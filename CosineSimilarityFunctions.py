import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Constants as Cont
import numpy as np
from scipy import stats
import re, openai, torch, transformers, json, pickle
import seaborn as snb
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv, dotenv_values
import accelerate
from torch import cuda, bfloat16
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from sentence_transformers import SentenceTransformer, util
from pprint import pprint
import os

load_dotenv()
config = dotenv_values(".env")
client = OpenAI(api_key=config["OpenAIKey"])

def get_open_ai():
    openai.api_key = config["OpenAIKey"]
    return openai


def calc_gpt_cosine(results_dir, qa_model, vector_type, model_name, prompt_, abm_model_, temp):

    Similarities_Path = "{}/Similarities/{}/{}-{}-{}-{}-{}-{}.pickle"  # LLM/ABM-LLM-P-Temp-VectType-EmbedModel
    file_name = "{}/{}/{}-{}-{}-{}.json".format(results_dir,
                                                qa_model.split("/")[-1],
                                                abm_model_,
                                                qa_model.split("/")[-1],
                                                prompt_,
                                                temp)
    sim_path = Similarities_Path.format(
        results_dir, qa_model, abm_model_, qa_model, prompt_, temp, vector_type, qa_model
    )
    print(file_name)
    try:
        similarities_ = read_similarities(sim_path)
        print("Similarities read from db.", sim_path)
        return similarities_
    except:
        # Read the json file of results
        texts = read_json(file_name)
        # turn them to a list of texts
        texts = list(texts.values())

        # sometimes commercial ones do not compile the job, and instead returns the prompt. Exclude them from analysis.
        # the retrieved answer includes ```json in its first line
        texts = [t for t in texts if t.split("\n")[0]=="```json"]
        print(len(texts))
        expected_output = Cont.PsAndOutputs[prompt_][abm_model_]
        expected_output = expected_output.replace("\n", "")
        expected_output = expected_output.replace("  ", "")
        texts_list = [expected_output] + texts
        if(len(texts_list)>1):
            similarities_=[]
            if vector_type == "TFIDF":
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(texts_list)
                # Calculate the cosine similarity between the two texts
                similarities_ = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten().tolist()
                # print("Similarities generated.")
            if vector_type == "GPT40":
                embeddings_list = []
                [embeddings_list.append(openai_embeddings(r, model_name)) for r in texts_list]
                similarities_ = cosine_similarity(embeddings_list[0:1], embeddings_list[1:]).flatten().tolist()
            if vector_type == "OpenSource":
                embeddings_list = []
                [embeddings_list.append(open_source_embeddings(r, model_name)) for r in texts_list]
                similarities_ = cosine_similarity(embeddings_list[0:1], embeddings_list[1:]).flatten().tolist()
            if vector_type == "SentenceTransformer":
                m = SentenceTransformer(model_name)
                embeddings_list = m.encode(texts_list)
                similarities_ = util.pytorch_cos_sim(embeddings_list[0], embeddings_list[1:]).numpy().tolist()
                similarities_ = similarities_[0]
            write_similarities(sim_path, similarities_)
            return similarities_
        else:
            return []


def save_to_json(file_name, data):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
        file.close()


def read_json(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        content = json.load(file)
        file.close()
        return content


def GPTSimilarities(results_dir, prompt, qa_model, VecType, VectModel):
    dt = []
    for abm in Cont.AbmModels[:4]:
        Temperature_Range = [float(r"{:.1f}".format(a)) for a in np.arange(0.1, 1, 0.2)]
        similarities = [calc_gpt_cosine(results_dir, qa_model, VecType, VectModel, prompt, abm, t) for t in Temperature_Range]
        similarities = [s for sims in similarities for s in sims]
        # similarities = calc_gpt_cosine(texts, VecType, VectModel, prompt, abm)
        # If we want to have all similarities for abm, llm per prompt, we need to flatten the similarities list
        # No need to flatten this, because there is no (nested) similarities per temperature
        # similarities = [s for sims in similarities for s in sims]
        # print("********** \n", similarities)
        # sts = [stats.describe(s) for s in similarities]
        if(len(similarities)>0):
            sts = stats.describe(similarities)
            # print(len(sts))
            dt.append([prompt, qa_model, abm, sts.mean, sts.variance])
        else:
            dt.append([prompt, qa_model, abm, 0, 0])
    return dt


def load_model(model_name):
    """
    This function receives the name of the open source model and return its model and tokenizer
    :param model_name:
    :return:
    model: Returns the model to make the chat chain
    tokenizer: Returns the tokenizer to make the chat chain or embeddings

    """
    torch.cuda.empty_cache()
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    # begin initializing HF items, you need an access token
    hf_auth = config["HuggingFaceToken"]
    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        token=hf_auth
    )
    # ####Without quantization_config, you can not run the model on small GPUs.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device,
        token=hf_auth,
        cache_dir="./Models/Cache/"
    )

    # enable evaluation mode to allow model inference
    model.eval()

    print(f"Model loaded on {device}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name,
                                                           token=hf_auth,
                                                           use_fast=False,  # For larger models this is necessary
                                                           device_map="auto",  # This gives error for larger models
                                                           cache_dir="./Models/Cache/"
                                                           )
    return model, tokenizer


def open_source_embeddings(text, model_name="bert-base-uncased"):
    """
    This function returns the embedding of the open source pre-trained model
    :param text: The text to be embedded
    :param model_name: The name of the to be used for the embedding
    :return: Embedded vector of the input text
    """
    model, tokenizer = load_model(model_name)
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       padding=True
                       )
    # Get the embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the mean of the embeddings
    embeddings = outputs[0].mean(dim=1)
    return embeddings.numpy()[0]


def openai_embeddings(text, model_name="text-embedding-3-large"):
    """
    This function returns the embedding of the input text based on the OpenAI commercial models.
    :param text: input text
    :param model_name: model_name
    :return:
    """
    response = client.embeddings.create(
        input=text,
        model=model_name # Use a specific model available for embeddings
    )
    return response.data[0].embedding


def read_output_texts(file_path, abm_model_, prompt_):
    """

    :param file_path:
    :param abm_model_:
    :param prompt_:
    :return:
    """
    with open(file_path, 'r', encoding="utf8") as f:
        res = f.read()
        f.close()
    # Each print output adds \n so it can be used to devide lines of epochs
    stop = r"\*\*\*\*\*\*\*\*\*\*\* Run (\d+) Ends\*\*\*\*\*\*\*\*\*\*|\*\*\*\*\*\*\*\*\*\*\* Run (\d+) Starts\*\*\*\*\*\*\*\*\*\*"
    res = res.replace("\n", "")
    results = re.split(stop, res)
    results = [s for s in results if not s in ["", " ", None]]
    results = [s for s in results if not s.isdigit()]

    # Remove \n and extra white space from expected outputs
    expected_output = Cont.PsAndOutputs[prompt_][abm_model_]
    expected_output = expected_output.replace("\n", "")
    expected_output = expected_output.replace("  ", "")
    return expected_output, results


def read_similarities(file_path):
    with open(file_path, 'rb' ) as f:
        # similarities_ = f.read()
        similarities_ = pickle.load(f)
        f.close()
    return similarities_


def write_similarities(file_path, similarities_):
    with open(file_path, "wb") as file:
        # file.write("{}".format(similarities_))
        pickle.dump(similarities_, file)
        file.close()


def read_embeddings(file_path):
    with open(file_path, 'rb' ) as f:
        # embeddings = f.read()
        embeddings = pickle.load(f)
        f.close()
    return embeddings


def write_embeddings(file_path, embeddings):
    with open(file_path, "wb" ) as file:
        # file.write("{}".format(embeddings))
        pickle.dump(embeddings, file)
        file.close()


def cosine_sim(results_dir, abm_model_, llm_model, prompt_, temp, vector_type="TFIDF", model_name=""):

    """
    This function calculates the cosine similarity between the reference text and the output texts. It reads the
    reference text from constants.py and the outputs from ./Results/LLM_MODEL_NAME
    :param abm_model_: The name of the ABM model
    :param llm_model: The name of the LLM Model
    :param prompt_: The prompt name
    :param temp: Temperature relevant to the LLM initial setup.
    :param vector_type: The type of the vectorization
    :param model_name: The model name relevant to the open source and sentence similarities
    :return:
    This function gets parameters for cosine similarity and returns the cosine similarity of all epoch results.
    """

    llm_model_name = llm_model.split('/')[-1]
    # print(llm_model_name)

    Embeddings_Path = "{}/Embeddings/{}/{}-{}-{}-{}-{}-{}.pickle"  # LLM/ABM-LLM-P-Temp-VectType-EmbedModel
    Similarities_Path = "{}/Similarities/{}/{}-{}-{}-{}-{}-{}.pickle"  # LLM/ABM-LLM-P-Temp-VectType-EmbedModel
    Results_Path = "{}/{}/{}-{}-{}-{:.1f}.txt"
    file_path = Results_Path.format(results_dir, llm_model_name, abm_model_, llm_model_name, prompt_, temp)

    sim_path = Similarities_Path.format(
        results_dir, llm_model_name, abm_model_, llm_model_name, prompt_, temp, vector_type, llm_model_name
    )
    embedd_path = Embeddings_Path.format(
        results_dir, llm_model_name, abm_model_, llm_model_name, prompt_, temp, vector_type, llm_model_name
    )
    try:
        similarities_ = read_similarities(sim_path)
        print("Similarities read from db.", file_path)
        return similarities_
    except:
        print("No similarities found.")
        # Convert the texts to TF-IDF vectors
        if vector_type == "TFIDF":
            expected_output, results = read_output_texts(file_path, abm_model_, prompt_)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([expected_output]+results)
            # Calculate the cosine similarity between the two texts
            similarities_ = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten().tolist()
            write_similarities(sim_path, similarities_)
            # print("Similarities generated.")
            return similarities_
        else:
            try:
                # If there are embeddings saved, return them
                embeddings_list = read_embeddings(embedd_path)
                # print("Embeddings found.")
            except:
                # Else make the embeddings and save them
                print("Embeddings could not found.")
                expected_output, results = read_output_texts(file_path, abm_model_, prompt_)
                embeddings_list = []
                texts_list = [expected_output] + results
                if vector_type == "GPT40":
                    [embeddings_list.append(openai_embeddings(r, model_name)) for r in texts_list]
                if vector_type == "OpenSource":
                    [embeddings_list.append(open_source_embeddings(r, model_name)) for r in texts_list]
                if vector_type == "SentenceTransformer":
                    m = SentenceTransformer(model_name)
                    embeddings_list = m.encode(texts_list)
                write_embeddings(embedd_path, embeddings_list)
                # print("Embeddings generated.")
            similarities_ = []
            if vector_type == "GPT40":
                similarities_ = cosine_similarity(embeddings_list[0:1], embeddings_list[1:]).flatten().tolist()
            if vector_type == "OpenSource":
                similarities_ = cosine_similarity(embeddings_list[0:1], embeddings_list[1:]).flatten().tolist()
            if vector_type == "SentenceTransformer":
                similarities_ = util.pytorch_cos_sim(embeddings_list[0], embeddings_list[1:]).numpy().tolist()
                similarities_ = similarities_[0]
            write_similarities(sim_path, similarities_)
            # print("Similarities generated.")
            return similarities_


def plot_thresholds(ax2, range_):
    ax2.plot(range_, [0.8]*len(range_), "--", label="Very High Cos")
    ax2.plot(range_, [0.5]*len(range_), "--", label="High Cos")
    ax2.plot(range_, [0.25]*len(range_), "--", label="Moderate Cos")


def plot_sts_temp(sts, temperature_range):
    plt.figure(figsize=(10, 10))
    plt.scatter(temperature_range, [s.mean for s in sts])
    plt.show()


def plot_box_temp(data, sts, p, temperature_range, abm_model, llm_model, epochs):
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20, 10))
    ax1.boxplot(data)
    ax2.scatter(temperature_range, [s.mean for s in sts])
    b, a = np.polyfit(temperature_range, [s.mean for s in sts], deg=1)
    ax2.plot(temperature_range, a + b * temperature_range)
    ax1.set_title("Box PLot of cosine similarity")
    ax2.set_title("Mean of cosine similarity")
    fig.suptitle("Cosine Similarity {ABM}, {LLM}, {Prompt} for temperature[0.1,0.9] in {epochs} Epochs.".format(
        ABM=abm_model, LLM=llm_model, Prompt=p, epochs=epochs))
    plt.legend()
    plt.show()


def plot_box_temp_ps(data, sts, p, ax1, ax2, temperature_range):
    ax1.scatter(temperature_range, [s.variance for s in sts])
    ax2.scatter(temperature_range, [s.mean for s in sts])
    b, a = np.polyfit(temperature_range, [s.mean for s in sts], deg=1)
    ax2.plot(temperature_range, a + b*np.array(temperature_range), label=p)


def plot_all_tempos(llm_models):
    for p in Cont.Prompts:
        for llm_ in llm_models:
            for abm in Cont.AbmModels[:4]:
                similarities = [cosine_sim(abm, llm_, p, t) for t in np.arange(0.1, 1, 0.1)]
                sts = [stats.describe(s) for s in similarities]
                plot_box_temp(similarities, sts, p)


def plot_all_tempos_ps_together(abm_model, llm_, temperature_range, epochs):
    for abm in Cont.AbmModels[:4]:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20, 10))
        for p in Cont.Prompts:
            similarities = [cosine_sim(abm, llm_, p, t) for t in temperature_range]
            sts = [stats.describe(s) for s in similarities]
            plot_box_temp_ps(similarities, sts, p, ax1, ax2, temperature_range)
        ax1.set_title("Box PLot of cosine similarity")
        ax2.set_title("Mean of cosine similarity")
        title = "Cosine Similarity {ABM}, {LLM}, for all Ps temperature[0.1,0.9] in {epochs} Epochs."
        fig.suptitle(title.format(ABM=abm_model, LLM=llm_, epochs=epochs))
        plt.legend()
        plt.show()


def plot_box_temp_abm(data, sts, abm, ax1, ax2, temperature_range):
    y = temperature_range
    ax1.scatter(y, [s.variance for s in sts])
    ax2.scatter(y, [s.mean for s in sts])
    b, a = np.polyfit(y, [s.mean for s in sts], deg=1)
    ax2.plot(y, a + b*np.array(y), label=abm)


def plot_all_tempos_ABMs_together(LLM, temperature_range, epochs, vectoriztion_type="TFIDF", vect_model=""):
    for p in Cont.Prompts:
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(20, 10))
        for abm in Cont.AbmModels[:4]:
            similarities = [cosine_sim(abm, LLM, p, t, vectoriztion_type) for t in temperature_range]
            sts = [stats.describe(s) for s in similarities]
            plot_box_temp_abm(similarities, sts, abm, ax1, ax2)
        ax1.set_title("Box PLot of cosine similarity")
        ax2.set_title("Mean of cosine similarity")
        ax1.set_yticks(temperature_range)
        ax2.set_yticks(temperature_range)
        ax1.set_yticklabels(["{}:.1f".format(a) for a in temperature_range])
        ax2.set_yticklabels(temperature_range)
        plot_thresholds(ax2, temperature_range)

        title_txt = ("Cosine Similarity All ABms, {LLM}, {Prompt} for temperature[0.1,0.9] in "
                     "{epochs} Epochs, Vectorizer: {vect} Vect_model ={vect_model}.")
        fig.suptitle(title_txt.format(LLM=LLM, Prompt=p, epochs=epochs, vect=vectoriztion_type, vect_model=vect_model))
        plt.legend()
        plt.show()


def plot_heatmaps(dt):
    # Setup fonts.
    font = {'size': 16}
    plt.rc('font', **font)

    # Prepare data for plotting
    dt["LLM | ABM"] = [s1 + " | " + s2 for s1,s2 in list(zip(dt["LLM"].values, dt["ABM"].values))]

    # Mean of the similarities
    dt_plot = dt.pivot_table(index=["LLM | ABM"], columns="Prompt", values="Mean")
    # Pivot reshuffles the order of dt, to solve, we reset the index
    dt_plot = dt_plot.reindex(list(dict.fromkeys(dt["LLM | ABM"])))
    plt.figure(figsize=(10, 15))
    snb.heatmap(dt_plot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Mean of Cosine Similarities per Prompt/LLM/ABM")
    plt.show()

    # Std of the similarities
    dt_plot = dt.pivot_table(index=["LLM | ABM"], columns="Prompt", values="Std")
    dt_plot = dt_plot.reindex(list(dict.fromkeys(dt["LLM | ABM"])))
    plt.figure(figsize=(10, 15))
    snb.heatmap(dt_plot, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Std of Cosine Similarities per Prompt/LLM/ABM")
    plt.show()


def calculate_all_similarities(MyResultsDir:str, similarities_file_postfix:str, VecType: str, VectModel:str):
    results_dir = "./Results/{}".format(MyResultsDir)
    # We will save the result of calculated similarities in the following path.
    similarities_path = "{results_dir}/{vect_type}_{vect_model}_Results_{postfix}.pkl".format(results_dir=results_dir,
                                                                                              vect_type=VecType,
                                                                                              vect_model=
                                                                                              VectModel.split("/")[-1],
                                                                                              postfix=similarities_file_postfix)

    Temperature_Range = [float(r"{:.1f}".format(a)) for a in np.arange(0.1, 1, 0.2)]
    dt = []
    try:
        dt = pd.read_pickle(similarities_path)
    except:
        for p in Cont.Prompts:
            dtGPT = GPTSimilarities(results_dir, p, Cont.Commercial_QA[0], VecType,
                                    VectModel)  # This function adds Chatgpt similarities to the dt
            [dt.append(d) for d in dtGPT]
            for LLM in Cont.LLMModels_:
                for abm in Cont.AbmModels[:4]:
                    # Each cosine similarity is based on abm, llm, p, t, epochs
                    # each similarity has 5 lists each per temperature 0.1, 0.3, 0.5, 0.7, 0.9
                    similarities = [cosine_sim(results_dir, abm, LLM, p, t, VecType, VectModel) for t in Temperature_Range]
                    # If we want to have all similarities for abm, llm per prompt, we need to flatten the similarities list
                    similarities = [s for sims in similarities for s in sims]
                    # print("********** \n", similarities)
                    # sts = [stats.describe(s) for s in similarities]
                    sts = stats.describe(similarities)
                    # print(len(sts))
                    dt.append([p, LLM.split("/")[-1], abm, sts.mean, sts.variance])
                    #print("********** prompt: {}, LLM: {}, abm: {} \n".format( p, LLM.split("/")[-1], abm), sts.mean, sts.variance)
        dt = pd.DataFrame(dt, columns=["Prompt", "LLM", "ABM", "Mean", "Std"])
        dt.to_pickle(similarities_path)
    return dt


def similarities(MyResultsDir: str, similarities_file_postfix: str,  plot_heatmap: bool = True):
    """
    This function calculates and stores similarities. It can also plot the heatmap.
    # Similarities_dt contains the results of calculated similarities.
    # We cache our calculations in similarities_path to save time for later.
    # Plot similarities heatmap based on similarities_dt
    # To transfer generated outputs to embeddings, we need to know what type of embedding we use (VectType) and what model
    # of that embedding type is used (VectModel).

    :param MyResultsDir: str
    Calculating similarities takes a considerable amount of time. Therefore, the results of these calculations are saved
     in a directory called MyResultsDir, and they will be retrieved if accessible. If you need to save a new set of
     calculations, you can change this directory to specify a different location for saving.

    :param plot_heatmap: bool = True
    This will indicate whether a plot should be done or not.

    :param similarities_file_postfix: str
    # Is used for versioning new similarity files.

    """
    VecType = Cont.VecList[2]
    VectModel = Cont.SentenceTransformerModels[0]

    similarities_dt = calculate_all_similarities(MyResultsDir, similarities_file_postfix, VecType, VectModel)
    if plot_heatmap:
        plot_heatmaps(similarities_dt)
