import torch
import accelerate
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv,  find_dotenv, dotenv_values
import openai
import itertools
from Constants import (AbmModels, LLMModels, Prompts, PromptsContents, DataPath, Model_files,
                       PsAndValues, Model_Ids, Instruction, Vector_IDS, Commercial_QA, Epochs)
import numpy as np
import pandas as pd
import json, time, os, csv
from CosineSimilarityFunctions import read_json, save_to_json

# Load confidential from .env file
load_dotenv()
# .env path
EnvPath = ".env"
Secrets = dotenv_values(EnvPath)


def make_folder(folder_name: str):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists.")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def load_model(model_id):
    """

    :param model_id:
    :param temperature:
    :return:
    model: Returns the model to make the chat chain

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
    hf_auth = Secrets["HuggingFaceToken"]
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth
    )
    #### #Without quantization_config, you can not run the model on small GPUs.
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,
                                                           token=hf_auth,
                                                           use_fast=False,  # For larger models this is necessary
                                                           device_map="auto",  # This gives error for larger models
                                                           cache_dir="./Models/Cache/"
                                                           )
    stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])
    return model, tokenizer, stopping_criteria


def build_the_chain(model, tokenizer, pdf_text, stopping_criteria, temperature=0.1):

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )
    # prompt = "What is agent-based modeling."
    # res = generate_text(prompt)
    # print(res[0]["generated_text"])
    llm = HuggingFacePipeline(pipeline=generate_text)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    # If loaded using document loader
    # all_splits = text_splitter.split_documents(documents)
    all_splits = text_splitter.split_text(pdf_text)

    # Creating Embeddings and Storing in Vector Store

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    # vectorstore = FAISS.from_documents(all_splits, embeddings)
    vectorstore = FAISS.from_texts(all_splits,
                                   embeddings
                                   )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=(
                Instruction +
                "Context:\n{context}\n\n"
                "Chat History:\n{chat_history}\n\n"
                "User Question:\n{question}\n\n"
                "Your Answer:"
        )
    )
    # Initializing Chain
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                  vectorstore.as_retriever(),
                                                  condense_question_prompt=custom_prompt,
                                                  return_source_documents=False)
    return chain


def QA(chain, prompt):
    """
    from langchain.document_loaders import WebBaseLoader
    web_links = ["https://www.databricks.com/","https://help.databricks.com","https://databricks.com/try-databricks","https://help.databricks.com/s/","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/release-notes/index.html","http://docs.databricks.com/ingestion/index.html","http://docs.databricks.com/exploratory-data-analysis/index.html","http://docs.databricks.com/data-preparation/index.html","http://docs.databricks.com/data-sharing/index.html","http://docs.databricks.com/marketplace/index.html","http://docs.databricks.com/workspace-index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html","http://docs.databricks.com/delta/index.html","http://docs.databricks.com/dev-tools/index.html","http://docs.databricks.com/integrations/index.html","http://docs.databricks.com/administration-guide/index.html","http://docs.databricks.com/security/index.html","http://docs.databricks.com/data-governance/index.html","http://docs.databricks.com/lakehouse-architecture/index.html","http://docs.databricks.com/reference/api.html","http://docs.databricks.com/resources/index.html","http://docs.databricks.com/whats-coming.html","http://docs.databricks.com/archive/index.html","http://docs.databricks.com/lakehouse/index.html","http://docs.databricks.com/getting-started/quick-start.html","http://docs.databricks.com/getting-started/etl-quick-start.html","http://docs.databricks.com/getting-started/lakehouse-e2e.html","http://docs.databricks.com/getting-started/free-training.html","http://docs.databricks.com/sql/language-manual/index.html","http://docs.databricks.com/error-messages/index.html","http://www.apache.org/","https://databricks.com/privacy-policy","https://databricks.com/terms-of-use"] 

    loader = WebBaseLoader(web_links)
    documents = loader.load()
    print(documents)
    """

    chat_history = []
    query = prompt
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']


def get_open_ai():
    openai.api_key = Secrets["OpenAIKey"]
    return openai


def init_progress():
    """
    This function makes a dataframe to save the progress.
    """
    cols = ["abm_model", "llm_model", "prompt", "Temperature", "Epoch", "ThreadId",	"RunId", "IsSaved"]
    return pd.DataFrame(columns=cols)


def run_batches_opensource(abm_model: str, llm_model: str, prompt: str, my_results_dir: str):
    """

    :param abm_model: str
    This is the name of the ABM model. This will be retrieved from the names in the Constants.py > ABMModels list.
    It should also contain its corresponding information in other constants.
    :param llm_model: str
    This reflects the name of the LLM supported by this code.
    This name should exist in Constants.py under the LLMModels or Commercial_QA lists.
    :param prompt: str
    The prompt to pass to the LLM.

    :param my_results_dir: str
    The results will be saved in /Results/ directory in MyResultsDir sub-directory. The process.csv also will be retrieved
    from that particular root.
    """
    # Reading progress db file .
    results_dir = "./Results/{}".format(my_results_dir)
    progress_dir = "{}/progress.csv".format(results_dir)

    try:
        print(progress_dir)
        progress = pd.read_csv(progress_dir)
    except:
        make_folder(results_dir)
        progress = init_progress()

    # Read abm model source files (pdf, txt or docs) to retrieve context.
    pdf_text = extract_text_from_pdf(DataPath + Model_files[abm_model])

    # Pass the text and model to get LLM model, tokenizer and stop criteria.
    model, tokenizer, stop_criteria = load_model(llm_model)

    # Iterate over different temperatures, starting from 0.1 till 1 with 0.2 steps.
    for temperature in np.arange(0.1, 1, 0.2):
        # Reformation temperature to have 1 digit.
        temperature = float("{:.1f}".format(temperature))

        # Generating llm chain which holds the model config and history of chat.
        llm_chain = build_the_chain(model, tokenizer, pdf_text, stop_criteria, temperature)
        for epoch in range(Epochs):
            print(abm_model, llm_model.split("/")[-1], prompt, temperature, epoch)
            # Check progress db to see this has been already done.
            if (
                    (progress["abm_model"] == abm_model) & (progress["llm_model"] == llm_model.split("/")[-1]) &
                    (progress["prompt"] == prompt) & (progress["Temperature"] == temperature) &
                    (progress["Epoch"] == epoch)
            ).any():
                print("Already Done!")
                pass
            else:
                # print("Epoch: ", i, ", Temperature: ", temperature)
                # Pass LLm, VectorStore and prompt to the model.
                # Some prompts need inputs based on their abm model like specific agent-type name for specific ABM.
                # PsAndValues dict in Constants holds these values.
                # some values are empty, thus, by using **dict, it will pass non-empty parameters into the formatter.
                # qa_results, will have the string specific for eac prompt.
                qa_result = QA(llm_chain, PromptsContents[prompt].format(
                    **PsAndValues[prompt][abm_model]
                ))
                # Generate the file name to save the generated output.
                file_name = "{}/{}/{}-{}-{}-{}.json".format(results_dir,
                                                            llm_model.split("/")[-1],
                                                            abm_model,
                                                            llm_model.split("/")[-1],
                                                            prompt,
                                                            temperature)
                try:
                    # We first read the file to add the new record to the previous records.
                    all_responses = read_json(file_name)
                except:
                    # If there is no previous records, init it.
                    all_responses = {}

                # Add the new record to the previous records.
                all_responses[int(epoch)] = qa_result
                # Save records back to the same file.
                save_to_json(file_name, all_responses)

                # Update the progress db.
                with open(progress_dir, "a", encoding="utf-8") as file:
                    csv.writer(file).writerow([abm_model, llm_model.split("/")[-1], prompt, temperature, epoch])
                    file.close()
                print("Done!")


def run_batches_gpt(abm_model: str, qa_model: str, prompt: str, my_results_dir: str, upload_file: bool = False):
    """
    This function runs a request for specific ABM, OpenAI LLM and prompt.
    :param llm_model: str
    This reflects the name of the LLM supported by this code.
    This name should exist in Constants.py under the LLMModels or Commercial_QA lists.
    :param prompt: str
    The prompt to pass to the LLM.

    :param my_results_dir: str

    :param upload_file: bool = False
    Upload files if it is not openai server?


    """
    results_dir = "./Results/{}".format(my_results_dir)
    progress_dir = "{}/progress.csv".format(results_dir)

    try:
        progress = pd.read_csv(progress_dir)
    except:
        make_folder(results_dir)
        progress = init_progress()

    # Establish the connection with OpenAI server.
    OAI = get_open_ai()
    # Upload a file with an "assistants" purpose
    instruction = Instruction
    # Upload files if it is not openai server?
    # upload_file = False

    # setting prompt for specific case.
    prompt_content = PromptsContents[prompt].format(
        **PsAndValues[prompt][abm_model]
    )
    # Iterate over temperatures.
    for temp in np.arange(0.1, 1, 0.2):
        temp = float("{:.1f}".format(temp))
        for epoch in range(Epochs):
            print(prompt, abm_model, qa_model, temp, epoch)
            # Check whether this has been done.
            if (
                    (progress["abm_model"] == abm_model) & (progress["llm_model"] == qa_model.split("/")[-1]) &
                    (progress["prompt"] == prompt) & (progress["Temperature"] == temp) &
                    (progress["Epoch"] == epoch)
            ).any():
                print("Already Done!")
            else:
                # file_id = Model_Ids[abm_model]
                if upload_file:
                    file = OAI.files.create(
                        # file=open(os.getcwd()+"/Data/PovertyPDF.pdf", "rb"),
                        file=open(os.getcwd() + "/Data/{}".format(Model_files[abm_model]), "rb"),
                        purpose='assistants'
                    )
                    file_id = file.id
                    print("Save this id in constants:\n", abm_model, file_id)

                # Getting vector_store
                vector_store = OAI.beta.vector_stores.retrieve(Vector_IDS[abm_model])
                # Init GPT Assistant
                assistant = OAI.beta.assistants.create(
                    instructions=instruction,
                    model=qa_model,
                    tools=[{"type": "file_search"}],
                    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
                    temperature=temp,
                )

                # Generate OpenAI Thread
                thread = OAI.beta.threads.create(
                    messages=[{"role": "user",
                               "content": prompt_content}],
                    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
                )

                # Init the run
                run = OAI.beta.threads.runs.create(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                )
                # OpenAI needs time to generate answers.
                # To track the response, it returns the thread id and run id which needs
                # to be stored. We will store it in the progress csv.
                # then these threads and runs can be read.
                with open(progress_dir, "a", encoding="utf-8") as file:
                    csv.writer(file).writerow(
                        [abm_model, qa_model.split("/")[-1], prompt, temp, epoch, thread.id, run.id])
                    file.close()
                print("Done!")
                # There is a request frequency limit on OpenAI servers.
                time.sleep(5)


def retrieve_openai_answers(my_results_dir: str):
    """
    This function iterates over progress.csv and retrieves any answers in this file that are relevant to OpenAI
    and have not been marked as retrieved.

    :param my_results_dir: str
    """
    # Connect to OpenAI
    OAI = get_open_ai()

    # Load the progress file
    results_dir = "./Results/{}".format(my_results_dir)
    progress_dir = "{}/progress.csv".format(results_dir)
    progress = pd.read_csv(progress_dir)
    # Filer the progress to get Commercial relavant ones.
    progress_openai = progress[progress["llm_model"].isin(Commercial_QA)]

    for prog_index, this_prog in progress_openai.iterrows():
        # this_prog = progress_openai.head(1)
        print(prog_index)
        qa_model = this_prog["llm_model"]
        abm_model = this_prog["abm_model"]
        prompt = this_prog["prompt"]
        temp = this_prog["Temperature"]
        epoch = this_prog["Epoch"]
        thread_id = this_prog["ThreadId"]
        run_id = this_prog["RunId"]
        is_saved = this_prog["IsSaved"]
        print(is_saved)
        file_name = "{}/{}/{}-{}-{}-{}.json".format(results_dir,
                                                    qa_model.split("/")[-1],
                                                    abm_model,
                                                    qa_model.split("/")[-1],
                                                    prompt,
                                                    temp)

        if is_saved != True:
            print("retrieving answers")

            class RetrieveObj():
                def __init__(self):
                    self.status = None

            retrieve_run = RetrieveObj()

            answer = OAI.beta.threads.messages.list(thread_id)
            answer = answer.data[0].content[0].text.value
            # Save answer to the file
            try:
                all_responses = read_json(file_name)
            except:
                all_responses = {}
            all_responses[int(epoch)] = answer
            save_to_json(file_name, all_responses)

            # change save flag
            new_progress = this_prog.values.tolist()
            new_progress[-1] = True
            progress.loc[prog_index] = new_progress

            progress.to_csv(progress_dir, index=False)

def run_opensource_models(AbmModels, LLMModels, Prompts, MyResultsDir):
    # This code generates a list of inputs for qa batch run.
    open_source_combinations = list(itertools.product(AbmModels, LLMModels, Prompts, [MyResultsDir]))
    # Open source models are ran on local machine, so they will both generate and save information in the same process.
    [run_batches_opensource(*combo) for combo in open_source_combinations]


def run_gpt_api(AbmModels, Commercial_QA, Prompts, MyResultsDir):
    # This code generates a list of inputs for qa batch run.
    # Commercial_QA[0:1] selects only the first item in the list, which is gpt4-0
    openai_combos = list(itertools.product(AbmModels, Commercial_QA, Prompts, [MyResultsDir]))
    [run_batches_gpt(*combo) for combo in openai_combos]

    # Commercial models operate on their own servers, which means they need some time to process information.
    # To manage this delay, these models typically send back tracking information that allows users to retrieve data later.
    # For example, ChatGPT provides a Thread ID and Run ID.

    # During the request process, all of this information is saved in a local file named `process.csv`.
    # The function `retrieve_openai_answers()` extracts this information and retrieves answers from the OpenAI server.
    retrieve_openai_answers(MyResultsDir)