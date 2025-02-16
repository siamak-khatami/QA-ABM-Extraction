# Run setup

## 1. Requirements.txt
Please install all packages mentioned in Requirements.txt.
1. Ensure you have your python installed. 
   Open your terminal and check:
   `python --version`
2. Ensure you have pip installed
3. Install requirements:
   In your terminal in the code directory:
   `pip install -r Requirements.txt`

## 2. Hardware
### 1. Commercial models: 
No Specific hardware is required. 
### 2. Open Source Models
   To run open-source models, you need GPUs that are compatible with CUDA. 
All GPUs available on cloud providers like Google Colab are suitable. 
However, using Jupyter on Google Colab will give you limited access.

## 3. Commercial Models Configurations

### OpenAI API Token
In order to use the OpenAI API, you need to make a `.env` file with no name. Then you need to include a key called `OpenAIKey`. 
### Uploading Files to OpenAI
After uploading files to the OpenAI api dashboard, you can copy and paste file and vector IDs into the `constant.py` in `Model_Ids` and `Vector_IDS` dictionaries . 


## 4. Open Source QA Models Configuration

### Hugging face token
In your `.env` file, you need to add your hugging face token as `HuggingFaceToken`
### QA Models
To include any QA models from Hugging Face repo, you just need to add the 
model name to `LLMModels` in `Constants.py` Constants
### Pre-trained LLMs
There are two types of pre-trained LLMs. The first branch of similar to QA 
models, you just need to follow their instructions on Hugging Face and include
their name in `LLMModels` list. 
The second types like Llama and Gemma models need some configurations. 
For LLama you need to request access for your hugging face token.
For gemma you need to down the code to the `"./Models/Downloaded/MODELNAME`
repo and include this name into `LLMModels`.

## Running Code
### 1. Generating Outputs
To generate outputs, you need to run the `main.py` file. If you want to regenerate outputs, you can make a new folder with `ANYNAME` and
change `MyResultsDir` variable content in the `main.py`. You should have the structure as follow:
- Folder for each model name
- Similarities
- Embeddings
### 2. Calculating Cosine Similarities
After generating your outputs, you can change the `MyResultsDir` in `CosineSimilarity.py` file and run the file. This will generate  
