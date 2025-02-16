"""
In this file we aim to make a program which tests Llama and other open source models to be able to create a QA
system in which answers our 9 prompts relevant to the ABM auto-generation.
Created on Mon May 27 11:17:05 2024
Documents:
https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476
@author: siamakkh
!pip install accelerate==0.21.0 transformers==4.31.0 tokenizers==0.13.3
!pip install bitsandbytes==0.40.0 einops==0.6.1
!pip install xformers==0.0.22.post7
!pip install langchain==0.1.4
!pip install faiss-gpu==1.7.1.post3
!pip install sentence_transformers
"""
from Functions import run_opensource_models, run_gpt_api
from Constants import AbmModels, LLMModels, Prompts, Commercial_QA

"""
############## General Information ###################
ALL constant variables including the list of ABM models, QA models, Prompts and etc are saved in Constants.py file.
The results will be saved in /Results/{MyResultsDir}/ directory in MyResultsDir sub-directory. The process.csv also 
will be retrieved from that particular root. 
"""
MyResultsDir = "BridgingGap"

"""
############## Open Source Run Commands ###################
Please refer to ReadMe.md for full description.
"""

run_opensource_models(AbmModels, LLMModels, Prompts, MyResultsDir)
"""
############## Commercial Run Commands ###################
Commercial models demand API setup including API token generation, files uploading, and financial support on your account.
Please refer to ReadMe.md for full description.
ABMModels: list[str]
Commercial_QA: list[str]
Prompts: list[str]
MyResultsDir: str
"""
run_gpt_api(AbmModels, Commercial_QA[0:1], Prompts, MyResultsDir)

"""
#################### Results Location #####################
After running these codes, all generated outputs will be saved in `/Results/LLMModelName/` direction. Each model should
have a folder named similar to model without any slashes. 
"""

"""
################### Cosine Similarities #####################
After generating results, you can run the code in the `CosineSimilarities.py` file, 
which will calculate the similarities. Calculating similarities is a time-consuming process, 
so the results will be saved in the `/Results/Similarities/` folder. 
If you generate new outputs, make sure to move the old similarity files to another folder to avoid having multiple
 similarity files in the same location. 

Please review the `CosineSimilarities.py` file for more info.
"""