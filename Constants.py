# ABMModels * LLM Models * Epochs * Tempts * Prompts * 30
DataPath = "./Data/"

# To ensure a consistent result, we re-run the same set of requests for each QA model.
# Epoch represents the number of those iterations.
Epochs = 10

Model_files = {
    "PovertyPDF": "PovertyPDF.pdf",
    "TroutPDF": "Trout.pdf",
    "Tiger": "tiger.pdf",
    "Elites": "elites.pdf",
}

AbmModels = [
    "PovertyPDF",
    "TroutPDF",
    "Tiger",
    "Elites"
]

""" Model Ids and Vector Ids"""
# As context, we need to upload ABM models. To use ABM models using OpenAI models (ChatGPT), we need to upload
# files on OpenAI dashboard and receive their file and vector id.
# You can use this URL to upload your files and vectors.
# https://platform.openai.com/storage/files

Model_Ids = {
    "Poverty": "file-HbNEYWNVBEHAT1aiPMrc6kIl",
    "PovertyPDF": "file-OJQbTyp22kOsx8hbdp9tvKG8",
    "Tiger": "file-Pl7eOhKgETlBJYX87ZkWZWjz",
    "TroutPDF": "file-e7TygYOi5xbjOlN9zLDR6SVY",
    "Elites": "file-WAzk4PPfpCPqxio5lnc4h98a",
}

Vector_IDS = {
    "Poverty": "vs_WoPPg2mvH8lMlXtIueLtgnxE",
    "PovertyPDF": "vs_32626VB3N3wLyBaMo1OzNSBd",
    "TroutPDF": "vs_pgSMCrkJ9VeCvloZzFdUzlSK",
    "Tiger": "vs_8l4ITTPVGfC3fLrKCyMmX5rh",
    "Elites": "vs_2iZ29ZbzCWxDXfgUEQuBwWKi",
}

LLMModels = [
    'NousResearch/Llama-2-7b-chat-hf',
    'NousResearch/Meta-Llama-3-8B',
    "NousResearch/Meta-Llama-3-70B",
    'NousResearch/Llama-2-13b-chat-hf',
    "NousResearch/Llama-2-70b-hf",
    # "google/gemma-2b", # Local Model, requires to be downloaded
    # "google/gemma-7b", # Local Model, requires to be downloaded
    # "./Models/Downloaded/gemma-2-27b", # requires to be downloaded
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2"   
    "microsoft/Phi-3-mini-4k-instruct"
]

LLMModels_ = [
        'NousResearch/Llama-2-7b-chat-hf',
        'NousResearch/Llama-2-13b-chat-hf',
        'NousResearch/Meta-Llama-3-8B',
        "./Models/Downloaded/gemma-2b", # Local Model, Not relevant answer
        "./Models/Downloaded/gemma-7b", # Local Model, Not relevant answer
        "./Models/Downloaded/gemma-2-9b",
        "./Models/Downloaded/gemma-2-27b",
        "microsoft/phi-1", # Not a relevant answer
        "microsoft/phi-1_5",  # Not a relevant answer
        "microsoft/phi-2"  # Not a relevant answer
    ]

Commercial_QA = ["gpt-4o", "gpt-4-0125-preview", "gpt-3.5-turbo-0125"]

Prompts = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

Instruction = ("You are an Agent based modeling specialist. Your duty is to help users in extracting information"
               "from ABM texts for coding purposes. Get the user messages to extract the relevant information from "
               "uploaded file. Do not summarize information and return a full report of the expected information. "
               "Return expected information just in the json format without any extra text around it. ")

PromptsContents = {
    #   Model Purpose
    "P1": """Analyze the provided ABM text to identify the purpose of the model. 
    Present the extracted data exclusively in JSON format, ensuring that the JSON object is comprehensive 
    and contains all requested information. Avoid any form of data truncation or summarizing, and ensure 
    that the response is strictly limited to the JSON object without any supplementary text. Please don't 
    generate extra text and answer. The JSON should follow this structure: 
    {{'Model_Purpose': {{'full_description':Full_DESCRIPTION, 'research_questions': 
    ['RESEARCH_QUESTION_1', 'RESEARCH_QUESTION_2', ... ], 'system_boundaries': [], 'outcome_variables': 
    {{VAR1_NAME: SHORT_DESCRIPTION, VAR2_NAME: SHORT_DESCRIPTION, ....}} }} }}""",
    #   List of Agent Sets
    "P2": """Analyze the provided ABM text to identify the list of all agent sets, a short description, and their
    agent role in the system. Present the extracted data exclusively in JSON format, ensuring that the JSON
    object is comprehensive and contains all requested information. Avoid any form of data truncation or
    summarizing, and ensure that the response is strictly limited to the JSON object without any supplementary 
    text. The JSON should follow this structure: {{AGENT_SET_1_NAME: {{ 'short_description':SHORT_DESCRIPTION, 
    'agent_role': SHORT_DESCRIPTION_AGENT_ROLE}}, ...}}""",
    #   List of Variables for AgentSets, a short description, their data types and initial value
    #   Input: {AGENT_SET} =  The name of the Agent-set Extracted from P2
    "P3": """Analyze the provided ABM text to identify and extract the complete list of variables, variable data
    type, and initial value related to the '{AGENT_SET}' agent. Please ensure you extract all variables and
    characteristics. Present the extracted data exclusively in JSON format, ensuring that the JSON object is
    comprehensive and contains all requested information. Avoid any form of data truncation or summarization, 
    and ensure that the response is strictly limited to the JSON object without any supplementary text. The JSON
    should follow this structure: {{ '{AGENT_SET}' :{{VAR1:{{'short_description':SHORT_DESCRIPTION, 'data_type': 
    DATA_TYPE, 'initial_value':INITIAL_VALUE, }}, VAR2 :{{...}} }} }}""",
    #   List of variables for a specific variable related to a specific agent-set to get their value_boundaries,
    #   equation, order numbers, frequency of execution.
    #   Input: '{AGENT_SET}', '{VAR}'
    "P4": """Analyze the provided ABM text to identify and extract the value boundaries, equation, order of 
          execution, and frequency of execution related to the '{VAR}' variable of '{AGENT_SET}' agent. Please ensure 
          you extract all variables and characteristics. Present the extracted data exclusively in JSON format, 
          ensuring that the JSON object is comprehensive and contains all requested information. Avoid any form of 
          data truncation or summarization, and ensure that the response is strictly limited to the JSON object 
          without any supplementary text. The JSON should follow this structure: {{ '{AGENT_SET}':{{ '{VAR}': 
          {{'value_boundaries':VALUE_BOUNDARIES, 'equation': EQUATION, 'order_number':ORDER_NUMBER, 
          'frequency': FREQUENCY }}, VAR2 :{{...}}}}}} """,
    #   Extracting the information relevant to the environment (space) characteristics of the abm model, with a
    #   short description and the space type.
    "P5": """Analyze the provided ABM text to identify and extract the information about the ABM simulation Space 
           (environment) type and Space (environment) short description. Present the extracted data exclusively in JSON
            format, ensuring that the JSON object is comprehensive and contains all requested information. 
            Avoid any form of data truncation or summarization, and ensure that the response is 
          strictly limited to the JSON object without any supplementary text. The JSON should follow this structure: 
          {{'Space': {{'short_description':SHORT_DESCRIPTION, 'type': TYPE }}}}""",
    #   Extracting the list of space variables
    "P6": """Analyze the provided ABM text to identify and extract the complete list of variables, variable data 
          type, and initial value related to the model space. Please ensure you extract all variables and 
          characteristics. Present the extracted data exclusively in JSON format, ensuring that the JSON object is 
          comprehensive and contains all requested information. Avoid any form of data truncation or summarization, 
          and ensure that the response is strictly limited to the JSON object without any supplementary text. The 
          JSON should follow this structure: {{'SPACE':{{VAR1:{{'short_description':SHORT_DESCRIPTION, 'data_type': 
          DATA_TYPE, 'initial_value':INITIAL_VALUE, }}, VAR2 :{{...}}}}}}""",
    #   Extracting the characteristics of each space variable
    #   Input: '{VAR}'
    "P7": """Analyze the provided ABM text to identify and extract the value boundaries, equation, order of 
    execution, and frequency of execution related to the '{VAR}' variable of model space. Please ensure you extract
    all variables and characteristics. Present the extracted data exclusively in JSON format, ensuring that the
    JSON object is comprehensive and contains all requested information. Avoid any form of data truncation or
    summarization, and ensure that the response is strictly limited to the JSON object without any 
    supplementary text. The JSON should follow this structure: {{'SPACE':{{'{VAR}':{{'value_boundaries': 
    VALUE_BOUNDARIES, 'equation': EQUATION, 'order_number':ORDER_NUMBER, 'frequency': FREQUENCY }} }} }}""",
    #   Extracting the list of model level variables
    "P8": """Analyze the provided ABM text to identify and extract the complete list of model-level variables, 
    variable data type, and initial value. Please ensure you extract all variables and characteristics only for 
    Model-level like step. Present the extracted data exclusively in JSON format, ensuring that the JSON object 
    is comprehensive and contains all requested information. Avoid any form of data truncation or summarization,
    and ensure that the response is strictly limited to the JSON object without any supplementary text. The 
    JSON should follow this structure: {{'Model-Level':{{VAR1:{{'short_description':SHORT_DESCRIPTION, 'data_type':
    DATA_TYPE, 'initial_value':INITIAL_VALUE, }}, VAR2 :{{...}} }} }}""",
    #   Extracting the characteristics of each model variable
    #   Input: '{VAR}'
    "P9": """Analyze the provided ABM text to identify and extract the value boundaries, equation, order of 
          execution, and frequency of execution related to the '{VAR}' variable of Model-level variables. Please 
          ensure you extract all variables and characteristics. Present the extracted data exclusively in JSON format,
           ensuring that the JSON object is comprehensive and contains all requested information. Avoid any form of 
          data truncation or summarization, and ensure that the response is strictly limited to the JSON object 
          without any supplementary text. The JSON should follow this structure: {{'Model-Level':{{'{VAR}':
          {{'value_boundaries':VALUE_BOUNDARIES, 'equation': EQUATION, 'order_number':ORDER_NUMBER, 
          'frequency': FREQUENCY }} }} }}""",
}

VecList = ["TFIDF", "OpenSource", "SentenceTransformer", "GPT40"]
SentenceTransformerModels = ['sentence-transformers/all-MiniLM-L6-v2']

PsAndValues = {
    "P1": {
        "PovertyTxT": {},
        "Poverty": {},
        "PovertyPDF": {},
        "Trout": {},
        "TroutPDF": {},
        "Tiger": {},
        "Elites": {}
    },
    "P2": {
        "PovertyTxT": {},
        "Poverty": {},
        "PovertyPDF": {},
        "Trout": {},
        "TroutPDF": {},
        "Tiger": {},
        "Elites": {}
    },
    # Input: {AGENT_SET}
    "P3": {
        "PovertyTxT": {
            "AGENT_SET": "HouseHold"
        },
        "Poverty": {
            "AGENT_SET": "HouseHold"
        },
        "PovertyPDF": {
            "AGENT_SET": "HouseHold"
        },
        "Trout": {
            "AGENT_SET": "trout"
        },
        "TroutPDF": {
            "AGENT_SET": "trout"
        },
        "Tiger": {
            "AGENT_SET": "Female Tigers"
        },
        "Elites": {
            "AGENT_SET": "Humanitarian Advocates"
        }
    },
    # Input: '{AGENT_SET}', '{VAR}'
    "P4": {
        "PovertyTxT": {
            "AGENT_SET": "HouseHold",
            "VAR": "income"
        },
        "Poverty": {
            "AGENT_SET": "HouseHold",
            "VAR": "income"
        },
        "PovertyPDF": {
            "AGENT_SET": "HouseHold",
            "VAR": "income"
        },
        "Trout": {
            "AGENT_SET": "trout",
            "VAR": "weight"
        },
        "TroutPDF": {
            "AGENT_SET": "trout",
            "VAR": "weight"
        },
        "Tiger": {
            "AGENT_SET": "Female Tigers",
            "VAR": "Age"
        },
        "Elites": {
            "AGENT_SET": "Humanitarian Advocates",
            "VAR": "advocate_zeal"
        }
    },
    "P5": {
        "PovertyTxT": {},
        "Poverty": {},
        "PovertyPDF": {},
        "Trout": {},
        "TroutPDF": {},
        "Tiger": {},
        "Elites": {}
    },
    "P6": {
        "PovertyTxT": {},
        "Poverty": {},
        "PovertyPDF": {},
        "Trout": {},
        "TroutPDF": {},
        "Tiger": {},
        "Elites": {}
    },
    #   Input: '{VAR}'
    "P7": {
        "PovertyTxT": {
            "VAR": "cell coordinates"
        },
        "Poverty": {
            "VAR": "cell coordinates"
        },
        "PovertyPDF": {
            "VAR": "cell coordinates"
        },
        "Trout": {
            "VAR": "Cell depth"
        },
        "TroutPDF": {
            "VAR": "Cells"
        },
        "Tiger": {
            "VAR": "cell coordinates"
        },
        "Elites": {
            "VAR": ""
        }
    },
    "P8": {
        "PovertyTxT": {},
        "Poverty": {},
        "PovertyPDF": {},
        "Trout": {},
        "TroutPDF": {},
        "Tiger": {},
        "Elites": {}
    },
    # Input: '{VAR}'
    "P9": {
        "PovertyTxT": {
            "VAR": "income_mean"
        },
        "Poverty": {
            "VAR": "income_mean"
        },
        "PovertyPDF": {
            "VAR": "income_mean"
        },
        "Trout": {
            "VAR": "habDriftRegenDist"
        },
        "TroutPDF": {
            "VAR": "habDriftRegenDist"
        },
        "Tiger": {
            "VAR": "total_population"
        },
        "Elites": {
            "VAR": "percent-elites"
        }
    },
}

PsAndOutputs = {
    "P1": {
        "PovertyPDF": """
                        {
                 "Model_Purpose": {
                 "full_description": "The primary objective of this study is to use Agent-Based Modeling (ABM) to examine the correlation between income levels and poverty levels by simulating the complex demand behavior of households in the market. The model aims to provide a detailed analysis of economic factors that affect poverty and to facilitate deeper investigations into how income levels and price levels correlate, potentially leading to increased inflation and poverty. Additionally, the model seeks to offer fundamental analytical value by aligning its output with empirical data and discussing future extensions to cover a broader spectrum of economic complexities.",
                 "research_questions": ["How do income levels influence poverty levels in a nation?", "What is the correlation between income and the price of a basket of goods in determining Poverty Lines (PL)?", "How does household demand behavior in the market affect macroeconomic outcomes related to poverty?"],
                 "system_boundaries": [
                 "The model focuses on the demand-side behavior of individuals with assumptions of limitless supply and a fixed price point.",
                "The supply subsystem and government are assumed to be exogenous systems."],
                 "outcome_variables": {
                 "income_mean": "The average income level of agents (households) in the simulation, directly influencing their demand.",
                 "citizen_desired_demand": "The preferred level of consumption by agents, representing the value of the basket of goods.",
                 "current_demand_level": "The actual level of consumption achieved by agents, indicating the satisfaction of their demand.",
                 "savings": "The amount of income saved by agents after meeting their demand, divided into regular savings and essential investment savings (EIG).",
                 "disposable_income": "The remaining income of agents after subtracting their current demand level, used to measure the potential for increased consumption or savings."
                 }
                 }
                }
        """,
        "TroutPDF": """
            {
                "Model_Purpose": {
                    "full_description": "InSTREAM-Gen is particularly suited to simulate the eco-evolutionary consequences of river management decisions for fish population dynamics and persistence under a climate change context. It is an eco-genetic version of inSTREAM with genetic transmission of traits to allow evolutionary changes in trout life history.",
                    "research_questions": [
                        "What are the demographic and evolutionary dynamics under warming resulting from climate change?",
                        "What are the demographic and evolutionary dynamics under climate change-induced warming plus stream flow reduction resulting from land use change?",
                        "How does combining warming with land use change impact trout populations compared to a baseline scenario without environmental change?"
                    ],
                    "system_boundaries": [
                        "The model represents one reach of a stream typically a few hundred meters in length.",
                        "Includes entities such as cells, trout, and redds."
                    ],
                    "outcome_variables": {
                        "abundance": "Controls the relation between fish condition and starvation probability.",
                        "biomass": "Total energy available to the population influenced by various parameters.",
                        "genotypic values": "Heritable traits influenced by various parameters including food availability and temperature.",
                        "length-at-emergence": "Evolving trait crucial for demographic output influenced by parameters such as size at emergence.",
                        "size maturity threshold": "Heritable trait defined by various parameters influencing spawning and growth."
                    }
                }
            }
        """,
        "Tiger": """
            {
                "Model_Purpose": {
                    "full_description": "The proximate purpose of the model is to predict the dynamics of the number, location, and size of tiger territories in response to habitat quality and tiger density. To allow for predictions to new conditions, for which no data exist, territories are not imposed but emerge from the tigers’ perception of habitat quality and from their interactions with each other. Tiger population dynamics is deduced from merging territory dynamics with observed demographic rates. The ultimate purpose of the model is to explore human-tiger interactions and assess threats to tiger populations across contexts and scales. The model can thus be used to better inform decision makers on how to conserve tigers under uncertain and changing future conditions.",
                    "research_questions": [
                        "How do habitat quality and tiger density affect the number, location, and size of tiger territories?",
                        "How do tiger territories emerge based on tigers’ perception of habitat quality and their interactions?",
                        "How can the model be used to predict tiger population dynamics under new conditions?",
                        "What are the impacts of various threats, such as poaching and habitat change, on tiger populations?",
                        "How do male and female tiger interactions influence territory and population dynamics?"
                    ],
                    "system_boundaries": [
                        "Spatial boundaries of habitat cells (250 m sides) within Chitwan National Park, Nepal",
                        "Temporal boundaries up to 20 years with each time step equivalent to 1 month",
                        "Initial population size and structure based on observed data from Chitwan National Park"
                    ],
                    "outcome_variables": {
                        "number_of_territories": "The total count of distinct territories established by tigers.",
                        "territory_location": "The spatial coordinates representing the center of tiger territories.",
                        "territory_size": "The area covered by tiger territories, influenced by prey availability and interactions.",
                        "population_size": "The overall size of the tiger population within the simulation landscape.",
                        "reproduction_rate": "Rate of births within the tiger population, affected by territory quality and tiger density.",
                        "mortality_rate": "Rate of deaths within the tiger population, including causes such as infanticide or territorial fights.",
                        "prey_biomass": "The amount of prey available in each habitat cell, influencing territory size and location."
                    }
                }
            }
        """,
        "Elites": """
            {
                "Model_Purpose": {
                    "full_description": "The purpose of this model is to understand the impact of elites in the spread of norms as an extension of a previous model. The model explores the diffusion of non-punishment-enforced humanitarian norms through an artificial society using an agent-based modeling approach. It examines how networked elites and norm advocates influence general individuals' adoption of the norm through interactions and peer pressure. The overall objective is to test assumptions of norm diffusion theories in an environment where peer pressure is the only mechanism for enforcement of norm adoption.",
                    "research_questions": [
                        "How do networked elites influence the spread of humanitarian norms in a society?",
                        "What impact does peer pressure have on the adoption of non-punishment-enforced norms?",
                        "Can we understand the underlying mechanisms of norm lock-in and explore ways to dismantle it for greater social good?"
                    ],
                    "system_boundaries": [],
                    "outcome_variables": {
                        "percent_elites": "Percentage of population who are elites (influencers)",
                        "advocate_weight_e": "Weight advocate exerts on the elite agent’s humanitarianism values",
                        "peer_weight_e": "Weight placed on interactions with other elite agents",
                        "network_size": "Randomized size of network of ‘followers’ of elites",
                        "advocate_weight_g": "Weight the advocate exerts on general agents",
                        "elite_weight": "Weight given to a networked elite’s norm adoption",
                        "general_weight": "Weight given to a one-on-one interaction with a general agent",
                        "peer_weight_g": "Weight given to average humanitarian values of a group of nearby agents",
                        "percent_advocates": "What percentage of agent population are advocates who promote a norm",
                        "advocate_zeal": "Homogeneous value given to advocate agents’ humanitarianism values",
                        "degrade_percent": "Amount per time step that an agent degrades its humanitarianism values, to represent donor/compassion fatigue"
                    }
                }
            }
        """
    },
    "P2": {
        "PovertyTxT": """
                {
            "Households": {
                "short_description": "Agents representing everyday purchasing decisions shaping aggregate economic demand.",
                "agent_role": "To simulate demand-side behavior in the modeled economy by adjusting their consumption, savings, and investment based on income and demand gaps."
            }
        }
        """,
        "Poverty": """
                    {
                "Households": {
                    "short_description": "Agents representing everyday purchasing decisions shaping aggregate economic demand.",
                    "agent_role": "To simulate demand-side behavior in the modeled economy by adjusting their consumption, savings, and investment based on income and demand gaps."
                }
            }
        """,
        "PovertyPDF": """
                {
            "Households": {
                "short_description": "Agents representing everyday purchasing decisions shaping aggregate economic demand.",
                "agent_role": "To simulate demand-side behavior in the modeled economy by adjusting their consumption, savings, and investment based on income and demand gaps."
            }
        }
        """,
        "Trout": """
                {
            
                    "Trout": {
                        "short_description": "Individuals with unique values of body length, weight, and condition including phenotypic and genotypic values for heritable traits",
                        "agent_role": "Modelled as individuals which can grow, spawn, select habitat, and have survival rates influenced by different environmental and biological factors."
                    },
                    "Redds": {
                        "short_description": "Spawning nests made by trout containing eggs with the genetic information of the female and male spawners",
                        "agent_role": "Responsible for the genetic transmission of traits to offspring; egg survival and development is subject to environmental conditions"
                    }
                }
        """,
        "TroutPDF": """
                {
            
                    "Trout": {
                        "short_description": "Individuals with unique values of body length, weight, and condition including phenotypic and genotypic values for heritable traits",
                        "agent_role": "Modelled as individuals which can grow, spawn, select habitat, and have survival rates influenced by different environmental and biological factors."
                    },
                    "Redds": {
                        "short_description": "Spawning nests made by trout containing eggs with the genetic information of the female and male spawners",
                        "agent_role": "Responsible for the genetic transmission of traits to offspring; egg survival and development is subject to environmental conditions"
                    }
                }
        """,
        "Tiger": """
            {
                "Female Tigers": {
                    "short_description": "Female tigers in the model represent the primary reproductive agents. They establish and expand territories based on prey availability and the presence of other females.",
                    "agent_role": "Females contribute to the reproduction and territorial dynamics by selecting and maintaining territories, gestating cubs, and caring for them until they disperse."
                },
                "Male Tigers": {
                    "short_description": "Male tigers are modeled with behaviors that are largely determined by their need to overlap territories with multiple females to maximize mating opportunities.",
                    "agent_role": "Males play a crucial role in maintaining genetic diversity and population stability. They establish territories that encompass those of females and engage in competitive interactions with other males to gain and defend access to females."
                } 
            }
        """,
        "Elites": """
                {
                    "Humanitarian Advocates": {
                        "short_description": "Humanitarian advocates work enthusiastically in the initial phase of the norm lifecycle to promote the norm to elites and other individuals. They have already internalized the humanitarian norm through various personal experiences or encounters.",
                        "agent_role": "Advocates operate like the general population but specifically seek out elites who have not adopted the norm to convince. They never adjust their level of enthusiasm for the norm."
                    },
                    "Elites": {
                        "short_description": "Elites include famous, influential people such as actors, athletes, religious figures, and other types of mavens with large social networks.",
                        "agent_role": "Elites attempt to convince political leaders, disperse information, and influence others through their networks. They may change their mind during norm adoption and primarily interact with advocates."
                    },
                    "General Population": {
                        "short_description": "General Population represents the most common agents who act based on the logic of consequences and seek to maximize their utility.",
                        "agent_role": "General individuals consider interactions with all agent types to change their opinion both for and against norms. They are influenced greatly by peer pressure and interactions with nearby agents."
                    }
                }
        """
    },
    # Input: {AGENT_SET}
    "P3": {
        "PovertyTxT": """
        {
    "HouseHold": {
        "income_mean": {
            "short_description": "Income mean of the Household agent",
            "data_type": "Variable",
            "initial_value": "f(time)"
        },
        "income_std": {
            "short_description": "Standard deviation of income for the Household agent",
            "data_type": "Constant",
            "initial_value": "10"
        },
        "citizen_desired_demand": {
            "short_description": "Constant value representing desired demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "50"
        },
        "citizen_required_demand": {
            "short_description": "Constant value determining required demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "citizen_desired_demand * 2"
        },
        "income": {
            "short_description": "Income of the Household agent",
            "data_type": "Variable",
            "initial_value": "Gaussian(income_mean, income_std)"
        },
        "eig_to_income_ratio": {
            "short_description": "Ratio of Essential Investment Goods (EIG) to income for the Household agent",
            "data_type": "Constant",
            "initial_value": "0.25"
        },
        "median_income_level": {
            "short_description": "Median income level of the Household agent",
            "data_type": "Variable",
            "initial_value": "Median point of all incomes"
        },
        "EIG_Price": {
            "short_description": "Price of Essential Investment Goods (EIG) for the Household agent",
            "data_type": "Variable",
            "initial_value": "median_income_level * eig_to_income_ratio"
        },
        "demand_gap": {
            "short_description": "Gap between desired demand and current demand level for the Household agent",
            "data_type": "Variable",
            "initial_value": "citizen_desired_demand - current_demand_level"
        },
        "disposable_income": {
            "short_description": "Remaining income after consumption for the Household agent",
            "data_type": "Variable",
            "initial_value": "income - current_demand_level"
        }
    }
}""",
        "Poverty": """
        {
    "HouseHold": {
        "income_mean": {
            "short_description": "Income mean of the Household agent",
            "data_type": "Variable",
            "initial_value": "f(time)"
        },
        "income_std": {
            "short_description": "Standard deviation of income for the Household agent",
            "data_type": "Constant",
            "initial_value": "10"
        },
        "citizen_desired_demand": {
            "short_description": "Constant value representing desired demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "50"
        },
        "citizen_required_demand": {
            "short_description": "Constant value determining required demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "citizen_desired_demand * 2"
        },
        "income": {
            "short_description": "Income of the Household agent",
            "data_type": "Variable",
            "initial_value": "Gaussian(income_mean, income_std)"
        },
        "eig_to_income_ratio": {
            "short_description": "Ratio of Essential Investment Goods (EIG) to income for the Household agent",
            "data_type": "Constant",
            "initial_value": "0.25"
        },
        "median_income_level": {
            "short_description": "Median income level of the Household agent",
            "data_type": "Variable",
            "initial_value": "Median point of all incomes"
        },
        "EIG_Price": {
            "short_description": "Price of Essential Investment Goods (EIG) for the Household agent",
            "data_type": "Variable",
            "initial_value": "median_income_level * eig_to_income_ratio"
        },
        "demand_gap": {
            "short_description": "Gap between desired demand and current demand level for the Household agent",
            "data_type": "Variable",
            "initial_value": "citizen_desired_demand - current_demand_level"
        },
        "disposable_income": {
            "short_description": "Remaining income after consumption for the Household agent",
            "data_type": "Variable",
            "initial_value": "income - current_demand_level"
        }
    }
}""",
        "PovertyPDF": """
        {
    "HouseHold": {
        "income_mean": {
            "short_description": "Income mean of the Household agent",
            "data_type": "Variable",
            "initial_value": "f(time)"
        },
        "income_std": {
            "short_description": "Standard deviation of income for the Household agent",
            "data_type": "Constant",
            "initial_value": "10"
        },
        "citizen_desired_demand": {
            "short_description": "Constant value representing desired demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "50"
        },
        "citizen_required_demand": {
            "short_description": "Constant value determining required demand for the Household agent",
            "data_type": "Constant",
            "initial_value": "citizen_desired_demand * 2"
        },
        "income": {
            "short_description": "Income of the Household agent",
            "data_type": "Variable",
            "initial_value": "Gaussian(income_mean, income_std)"
        },
        "eig_to_income_ratio": {
            "short_description": "Ratio of Essential Investment Goods (EIG) to income for the Household agent",
            "data_type": "Constant",
            "initial_value": "0.25"
        },
        "median_income_level": {
            "short_description": "Median income level of the Household agent",
            "data_type": "Variable",
            "initial_value": "Median point of all incomes"
        },
        "EIG_Price": {
            "short_description": "Price of Essential Investment Goods (EIG) for the Household agent",
            "data_type": "Variable",
            "initial_value": "median_income_level * eig_to_income_ratio"
        },
        "demand_gap": {
            "short_description": "Gap between desired demand and current demand level for the Household agent",
            "data_type": "Variable",
            "initial_value": "citizen_desired_demand - current_demand_level"
        },
        "disposable_income": {
            "short_description": "Remaining income after consumption for the Household agent",
            "data_type": "Variable",
            "initial_value": "income - current_demand_level"
        }
    }
}""",
        "Trout": """
        {
    "Trout": {
        "body_length": {
            "short_description": "Unique body length of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "weight": {
            "short_description": "Unique weight of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "condition": {
            "short_description": "Body condition of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "phenotypic_value_size_emergence": {
            "short_description": "Phenotypic value for size at emergence",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "genotypic_value_size_emergence": {
            "short_description": "Genotypic value for size at emergence",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "phenotypic_value_size_maturity_threshold": {
            "short_description": "Phenotypic value for size maturity threshold",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "genotypic_value_size_maturity_threshold": {
            "short_description": "Genotypic value for size maturity threshold",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "starvation_probability": {
            "short_description": "Probability of starvation",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "high_temperature_mortality": {
            "short_description": "Mortality due to high temperatures",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "high_velocity_mortality": {
            "short_description": "Mortality due to high water velocities",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "stranding_mortality": {
            "short_description": "Mortality due to stranding",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "terrestrial_predation_mortality": {
            "short_description": "Mortality due to predation by terrestrial animals",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "aquatic_predation_mortality": {
            "short_description": "Mortality due to predation by piscivorous trout",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "angler_mortality": {
            "short_description": "Mortality due to angling",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "survival_probability": {
            "short_description": "Daily survival probability",
            "data_type": "float",
            "initial_value": "derived from various mortality probabilities"
        },
        "fitness_measure": {
            "short_description": "Expected Maturity (EM)",
            "data_type": "float",
            "initial_value": "calculated using growth and predation risk"
        },
        "growth_rate": {
            "short_description": "Daily growth rate",
            "data_type": "float",
            "initial_value": "calculated based on bioenergetics modeling"
        },
        "food_intake": {
            "short_description": "Daily food intake",
            "data_type": "float",
            "initial_value": "calculated based on habitat characteristics"
        },
        "metabolic_costs": {
            "short_description": "Daily metabolic costs",
            "data_type": "float",
            "initial_value": "calculated based on trout size, swimming speed, and water temperature"
        }
    }
}
        """,
        "TroutPDF": """
        {
    "Trout": {
        "body_length": {
            "short_description": "Unique body length of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "weight": {
            "short_description": "Unique weight of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "condition": {
            "short_description": "Body condition of the trout",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "phenotypic_value_size_emergence": {
            "short_description": "Phenotypic value for size at emergence",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "genotypic_value_size_emergence": {
            "short_description": "Genotypic value for size at emergence",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "phenotypic_value_size_maturity_threshold": {
            "short_description": "Phenotypic value for size maturity threshold",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "genotypic_value_size_maturity_threshold": {
            "short_description": "Genotypic value for size maturity threshold",
            "data_type": "float",
            "initial_value": "varies per individual"
        },
        "starvation_probability": {
            "short_description": "Probability of starvation",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "high_temperature_mortality": {
            "short_description": "Mortality due to high temperatures",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "high_velocity_mortality": {
            "short_description": "Mortality due to high water velocities",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "stranding_mortality": {
            "short_description": "Mortality due to stranding",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "terrestrial_predation_mortality": {
            "short_description": "Mortality due to predation by terrestrial animals",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "aquatic_predation_mortality": {
            "short_description": "Mortality due to predation by piscivorous trout",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "angler_mortality": {
            "short_description": "Mortality due to angling",
            "data_type": "float",
            "initial_value": "calculated per individual"
        },
        "survival_probability": {
            "short_description": "Daily survival probability",
            "data_type": "float",
            "initial_value": "derived from various mortality probabilities"
        },
        "fitness_measure": {
            "short_description": "Expected Maturity (EM)",
            "data_type": "float",
            "initial_value": "calculated using growth and predation risk"
        },
        "growth_rate": {
            "short_description": "Daily growth rate",
            "data_type": "float",
            "initial_value": "calculated based on bioenergetics modeling"
        },
        "food_intake": {
            "short_description": "Daily food intake",
            "data_type": "float",
            "initial_value": "calculated based on habitat characteristics"
        },
        "metabolic_costs": {
            "short_description": "Daily metabolic costs",
            "data_type": "float",
            "initial_value": "calculated based on trout size, swimming speed, and water temperature"
        }
    }
}
        """,
        "Tiger": """
            {
                "Female Tigers": {
                    "Age": {
                        "short_description": "Age in months",
                        "data_type": "integer",
                        "initial_value": "1–180"
                    },
                    "Fertile?": {
                        "short_description": "Indicates whether female is fertile",
                        "data_type": "boolean",
                        "initial_value": "True/false"
                    },
                    "Gestating?": {
                        "short_description": "Indicates whether female is gestating",
                        "data_type": "boolean",
                        "initial_value": "True/false"
                    },
                    "Males-in-my-territory": {
                        "short_description": "Identities of males overlapping female territory",
                        "data_type": "set",
                        "initial_value": "Set of male identities"
                    },
                    "My-mom": {
                        "short_description": "Identity of mom",
                        "data_type": "identifier",
                        "initial_value": "Identity of female tiger"
                    },
                    "My-offspring": {
                        "short_description": "Number of offspring in current litter",
                        "data_type": "integer",
                        "initial_value": "1–5"
                    },
                    "Natal-origin": {
                        "short_description": "Cell where female was initialized at or the centroid cell of mother’s territory",
                        "data_type": "coordinates",
                        "initial_value": "0 – max X, 0 – max Y"
                    },
                    "Num-litters": {
                        "short_description": "Total number of litters the female has had up until current time",
                        "data_type": "integer",
                        "initial_value": "0 – max number of litters over lifetime"
                    },
                    "Age-class": {
                        "short_description": "Indicates development stage of female",
                        "data_type": "categorical",
                        "initial_value": "Cub, Juvenile, Transient, or Breeder"
                    },
                    "Territory": {
                        "short_description": "Set of cells belonging to territory",
                        "data_type": "set",
                        "initial_value": "Set of cell coordinates"
                    },
                    "Terr-orig": {
                        "short_description": "Cell that female was initialized at or first cell of territory",
                        "data_type": "coordinates",
                        "initial_value": "0 – max X, 0 – max Y"
                    },
                    "T-gestation": {
                        "short_description": "Indicates how long female has gestated",
                        "data_type": "integer",
                        "initial_value": "0-3 or 4"
                    },
                    "T-parenting": {
                        "short_description": "Indicates how long female has been a parent of current litter",
                        "data_type": "integer",
                        "initial_value": "0-24"
                    }
            }
        }
        """,
        "Elites": """
            {
                "Humanitarian Advocates": {
                    "percent_advocates": {
                                "short_description": "What percentage of agent population are advocates who promote a norm",
                                "data_type": "float",
                                "initial_value": 0.0
                            },
                            "advocate_zeal": {
                                "short_description": "Homogeneous value given to advocate agents’ humanitarianism values",
                                "data_type": "float",
                                "initial_value": 1.0
                            },
                            "advocate_weight_e": {
                                "short_description": "Weight advocate exerts on the elite agent’s humanitarianism values",
                                "data_type": "float",
                                "initial_value": 0.5
                            },
                            "advocate_weight_g": {
                                "short_description": "Weight the advocate exerts on general agents",
                                "data_type": "float",
                                "initial_value": 0.5
                            }
                }
            }
        """
    },
    # Input: '{AGENT_SET}', '{VAR}'
    "P4": {
        "PovertyTxT": """
                {
                    "HouseHold": {
                        "income": {
                            "value_boundaries": "0-200",
                            "equation": "income = Gaussian(µ = income_mean, σ = income_std)",
                            "order_number": 1,
                            "frequency": "every tick (per year)"
                        }
                    }
                }
        """,
        "Poverty": """
                {
                    "HouseHold": {
                        "income": {
                            "value_boundaries": "0-200",
                            "equation": "income = Gaussian(µ = income_mean, σ = income_std)",
                            "order_number": 1,
                            "frequency": "every tick (per year)"
                        }
                    }
                }
        """,
        "PovertyPDF": """
                {
                    "HouseHold": {
                        "income": {
                            "value_boundaries": "0-200",
                            "equation": "income = Gaussian(µ = income_mean, σ = income_std)",
                            "order_number": 1,
                            "frequency": "every tick (per year)"
                        }
                    }
                }
        """,
        "Trout":  """
            {
                "trout": {
                    "weight": {
                        "value_boundaries": "Not specified",
                        "equation": "Fish growth is modelled as net energy intake, the difference between energy intake from food and metabolic costs.",
                        "order_number": "5",
                        "frequency": "Daily"
                    }
                }
            }
        """,
        "TroutPDF": """
            {
                "trout": {
                    "weight": {
                        "value_boundaries": "Not specified",
                        "equation": "Fish growth is modelled as net energy intake, the difference between energy intake from food and metabolic costs.",
                        "order_number": "5",
                        "frequency": "Daily"
                    }
                }
            }
        """,
        "Tiger_weight": """
                    {
                        "Female Tigers": {
                            "weight": {
                                "value_boundaries": "Based on empirical data from female tigers in Chitwan, the lower limit is 76 kg/month, and the upper limit is 167.3 kg/month, calculated by the daily consumption rates of 5 to 6 kg/day.",
                                "equation": "The female tigers achieve basal metabolic energy demands when they have access to 76 kg/month within their territory. The maximum consumption is capped at 167.3 kg/month.",
                                "order_number": 2,
                                "frequency": "1 month step"
                            }
                        }
                    }
        """,
        "Tiger": """
                    {
                      "Female Tigers": {
                        "Age": {
                          "value_boundaries": [1, 180],
                          "equation": "Age increases by 1 month at each time step",
                          "order_number": 2,
                          "frequency": "1 time step (1 month)"
                        }
                      }
                    }
        """,
        "Elites": """
                {
                    "Humanitarian Advocates": {
                        "advocate_zeal": {
                            "value_boundaries": "[0, 1]",
                            "equation": "The exact equation is not provided in the text, but the variable appears in regression models indicating its effect on norm adoption.",
                            "order_number": "Not specified in the text",
                            "frequency": "The text suggests that agents update their opinions each time step, implying that 'advocate zeal' is considered in every time step."
                        }
                    }
                }
        """
    },
    "P5": {
        "PovertyTxT": """
            {
                "Space": {
                    "short_description": "The simulation space represents the demand sub-system of the market, focusing on household
                     demand behavior towards consumption and savings amidst different income growth functions. This model replicates
                      the interactions between income levels and demand levels in an economy",
                    "type": "Grid"
                }
            }
        """,
        "Poverty": """
            {
                "Space": {
                    "short_description": "The simulation space represents the demand sub-system of the market, focusing on household
                     demand behavior towards consumption and savings amidst different income growth functions. This model replicates
                      the interactions between income levels and demand levels in an economy",
                    "type": "Grid"
                }
            }
        """,
        "PovertyPDF": """
            {
                "Space": {
                    "short_description": "The simulation space represents the demand sub-system of the market, focusing on household
                     demand behavior towards consumption and savings amidst different income growth functions. This model replicates
                      the interactions between income levels and demand levels in an economy",
                    "type": "Grid"
                }
            }
        """,
        "Trout": """
            {
                "Space": {
                    "short_description": "The model represents one reach of a stream typically a few hundred meters in length. It includes entities such as cells, trout, and redds. Cells are objects that represent patches of relatively uniform habitat within the reach.",
                    "type": "spatially explicit, individual-based eco-genetic structure"
                }
            }
        """,
        "TroutPDF": """
            {
                "Space": {
                    "short_description": "The model represents one reach of a stream typically a few hundred meters in length. It includes entities such as cells, trout, and redds. Cells are objects that represent patches of relatively uniform habitat within the reach.",
                    "type": "spatially explicit, individual-based eco-genetic structure"
                }
            }
        """,
        "Tiger": """
                {
                    "Space": {
                        "short_description": "The simulations were carried out on landscapes of different sizes: a small landscape of 40 × 40 cells (100 km²), a larger landscape of 128 × 125 cells (1000 km²), and the Chitwan landscape of 157 × 345 cells (3385 km², though only 1239 km² of it comprises the park). The boundaries in the model landscapes were impermeable.",
                        "type": "Spatially explicit landscape with varying sizes and impermeable boundaries."
                    }
                }
        """,
        "Elites": """
            No space information provided.
        """
    },
    "P6": {
        "PovertyTxT": """
        No space information provided.
        """,
        "Poverty": """No space information provided.
        """,
        "PovertyPDF": """No space information provided. """,
        "Trout": """
            {
                "SPACE": {
                    "daily_flow": {
                        "short_description": "Daily flow values of the reach",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "temperature": {
                        "short_description": "Daily temperature values of the reach",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "depth": {
                        "short_description": "Depth of each cell",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "velocity": {
                        "short_description": "Velocity of each cell",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "food_availability": {
                        "short_description": "Food availability in each cell",
                        "data_type": "float",
                        "initial_value": "calculated from area, depth, velocity and food parameters"
                    }
                }
            }
        """,
        "TroutPDF": """
            {
                "SPACE": {
                    "daily_flow": {
                        "short_description": "Daily flow values of the reach",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "temperature": {
                        "short_description": "Daily temperature values of the reach",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "depth": {
                        "short_description": "Depth of each cell",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "velocity": {
                        "short_description": "Velocity of each cell",
                        "data_type": "float",
                        "initial_value": "from input files"
                    },
                    "food_availability": {
                        "short_description": "Food availability in each cell",
                        "data_type": "float",
                        "initial_value": "calculated from area, depth, velocity and food parameters"
                    }
                }
            }
        """,
        "Tiger": """
        {
    "SPACE": {
        "Habitat_Cells": {
            "prey_density": {
                "short_description": "Density of prey in the habitat cell",
                "data_type": "float",
                "initial_value": "varies per cell"
            },
            "forest_cover": {
                "short_description": "Percentage of forest cover in the habitat cell",
                "data_type": "float",
                "initial_value": "varies per cell"
            },
            "water_access": {
                "short_description": "Access to water resources",
                "data_type": "binary",
                "initial_value": "0 or 1"
            },
            "human_density": {
                "short_description": "Density of human population in the habitat cell",
                "data_type": "float",
                "initial_value": "varies per cell"
            },
            "cell_coordinates": {
                "short_description": "Coordinates of the habitat cell in the landscape",
                "data_type": "tuple (int, int)",
                "initial_value": "determined at setup"
            },
            "cell_type": {
                "short_description": "Type of habitat in the cell (forest, grassland, etc.)",
                "data_type": "string",
                "initial_value": "determined at setup"
            }
        }
    }
}
        """,
        "Elites": """
            No space information provided.
        """
    },
    #   Input: '{VAR}'
    "P7": {
        "PovertyTxT": """No space information provided.""",
        "Poverty": """No space information provided.""",
        "PovertyPDF": """No space information provided.""",
        "Trout": """
                    {
                "SPACE": {
                    "cell depth": {
                        "value_boundaries": "mean depth and water velocity that are functions of stream flow area of velocity shelter for drift feeding spawning gravel area and mean distance to hiding cover",
                        "equation": "PHABSIM v.1.5.1 hydraulic models to predict depth and velocity in each cell at each simulated flow rate",
                        "order_number": 1,
                        "frequency": "daily"
                    }
                }
            }
        """,
        "TroutPDF": """
                    {
                "SPACE": {
                    "cell depth": {
                        "value_boundaries": "mean depth and water velocity that are functions of stream flow area of velocity shelter for drift feeding spawning gravel area and mean distance to hiding cover",
                        "equation": "PHABSIM v.1.5.1 hydraulic models to predict depth and velocity in each cell at each simulated flow rate",
                        "order_number": 1,
                        "frequency": "daily"
                    }
                }
            }
        """,
        "Tiger": """
                {
                    "SPACE": {
                        "Habitat_Cells": {
                            "value_boundaries": "2.05 kg/cell/month to 10.46 kg/cell/month",
                            "equation": "Value boundaries were obtained using a uniform distribution based on the Chitwan data",
                            "order_number": 5,
                            "frequency": "Every time step (1 month)"
                        }
                    }
                }
        """,
        "Elites": """
            No space information provided.
        """
    },
    "P8": {
        "PovertyTxT": """
                {
                    "Model-Level": {
                        "income_mean": {
                            "short_description": "Mean income",
                            "data_type": "Variable",
                            "initial_value": "f(time)"
                        },
                        "income_std": {
                            "short_description": "Income standard deviation",
                            "data_type": "Constant",
                            "initial_value": 10
                        },
                        "citizen_desired_demand": {
                            "short_description": "Desired demand of citizens",
                            "data_type": "Constant",
                            "initial_value": 50
                        },
                        "citizen_required_demand": {
                            "short_description": "Minimum required demand for citizens",
                            "data_type": "Constant",
                            "initial_value": "citizen_desired_demand/2"
                        },
                        "Steps": {
                            "short_description": "Time unit of the model",
                            "data_type": "Variable",
                            "initial_value": "years (i.e., one year per step/tick)"
                        },
                        "income_dir_change_step": {
                            "short_description": "Income pattern turnover step",
                            "data_type": "Constant",
                            "initial_value": 200
                        },
                        "growth_direction": {
                            "short_description": "Direction of the income growth",
                            "data_type": "Constant",
                            "initial_value": "1 or -1"
                        },
                        "n_citizen": {
                            "short_description": "Number of citizens",
                            "data_type": "Constant",
                            "initial_value": 100
                        },
                        "emergency_savings_preference": {
                            "short_description": "Salaries meant to be saved in savings",
                            "data_type": "Constant",
                            "initial_value": 5
                        },
                        "eig_to_income_ratio": {
                            "short_description": "Annual income needed to buy property",
                            "data_type": "Constant",
                            "initial_value": 4
                        },
                        "share_eig_in_demands": {
                            "short_description": "Portion of rents in the basket of goods",
                            "data_type": "Constant",
                            "initial_value": 0.3
                        },
                        "median_income_level": {
                            "short_description": "Median point of all incomes",
                            "data_type": "Variable",
                            "initial_value": ""
                        },
                        "EIG_Price": {
                            "short_description": "Price of Essential Investment Goods",
                            "data_type": "Variable",
                            "initial_value": "median_income_level * eig_to_income_ratio"
                        }
                    }
                }
        """,
        "Poverty": """
                {
                    "Model-Level": {
                        "income_mean": {
                            "short_description": "Mean income",
                            "data_type": "Variable",
                            "initial_value": "f(time)"
                        },
                        "income_std": {
                            "short_description": "Income standard deviation",
                            "data_type": "Constant",
                            "initial_value": 10
                        },
                        "citizen_desired_demand": {
                            "short_description": "Desired demand of citizens",
                            "data_type": "Constant",
                            "initial_value": 50
                        },
                        "citizen_required_demand": {
                            "short_description": "Minimum required demand for citizens",
                            "data_type": "Constant",
                            "initial_value": "citizen_desired_demand/2"
                        },
                        "Steps": {
                            "short_description": "Time unit of the model",
                            "data_type": "Variable",
                            "initial_value": "years (i.e., one year per step/tick)"
                        },
                        "income_dir_change_step": {
                            "short_description": "Income pattern turnover step",
                            "data_type": "Constant",
                            "initial_value": 200
                        },
                        "growth_direction": {
                            "short_description": "Direction of the income growth",
                            "data_type": "Constant",
                            "initial_value": "1 or -1"
                        },
                        "n_citizen": {
                            "short_description": "Number of citizens",
                            "data_type": "Constant",
                            "initial_value": 100
                        },
                        "emergency_savings_preference": {
                            "short_description": "Salaries meant to be saved in savings",
                            "data_type": "Constant",
                            "initial_value": 5
                        },
                        "eig_to_income_ratio": {
                            "short_description": "Annual income needed to buy property",
                            "data_type": "Constant",
                            "initial_value": 4
                        },
                        "share_eig_in_demands": {
                            "short_description": "Portion of rents in the basket of goods",
                            "data_type": "Constant",
                            "initial_value": 0.3
                        },
                        "median_income_level": {
                            "short_description": "Median point of all incomes",
                            "data_type": "Variable",
                            "initial_value": ""
                        },
                        "EIG_Price": {
                            "short_description": "Price of Essential Investment Goods",
                            "data_type": "Variable",
                            "initial_value": "median_income_level * eig_to_income_ratio"
                        }
                    }
                }
        """,
        "PovertyPDF": """
                {
                    "Model-Level": {
                        "income_mean": {
                            "short_description": "Mean income",
                            "data_type": "Variable",
                            "initial_value": "f(time)"
                        },
                        "income_std": {
                            "short_description": "Income standard deviation",
                            "data_type": "Constant",
                            "initial_value": 10
                        },
                        "citizen_desired_demand": {
                            "short_description": "Desired demand of citizens",
                            "data_type": "Constant",
                            "initial_value": 50
                        },
                        "citizen_required_demand": {
                            "short_description": "Minimum required demand for citizens",
                            "data_type": "Constant",
                            "initial_value": "citizen_desired_demand/2"
                        },
                        "Steps": {
                            "short_description": "Time unit of the model",
                            "data_type": "Variable",
                            "initial_value": "years (i.e., one year per step/tick)"
                        },
                        "income_dir_change_step": {
                            "short_description": "Income pattern turnover step",
                            "data_type": "Constant",
                            "initial_value": 200
                        },
                        "growth_direction": {
                            "short_description": "Direction of the income growth",
                            "data_type": "Constant",
                            "initial_value": "1 or -1"
                        },
                        "n_citizen": {
                            "short_description": "Number of citizens",
                            "data_type": "Constant",
                            "initial_value": 100
                        },
                        "emergency_savings_preference": {
                            "short_description": "Salaries meant to be saved in savings",
                            "data_type": "Constant",
                            "initial_value": 5
                        },
                        "eig_to_income_ratio": {
                            "short_description": "Annual income needed to buy property",
                            "data_type": "Constant",
                            "initial_value": 4
                        },
                        "share_eig_in_demands": {
                            "short_description": "Portion of rents in the basket of goods",
                            "data_type": "Constant",
                            "initial_value": 0.3
                        },
                        "median_income_level": {
                            "short_description": "Median point of all incomes",
                            "data_type": "Variable",
                            "initial_value": ""
                        },
                        "EIG_Price": {
                            "short_description": "Price of Essential Investment Goods",
                            "data_type": "Variable",
                            "initial_value": "median_income_level * eig_to_income_ratio"
                        }
                    }
                }
        """,
        "Trout": """
            {
    "Model-Level": {
        "step": {
            "short_description": "Simulation step",
            "data_type": "integer",
            "initial_value": "1"
        },
        "habDriftRegenDist": {
            "short_description": "Habitat drift regeneration distance",
            "data_type": "float",
            "initial_value": "600.0"
        },
        "habDriftConc": {
            "short_description": "Drift concentration in habitat",
            "data_type": "float",
            "initial_value": "2.1E-10"
        },
        "habSearchProd": {
            "short_description": "Search food production rate in habitat",
            "data_type": "float",
            "initial_value": "4.8E-7"
        },
        "habPreyEnergyDensity": {
            "short_description": "Prey energy density in habitat",
            "data_type": "float",
            "initial_value": "5200"
        },
        "mortFishAqPredMin": {
            "short_description": "Minimum aquatic predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "0.984"
        },
        "mortFishTerrPredMin": {
            "short_description": "Minimum terrestrial predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "0.996"
        },
        "habDriftRegenDist_increase": {
            "short_description": "Increment in habitat drift regeneration distance",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habDriftConc_increase": {
            "short_description": "Increment in drift concentration in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habSearchProd_increase": {
            "short_description": "Increment in search food production rate in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habPreyEnergyDensity_increase": {
            "short_description": "Increment in prey energy density in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "mortFishAqPredMin_increase": {
            "short_description": "Increment in minimum aquatic predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "increment"
        },
        "mortFishTerrPredMin_increase": {
            "short_description": "Increment in minimum terrestrial predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "increment"
        }
    }
}
        """,
        "TroutPDF": """
        {
    "Model-Level": {
        "step": {
            "short_description": "Simulation step",
            "data_type": "integer",
            "initial_value": "1"
        },
        "habDriftRegenDist": {
            "short_description": "Habitat drift regeneration distance",
            "data_type": "float",
            "initial_value": "600.0"
        },
        "habDriftConc": {
            "short_description": "Drift concentration in habitat",
            "data_type": "float",
            "initial_value": "2.1E-10"
        },
        "habSearchProd": {
            "short_description": "Search food production rate in habitat",
            "data_type": "float",
            "initial_value": "4.8E-7"
        },
        "habPreyEnergyDensity": {
            "short_description": "Prey energy density in habitat",
            "data_type": "float",
            "initial_value": "5200"
        },
        "mortFishAqPredMin": {
            "short_description": "Minimum aquatic predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "0.984"
        },
        "mortFishTerrPredMin": {
            "short_description": "Minimum terrestrial predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "0.996"
        },
        "habDriftRegenDist_increase": {
            "short_description": "Increment in habitat drift regeneration distance",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habDriftConc_increase": {
            "short_description": "Increment in drift concentration in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habSearchProd_increase": {
            "short_description": "Increment in search food production rate in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "habPreyEnergyDensity_increase": {
            "short_description": "Increment in prey energy density in habitat",
            "data_type": "float",
            "initial_value": "increment"
        },
        "mortFishAqPredMin_increase": {
            "short_description": "Increment in minimum aquatic predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "increment"
        },
        "mortFishTerrPredMin_increase": {
            "short_description": "Increment in minimum terrestrial predation mortality risk for fish",
            "data_type": "float",
            "initial_value": "increment"
        }
    }
}
        """,
        "Tiger": """
        {
    "Model-Level": {
        "step": {
            "short_description": "Indicates each time step within the model, often representing a monthly interval in simulations.",
            "data_type": "integer",
            "initial_value": 0
        },
        "prey_biomass_production": {
            "short_description": "Prey biomass production rates (kg/month/cell) across the landscape.",
            "data_type": "float",
            "initial_value": "Calculated within 2.05 to 10.46 based on cell size"
        },
        "age_classes": {
            "short_description": "Categorizes tiger based on age stages: Breeding, Transient, Juvenile, Cub.",
            "data_type": "string",
            "initial_value": {
                "Breeding": "3+ years old",
                "Transient": "2-3 years old",
                "Juvenile": "1-2 years old",
                "Cub": "0-1 years old"
            }
        },
        "litter_size_distribution": {
            "short_description": "Distribution probabilities for different litter sizes.",
            "data_type": "float",
            "initial_value": {
                "1": 0,
                "2": 0.23,
                "3": 0.58,
                "4": 0.17,
                "5": 0.02
            }
        },
        "max_cells_per_time_step": {
            "short_description": "Maximum number of cells a female tiger can add to its territory per time step.",
            "data_type": "integer",
            "initial_value": 48
        },
        "annual_survival": {
            "short_description": "Annual survival rates for different tiger demographics.",
            "data_type": "float",
            "initial_value": {
                "Breeding male": 0.8,
                "Breeding female": 0.9,
                "Dispersal male": 0.65,
                "Transient male": 0.65,
                "Transient female": 0.7,
                "Juvenile": 0.9,
                "Cub": 0.6
            }
        },
        "annual_fecundity": {
            "short_description": "Probability that resident female tigers breed if fertile.",
            "data_type": "float",
            "initial_value": {
                "3-year old female": 0.9,
                "4+ years old female": 1
            }
        },
        "max_dispersal_distance": {
            "short_description": "Maximum possible dispersal distance from natal range for transient tigers.",
            "data_type": "float",
            "initial_value": {
                "Transient male": 66,
                "Transient female": 33
            }
        },
        "prey_thresholds": {
            "short_description": "Minimum and maximum prey biomass within a territory.",
            "data_type": "float",
            "initial_value": {
                "Minimum": 76,
                "Maximum": 167.3
            }
        },
        "dominant_female_prob": {
            "short_description": "Probability that a dominant female takes a territory cell from a subordinate female if the cell has the highest prey.",
            "data_type": "float",
            "initial_value": 0.25
        },
        "territory_prey_utilization": {
            "short_description": "Proportion of prey within a territory utilized by a female tiger.",
            "data_type": "float",
            "initial_value": 0.1
        },
        "male_search_radius": {
            "short_description": "Radius within which breeding males will search for nearby breeding females.",
            "data_type": "float",
            "initial_value": 3
        },
        "max_female_territories_overlap": {
            "short_description": "Maximum number of female territories a male can overlap.",
            "data_type": "integer",
            "initial_value": 6
        }
    }
}
        """,
        "Elites": """
        {
    "Model-Level": {
        "percent-elites": {
            "short_description": "Percentage of population who are elites (influencers)",
            "data_type": "float",
            "initial_value": [
                0,
                2
            ]
        },
        "advocate-weight-e": {
            "short_description": "Weight advocate exerts on the elite agent’s humanitarianism values",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "peer-weight-e": {
            "short_description": "Weight placed on interactions with other elite agents",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "network-size": {
            "short_description": "Randomized size of network of ‘followers’ of elites",
            "data_type": "integer",
            "initial_value": [
                0,
                500
            ]
        },
        "advocate-weight-g": {
            "short_description": "Weight the advocate exerts on general agents",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "elite-weight": {
            "short_description": "Weight given to a networked elite’s norm adoption",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "general-weight": {
            "short_description": "Weight given to a one-on-one interaction with a general agent",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "peer_weight_g": {
            "short_description": "Weight given to average humanitarian values of a group of nearby agents",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "percent-advocates": {
            "short_description": "What percentage of agent population are advocates who promote a norm",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "advocate-zeal": {
            "short_description": "Homogeneous value given to advocate agents’ humanitarianism values",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "degrade-percent": {
            "short_description": "Amount per time step that an agent degrades its humanitarianism values, to represent donor/compassion fatigue",
            "data_type": "float",
            "initial_value": [
                0,
                100
            ]
        },
        "mu-activist-state": {
            "short_description": "Mean humanitarianism value of all general agents (heterogeneous)",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "mu-activist-threshold": {
            "short_description": "Mean threshold value of all general agents beyond which they become active humanitarian actors (heterogeneous)",
            "data_type": "float",
            "initial_value": [
                0,
                1
            ]
        },
        "Population": {
            "short_description": "Total population of agents",
            "data_type": "integer",
            "initial_value": [
                500,
                1000
            ]
        }
    }
}
        """
    },
    # Input: '{VAR}'
    "P9": {
        "PovertyTxT": """
            {
                "Model-Level": {
                    "income_mean": {
                        "value_boundaries": "0-200",
                        "equation": {
                            "linear": "Income_Mean = a * time",
                            "logarithmic": "Income_Mean = 10 * LN(Time / 0.13)",
                            "exponential": "Income_Mean = 30 * e^(0.0206 * Time) - 30"
                        },
                        "order_number": "Update at the beginning of each tick",
                        "frequency": "Yearly (per tick)"
                    }
                }
            }
        """,
        "Poverty": """
            {
                "Model-Level": {
                    "income_mean": {
                        "value_boundaries": "0-200",
                        "equation": {
                            "linear": "Income_Mean = a * time",
                            "logarithmic": "Income_Mean = 10 * LN(Time / 0.13)",
                            "exponential": "Income_Mean = 30 * e^(0.0206 * Time) - 30"
                        },
                        "order_number": "Update at the beginning of each tick",
                        "frequency": "Yearly (per tick)"
                    }
                }
            }
        """,
        "PovertyPDF": """
            {
                "Model-Level": {
                    "income_mean": {
                        "value_boundaries": "0-200",
                        "equation": {
                            "linear": "Income_Mean = a * time",
                            "logarithmic": "Income_Mean = 10 * LN(Time / 0.13)",
                            "exponential": "Income_Mean = 30 * e^(0.0206 * Time) - 30"
                        },
                        "order_number": "Update at the beginning of each tick",
                        "frequency": "Yearly (per tick)"
                    }
                }
            }
        """,
        "Trout": """
        {
            "Model-Level": {
                "habDriftRegenDist": {
                    "value_boundaries": "600 cm",
                    "equation": "Not Available",
                    "order_number": "Not Available",
                    "frequency": "Not Available"
                }
            }
        }
        """,
        "TroutPDF": """
        {
            "Model-Level": {
                "habDriftRegenDist": {
                    "value_boundaries": "600 cm",
                    "equation": "Not Available",
                    "order_number": "Not Available",
                    "frequency": "Not Available"
                }
            }
        }
        """,
        "Tiger":"""
            {
                "Model-Level": {
                    "total_population": {
                        "value_boundaries": null,
                        "equation": "pop size * ((1 + avg growth rate) ^ n)",
                        "order_number": 12,
                        "frequency": "monthly"
                    }
                }
            }
        """,
        "Elites": """
        {
                "Model-Level": {
                    "percent-elites": {
                        "value_boundaries": "[0, 2]",
                        "equation": null,
                        "order_number": null,
                        "frequency": "each time step"
                    }
                }
            }
        """
    },
}