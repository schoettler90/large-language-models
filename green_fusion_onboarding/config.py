import os
from dotenv import load_dotenv

load_dotenv()

# TRAINING PARAMETERS in training.py
HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 8
DROPOUT = 0.2
WEIGHT_DECAY = 1e-2


# PATHS
# original Excel file with data
ORIGINAL_DATA_PATH = 'data/sensors_example.xlsx'

# resulting cleaned data of preprocessing.py
CLEAN_DATA_PATH = 'data/sensors_cleaned.pkl'

# resulting embeddings of feature_extraction.py
EMBEDDINGS_DATA_PATH = 'data/sensors_embeddings.pkl'

# results of training.py
RESULTS_PATH = 'data/results.csv'
MODEL_SAVE_PATH = "models/multi_head_classifier.pt"

# TRANSFORMERS from env. Main model is E5 until now

# https://huggingface.co/intfloat/multilingual-e5-large
E5 = os.getenv("E5")

# https://huggingface.co/dbmdz/bert-base-german-uncased
BERT = os.getenv("BERT")

# https://huggingface.co/distilbert-base-german-cased
DISTIL_BERT = os.getenv("DISTIL_BERT")

# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
LLAMA = os.getenv("LLAMA")

TRANSFORMER_PATH = E5


# PREPROCESSING MAPPINGS done in preprocessing.py
NODE_ID_MAPPINGS = {
    "_r": " ",
    "_x": " ",
    "_": " ",
}

NAME_CUST_DICT = {
    "_ival": "",
    ".ival": "",
    "ival": "",
    "_": " ",
}

GENERAL_MAPPINGS = {
    "wmz": "wärmemengenzähler",
    "wwb": "wasserwerkbereiter",
    "hk": "heizkreis",
    "temp ": "temperatur",
    "nan": "nicht bekannt",
}
