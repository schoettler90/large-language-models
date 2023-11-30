import os
from dotenv import load_dotenv

load_dotenv()

# TRAINING
HIDDEN_DIM = 256
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 8
DROPOUT = 0.2
WEIGHT_DECAY = 1e-2


# PATHS
ORIGINAL_DATA_PATH = 'data/sensors_example.xlsx'
CLEAN_DATA_PATH = 'data/sensors_cleaned.pkl'
EMBEDDINGS_DATA_PATH = 'data/sensors_embeddings.pkl'

MODEL_SAVE_PATH = "models/multi_head_classifier.pt"

# TRANSFORMERS from env
E5 = os.getenv("E5")
BERT = os.getenv("BERT")
DISTIL_BERT = os.getenv("DISTIL_BERT")
SGPT = os.getenv("SGPT")
LLAMA = os.getenv("LLAMA")

TRANSFORMER_PATH = LLAMA


# PREPROCESSING MAPPINGS

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
