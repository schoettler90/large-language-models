# TRAINING
HIDDEN_DIM = 512
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 32
DROPOUT = 0.2
WEIGHT_DECAY = 1e-3


# PATHS
ORIGINAL_DATA_PATH = 'data/sensors_example.xlsx'
CLEAN_DATA_PATH = 'data/sensors_cleaned.pkl'
EMBEDDINGS_DATA_PATH = 'data/sensors_embeddings.pkl'

MODEL_SAVE_PATH = "models/three_head_classifier.pt"

# EMBEDDINGS MODEL PATHS
ROBERTA = 'D:\models\multilingual-e5-large'
BERT = "D:\models\distilbert-base-german-cased"

TRANSFORMER_PATH = BERT


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
    "temp": "temperatur",
    "nan": "nicht bekannt",
}
