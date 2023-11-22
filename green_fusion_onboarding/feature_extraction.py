import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import utils

# set the device to cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def get_embeddings(data_path, model_name):
    # Read in the data
    if data_path.endswith('.pkl'):
        df = pd.read_pickle(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("Data path must be a .pkl or .csv file.")

    input_texts = df['text'].tolist()

    print("Number of input texts: ", len(input_texts))

    # Load the model

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=DEVICE)
    model = AutoModel.from_pretrained(model_name, device_map=DEVICE)

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Get the embeddings
    outputs = model(**batch_dict)
    embeddings = utils.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    print("Embeddings:")
    print(embeddings.shape)

    # Save the embeddings
    torch.save(embeddings, 'data/embeddings.pt')

    # save the embeddings in df
    df['embeddings'] = embeddings.tolist()
    df.to_pickle('data/sensors_embeddings.pkl')
    df.to_csv('data/sensors_embeddings.csv')


def main():
    data_path = 'data/sensors_cleaned.pkl'
    model_name = 'D:\models\multilingual-e5-large'
    # model_name = "D:\models\distilbert-base-german-cased"
    get_embeddings(data_path, model_name)


if __name__ == '__main__':
    main()
