import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import utils

# set the device to cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def load_cleaned_data(data_path: str) -> pd.DataFrame:
    """
    Load the cleaned data
    :param data_path: path to the cleaned data
    :return: pandas dataframe
    """

    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.pkl'):
        data = pd.read_pickle(data_path)
    else:
        raise ValueError("The data_path must be either .csv or .pkl")
    return data


def get_embeddings(data: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Get the embeddings of the input texts
    :param data: cleaned data with the text column
    :param model_name: path or name to the model
    :return:
    """
    input_texts = data['text'].tolist()

    print("Number of cases: ", len(input_texts))

    # Load the model
    print("Loading the model: ", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=DEVICE)
    model = AutoModel.from_pretrained(model_name, device_map=DEVICE)

    # print model architecture
    print(model)
    # print number of parameters in millions
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=64, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Get the embeddings
    outputs = model(**batch_dict)
    embeddings = utils.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    # explain the above line
    # outputs.last_hidden_state.shape = (batch_size, seq_len, hidden_dim)
    # batch_dict['attention_mask'].shape = (batch_size, seq_len)
    # outputs.last_hidden_state.masked_fill(~batch_dict['attention_mask'][..., None].bool(), 0.0)
    # outputs.last_hidden_state.shape = (batch_size, seq_len, hidden_dim)
    # outputs.last_hidden_state.sum(dim=1).shape = (batch_size, hidden_dim)
    # outputs.last_hidden_state.sum(dim=1) / batch_dict['attention_mask'].sum(dim=1)[..., None].shape
    # = (batch_size, hidden_dim)

    print("Embeddings:")
    print(embeddings.shape)

    # Save the embeddings
    torch.save(embeddings, 'data/embeddings.pt')

    # save the embeddings in df
    data['embeddings'] = embeddings.tolist()

    return data


def main():
    import config
    clean_data_path = config.CLEAN_DATA_PATH
    model_name = config.TRANSFORMER_PATH

    clean_data = load_cleaned_data(clean_data_path)

    df = get_embeddings(clean_data, model_name)

    df.to_pickle(config.EMBEDDINGS_DATA_PATH)
    df.to_csv('data/sensors_embeddings.csv')

    print("Data saved to: ", config.EMBEDDINGS_DATA_PATH)


if __name__ == '__main__':
    main()
