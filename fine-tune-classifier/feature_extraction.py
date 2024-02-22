import os

import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

import utils, config

load_dotenv()
access_token = os.getenv("HUGGINGFACE_WRITE")

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
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=DEVICE, access_token=access_token)
    model = AutoModel.from_pretrained(model_name, device_map=DEVICE, torch_dtype=torch.float16)

    # put the model in evaluation mode
    model.eval()

    # print model architecture
    print(model)
    # print number of parameters in millions
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=64, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    # Get the embeddings
    outputs = model(**batch_dict)
    embeddings = utils.average_pooling(outputs.last_hidden_state, batch_dict['attention_mask'])

    print("Embeddings:")
    print(embeddings.shape)

    # Save the embeddings
    torch.save(embeddings, 'data/embeddings.pt')

    # save the embeddings in df
    data['embeddings'] = embeddings.tolist()

    return data


def encode_sentence_llm(input_text: str, model: torch.nn.Module, tokenizer) -> torch.Tensor:
    """

    :param input_text: input text with batch size of 1
    :param model: large language model
    :param tokenizer: tokenizer of the large language model
    :return: tensor of shape (hidden_dim) with the embeddings
    """

    # Tokenize the input texts
    input_ids = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    # Get the embeddings
    output = model(**input_ids)

    embeddings = utils.weighted_average_pool(output.last_hidden_state, input_ids['attention_mask'])

    return embeddings.squeeze(0)


def get_embeddings_llm(data: pd.DataFrame, model_name: str) -> pd.DataFrame:
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        device_map=DEVICE,
        trust_remote_code=True,
        token=access_token,
    )

    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=DEVICE,
        token=access_token,
    )

    # put the model in evaluation mode
    model.eval()

    # print model architecture
    print(model)
    # print number of parameters in millions
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Get the embeddings
    embeddings = []
    for text in input_texts:
        embedding = encode_sentence_llm(text, model, tokenizer)
        print("Text: ", text)
        print("Embeddings Shape: ", embedding.shape)
        embeddings.append(embedding)

    # convert list of tensors to tensor
    embeddings = torch.stack(embeddings, dim=0)

    print("Concatenated Embeddings Shape: ", embeddings.shape)

    # save the embeddings in df
    data['embeddings'] = embeddings.tolist()

    return data


def main():
    clean_data_path = config.CLEAN_DATA_PATH
    model_name = config.TRANSFORMER_PATH

    clean_data = load_cleaned_data(clean_data_path)

    if "Llama" in model_name or "SGPT" in model_name:
        df = get_embeddings_llm(clean_data, model_name)
    else:
        df = get_embeddings(clean_data, model_name)

    df.to_pickle(config.EMBEDDINGS_DATA_PATH)
    df.to_csv('data/sensors_embeddings.csv')

    print("Data saved to: ", config.EMBEDDINGS_DATA_PATH)


if __name__ == '__main__':
    main()
