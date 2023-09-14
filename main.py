import torch
from transformers import GPT2Tokenizer, GPT2Model


def generate_gpt2_embedding(sentence):
    # Load pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')

    # Tokenize sentence and convert to tensor
    inputs = tokenizer.encode(sentence, return_tensors='pt')

    # Generate sentence embedding
    with torch.no_grad():
        outputs = model(inputs)[0][:, 0, :]
    embedding = torch.mean(outputs, dim=0)

    # Convert embedding tensor to numpy array and return
    return embedding.numpy()


def variables():
    from dotenv import load_dotenv
    import os

    load_dotenv()

    OPENAI_KEY = os.getenv("OPENAI_KEY")
    print(f"OPENAI_KEY: {OPENAI_KEY}")


def main():
    print("Generating GPT-2 embedding...")
    sentence = "Hello, my dog is cute"
    print("Sentence: ", sentence)
    embedding = generate_gpt2_embedding(sentence)
    print("Length of the embedding: ", len(embedding))
    print("First 10 values of embedding: ")
    print(embedding[:10])


if __name__ == "__main__":
    variables()
