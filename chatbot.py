import os
from dotenv import load_dotenv

import torch
import transformers


def main():
    # model_path = "tiiuae/falcon-7b-instruct"
    # model_path = 'meta-llama/Llama-2-7b-chat-hf'
    model_path = 'meta-llama/Llama-2-13b-chat-hf'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("Pipeline:")
    print(pipeline.model)

    # print on which device the model is running
    print("Device:", pipeline.device)

    input_text = "What is the recipe for spaghetti bolognese?"

    sequences = pipeline(
        input_text,
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def load_model() -> transformers.Pipeline:
    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_WRITE")

    # model_path = "tiiuae/falcon-7b-instruct"
    model_path = 'meta-llama/Llama-2-7b-chat-hf'
    # model_path = 'meta-llama/Llama-2-13b-chat-hf
    model_path = '../models/Llama-2-7b-chat-hf'
    # model_path = '../models/Llama-2-13b-chat-hf'

    print("Loading model...")
    print(f"Model name: {model_path}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        device_map="auto",
        token=access_token,
    )

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        token=access_token,
    )

    return pipeline


def chat_with_model(pipeline: transformers.Pipeline):
    print("Welcome! You can start a conversation by typing your message, or type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        # Check if the user wants to exit the conversation
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        sequences = pipeline(
            user_input,
            max_length=1000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            pad_token_id=pipeline.tokenizer.eos_token_id,
        )

        for seq in sequences:
            print(f"Model: {seq['generated_text']}")


if __name__ == "__main__":
    chatbot = load_model()
    chat_with_model(chatbot)
