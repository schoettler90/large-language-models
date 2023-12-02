import os
from dotenv import load_dotenv

import torch
import transformers

# if cuda is available, use it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_generate():
    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_WRITE")
    # model_path = "tiiuae/falcon-7b-instruct"
    # model_path = 'meta-llama/Llama-2-7b-chat-hf'
    # model_path = 'meta-llama/Llama-2-13b-chat-hf'
    model_path = r"../models/Llama-2-7b-chat-hf"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, device_map=DEVICE, token=access_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=DEVICE,
        token=access_token)

    print("Model:")
    print(model)

    # generate text
    input_text = "Give me the recipe to cook spaghetti carbonara."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(
        input_ids.cuda(),
        max_length=1000,
        do_sample=False,
        top_k=1,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


def load_model():
    load_dotenv()
    access_token = os.getenv("HUGGINGFACE_WRITE")
    # model_path = "tiiuae/falcon-7b-instruct"
    # model_path = 'meta-llama/Llama-2-7b-chat-hf'
    # model_path = 'meta-llama/Llama-2-13b-chat-hf'
    # model_path = r"D:\models\tiiuae\falcon-7b"
    model_path = r"D:\models\Llama-2-7b-chat-hf"
    # model_path = r"D:\models\Llama-2-13b-chat-hf"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        device_map=DEVICE,
        trust_remote_code=True,
        token=access_token)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=DEVICE,
        token=access_token)

    print("Model:")
    print(model)

    return model, tokenizer


def chat_with_model(model, tokenizer):
    print("Welcome! You can start a conversation by typing your message, or type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        # Check if the user wants to exit the conversation
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Encode the user's input and generate a response
        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(DEVICE)
        output = model.generate(
            input_ids,
            max_length=1000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode and print the model's response
        print("Model:", tokenizer.decode(output[0], skip_special_tokens=True))


def run_chatbot():
    model, tokenizer = load_model()
    chat_with_model(model, tokenizer)


if __name__ == "__main__":
    run_chatbot()
