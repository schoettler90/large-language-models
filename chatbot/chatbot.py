import os

import torch
import transformers
from dotenv import load_dotenv
import config

load_dotenv()

# if cuda is available, use it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# add your hugingface token to the environment variables
ACCESS_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Load the model from the config file
MODEL_PATH = config.LLAMA


def load_model():
    load_dotenv()

    print("Loading model from: ", MODEL_PATH)

    # Load the model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        token=ACCESS_TOKEN,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        token=ACCESS_TOKEN,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    print("Model:")
    print(model)

    return model, tokenizer


def chat_with_model(model, tokenizer):
    print("Welcome! You can start a conversation by typing your message, or type 'exit' to end the conversation.")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # initialize conversation
    conversation = "You are a helpful assistant, that answers questions precisely and informatively.\n"

    while True:
        user_input = input("You: ")

        # Check if the user wants to exit the conversation
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # add user input to conversation
        conversation += user_input + "\n"

        input_ids = tokenizer.encode(conversation, return_tensors="pt").to(DEVICE)

        outputs = model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
        )

        # get the last response
        response = outputs[0][input_ids.shape[-1]:]

        # decode and print only the last response
        response = tokenizer.decode(response, skip_special_tokens=True)

        print("Model:", response)

        # add model response to conversation
        conversation += response + "\n"


def run_chatbot():
    model, tokenizer = load_model()
    chat_with_model(model, tokenizer)


if __name__ == "__main__":
    run_chatbot()
