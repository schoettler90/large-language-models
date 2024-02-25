import os

import torch
import transformers
from dotenv import load_dotenv

# if cuda is available, use it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACCESS_TOKEN = os.getenv("HUGGINGFACE_WRITE")

MODEL_PATH = r"D:\models\Llama-2-7b-chat-hf"
# MODEL_PATH = r"D:\models\Orca-2-7b"


def load_model():
    load_dotenv()

    print("Loading model from: ", MODEL_PATH)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH,
        device_map=DEVICE,
        trust_remote_code=True,
        token=ACCESS_TOKEN)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=DEVICE,
        token=ACCESS_TOKEN)

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
