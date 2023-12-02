import os
from dotenv import load_dotenv

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load the LLAMA 2 7B model
load_dotenv()
model_name = "D:\models\Llama-2-7b-chat-hf"
access_token = os.getenv("HUGGINGFACE_WRITE")

model = LlamaForCausalLM.from_pretrained(model_name, token=access_token, device_map="cuda:0")
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token, device_map="cuda:0")

print("Model:")
print(model)

# generate text
input_text = "Who will win the war in Ukraine?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("Token IDs: ")
print(input_ids)

output = model.generate(input_ids.cuda(), max_length=200, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
