# Load model directly
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
load_dotenv()
access_token = os.getenv("HUGGINGFACE_WRITE")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf", token=access_token, device_map="cpu")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", token=access_token, device_map="cpu")

print(model)


# generate text
input_text = "Who will win the war in Ukraine?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
print("Token IDs: ")
print(input_ids)

output = model.generate(input_ids, max_length=200, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
