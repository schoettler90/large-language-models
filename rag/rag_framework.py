import os
from langchain import hub
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.5, api_key=TOKEN)

email = (f"Dear Christian. I just siged the contract. I am looking forward to working with you. "
         f"Best regards, "
         f"Ricardo")

messages = [
    ("system", "You are a helpful assistant that translates English to German."),
    ("human", f"Translate this sentence from English to German: {email}"),
]

response = llm.invoke(messages)
print(response.content)
