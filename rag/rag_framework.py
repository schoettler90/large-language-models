import os
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=1., api_key=TOKEN)

promt = "Explain the difference between a encoder and a decoder transformer model. "

messages = [
    ("system", "You are a helpful assistant."),
    ("human", promt),
]

response = llm.invoke(messages)
print(response.content)
