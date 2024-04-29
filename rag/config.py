"""
This file contains the configuration for the RAG model.
"""

device = "cuda"
documents_path = r"./documents"
database_path = r"./db"
collection_name = "my_papers"

embedding_model = "Alibaba-NLP/gte-large-en-v1.5"
