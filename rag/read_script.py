import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import config

# Initialize Chroma DB client
client = chromadb.PersistentClient(path=config.database_path)
collection = client.get_collection(name=config.collection_name)

model_kwargs = {'device': config.device, 'trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(
    model_name=config.embedding_model,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Get user input
query = "Search information about indexing optimization"

# Convert query to vector representation
query_vector = embeddings.embed_query(query)

# Query Chroma DB with the vector representation
results = collection.query(query_embeddings=query_vector, n_results=4, include=["documents"])

# Print results
for result in results["documents"]:
    for i in result:
        print("__" * 40)
        print(i)
