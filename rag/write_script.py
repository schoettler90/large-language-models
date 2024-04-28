import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def pdf_to_text(file_path: str) -> str:
    """
    Function to convert a PDF file to text
    :param file_path: path to the PDF file
    :return: text extracted from the PDF

    """
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()

    pdf_file.close()

    return text


def main():
    """
    Main function to process PDFs in the ./documents directory and store them in Chroma DB
    """
    import config


    model_kwargs = {'device': config.device}
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize text splitter and embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    # Initialize Chroma DB client
    client = chromadb.PersistentClient(path=config.db_path)
    collection = client.create_collection(name=config.database_name)

    # Process each PDF in the ./input directory
    for filename in os.listdir(config.documents_path):
        if filename.endswith('.pdf'):

            print(f"Processing {filename}")

            # Convert PDF to text
            text = pdf_to_text(os.path.join(config.documents_path, filename))

            # Split text into chunks
            chunks = text_splitter.split_text(text)

            # Convert chunks to vector representations and store in Chroma DB
            documents_list = []
            embeddings_list = []
            ids_list = []

            for i, chunk in enumerate(chunks):
                vector = embeddings.embed_query(chunk)

                documents_list.append(chunk)
                embeddings_list.append(vector)
                ids_list.append(f"{filename}_{i}")

            collection.add(
                embeddings=embeddings_list,
                documents=documents_list,
                ids=ids_list
            )



if __name__ == "__main__":
    main()
