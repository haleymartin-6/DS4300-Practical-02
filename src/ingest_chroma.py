import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Ensure the collection exists
try:
    collection = chroma_client.get_collection(name="embedding_collection")
except chromadb.errors.InvalidCollectionException:
    collection = chroma_client.create_collection(name="embedding_collection")

VECTOR_DIM = 768


def clear_chroma_store():
    print("Clearing existing ChromaDB store...")
    chroma_client.delete_collection(name="embedding_collection")
    global collection
    collection = chroma_client.create_collection(name="embedding_collection")
    print("ChromaDB store cleared.")


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    print(f"Stored embedding for: {chunk}")


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = [(page_num, page.get_text()) for page_num, page in enumerate(doc)]
    return text_by_page


def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks


def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_chroma(query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    if "ids" in results and results["ids"]:
        for doc, score in zip(results["ids"][0], results["distances"][0]):
            print(f"Document ID: {doc} \n ----> Score: {score}\n")
    else:
        print("No results found.")


def main():
    clear_chroma_store()
    process_pdfs("../notes/")
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()