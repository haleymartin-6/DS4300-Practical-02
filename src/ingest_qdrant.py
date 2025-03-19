import ollama
import numpy as np
import os
import fitz
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize Qdrant connection
client = QdrantClient(url="http://localhost:6333")

VECTOR_DIM = 768
COLLECTION_NAME = "document_embeddings"


# Clear the Qdrant collection
def clear_qdrant_store():
    print("Clearing existing Qdrant collection...")
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("Qdrant collection deleted.")
    except Exception as e:
        print(f"Collection doesn't exist or couldn't be deleted: {e}")


# Create a collection in Qdrant
def create_qdrant_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIM,
            distance=models.Distance.COSINE
        )
    )
    print("Qdrant collection created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in Qdrant
def store_embedding(file: str, page: str, chunk: str, embedding: list, chunk_id: int):
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "file": file,
                    "page": page,
                    "chunk": chunk,
                }
            )
        ]
    )
    print(f"Stored embedding for: {chunk[:50]}...")


# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):
    chunk_counter = 0
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                        chunk_id=chunk_counter
                    )
                    chunk_counter += 1
            print(f" -----> Processed {file_name}")


def query_qdrant(query_text: str, limit=5):
    query_embedding = get_embedding(query_text)

    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=limit
    )

    print(f"Query: {query_text}\n")
    for result in search_result:
        print(f"ID: {result.id}, Score: {result.score}")
        print(f"File: {result.payload['file']}, Page: {result.payload['page']}")
        print(f"Chunk: {result.payload['chunk'][:100]}...\n")


def main():
    clear_qdrant_store()
    create_qdrant_collection()

    process_pdfs("../notes/")
    print("\n---Done processing PDFs---\n")
    query_qdrant("What is the capital of France?")


if __name__ == "__main__":
    main()