import ollama
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "document_embeddings"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generate an embedding using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    """Search for the most similar vectors in Qdrant."""
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query)

        # Search for similar vectors in Qdrant
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k
        )

        # Transform results into the expected format
        top_results = []
        for result in search_results:
            top_results.append({
                "file": result.payload.get("file", "Unknown file"),
                "page": result.payload.get("page", "Unknown page"),
                "chunk": result.payload.get("chunk", "Unknown chunk"),
                "similarity": result.score
            })

            # Print results for debugging
            print(
                f"---> File: {result.payload.get('file')}, Page: {result.payload.get('page')}, Similarity: {result.score}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    """Generate a response using retrieved context."""
    # Extract text from context results
    context_chunks = [result.get("chunk", "") for result in context_results]

    # Prepare context string with source information
    context_with_sources = []
    for i, result in enumerate(context_results):
        source_info = f"Source {i + 1}: {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')})"
        context_with_sources.append(f"{source_info}\n{result.get('chunk', '')}")

    context_str = "\n\n".join(context_with_sources)

    print(f"\nContext being sent to LLM:")
    print("-" * 50)
    print(context_str[:500] + "..." if len(context_str) > 500 else context_str)
    print("-" * 50)

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 Qdrant RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        print(f"\nSearching for: '{query}'")
        context_results = search_embeddings(query)

        if not context_results:
            print("No relevant results found.")
            continue

        # Generate RAG response
        print("\nGenerating response...")
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    interactive_search()