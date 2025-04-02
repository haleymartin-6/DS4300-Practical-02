# DS4300-Practical-02


## Overview
### The RAG system works in three main steps:

<b>Indexing</b>: Documents are processed, split into chunks, and stored as vector embeddings in Redis

<b>Retrieval</b>: User queries are converted to embeddings and similar document chunks are retrieved

<b>Generation</b>: Retrieved context is sent to an LLM along with the query to generate an informed response

## Requirements

Python 3.8+

Redis (with RediSearch module)

Ollama

PyMuPDF (fitz)

NumPy

## Installation

### ChromaDB Setup
ChromaDB is an open-source embedding database designed for storing and searching vector embeddings for LLM applications.
Setting up ChromaDB with Docker

Pull the ChromaDB Docker image:

```bash
docker pull chromadb/chroma
```
Create a persistent volume for data storage:
```bash
docker volume create chroma-data
```
Install the Python client:
```bash 
pip install chromadb
```

### Qdrant Setup
Qdrant is a vector similarity search engine with extended filtering support.
Setting up Qdrant with Docker

Pull the Qdrant Docker image:
```bash
docker pull qdrant/qdrant
```

Create a persistent volume for data storage:
```bash
docker volume create qdrant-data
```

Run Qdrant container:
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant-data:/qdrant/storage \
  qdrant/qdrant
```
Install the Python client:
```bash
pip install qdrant-client
```

### Redis Setup
Install Redis with vector search capabilities:

Docker installation example
```bash
docker run -d --name redis-vector -p 6380:6379 redis/redis-stack
```

Install Ollama:

```bash
Follow instructions at https://ollama.ai/
```

Install Python dependencies:

```bash
pip install redis numpy pymupdf ollama
```

Pull the required models in Ollama:

```bash
ollama pull nomic-embed-text
ollama pull mistral
```
## Running the model

Run the script search.py then enter your questions at the prompt to receive responses based on the indexed documents :)

