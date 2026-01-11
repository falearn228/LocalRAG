# Ollama Hello World - RAG System with LocalAI and Qdrant

A Retrieval-Augmented Generation (RAG) system built with Go that uses LocalAI for running local LLM models and Qdrant as a vector database for semantic search.

## Features

- **Local LLM Inference**: Run LLaMA and other models locally using LocalAI
- **Vector Database**: Store and search document embeddings with Qdrant
- **RAG Pipeline**: Retrieve relevant context and generate responses
- **Streaming Responses**: Real-time streaming of LLM outputs
- **Docker Compose**: Easy setup with GPU support for NVIDIA devices

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   User      │────▶│   Go App    │────▶│  LocalAI    │
│  (CLI)      │     │             │◀────│  (LLM/Emb)  │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   Qdrant    │
                    │ (Vector DB) │
                    └─────────────┘
```

## Prerequisites

- Go 1.25.5 or higher
- Docker and Docker Compose
- NVIDIA GPU (optional, for GPU acceleration)
- NVIDIA Container Toolkit (for GPU support)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ollama-hello-world
```

### 2. Download Go dependencies

```bash
go mod download
```

### 3. Prepare model files

Place your GGUF model files in the `models/` directory:

- `models/llama-3.2-1b-instruct-q4_k_m.gguf` - LLM model
- `models/all-MiniLM-L6-v2-Q3_K_L.gguf` - Embedding model

Configure model settings in:
- `models/local-llm.yaml` - LLM configuration
- `models/local-embed.yaml` - Embedding model configuration

### 4. Start services with Docker Compose

```bash
docker-compose up -d
```

This will start:
- **Qdrant** on port `6333` - Vector database
- **LocalAI** on port `8080` - LLM and embedding service

### 5. Run the application

```bash
go run main.go
```

## Usage

Once the application is running, you can interact with it through the CLI:

```
=== RAG система с LocalAI и Qdrant ===
Введите ваш вопрос (или 'exit' для выхода):

> Что такое Ollama?
Ответ: Ollama - это инструмент для запуска локальных LLM моделей.

> Как работает RAG?
Ответ: RAG (Retrieval-Augmented Generation) - это подход, объединяющий поиск информации и генерацию текста.

> exit
Завершение работы...
```

## Project Structure

```
.
├── main.go                 # Main application entry point
├── go.mod                  # Go module definition
├── go.sum                  # Go dependencies checksums
├── docker-compose.yml      # Docker services configuration
├── models/                 # Model files and configurations
│   ├── local-llm.yaml      # LLM model configuration
│   ├── local-embed.yaml    # Embedding model configuration
│   └── *.gguf              # Model weights
└── qdrant_data/            # Qdrant persistent storage (gitignored)
```

## Key Components

### QdrantStore
Handles vector storage and retrieval operations with Qdrant:
- `AddDocuments()` - Store documents with embeddings
- `SimilaritySearch()` - Find similar documents by query

### LocalAILLM
Provides LLM inference through LocalAI:
- `PredictStream()` - Stream responses from the model

### LocalAIEmbedder
Generates embeddings for documents and queries:
- `EmbedQuery()` - Create embedding for a single text
- `EmbedDocuments()` - Create embeddings for multiple texts

## Dependencies

- [`github.com/tmc/langchaingo`](https://github.com/tmc/langchaingo) - LangChain for Go
- [`github.com/google/uuid`](https://github.com/google/uuid) - UUID generation
- [`github.com/go-skynet/go-llama.cpp`](https://github.com/go-skynet/go-llama.cpp) - LLaMA.cpp bindings

## Configuration

### LocalAI Models

Edit the model configuration files in `models/`:

**local-llm.yaml**:
```yaml
name: local-llm
backend: llama
context_size: 2048
f16: true
...
```

**local-embed.yaml**:
```yaml
name: local-embed
backend: llama
embedding: true
...
```

### Qdrant

The Qdrant vector database is configured in `docker-compose.yml`:
- Port: `6333`
- Storage: `./qdrant_data`
- Distance metric: Cosine

## Troubleshooting

### LocalAI fails to start
- Ensure model files are correctly placed in `models/`
- Check Docker logs: `docker-compose logs localai`
- Verify GPU access: `nvidia-smi`

### Qdrant connection errors
- Check if Qdrant is running: `docker-compose ps`
- Verify port 6333 is available
- Check Qdrant logs: `docker-compose logs qdrant`

### Embedding dimension mismatch
- Ensure the embedding model is correctly configured
- Delete Qdrant data and restart: `rm -rf qdrant_data && docker-compose restart qdrant`

## License

MIT License