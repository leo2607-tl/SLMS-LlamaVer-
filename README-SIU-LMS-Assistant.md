# SIU LMS Assistant

SIU LMS Assistant is an AI-powered assistant system for Learning Management System workflows, built on a Retrieval-Augmented Generation architecture that combines Elasticsearch and Qdrant for context-aware question answering over ingested documents.

## Overview

The project is designed to answer user questions based on internal learning materials through a retrieval-first pipeline. It combines document ingestion, embedding generation, hybrid retrieval, language model inference, and agent-based orchestration into a single backend service.

## Features

- FastAPI backend for AI-powered query handling.
- Retrieval-Augmented Generation pipeline using Elasticsearch and Qdrant.
- Embedding-based retrieval with HuggingFace models and LlamaIndex integration.
- Agent-based workflow for coordinating contextual query tools.
- Support for ingesting `.txt` documents from a dataset directory.
- Docker Compose setup for Elasticsearch and Qdrant services.
- API endpoint for submitting natural language queries.

## Tech Stack

- Python
- FastAPI
- llama_index
- vLLM
- Elasticsearch
- Qdrant
- HuggingFace
- Docker Compose

## Architecture

```text
services/
└── api.py                      # FastAPI entry point and query endpoint
agent/
└── assistant.py                # Assistant logic and agent workflow
config/
└── config.py                   # System configuration and model settings
database/
├── elasticsearch_database.py   # Elasticsearch integration
├── qdrant_database.py          # Qdrant integration
└── ingest.py                   # Data ingestion workflow
utils/
├── ingestor.py                 # Reads dataset files and triggers ingestion
├── elasticsearch_utils.py      # Elasticsearch utilities
└── qdrant_utils.py             # Qdrant utilities
docker/
└── docker-compose.yml          # Elasticsearch and Qdrant services
```

## Requirements

- Python 3.10+ or 3.11+
- Docker and Docker Compose
- GPU recommended when using larger `vLLM` models
- Hugging Face token for restricted or externally hosted models when required

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn python-dotenv llama-index transformers vllm qdrant-client elasticsearch numpy
```

## Environment Variables

Create a `.env` file in the project root:

```bash
HUGGING_FACE_TOKEN=<your_huggingface_token>
```

If model paths, cache locations, or system endpoints need to be changed, update the project configuration accordingly.

## Running the Services

### 1. Start Elasticsearch and Qdrant

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 2. Start the FastAPI service

```bash
python services/api.py
```

### 3. Default API endpoint

```text
http://0.0.0.0:8000
```

### 4. Query endpoint

```text
POST http://localhost:8000/query
```

Example request body:

```json
{ "query": "Your question here" }
```

## Data Ingestion

Documents in `.txt` format are expected inside the `dataset` directory.

Run the ingestion script with:

```bash
python utils/ingestor.py
```

The ingestion pipeline reads the files, generates embeddings, and stores the results in both Elasticsearch and Qdrant.

## Notes

- The project configuration defines the default Elasticsearch and Qdrant endpoints.
- A shared index or collection is used for storing ingested document representations.
- Different language model settings may be used for testing and API runtime environments.
- If GPU resources are limited, model size and inference settings should be adjusted accordingly.

## Future Improvements

- Add a frontend interface for end users.
- Introduce authentication and access control.
- Extend the ingestion pipeline to support more document formats.
- Add monitoring dashboards for retrieval infrastructure.
- Improve observability for agent and query execution workflows.

## Goal

SIU LMS Assistant is designed to provide a scalable AI assistant for LMS environments by combining retrieval, language generation, and orchestration into a document-grounded question answering platform.
