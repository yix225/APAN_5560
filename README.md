# Assignment 1 - Word Embedding API

This project implements a FastAPI endpoint that returns word embeddings using spaCy.

## Setup

Install dependencies:

uv sync

Install the spaCy English model:

uv run python -m spacy download en_core_web_md

## Run

uv run fastapi dev main.py

Open the API documentation at:

http://127.0.0.1:8000/docs

## Embedding Endpoint

POST /embedding

Example request:

{
    "word": "computer"
}

The API returns the word, embedding dimension, and embedding vector.