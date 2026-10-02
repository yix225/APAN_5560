from fastapi import FastAPI
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel


app = FastAPI()


# -------------------------
# Bigram model
# -------------------------

corpus = [
    "the cat sat on the mat",
    "the cat ate the food",
    "the dog sat on the floor",
    "the dog ate the food"
]

bigram_model = BigramModel(corpus)


class GenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {
        "message": "Generative Text API is running"
    }


@app.post("/generate")
def generate_text(request: GenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


# -------------------------
# Embedding model
# -------------------------

embedding_model = EmbeddingModel()


class EmbeddingRequest(BaseModel):
    word: str


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    word = request.word.strip()

    if not word:
        return {
            "error": "Word cannot be empty"
        }

    embedding = embedding_model.get_embedding(word)

    return {
        "word": word,
        "dimension": len(embedding),
        "embedding": embedding
    }