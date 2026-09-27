from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.embedding_model import EmbeddingModel

app = FastAPI(
    title="SPS GenAI",
    description="Bigram text generation and spaCy word embeddings.",
)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)
embedding_model = EmbeddingModel("en_core_web_lg")


class TextGenerationRequest(BaseModel):
    start_word: str = Field(..., min_length=1, examples=["the"])
    length: int = Field(..., ge=1, le=100, examples=[10])


def require_vector(word: str):
    if not embedding_model.has_vector(word):
        raise HTTPException(status_code=404, detail=f"No embedding for '{word}'")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vocabulary")
def get_vocabulary():
    """Words in the bigram model's corpus."""
    vocabulary = bigram_model.vocabulary
    return {"size": len(vocabulary), "words": vocabulary}


@app.get("/next_words/{word}")
def get_next_words(word: str):
    """Probability of each word that can follow `word`."""
    probs = bigram_model.next_word_probs(word)
    if not probs:
        raise HTTPException(status_code=404, detail=f"No words follow '{word}' in the corpus")
    return {"word": word.lower(), "next_word_probabilities": probs}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """Generate up to `length` words starting from `start_word`."""
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.get("/embedding/{word}")
def get_embedding(word: str, limit: int | None = Query(None, ge=1, le=300)):
    """spaCy embedding for `word`. Use `limit` to return only the first N values."""
    require_vector(word)
    embedding = embedding_model.calculate_embedding(word)
    return {
        "word": word,
        "model": embedding_model.model_name,
        "dimension": len(embedding),
        "embedding": embedding[:limit],
    }


@app.get("/similarity")
def get_similarity(word1: str, word2: str):
    """Cosine similarity between the embeddings of two words."""
    require_vector(word1)
    require_vector(word2)
    similarity = embedding_model.calculate_similarity(word1, word2)
    return {"word1": word1, "word2": word2, "similarity": similarity}
