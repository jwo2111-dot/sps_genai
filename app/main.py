from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel

app = FastAPI(
    title="SPS GenAI: Bigram Text Generator",
    description="Generate text from a simple bigram language model.",
)

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(..., min_length=1, examples=["the"])
    length: int = Field(..., ge=1, le=100, examples=[10])


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/vocabulary")
def get_vocabulary():
    """List every word the model knows."""
    vocabulary = bigram_model.vocabulary
    return {"size": len(vocabulary), "words": vocabulary}


@app.get("/next_words/{word}")
def get_next_words(word: str):
    """Show the probability of each word that can follow `word`."""
    probs = bigram_model.next_word_probs(word)
    if not probs:
        raise HTTPException(status_code=404, detail=f"No words follow '{word}' in the corpus")
    return {"word": word.lower(), "next_word_probabilities": probs}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}
