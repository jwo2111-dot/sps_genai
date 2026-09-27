# SPS GenAI: Text Generation and Word Embedding API

A FastAPI service that

- generates text with a bigram language model (Module 3 class activity), and
- returns spaCy word embeddings and word similarity (Assignment 1, using `en_core_web_lg` as in the Module 2 practical).

## Project structure

```
sps_genai/
├── app/
│   ├── bigram_model.py     # BigramModel: tokenizes the corpus, builds P(next | current), generates text
│   ├── embedding_model.py  # EmbeddingModel: spaCy en_core_web_lg embeddings and similarity
│   └── main.py             # FastAPI app and endpoints
├── tests/test_api.py       # unit and API tests (pytest)
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs for the interactive API docs. The image is about 1.5 GB because it includes the spaCy `en_core_web_lg` model; the first build takes a few minutes.

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
uv run pytest        # run the tests
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Hello World |
| GET | `/health` | Health check |
| GET | `/vocabulary` | Every word the model knows |
| GET | `/next_words/{word}` | Probability of each word that can follow `word` |
| POST | `/generate` | Generate text from a start word |
| GET | `/embedding/{word}` | 300-dimensional spaCy embedding for `word` (optional `?dimensions=N` returns only the first N values) |
| GET | `/similarity?word1=&word2=` | Cosine similarity between two words' embeddings |

### Examples

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
# {"generated_text": "the count of monte cristo is a novel written by alexandre"}

curl http://127.0.0.1:8000/next_words/bigram
# {"word": "bigram", "next_word_probabilities": {"probabilities": 0.5, "models": 0.5}}


curl "http://127.0.0.1:8000/embedding/apple?dimensions=5"
# {"word": "apple", "model": "en_core_web_lg", "dimension": 300, "embedding": [-0.3639, 0.4377, -0.2045, -0.2289, -0.1423]}

curl "http://127.0.0.1:8000/similarity?word1=apple&word2=orange"
# {"word1": "apple", "word2": "orange", "similarity": 0.5619}
```

Words that are not in the spaCy vocabulary return a 404.

`length` must be between 1 and 100. Generation stops early if it reaches a word that nothing follows in the corpus.
