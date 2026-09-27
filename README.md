# SPS GenAI: Text Generation and Word Embedding API

FastAPI service for Applied Generative AI:

- **Module 3 class activity:** text generation with a bigram language model
- **Assignment 1:** word embeddings and similarity with spaCy's `en_core_web_lg` (as in the Module 2 practical)

Built with help from Claude Code (the course allows LLM coding assistants).

## Project structure

```
sps_genai/
├── app/
│   ├── bigram_model.py     # bigram probabilities and text generation
│   ├── embedding_model.py  # spaCy embeddings and similarity
│   └── main.py             # API endpoints
├── tests/test_api.py
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs. The image is about 1.5 GB because it includes the `en_core_web_lg` model, so the first build takes a few minutes.

## Run locally with uv

```bash
uv sync
uv run fastapi dev app/main.py
uv run pytest
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Hello World |
| GET | `/health` | Health check |
| GET | `/vocabulary` | Every word the model knows |
| GET | `/next_words/{word}` | Probability of each word that can follow `word` |
| POST | `/generate` | Generate text from a start word |
| GET | `/embedding/{word}` | 300-dimensional embedding for `word` (`?limit=N` returns the first N values) |
| GET | `/similarity?word1=&word2=` | Cosine similarity between two words |

### Examples

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
# {"generated_text": "the count of monte cristo is a novel written by alexandre"}  (output is random)

curl http://127.0.0.1:8000/next_words/bigram
# {"word": "bigram", "next_word_probabilities": {"probabilities": 0.5, "models": 0.5}}

curl "http://127.0.0.1:8000/embedding/apple?limit=5"
# {"word": "apple", "model": "en_core_web_lg", "dimension": 300, "embedding": [-0.3639, 0.4377, -0.2045, -0.2289, -0.1423]}

curl "http://127.0.0.1:8000/similarity?word1=apple&word2=orange"
# {"word1": "apple", "word2": "orange", "similarity": 0.5619}
```

`length` must be between 1 and 100. Generation stops early at a word that nothing follows in the corpus. Words without an embedding return 404.
