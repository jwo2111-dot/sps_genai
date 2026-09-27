# SPS GenAI: Bigram Text Generator API

A FastAPI service that generates text with a bigram language model (Module 3 activity).

## Project structure

```
sps_genai/
├── app/
│   ├── bigram_model.py   # BigramModel: tokenizes the corpus, builds P(next | current), generates text
│   └── main.py           # FastAPI app and endpoints
├── tests/test_api.py     # unit and API tests (pytest)
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Run with Docker

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://127.0.0.1:8000/docs for the interactive API docs.

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

### Examples

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
# {"generated_text": "the count of monte cristo is a novel written by alexandre"}

curl http://127.0.0.1:8000/next_words/bigram
# {"word": "bigram", "next_word_probabilities": {"probabilities": 0.5, "models": 0.5}}
```

`length` must be between 1 and 100. Generation stops early if it reaches a word that nothing follows in the corpus.
