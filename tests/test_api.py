from fastapi.testclient import TestClient

from app.bigram_model import BigramModel
from app.main import app

client = TestClient(app)


def test_bigram_probabilities_use_mle_formula():
    # p(w2 | w1) = C(w1, w2) / C(w1)
    model = BigramModel(["a b a c", "a b"])
    assert model.next_word_probs("a") == {"b": 2 / 3, "c": 1 / 3}
    # "b" appears twice but is followed by "a" only once
    assert model.next_word_probs("b") == {"a": 0.5}


def test_lecture_example():
    model = BigramModel("Darkness cannot drive out darkness, only light can do that")
    assert model.next_word_probs("darkness") == {"cannot": 0.5, "only": 0.5}


def test_generate_stops_at_dead_end():
    model = BigramModel(["hello world"])
    assert model.generate_text("hello", 10) == "hello world"
    assert model.generate_text("unknown", 5) == "unknown"


def test_root():
    assert client.get("/").json() == {"Hello": "World"}


def test_generate_endpoint():
    response = client.post("/generate", json={"start_word": "bigram", "length": 5})
    assert response.status_code == 200
    words = response.json()["generated_text"].split()
    assert words[0] == "bigram"
    assert 1 <= len(words) <= 5


def test_generate_rejects_bad_length():
    response = client.post("/generate", json={"start_word": "the", "length": 0})
    assert response.status_code == 422


def test_next_words_endpoint():
    response = client.get("/next_words/bigram")
    assert response.status_code == 200
    assert response.json()["next_word_probabilities"] == {"probabilities": 0.5, "models": 0.5}
    assert client.get("/next_words/zebra").status_code == 404


def test_vocabulary_endpoint():
    body = client.get("/vocabulary").json()
    assert "monte" in body["words"]
    assert body["size"] == len(body["words"])
