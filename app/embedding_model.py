"""Word embeddings with spaCy, adapted from Module 2, Practical 3 (Word Embeddings)."""

import spacy


class EmbeddingModel:
    """Loads a spaCy model once and exposes the Module 2 embedding and similarity functions."""

    def __init__(self, model_name="en_core_web_lg"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def has_vector(self, text):
        """True if at least one token in `text` is in the model's vocabulary."""
        return any(token.has_vector for token in self.nlp(text))

    def calculate_embedding(self, input_word):
        """Return the embedding vector for a word (or the average over the words in a phrase)."""
        word = self.nlp(input_word)
        return word.vector.tolist()

    def calculate_similarity(self, word1, word2):
        """Cosine similarity between the embeddings of two words or phrases."""
        return float(self.nlp(word1).similarity(self.nlp(word2)))
