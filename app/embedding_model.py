"""Word embeddings with spaCy, adapted from Module 2, Practical 3 (Word Embeddings)."""

import spacy


class EmbeddingModel:
    def __init__(self, model_name="en_core_web_lg"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def has_vector(self, text):
        return any(token.has_vector for token in self.nlp(text))

    def calculate_embedding(self, input_word):
        """Averaged over tokens if given a phrase."""
        word = self.nlp(input_word)
        return word.vector.tolist()

    def calculate_similarity(self, word1, word2):
        return float(self.nlp(word1).similarity(self.nlp(word2)))
