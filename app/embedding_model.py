"""Word embeddings with spaCy, adapted from Module 2, Practical 3 (Word Embeddings)."""

import spacy


class EmbeddingModel:
    """spaCy embedding and similarity functions from Module 2."""

    def __init__(self, model_name="en_core_web_lg"):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def has_vector(self, text):
        """True if any token in `text` has a vector in the model."""
        return any(token.has_vector for token in self.nlp(text))

    def calculate_embedding(self, input_word):
        """Embedding vector for a word (averaged over tokens for a phrase)."""
        word = self.nlp(input_word)
        return word.vector.tolist()

    def calculate_similarity(self, word1, word2):
        """Cosine similarity between the embeddings of two words."""
        return float(self.nlp(word1).similarity(self.nlp(word2)))
