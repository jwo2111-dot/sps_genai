"""Bigram language model adapted from Module 2, Practical 2 (Word Sampling)."""

import random
import re
from collections import Counter, defaultdict
from itertools import pairwise


def simple_tokenizer(text, frequency_threshold=None):
    """Split text into lowercase words, optionally dropping words rarer than frequency_threshold."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    return [token for token in tokens if word_counts[token] >= frequency_threshold]


def analyze_bigrams(documents, frequency_threshold=None):
    """Compute bigram probabilities p(w2 | w1) = C(w1, w2) / C(w1).

    Documents are tokenized separately so no bigram spans two documents.
    """
    bigram_counts = Counter()
    unigram_counts = Counter()
    for document in documents:
        words = simple_tokenizer(document, frequency_threshold)
        bigram_counts.update(pairwise(words))
        unigram_counts.update(words)

    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    return list(unigram_counts.keys()), dict(bigram_probs)


class BigramModel:
    """Bigram model from Module 2, built once from a corpus and reused by the API."""

    def __init__(self, corpus, frequency_threshold=None, seed=None):
        if isinstance(corpus, str):
            corpus = [corpus]
        vocab, self.bigram_probs = analyze_bigrams(corpus, frequency_threshold)
        self.vocabulary = sorted(vocab)
        self.rng = random.Random(seed)

    def next_word_probs(self, word):
        """p(next | word) for each word seen after `word`, most likely first."""
        next_words = self.bigram_probs.get(word.lower(), {})
        return dict(sorted(next_words.items(), key=lambda item: item[1], reverse=True))

    def generate_text(self, start_word, num_words=20):
        """Generate text based on bigram probabilities."""
        current_word = start_word.lower()
        generated_words = [current_word]

        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # If no bigrams for the current word, stop generating
                break

            # Choose the next word based on probabilities
            next_word = self.rng.choices(
                list(next_words.keys()), weights=list(next_words.values())
            )[0]
            generated_words.append(next_word)
            current_word = next_word  # Move to the next word

        return " ".join(generated_words)
