"""
Simple baseline model for next-byte prediction.

This is a basic unigram model that predicts based on general byte frequencies.
It doesn't require training and serves as a starting point (~5-6 bits/byte).

To improve:
1. Train on the actual dataset using train_ngram.py
2. Use a neural network approach
3. Use a pretrained language model
"""

import math
from pathlib import Path


class Model:
    def __init__(self, submission_dir: Path):
        """
        Initialize the model.

        This baseline uses hardcoded byte frequencies typical of Icelandic text.
        For better results, train on the actual dataset!
        """
        # Try to load trained counts if available
        counts_file = submission_dir / "counts.json.gz"
        if counts_file.exists():
            self._load_trained_model(submission_dir)
        else:
            self._init_baseline()

    def _init_baseline(self):
        """Initialize with a simple baseline (no training needed)."""
        # Basic ASCII/UTF-8 byte frequencies for text
        # Common bytes: space, lowercase letters, newlines
        self.logits_cache = {}

        # Default logits: slightly favor common text bytes
        self.default_logits = [0.0] * 256

        # Boost common ASCII characters
        # Space (32) is very common
        self.default_logits[32] = 3.0  # space
        self.default_logits[10] = 1.5  # newline

        # Lowercase letters (97-122) are common
        for i in range(97, 123):
            self.default_logits[i] = 2.0

        # Uppercase letters (65-90) less common
        for i in range(65, 91):
            self.default_logits[i] = 1.0

        # Digits (48-57)
        for i in range(48, 58):
            self.default_logits[i] = 0.5

        # Common punctuation
        self.default_logits[46] = 1.0  # period
        self.default_logits[44] = 1.0  # comma

        # Icelandic-specific: UTF-8 continuation bytes are common
        # UTF-8 continuation bytes: 128-191
        for i in range(128, 192):
            self.default_logits[i] = 1.0

        # UTF-8 2-byte start: 192-223
        for i in range(192, 224):
            self.default_logits[i] = 0.5

        self.trained = False
        print("Using baseline model (no training data)")

    def _load_trained_model(self, submission_dir: Path):
        """Load trained n-gram counts."""
        import gzip
        import json

        counts_file = submission_dir / "counts.json.gz"
        with gzip.open(counts_file, 'rt') as f:
            raw_counts = json.load(f)

        # Convert to lookup format
        self.counts = {}
        for context_str, byte_counts in raw_counts.items():
            context = tuple(json.loads(context_str))
            self.counts[context] = {b: c for b, c in byte_counts}

        # Build unigram fallback
        self.unigram = [1] * 256
        for byte_counts in self.counts.values():
            for byte, count in byte_counts.items():
                self.unigram[byte] += count

        total = sum(self.unigram)
        self.unigram_logits = [math.log(c + 1) for c in self.unigram]

        self.trained = True
        print(f"Loaded trained model with {len(self.counts)} contexts")

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        """
        Predict next byte for each context.

        Args:
            contexts: List of byte sequences (each is list of ints 0-255)

        Returns:
            List of logit vectors, shape [batch_size, 256]
        """
        if self.trained:
            return [self._predict_trained(ctx) for ctx in contexts]
        else:
            return [self._predict_baseline(ctx) for ctx in contexts]

    def _predict_baseline(self, context: list[int]) -> list[float]:
        """Simple baseline prediction."""
        # Use context to slightly adjust predictions
        logits = self.default_logits.copy()

        if len(context) > 0:
            last_byte = context[-1]

            # After space, boost lowercase letters
            if last_byte == 32:
                for i in range(97, 123):
                    logits[i] += 1.0

            # After newline, boost uppercase/space
            elif last_byte == 10:
                for i in range(65, 91):
                    logits[i] += 0.5
                logits[32] += 0.5

            # After period, boost space
            elif last_byte == 46:
                logits[32] += 2.0
                logits[10] += 1.0

        return logits

    def _predict_trained(self, context: list[int]) -> list[float]:
        """Prediction using trained n-gram counts."""
        # Try progressively shorter contexts
        for length in range(min(len(context), 6), -1, -1):
            if length == 0:
                break
            ctx_tuple = tuple(context[-length:])
            if ctx_tuple in self.counts:
                return self._counts_to_logits(self.counts[ctx_tuple])

        return self.unigram_logits

    def _counts_to_logits(self, byte_counts: dict) -> list[float]:
        """Convert counts to logits."""
        logits = [0.0] * 256
        for byte, count in byte_counts.items():
            logits[byte] = math.log(count + 1)
        return logits
