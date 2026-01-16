#!/usr/bin/env python3
"""
Train an n-gram model for next-byte prediction.

This creates a simple but effective baseline model that:
1. Counts byte sequences in training data
2. Saves the counts as a compressed JSON file
3. At inference, predicts based on observed frequencies

Usage:
    # From HuggingFace dataset (arrow format):
    python train_ngram.py --data /path/to/igc_full --n 3

    # From text files:
    python train_ngram.py --data /path/to/texts --n 3 --text-mode

The output goes to submission/counts.json.gz
"""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path


def load_from_hf_dataset(data_path: Path, max_docs: int = None) -> list[bytes]:
    """Load training data from HuggingFace datasets format (arrow files)."""
    try:
        from datasets import load_from_disk
    except ImportError:
        print("ERROR: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        raise

    print(f"Loading HuggingFace dataset from {data_path}...")
    dataset = load_from_disk(str(data_path))

    num_docs = min(len(dataset), max_docs) if max_docs else len(dataset)
    print(f"Loading {num_docs:,} of {len(dataset):,} documents...")

    texts = []
    for i in range(num_docs):
        item = dataset[i]
        if "text" in item:
            texts.append(item["text"].encode("utf-8"))

    print(f"Loaded {len(texts)} documents")
    return texts


def load_from_text_files(data_path: Path) -> list[bytes]:
    """Load training data from text files."""
    texts = []

    if data_path.is_file():
        texts.append(data_path.read_bytes())
    elif data_path.is_dir():
        for f in data_path.glob("**/*.txt"):
            texts.append(f.read_bytes())
    else:
        raise ValueError(f"Data path not found: {data_path}")

    print(f"Loaded {len(texts)} files")
    return texts


def train_ngram(texts: list[bytes], n: int, min_count: int = 2) -> dict:
    """
    Train n-gram counts from training data.

    Args:
        texts: List of byte sequences
        n: N-gram order (e.g., 3 for trigrams)
        min_count: Minimum count to keep (prunes rare n-grams)

    Returns:
        Dictionary mapping context strings to [[next_byte, count], ...]
    """
    # Count n-grams: context (n-1 bytes) -> next byte -> count
    counts = defaultdict(lambda: defaultdict(int))

    total_bytes = sum(len(t) for t in texts)
    print(f"Training on {total_bytes:,} bytes...")

    total_ngrams = 0
    for text in texts:
        for i in range(len(text)):
            # Get context (previous n-1 bytes, or less at start)
            start = max(0, i - n + 1)
            context = tuple(text[start:i])
            next_byte = text[i]
            counts[context][next_byte] += 1
            total_ngrams += 1

    print(f"Counted {total_ngrams:,} n-grams")
    print(f"Unique contexts: {len(counts):,}")

    # Prune rare n-grams to save space
    pruned_counts = {}
    for context, byte_counts in counts.items():
        total = sum(byte_counts.values())
        if total >= min_count:
            # Convert to list format: [[byte, count], ...]
            pruned_counts[str(list(context))] = [
                [b, c] for b, c in byte_counts.items() if c >= min_count
            ]

    print(f"After pruning (min_count={min_count}): {len(pruned_counts):,} contexts")
    return pruned_counts


def save_counts(counts: dict, output_path: Path):
    """Save counts to gzipped JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure .gz extension
    if not str(output_path).endswith('.gz'):
        output_path = output_path.with_suffix('.json.gz')

    json_str = json.dumps(counts, separators=(',', ':'))

    with gzip.open(output_path, 'wt') as f:
        f.write(json_str)

    size_kb = output_path.stat().st_size / 1024
    print(f"Saved to {output_path} ({size_kb:.1f} KB)")

    if size_kb > 900:
        print("\nWARNING: File is close to 1 MB limit!")
        print("Try increasing --min-count or decreasing --n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train n-gram model for byte prediction")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to training data (HuggingFace dataset dir or text files)")
    parser.add_argument("--n", type=int, default=1,
                        help="N-gram order (default: 1 = unigram)")
    parser.add_argument("--min-count", type=int, default=2,
                        help="Minimum count to keep (default: 2)")
    parser.add_argument("--output", type=Path, default=Path("submission/counts.json.gz"),
                        help="Output file path")
    parser.add_argument("--text-mode", action="store_true",
                        help="Load from .txt files instead of HuggingFace dataset")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to load (default: all)")
    args = parser.parse_args()

    print(f"Training {args.n}-gram model")
    print(f"Data: {args.data}")
    print(f"Min count: {args.min_count}")
    print()

    # Load data
    if args.text_mode:
        texts = load_from_text_files(args.data)
    else:
        texts = load_from_hf_dataset(args.data, max_docs=args.max_docs)

    total_bytes = sum(len(t) for t in texts)
    print(f"Total: {total_bytes:,} bytes from {len(texts)} documents")
    print()

    # Train
    counts = train_ngram(texts, args.n, args.min_count)

    # Save
    output_path = save_counts(counts, args.output)

    print()
    print("Done! Now run:")
    print("  python create_submission.py")
    print()
    print("Then upload submission.zip to the competition website.")


if __name__ == "__main__":
    main()
