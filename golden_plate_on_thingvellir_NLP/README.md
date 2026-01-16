# The Golden Plate at Þingvellir (BabyLM - NLP):

The government is preparing a digital time capsule to be buried at Þingvellir. You have been tasked with creating an Icelandic language model that will be written on a golden plate and stored in the time capsule. This model should preserve as much Icelandic linguistic knowledge as possible so it can be recovered later if the future takes a turn for the worst. However, the catch is that the golden plate can only hold one megabyte. You must gather text data and teach a small model as much general linguistic knowledge as you can. The model will then be evaluated on hidden data from Risamálheild to test its Icelandic capabilities. The future of the Icelandic language is in your hands!


**Goal:** Build a model that predicts the next byte in a sequence of text. Lower bits-per-byte = better compression = higher score on the leaderboard.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train your own model on the training data
python train_ngram.py --data data/igc_full

# 3. Create submission zip
python create_submission.py

# 4. Validate your submission (recommended!)
python check_submission.py

# 5. Upload submission.zip to the competition website
```

> **Note:** Run `python create_dataset.py` first to download the training data from HuggingFace.

**The included `submission.zip` is just a starting point!** We recommend you train your own model using the provided training code - even a simple n-gram model trained on more data will score better.

---

## Submission Format

Your submission must be a `.zip` file (max 1 MB) with this structure:

```
submission.zip
├── model.py        # REQUIRED - must contain a Model class
├── weights.bin     # Optional - model weights
├── config.json     # Optional - configuration
└── ...             # Any other files your model needs
```

### Required Interface

Your `model.py` must implement this exact interface:

```python
from pathlib import Path

class Model:
    def __init__(self, submission_dir: Path):
        """
        Load your model here.

        Args:
            submission_dir: Path to the extracted submission folder
                           containing all your files (weights, config, etc.)
        """
        pass

    def predict(self, contexts: list[list[int]]) -> list[list[float]]:
        """
        Predict the next byte for each context.

        Args:
            contexts: Batch of byte sequences. Each sequence is a list of
                     integers (0-255) representing the preceding bytes.
                     Sequences have variable length (0 to 512 bytes).

        Returns:
            Logits for next byte prediction.
            Shape: [batch_size, 256] - one row per context, 256 possible bytes.
            These are RAW LOGITS (will be passed through softmax for scoring).
        """
        pass
```

---

## Scoring

Your model is evaluated on how well it predicts the next byte:

1. For each position in the test data, your model receives the previous 512 bytes as context
2. Your model returns logits (unnormalized probabilities) for each of the 256 possible next bytes
3. We compute cross-entropy loss: `-log2(softmax(logits)[correct_byte])`
4. Final score = average bits per byte across all predictions

**Lower is better!**
- ~8 bits/byte = random guessing (uniform distribution)
- ~5 bits/byte = basic compression (baseline)
- ~2 bits/byte = good language model
- ~1 bit/byte = excellent compression

---

## Constraints

| Constraint | Value |
|------------|-------|
| Max zip size | **1 MB** |
| Max uncompressed | 50 MB |
| Memory limit | 4 GB |
| CPU | 2 cores (no GPU) |
| Network | **None** - completely sandboxed |
| Context window | 512 bytes max |
| Batch size | 1024 contexts per call |
| Time limit | 10 minutes total |

---

## Available Packages

The evaluation environment has these packages pre-installed:

| Package | Version | Notes |
|---------|---------|-------|
| `torch` | 2.9.1 | **CPU-only** - main framework |
| `transformers` | 4.57.6 | HuggingFace models |
| `tensorflow-cpu` | 2.20.0 | TensorFlow (CPU) |
| `jax` | 0.8.2 | JAX ecosystem |
| `flax` | 0.12.2 | JAX ecosystem |
| `numpy` | 2.4.1 | Numerical computing |
| `scipy` | 1.17.0 | Scientific computing |
| `safetensors` | 0.7.0 | Fast weight loading |
| `datasets` | 4.5.0 | HuggingFace datasets |
| `pyarrow` | 22.0.0 | Data serialization |

### Need a different package?

If you need a package that's not listed above, **reach out on Discord early in the week** (early in the week). We may be able to add it to the environment, but we need time to test and redeploy.

**Do NOT assume we can add packages last-minute!**

---

## The Dataset

The training data comes from the **IGC-2024** (Icelandic Gigaword Corpus), a large collection of Icelandic text.

```bash
# Download and prepare the training data (~2.1M documents)
python create_dataset.py
```

This creates `data/igc_full/` containing the training documents.

**Important:** The validation and test sets used for evaluation are sampled from the same IGC corpus.

---

## Validating Your Submission

Before uploading, always validate your submission:

```bash
python check_submission.py submission.zip
```

This checks:
- File size limits (1 MB compressed, 50 MB uncompressed)
- ZIP format and structure
- `model.py` exists with correct `Model` class
- `__init__(self, submission_dir)` and `predict()` methods exist
- Model can be instantiated and runs without errors
- Output format is correct (list of 256 logits per context)
- Can handle batch of 1024 contexts

**Always run this before submitting!** It catches most common errors.

---

## Training Your Own Model

We provide code to train n-gram models. **We recommend you start by training your own model** - it's easy and will improve your score!

```bash
# Step 1: Download the training data (only need to do this once)
python create_dataset.py

# Step 2: Train a model (try different n values!)
python train_ngram.py --data data/igc_full --n 2   # bigram
python train_ngram.py --data data/igc_full --n 3   # trigram (recommended)

# Step 3: Create and validate your submission
python create_submission.py
python check_submission.py
```

**N-gram options:**
- `--n 1` = Unigram (predicts based on global byte frequencies)
- `--n 2` = Bigram (uses the previous byte as context)
- `--n 3` = Trigram (uses the previous 2 bytes as context)
- `--n 4+` = Higher order (more context, but larger file)

Higher n = more context = better predictions, but larger file size. Use `--min-count` to prune rare patterns and keep file under 1 MB.

**We encourage you to explore different approaches!** N-gram models are just a starting point - consider neural networks, transformers, or other creative solutions.

---

## Tips

1. **Start simple** - The provided n-gram baseline is a good starting point
2. **Test locally** - Make sure your model loads and runs before submitting
3. **Watch the size limit** - 1 MB compressed, 50 MB uncompressed
4. **No network** - Your model cannot download weights at runtime
5. **CPU only** - Optimize for CPU inference, not GPU
6. **Batch efficiently** - You receive 1024 contexts at once, vectorize!

---

## Files in This Repository

| File | Description |
|------|-------------|
| `README.md` | This file |
| `requirements.txt` | Python dependencies |
| `create_dataset.py` | Downloads training data from HuggingFace |
| `train_ngram.py` | Script to train n-gram model |
| `create_submission.py` | Script to create submission.zip |
| `check_submission.py` | Validates your submission before upload |
| `submission/model.py` | Example model implementation |
| `submission.zip` | Ready-to-upload unigram baseline |

---

## Questions?

Ask on Discord! But remember - if you need extra packages, ask **early in the week**.

Good luck!
