# BPE-Tokenizer

![Tests](https://github.com/tomciolowsky/BPE-Tokenizer/actions/workflows/test.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/uv-managed-DE5FE9.svg?logo=uv)

A pure-Python educational / toy implementation of the **Byte Pair Encoding (BPE)** tokenization algorithm based on OpenAI's GPT-2 tokenizer.

It implements vocabulary loading, rank merging, whitespace-preserving pre-tokenization, encoding, and decoding in pure Python, validated against OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) library for verified token parity across common test suites.

## Features

* **GPT-2 Vocabulary & Merge Ranks**: Loads official GPT-2 vocabulary (`encoder.json`) and merge ranks (`vocab.bpe`).
* **Verified Parity**: Validated against OpenAI's `tiktoken` (`gpt2` encoding) across standard sentences, punctuation, multi-line blocks, and repeated character sequences.
* **Pure Python**: Zero external dependencies for core inference (external tools like `tiktoken` are only used as test-suite oracle dependencies).

## Quick Start

```python
from bpe_tokenizer import Tokenizer

# Initialize tokenizer (loads GPT-2 vocabulary and ranks)
tokenizer = Tokenizer()

# Text to tokenize
text = "This is text with out-of-vocabulary words: Superfluous, Arachnophobia!"

# 1. Encode text to token IDs
tokens = tokenizer.encode(text)
print("Token IDs:", tokens)

# 2. Decode token IDs back to string
decoded_text = tokenizer.decode(tokens)
print("Decoded  :", decoded_text)

assert text == decoded_text
```

Run the example from the repository root:

```bash
uv run python working_examples/tokenizer_demo.py
```

## Installation

### Using uv:
Requires [uv](https://github.com/astral-sh/uv)

```bash
git clone https://github.com/tomciolowsky/BPE-Tokenizer.git
cd BPE-Tokenizer

# Install runtime and dev dependencies (pytest, ruff)
uv sync --all-groups
```

### Using pip:

```bash
git clone https://github.com/tomciolowsky/BPE-Tokenizer.git
cd BPE-Tokenizer
pip install -e .[dev]
```

## Running Tests & Linting

```bash
uv run pytest -v
uv run ruff check .
```

## Resources

* **Learning Materials**:
  * [Sebastian Raschka's Blog](https://sebastianraschka.com/)
* **Test Vectors & Reference**:
  * [OpenAI tiktoken Repository](https://github.com/openai/tiktoken)
  * [OpenAI tiktoken Tests](https://github.com/openai/tiktoken/tree/main/tests)

## License

MIT License — see [LICENSE](LICENSE) for details.
