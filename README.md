# BPE-Tokenizer

![Tests](https://github.com/tomciolowsky/BPE-Tokenizer/actions/workflows/test.yml/badge.svg)

My implementation of the **Byte Pair Encoding (BPE)** tokenization algorithm used in OpenAI's GPT-2 model.
Its a toy version of OpenAI's library **tiktoken**.
I implemented: vocabulary loading, rank merging, pre-tokenization, encoding and decoding in pure Python


## Features

* **GPT-2 Tokenization**: I used OpenAI's vocabulary (`encoder.json`) and ranks (`vocab.bpe`).
* **Verified Parity**: I tested against OpenAI's [`tiktoken`](https://github.com/openai/tiktoken) library to produce 100% identical token IDs.
* **Pure Python**: No external libraries.

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

Run example from the repo:
```bash
python working_examples/tokenizer_demo.py
```

## Installation

Standard Installation:

```bash
git clone https://github.com/tomciolowsky/BPE-Tokenizer.git
cd BPE-Tokenizer
pip install .
```

Development Installation (modify or run tests):

```bash
git clone https://github.com/tomciolowsky/BPE-Tokenizer.git
cd BPE-Tokenizer
pip install -e .[dev]
```

## Running Tests

I used `pytest` for testing.

To run tests:

```bash
pytest
```

## Resources

* Learning materials:

    [Sebastian Raschka Blog](https://sebastianraschka.com/)

* Tests:

    [OpenAI tiktoken Repo](https://github.com/openai/tiktoken/tree/main/tests)