# Word Embeddings from Scratch

A minimal, ground-up implementation of **Word2Vec (Skip-gram)** using only NumPy — no PyTorch, no TensorFlow, no magic.

Built to understand what word embeddings actually are and how they emerge from training a simple two-layer neural network.

---

## What it does

Given a raw text corpus, this implementation:

1. **Tokenizes** the text using regex
2. **Builds a vocabulary** and bi-directional word↔id mappings
3. **Generates training pairs** — for each word, pairs it with surrounding words within a configurable window
4. **One-hot encodes** inputs and labels
5. **Trains a 2-layer neural network** from scratch (forward pass + backpropagation + gradient descent)
6. **Extracts word embeddings** from the first weight matrix after training
7. **Plots the training loss** curve with matplotlib

The key insight: the rows of the first weight matrix `W1` *are* the embeddings. A one-hot input vector just selects a row — making the weight matrix a learned lookup table that maps sparse, high-dimensional word IDs to dense, low-dimensional semantic vectors.

---

## Architecture

```
Input (one-hot, vocab_size)
        ↓  W1 (vocab_size × n_embedding)
   Hidden layer (n_embedding)              ← embeddings live here
        ↓  W2 (n_embedding × vocab_size)
  Output (vocab_size)
        ↓  Softmax
   Probability distribution over vocabulary
```

- No bias terms (intentional — avoids shifting all embeddings by the same constant)
- No activation function on the hidden layer (keeps the embedding space linear)
- Loss function: categorical cross-entropy
- Optimizer: vanilla gradient descent

---

## Quickstart

```bash
# Clone the repo
git clone https://github.com/prashant-gulati/wordembeddings
cd wordembeddings

# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run
python word_embeddings.py
```

---

## Key functions

| Function | What it does |
|---|---|
| `tokenize(text)` | Regex-based tokenizer, returns lowercase word tokens |
| `mapping(tokens)` | Builds `word_to_id` and `id_to_word` dicts from the vocabulary |
| `generate_training_data(tokens, word_to_id, window)` | Produces skip-gram (X, y) pairs as one-hot NumPy arrays |
| `init_network(vocab_size, n_embedding)` | Initializes W1 and W2 weight matrices randomly |
| `forward(model, X)` | Forward pass: linear → linear → softmax |
| `backward(model, X, y, alpha)` | Backprop + gradient descent update, returns cross-entropy loss |
| `get_embedding(model, word)` | Looks up the embedding vector for a word from the trained W1 |

---

## Dependencies

```
numpy
matplotlib
```

---

**Virtual environment & package installation**
```
python3 -m venv /Users/prashantgulati/Documents/dev/python/word_embeddings/.venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Colab version**
https://colab.research.google.com/drive/1J9MgDsVLrBsmNCIkOYmCs1BOqkfolVkR#scrollTo=i6SNxJxFiW9-

**Inspired by**
https://jaketae.github.io/study/word2vec/

**Github**
```
git init
git remote add origin https://github.com/prashant-gulati/wordembeddings
git add .gitignore README.md requirements.txt word_embeddings.py
git commit -m "$(cat <<'EOF'
Initial commit: word embeddings project. Add word_embeddings.py, README, requirements, and .gitignore.
EOF
)"
git push -u origin main 2>&1
```
