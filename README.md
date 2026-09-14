# BoundedGlitchEngine-GPT

A minimal, educational GPT implementation for learning how transformer-based language models work.

**Philosophy**: Start with one monolithic `model.py` file that contains everything. Once you understand the architecture, we refactor into separate modules without changing the underlying design.

---

## 📁 Repository Structure

```
The-BoundedGlitchGPT/
├── data/                    # Training data goes here
├── tokenizer/               # Tokenizer utilities (folder for future expansion)
├── model/
│   └── model.py             # **THE CORE FILE** - entire GPT architecture
├── training/
│   └── train.py             # Training loop (coming next)
├── inference/
│   └── generate.py          # Text generation (coming next)
├── tests/                   # Unit tests (folder for future expansion)
├── server/                  # API server (folder for future expansion)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git configuration
└── README.md                # This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Test that the model works

```bash
python model/model.py
```

You should see:
- Model configuration printed
- A forward pass example
- "SUCCESS: Model is working!" message

**What this does:**
- Initializes a GPT with the default config
- Runs a simple prompt through the model
- Shows the predicted next token

---

## 🧠 Understanding the Model

### What is this code?

A **Transformer-based language model** that predicts the next token in a sequence.

### How does it work?

1. **Tokenization**: Text → list of integers
2. **Embedding**: Integers → vectors (token embeddings + positional embeddings)
3. **Transformer Blocks**: Stack of attention + feed-forward layers
4. **Output Head**: Final layer converts hidden states → next-token probabilities

### Key Components (in `model/model.py`)

| Class | What It Does |
|-------|---|
| `GPTConfig` | Hyperparameters (vocabulary size, embedding dim, num layers, etc.) |
| `SimpleTokenizer` | Converts text ↔ token IDs |
| `PositionalEmbedding` | Adds position information to embeddings |
| `MultiHeadAttention` | Self-attention mechanism (key innovation of transformers) |
| `FeedForward` | Position-wise fully-connected network |
| `TransformerBlock` | Combines attention + FFN with residual connections |
| `GPT` | Full model (embeddings + transformer blocks + output head) |

### Important Methods

```python
# Training: forward pass with loss
logits, loss = model(token_ids, targets=target_ids)

# Inference: generate new tokens
generated_ids = model.generate(prompt_ids, max_new_tokens=100)
```

---

## 📊 Model Architecture Diagram

```
Input: "Hello, world!"
    ↓
[Tokenizer]
Token IDs: [123, 45, 67, ...]
    ↓
[Token Embedding + Positional Embedding]
Embedded vectors: (batch_size, seq_len, embed_dim)
    ↓
[Transformer Block 1]  ← Self-Attention → Multi-Head Attention
  └→ Residual Connection
  └→ Feed-Forward Network
  └→ Residual Connection
    ↓
[Transformer Block 2]  ← (repeat num_layers times)
  ...
    ↓
[Transformer Block N]
    ↓
[Output Head]
Logits: (batch_size, seq_len, vocab_size)
    ↓
[Softmax]
Next token probabilities
    ↓
Output: Most likely next token(s)
```

---

## 🔧 Configuration

Edit `GPTConfig` in `model/model.py` to adjust:

```python
class GPTConfig:
    vocab_size = 50_257           # How many tokens?
    max_context_length = 512      # How long is the context?
    embed_dim = 768               # How big are embeddings?
    num_heads = 12                # How many attention heads?
    num_layers = 12               # How many transformer blocks?
    dropout = 0.1                 # Regularization
    ffn_hidden_dim = 3072         # Feed-forward intermediate size
```

**For learning**: Keep these small (smaller = faster to experiment)

```python
# Small model for quick testing
embed_dim = 64
num_heads = 4
num_layers = 2
```

**For production**: Make them larger (bigger = more powerful, but slower)

---

## 📚 Comments in the Code

Every class and method in `model.py` has:
- **What it does** (in English)
- **Why it matters** (context for learning)
- **How it works** (the mechanism)
- **Shapes of tensors** (for debugging)

Read through `model.py` and pause at each comment block to understand.

---

## 🎯 Next Steps

1. ✅ **Read `model/model.py`** — Understand each component
2. 📝 **Run the example** — `python model/model.py` should work
3. 🏋️ **Implement training** — Write `training/train.py`
4. 🎲 **Implement generation** — Write `inference/generate.py`
5. 📊 **Train a model** — Feed it some text data
6. 🚀 **Generate text** — See what it learns

---

## 💡 Common Questions

### Why is everything in one file?

**For learning.** Once you understand how the pieces fit together, we can split it into:
- `model/layers/attention.py`
- `model/layers/feedforward.py`
- `model/transformer_block.py`
- etc.

Refactoring is easy. Understanding is hard. We prioritize understanding first.

### What does `register_buffer` do?

It's like registering a parameter, but the tensor isn't trained. It gets saved/loaded with the model but doesn't get gradient updates. Perfect for fixed embeddings like positional encodings.

### Why causal masking?

In a language model, the model should predict the *next* token. It shouldn't peek at future tokens. Causal masking prevents attention from looking ahead—each position only attends to itself and previous positions.

### What's the difference between attention and self-attention?

**Attention**: Query from one thing, Key/Value from another thing  
**Self-attention**: Query, Key, Value all from the same sequence

In this model, it's all self-attention: the sequence attends to itself.

### Why multiple heads?

Different heads learn different ways to attend:
- Some might focus on nearby tokens
- Some might focus on distant tokens
- Some might look for specific patterns

12 heads = 12 different attention patterns running in parallel.

---

## 📖 Resources for Learning

- **"Attention is All You Need"** (original Transformer paper) — bit.ly/attention-is-all
- **The Illustrated Transformer** — jalammar.github.io/illustrated-transformer/
- **A Friendly Introduction to Attention** — Andrej Karpathy's video

---

## ⚖️ License

The Unlicense (public domain). Do whatever you want with this code.

---

## 🤝 Contributing

This is a learning project. If you:
- Fix bugs
- Add clearer comments
- Improve the examples
- Add tests

...please do! Open an issue or submit a PR.

---

## 📝 Notes

- **Tokenizer**: This uses character-level encoding. In reality, you'd use BPE or WordPiece.
- **Efficiency**: This code prioritizes clarity over speed. Production systems optimize differently.
- **Accuracy**: No pretrained weights. You have to train from scratch.

---

**Ready? Start here:**
```bash
python model/model.py
```

Then read through the comments in `model.py` to understand what's happening.

After that, let's implement training!
