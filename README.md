# BaNaNa SLM 300M

A lightweight decoder-only Transformer language model inspired by modern architectures such as Llama, Gemma, and Mistral.

BaNaNa SLM 300M is designed to be:
- simple
- hackable
- efficient
- modern
- easy to train
- small enough for consumer hardware
- fully implemented in raw PyTorch

The model uses:
- RoPE positional embeddings
- RMSNorm
- SwiGLU feed-forward blocks
- Grouped Query Attention (GQA)
- causal decoder-only Transformer architecture
- tied embeddings
- Flash/SDPA attention through PyTorch

---

# Overview

This project focuses on building a modern Small Language Model (SLM) with approximately 300 million parameters while keeping the implementation extremely clean and understandable.

The architecture avoids unnecessary complexity while still including most improvements used in current production LLMs.

Main goals:
- educational value
- research experimentation
- lightweight pretraining
- instruction tuning
- local inference
- architecture exploration

---

# Model Architecture

VOCAB_SIZE → 32000  
HIDDEN → 1024  
INTER → 2816  
N_LAYERS → 24  
N_HEADS → 16  
N_KV_HEADS → 8  
HEAD_DIM → 64  
MAX_SEQ → 2048  
ROPE_THETA → 10000.0  
RMS_EPS → 1e-5  
TIE_EMBED → True  

Approximate parameter count:

~300 million parameters

---

# Architecture Details

## Embedding Layer

The model starts with a learned token embedding table using:

nn.Embedding(VOCAB_SIZE, HIDDEN)

This converts token IDs into dense hidden vectors.

---

## Rotary Positional Embeddings (RoPE)

BaNaNa uses Rotary Positional Embeddings instead of learned positional embeddings.

Advantages:
- better extrapolation
- efficient long-context behavior
- no extra positional parameters
- modern LLM standard

Implemented through sinusoidal rotation of Q/K vectors.

---

## RMSNorm

Instead of LayerNorm, the model uses RMSNorm:

x / sqrt(mean(x²) + eps)

Advantages:
- faster
- simpler
- more numerically stable
- widely used in modern LLMs

---

## SwiGLU Feed Forward

The FFN uses SwiGLU activation:

SiLU(gate(x)) * up(x)

Advantages:
- stronger expressiveness
- smoother gradients
- better parameter efficiency

Used in:
- Llama
- Gemma
- PaLM

---

## Grouped Query Attention (GQA)

BaNaNa implements GQA:

16 query heads  
8 key/value heads

Advantages:
- lower KV-cache memory usage
- faster inference
- reduced bandwidth cost
- modern inference optimization

---

## Attention Backend

Attention is computed using:

torch.nn.functional.scaled_dot_product_attention

This enables:
- Flash Attention kernels
- memory-efficient attention
- optimized CUDA kernels
- PyTorch fused attention paths

---

# Training Configuration

SEQ_LEN → 512  
BATCH → 8  
GRAD_ACCUM → 4  
MAX_STEPS → 1080  
MAX_LR → 3e-4  
WEIGHT_DECAY → 0.1  
GRAD_CLIP → 1.0  

Effective batch size:

32

Tokens processed per optimizer step:

16384

Total training tokens:

~17.7 million tokens

---

# Dataset Composition

Example training mixture:

50% FineWeb  
30% C4  
10% Mathematics  
9% Code  
1% OASST1  

The model can be pretrained on:
- web text
- programming code
- instruction datasets
- mathematical corpora

---

# Initialization Strategy

BaNaNa uses carefully scaled initialization.

## Embeddings

std = HIDDEN ** -0.5

This keeps initial logits near zero.

---

## Linear Layers

N(0, 0.02)

Standard Transformer initialization.

---

## Residual Scaling

Output projections are scaled using residual normalization scaling to stabilize deep residual streams.

This helps prevent exploding activations during training.

---

# Features

- Modern decoder-only architecture
- Fully causal autoregressive training
- Flash Attention compatible
- BF16 support
- Gradient checkpointing compatible
- GQA inference optimization
- RoPE positional encoding
- Lightweight implementation
- Pure PyTorch
- Easy to modify
- HuggingFace compatible export

---

# Why This Architecture?

BaNaNa intentionally focuses on:
- simplicity
- readability
- modern best practices
- efficiency

The project avoids:
- unnecessary abstraction
- giant framework overhead
- overengineered training systems

The result is a compact research-friendly codebase.

---

# Hardware Targets

The model is intended to run on:
- consumer GPUs
- Colab
- local Linux machines
- low-memory setups
- experimental CPU training

Training large runs still benefits heavily from GPUs.

---

# Intended Use Cases

- language modeling research
- tokenizer experiments
- architecture prototyping
- educational purposes
- small-scale finetuning
- local inference
- lightweight chatbots
- code generation experiments

---

# Future Improvements

Possible upgrades:
- Mixture of Experts (MoE)
- sliding window attention
- speculative decoding
- YaRN / NTK-aware RoPE scaling
- quantization
- KV-cache optimization
- distributed training
- inference kernels
- RLHF / DPO alignment
- multilingual training

---

# Example Forward Pass

x = self.embed(input_ids)

for block in self.blocks:
    x = block(x)

logits = self.lm_head(self.norm(x))

Simple, clean, and efficient.

---

# Philosophy

BaNaNa exists to prove that:
- modern LLMs are understandable
- clean code matters
- small models are valuable
- experimentation should be accessible

The project prioritizes transparency over complexity.

---

# License

Choose any license you prefer:
- MIT
- Apache-2.0
- GPLv3
- custom research license

---

# Acknowledgements

Inspired by ideas from:
- Llama
- Gemma
- Mistral
- Transformer research papers
- the open-source ML community

---

# Final Notes

BaNaNa SLM 300M is not intended to compete with frontier-scale models.

Instead, it focuses on:
- accessibility
- experimentation
- education
- lightweight research

A small, modern Transformer can still teach an enormous amount about how large language models actually work.
