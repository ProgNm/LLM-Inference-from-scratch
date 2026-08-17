

## 1. The Story -- Why We Built This

The real **LLM** (from UC Berkeley, 2023) is a production LLM serving engine known for near-perfect GPU memory utilisation and up to 24x higher throughput than naive inference servers. Its two core ideas -- **PagedAttention** and **Continuous Batching** -- were originally described in:

> Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023)

This project is a **ground-up re-implementation** of those ideas. The goal is not just to copy LLM API, but to deeply understand:

- **How a GPT-style Transformer works internally** -- token embeddings, positional embeddings, multi-head attention (QKV), feed-forward network, layer norms, residual connections, the LM head.
- **Why the KV Cache exists** -- and exactly what goes wrong (position errors, wrong logits) if you implement it incorrectly.
- **How PagedAttention eliminates memory fragmentation** -- the direct analogy to OS virtual memory / paging.
- **Why Continuous Batching matters** -- static batching wastes GPU cycles; continuous batching fills every slot immediately when one sequence finishes.
- **How to load HuggingFace weights into a custom model** -- the Conv1D to nn.Linear transposition trick, fused QKV splitting.
- **How to wrap it all in a production REST API** -- FastAPI + uvicorn + Pydantic schemas + Docker.

Every component was verified with mathematical correctness proofs baked into the test suite.

---

## 2. High-Level Architecture

The engine has four layers that communicate top-down:

`
HTTP Request
    |
    v  [FastAPI Server]  serving/api.py
    |
    |  add_request()
    v  [SimpleLLMEngine]  engine/inference.py
    |
    +-- ContinuousBatchingScheduler  engine/scheduler.py
    |       State machine: WAITING -> RUNNING -> FINISHED
    |       FIFO queue, max_batch_size enforcement
    |
    +-- BlockManager  engine/block_manager.py
    |       16-token KV blocks, free_blocks set, block_table dict
    |       Mirrors OS virtual memory paging
    |
    |  forward pass per sequence
    v  [GPTModel]  model/transformer.py
    |
    |  Token Embedding + Positional Embedding (with past_length offset)
    |  [TransformerBlock x N layers]
    |      LayerNorm -> MultiHeadAttention (+ per-sequence KV Cache) -> Residual
    |      LayerNorm -> FeedForwardNetwork (GELU) -> Residual
    |  Final LayerNorm -> LM Head (weight-tied to token embedding)
    |
    |  logits (vocab_size,)
    v  [Sampling Pipeline]  model/sampling.py
    |
    rep_penalty -> temperature -> top-k -> top-p -> multinomial -> next_token_id

Pre-trained weights: HuggingFace GPT-2 loaded by model/load_weights.py
`

---

## 3. Project Structure

`
LLM-from-scratch/
|
+-- main.py                   # Entry point: runs the full 9-test suite
|
+-- model/
|   +-- __init__.py
|   +-- transformer.py        # GPTModel, TransformerBlock, FFN, RoPE
|   +-- attention.py          # MultiHeadAttention, PagedAttention
|   +-- kv_cache.py           # KVCache, MultiSeqKVCache
|   +-- sampling.py           # SamplingParams, sample_token pipeline
|   +-- load_weights.py       # HuggingFace GPT-2 weight loader
|
+-- engine/
|   +-- __init__.py
|   +-- block_manager.py      # BlockManager (PagedAttention memory)
|   +-- scheduler.py          # ContinuousBatchingScheduler
|   +-- inference.py          # SimpleLLMEngine (the main orchestrator)
|
+-- serving/
|   +-- __init__.py
|   +-- api.py                # FastAPI server with uvicorn
|
+-- docs/
|   +-- 01_ATTENTION.md
|   +-- 02_KVCACHE.md
|   +-- 03_TRANSFORMER.md
|   +-- 04_BLOCKMANAGER.md
|   +-- 05_SCHEDULER.md
|   +-- 06_INFERENCE.md
|   +-- 07_API.md
|   +-- 08_TESTS.md
|   +-- DOCUMENTATION.md
|
+-- Dockerfile                # Production Docker image
+-- requirements.txt          # Pinned Python dependencies
+-- output.txt                # Captured output of python main.py
+-- new_readme.md             # This file
`
