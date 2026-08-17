PICOLLM:

PicoLLM is a from-scratch LLM inference engine built in PyTorch for autoregressive GPT-2 text generation. It implements KV caching, PagedAttention-style memory management, continuous batching, request scheduling, and configurable sampling, with correctness validated against independent generation. The engine is exposed through a FastAPI REST API and can be containerized for deployment.

Transformer Model - Multi-head self-attention with RoPE position encoding
KV-Cache - Efficient token caching for faster inference
PagedAttention - Block-based memory management (like OS virtual memory)
Continuous Batching - Dynamic scheduling for high throughput
FastAPI Server - Production-ready REST API for inference
Project Structure
# PicoLLM — LLM Inference Engine from Scratch

PicoLLM is a **from-scratch LLM inference engine built in PyTorch** for autoregressive GPT-style text generation. It implements core inference-system components including **KV caching, PagedAttention-style memory management, continuous batching, request scheduling, and configurable sampling**.

The engine is exposed through a **FastAPI REST API** and is designed to be containerized for deployment.

The project focuses on understanding and implementing the systems behind modern LLM inference engines rather than relying on existing inference frameworks.

---

## Key Features

- **Transformer Model**
  - Multi-head self-attention
  - Rotary Position Embeddings (RoPE)
  - Feed-forward network (FFN)
  - Autoregressive text generation

- **KV Cache**
  - Caches previously computed key/value states
  - Avoids redundant computation during autoregressive decoding
  - Supports multiple concurrent sequences

- **PagedAttention-style Memory Management**
  - Block-based KV-cache allocation
  - Dynamic memory management
  - Inspired by OS-style virtual memory concepts

- **Continuous Batching**
  - Dynamically adds and removes requests during inference
  - Improves GPU utilization and throughput
  - Supports concurrent sequence generation

- **Request Scheduling**
  - Manages active inference requests
  - Controls request execution and completion
  - Coordinates with the KV-cache and block manager

- **Configurable Sampling**
  - Temperature-based sampling
  - Top-k sampling
  - Configurable generation parameters

- **FastAPI Inference Server**
  - REST API for text generation
  - Easy integration with external applications
  - Suitable for containerized deployment

- **Correctness Validation**
  - Independent generation tests
  - Component-level testing
  - End-to-end inference validation

---

## Architecture

```text
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Inference Engine   │
                    │    Orchestration     │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │     Scheduler    │          │   KV Cache       │
      │ Continuous       │          │ Multi-Sequence   │
      │ Batching         │          │ Cache            │
      └────────┬─────────┘          └────────┬─────────┘
               │                             │
               ▼                             ▼
      ┌──────────────────┐          ┌──────────────────┐
      │  Block Manager   │◄────────►│ PagedAttention   │
      │ Block Allocation │          │ Memory Access    │
      └────────┬─────────┘          └────────┬─────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    ┌──────────────────────┐
                    │    Transformer      │
                    │  Multi-Head Attn.   │
                    │       + RoPE        │
                    │        + FFN         │
                    └──────────────────────┘
