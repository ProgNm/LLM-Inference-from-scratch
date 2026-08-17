PICOLLM:

PicoLLM is a from-scratch LLM inference engine built in PyTorch for autoregressive GPT-2 text generation. It implements KV caching, PagedAttention-style memory management, continuous batching, request scheduling, and configurable sampling, with correctness validated against independent generation. The engine is exposed through a FastAPI REST API and can be containerized for deployment.

Transformer Model - Multi-head self-attention with RoPE position encoding
KV-Cache - Efficient token caching for faster inference
PagedAttention - Block-based memory management (like OS virtual memory)
Continuous Batching - Dynamic scheduling for high throughput
FastAPI Server - Production-ready REST API for inference
Project Structure
d:\llm\
├── venv/                          # Virtual environment
│   ├── attention.py               # Phase 1: MultiHeadAttention, PagedAttention
│   ├── kv_cache.py                # Phase 2: KVCache, MultiSeqKVCache
│   └── transformer.py             # Phase 1: GPTModel, RotaryEmbedding, FFN
│   ├── block_manager.py           # Phase 3: BlockManager, paged memory
│   ├── scheduler.py               # Phase 4: ContinuousBatchingScheduler
│   └── inference.py               # Phase 5: SimpleLLMEngine orchestration
│   └── api.py                     # Phase 5: FastAPI server
│   ├── 01_ATTENTION.md            # Detailed attention docs
│   ├── 02_KVCACHE.md              # KV-Cache documentation
│   ├── 03_TRANSFORMER.md          # Transformer model docs
│   ├── 04_BLOCKMANAGER.md         # BlockManager documentation
│   ├── 05_SCHEDULER.md            # Scheduler documentation
│   ├── 06_INFERENCE.md            # Inference engine docs
│   ├── 07_API.md                  # FastAPI server docs
│   └── 08_TESTS.md                # Test suite documentation
├── main.py                        # Complete test suite
├── requirements.txt               # Python dependencies
└── README.md                      # This file
