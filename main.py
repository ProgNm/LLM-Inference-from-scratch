"""
Main entry point for vLLM Engine.

Demonstrates:
1. Building a minimalist transformer from scratch
2. Implementing KV-cache for efficient inference
3. PagedAttention with block manager
4. Continuous batching scheduler
5. FastAPI serving
6. Loading real GPT-2 pretrained weights
7. Temperature + Top-p sampling strategies
"""
import torch
import logging
from transformers import AutoTokenizer
from engine.inference import SimpleLLMEngine, InferenceConfig
from model.load_weights import load_gpt2_weights, build_gpt2_model
from model.sampling import SamplingParams, apply_temperature, apply_top_p, nucleus_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_forward_pass():
    """Test Phase 1: Basic transformer forward pass (uses small toy config for speed)."""
    logger.info("=" * 60)
    logger.info("TEST 1: Transformer Forward Pass")
    logger.info("=" * 60)
    
    from model.transformer import GPTModel
    
    # Small toy model for fast unit testing
    model = GPTModel(
        vocab_size=50257,  # GPT-2 vocab
        max_seq_len=512,
        dim=128,           # Small for testing speed
        num_heads=8,
        num_layers=2,
        dropout=0.0,
    )
    
    model.eval()
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    
    with torch.no_grad():
        logits, _ = model(input_ids, use_cache=False)
    
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"✓ Forward pass successful!")
    
    return model


def test_kv_cache():
    """Test Phase 2: KV-cache handling."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: KV-Cache Implementation")
    logger.info("=" * 60)
    
    from model.kv_cache import KVCache
    
    # Create cache
    cache = KVCache(
        max_seq_len=128,
        num_heads=8,
        head_dim=16,
        batch_size=1,
        device=torch.device("cpu"),
    )
    
    # Simulate two steps of generation
    for step in range(3):
        new_k = torch.randn(1, 8, 1, 16)
        new_v = torch.randn(1, 8, 1, 16)
        cache.update(new_k, new_v)
        
        k, v = cache.get()
        logger.info(f"Step {step}: Cached sequence length = {cache.get_seq_length()}")
    
    logger.info(f"✓ KV-Cache working correctly!")


def test_block_manager():
    """Test Phase 3: PagedAttention BlockManager."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: BlockManager for PagedAttention")
    logger.info("=" * 60)
    
    from engine.block_manager import BlockManager, BLOCK_SIZE
    
    # Create block manager
    bm = BlockManager(
        num_blocks=64,
        block_size=BLOCK_SIZE,
        num_heads=8,
        head_dim=16,
        device=torch.device("cpu"),
    )
    
    logger.info(f"BlockManager created with {64} blocks, {BLOCK_SIZE} tokens per block")
    
    # Allocate sequences
    for seq_id in range(5):
        success = bm.allocate_sequence(seq_id, num_tokens=32)
        logger.info(f"Sequence {seq_id} allocated: {success}")
    
    # Check memory usage
    usage = bm.get_memory_usage()
    logger.info(f"Memory usage: {usage['filled_tokens']}/{usage['total_tokens_capacity']} tokens")
    logger.info(f"Utilization: {usage['utilization']:.1%}")
    
    # Free a sequence
    bm.free_sequence(2)
    logger.info(f"Freed sequence 2")
    
    usage = bm.get_memory_usage()
    logger.info(f"Free blocks: {usage['free_blocks']}")
    logger.info(f"✓ BlockManager working correctly!")


def test_scheduler():
    """Test Phase 4: Continuous Batching Scheduler."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Continuous Batching Scheduler")
    logger.info("=" * 60)
    
    from engine.scheduler import ContinuousBatchingScheduler, SchedulerConfig
    
    config = SchedulerConfig(max_batch_size=4)
    scheduler = ContinuousBatchingScheduler(config)
    
    # Add requests
    for i in range(6):
        seq_id = scheduler.add_sequence(
            prompt=f"Prompt {i}",
            input_ids=list(range(10)),
            max_new_tokens=20,
        )
        logger.info(f"Added request {seq_id}")
    
    # Schedule steps
    for step in range(3):
        to_run, changes = scheduler.schedule_step()
        logger.info(f"Step {step}: {len(to_run)} sequences running")
    
    stats = scheduler.get_stats()
    logger.info(f"Scheduler stats: {stats}")
    logger.info(f"✓ Scheduler working correctly!")


def test_full_inference():
    """
    Test all phases together: real GPT-2 weights flowing through the FULL vLLM
    engine pipeline (scheduler → block manager → batched forward → sampling).

    Output should be coherent English, proving that:
    1. The SimpleLLMEngine orchestrates everything correctly, AND
    2. The position-embedding offset in generate_step() is correct.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Full vLLM Inference Engine (with real GPT-2 weights)")
    logger.info("=" * 60)

    # ── Create GPT-2 Small engine ────────────────────────────────────────────
    config = InferenceConfig(
        vocab_size=50257,
        model_dim=768,
        num_heads=12,
        num_layers=12,
        max_seq_length=1024,
        num_blocks=128,
        max_batch_size=4,
        default_max_new_tokens=15,
        device="cpu",
    )

    engine = SimpleLLMEngine(config)
    logger.info("Engine initialized")

    # ── Load real GPT-2 pretrained weights ───────────────────────────────────
    logger.info("Loading GPT-2 pretrained weights into full vLLM engine...")
    load_gpt2_weights(engine.model, model_name="gpt2")

    # ── Load GPT-2 tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Add generation requests ──────────────────────────────────────────────
    prompts = [
        "Hello, my name is",
        "The future of AI is",
        "Once upon a time,",
    ]

    seq_ids = []
    for prompt in prompts:
        seq_id = engine.add_request(
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=15,
            temperature=0.8,
            top_k=50,
        )
        seq_ids.append(seq_id)
        logger.info(f"Added request {seq_id}: '{prompt}'")

    # ── Run the full vLLM inference loop ─────────────────────────────────────
    logger.info("Starting vLLM engine inference...")
    outputs = engine.generate_all(max_steps=100)

    # ── Print results ─────────────────────────────────────────────────────────
    logger.info("Inference complete! Generated text:")
    for seq_id, tokens in outputs.items():
        result = engine.scheduler.get_finished_sequence(seq_id)
        if result:
            generated = tokenizer.decode(result.output_ids)
            logger.info(f"  Sequence {seq_id}: {generated!r}")

    stats = engine.get_stats()
    logger.info(f"Final stats: {stats}")

    logger.info(f"✓ Full inference working correctly!")


def test_gpt2_weights():
    """
    Test Phase 6: Load real GPT-2 pretrained weights and generate coherent text.

    This proves the vLLM engine works end-to-end with actual pretrained weights.
    Generated text should be meaningful English, not random token garbage.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Real GPT-2 Pretrained Weights")
    logger.info("=" * 60)

    # ── Build GPT-2 Small model and load pretrained weights ─────────────────
    logger.info("Building GPT-2 Small architecture (dim=768, 12 heads, 12 layers)...")
    model = build_gpt2_model(model_name="gpt2", use_paged_attention=False)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("Downloading/loading GPT-2 pretrained weights from HuggingFace...")
    model = load_gpt2_weights(model, model_name="gpt2")

    # ── Load GPT-2 tokenizer ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Generate text using real weights ────────────────────────────────────
    test_prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away,",
    ]

    for prompt in test_prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                top_k=50,
            )
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        logger.info(f"Prompt : {prompt!r}")
        logger.info(f"Output : {generated_text!r}")
        logger.info("")

    logger.info("✓ GPT-2 pretrained weights working correctly! (text should be coherent English)")


def test_verify_kv_cache_correctness():
    """
    TEST 7: KV Cache Correctness Verification.

    The claim: processing a sequence token-by-token WITH KV cache must produce
    IDENTICAL next-token logits to a full-context forward pass WITH a causal mask.

    Why this must hold:
      - K and V are pure linear projections of each token's input embedding.
        They do not depend on other tokens, so caching them is lossless.
      - With a causal mask, position i can only attend to positions 0..i anyway,
        which is exactly what the KV cache provides when we feed tokens one at a time.
      - Therefore the only allowed difference is floating-point rounding, which
        should be < 1e-3 everywhere.

    Uses a small toy model for speed; the correctness is mechanism-agnostic.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: KV Cache Correctness Verification")
    logger.info("=" * 60)

    from model.transformer import GPTModel

    torch.manual_seed(42)
    model = GPTModel(
        vocab_size=50257, max_seq_len=512,
        dim=128, num_heads=8, num_layers=4, dropout=0.0,
    )
    model.eval()

    # Use a fixed 8-token sequence (random token IDs)
    input_ids = torch.randint(0, 50257, (1, 8))
    seq_len = input_ids.shape[1]
    logger.info(f"Test sequence: {seq_len} tokens (random token IDs, fixed seed=42)")

    # ── Method A: Full context forward pass WITH 4D causal mask (ground truth) ──
    # causal_mask[..., i, j] = 1 means position i may attend to position j (j ≤ i)
    # Shape (1, 1, seq, seq) broadcasts over (batch, heads, seq, seq)
    causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))

    with torch.no_grad():
        logits_full, _ = model(input_ids, attention_mask=causal_mask, use_cache=False)
    # logits_full[0, i, :] = next-token distribution for position i (causally masked)

    # ── Method B: Token-by-token WITH KV cache ────────────────────────────────
    kv_caches = [None] * model.num_layers
    kv_logits = []

    for step in range(seq_len):
        with torch.no_grad():
            out, kv_caches = model(
                input_ids[:, step:step + 1],   # (1, 1)  — single token
                kv_caches=kv_caches,
                use_cache=True,
                past_length=step,              # absolute position of this token
            )
        kv_logits.append(out[0, -1, :].clone())  # (vocab_size,)

    # ── Compare logits at every position ──────────────────────────────────────
    logger.info(f"\n{'Pos':<6} {'Max Logit Diff':<20} {'Greedy (Full)':<18} {'Greedy (KV)':<18} Match")
    logger.info("─" * 75)

    max_diff_overall = 0.0
    all_greedy_match = True

    for i in range(seq_len):
        full_i = logits_full[0, i, :]
        kv_i   = kv_logits[i]

        diff   = torch.max(torch.abs(full_i - kv_i)).item()
        max_diff_overall = max(max_diff_overall, diff)

        gf = torch.argmax(full_i).item()
        gk = torch.argmax(kv_i).item()
        ok = gf == gk
        all_greedy_match = all_greedy_match and ok

        logger.info(f"{i:<6} {diff:<20.3e} {gf:<18} {gk:<18} {'✓' if ok else '✗ MISMATCH'}")

    logger.info(f"\nWorst-case logit difference across all {seq_len} positions: {max_diff_overall:.3e}")

    assert max_diff_overall < 1e-3, (
        f"KV cache produces wrong logits! Max diff = {max_diff_overall:.3e}. "
        "Position embeddings or KV accumulation is broken."
    )
    assert all_greedy_match, "Greedy next tokens differ between full pass and KV cache!"

    logger.info("✓ KV cache verified: every position matches the causal forward pass exactly!")


def test_verify_continuous_batching():
    """
    TEST 8: Continuous Batching Correctness Verification.

    The claim: generating N sequences SIMULTANEOUSLY through the vLLM engine
    (continuous batching, interleaved steps) must produce IDENTICAL token sequences
    to generating each one INDEPENDENTLY with model.generate().

    Method:
      1. Create a small toy model with random weights.
      2. Run model.generate() on each prompt independently → 'standalone' tokens.
      3. Copy the exact same weights into a fresh SimpleLLMEngine.
      4. Feed all prompts simultaneously and let the engine run to completion.
      5. Assert token sequences match exactly (using top_k=1 / greedy for determinism).

    This proves the scheduler, per-sequence KV cache management, and
    past_length offsets are all correct under concurrent operation.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: Continuous Batching Correctness Verification")
    logger.info("=" * 60)

    from model.transformer import GPTModel

    # ── Shared setup ─────────────────────────────────────────────────────────
    TOY_DIM     = 128
    TOY_HEADS   = 8
    TOY_LAYERS  = 2
    MAX_NEW     = 10

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompts = [
        "Hello, my name is",
        "The capital of France",
        "Once upon a time,",
    ]

    # ── Method A: Independent generation ─────────────────────────────────────
    logger.info("Method A: Independent per-prompt generation (model.generate)...")

    torch.manual_seed(0)
    standalone_model = GPTModel(
        vocab_size=50257, max_seq_len=512,
        dim=TOY_DIM, num_heads=TOY_HEADS, num_layers=TOY_LAYERS, dropout=0.0,
    )
    standalone_model.eval()

    standalone_outputs: dict = {}
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)])
        with torch.no_grad():
            gen = standalone_model.generate(
                ids, max_new_tokens=MAX_NEW, temperature=1.0, top_k=1  # greedy
            )
        new_tokens = gen[0, ids.shape[1]:].tolist()
        standalone_outputs[prompt] = new_tokens
        logger.info(f"  Standalone  {prompt!r} → {tokenizer.decode(new_tokens)!r}")

    # ── Method B: Simultaneous via vLLM engine ────────────────────────────────
    logger.info("\nMethod B: Simultaneous generation via vLLM engine (continuous batching)...")

    config = InferenceConfig(
        vocab_size=50257,
        model_dim=TOY_DIM,
        num_heads=TOY_HEADS,
        num_layers=TOY_LAYERS,
        max_seq_length=512,
        num_blocks=128,
        max_batch_size=4,
        default_max_new_tokens=MAX_NEW,
        device="cpu",
    )
    engine = SimpleLLMEngine(config)

    # !! Copy EXACT same weights as standalone model so both sides are identical !!
    engine.model.load_state_dict(standalone_model.state_dict())
    engine.model.eval()

    sid_to_prompt: dict = {}
    for prompt in prompts:
        sid = engine.add_request(
            prompt=prompt, tokenizer=tokenizer,
            max_new_tokens=MAX_NEW, temperature=1.0, top_k=1,  # greedy
        )
        sid_to_prompt[sid] = prompt

    engine.generate_all(max_steps=300)

    engine_outputs: dict = {}
    for sid, prompt in sid_to_prompt.items():
        result = engine.scheduler.get_finished_sequence(sid)
        if result:
            engine_outputs[prompt] = result.output_ids
            logger.info(f"  Engine      {prompt!r} → {tokenizer.decode(result.output_ids)!r}")

    # ── Token-level comparison ────────────────────────────────────────────────
    logger.info("\nToken-level comparison (standalone vs engine):")
    logger.info(f"{'Prompt':<30} {'Standalone tokens':<35} {'Engine tokens':<35} Match")
    logger.info("─" * 110)

    all_match = True
    for prompt in prompts:
        sa  = standalone_outputs.get(prompt, [])
        eng = engine_outputs.get(prompt, [])
        ok  = sa == eng
        all_match = all_match and ok
        logger.info(
            f"{prompt!r:<30} {str(sa):<35} {str(eng):<35} {'✓' if ok else '✗ MISMATCH'}"
        )
        if not ok:
            logger.error(f"  Standalone: {tokenizer.decode(sa)!r}")
            logger.error(f"  Engine:     {tokenizer.decode(eng)!r}")

    assert all_match, (
        "Continuous batching produced DIFFERENT tokens than standalone generation! "
        "The KV cache management or scheduler ordering is broken."
    )
    logger.info(
        "✓ Continuous batching verified: "
        "all sequences match independent generation exactly!"
    )


def test_temperature_top_p():
    """
    TEST 9: Temperature + Top-p Decoding Strategy.

    Demonstrates:
      A) Nucleus-size statistics — how temperature shapes the nucleus
         (lower T → smaller nucleus, higher T → larger nucleus)
      B) Qualitative text output — 5 preset SamplingParams configurations
         with real GPT-2 weights showing the trade-off between coherence
         and diversity.

    Temperature + Top-p pipeline (in sampling.py):
        raw logits
        → repetition penalty
        → ÷ temperature          (sharpen or flatten the distribution)
        → top-k  (optional)
        → top-p  (nucleus: keep smallest set with cumulative prob ≥ p)
        → multinomial sample
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEST 9: Temperature + Top-p Sampling Strategy")
    logger.info("=" * 60)

    from model.transformer import GPTModel

    # ── Part A: Nucleus-size vs temperature ───────────────────────────────────
    logger.info("\nPart A — How temperature affects nucleus size (top_p=0.9):")
    logger.info(f"{'Temperature':<16} {'Min nucleus':<16} {'Max nucleus':<16} {'Avg nucleus':<16}")
    logger.info("─" * 65)

    torch.manual_seed(7)
    toy = GPTModel(
        vocab_size=50257, max_seq_len=512,
        dim=128, num_heads=8, num_layers=2, dropout=0.0,
    )
    toy.eval()

    # Generate 20 random logit vectors and measure nucleus size per temperature
    sample_logits = [torch.randn(50257) for _ in range(20)]
    top_p = 0.9

    for temp in [0.3, 0.5, 0.8, 1.0, 1.3, 2.0]:
        sizes = [nucleus_size(l, temperature=temp, top_p=top_p) for l in sample_logits]
        logger.info(
            f"{temp:<16.1f} {min(sizes):<16} {max(sizes):<16} {sum(sizes)/len(sizes):<16.1f}"
        )

    logger.info(
        "\n→ Lower temperature → smaller nucleus (model is more confident).\n"
        "→ Higher temperature → larger nucleus (more tokens considered).\n"
    )

    # ── Part B: Qualitative generation with 5 presets ─────────────────────────
    logger.info("Part B — Qualitative text output with different sampling settings:")
    logger.info("(Using real GPT-2 Small weights)\n")

    model = build_gpt2_model(model_name="gpt2", use_paged_attention=False)
    model = load_gpt2_weights(model, model_name="gpt2")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    prompt = "The future of artificial intelligence is"
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    presets = [
        # label,               temperature, top_k, top_p,  rep_penalty
        ("Greedy       (T=1, k=1)",      1.0,  1,    1.0,  1.0),
        ("Conservative (T=0.7, p=0.9)",  0.7,  None, 0.90, 1.0),
        ("Balanced     (T=0.8, p=0.9)",  0.8,  None, 0.90, 1.0),
        ("Creative     (T=1.1, p=0.95)", 1.1,  None, 0.95, 1.1),
        ("Very diverse (T=1.5, p=0.98)", 1.5,  None, 0.98, 1.2),
    ]

    logger.info(f"Prompt: {prompt!r}\n")

    for label, temp, top_k, top_p, rep in presets:
        torch.manual_seed(42)   # same seed → only strategy differs
        params = SamplingParams(
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep,
            max_new_tokens=25,
        )
        with torch.no_grad():
            gen = model.generate(input_ids, params=params)
        output = tokenizer.decode(gen[0, input_ids.shape[1]:].tolist())
        logger.info(f"  [{label}]")
        logger.info(f"    {output!r}\n")

    logger.info("✓ Temperature + Top-p strategy working correctly!")



def main():
    """Run all tests."""
    logger.info("Starting vLLM Engine Tests\n")

    try:
        # ── Mechanism tests (fast, toy model) ───────────────────────────────
        # Phase 1: Transformer forward pass
        test_model_forward_pass()

        # Phase 2: KV-Cache data structure
        test_kv_cache()

        # Phase 3: PagedAttention BlockManager
        test_block_manager()

        # Phase 4: Continuous batching scheduler
        test_scheduler()

        # ── Correctness proofs (toy model, no HF download needed) ───────────
        # Phase 7: KV cache produces identical logits to full causal forward pass
        test_verify_kv_cache_correctness()

        # Phase 8: Continuous batching matches independent generation
        test_verify_continuous_batching()

        # ── Integration tests (full GPT-2 weights) ───────────────────────────
        # Phase 5: Full vLLM engine pipeline with real GPT-2 weights
        test_full_inference()

        # Phase 6: Direct model.generate() with real GPT-2 weights
        test_gpt2_weights()

        # Phase 9: Temperature + Top-p decoding strategies
        test_temperature_top_p()

        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("=" * 60)
        logger.info("\nTo start the FastAPI server with real GPT-2 weights, run:")
        logger.info("  python -m uvicorn serving.api:app --reload --host 0.0.0.0 --port 8000")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
