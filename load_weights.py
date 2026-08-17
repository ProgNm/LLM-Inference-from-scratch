"""
GPT-2 Pretrained Weight Loader.

Downloads GPT-2 weights from HuggingFace and maps them into our custom GPTModel.

Key challenges handled here:
  1. HuggingFace GPT-2 uses Conv1D (weight shape: [in, out]) instead of nn.Linear
     ([out, in]), so every weight tensor must be transposed before loading.
  2. GPT-2 fuses Q, K, V into a single 'c_attn' projection of size [dim, 3*dim].
     We split it into our three separate q_proj, k_proj, v_proj layers.
  3. The causal attention mask ('attn.bias', 'attn.masked_bias') stored in the
     HuggingFace state_dict are buffers, not learned parameters — we skip them.
  4. The lm_head weight is tied to token_embedding, so we also skip it (it is
     already handled by weight tying in GPTModel.__init__).
"""
import logging
from typing import Optional
import torch
from model.transformer import GPTModel

logger = logging.getLogger(__name__)

# GPT-2 variant specs  {model_name: (dim, num_heads, num_layers, max_seq_len)}
GPT2_CONFIGS = {
    "gpt2":        (768,  12, 12, 1024),   # 117M  — GPT-2 Small
    "gpt2-medium": (1024, 16, 24, 1024),   # 345M  — GPT-2 Medium
    "gpt2-large":  (1280, 20, 36, 1024),   # 774M  — GPT-2 Large
    "gpt2-xl":     (1600, 25, 48, 1024),   # 1.5B  — GPT-2 XL
}


def build_gpt2_model(model_name: str = "gpt2", use_paged_attention: bool = True) -> GPTModel:
    """
    Construct a GPTModel with the exact hyperparameters for the requested
    GPT-2 variant.  The returned model has *random* weights — call
    load_gpt2_weights() to fill in the pretrained values.

    Args:
        model_name: One of "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl".
        use_paged_attention: Whether to use PagedAttention (for serving).

    Returns:
        GPTModel instance with correct architecture.
    """
    if model_name not in GPT2_CONFIGS:
        raise ValueError(f"Unknown model_name '{model_name}'. Choose from: {list(GPT2_CONFIGS)}")

    dim, num_heads, num_layers, max_seq_len = GPT2_CONFIGS[model_name]
    logger.info(f"Building {model_name} model: dim={dim}, heads={num_heads}, layers={num_layers}")

    model = GPTModel(
        vocab_size=50257,
        max_seq_len=max_seq_len,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.0,
        use_paged_attention=use_paged_attention,
    )
    return model


def load_gpt2_weights(model: GPTModel, model_name: str = "gpt2") -> GPTModel:
    """
    Download GPT-2 pretrained weights from HuggingFace and copy them into
    our custom GPTModel.

    This function:
      - Downloads the HuggingFace GPT2LMHeadModel weights (cached after first run).
      - Transposes every Conv1D weight to match nn.Linear layout.
      - Splits the fused c_attn projection into separate q, k, v projections.
      - Strictly loads the resulting state dict into `model`.

    Args:
        model: GPTModel instance with architecture matching `model_name`.
        model_name: HuggingFace model identifier (default: "gpt2").

    Returns:
        The same model instance with pretrained weights loaded, in eval mode.
    """
    logger.info(f"Loading pretrained weights from HuggingFace: '{model_name}'...")

    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("transformers package is required. Run: pip install transformers")

    # ── Download / load from cache ──────────────────────────────────────────
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_sd = hf_model.state_dict()
    logger.info(f"HuggingFace state dict loaded ({len(hf_sd)} tensors)")

    # ── Build mapping to our state dict ────────────────────────────────────
    our_sd = {}

    # Helper: Conv1D weight (HF) → nn.Linear weight (ours)
    # HF Conv1D stores weights as [in_features, out_features]
    # nn.Linear stores weights as [out_features, in_features]
    def conv_to_linear(t: torch.Tensor) -> torch.Tensor:
        return t.t().contiguous()

    # ── Token & position embeddings ─────────────────────────────────────────
    our_sd["token_embedding.weight"] = hf_sd["transformer.wte.weight"]
    our_sd["pos_embedding.weight"]   = hf_sd["transformer.wpe.weight"]

    # ── Transformer blocks ──────────────────────────────────────────────────
    num_layers = model.num_layers
    dim        = model.dim

    for i in range(num_layers):
        hf_pre  = f"transformer.h.{i}"   # HuggingFace prefix
        our_pre = f"blocks.{i}"           # Our prefix

        # Pre-attention LayerNorm
        our_sd[f"{our_pre}.norm1.weight"] = hf_sd[f"{hf_pre}.ln_1.weight"]
        our_sd[f"{our_pre}.norm1.bias"]   = hf_sd[f"{hf_pre}.ln_1.bias"]

        # ── Attention projections ────────────────────────────────────────
        # c_attn is a fused [dim, 3*dim] Conv1D projection (Q||K||V)
        # After transposing: [3*dim, dim] — then split along dim-0
        c_attn_w = conv_to_linear(hf_sd[f"{hf_pre}.attn.c_attn.weight"])  # [3*dim, dim]
        c_attn_b = hf_sd[f"{hf_pre}.attn.c_attn.bias"]                    # [3*dim]

        our_sd[f"{our_pre}.attn.q_proj.weight"] = c_attn_w[0:dim, :]
        our_sd[f"{our_pre}.attn.q_proj.bias"]   = c_attn_b[0:dim]

        our_sd[f"{our_pre}.attn.k_proj.weight"] = c_attn_w[dim:2*dim, :]
        our_sd[f"{our_pre}.attn.k_proj.bias"]   = c_attn_b[dim:2*dim]

        our_sd[f"{our_pre}.attn.v_proj.weight"] = c_attn_w[2*dim:3*dim, :]
        our_sd[f"{our_pre}.attn.v_proj.bias"]   = c_attn_b[2*dim:3*dim]

        # Output projection (c_proj)
        our_sd[f"{our_pre}.attn.out_proj.weight"] = conv_to_linear(hf_sd[f"{hf_pre}.attn.c_proj.weight"])
        our_sd[f"{our_pre}.attn.out_proj.bias"]   = hf_sd[f"{hf_pre}.attn.c_proj.bias"]

        # Post-attention LayerNorm
        our_sd[f"{our_pre}.norm2.weight"] = hf_sd[f"{hf_pre}.ln_2.weight"]
        our_sd[f"{our_pre}.norm2.bias"]   = hf_sd[f"{hf_pre}.ln_2.bias"]

        # ── Feed-Forward Network ─────────────────────────────────────────
        # c_fc: [dim, 4*dim] Conv1D  →  fc1: [4*dim, dim] Linear
        our_sd[f"{our_pre}.ffn.fc1.weight"] = conv_to_linear(hf_sd[f"{hf_pre}.mlp.c_fc.weight"])
        our_sd[f"{our_pre}.ffn.fc1.bias"]   = hf_sd[f"{hf_pre}.mlp.c_fc.bias"]

        # c_proj: [4*dim, dim] Conv1D  →  fc2: [dim, 4*dim] Linear
        our_sd[f"{our_pre}.ffn.fc2.weight"] = conv_to_linear(hf_sd[f"{hf_pre}.mlp.c_proj.weight"])
        our_sd[f"{our_pre}.ffn.fc2.bias"]   = hf_sd[f"{hf_pre}.mlp.c_proj.bias"]

    # ── Final LayerNorm ──────────────────────────────────────────────────────
    our_sd["norm.weight"] = hf_sd["transformer.ln_f.weight"]
    our_sd["norm.bias"]   = hf_sd["transformer.ln_f.bias"]

    # ── lm_head ──────────────────────────────────────────────────────────────
    # Weight-tied to token_embedding: already set above via token_embedding.weight.
    # Copy explicitly so strict loading works.
    our_sd["lm_head.weight"] = hf_sd["transformer.wte.weight"]

    # ── Load into model (strict=True to catch any mismatches) ───────────────
    missing, unexpected = model.load_state_dict(our_sd, strict=True)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    logger.info(f"✓ GPT-2 pretrained weights loaded successfully ({model_name})")

    # Free HuggingFace model memory
    del hf_model, hf_sd

    return model
