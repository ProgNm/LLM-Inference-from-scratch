"""
sampling.py — Composable decoding strategies for autoregressive generation.

Decoding pipeline (applied in this order):
    1. Repetition penalty  — down-weight already-generated tokens
    2. Temperature scaling  — T<1 sharpens, T>1 flattens the distribution
    3. Top-k filtering      — keep only the k most probable tokens
    4. Top-p / Nucleus      — keep the smallest set whose cumulative prob ≥ p
    5. Multinomial sample   — draw from the filtered distribution

Why this order?
    Temperature must come BEFORE top-p so the nucleus is built on the
    re-weighted distribution.  For example:
        T=0.5 → sharp peaks → small nucleus (focused)
        T=1.5 → flat peaks  → large nucleus (diverse)
    Applying top-p on raw logits and then rescaling defeats the purpose.

References:
    Top-p: "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2020)
    Repetition penalty: "CTRL: A Conditional Transformer Language Model for
                         Controllable Generation" (Keskar et al., 2019)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


# ─── SamplingParams ───────────────────────────────────────────────────────────

@dataclass
class SamplingParams:
    """
    All knobs that control how the next token is sampled.

    Recommended presets
    -------------------
    Greedy (deterministic):  top_k=1,  temperature=1.0
    Conservative:            top_p=0.9,  temperature=0.7
    Balanced (default):      top_p=0.9,  temperature=0.8
    Creative:                top_p=0.95, temperature=1.1
    Very creative:           top_p=0.98, temperature=1.3, repetition_penalty=1.2
    """
    temperature: float = 0.8
    top_k: Optional[int] = None
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 50

    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be ≥ 1.0, got {self.repetition_penalty}")


# ─── Individual filters ────────────────────────────────────────────────────────

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Divide logits by temperature.

    Temperature effect on the distribution:
        T → 0:   approaches greedy (argmax); distribution collapses to one token
        T = 1:   distribution unchanged
        T → ∞:   approaches uniform; all tokens equally likely

    Args:
        logits: 1D (vocab_size,) tensor — raw model output for the next token
        temperature: positive float

    Returns:
        Temperature-scaled logits (same shape, in-place-safe copy)
    """
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Keep only the top-k most probable tokens; mask the rest to -inf.

    Args:
        logits: 1D (vocab_size,) tensor
        top_k:  number of tokens to keep (≥ 1)

    Returns:
        Filtered logits (same shape)
    """
    if top_k <= 0:
        return logits
    top_k = min(top_k, logits.shape[-1])
    # Threshold = value of the k-th largest logit
    kth_value = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < kth_value, float('-inf'))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (Top-p) sampling — Holtzman et al., 2020.

    Algorithm:
        1. Sort tokens by probability (descending).
        2. Compute cumulative probability of sorted tokens.
        3. Discard all tokens whose cumulative probability BEFORE adding
           them already exceeds top_p.  This guarantees:
               • At least one token always survives.
               • The nucleus sum is always ≥ top_p.
        4. Scatter the removal mask back to the original token ordering.

    Example (top_p = 0.75):
        sorted probs  = [0.40, 0.30, 0.20, 0.10]
        cumsum        = [0.40, 0.70, 0.90, 1.00]
        shifted cumsum= [0.00, 0.40, 0.70, 0.90]   ← compare vs top_p
        remove?       = [ No,   No,   No,  Yes ]   ← keep tokens 0-2 (sum=0.90)

    IMPORTANT: Apply temperature BEFORE this function so that the nucleus
    is built on the correctly re-weighted distribution.

    Args:
        logits: 1D (vocab_size,) tensor — already temperature-scaled
        top_p:  float in (0, 1]; 1.0 → no filtering

    Returns:
        Filtered logits (same shape)
    """
    if top_p >= 1.0:
        return logits

    # Sort descending
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # Probabilities and their cumulative sum in sorted order
    sorted_probs   = F.softmax(sorted_logits, dim=-1)
    cumulative     = torch.cumsum(sorted_probs, dim=-1)

    # "Shifted" cumsum: cumulative probability BEFORE adding each token.
    # Token i is removed if already more than top_p is covered without it.
    shifted        = cumulative - sorted_probs
    sorted_remove  = shifted > top_p                    # bool mask (sorted order)

    # Scatter back to the original (unsorted) vocabulary ordering
    remove = torch.zeros_like(logits, dtype=torch.bool)
    remove.scatter_(0, sorted_indices, sorted_remove)

    return logits.masked_fill(remove, float('-inf'))


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    """
    Penalise tokens that have already been generated.

    For a token t already in generated_ids:
        logit[t] > 0  →  logit[t] / penalty   (reduces positive score)
        logit[t] < 0  →  logit[t] * penalty   (makes negative score more negative)

    Args:
        logits:        1D (vocab_size,) tensor
        generated_ids: list of integer token IDs generated so far
        penalty:       float ≥ 1.0; 1.0 means no penalty

    Returns:
        Penalised logits (same shape, modified in a copy)
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    unique_ids = list(set(generated_ids))
    scores = logits[unique_ids]
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits[unique_ids] = scores
    return logits


# ─── Nucleus-size diagnostic ───────────────────────────────────────────────────

def nucleus_size(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """
    Return the number of tokens that would be inside the nucleus after
    applying temperature + top-p.  Useful for monitoring/debugging.

    Args:
        logits:      1D (vocab_size,) raw logits
        temperature: temperature scaling factor
        top_p:       nucleus threshold

    Returns:
        Number of tokens in the nucleus (int)
    """
    scaled = apply_temperature(logits.clone(), temperature)
    filtered = apply_top_p(scaled, top_p)
    return int((filtered > float('-inf')).sum().item())


# ─── Combined sampling pipeline ───────────────────────────────────────────────

def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[List[int]] = None,
) -> int:
    """
    Full decoding pipeline: penalty → temperature → top-k → top-p → sample.

    This is the single entry point for all sampling strategies.

    Args:
        logits:             1D (vocab_size,) raw next-token logits
        temperature:        scaling factor (default 1.0 = no change)
        top_k:              if set, keep only top-k tokens (applied after temperature)
        top_p:              nucleus threshold; None or 1.0 = no filtering
        repetition_penalty: penalise already-generated tokens (≥ 1.0)
        generated_ids:      list of token IDs generated so far

    Returns:
        Sampled next token ID (int)

    Usage examples:
        # Greedy
        token = sample_token(logits, top_k=1)

        # Balanced (temperature + nucleus)
        token = sample_token(logits, temperature=0.8, top_p=0.9)

        # Creative with repetition suppression
        token = sample_token(logits, temperature=1.1, top_p=0.95,
                             repetition_penalty=1.2, generated_ids=past_tokens)
    """
    assert logits.dim() == 1, (
        f"sample_token expects 1D logits (vocab_size,), got shape {tuple(logits.shape)}"
    )

    # 1. Repetition penalty (before temperature to keep scales consistent)
    if repetition_penalty != 1.0 and generated_ids:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # 2. Temperature
    logits = apply_temperature(logits, temperature)

    # 3. Top-k
    if top_k is not None and top_k > 0:
        logits = apply_top_k(logits, top_k)

    # 4. Top-p (nucleus)
    if top_p is not None and top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    # 5. Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_token_from_params(
    logits: torch.Tensor,
    params: SamplingParams,
    generated_ids: Optional[List[int]] = None,
) -> int:
    """
    Convenience wrapper: apply a SamplingParams object in one call.

    Args:
        logits:        1D (vocab_size,) raw logits
        params:        SamplingParams instance
        generated_ids: list of already-generated token IDs (for rep. penalty)

    Returns:
        Sampled next token ID (int)
    """
    return sample_token(
        logits,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
        generated_ids=generated_ids,
    )
