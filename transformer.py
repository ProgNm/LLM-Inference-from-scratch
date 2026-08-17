"""
Minimalist GPT-style Transformer model.
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from model.sampling import sample_token, SamplingParams
from model.attention import MultiHeadAttention, PagedAttention
from model.kv_cache import KVCache, MultiSeqKVCache


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embeddings (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Pre-compute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length
            device: Device
            
        Returns:
            Rotation matrix for position embeddings
        """
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        return emb


class FeedForwardNetwork(nn.Module):
    """Simple feed-forward network."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        
        self.fc1 = nn.Linear(dim, hidden_dim, bias=True)  # bias=True to match GPT-2
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)  # bias=True to match GPT-2
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with attention, FFN, and layer norms."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_paged_attention: bool = False,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = (
            PagedAttention(dim, num_heads, dropout)
            if use_paged_attention
            else MultiHeadAttention(dim, num_heads, dropout)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, ffn_hidden_dim, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: (batch_size, seq_len, dim)
            kv_cache: Optional KV cache from previous steps
            attention_mask: Optional attention mask
            use_cache: Whether to return KV cache
            
        Returns:
            output: (batch_size, seq_len, dim)
            new_kv_cache: Updated KV cache if use_cache=True
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, new_kv_cache = self.attn(
            normed,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        x = x + attn_out
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x, new_kv_cache


class GPTModel(nn.Module):
    """Minimal GPT-style transformer model."""
    
    def __init__(
        self,
        vocab_size: int = 50257,       # GPT-2 vocabulary size
        max_seq_len: int = 1024,        # GPT-2 context length
        dim: int = 768,                 # GPT-2 Small hidden dimension
        num_heads: int = 12,            # GPT-2 Small attention heads
        num_layers: int = 12,           # GPT-2 Small transformer layers
        ffn_hidden_dim: Optional[int] = None,  # defaults to 4*dim = 3072
        dropout: float = 0.0,
        use_paged_attention: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                use_paged_attention=use_paged_attention,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_length: int = 0,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            kv_caches: Optional list of KV caches for each layer
            attention_mask: (batch_size, seq_len)
            use_cache: Whether to return KV caches
            past_length: Number of tokens already in the KV cache (for position offset).
                         During the prompt phase this is 0; during decoding it equals
                         the number of tokens already generated so positions are correct.
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            new_kv_caches: Updated KV caches if use_cache=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Position embeddings: offset by past_length so decoding tokens get
        # the correct absolute position index, not position 0 every step.
        positions = torch.arange(past_length, past_length + seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)
        
        # Pass through transformer blocks
        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv_cache = block(
                x,
                kv_cache=kv_cache,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            if use_cache:
                new_kv_caches.append(new_kv_cache)
        
        # Final layer norm and language model head
        x = self.norm(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        return logits, new_kv_caches if use_cache else None
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        params: Optional[SamplingParams] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation using KV cache + the sampling pipeline
        from model/sampling.py (temperature → top-k → top-p → multinomial).

        Args:
            input_ids:          (batch_size, prompt_len) — prompt token IDs
            max_new_tokens:     maximum tokens to generate
            temperature:        sharpness of sampling distribution (default 0.8)
            top_k:              if set, keep only top-k tokens
            top_p:              nucleus threshold (default 0.9)
            repetition_penalty: penalise repeated tokens (≥ 1.0)
            params:             if provided, overrides individual sampling args

        Returns:
            Generated token IDs: (batch_size, prompt_len + max_new_tokens)
        """
        # Resolve sampling params
        if params is not None:
            temperature        = params.temperature
            top_k              = params.top_k
            top_p              = params.top_p
            repetition_penalty = params.repetition_penalty
            max_new_tokens     = params.max_new_tokens

        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        device     = input_ids.device

        # Initialise KV caches
        kv_caches = [None] * self.num_layers

        # Prompt phase: process full prompt (positions 0..prompt_len-1)
        with torch.no_grad():
            logits, kv_caches = self.forward(
                input_ids, kv_caches=kv_caches, use_cache=True, past_length=0
            )

        generated = input_ids.clone()

        for step in range(max_new_tokens):
            # Logits for the last token position → next-token distribution
            next_logits = logits[0, -1, :]   # (vocab_size,)

            # All tokens generated so far (for repetition penalty)
            all_generated_ids = generated[0].tolist()

            # ── Apply full sampling pipeline ──────────────────────────────
            next_token_id = sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=all_generated_ids,
            )
            next_token = torch.tensor([[next_token_id]], device=device)

            # Append to output
            generated = torch.cat([generated, next_token], dim=1)

            # Decode phase: one token, correct absolute position
            past_length = prompt_len + step
            with torch.no_grad():
                logits, kv_caches = self.forward(
                    next_token,
                    kv_caches=kv_caches,
                    use_cache=True,
                    past_length=past_length,
                )

        return generated
