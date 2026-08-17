"""
Multi-Head Self-Attention with support for KV-cache and PagedAttention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with optional KV-cache."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections (bias=True to match GPT-2 pretrained weights)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
    
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
            kv_cache: Optional tuple of (k_cache, v_cache) from previous steps
            attention_mask: (batch_size, seq_len) or (batch_size, seq_len, seq_len)
            use_cache: Whether to compute and return KV cache
            
        Returns:
            output: (batch_size, seq_len, dim)
            kv_cache: Optional updated cache if use_cache=True
        """
        batch_size, seq_len, dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch_size, seq_len, dim)
        k = self.k_proj(x)  # (batch_size, seq_len, dim)
        v = self.v_proj(x)  # (batch_size, seq_len, dim)
        
        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use KV cache if provided (during decoding)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # Append new tokens
            v = torch.cat([v_cache, v], dim=2)
        
        # Store cache for next step
        new_kv_cache = (k, v) if use_cache else None
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:  # (batch_size, seq_len)
                attention_mask = attention_mask[:, None, None, :]  # Expand for broadcasting
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len_q, head_dim)
        
        # Reshape back to (batch_size, seq_len, dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output, new_kv_cache


class PagedAttention(nn.Module):
    """Multi-head attention using PagedAttention memory layout."""
    
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections (bias=True to match GPT-2 pretrained weights)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward_with_paged_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_table: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q: Query (batch_size, num_heads, seq_len_q, head_dim)
            k: Key (batch_size, num_heads, seq_len_k, head_dim) 
            v: Value (batch_size, num_heads, seq_len_v, head_dim)
            block_table: Mapping from logical blocks to physical memory
            attention_mask: Optional mask
            
        Returns:
            output: Attention output (batch_size, num_heads, seq_len_q, head_dim)
        """
        # Standard attention computation (block_table mapping handled at engine level)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Standard forward pass (PagedAttention logic handled at engine level)."""
        batch_size, seq_len, dim = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # KV cache handling
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        new_kv_cache = (k, v) if use_cache else None
        
        # Compute attention
        output = self.forward_with_paged_kv(q, k, v, attention_mask=attention_mask)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, dim)
        output = self.out_proj(output)
        
        return output, new_kv_cache
