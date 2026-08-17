"""
Key-Value Cache implementation for efficient sequence generation.
"""
import torch
from typing import Tuple, Optional


class KVCache:
    """Simple KV-cache for storing keys and values during autoregressive generation."""
    
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            batch_size: Batch size
            device: Device to allocate cache on
        """
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.device = device
        self.cur_len = 0
        
        # Allocate cache tensors: (batch_size, num_heads, max_seq_len, head_dim)
        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float32
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float32
        )
    
    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """
        Update cache with new key-value pairs.
        
        Args:
            new_k: New keys (batch_size, num_heads, seq_len, head_dim)
            new_v: New values (batch_size, num_heads, seq_len, head_dim)
        """
        seq_len = new_k.shape[2]
        
        # Write to cache at current position
        self.k_cache[:, :, self.cur_len : self.cur_len + seq_len] = new_k
        self.v_cache[:, :, self.cur_len : self.cur_len + seq_len] = new_v
        
        self.cur_len += seq_len
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the accumulated cache up to current position.
        
        Returns:
            k_cache: (batch_size, num_heads, cur_len, head_dim)
            v_cache: (batch_size, num_heads, cur_len, head_dim)
        """
        return (
            self.k_cache[:, :, :self.cur_len],
            self.v_cache[:, :, :self.cur_len],
        )
    
    def reset(self) -> None:
        """Reset cache for a new sequence."""
        self.cur_len = 0
    
    def get_seq_length(self) -> int:
        """Get current sequence length in cache."""
        return self.cur_len


class MultiSeqKVCache:
    """Manages KV-caches for multiple sequences (batches)."""
    
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        max_batch_size: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.device = device
        
        # Shared cache for all sequences
        self.k_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float32
        )
        self.v_cache = torch.zeros(
            max_batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=torch.float32
        )
        
        # Track current length per sequence
        self.seq_lengths = {}  # seq_id -> int
    
    def allocate(self, seq_id: int) -> None:
        """Allocate cache slot for a new sequence."""
        self.seq_lengths[seq_id] = 0
    
    def free(self, seq_id: int) -> None:
        """Free cache slot for a finished sequence."""
        if seq_id in self.seq_lengths:
            del self.seq_lengths[seq_id]
    
    def update(
        self,
        seq_id: int,
        batch_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> None:
        """
        Update cache for a specific sequence.
        
        Args:
            seq_id: Sequence identifier
            batch_idx: Index in batch
            new_k: New keys (1, num_heads, seq_len, head_dim)
            new_v: New values (1, num_heads, seq_len, head_dim)
        """
        cur_len = self.seq_lengths[seq_id]
        seq_len = new_k.shape[2]
        
        self.k_cache[batch_idx, :, cur_len:cur_len + seq_len] = new_k.squeeze(0)
        self.v_cache[batch_idx, :, cur_len:cur_len + seq_len] = new_v.squeeze(0)
        
        self.seq_lengths[seq_id] = cur_len + seq_len
    
    def get(self, seq_id: int, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cache for a specific sequence."""
        cur_len = self.seq_lengths[seq_id]
        return (
            self.k_cache[batch_idx : batch_idx + 1, :, :cur_len],
            self.v_cache[batch_idx : batch_idx + 1, :, :cur_len],
        )
    
    def reset(self, seq_id: int) -> None:
        """Reset cache for a sequence."""
        if seq_id in self.seq_lengths:
            self.seq_lengths[seq_id] = 0
