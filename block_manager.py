"""
Block Manager for PagedAttention memory management.

Inspired by OS virtual memory paging:
- Pages map to KV blocks
- Block table maps logical blocks to physical memory blocks
- Allocates fixed-size blocks to eliminate fragmentation
"""
import torch
from typing import Dict, List, Optional, Set
from dataclasses import dataclass


BLOCK_SIZE = 16  # Tokens per KV block


@dataclass
class BlockTableEntry:
    """Entry in the block table for a sequence."""
    seq_id: int
    block_ids: List[int]  # Logical blocks -> physical block IDs
    num_filled_tokens: List[int]  # Tokens filled in each block


class BlockManager:
    """
    Manages KV cache blocks for multiple sequences.
    
    Similar to OS page tables:
    - Physical memory: num_blocks * BLOCK_SIZE token capacity
    - Logical blocks: tokens are mapped to physical blocks
    - Block table: maps sequence -> physical block IDs
    """
    
    def __init__(
        self,
        num_blocks: int,
        block_size: int = BLOCK_SIZE,
        num_heads: int = 8,
        head_dim: int = 16,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            num_blocks: Number of physical blocks to allocate
            block_size: Tokens per block
            num_heads: Number of attention heads
            head_dim: Dimension per head
            device: GPU/CPU device
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Track free and allocated blocks
        self.free_blocks: Set[int] = set(range(num_blocks))
        self.allocated_blocks: Dict[int, int] = {}  # block_id -> seq_id
        
        # Block table: maps sequence_id -> list of physical block IDs
        self.block_table: Dict[int, List[int]] = {}
        
        # Track filled tokens in each block
        self.block_filled_tokens: Dict[int, int] = {}  # block_id -> filled_tokens
        
        # Physical KV cache storage
        self.k_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=torch.float32
        )
        self.v_cache = torch.zeros(
            num_blocks, block_size, num_heads, head_dim,
            device=device, dtype=torch.float32
        )
    
    def allocate_sequence(self, seq_id: int, num_tokens: int) -> bool:
        """
        Allocate blocks for a new sequence.
        
        Args:
            seq_id: Sequence ID
            num_tokens: Initial number of tokens in the sequence
            
        Returns:
            Success or failure
        """
        if seq_id in self.block_table:
            return False  # Already allocated
        
        # Calculate blocks needed
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        
        # Check if enough free blocks
        if len(self.free_blocks) < num_blocks_needed:
            return False
        
        # Allocate blocks
        blocks = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            blocks.append(block_id)
            self.allocated_blocks[block_id] = seq_id
            self.block_filled_tokens[block_id] = 0
        
        self.block_table[seq_id] = blocks
        return True
    
    def extend_sequence(self, seq_id: int, num_new_tokens: int) -> bool:
        """
        Add more tokens to an existing sequence (allocate new blocks if needed).
        
        Args:
            seq_id: Sequence ID
            num_new_tokens: Number of new tokens to accommodate
            
        Returns:
            Success or failure
        """
        if seq_id not in self.block_table:
            return False
        
        blocks = self.block_table[seq_id]
        
        # Get current and needed capacity
        current_capacity = len(blocks) * self.block_size
        total_tokens = current_capacity + num_new_tokens
        num_blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
        
        # Allocate additional blocks
        additional_blocks_needed = num_blocks_needed - len(blocks)
        if additional_blocks_needed > 0:
            if len(self.free_blocks) < additional_blocks_needed:
                return False
            
            for _ in range(additional_blocks_needed):
                block_id = self.free_blocks.pop()
                blocks.append(block_id)
                self.allocated_blocks[block_id] = seq_id
                self.block_filled_tokens[block_id] = 0
        
        return True
    
    def add_tokens_to_block(
        self,
        seq_id: int,
        block_index: int,
        num_tokens: int,
    ) -> bool:
        """
        Record that tokens have been added to a block.
        
        Args:
            seq_id: Sequence ID
            block_index: Index in the block list for this sequence
            num_tokens: Number of tokens added
            
        Returns:
            Success or failure
        """
        if seq_id not in self.block_table:
            return False
        
        blocks = self.block_table[seq_id]
        if block_index >= len(blocks):
            return False
        
        block_id = blocks[block_index]
        current_filled = self.block_filled_tokens[block_id]
        new_filled = min(current_filled + num_tokens, self.block_size)
        
        self.block_filled_tokens[block_id] = new_filled
        return True
    
    def free_sequence(self, seq_id: int) -> bool:
        """
        Free all blocks allocated to a sequence.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Success or failure
        """
        if seq_id not in self.block_table:
            return False
        
        blocks = self.block_table[seq_id]
        for block_id in blocks:
            self.free_blocks.add(block_id)
            del self.allocated_blocks[block_id]
            del self.block_filled_tokens[block_id]
        
        del self.block_table[seq_id]
        return True
    
    def get_block_table(self, seq_id: int) -> Optional[List[int]]:
        """Get the block table (physical block IDs) for a sequence."""
        return self.block_table.get(seq_id)
    
    def write_to_block(
        self,
        seq_id: int,
        block_index: int,
        k_data: torch.Tensor,
        v_data: torch.Tensor,
        token_offset: int = 0,
    ) -> bool:
        """
        Write K-V data to a specific block.
        
        Args:
            seq_id: Sequence ID
            block_index: Index in block table for this sequence
            k_data: Key data (num_heads, seq_len, head_dim)
            v_data: Value data (num_heads, seq_len, head_dim)
            token_offset: Write offset within the block (in tokens)
            
        Returns:
            Success or failure
        """
        if seq_id not in self.block_table:
            return False
        
        blocks = self.block_table[seq_id]
        if block_index >= len(blocks):
            return False
        
        block_id = blocks[block_index]
        seq_len = k_data.shape[1]
        
        # Check bounds
        if token_offset + seq_len > self.block_size:
            return False
        
        # Write to physical cache
        self.k_cache[block_id, token_offset:token_offset+seq_len] = k_data.transpose(0, 1)
        self.v_cache[block_id, token_offset:token_offset+seq_len] = v_data.transpose(0, 1)
        
        return True
    
    def read_from_block(
        self,
        seq_id: int,
        block_index: int,
        start_token: int = 0,
        end_token: Optional[int] = None,
    ) -> Optional[tuple]:
        """
        Read K-V data from a specific block.
        
        Returns:
            (k_data, v_data) or None if not found
        """
        if seq_id not in self.block_table:
            return None
        
        blocks = self.block_table[seq_id]
        if block_index >= len(blocks):
            return None
        
        block_id = blocks[block_index]
        if end_token is None:
            end_token = self.block_filled_tokens[block_id]
        
        k_data = self.k_cache[block_id, start_token:end_token].transpose(0, 1)
        v_data = self.v_cache[block_id, start_token:end_token].transpose(0, 1)
        
        return k_data, v_data
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        total_blocks = self.num_blocks
        allocated_blocks = len(self.allocated_blocks)
        free_blocks = len(self.free_blocks)
        
        total_tokens_capacity = total_blocks * self.block_size
        filled_tokens = sum(self.block_filled_tokens.values())
        
        return {
            "total_blocks": total_blocks,
            "allocated_blocks": allocated_blocks,
            "free_blocks": free_blocks,
            "total_tokens_capacity": total_tokens_capacity,
            "filled_tokens": filled_tokens,
            "utilization": filled_tokens / total_tokens_capacity,
        }
    
    def get_num_free_blocks(self) -> int:
        """Get number of free blocks."""
        return len(self.free_blocks)
    
    def get_num_allocated_sequences(self) -> int:
        """Get number of allocated sequences."""
        return len(self.block_table)
