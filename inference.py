"""
Main inference engine that orchestrates the entire vLLM system.

Combines:
- Transformer model
- KV Cache
- PagedAttention BlockManager
- Continuous Batching Scheduler
"""
import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Any
from model.transformer import GPTModel
from model.sampling import sample_token, SamplingParams
from engine.block_manager import BlockManager, BLOCK_SIZE
from engine.scheduler import ContinuousBatchingScheduler, SchedulerConfig, SequenceState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceConfig:
    """Configuration for the inference engine."""
    
    def __init__(
        self,
        # Model config — defaults match GPT-2 Small (117M)
        vocab_size: int = 50257,
        model_dim: int = 768,          # GPT-2 Small hidden dim
        num_heads: int = 12,           # GPT-2 Small attention heads
        num_layers: int = 12,          # GPT-2 Small transformer layers
        max_seq_length: int = 1024,    # GPT-2 context window
        # Memory config
        num_blocks: int = 256,         # Total KV-cache blocks for all sequences
        # Scheduling config
        max_batch_size: int = 32,
        max_tokens_per_batch: int = 4096,
        # Generation config
        default_max_new_tokens: int = 50,
        device: str = "cpu",
    ):
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.num_blocks = num_blocks
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.default_max_new_tokens = default_max_new_tokens
        self.device = device


class SimpleLLMEngine:
    """
    Complete vLLM inference engine.
    
    Features:
    - PagedAttention with BlockManager for efficient KV cache
    - Continuous batching scheduler for dynamic request handling
    - Autoregressive generation with sampling
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        logger.info(f"Initializing LLM Engine on {self.device}")
        
        # Initialize model
        self.model = GPTModel(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_length,
            dim=config.model_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            use_paged_attention=True,
        ).to(self.device)
        
        self.model.eval()
        
        # Initialize block manager for PagedAttention
        self.block_manager = BlockManager(
            num_blocks=config.num_blocks,
            block_size=BLOCK_SIZE,
            num_heads=config.num_heads,
            head_dim=config.model_dim // config.num_heads,
            device=self.device,
        )
        
        # Initialize scheduler
        scheduler_config = SchedulerConfig(
            max_batch_size=config.max_batch_size,
            max_seq_length=config.max_seq_length,
            max_tokens_per_batch=config.max_tokens_per_batch,
        )
        self.scheduler = ContinuousBatchingScheduler(scheduler_config)
        
        # KV caches per sequence
        self.seq_kv_caches: Dict[int, List] = {}  # seq_id -> list of kv_caches per layer
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Block manager: {config.num_blocks} blocks * {BLOCK_SIZE} tokens")
    
    def add_request(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> int:
        """
        Add a new generation request.
        
        Args:
            prompt: Text prompt
            tokenizer: HuggingFace tokenizer
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            seq_id: Unique sequence identifier
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.default_max_new_tokens
        
        # Tokenize
        tokens = tokenizer.encode(prompt)
        
        # Add to scheduler
        seq_id = self.scheduler.add_sequence(
            prompt=prompt,
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Allocate blocks in block manager
        prompt_length = len(tokens)
        if not self.block_manager.allocate_sequence(seq_id, prompt_length + max_new_tokens):
            logger.warning(f"Failed to allocate blocks for sequence {seq_id}")
        
        logger.info(f"Added request {seq_id}: prompt_length={prompt_length}, max_new_tokens={max_new_tokens}")
        
        return seq_id
    
    def _prepare_batch(self, sequences: List) -> tuple:
        """
        Prepare a batch of sequences for model forward pass.
        
        Returns:
            (input_ids, batch_size, seq_to_batch_idx, past_length)
            - past_length: positional offset for the new tokens being processed.
              Prompt phase = 0 (positions start at 0).
              Decode phase = len(prompt_tokens) + tokens_already_generated - 1
              (because output_ids already contains the token being re-fed as input).
        """
        batch_input_ids = []
        seq_to_batch_idx = {}
        
        for batch_idx, seq in enumerate(sequences):
            seq_to_batch_idx[seq.seq_id] = batch_idx
            
            # For prompt: use full input_ids
            # For generation step: use only last token
            if len(seq.output_ids) == 0:
                # Prompt phase
                input_ids = seq.input_ids
            else:
                # Generation phase: pass only the last generated token
                input_ids = [seq.output_ids[-1]]
            
            batch_input_ids.append(input_ids)
        
        # Compute past_length using the first sequence as representative.
        # Prompt phase: past_length = 0  (positions run 0..prompt_len-1)
        # Decode phase: the token being fed is at position
        #               len(input_ids) + len(output_ids) - 1
        first_seq = sequences[0]
        if len(first_seq.output_ids) == 0:
            past_length = 0
        else:
            past_length = len(first_seq.input_ids) + len(first_seq.output_ids) - 1
        
        # Pad and stack (pad to max length in batch)
        max_len = max(len(ids) for ids in batch_input_ids)
        padded = []
        for ids in batch_input_ids:
            padded.append(ids + [0] * (max_len - len(ids)))
        
        input_tensors = torch.tensor(padded, device=self.device, dtype=torch.long)
        
        return input_tensors, len(sequences), seq_to_batch_idx, past_length
    
    def generate_step(self) -> Dict[int, Any]:
        """
        Execute one generation step.

        Each sequence is processed independently (one forward pass each) so that:
        - Its own KV cache is correctly used and updated.
        - Its own absolute position offset (past_length) is respected.

        This sacrifices raw GPU throughput compared to fully-batched inference,
        but is necessary for correct multi-sequence generation with separate KV
        caches. A future optimisation could batch sequences that share the same
        prompt length.

        Returns:
            results: Dict of seq_id -> generated token
        """
        # Get sequences to run this step
        sequences_to_run, state_changes = self.scheduler.schedule_step()

        if not sequences_to_run:
            return {}

        logger.info(f"Generation step: running {len(sequences_to_run)} sequences")

        results = {}

        for seq in sequences_to_run:
            # ── Build input for this sequence ─────────────────────────────────
            if len(seq.output_ids) == 0:
                # Prompt phase: feed the full prompt, positions start at 0
                input_ids = torch.tensor(
                    [seq.input_ids], device=self.device, dtype=torch.long
                )
                past_length = 0
            else:
                # Decode phase: feed only the last generated token.
                # past_length = number of tokens already in this seq's KV cache
                # = len(prompt) + (generated tokens so far - 1)
                input_ids = torch.tensor(
                    [[seq.output_ids[-1]]], device=self.device, dtype=torch.long
                )
                past_length = len(seq.input_ids) + len(seq.output_ids) - 1

            # ── Initialise KV cache slot if new ──────────────────────────────
            if seq.seq_id not in self.seq_kv_caches:
                self.seq_kv_caches[seq.seq_id] = [None] * self.config.num_layers

            # ── Forward pass ──────────────────────────────────────────────────
            with torch.no_grad():
                logits, new_kv_caches = self.model(
                    input_ids,
                    kv_caches=self.seq_kv_caches[seq.seq_id],
                    use_cache=True,
                    past_length=past_length,
                )

            # Update this sequence's KV cache with the fresh keys/values
            self.seq_kv_caches[seq.seq_id] = new_kv_caches

            # ── Sample the next token ─────────────────────────────────────────
            seq_logits = logits[0, -1, :]  # (vocab_size,)

            # ── Sample the next token via full sampling pipeline ──────────────
            next_token_id = sample_token(
                seq_logits,
                temperature=seq.temperature,
                top_k=seq.top_k,
                top_p=seq.top_p,
                repetition_penalty=getattr(seq, 'repetition_penalty', 1.0),
                generated_ids=seq.input_ids + seq.output_ids,
            )

            seq.output_ids.append(next_token_id)
            results[seq.seq_id] = next_token_id

            # ── Check if this sequence is finished ────────────────────────────
            self.scheduler.mark_sequence_token_generated(seq.seq_id)
            if seq.current_length >= seq.max_new_tokens:
                self.scheduler.mark_sequence_finished(seq.seq_id, seq.output_ids)
                self.block_manager.free_sequence(seq.seq_id)
                if seq.seq_id in self.seq_kv_caches:
                    del self.seq_kv_caches[seq.seq_id]

        return results

    
    def generate_all(self, max_steps: Optional[int] = None) -> Dict[int, List[int]]:
        """
        Run inference until all sequences are finished.
        
        Args:
            max_steps: Maximum generation steps (for safety)
            
        Returns:
            Dict of seq_id -> output_ids
        """
        step = 0
        max_steps = max_steps or 1000
        
        while step < max_steps:
            results = self.generate_step()
            
            if not results:
                # All done
                break
            
            step += 1
            
            # Log progress
            if step % 10 == 0:
                stats = self.scheduler.get_stats()
                logger.info(f"Step {step}: {stats}")
        
        logger.info(f"Generation complete in {step} steps")
        
        # Collect results
        outputs = {}
        for seq_id in range(self.scheduler.next_seq_id):
            finished_seq = self.scheduler.get_finished_sequence(seq_id)
            if finished_seq:
                outputs[seq_id] = finished_seq.output_ids
        
        return outputs
    
    def get_status(self, seq_id: int) -> Optional[Dict]:
        """Get status of a sequence."""
        return self.scheduler.get_sequence_status(seq_id)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        block_stats = self.block_manager.get_memory_usage()
        scheduler_stats = self.scheduler.get_stats()
        
        return {
            "block_manager": block_stats,
            "scheduler": scheduler_stats,
        }
