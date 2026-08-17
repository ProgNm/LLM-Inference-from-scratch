"""
Continuous Batching Scheduler for efficient inference.

Manages sequence states and decides which sequences to run each step:
- WAITING: Queued, waiting to start
- RUNNING: Currently being processed
- FINISHED: Completed generation
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import time


class SequenceState(Enum):
    """State of a sequence."""
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    """Represents a generation sequence."""
    seq_id: int
    prompt: str
    state: SequenceState = SequenceState.WAITING
    input_ids: List[int] = field(default_factory=list)
    output_ids: List[int] = field(default_factory=list)
    max_new_tokens: int = 50
    current_length: int = 0  # Current number of generated tokens
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    created_time: float = field(default_factory=time.time)
    started_time: Optional[float] = None
    finished_time: Optional[float] = None


class SchedulerConfig:
    """Configuration for the scheduler."""
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_length: int = 2048,
        max_tokens_per_batch: int = 4096,
        use_preemption: bool = False,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch
        self.use_preemption = use_preemption


class ContinuousBatchingScheduler:
    """
    Scheduler for continuous batching.
    
    Unlike static batching where all sequences finish together,
    continuous batching allows:
    - New sequences to join mid-flight
    - Early stopping for finished sequences
    - Better GPU utilization
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        
        # Sequence management
        self.waiting_queue: List[Sequence] = []  # FIFO queue of waiting sequences
        self.running_sequences: Dict[int, Sequence] = {}  # seq_id -> sequence
        self.finished_sequences: Dict[int, Sequence] = {}  # seq_id -> sequence
        
        self.next_seq_id = 0
    
    def add_sequence(
        self,
        prompt: str,
        input_ids: List[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> int:
        """
        Add a new sequence to the scheduler.
        
        Args:
            prompt: Text prompt
            input_ids: Token IDs for the prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            seq_id: Unique sequence identifier
        """
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            state=SequenceState.WAITING,
            input_ids=input_ids.copy(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            current_length=0,
        )
        
        self.waiting_queue.append(seq)
        return seq_id
    
    def schedule_step(self) -> tuple:
        """
        Decide which sequences to run in this step.
        
        Returns:
            (sequences_to_run, state_changes)
            - sequences_to_run: List of Sequence objects to run
            - state_changes: Dict of seq_id -> new_state for status updates
        """
        to_run = []
        state_changes = {}
        
        # Transition waiting sequences to running
        while (
            len(self.waiting_queue) > 0
            and len(self.running_sequences) + len(to_run) < self.config.max_batch_size
        ):
            seq = self.waiting_queue.pop(0)
            seq.state = SequenceState.RUNNING
            seq.started_time = time.time()
            self.running_sequences[seq.seq_id] = seq
            to_run.append(seq)
            state_changes[seq.seq_id] = SequenceState.RUNNING
        
        # Add existing running sequences
        for seq_id, seq in list(self.running_sequences.items()):
            to_run.append(seq)
        
        return to_run, state_changes
    
    def mark_sequence_finished(self, seq_id: int, output_ids: List[int]) -> bool:
        """
        Mark a sequence as finished.
        
        Args:
            seq_id: Sequence ID
            output_ids: Generated output token IDs
            
        Returns:
            Success or failure
        """
        if seq_id not in self.running_sequences:
            return False
        
        seq = self.running_sequences.pop(seq_id)
        seq.state = SequenceState.FINISHED
        seq.output_ids = output_ids
        seq.finished_time = time.time()
        self.finished_sequences[seq_id] = seq
        
        return True
    
    def mark_sequence_token_generated(self, seq_id: int) -> bool:
        """
        Update sequence after one token is generated.
        
        Args:
            seq_id: Sequence ID
            
        Returns:
            Whether sequence should continue
        """
        if seq_id not in self.running_sequences:
            return False
        
        seq = self.running_sequences[seq_id]
        seq.current_length += 1
        
        # Check if finished
        if seq.current_length >= seq.max_new_tokens:
            return False  # Should terminate
        
        return True
    
    def get_running_sequences(self) -> List[Sequence]:
        """Get all currently running sequences."""
        return list(self.running_sequences.values())
    
    def get_waiting_sequences(self) -> List[Sequence]:
        """Get all waiting sequences."""
        return self.waiting_queue.copy()
    
    def get_finished_sequence(self, seq_id: int) -> Optional[Sequence]:
        """Get a finished sequence by ID."""
        return self.finished_sequences.get(seq_id)
    
    def get_sequence_status(self, seq_id: int) -> Optional[Dict]:
        """Get detailed status of a sequence."""
        # Check all states
        if seq_id in self.running_sequences:
            seq = self.running_sequences[seq_id]
            return {
                "seq_id": seq_id,
                "state": seq.state.value,
                "current_length": seq.current_length,
                "max_new_tokens": seq.max_new_tokens,
                "progress": seq.current_length / seq.max_new_tokens,
            }
        elif seq_id in self.finished_sequences:
            seq = self.finished_sequences[seq_id]
            elapsed = (seq.finished_time - seq.started_time) if seq.started_time else 0
            return {
                "seq_id": seq_id,
                "state": seq.state.value,
                "tokens_generated": seq.current_length,
                "max_new_tokens": seq.max_new_tokens,
                "generation_time": elapsed,
                "tokens_per_second": seq.current_length / elapsed if elapsed > 0 else 0,
            }
        else:
            for seq in self.waiting_queue:
                if seq.seq_id == seq_id:
                    return {
                        "seq_id": seq_id,
                        "state": seq.state.value,
                        "queue_position": self.waiting_queue.index(seq),
                    }
        return None
    
    def clear_finished_sequences(self) -> int:
        """
        Clear finished sequences from memory.
        
        Returns:
            Number of sequences cleared
        """
        count = len(self.finished_sequences)
        self.finished_sequences.clear()
        return count
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_sequences),
            "finished": len(self.finished_sequences),
            "total_sequences_processed": self.next_seq_id,
        }
