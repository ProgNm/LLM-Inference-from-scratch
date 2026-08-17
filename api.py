"""
FastAPI server for vLLM inference.

Provides REST API endpoints for:
- Adding generation requests
- Polling for results
- Getting system status
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import logging
from functools import lru_cache

from engine.inference import SimpleLLMEngine, InferenceConfig
from model.load_weights import load_gpt2_weights

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class GenerateRequest(BaseModel):
    """Request to generate text."""
    prompt: str
    max_new_tokens: Optional[int] = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class GenerateResponse(BaseModel):
    """Response from generation endpoint."""
    request_id: str
    status: str  # "accepted", "running", "completed", "error"
    message: Optional[str] = None


class ResultResponse(BaseModel):
    """Response with generation results."""
    request_id: str
    prompt: str
    generated_text: str
    tokens_generated: int
    generation_time: Optional[float] = None
    tokens_per_second: Optional[float] = None


class StatusResponse(BaseModel):
    """System status response."""
    status: str
    active_requests: int
    waiting_requests: int
    completed_requests: int
    memory_usage: Dict[str, Any]


# Global engine instance
_engine: Optional[SimpleLLMEngine] = None
_tokenizer: Optional[Any] = None


def init_engine(config: Optional[InferenceConfig] = None):
    """Initialize the inference engine with GPT-2 Small architecture."""
    global _engine
    
    if config is None:
        # Defaults match GPT-2 Small (117M parameters)
        config = InferenceConfig(
            vocab_size=50257,
            model_dim=768,
            num_heads=12,
            num_layers=12,
            max_seq_length=1024,
            num_blocks=256,
            device="cpu",
        )
    
    _engine = SimpleLLMEngine(config)
    logger.info("Engine initialized")


def init_tokenizer():
    """Initialize tokenizer from HuggingFace."""
    global _tokenizer
    
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("Tokenizer initialized (GPT-2)")
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {e}")


@lru_cache(maxsize=1)
def get_engine() -> SimpleLLMEngine:
    """Get or initialize the engine."""
    if _engine is None:
        init_engine()
    return _engine


@lru_cache(maxsize=1)
def get_tokenizer():
    """Get or initialize the tokenizer."""
    if _tokenizer is None:
        init_tokenizer()
    return _tokenizer


# Create FastAPI app
app = FastAPI(
    title="vLLM Engine",
    description="Simple vLLM implementation with PagedAttention and continuous batching",
    version="0.1.0",
)


# ===== Endpoints =====

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Submit a generation request.
    
    Creates a request in the scheduler and returns immediately.
    Use /status and /result endpoints to check progress.
    """
    try:
        engine = get_engine()
        tokenizer = get_tokenizer()
        
        # Add request to engine
        seq_id = engine.add_request(
            prompt=request.prompt,
            tokenizer=tokenizer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
        
        return GenerateResponse(
            request_id=str(seq_id),
            status="accepted",
            message=f"Request {seq_id} queued for processing",
        )
    
    except Exception as e:
        logger.error(f"Error in /generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/{request_id}/status")
async def get_status(request_id: str) -> Dict[str, Any]:
    """Get the status of a generation request."""
    try:
        seq_id = int(request_id)
        engine = get_engine()
        
        status = engine.get_status(seq_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Request {request_id} not found")
        
        return status
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id")
    except Exception as e:
        logger.error(f"Error in /generate/{request_id}/status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/{request_id}/result", response_model=ResultResponse)
async def get_result(request_id: str):
    """Get the result of a completed generation request."""
    try:
        seq_id = int(request_id)
        engine = get_engine()
        tokenizer = get_tokenizer()
        
        # Check if finished
        finished_seq = engine.scheduler.get_finished_sequence(seq_id)
        if finished_seq is None:
            raise HTTPException(status_code=202, detail=f"Request {request_id} not yet complete")
        
        # Decode tokens
        generated_text = tokenizer.decode(finished_seq.output_ids)
        
        # Calculate stats
        generation_time = (
            (finished_seq.finished_time - finished_seq.started_time)
            if finished_seq.started_time else 0
        )
        tokens_per_second = (
            finished_seq.current_length / generation_time
            if generation_time > 0 else 0
        )
        
        return ResultResponse(
            request_id=request_id,
            prompt=finished_seq.prompt,
            generated_text=generated_text,
            tokens_generated=finished_seq.current_length,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
        )
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /generate/{request_id}/result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def system_status() -> StatusResponse:
    """Get system status and statistics."""
    try:
        engine = get_engine()
        stats = engine.get_stats()
        scheduler_stats = stats["scheduler"]
        
        return StatusResponse(
            status="running",
            active_requests=scheduler_stats["running"],
            waiting_requests=scheduler_stats["waiting"],
            completed_requests=scheduler_stats["finished"],
            memory_usage=stats["block_manager"],
        )
    
    except Exception as e:
        logger.error(f"Error in /status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run_inference")
async def run_inference():
    """
    Run inference for all queued requests.
    
    This endpoint should be called periodically or via a background task
    to actually process the requests in the queue.
    """
    try:
        engine = get_engine()
        
        # Run inference
        outputs = engine.generate_all(max_steps=1000)
        
        return {
            "status": "completed",
            "results_count": len(outputs),
            "stats": engine.get_stats(),
        }
    
    except Exception as e:
        logger.error(f"Error in /run_inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.on_event("startup")
async def startup_event():
    """Initialize engine and load pretrained GPT-2 weights on startup."""
    logger.info("Starting vLLM Server")
    init_engine()
    init_tokenizer()
    # Load real GPT-2 pretrained weights into the model
    logger.info("Loading GPT-2 pretrained weights — this may take a moment...")
    load_gpt2_weights(_engine.model, model_name="gpt2")
    logger.info("✓ Server ready with real GPT-2 weights")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down vLLM Server")


if __name__ == "__main__":
    import uvicorn
    
    # Initialize
    init_engine()
    init_tokenizer()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
