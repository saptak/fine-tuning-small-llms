#!/usr/bin/env python3
"""
FastAPI Application for LLM Model Serving
From: Fine-Tuning Small LLMs with Docker Desktop - Part 5
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
import os
import httpx
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import redis.asyncio as redis

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_REQUESTS = Gauge('http_active_requests', 'Active HTTP requests')
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time')
MODEL_QUEUE_LENGTH = Gauge('model_inference_queue_length', 'Model inference queue length')

# Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost')
OLLAMA_PORT = int(os.getenv('OLLAMA_PORT', '11434'))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MODEL_NAME = os.getenv('MODEL_NAME', 'sql-expert')
API_TOKEN = os.getenv('API_TOKEN', 'demo-token-12345')

# Global variables
redis_client: Optional[redis.Redis] = None
ollama_client: Optional[httpx.AsyncClient] = None

# Pydantic models
class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Whether to stream the response")

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Request model for chat completion"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    stream: bool = Field(False, description="Whether to stream the response")

class GenerateResponse(BaseModel):
    """Response model for generation"""
    text: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    response_time_ms: float = Field(..., description="Response time in milliseconds")

class ChatResponse(BaseModel):
    """Response model for chat completion"""
    message: ChatMessage = Field(..., description="Generated message")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    response_time_ms: float = Field(..., description="Response time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model loading status")

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    size: str = Field(..., description="Model size")
    format: str = Field(..., description="Model format")
    family: str = Field(..., description="Model family")
    loaded: bool = Field(..., description="Whether model is loaded")

# Authentication
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("Starting LLM API service...")
    
    global redis_client, ollama_client
    
    # Initialize Redis client
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.warning(f"⚠️  Redis connection failed: {e}")
        redis_client = None
    
    # Initialize Ollama client
    try:
        ollama_client = httpx.AsyncClient(
            base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}",
            timeout=httpx.Timeout(60.0)
        )
        
        # Test Ollama connection
        response = await ollama_client.get("/api/tags")
        if response.status_code == 200:
            logger.info("✅ Ollama connection established")
        else:
            logger.warning(f"⚠️  Ollama connection test failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️  Ollama connection failed: {e}")
        ollama_client = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM API service...")
    
    if redis_client:
        await redis_client.close()
    
    if ollama_client:
        await ollama_client.aclose()

# FastAPI app
app = FastAPI(
    title="LLM Fine-Tuning API",
    description="Production-ready API for fine-tuned language model inference",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to collect request metrics"""
    
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        logger.error(f"Request failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        ACTIVE_REQUESTS.dec()
        
        # Record metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()
        
        REQUEST_DURATION.observe(duration)
    
    return response

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Fine-Tuning API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    model_status = "unknown"
    
    if ollama_client:
        try:
            response = await ollama_client.get("/api/tags")
            if response.status_code == 200:
                tags = response.json()
                model_loaded = any(model['name'].startswith(MODEL_NAME) for model in tags.get('models', []))
                model_status = "loaded" if model_loaded else "not_loaded"
            else:
                model_status = "error"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            model_status = "error"
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        model_status=model_status
    )

@app.get("/api/v1/models", response_model=List[ModelInfo])
async def list_models(token: str = Depends(verify_token)):
    """List available models"""
    
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    try:
        response = await ollama_client.get("/api/tags")
        if response.status_code != 200:
            raise HTTPException(status_code=503, detail="Failed to fetch models from Ollama")
        
        data = response.json()
        models = []
        
        for model in data.get('models', []):
            models.append(ModelInfo(
                name=model['name'],
                size=model.get('size', 'unknown'),
                format=model.get('details', {}).get('format', 'unknown'),
                family=model.get('details', {}).get('family', 'unknown'),
                loaded=True
            ))
        
        return models
        
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise HTTPException(status_code=503, detail="Ollama service unavailable")

@app.post("/api/v1/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Generate text completion"""
    
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    start_time = time.time()
    MODEL_QUEUE_LENGTH.inc()
    
    try:
        # Check cache if Redis is available
        cache_key = None
        if redis_client and not request.stream:
            cache_key = f"generate:{hash(request.prompt)}:{request.max_tokens}:{request.temperature}"
            cached_response = await redis_client.get(cache_key)
            if cached_response:
                logger.info("Cache hit for generation request")
                return GenerateResponse.parse_raw(cached_response)
        
        # Prepare Ollama request
        ollama_request = {
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
        
        if request.stop:
            ollama_request["options"]["stop"] = request.stop
        
        # Make request to Ollama
        response = await ollama_client.post("/api/generate", json=ollama_request)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama generation failed: {response.status_code}"
            )
        
        data = response.json()
        generated_text = data.get("response", "")
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Estimate token usage (approximate)
        input_tokens = len(request.prompt.split())
        output_tokens = len(generated_text.split())
        
        result = GenerateResponse(
            text=generated_text,
            model=MODEL_NAME,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            },
            response_time_ms=response_time_ms
        )
        
        # Cache result if Redis is available
        if redis_client and cache_key and not request.stream:
            background_tasks.add_task(
                cache_response, cache_key, result.json(), expire_seconds=3600
            )
        
        # Record metrics
        MODEL_INFERENCE_TIME.observe(response_time_ms / 1000)
        
        return result
        
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    
    finally:
        MODEL_QUEUE_LENGTH.dec()

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
):
    """Generate chat completion"""
    
    if not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama service not available")
    
    start_time = time.time()
    MODEL_QUEUE_LENGTH.inc()
    
    try:
        # Convert chat messages to prompt format
        prompt_parts = []
        for message in request.messages:
            if message.role == "system":
                prompt_parts.append(f"System: {message.content}")
            elif message.role == "user":
                prompt_parts.append(f"User: {message.content}")
            elif message.role == "assistant":
                prompt_parts.append(f"Assistant: {message.content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        # Prepare Ollama request
        ollama_request = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": request.stream,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
                "stop": ["User:", "System:"]
            }
        }
        
        # Make request to Ollama
        response = await ollama_client.post("/api/generate", json=ollama_request)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama generation failed: {response.status_code}"
            )
        
        data = response.json()
        generated_text = data.get("response", "").strip()
        
        # Calculate metrics
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # Estimate token usage
        total_prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = len(generated_text.split())
        
        result = ChatResponse(
            message=ChatMessage(role="assistant", content=generated_text),
            model=MODEL_NAME,
            usage={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_prompt_tokens + completion_tokens
            },
            response_time_ms=response_time_ms
        )
        
        # Record metrics
        MODEL_INFERENCE_TIME.observe(response_time_ms / 1000)
        
        return result
        
    except httpx.RequestError as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise HTTPException(status_code=503, detail="Ollama service unavailable")
    
    finally:
        MODEL_QUEUE_LENGTH.dec()

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/api/v1/status")
async def get_status(token: str = Depends(verify_token)):
    """Get service status and statistics"""
    
    status_info = {
        "service": "llm-api",
        "status": "running",
        "timestamp": time.time(),
        "model": MODEL_NAME,
        "ollama_available": ollama_client is not None,
        "redis_available": redis_client is not None
    }
    
    # Add model information if available
    if ollama_client:
        try:
            response = await ollama_client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                status_info["available_models"] = [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
    
    return status_info

# Background tasks
async def cache_response(key: str, value: str, expire_seconds: int = 3600):
    """Cache response in Redis"""
    if redis_client:
        try:
            await redis_client.setex(key, expire_seconds, value)
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info",
        reload=os.getenv("ENVIRONMENT", "development") == "development"
    )
