"""
Agent Platform (fka Vertex AI) Custom Container Adapter for Hugging Face LLMs.
Bridges Agent Platform (fka Vertex AI) prediction request protocol and OpenAI routes with vLLM engine.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geap-adapter")

# Environment configurations
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}"
DEFAULT_MODEL = os.getenv("MODEL_ID", "openai/gpt-oss-120b")
TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300.0"))

# Global HTTP client
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        base_url=VLLM_BASE_URL,
        timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=200),
    )
    logger.info(f"Initialized GEAP adapter forwarding to vLLM at {VLLM_BASE_URL}")
    yield
    if http_client:
        await http_client.aclose()
    logger.info("Adapter shutdown complete.")


app = FastAPI(
    title="Agent Platform LLM Custom Container Adapter",
    description="Agent Platform (fka Vertex AI) & OpenAI-compatible inference adapter for vLLM",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request/Response Schemas for Vertex AI Protocol ---

class PredictionInstance(BaseModel):
    """Represents a single prediction instance passed in Vertex AI format."""
    messages: Optional[List[Dict[str, Any]]] = None
    prompt: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    model: Optional[str] = None
    reasoning_effort: Optional[str] = None  # e.g., 'low', 'medium', 'high' for gpt-oss
    extra_body: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class VertexPredictRequest(BaseModel):
    """Vertex AI Standard Prediction Request payload."""
    instances: List[Union[PredictionInstance, Dict[str, Any]]]
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class VertexPredictResponse(BaseModel):
    """Vertex AI Standard Prediction Response payload."""
    predictions: List[Any]


# --- Health Endpoints for Vertex AI Probes ---

@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/v1/health", status_code=status.HTTP_200_OK)
@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint for Vertex AI / GEAP liveness and readiness probes.
    Queries the underlying vLLM engine to ensure the model weights are fully loaded.
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client uninitialized")

    try:
        response = await http_client.get("/health")
        if response.status_code == 200:
            return {"status": "healthy", "model": DEFAULT_MODEL}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "vLLM initializing", "upstream_code": response.status_code},
            )
    except Exception as exc:
        logger.debug(f"Health check probe failed: {exc}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "vLLM not reachable yet", "error": str(exc)},
        )


# --- Vertex AI Prediction Route ---

@app.post("/predict", response_model=VertexPredictResponse)
@app.post("/v1/models/{model_name:path}:predict", response_model=VertexPredictResponse)
async def vertex_predict(request: VertexPredictRequest):
    """
    Vertex AI standard predict route (`/predict`).
    Unpacks instances, handles multi-turn chat messages, tool definitions,
    and returns predictions matching the Vertex AI contract.
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client uninitialized")

    if not request.instances:
        raise HTTPException(status_code=400, detail="Instances list cannot be empty.")

    predictions = []
    global_params = request.parameters or {}

    for instance in request.instances:
        inst_dict = instance.dict(exclude_none=True) if isinstance(instance, BaseModel) else dict(instance)
        
        # Merge global parameters with instance-specific parameters (instance overrides parameters)
        merged_payload = {**global_params, **inst_dict}
        
        # Ensure model is set
        if "model" not in merged_payload or not merged_payload["model"]:
            merged_payload["model"] = DEFAULT_MODEL

        # Convert to OpenAI chat completions format if messages are present
        if "messages" in merged_payload:
            endpoint = "/v1/chat/completions"
        elif "prompt" in merged_payload:
            # Fallback to completions if raw prompt string given
            endpoint = "/v1/completions"
        else:
            raise HTTPException(
                status_code=400,
                detail="Instance must contain either 'messages' (for chat/tools) or 'prompt'.",
            )

        try:
            vllm_resp = await http_client.post(endpoint, json=merged_payload)
            if vllm_resp.status_code != 200:
                logger.error(f"vLLM error {vllm_resp.status_code}: {vllm_resp.text}")
                raise HTTPException(
                    status_code=vllm_resp.status_code,
                    detail=f"vLLM upstream error: {vllm_resp.text}",
                )
            result = vllm_resp.json()
            predictions.append(result)
        except httpx.RequestError as exc:
            logger.error(f"Failed to communicate with vLLM: {exc}")
            raise HTTPException(
                status_code=502, detail=f"Failed to communicate with vLLM engine: {exc}"
            )

    return VertexPredictResponse(predictions=predictions)


# --- OpenAI-Compatible Pass-Through Routes ---

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    Direct OpenAI-compatible chat completions proxy with streaming support.
    Supports tool calling, multi-turn messages, reasoning tokens.
    """
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client uninitialized")

    body = await request.json()
    if "model" not in body or not body["model"]:
        body["model"] = DEFAULT_MODEL

    is_stream = body.get("stream", False)

    if is_stream:
        async def stream_generator():
            try:
                async with http_client.stream("POST", "/v1/chat/completions", json=body) as stream_resp:
                    async for chunk in stream_resp.aiter_bytes():
                        yield chunk
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode("utf-8")

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            resp = await http_client.post("/v1/chat/completions", json=body)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except httpx.RequestError as exc:
            logger.error(f"vLLM upstream error: {exc}")
            raise HTTPException(status_code=502, detail=f"vLLM error: {exc}")


@app.get("/v1/models")
async def openai_models():
    """Lists available models from vLLM."""
    if not http_client:
        raise HTTPException(status_code=503, detail="HTTP client uninitialized")
    try:
        resp = await http_client.get("/v1/models")
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
