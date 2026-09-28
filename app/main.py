import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import ORJSONResponse

from app.config import settings
from app.embedding import embedding_service
from app.middleware import verify_api_key
from app.validation import ensure_model_ready, validate_text, validate_texts
from app.schemas import (
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(
        target=embedding_service.load, name="model-loader", daemon=True
    ).start()
    yield


app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.get("/ready", response_model=ReadyResponse, dependencies=[Depends(ensure_model_ready)])
def ready():
    return {
        "status": "ready",
        "service": settings.app_name,
        "model_loaded": True,
    }


@app.get("/model-info", response_model=ModelInfoResponse, dependencies=[Depends(ensure_model_ready)])
def model_info():
    return {
        "model_name": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "device": embedding_service.device,
        "normalize_embeddings": embedding_service.normalize,
        "max_seq_length": embedding_service.max_seq_length,
    }


@app.post(
    "/embed",
    response_model=EmbedResponse,
    dependencies=[Depends(verify_api_key), Depends(ensure_model_ready)],
)
def embed(req: EmbedRequest):
    text = validate_text(req.text)
    embedding, truncated = embedding_service.embed(text)

    return {
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "embedding": embedding,
        "truncated": truncated,
    }


@app.post(
    "/embed-batch",
    response_model=EmbedBatchResponse,
    dependencies=[Depends(verify_api_key), Depends(ensure_model_ready)],
)
def embed_batch(req: EmbedBatchRequest):
    texts = validate_texts(req.texts)
    embeddings, truncated_count = embedding_service.embed_batch(texts)

    return {
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "count": len(embeddings),
        "embeddings": embeddings,
        "truncated_count": truncated_count,
    }
