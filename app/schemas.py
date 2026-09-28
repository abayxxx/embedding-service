from pydantic import BaseModel, Field

from app.config import settings


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedBatchRequest(BaseModel):
    # max_length caps the batch at the Pydantic layer so an oversized payload is
    # rejected before we parse/strip the whole list (fix #3).
    texts: list[str] = Field(
        ..., min_length=1, max_length=settings.embedding_max_batch_size
    )


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embedding: list[float]
    # True when the input exceeded the model's max token length and was
    # silently truncated by the tokenizer (fix #4). Additive/backward-safe.
    truncated: bool = False


class EmbedBatchResponse(BaseModel):
    model: str
    dimension: int
    count: int
    embeddings: list[list[float]]
    # Number of inputs in this batch that were truncated (fix #4).
    truncated_count: int = 0


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    dimension: int
    device: str
    normalize_embeddings: bool
    max_seq_length: int
