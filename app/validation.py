from fastapi import HTTPException
from app.embedding import embedding_service
from app.config import settings

def ensure_model_ready():
    """Dependency guard to ensure the embedding model is loaded before serving requests."""
    if not embedding_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

def validate_text(text: str) -> str:
    """Validate and sanitize single text input."""
    cleaned = text.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(cleaned) > settings.embedding_max_text_length:
        raise HTTPException(status_code=400, detail="Text is too long")
    return cleaned

def validate_texts(texts: list[str]) -> list[str]:
    """Validate and sanitize batch text inputs."""
    cleaned = [t.strip() for t in texts if t.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Texts cannot be empty")
    for t in cleaned:
        if len(t) > settings.embedding_max_text_length:
            raise HTTPException(status_code=400, detail="One or more texts are too long")
    return cleaned
