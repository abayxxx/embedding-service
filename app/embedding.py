import logging
import threading

from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Loads the embedding model and turns text into vectors.

    The model is loaded explicitly via ``load()`` (called from the FastAPI
    lifespan handler) instead of at import time, so the process can start
    serving liveness checks while the ~2GB model is still downloading/loading
    (fix #3).
    """

    def __init__(self):
        self.model_name = settings.embedding_model_name
        self.device = settings.embedding_device
        self.normalize = settings.embedding_normalize

        self.model: SentenceTransformer | None = None
        self.dimension: int = 0
        self.max_seq_length: int = 0

        # Backpressure: serialize (or bound) concurrent inference so we don't
        # oversubscribe the CPU when many requests arrive at once (fix #1).
        # Routes run in Starlette's threadpool, so a threading primitive (not
        # asyncio) is the correct one to block on here.
        self._inference_semaphore = threading.BoundedSemaphore(
            max(1, settings.embedding_max_concurrency)
        )

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None:
            return

        if settings.embedding_num_threads and settings.embedding_num_threads > 0:
            # Limit CPU oversubscription when multiple requests run at once.
            import torch

            torch.set_num_threads(settings.embedding_num_threads)

        logger.info(
            "Loading embedding model '%s' on device '%s' (first run downloads from Hugging Face)...",
            self.model_name,
            self.device,
        )
        model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = model.get_sentence_embedding_dimension()
        self.max_seq_length = int(model.max_seq_length)

        # Warm up the model so the first real request doesn't pay torch's
        # one-time kernel/graph initialization cost (several seconds on
        # CPU/MPS). Running one throwaway encode here moves that cost into
        # startup, before /ready flips true and any traffic arrives.
        model.encode(
            "warmup",
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        # Publish the model reference last, so ``is_ready`` (and therefore
        # /ready and the request guards) only flip true once dimension/
        # max_seq_length are set and the warmup is done. This matters now that
        # load() runs in a background thread and requests can arrive mid-load.
        self.model = model

        logger.info(
            "Model loaded and warmed: dimension=%d max_seq_length=%d",
            self.dimension,
            self.max_seq_length,
        )

    def _is_truncated(self, text: str) -> bool:
        """Token-aware truncation check (fix #4).

        ``embedding_max_text_length`` guards on characters, but the model
        truncates on *tokens* (max_seq_length). This tells us whether a given
        input will actually be cut so we can surface it instead of silently
        embedding a partial chunk.

        Cheap gate (fix #2): this tokenizer emits roughly <=1 token per
        character, so a text whose character length is already within the token
        budget cannot be truncated. We only pay for the extra tokenization when
        the input is long enough to plausibly exceed the limit, which avoids
        double-tokenizing every normal-sized chunk.
        """
        if len(text) <= self.max_seq_length:
            return False
        token_ids = self.model.tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids) > self.max_seq_length

    def embed(self, text: str) -> tuple[list[float], bool]:
        truncated = self._is_truncated(text)
        if truncated:
            logger.warning(
                "Input exceeds model max_seq_length (%d tokens); it will be truncated.",
                self.max_seq_length,
            )

        with self._inference_semaphore:
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        return embedding.tolist(), truncated

    def embed_batch(self, texts: list[str]) -> tuple[list[list[float]], int]:
        truncated_count = sum(1 for t in texts if self._is_truncated(t))
        if truncated_count:
            logger.warning(
                "%d of %d inputs exceed model max_seq_length (%d tokens); they will be truncated.",
                truncated_count,
                len(texts),
                self.max_seq_length,
            )

        with self._inference_semaphore:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                batch_size=min(len(texts), settings.embedding_max_batch_size),
                show_progress_bar=False,
            )
        return embeddings.tolist(), truncated_count


embedding_service = EmbeddingService()
