# Python Embedding Service Specification

## 1. Purpose

Build a standalone Python service that generates text embeddings using a local Hugging Face model.

This service will be called by the main Go backend through HTTP.

The Go backend will be responsible for:

- Receiving user requests
- Managing authentication and tenant filtering
- Extracting and chunking documents
- Calling this embedding service
- Storing returned vectors into PostgreSQL + pgvector
- Performing vector similarity search
- Calling external LLM APIs such as GPT or Claude

The Python service will only be responsible for:

- Loading the embedding model
- Receiving text input
- Returning vector embeddings
- Supporting batch embedding for document indexing

Recommended architecture:

```text
Go API / Worker
    ↓ HTTP
Python Embedding Service
    ↓
BAAI/bge-m3 model
    ↓
Vector result
    ↓ HTTP response
Go API / Worker
    ↓
PostgreSQL + pgvector
```

---

## 2. Why Use a Separate Embedding Service?

The embedding model should not be loaded directly inside the Go API.

Using a separate Python service is better because Python has stronger machine learning ecosystem support, especially for Hugging Face, PyTorch, and sentence-transformers.

### Benefits

| Benefit | Explanation |
|---|---|
| Clean separation | Go handles business logic, Python handles ML embedding |
| Easier model replacement | Model can be changed without major Go code changes |
| Better ML support | Python has mature libraries for Hugging Face models |
| Go app stays lightweight | Go service does not need PyTorch or ML dependencies |
| Local data control | Text is embedded locally, not sent to external embedding API |
| Lower embedding cost | No per-token embedding API cost |
| Easier scaling | Embedding service can be scaled independently |
| Future ready | Later can add GPU, reranker, or local LLM service separately |

### Trade-offs

| Trade-off | Explanation |
|---|---|
| Extra service to maintain | Need to deploy Go API and Python embedding service |
| More RAM usage | Model must stay loaded in memory |
| Startup can be slow | First startup downloads and loads model |
| CPU can be slow | Large document indexing can be slow without GPU |
| Need batching | Embedding chunks one by one is inefficient |
| Need failure handling | Go must handle timeout, retry, and service unavailable cases |
| Need model versioning | If model changes, old vectors may need reindexing |

---

## 3. Recommended Model

Use:

```text
BAAI/bge-m3
```

Hugging Face model ID:

```text
BAAI/bge-m3
```

Reason:

- Good multilingual embedding model
- Suitable for Indonesian + English mixed documents
- Good for semantic search and RAG
- Can be used through `sentence-transformers`
- Common output dimension is `1024`

Important:

The PostgreSQL pgvector column must match the model output dimension.

For BGE-M3:

```sql
embedding vector(1024)
```

---

## 4. Service Responsibilities

The Python embedding service must provide:

1. Health check endpoint
2. Single text embedding endpoint
3. Batch text embedding endpoint
4. Model metadata endpoint
5. Basic input validation
6. Error responses
7. Optional API key protection for internal service communication

This service should not:

- Store vectors in database
- Access PostgreSQL
- Perform RAG retrieval
- Call GPT/Claude
- Manage users or tenants
- Make business decisions

Those responsibilities belong to the Go backend.

---

## 5. Project Structure

Recommended folder structure:

```text
embedding-service/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── embedding.py
│   └── middleware.py
├── Dockerfile
├── docker-compose.example.yml
├── requirements.txt
├── .env.example
└── README.md
```

Explanation:

| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app and route definitions |
| `app/config.py` | Environment variable configuration |
| `app/schemas.py` | Request and response models |
| `app/embedding.py` | Model loading and embedding logic |
| `app/middleware.py` | Optional API key middleware |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build definition |
| `.env.example` | Example environment config |

---

## 6. Environment Variables

Create `.env.example`:

```env
APP_NAME=embedding-service
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8081

EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_NORMALIZE=true
EMBEDDING_DEVICE=cpu
EMBEDDING_MAX_BATCH_SIZE=64
EMBEDDING_MAX_TEXT_LENGTH=8000

API_KEY=
```

Explanation:

| Variable | Description |
|---|---|
| `EMBEDDING_MODEL_NAME` | Hugging Face model name |
| `EMBEDDING_NORMALIZE` | Whether to normalize embeddings |
| `EMBEDDING_DEVICE` | Use `cpu`, `cuda`, or `mps` |
| `EMBEDDING_MAX_BATCH_SIZE` | Maximum texts per batch request |
| `EMBEDDING_MAX_TEXT_LENGTH` | Maximum text length per item |
| `API_KEY` | Optional internal API key |

For first version, use:

```env
EMBEDDING_DEVICE=cpu
```

If using NVIDIA GPU later:

```env
EMBEDDING_DEVICE=cuda
```

If using Apple Silicon locally:

```env
EMBEDDING_DEVICE=mps
```

---

## 7. Python Dependencies

Create `requirements.txt`:

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
sentence-transformers==3.3.1
pydantic==2.10.4
pydantic-settings==2.7.0
python-dotenv==1.0.1
```

Install locally:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 8. API Contract

### 8.1 Health Check

Endpoint:

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "embedding-service"
}
```

Purpose:

Used by Docker, Kubernetes, load balancer, or Go backend to check if the service is alive.

---

### 8.2 Model Info

Endpoint:

```http
GET /model-info
```

Response:

```json
{
  "model_name": "BAAI/bge-m3",
  "dimension": 1024,
  "device": "cpu",
  "normalize_embeddings": true
}
```

Purpose:

Allows the Go backend or developer to confirm model name and vector dimension.

---

### 8.3 Single Embedding

Endpoint:

```http
POST /embed
```

Request:

```json
{
  "text": "How do I open a trading account?"
}
```

Response:

```json
{
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "embedding": [0.0123, -0.0456, 0.0789]
}
```

Use case:

- User asks a question
- Go API sends question to embedding service
- Go API receives question vector
- Go API searches PostgreSQL pgvector

---

### 8.4 Batch Embedding

Endpoint:

```http
POST /embed-batch
```

Request:

```json
{
  "texts": [
    "Chunk 1 content...",
    "Chunk 2 content...",
    "Chunk 3 content..."
  ]
}
```

Response:

```json
{
  "model": "BAAI/bge-m3",
  "dimension": 1024,
  "count": 3,
  "embeddings": [
    [0.01, -0.02, 0.03],
    [0.04, -0.05, 0.06],
    [0.07, -0.08, 0.09]
  ]
}
```

Use case:

- Document indexing worker extracts text
- Worker splits document into chunks
- Worker sends many chunks to `/embed-batch`
- Worker stores returned vectors into `ai_document_chunks`

Important:

Use `/embed-batch` for document indexing.
Do not embed document chunks one by one unless the document is very small.

---

## 9. Implementation Code

### 9.1 `app/config.py`

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "embedding-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8081

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_normalize: bool = True
    embedding_device: str = "cpu"
    embedding_max_batch_size: int = 64
    embedding_max_text_length: int = 8000

    api_key: str | None = None

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
```

---

### 9.2 `app/schemas.py`

```python
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embedding: list[float]


class EmbedBatchResponse(BaseModel):
    model: str
    dimension: int
    count: int
    embeddings: list[list[float]]


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelInfoResponse(BaseModel):
    model_name: str
    dimension: int
    device: str
    normalize_embeddings: bool
```

---

### 9.3 `app/embedding.py`

```python
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingService:
    def __init__(self):
        self.model_name = settings.embedding_model_name
        self.device = settings.embedding_device
        self.normalize = settings.embedding_normalize

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device
        )

        test_embedding = self.model.encode(
            "dimension test",
            normalize_embeddings=self.normalize
        )
        self.dimension = len(test_embedding)

    def embed(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=min(len(texts), settings.embedding_max_batch_size)
        )
        return embeddings.tolist()


embedding_service = EmbeddingService()
```

---

### 9.4 `app/middleware.py`

```python
from fastapi import Header, HTTPException
from app.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)):
    if not settings.api_key:
        return

    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

---

### 9.5 `app/main.py`

```python
from fastapi import FastAPI, Depends, HTTPException
from app.config import settings
from app.embedding import embedding_service
from app.middleware import verify_api_key
from app.schemas import (
    EmbedRequest,
    EmbedBatchRequest,
    EmbedResponse,
    EmbedBatchResponse,
    HealthResponse,
    ModelInfoResponse,
)

app = FastAPI(title=settings.app_name)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return {
        "model_name": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "device": embedding_service.device,
        "normalize_embeddings": embedding_service.normalize,
    }


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, _: None = Depends(verify_api_key)):
    text = req.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > settings.embedding_max_text_length:
        raise HTTPException(status_code=400, detail="Text is too long")

    embedding = embedding_service.embed(text)

    return {
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "embedding": embedding,
    }


@app.post("/embed-batch", response_model=EmbedBatchResponse)
def embed_batch(req: EmbedBatchRequest, _: None = Depends(verify_api_key)):
    texts = [text.strip() for text in req.texts if text.strip()]

    if not texts:
        raise HTTPException(status_code=400, detail="Texts cannot be empty")

    if len(texts) > settings.embedding_max_batch_size:
        raise HTTPException(status_code=400, detail="Batch size is too large")

    for text in texts:
        if len(text) > settings.embedding_max_text_length:
            raise HTTPException(status_code=400, detail="One or more texts are too long")

    embeddings = embedding_service.embed_batch(texts)

    return {
        "model": embedding_service.model_name,
        "dimension": embedding_service.dimension,
        "count": len(embeddings),
        "embeddings": embeddings,
    }
```

---

## 10. Running Locally

Start service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8081
```

First run may be slow because the model is downloaded from Hugging Face.
After that, the model is loaded from local cache.

Test health:

```bash
curl http://localhost:8081/health
```

Test model info:

```bash
curl http://localhost:8081/model-info
```

Test single embedding:

```bash
curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I open a trading account?"}'
```

Test batch embedding:

```bash
curl -X POST http://localhost:8081/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Chunk one content","Chunk two content"]}'
```

---

## 11. Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8081

EXPOSE 8081

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
```

Build:

```bash
docker build -t embedding-service .
```

Run:

```bash
docker run -p 8081:8081 embedding-service
```

---

## 12. Docker Compose Example

Create `docker-compose.example.yml`:

```yaml
services:
  embedding-service:
    build: .
    container_name: embedding-service
    ports:
      - "8081:8081"
    environment:
      APP_ENV: production
      EMBEDDING_MODEL_NAME: BAAI/bge-m3
      EMBEDDING_NORMALIZE: "true"
      EMBEDDING_DEVICE: cpu
      EMBEDDING_MAX_BATCH_SIZE: 64
      EMBEDDING_MAX_TEXT_LENGTH: 8000
    restart: unless-stopped
```

In the full CRM stack, the Go API can call:

```text
http://embedding-service:8081/embed
```

or:

```text
http://embedding-service:8081/embed-batch
```

---

## 13. PostgreSQL pgvector Setup

The Python service does not store vectors.
The Go backend stores vectors into PostgreSQL.

Enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Example document chunks table:

```sql
CREATE TABLE ai_document_chunks (
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_uuid UUID NOT NULL,
    client_uuid UUID NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT,
    embedding vector(1024),
    source_page INT,
    source_section TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT now()
);
```

Create vector index:

```sql
CREATE INDEX idx_ai_chunks_embedding_hnsw
ON ai_document_chunks
USING hnsw (embedding vector_cosine_ops);
```

Important:

Because embeddings are normalized, cosine similarity is recommended.

Example similarity query:

```sql
SELECT
    uuid,
    content,
    1 - (embedding <=> $1) AS similarity
FROM ai_document_chunks
WHERE client_uuid = $2
ORDER BY embedding <=> $1
LIMIT 8;
```

---

## 14. Go Integration Concept

The Go backend should call the Python service through HTTP.

### 14.1 Single Embedding Flow

Used when user asks a question.

```text
User asks question
    ↓
Go API receives question
    ↓
Go calls POST /embed
    ↓
Python returns vector
    ↓
Go searches pgvector
    ↓
Go sends retrieved chunks to GPT/Claude
    ↓
Go returns answer
```

### 14.2 Batch Embedding Flow

Used when indexing documents.

```text
User uploads document
    ↓
Go saves document
    ↓
Go pushes indexing job
    ↓
Worker extracts text
    ↓
Worker chunks text
    ↓
Worker calls POST /embed-batch
    ↓
Python returns vectors
    ↓
Worker stores chunks + vectors in PostgreSQL
```

---

## 15. Go HTTP Client Example

Example Go struct:

```go
type EmbedRequest struct {
    Text string `json:"text"`
}

type EmbedResponse struct {
    Model     string    `json:"model"`
    Dimension int       `json:"dimension"`
    Embedding []float32 `json:"embedding"`
}
```

Example call:

```go
func Embed(ctx context.Context, baseURL string, text string) ([]float32, error) {
    payload := EmbedRequest{Text: text}

    bodyBytes, err := json.Marshal(payload)
    if err != nil {
        return nil, err
    }

    req, err := http.NewRequestWithContext(
        ctx,
        http.MethodPost,
        baseURL+"/embed",
        bytes.NewReader(bodyBytes),
    )
    if err != nil {
        return nil, err
    }

    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{
        Timeout: 10 * time.Second,
    }

    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    if resp.StatusCode != http.StatusOK {
        return nil, fmt.Errorf("embedding service returned status %d", resp.StatusCode)
    }

    var result EmbedResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return result.Embedding, nil
}
```

Recommended timeout:

- `/embed`: 5-10 seconds
- `/embed-batch`: 30-120 seconds depending on batch size and server speed

---

## 16. Performance Notes

### User Question Embedding

Usually fast because it is only one short text.

Expected latency on CPU:

```text
100ms - 1500ms
```

### Document Indexing

Can be slower because documents produce many chunks.

Example:

```text
100 page PDF
    ↓
100-300 chunks
    ↓
100-300 embeddings
```

This can take seconds to minutes on CPU.

Therefore:

- Do not embed documents inside the upload HTTP request
- Use a background worker
- Use `/embed-batch`
- Store document status as `pending`, `processing`, `indexed`, or `failed`

---

## 17. Recommended Production Rules

1. Keep embedding service internal only.
2. Do not expose it publicly to the internet.
3. Use API key or private network between Go and Python service.
4. Use `/embed-batch` for indexing documents.
5. Use `/embed` for user questions.
6. Log latency in Go backend.
7. Store model name and dimension in retrieval logs.
8. If model changes, reindex old documents.
9. Always use same model for document chunks and user questions.
10. Use `client_uuid` filtering in Go/PostgreSQL, not in the embedding service.

---

## 18. Important Model Versioning Rule

The same embedding model must be used for both indexing and searching.

Correct:

```text
Document chunks → BAAI/bge-m3
User question   → BAAI/bge-m3
```

Wrong:

```text
Document chunks → BAAI/bge-m3
User question   → OpenAI embedding
```

If the model changes later, old document chunks should be reindexed.

Suggested metadata to store in `ai_documents` or `ai_document_chunks`:

```json
{
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "normalized": true
}
```

---

## 19. Error Handling

The service should return clear errors.

### Empty Text

Status:

```http
400 Bad Request
```

Response:

```json
{
  "detail": "Text cannot be empty"
}
```

### Text Too Long

Status:

```http
400 Bad Request
```

Response:

```json
{
  "detail": "Text is too long"
}
```

### Batch Too Large

Status:

```http
400 Bad Request
```

Response:

```json
{
  "detail": "Batch size is too large"
}
```

### Invalid API Key

Status:

```http
401 Unauthorized
```

Response:

```json
{
  "detail": "Invalid API key"
}
```

---

## 20. Final Recommendation

Build this service as a small internal microservice:

```text
embedding-service
    ├── Python
    ├── FastAPI
    ├── sentence-transformers
    └── BAAI/bge-m3
```

The Go backend should call it through HTTP:

```text
POST /embed
POST /embed-batch
```

The Python service should only return embeddings.
The Go backend should store vectors, retrieve chunks, enforce tenant security, and call the final LLM.

This design is recommended because it keeps your Go backend clean, avoids embedding API cost, keeps document embedding local, and allows the embedding layer to scale independently later.
