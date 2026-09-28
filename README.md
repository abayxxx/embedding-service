# Embedding Service

A small, stateless Python microservice that turns text into vector embeddings
using a local Hugging Face model (`BAAI/bge-m3`, 1024-dim). It is meant to be
called over HTTP by the Go backend, which owns auth, tenancy, chunking, the
database, and the final LLM call. This service **only** returns vectors.

```
Go API / Worker  ──HTTP──▶  Embedding Service  ──▶  BAAI/bge-m3  ──▶  vectors
```

## Endpoints

| Method | Path           | Purpose                                              |
|--------|----------------|------------------------------------------------------|
| GET    | `/health`      | Liveness — process is up (does **not** need model)   |
| GET    | `/ready`       | Readiness — `200` only once the model is loaded      |
| GET    | `/model-info`  | Model name, dimension, device, max token length      |
| POST   | `/embed`       | Embed a single text                                  |
| POST   | `/embed-batch` | Embed many texts (use this for document indexing)    |

`/embed` and `/embed-batch` require the `X-API-Key` header **only if** `API_KEY`
is set in the environment.

### Notes vs. the original spec

Two improvements were folded into the spec build:

- **Liveness/readiness split.** The ~2GB model loads in the FastAPI lifespan, so
  the container can answer `/health` immediately and report `/ready` only after
  the model is usable. Orchestrators should use `/ready` to gate traffic.
- **Truncation detection.** `bge-m3` truncates inputs longer than its token
  limit (`max_seq_length`). Responses now include `truncated` (single) /
  `truncated_count` (batch) and the service logs a warning, so partial-chunk
  embeddings aren't silent. Chunk on a *token* budget on the Go side.

## Run locally (without Docker)

> Targets Python **3.13** (matches the container). Torch ships cp313 wheels, so
> no special handling is needed.

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional

uvicorn app.main:app --host 0.0.0.0 --port 8081
```

The first run downloads `BAAI/bge-m3` (~2GB) from Hugging Face; subsequent runs
load from the local cache. On Apple Silicon you can set `EMBEDDING_DEVICE=mps`
in `.env` for faster local inference (Docker on Mac is CPU-only).

## Run with Docker

```bash
docker compose -f docker-compose.example.yml up --build
```

## Quick test

```bash
curl http://localhost:8081/health
curl http://localhost:8081/ready
curl http://localhost:8081/model-info

curl -X POST http://localhost:8081/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"How do I open a trading account?"}'

curl -X POST http://localhost:8081/embed-batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Chunk one content","Chunk two content"]}'
```

## Configuration

See `.env.example`. Key variables: `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`
(`cpu` / `cuda` / `mps`), `EMBEDDING_MAX_BATCH_SIZE`, `EMBEDDING_MAX_TEXT_LENGTH`,
`EMBEDDING_NUM_THREADS` (cap CPU threads), and `API_KEY`.

## Production rules

- Keep this service internal; do not expose it to the public internet.
- Use `API_KEY` or a private network between Go and this service.
- Use `/embed-batch` for indexing, `/embed` for user questions.
- Always embed documents and queries with the **same** model; if the model
  changes, reindex. Store `embedding_model` + `dimension` alongside vectors.
- Tenant filtering (`client_uuid`) belongs in Go/Postgres, not here.
