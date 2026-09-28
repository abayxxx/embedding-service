FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user. Pre-create the Hugging Face cache dir and give it to
# appuser: a named volume mounted here would otherwise be created root-owned,
# and the (non-root) process couldn't write the downloaded model into it.
# HF_HOME pins the cache location regardless of whether $HOME is set for the user.
ENV HF_HOME=/home/appuser/.cache/huggingface
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "$HF_HOME" \
    && chown -R appuser:appuser /home/appuser/.cache
USER appuser

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8081

EXPOSE 8081

# Readiness probe: /ready returns 200 only once the model is loaded, so this
# also gates `depends_on: condition: service_healthy` in the compose stack.
# start-period is long because the first run downloads ~2GB before /ready
# succeeds; failing checks during that window don't count against --retries.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8081/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
