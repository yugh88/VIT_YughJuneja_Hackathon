# Sentinel demo image: builds the React dashboard, then serves it from FastAPI.
# docker compose up  ->  http://localhost:8000

# ---- web build (skipped gracefully if the web app is absent) ----
FROM node:20-slim AS web
WORKDIR /web
COPY src/web/ ./
RUN if [ -f package.json ]; then npm ci && npm run build; else mkdir -p dist; fi

# ---- runtime ----
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/models/hf SENTINEL_MODE=replay PYTHONPATH=/app/src
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home sentinel
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install -e . --no-deps
COPY data/ ./data/
COPY docs/ ./docs/
# Runtime models: ONNX classifiers (if built or downloaded) + fastText language ID.
COPY --chown=sentinel models/ ./models/
# Models built locally are copied above; a fresh clone downloads them from the GitHub Release
# (checksummed). If that is unreachable the engine still runs, in its documented fallback modes.
RUN [ -f models/lid.176.ftz ] || curl -fL -o models/lid.176.ftz https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz; \
    [ -f models/onnx/sentiment-en/model.int8.onnx ] || python -m engine.models_download || echo "models: release not reachable, fallback modes"
COPY --from=web /web/dist ./src/web/dist
RUN mkdir -p /app/data/output && chown -R sentinel /app/data/output && chown sentinel /app/models
USER sentinel
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s CMD curl -fs http://localhost:8000/v1/health || exit 1
CMD ["uvicorn", "engine.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
