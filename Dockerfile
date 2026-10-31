# parsrag API + web UI (lesson 12).
#   docker build -t parsrag .
#   docker run --rm -p 8000:8000 -v "$PWD/data/fixtures/corpus:/data/docs:ro" parsrag
# Offline by default (FakeLLM + HashingEmbedding); set PARSRAG_LLM_* for a real model.

FROM python:3.12-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Wheels for parsrag and its runtime extras; nothing from the build stage ships except these.
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[api,openai,loaders,pgvector]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PARSRAG_DOCS=/data/docs
RUN useradd --create-home --uid 10001 parsrag \
    && mkdir -p /data/docs && chown parsrag /data/docs
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels "parsrag[api,openai,loaders,pgvector]" \
    && rm -rf /wheels
USER parsrag
WORKDIR /home/parsrag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]
CMD ["parsrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
