# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: collect the wheel + dependencies.
# Pick optional feature sets with:  --build-arg EXTRAS="lancedb,postgres,parsing,docling"
# (docling = layout-aware parsing + OCR; pulls torch, so the image is larger)
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ARG EXTRAS="lancedb,postgres,parsing,docling"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml ./
COPY multimodal_rag ./multimodal_rag
RUN python -m pip install --prefix=/install ".[${EXTRAS}]"

# Static ffmpeg (single self-contained binary; we never use ffprobe).
# Pinned by digest for reproducible builds; bump deliberately.
FROM mwader/static-ffmpeg@sha256:028fb402231d4f5f2223c67fca5d0abc04f5a38cffcbadf9844f06985e2259b8 AS ffmpeg-static

# ---------------------------------------------------------------------------
# Shared runtime base (no ffmpeg).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime-base

COPY --from=builder /install /usr/local

RUN useradd --create-home --uid 10001 mrag \
    && mkdir -p /data \
    && chown -R mrag:mrag /data

ENV MRAG_HOME=/data \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER mrag
WORKDIR /home/mrag
VOLUME ["/data"]

ENTRYPOINT ["mrag-ingest"]
CMD ["run"]

# ---------------------------------------------------------------------------
# Document-only image (no ffmpeg): smallest footprint.
#   docker build --target runtime-slim ...
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime-slim

# ---------------------------------------------------------------------------
# Default image: adds static ffmpeg for audio/video extraction (~80 MB).
# ---------------------------------------------------------------------------
FROM runtime-base AS runtime
COPY --from=ffmpeg-static /ffmpeg /usr/local/bin/ffmpeg
