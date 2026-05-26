# Dockerfile
# FastAPI RAG System — Production Container

# ── Base Image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — smaller image, faster deploy
# WHY slim? Full image is 1GB+, slim is ~200MB
# Railway has faster deploys with smaller images
FROM python:3.13-slim

# ── Environment Variables ─────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
# Don't write .pyc files — saves space

ENV PYTHONUNBUFFERED=1
# Print logs immediately — important for Railway log viewer

ENV PYTHONPATH=/app
# Makes "from rag_core.xxx import" work inside container

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── System Dependencies ───────────────────────────────────────────────────────
# Install build tools for packages that need C compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*
# rm -rf cleans up apt cache — keeps image small

# ── Python Dependencies ───────────────────────────────────────────────────────
# Copy requirements first (Docker layer caching)
# WHY copy requirements before code?
#   If only CODE changes → Docker reuses cached pip install layer
#   If requirements change → Docker reinstalls packages
#   Saves 3-5 minutes on every redeploy
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy Application Code ─────────────────────────────────────────────────────
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health Check ──────────────────────────────────────────────────────────────
# Railway calls this to verify container is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start Command ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1"]
# workers=1 for Railway free tier
# Increase to 2-4 on paid tier
