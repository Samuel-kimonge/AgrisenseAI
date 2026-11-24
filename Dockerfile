# syntax=docker/dockerfile:1.7
# AgriSense AI - production image (CPU-only, non-root, read-only-filesystem friendly).
#
#   docker build -t agrisense-ai:0.1.0 .                 # runtime image (default)
#   docker build --target test -t agrisense-ai:test .    # image that runs the test-suite
#   docker run --rm -p 8501:8501 -v agrisense-data:/data agrisense-ai:0.1.0

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- builder
FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# CPU-only PyTorch wheels (a few hundred MB instead of multi-GB CUDA wheels).
# Versions match the ones the model was trained and tested with.
ARG TORCH_VERSION=2.5.1
ARG TORCHVISION_VERSION=0.20.1
RUN pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu

WORKDIR /build
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# ---------------------------------------------------------------- base (shared by test + runtime)
FROM python:${PYTHON_VERSION}-slim AS base
ARG APP_UID=10001
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=8 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ANONYMIZED_TELEMETRY=False \
    TORCH_THREADS=2 \
    OMP_NUM_THREADS=2 \
    DATABASE_PATH=/data/agrisense.db \
    UPLOADS_DIR=/data/uploads \
    CHROMA_DIR=/data/chroma

# Unprivileged user. /data is the ONLY writable location (plus /tmp): SQLite history,
# saved photos and the rebuildable Chroma index.
RUN groupadd --gid ${APP_UID} app \
 && useradd --uid ${APP_UID} --gid app --no-create-home --shell /usr/sbin/nologin app \
 && mkdir -p /data && chown app:app /data

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Application code is root-owned and read-only for the app user.
COPY app.py config.py ./
COPY pages ./pages
COPY src ./src
COPY data/knowledge_base ./data/knowledge_base
COPY models/production ./models/production
COPY reports/evaluation ./reports/evaluation

# ---------------------------------------------------------------- test
FROM base AS test
COPY tests ./tests
USER app
CMD ["pytest", "-q", "--no-header", "-p", "no:cacheprovider"]

# ---------------------------------------------------------------- runtime (default target)
FROM base AS runtime
USER app
VOLUME ["/data"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]
CMD ["streamlit", "run", "app.py"]
