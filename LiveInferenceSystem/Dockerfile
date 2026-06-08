FROM ubuntu:22.04

# System deps
RUN apt-get update && apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install Python deps via uv (dev group excluded — no pytest in prod image)
# UV_PYTHON pins to 3.11 to avoid issues with newer system pythons
WORKDIR /usr/src/install
COPY ./LiveInferenceSystem/pyproject.toml ./pyproject.toml
COPY ./LiveInferenceSystem/uv.lock ./uv.lock
ENV UV_PROJECT_ENVIRONMENT=/usr/src/venv
ENV UV_PYTHON=3.11
RUN uv python install 3.11 && uv sync --no-dev

# Bake HuggingFace model into image cache at build time
ENV HF_HOME=/usr/src/hf_cache
ENV TRANSFORMERS_CACHE=/usr/src/hf_cache
RUN <<'EOF'
/usr/src/venv/bin/python - <<'PYCODE'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="davethaler/whale-call-detector",
    local_dir="/usr/src/hf_cache",
    local_dir_use_symlinks=False,
    revision="d1eedf5c614268da7551039a84dfc35d317168b9",
    allow_patterns="*"
)
PYCODE
EOF

# At runtime, never hit the network for the model
ENV HF_HUB_OFFLINE=1

# App working directory
WORKDIR /usr/src/app

# PODS-AI source code
COPY ./src/ ./src/

# OrcaHello HLS client (required by LiveInferenceOrchestrator.py)
COPY ./external/orcahello/InferenceSystem/src/orcasound_hls/ ./external/orcahello/InferenceSystem/src/orcasound_hls/

ENTRYPOINT ["/usr/src/venv/bin/python3", "-u", "./src/LiveInferenceOrchestrator.py"]
CMD []
