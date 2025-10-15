# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PYTHON=python3.11

# Copy manifests + README first so uv can build the project
COPY pyproject.toml uv.lock README.md ./

# Copy source (so the package exists at /app/src during uv sync)
COPY src ./src
COPY scripts ./scripts
# (optional) COPY sample_images ./sample_images
# (optional) COPY assets ./assets

RUN pip install --no-cache-dir uv \
 && uv sync --frozen

# Install model/runtime deps (Torch CPU wheels available on Linux)
RUN uv add ultralytics pillow numpy 

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.inviol_image_analyser_assignment.app:app", "--host", "0.0.0.0", "--port", "8000"]
