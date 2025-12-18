
FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libgl1 \
   pkg-config \
   libopenblas-dev \
   liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Install dependencies
# --frozen: sync from lock file
# --no-install-project: don't install the current project (we just want deps first)
RUN uv sync --frozen --no-install-project --no-cache

# Add .venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

COPY . /app

RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

CMD ["bash"]