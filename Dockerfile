FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV, dlib, PyTorch, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
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

# Copiar archivos de dependencias
COPY pyproject.toml uv.lock /app/

# Instalar dependencias usando uv
# --frozen: usa el lock file sin actualizarlo
# --no-install-project: solo instala dependencias, no el proyecto
RUN uv sync --frozen --no-install-project --no-cache

# Agregar .venv al PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copiar el código fuente completo
COPY . .

# Crear directorios necesarios para la aplicación
RUN mkdir -p temp_uploads audit_storage

# Crear usuario no-root y asignar permisos
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Exponer puerto de la API
EXPOSE 8000

# Comando por defecto: iniciar API
CMD ["/app/.venv/bin/uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
