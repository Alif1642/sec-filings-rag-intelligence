FROM python:3.13.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY api ./api
COPY app ./app
COPY ingestion ./ingestion
COPY src ./src
COPY evals ./evals
COPY scripts ./scripts
COPY docs ./docs
COPY data ./data

RUN python -m pip install --upgrade pip && pip install .

EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
