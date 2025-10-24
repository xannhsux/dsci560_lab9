FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PERSIST_DIR=/app/vectorstore/faiss \
    UPLOAD_DIR=/app/uploads

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY beckend beckend
COPY htmlTemplates.py htmlTemplates.py
COPY frontend frontend

RUN mkdir -p "${PERSIST_DIR}" "${UPLOAD_DIR}" /app/data

EXPOSE 8000

CMD ["uvicorn", "beckend.app:app", "--host", "0.0.0.0", "--port", "8000"]
