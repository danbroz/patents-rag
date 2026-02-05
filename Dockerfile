FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Basic OS deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-mcp.txt /app/requirements-mcp.txt
RUN pip install --no-cache-dir -r /app/requirements-mcp.txt

COPY . /app

ENV PORT=9000 \
    FAISS_INDEX_PATH=/app/patents-embeddings/index.faiss \
    TITLES_PATH=/app/patents-index/patent_titles.npy

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "mcp_server.app:app", "--host", "0.0.0.0", "--port", "9000"]

