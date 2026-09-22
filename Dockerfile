FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for C-extensions (Levenshtein, tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY checkpoints/ checkpoints/
COPY tokenizer/ tokenizer/
COPY model/ model/
COPY service/ service/

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]