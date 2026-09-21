from python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY checkpoints/ checkpoints/
COPY tokenizer/ tokenizer/
COPY model/ model/
COPY service/ service/

EXPOSE 8000

ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]