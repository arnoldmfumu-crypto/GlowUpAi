FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=200 --retries=5 -r requirements.txt

COPY app.py .
COPY ingest.py .
COPY retriever.py .
COPY french_brands.py .
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
