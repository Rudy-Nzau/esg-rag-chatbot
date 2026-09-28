FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy app
COPY api/ ./api/
COPY rag/ ./rag/
COPY ingestion/ ./ingestion/

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
