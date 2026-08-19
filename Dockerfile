# RoadScan — FastAPI dashboard for Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 ROADSCAN_DEVICE=cpu

# opencv/ultralytics runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT (default 8080). Shell form so the variable is expanded.
ENV PORT=8080
EXPOSE 8080
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
