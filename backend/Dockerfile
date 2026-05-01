FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run training on first boot if models not present, then start API
CMD ["sh", "-c", "[ -f models/artifacts/scaler.pkl ] || python train.py && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]