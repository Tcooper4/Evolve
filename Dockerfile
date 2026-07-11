# Evolve - single-container production image.
# Stage 1: build the React frontend
FROM node:22-slim AS frontend
WORKDIR /app/web/frontend
COPY web/frontend/package*.json ./
RUN npm ci
COPY web/frontend ./
RUN npm run build

# Stage 2: Python runtime serving API + built frontend at :8000
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend /app/web/frontend/dist ./web/frontend/dist
EXPOSE 8000
# data/ (accounts, keys, memory, watchlists) should be a mounted volume
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
