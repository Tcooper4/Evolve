FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire trading directory
COPY trading/ ./trading/

# Copy necessary files from other directories
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/
COPY utils/ ./utils/
COPY memory/ ./memory/

# Create necessary directories
RUN mkdir -p logs models data plots research optimization visualization

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import redis; redis.Redis(host='redis', port=6379).ping()" || exit 1

# Default command (can be overridden)
CMD ["python", "trading/services/launch_model_builder.py"] 