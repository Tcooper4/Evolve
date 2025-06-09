#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting local deployment...${NC}"

# Create necessary directories
echo "Creating directories..."
mkdir -p automation/logs
mkdir -p automation/data

# Copy configuration
echo "Setting up configuration..."
if [ ! -f config.json ]; then
    cp config.example.json config.json
    echo "Created config.json from example"
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo "Checking service health..."
if curl -s http://localhost:5000/health | grep -q "healthy"; then
    echo -e "${GREEN}Application is healthy${NC}"
else
    echo -e "${RED}Application health check failed${NC}"
    exit 1
fi

# Initialize Ray cluster
echo "Initializing Ray cluster..."
docker-compose exec app python -c "import ray; ray.init(address='ray:10001')"

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "Access the application at:"
echo "- Web Interface: http://localhost:5000"
echo "- Monitoring Dashboard: http://localhost:5000/dashboard"
echo "- Prometheus: http://localhost:9090"
echo "- Grafana: http://localhost:3000 (admin/admin)"

# Show logs
echo -e "\n${GREEN}Showing application logs...${NC}"
docker-compose logs -f app 