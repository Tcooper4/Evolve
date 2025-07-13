# Docker Configuration

This directory contains all Dockerfiles for the Evolve AI Trading System.

## Dockerfile Overview

### Base Images
- **Dockerfile.base** - Base image with common dependencies and Python environment
- **Dockerfile** - Main application Dockerfile (from deploy/)

### Service-Specific Images
- **Dockerfile.api** - API service container
- **Dockerfile.web** - Web interface container  
- **Dockerfile.worker** - Background worker container
- **Dockerfile.monitor** - Monitoring and metrics container
- **Dockerfile.services** - Trading services container (from trading/services/)

### Production Images
- **Dockerfile.production** - Production-optimized main application

## Usage

### Development
```bash
# Build base image
docker build -f docker/Dockerfile.base -t evolve-base .

# Build specific service
docker build -f docker/Dockerfile.api -t evolve-api .
```

### Production
```bash
# Build production image
docker build -f docker/Dockerfile.production -t evolve-production .
```

## Docker Compose

Use the appropriate docker-compose file for your environment:
- `docker-compose.yml` - Development
- `docker-compose.production.yml` - Production

## Notes

- All images are based on Python 3.11+
- Base image includes common ML/AI dependencies
- Service-specific images extend the base image with additional dependencies
- Production image is optimized for size and security 