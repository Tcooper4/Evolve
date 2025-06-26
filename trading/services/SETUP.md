# Trading Agent Services Setup Guide

This guide provides step-by-step instructions for setting up and running the trading agent services architecture.

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Redis server
- Git

### 2. Install Dependencies

```bash
# Install Redis
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Install Python dependencies
cd trading/services
pip install -r requirements.txt
```

### 3. Start Redis

```bash
redis-server
```

### 4. Test the Setup

```bash
# Test basic functionality
python test_services.py

# Test specific components
python test_services.py --test redis
python test_services.py --test manager
python test_services.py --test client
```

### 5. Start Services

#### Option A: Using ServiceManager (Recommended)

```bash
# Start all services
python service_manager.py --action start-all

# Start specific service
python service_manager.py --action start --service model_builder

# Check status
python service_manager.py --action status

# Stop all services
python service_manager.py --action stop-all
```

#### Option B: Manual Launch

```bash
# Start services in separate terminals
python launch_model_builder.py
python launch_performance_critic.py
python launch_updater.py
python launch_research.py
python launch_meta_tuner.py
python launch_multimodal.py
python launch_prompt_router.py
```

### 6. Test Service Communication

```bash
# Test with client
python service_client.py --action build
python service_client.py --action search
python service_client.py --action route
```

## Docker Deployment

### 1. Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

### 2. Individual Service Containers

```bash
# Build service image
docker build -f services/Dockerfile -t evolve/services .

# Run individual service
docker run -d --name model_builder \
  --network evolve_network \
  -e REDIS_HOST=redis \
  evolve/services \
  python services/launch_model_builder.py
```

## Service Architecture Overview

### Core Components

1. **BaseService** - Abstract base class providing Redis pub/sub infrastructure
2. **ServiceManager** - Centralized orchestration and monitoring
3. **ServiceClient** - High-level API for service interaction
4. **Individual Services** - Wrappers around existing agents

### Communication Flow

```
Client Request → Redis Channel → Service → Agent → Response → Redis Channel → Client
```

### Service Channels

Each service uses three Redis channels:
- `{service_name}_input` - Receives requests
- `{service_name}_output` - Sends responses
- `{service_name}_control` - Control messages (start/stop/status)

## Configuration

### Environment Variables

```bash
# Redis Configuration
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

# OpenAI API (for research and multimodal services)
export OPENAI_API_KEY=your_openai_api_key

# Logging
export LOG_LEVEL=INFO
```

### Service Configuration

Each service can be configured via:
- Environment variables
- Configuration files
- Command line arguments

## Monitoring and Debugging

### Logs

Each service creates its own log file:
```bash
tail -f logs/model_builder_service.log
tail -f logs/performance_critic_service.log
# ... etc
```

### Redis Monitoring

```bash
# Monitor all Redis traffic
redis-cli monitor

# Subscribe to specific service output
redis-cli subscribe model_builder_output

# Check Redis info
redis-cli info
```

### Service Health Checks

```bash
# Check service status
python service_manager.py --action status

# Ping individual services
python service_client.py --action ping --service model_builder
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis is running
   redis-cli ping
   
   # Start Redis if needed
   redis-server
   ```

2. **Service Not Responding**
   ```bash
   # Check service logs
   tail -f logs/{service_name}_service.log
   
   # Restart service
   python service_manager.py --action stop --service {service_name}
   python service_manager.py --action start --service {service_name}
   ```

3. **Import Errors**
   ```bash
   # Check Python path
   export PYTHONPATH=/path/to/evolve_clean
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

### Performance Tuning

1. **Redis Performance**
   ```bash
   # Configure Redis memory
   redis-cli config set maxmemory 1gb
   redis-cli config set maxmemory-policy allkeys-lru
   ```

2. **Service Scaling**
   ```bash
   # Run multiple instances
   python launch_model_builder.py &
   python launch_model_builder.py &
   ```

## Development

### Adding New Services

1. Create service class inheriting from `BaseService`
2. Implement `process_message()` method
3. Create launcher script
4. Add to `ServiceManager`
5. Update documentation

### Testing

```bash
# Run all tests
python test_services.py

# Run specific tests
python test_services.py --test redis
python test_services.py --test communication
```

### Code Quality

```bash
# Format code
black services/

# Lint code
flake8 services/

# Type checking
mypy services/
```

## Production Deployment

### Security Considerations

1. **Redis Security**
   ```bash
   # Enable Redis authentication
   redis-cli config set requirepass your_password
   
   # Use SSL/TLS
   redis-cli config set tls-port 6380
   ```

2. **Network Security**
   ```bash
   # Restrict Redis access
   redis-cli config set bind 127.0.0.1
   
   # Use firewall rules
   ufw allow from 192.168.1.0/24 to any port 6379
   ```

### High Availability

1. **Redis Clustering**
   ```bash
   # Setup Redis cluster
   redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 127.0.0.1:7002
   ```

2. **Service Redundancy**
   ```bash
   # Run multiple service instances
   docker-compose up -d --scale model_builder=3
   ```

### Monitoring

1. **Metrics Collection**
   ```bash
   # Redis metrics
   redis-cli info memory
   redis-cli info stats
   
   # Service metrics
   python service_manager.py --action stats
   ```

2. **Alerting**
   ```bash
   # Monitor service health
   while true; do
     python service_manager.py --action status
     sleep 60
   done
   ```

## Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the troubleshooting section
3. Check Redis connectivity
4. Verify service dependencies

## Next Steps

After setting up the services:

1. **Integrate with existing agents** - Ensure all agent functionality is properly wrapped
2. **Add more services** - Create services for additional agents
3. **Implement load balancing** - Distribute requests across multiple service instances
4. **Add monitoring** - Implement comprehensive monitoring and alerting
5. **Optimize performance** - Profile and optimize service performance 