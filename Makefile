.PHONY: install test lint format clean docker-build docker-run docker-compose-up docker-compose-down

# Python version
PYTHON := python3.10

# Install dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

# Run tests
test:
	$(PYTHON) -m pytest tests/ -v --cov=trading --cov-report=term-missing

# Run linting
lint:
	$(PYTHON) -m flake8 trading tests
	$(PYTHON) -m black --check trading tests
	$(PYTHON) -m isort --check-only trading tests
	$(PYTHON) -m mypy trading tests

# Format code
format:
	$(PYTHON) -m black trading tests
	$(PYTHON) -m isort trading tests

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name "dist" -exec rm -r {} +
	find . -type d -name "build" -exec rm -r {} +

# Docker commands
docker-build:
	docker build -t trading-app .

docker-run:
	docker run -p 8501:8501 trading-app

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

# Development server
dev:
	$(PYTHON) -m streamlit run app.py

# Production server
prod:
	$(PYTHON) -m streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# Security checks
security:
	$(PYTHON) -m bandit -r trading
	$(PYTHON) -m safety check

# Performance benchmarks
benchmark:
	$(PYTHON) -m pytest tests/test_nlp/test_performance.py -v --benchmark-only

# Documentation
docs:
	$(PYTHON) -m pdoc --html --output-dir docs trading

# Help
help:
	@echo "Available commands:"
	@echo "  install          - Install dependencies"
	@echo "  test            - Run tests"
	@echo "  lint            - Run linting"
	@echo "  format          - Format code"
	@echo "  clean           - Clean up"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-run      - Run Docker container"
	@echo "  docker-compose-up   - Start Docker Compose services"
	@echo "  docker-compose-down - Stop Docker Compose services"
	@echo "  dev             - Run development server"
	@echo "  prod            - Run production server"
	@echo "  security        - Run security checks"
	@echo "  benchmark       - Run performance benchmarks"
	@echo "  docs            - Generate documentation" 