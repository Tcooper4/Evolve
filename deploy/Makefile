.PHONY: install install-dev test lint format typecheck clean docs serve-docs docker-build docker-run

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=. --cov-report=html

test-fast:
	pytest tests/ -v -n auto

# Code quality
lint:
	flake8 .
	black . --check
	isort . --check-only
	mypy .

format:
	black .
	isort .

typecheck:
	mypy .

# Documentation
docs:
	cd docs && make html

serve-docs:
	mkdocs serve

# Docker
docker-build:
	docker build -t trading-platform .

docker-run:
	docker run -p 8501:8501 trading-platform

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".coverage" -exec rm -r {} +
	find . -type d -name "htmlcov" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "*.log" -delete

# Development
dev:
	streamlit run app.py

# Security
security:
	bandit -r .
	safety check

# Performance
profile:
	python -m cProfile -o output.prof app.py
	snakeviz output.prof

# Help
help:
	@echo "Available commands:"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  test          - Run tests"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  test-fast     - Run tests in parallel"
	@echo "  lint          - Run linters"
	@echo "  format        - Format code"
	@echo "  typecheck     - Run type checker"
	@echo "  docs          - Build documentation"
	@echo "  serve-docs    - Serve documentation"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  clean         - Clean up cache files"
	@echo "  dev           - Run development server"
	@echo "  security      - Run security checks"
	@echo "  profile       - Profile application"
	@echo "  help          - Show this help message"
