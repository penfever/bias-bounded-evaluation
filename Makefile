# Differential Debiasing Development Makefile

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check clean build docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install package with all dev dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-performance - Run performance tests only"
	@echo "  lint         - Run linting (flake8)"
	@echo "  format       - Format code (black + isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build package"
	@echo "  docs         - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-performance:
	pytest tests/performance/

lint:
	flake8 differential_debiasing/ tests/ scripts/

format:
	black differential_debiasing/ tests/ scripts/
	isort differential_debiasing/ tests/ scripts/

type-check:
	mypy differential_debiasing/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	build:
	python -m build

docs:
	@echo "Documentation generation not yet implemented"

check-all: lint type-check test
	@echo "All checks passed!"
