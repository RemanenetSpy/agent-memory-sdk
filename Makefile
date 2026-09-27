# Makefile for Agent Memory Development

.PHONY: help install test lint typecheck format benchmark eval stress-charts clean docker-build docker-run docker-mcp

# Default target
help:
	@echo "Agent Memory - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install package in development mode with dev dependencies"
	@echo "  install-ci    Install for CI (no dev dependencies)"
	@echo ""
	@echo "Testing:"
	@echo "  test          Run all tests"
	@echo "  test-v        Run tests with verbose output"
	@echo "  test-cov      Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          Run ruff linter"
	@echo "  typecheck     Run mypy type checker"
	@echo "  format        Format code with ruff"
	@echo "  check         Run lint + typecheck"
	@echo ""
	@echo "Benchmarks & Evaluation:"
	@echo "  benchmark     Run quick benchmark"
	@echo "  benchmark-full Run full benchmark with seeded data"
	@echo "  eval          Run evaluation datasets"
	@echo "  stress-charts Regenerate stress-test charts (needs: pip install -e '.[charts]')"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run CLI in Docker"
	@echo "  docker-mcp    Run MCP server in Docker"
	@echo "  docker-compose-up  Start services with docker-compose"
	@echo "  docker-compose-down Stop services"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean         Remove build artifacts and cache"
	@echo "  reset-db      Reset ChromaDB database"

# Installation
install:
	pip install -e ".[dev]"

install-ci:
	pip install -e .

# Testing
test:
	python -m pytest tests/ -v

test-v:
	python -m pytest tests/ -vv --tb=long

test-cov:
	python -m pytest tests/ --cov=agent_memory --cov-report=term-missing --cov-report=html

# Code Quality
lint:
	ruff check agent_memory/ tests/

typecheck:
	mypy agent_memory/ --ignore-missing-imports

format:
	ruff format agent_memory/ tests/

check: lint typecheck

# Benchmarks & Evaluation
benchmark:
	agent-memory benchmark

benchmark-full:
	agent-memory benchmark --seed --repeat 3

eval:
	agent-memory eval

# Requires the optional charts extra: pip install -e ".[charts]"
stress-charts:
	uv run --extra charts python scripts/generate_stress_charts.py

# Docker
docker-build:
	docker build -t agent-memory:latest .

docker-run:
	docker run --rm -v agent_memory_data:/home/appuser/.agent_memory agent-memory:latest agent-memory $(ARGS)

docker-mcp:
	docker run -d --name agent-memory-mcp \
		-v agent_memory_data:/home/appuser/.agent_memory \
		-p 8000:8000 \
		agent-memory:latest agent-memory-mcp

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

# Maintenance
clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

reset-db:
	rm -rf .agent_memory .agent_memory_demo
	docker volume rm agent_memory_data 2>/dev/null || true

# Development shortcuts
dev: install
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make benchmark' to run benchmarks"

# Run example
example:
	python examples/basic_usage.py

# Run CLI help
cli-help:
	agent-memory --help

# Install pre-commit hooks
pre-commit-install:
	pip install pre-commit
	pre-commit install

# Run pre-commit on all files
pre-commit-run:
	pre-commit run --all-files