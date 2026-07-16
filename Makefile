.PHONY: run run-mcp test lint type-check audit clean install dev-install

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

run:
	python -m deeplens.main

run-mcp:
	python -m deeplens.mcp.server

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=deeplens --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

type-check:
	mypy src/deeplens/

audit:
	pip-audit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info
