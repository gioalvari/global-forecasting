# Makefile for global-forecasting project

.PHONY: all install clean format lint typecheck test run

VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python

all: lint typecheck test

$(VENV_DIR):
	uv venv

install: $(VENV_DIR) lock
	@echo "Installing dependencies..."
	uv pip sync requirements.txt

lock: $(VENV_DIR)
	@echo "Compiling lockfile..."
	uv pip compile pyproject.toml --extra dev -o requirements.txt

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.orig" -delete
	rm -f .coverage
	rm -rf htmlcov/

format: install
	@echo "Formatting code with Black and Ruff..."
	uv run python -m black src tests main.py
	uv run python -m ruff format src tests main.py

lint: install
	@echo "Linting code with Ruff..."
	uv run python -m ruff check src tests main.py

typecheck: install
	@echo "Running MyPy type checker..."
	uv run python -m mypy src tests main.py

test: install
	@echo "Running tests with Pytest..."
	uv run python -m pytest -ra -q --cov=global_forecasting --cov-report=term-missing

run: install
	@echo "Running the main application..."
	uv run python main.py
