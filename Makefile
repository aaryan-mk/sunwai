.PHONY: install format lint typecheck test test-integration check api lipsync-worker clean

install:
	uv sync --extra dev --frozen

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/sunwai

test:
	uv run pytest tests/unit tests/contracts -q

test-integration:
	uv run pytest tests/integration -q

check: lint typecheck test

api:
	uv run sunwai serve --host 127.0.0.1 --port 8000

# Usage: make lipsync-worker ROOT=./var/jobs
ROOT ?= ./var/jobs
lipsync-worker:
	uv run sunwai worker lipsync --root $(ROOT)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
