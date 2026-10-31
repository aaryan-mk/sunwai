.PHONY: install format lint typecheck test check api lipsync-worker config-map

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest

check: lint typecheck test

api:
	$(PY) -m sunwai.cli serve

lipsync-worker:
	$(PY) -m sunwai.cli worker lipsync --root "$${DUB_ROOT:-./jobs}" --id w1

config-map:
	$(PY) -m sunwai.cli config map
