.PHONY: install lint format test ci

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest

ci: lint test
