.PHONY: lint format

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
