format:
	poetry run ruff format .
	poetry run ruff check --fix-only .

lint:
	poetry run ruff check .
	poetry run ruff format --check .

check-type:
	poetry run pyright .

test:
	make unit-test
	make acceptance-test

unit-test:
	poetry run pytest -v  tests/unit

acceptance-test:
	poetry run pytest -v tests/acceptance

check-ci:
	make lint
	make check-type
	make unit-test
