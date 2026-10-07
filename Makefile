.PHONY: test smoke lint

test:
	python -m pytest

smoke:
	python -m postman.scripts.train --steps 100 --output outputs/smoke.npz

lint:
	ruff check src tests
