codespell:
	codespell -L astroid -w $(shell git ls-files)

codespell-check:
	codespell -L astroid $(shell git ls-files)

flake8:
	flake8 $(shell git ls-files | grep .py)

format:
	isort $(shell git ls-files | grep .py)
	black $(shell git ls-files | grep .py)

format-check:
	isort --check-only $(shell git ls-files | grep .py)
	black --diff --check $(shell git ls-files | grep .py)

docs-check:
	sphinx-build -b html --keep-going docs/source docs/build/html

lint:
	make format-check codespell-check flake8 docs-check
