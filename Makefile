codespell:
	codespell -w $(shell git ls-files)

codespell-check:
	codespell $(shell git ls-files)

flake8:
	flake8 examples/ tests/ xffl

format:
	isort examples/ tests/ xffl/
	black examples/ tests/ xffl/

format-check:
	isort --check-only  examples/ tests/ xffl/
	black --diff --check examples/ tests/ xffl/