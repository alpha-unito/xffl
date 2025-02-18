
format:
	isort xffl/ examples/
	black xffl/ examples/

format-check:
	isort --check-only  xffl/ examples/
	black --diff --check xffl/ examples/