.PHONY: quality style
PYTHON ?= python
check_dirs := src third_party/WAGLE setup.py

quality:
	$(PYTHON) -m ruff check $(check_dirs)
	$(PYTHON) -m ruff format --check $(check_dirs)

style:
	$(PYTHON) -m ruff check --fix $(check_dirs)
	$(PYTHON) -m ruff format $(check_dirs)
