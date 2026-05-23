PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
CONFIG ?= photo-unifier.local.yaml

.PHONY: setup doctor test status jobs init-config serve

setup:
	python3 -m venv .venv
	$(PIP) install -e .

doctor:
	$(PYTHON) -m photo_unifier.cli doctor --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -t . -q

status:
	$(PYTHON) -m photo_unifier.cli status --config $(CONFIG)

jobs:
	$(PYTHON) -m photo_unifier.cli jobs --config $(CONFIG)

init-config:
	$(PYTHON) -m photo_unifier.cli init-config --config $(CONFIG)

serve:
	$(PYTHON) -m photo_unifier.cli serve-api --config $(CONFIG) --host 127.0.0.1 --port 8000
