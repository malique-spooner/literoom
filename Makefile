PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
CONFIG ?= literoom.local.yaml

.PHONY: setup doctor test status jobs init-config serve

setup:
	python3 -m venv .venv
	$(PIP) install -e .

doctor:
	literoom doctor --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -t . -q

status:
	literoom status --config $(CONFIG)

jobs:
	literoom jobs --config $(CONFIG)

init-config:
	literoom init-config --config $(CONFIG)

serve:
	literoom serve-api --config $(CONFIG) --host 127.0.0.1 --port 8000
