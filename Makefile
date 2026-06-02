PYTHON := ./.venv/bin/python
PIP := ./.venv/bin/pip
CONFIG ?= literoom.local.yaml

.PHONY: setup doctor test status jobs init-config serve desktop package

setup:
	python3 -m venv .venv
	$(PIP) install -e .

doctor:
	$(PYTHON) -m literoom.cli doctor --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -t . -q

status:
	$(PYTHON) -m literoom.cli status --config $(CONFIG)

jobs:
	$(PYTHON) -m literoom.cli jobs --config $(CONFIG)

init-config:
	$(PYTHON) -m literoom.cli init-config --config $(CONFIG)

serve:
	$(PYTHON) -m literoom.cli serve-api --config $(CONFIG) --host 127.0.0.1 --port 8000

desktop:
	$(PYTHON) -m literoom.cli desktop --config $(CONFIG) --host 127.0.0.1 --port 8000

package:
	$(PYTHON) tools/package_macos.py
