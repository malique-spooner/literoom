from __future__ import annotations

import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_CACHE = Path(tempfile.gettempdir()) / "literoom-tests-cache"
TEST_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("PADDLEOCR_DISABLE_AUTO_LOGGING_CONFIG", "1")
os.environ.setdefault("MPLCONFIGDIR", str(TEST_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TEST_CACHE))
