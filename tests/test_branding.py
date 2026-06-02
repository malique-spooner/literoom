from __future__ import annotations

import unittest
from pathlib import Path

from literoom.config import DEFAULT_CONFIG_PATH


class BrandingTests(unittest.TestCase):
    def test_new_branding_is_visible(self):
        import literoom.api  # noqa: F401
        import literoom.cli  # noqa: F401
        import literoom.config  # noqa: F401
        import literoom.metadata.manifest  # noqa: F401
        import literoom.utils.location  # noqa: F401

        self.assertEqual(DEFAULT_CONFIG_PATH.name, "literoom.local.yaml")

    def test_no_old_product_name_remains_in_user_facing_files(self):
        root = Path(__file__).resolve().parents[1]
        tracked = [
            root / "README.md",
            root / "ROADMAP.md",
            root / "pyproject.toml",
            root / "Makefile",
        ]
        tracked.extend((root / "src").rglob("*.py"))
        old_name = "Photo " + "Unifier"
        for path in tracked:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(old_name, text, path.name)


if __name__ == "__main__":
    unittest.main()
