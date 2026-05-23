from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_unifier.config import AppConfig, load_config, save_config, write_default_config


class ConfigTests(unittest.TestCase):
    def test_write_and_load_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_default_config(config_path)
            config, resolved = load_config(config_path)

            self.assertEqual(resolved, config_path.resolve())
            self.assertEqual(config.db_path(resolved), (config_path.parent / ".photo_unifier/manifest.sqlite").resolve())
            self.assertEqual(config.library_dir(resolved), (config_path.parent / "library").resolve())
            self.assertEqual(config.previews_dir(resolved), (config_path.parent / "previews").resolve())
            self.assertIsNotNone(config.tools.exiftool)
            self.assertIsNotNone(config.tools.ffmpeg)
            self.assertIsNotNone(config.tools.tesseract)

    def test_save_config_persists_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            config = AppConfig(
                workspace_root="/tmp/archive",
                sources=["/Volumes/Extreme SSD/MSp/Camera"],
            )
            save_config(config, config_path)
            loaded, _resolved = load_config(config_path)

            self.assertEqual(loaded.workspace_root, "/tmp/archive")
            self.assertEqual(loaded.sources, ["/Volumes/Extreme SSD/MSp/Camera"])


if __name__ == "__main__":
    unittest.main()
