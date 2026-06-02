from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from literoom.config import AppConfig, load_config, save_config, write_default_config


class ConfigTests(unittest.TestCase):
    def test_write_and_load_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            write_default_config(config_path)
            config, resolved = load_config(config_path)

            self.assertEqual(resolved, config_path.resolve())
            self.assertEqual(config.db_path(resolved), (config_path.parent / ".literoom/manifest.sqlite").resolve())
            self.assertEqual(config.managed_library_dir(resolved), (config_path.parent / "library").resolve())
            self.assertEqual(config.derivatives_dir(resolved), (config_path.parent / "previews").resolve())

            config.ensure_workspace_dirs(resolved)
            self.assertTrue((config_path.parent / "imports").exists())
            self.assertTrue((config_path.parent / "library").exists())
            self.assertTrue((config_path.parent / "previews").exists())
            self.assertTrue((config_path.parent / "logs").exists())
            self.assertTrue((config_path.parent / "tmp").exists())
            self.assertTrue((config_path.parent / ".literoom" / "cache").exists())
            self.assertIsNotNone(config.tools.exiftool)
            self.assertIsNotNone(config.tools.ffmpeg)
            self.assertIsNotNone(config.tools.tesseract)
            self.assertEqual(config.tools.face_model, "antelopev2")

            config.prepare_runtime_environment(resolved)
            self.assertIn("insightface", Path(os.environ["INSIGHTFACE_HOME"]).name)

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
