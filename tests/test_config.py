from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from literoom.config import AppConfig, DEFAULT_WORKSPACE_ROOT, load_config, save_config, write_default_config


class ConfigTests(unittest.TestCase):
    def test_write_and_load_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            sandbox_root = Path(tmp) / "ssd"
            sandbox_root.mkdir()
            normalize = lambda value: os.path.realpath(str(value))
            with mock.patch("literoom.config.DEFAULT_WORKSPACE_ROOT", sandbox_root):
                write_default_config(config_path)
                config, resolved = load_config(config_path)
                expected_root = sandbox_root.resolve()
                resolved_root = config.resolve_root(resolved)

                self.assertEqual(resolved, config_path.resolve())
                self.assertEqual(normalize(resolved_root), normalize(expected_root))
                self.assertEqual(normalize(config.db_path(resolved)), normalize(expected_root / ".literoom/manifest.sqlite"))
                self.assertEqual(normalize(config.managed_library_dir(resolved)), normalize(expected_root / "library"))
                self.assertEqual(normalize(config.derivatives_dir(resolved)), normalize(expected_root / "previews"))

                config.ensure_workspace_dirs(resolved)
                self.assertTrue((expected_root / "imports").exists())
                self.assertTrue((expected_root / "library").exists())
                self.assertTrue((expected_root / "previews").exists())
                self.assertTrue((expected_root / "logs").exists())
                self.assertTrue((expected_root / "tmp").exists())
                self.assertTrue((expected_root / ".literoom" / "cache").exists())
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
