from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from literoom.cli import app


class CliTests(unittest.TestCase):
    def test_desktop_command_invokes_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.yaml"
            runner = CliRunner()
            with patch("literoom.cli.launch_desktop_app") as launcher:
                result = runner.invoke(
                    app,
                    [
                        "desktop",
                        "--config",
                        str(config_path),
                        "--host",
                        "0.0.0.0",
                        "--port",
                        "9000",
                        "--no-open-browser",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            launcher.assert_called_once()
            args, kwargs = launcher.call_args
            self.assertEqual(args[0], config_path)
            self.assertEqual(kwargs["host"], "0.0.0.0")
            self.assertEqual(kwargs["port"], 9000)
            self.assertFalse(kwargs["open_browser"])


if __name__ == "__main__":
    unittest.main()
