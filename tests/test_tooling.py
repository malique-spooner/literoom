from __future__ import annotations

import unittest

from literoom.config import ToolPaths
from literoom.tooling import build_tool_stack_report


class ToolingTests(unittest.TestCase):
    def test_build_tool_stack_report(self):
        report = build_tool_stack_report(
            ToolPaths(
                exiftool="/opt/homebrew/bin/exiftool",
                ffmpeg="/opt/homebrew/bin/ffmpeg",
                tesseract="/opt/homebrew/bin/tesseract",
                vips="/opt/homebrew/bin/vips",
            )
        )
        self.assertIn("required", report)
        self.assertIn("optional", report)
        self.assertIn("locked_stack", report)
        self.assertTrue(report["required_ready"])
        self.assertTrue(report["required"]["exiftool"]["ready"])
        self.assertTrue(report["required"]["ffmpeg"]["ready"])
        self.assertTrue(report["required"]["tesseract"]["ready"])
        self.assertNotIn("PaddleOCR", report["locked_stack"]["items"])
        self.assertNotIn("Whisper", report["locked_stack"]["items"])
        self.assertIn("SQLite FTS + vector search", report["locked_stack"]["items"])
        self.assertIn("core_ready", report)
        self.assertIn("locked_stack_ready", report)


if __name__ == "__main__":
    unittest.main()
