from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from literoom.source_analysis import analyze_sources


class SourceAnalysisTests(unittest.TestCase):
    def test_analyze_sources_reports_repeat_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "set-a" / "IMG_0001.jpg"
            second = root / "set-b" / "IMG_0001.jpg"
            clip = root / "set-c" / "clip.mp4"
            first.parent.mkdir(parents=True, exist_ok=True)
            second.parent.mkdir(parents=True, exist_ok=True)
            clip.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 64), color="white").save(first)
            Image.new("RGB", (64, 64), color="gray").save(second)
            clip.write_bytes(b"video-bytes")

            result = analyze_sources([root])

            self.assertEqual(result["media_files"], 3)
            self.assertEqual(result["image_files"], 2)
            self.assertEqual(result["video_files"], 1)
            self.assertEqual(result["duplicate_basename_groups"], 1)
            self.assertEqual(result["duplicate_basename_items"], 1)
            self.assertEqual(result["selected_sources"], 1)


if __name__ == "__main__":
    unittest.main()
