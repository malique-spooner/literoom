from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from photo_unifier.phase_metadata import gather, manifest


class GatherTests(unittest.TestCase):
    def test_ingest_local_directory_indexes_media_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media_dir = root / "media"
            media_dir.mkdir()
            (media_dir / "IMG_20240102_030405.jpg").write_bytes(b"fake-jpeg")
            (media_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
            db_path = root / "manifest.sqlite"

            rows = gather.run([media_dir], db_path=db_path, batch_size=10)
            overview = manifest.get_overview(db_path)

            self.assertEqual(rows, 1)
            self.assertEqual(overview["assets_total"], 1)


if __name__ == "__main__":
    unittest.main()
