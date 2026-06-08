from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from literoom.api import create_app
    from literoom.config import load_config
    from literoom.metadata import manifest

    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment]
    create_app = None  # type: ignore[assignment]
    load_config = None  # type: ignore[assignment]
    manifest = None  # type: ignore[assignment]


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi is not installed")
class RecoveryTests(unittest.TestCase):
    def _write_config(self, config_path: Path, db_path: Path, managed: Path, derived: Path, root: Path, sources: list[Path]) -> None:
        config_path.write_text(
            "\n".join(
                [
                    "workspace_root: .",
                    "sources:",
                    *(f"  - {source}" for source in sources),
                    "paths:",
                    f"  db_path: {db_path}",
                    f"  managed_library_dir: {managed}",
                    f"  derivatives_dir: {derived}",
                    f"  logs_dir: {root / 'logs'}",
                    f"  temp_dir: {root / 'tmp'}",
                ]
            ),
            encoding="utf-8",
        )

    def test_first_startup_keeps_import_folders_but_clears_database_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_path = root / "manifest.sqlite"
            managed = root / "library"
            derived = root / "derived"
            import_dir = root / "imports"
            managed.mkdir()
            derived.mkdir()
            import_dir.mkdir()
            manifest.init_db(db_path)
            job_id = manifest.create_job(db_path, "ingest", {"sources": [str(import_dir)]})
            manifest.start_job(db_path, job_id)
            manifest.complete_job(db_path, job_id, {"rows_upserted": 1})

            config_path = root / "config.yaml"
            self._write_config(config_path, db_path, managed, derived, root, [import_dir])
            client = TestClient(create_app(config_path))

            response = client.get("/app/actions/first-startup/confirm", follow_redirects=False)

            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/")
            self.assertEqual(manifest.list_jobs(db_path), [])
            loaded, _ = load_config(config_path)
            self.assertEqual(loaded.sources, [])
            self.assertTrue(import_dir.exists())
            page = client.get("/")
            self.assertIn("Welcome to Literoom", page.text)
            self.assertIn("Choose your folders to begin", page.text)
            self.assertNotIn("Startup", page.text)


if __name__ == "__main__":
    unittest.main()
