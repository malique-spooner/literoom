from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "Literoom"
APP_BUNDLE = DIST / f"{APP_NAME}.app"
ZIP_PATH = DIST / f"{APP_NAME}.zip"
LEGACY_COLLECT = DIST / APP_NAME
LEGACY_STAGING = DIST / "_dmg_staging"
DS_STORE = DIST / ".DS_Store"
PYINSTALLER_CONFIG_DIR = Path("/private/tmp/literoom-pyinstaller")


def _run(cmd: list[str]) -> None:
    env = dict(**os.environ)
    env["PYINSTALLER_CONFIG_DIR"] = str(PYINSTALLER_CONFIG_DIR)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def _clean(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def build_app() -> None:
    DIST.mkdir(exist_ok=True)
    PYINSTALLER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    pyinstaller = ROOT / ".venv" / "bin" / "pyinstaller"
    entry = ROOT / "src" / "literoom" / "bundle_entry.py"
    add_data = [
        "--add-data",
        f"{ROOT / 'literoom logo.png'}:.",
        "--add-data",
        f"{ROOT / 'literoom favicon.png'}:.",
    ]
    cmd = [
        str(pyinstaller),
        "--name",
        APP_NAME,
        "--windowed",
        "--noconfirm",
        "--clean",
        "--paths",
        str(ROOT / "src"),
        *add_data,
        str(entry),
    ]
    _run(cmd)


def build_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=DIST, base_dir=APP_BUNDLE.name)
    return ZIP_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Literoom macOS app bundle and zip archive.")
    parser.parse_args()

    _clean([BUILD, APP_BUNDLE, ZIP_PATH, LEGACY_COLLECT, LEGACY_STAGING, DIST / f"{APP_NAME}.dmg", DS_STORE])
    build_app()
    artifact = build_zip()
    _clean([LEGACY_COLLECT, DS_STORE])
    print(f"Built {APP_BUNDLE}")
    print(f"Built {artifact}")


if __name__ == "__main__":
    main()
