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
ICON_FILE = BUILD / f"{APP_NAME}.icns"
LEGACY_COLLECT = DIST / APP_NAME
LEGACY_STAGING = DIST / "_dmg_staging"
DS_STORE = DIST / ".DS_Store"
LOGO_SOURCE = ROOT / "literoom logo.png"
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


def build_icon() -> Path:
    _clean([ICON_FILE])
    BUILD.mkdir(exist_ok=True)
    from PIL import Image

    Image.open(LOGO_SOURCE).save(ICON_FILE)
    return ICON_FILE


def build_app() -> None:
    DIST.mkdir(exist_ok=True)
    PYINSTALLER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    pyinstaller = ROOT / ".venv" / "bin" / "pyinstaller"
    entry = ROOT / "src" / "literoom" / "bundle_entry.py"
    icon = build_icon()
    add_data = [
        "--add-data",
        f"{LOGO_SOURCE}:.",
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
        "--icon",
        str(icon),
        "--paths",
        str(ROOT / "src"),
        *add_data,
        str(entry),
    ]
    _run(cmd)


def build_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    subprocess.run(["ditto", "-c", "-k", "--keepParent", str(APP_BUNDLE), str(ZIP_PATH)], check=True, cwd=ROOT)
    return ZIP_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Literoom macOS app bundle and zip archive.")
    parser.parse_args()

    _clean([BUILD, APP_BUNDLE, ZIP_PATH, LEGACY_COLLECT, LEGACY_STAGING, DIST / f"{APP_NAME}.dmg", DS_STORE])
    build_app()
    artifact = build_zip()
    _clean([APP_BUNDLE, LEGACY_COLLECT, DS_STORE])
    shutil.rmtree(BUILD, ignore_errors=True)
    print(f"Built {artifact}")


if __name__ == "__main__":
    main()
