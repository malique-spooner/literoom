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
STAGING = DIST / "_dmg_staging"
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


def build_dmg() -> Path:
    _clean([STAGING])
    STAGING.mkdir(parents=True, exist_ok=True)
    shutil.copytree(APP_BUNDLE, STAGING / APP_BUNDLE.name)
    applications_link = STAGING / "Applications"
    if applications_link.exists() or applications_link.is_symlink():
        applications_link.unlink()
    applications_link.symlink_to("/Applications")
    dmg_path = DIST / f"{APP_NAME}.dmg"
    if dmg_path.exists():
        dmg_path.unlink()
    try:
        _run([
            "/usr/bin/hdiutil",
            "create",
            "-volname",
            APP_NAME,
            "-srcfolder",
            str(STAGING),
            "-ov",
            "-format",
            "UDZO",
            str(dmg_path),
        ])
        return dmg_path
    except subprocess.CalledProcessError:
        zip_path = DIST / f"{APP_NAME}.zip"
        if zip_path.exists():
            zip_path.unlink()
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=DIST, base_dir=APP_BUNDLE.name)
        return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Literoom macOS app bundle and DMG.")
    parser.add_argument("--skip-dmg", action="store_true", help="Build only the .app bundle.")
    args = parser.parse_args()

    _clean([BUILD, APP_BUNDLE, STAGING, DIST / f"{APP_NAME}.dmg"])
    build_app()
    if not args.skip_dmg:
        artifact = build_dmg()
    print(f"Built {APP_BUNDLE}")
    if not args.skip_dmg:
        print(f"Built {artifact}")


if __name__ == "__main__":
    main()
