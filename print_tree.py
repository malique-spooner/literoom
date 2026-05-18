#!/usr/bin/env python3
"""
print_tree.py — print a clean directory tree

Usage:
  python print_tree.py             # current directory
  python print_tree.py src         # specific path
  python print_tree.py -a          # include hidden files/dirs
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

# Folders/files to ignore by default
DEFAULT_IGNORE_DIRS = {
    "__pycache__", ".git", ".hg", ".svn",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", ".idea", ".vscode",
    "build", "dist",
}
DEFAULT_IGNORE_SUFFIX_DIRS = {".egg-info"}  # any dir ending with these
DEFAULT_IGNORE_FILES = {".DS_Store", "Thumbs.db"}
DEFAULT_IGNORE_FILE_SUFFIXES = {".pyc", ".pyo", ".pyd"}

def should_ignore_dir(name: str, include_hidden: bool) -> bool:
    if not include_hidden and name.startswith("."):
        return True
    if name in DEFAULT_IGNORE_DIRS:
        return True
    return any(name.endswith(suf) for suf in DEFAULT_IGNORE_SUFFIX_DIRS)

def should_ignore_file(name: str, include_hidden: bool) -> bool:
    if not include_hidden and name.startswith("."):
        return True
    if name in DEFAULT_IGNORE_FILES:
        return True
    return any(name.endswith(suf) for suf in DEFAULT_IGNORE_FILE_SUFFIXES)

def tree(root: Path, include_hidden: bool = False) -> None:
    root = root.resolve()
    print(str(root))

    def _iter(path: Path, prefix: str = ""):
        # List children with directories first, then files; both sorted case-insensitively
        try:
            entries = list(path.iterdir())
        except PermissionError:
            print(prefix + "└── [Permission denied]")
            return

        dirs = sorted([p for p in entries if p.is_dir() and not should_ignore_dir(p.name, include_hidden)],
                      key=lambda p: p.name.lower())
        files = sorted([p for p in entries if p.is_file() and not should_ignore_file(p.name, include_hidden)],
                       key=lambda p: p.name.lower())

        children = dirs + files
        for i, p in enumerate(children):
            is_last = i == len(children) - 1
            branch = "└── " if is_last else "├── "
            print(prefix + branch + p.name)
            if p.is_dir():
                extension = "    " if is_last else "│   "
                _iter(p, prefix + extension)

    _iter(root)

def main():
    parser = argparse.ArgumentParser(description="Print a clean directory tree.")
    parser.add_argument("path", nargs="?", default=".", help="Root path (default: current directory)")
    parser.add_argument("-a", "--all", action="store_true", help="Include hidden files/dirs")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        parser.error(f"path not found: {root}")

    tree(root, include_hidden=args.all)

if __name__ == "__main__":
    main()
