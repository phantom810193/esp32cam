#!/usr/bin/env python3
"""Repository-wide validation helpers.

The GitHub Actions workflow uses this script to ensure that every tracked
text file respects some basic hygiene rules before code is merged.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_git_ls_files() -> List[Path]:
    """Return all files tracked by git as absolute paths."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [REPO_ROOT / Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]


def is_text_file(path: Path, sample_size: int = 4096) -> bool:
    """Heuristically determine whether *path* is a UTF-8 text file."""
    try:
        with path.open("rb") as fp:
            sample = fp.read(sample_size)
    except OSError:
        return False

    if b"\x00" in sample:
        return False

    if not sample:
        return True

    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False

    return True


def validate_text_file(path: Path, errors: List[str]) -> None:
    """Validate newline and trailing whitespace for *path*."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        errors.append(f"{path.relative_to(REPO_ROOT)}: not valid UTF-8 text")
        return

    if content and not content.endswith("\n"):
        errors.append(f"{path.relative_to(REPO_ROOT)}: missing trailing newline")

    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.rstrip() != line:
            errors.append(
                f"{path.relative_to(REPO_ROOT)}:{line_number}: trailing whitespace"
            )


def main() -> int:
    errors: List[str] = []
    files: Iterable[Path] = run_git_ls_files()

    for path in files:
        if not path.exists() or path.is_dir():
            errors.append(f"{path.relative_to(REPO_ROOT)}: missing from filesystem")
            continue

        if not is_text_file(path):
            continue

        validate_text_file(path, errors)

    if errors:
        print("Repository validation failed:\n")
        for error in errors:
            print(f"- {error}")
        return 1

    print("All tracked text files passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
