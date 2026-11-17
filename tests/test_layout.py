import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_LEVEL_FOLDERS = (".github", "setup", "src", "tests")
TOP_LEVEL_FILES = (
    ".gitattributes",
    ".gitignore",
    ".lint-baseline.json",
    ".python-version",
    "CITATION.cff",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "features.json",
    "pyproject.toml",
    "uv.lock",
)


def listed_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p and (ROOT / p).exists()]


def test_every_file_sits_in_a_known_top_level_place():
    misplaced = [
        p
        for p in listed_files()
        if ("/" in p and p.split("/", 1)[0] not in TOP_LEVEL_FOLDERS)
        or ("/" not in p and p not in TOP_LEVEL_FILES)
    ]
    assert not misplaced, (
        f"files outside the known layout: {misplaced}. Move each under one of the top-level"
        f" folders {list(TOP_LEVEL_FOLDERS)}, or, for a new top-level file, add it to"
        f" TOP_LEVEL_FILES {list(TOP_LEVEL_FILES)} in tests/test_layout.py."
    )
