#!/usr/bin/env python3
"""Check relative Markdown links in the tracked release selection."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    selected = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    failures: list[str] = []
    for relative in selected:
        path = ROOT / relative
        if path.suffix.lower() != ".md" or not path.exists():
            continue
        for target in pattern.findall(path.read_text(encoding="utf-8", errors="replace")):
            clean = target.strip().strip("<>").split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            if not (path.parent / clean).resolve().exists():
                failures.append(f"{relative} -> {target}")
    if failures:
        raise SystemExit("Broken relative links:\n" + "\n".join(failures))
    print("Tracked Markdown links OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
