"""Fail when a document links to a file git does not ship.

Every link in the plan tree is a promise that a reader can open the evidence.
A link to a gitignored artifact keeps that promise only on the author's machine:
it resolves locally and 404s for anyone who clones.  The same mistake has now
reached `main` twice (a roadmap link to a `*.jsonl` that `.gitignore` excludes),
and the only thing that caught it was a full-suite test written after the fact.

This runs on the commit path (pre-commit) and in the fast CI layer, so the
mistake is named at `file:line` before it lands instead of after a one-hour
regression.

Scope is deliberately two-tiered:

* ``plans/active`` is the live contract -- every relative link there must
  resolve to a tracked file.  That is exactly what
  ``tests/seed/test_project_identity.py`` asserts, only earlier.
* the rest of ``plans`` is historical prose (2,400+ dead links accumulated
  since 2026-09) and is not retro-fixed.  Only one rule is absolute there:
  a ``.jsonl`` link can never resolve, because ``.gitignore`` excludes every
  ``*.jsonl`` in this repository.

Usage:
    python scripts/ci/check_doc_links.py [path ...]
Exit code 1 lists every violation; 0 means the checked links are honest.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
DEFAULT_SCOPES = ("plans",)
STRICT_SCOPE = Path("plans/active")


def tracked_markdown(paths: list[str]) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [ROOT / name for name in listed.split("\0") if name.endswith(".md")]


def ignored_paths(candidates: set[str]) -> set[str]:
    """Ask git which of these repository-relative paths .gitignore excludes."""
    if not candidates:
        return set()
    payload = "\n".join(sorted(candidates)).encode("utf-8")
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=ROOT,
        input=payload,
        capture_output=True,
    )
    return {
        line.strip()
        for line in result.stdout.decode("utf-8", errors="replace").splitlines()
        if line.strip()
    }


def links_in(path: Path) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in LINK.finditer(line):
            target = match.group(1).split("#", 1)[0].strip()
            if target and not target.startswith(SKIP_PREFIXES):
                found.append((number + 1, target))
    return found


def main(argv: list[str]) -> int:
    scopes = [item for item in argv if not item.startswith("-")] or list(DEFAULT_SCOPES)
    files = tracked_markdown(scopes)

    resolved: list[tuple[Path, int, str, Path]] = []
    for path in files:
        for number, target in links_in(path):
            resolved.append((path, number, target, (path.parent / target).resolve()))

    # Only paths inside the repository can be tracked or ignored; anything else
    # is a link out of the tree and is not this check's business.
    inside = {
        str(destination.relative_to(ROOT)).replace("\\", "/")
        for _, _, _, destination in resolved
        if destination.is_relative_to(ROOT)
    }
    ignored = ignored_paths(inside)

    violations: list[str] = []
    for path, number, target, destination in resolved:
        strict = path.is_relative_to(ROOT / STRICT_SCOPE)
        where = f"{path.relative_to(ROOT).as_posix()}:{number}"
        if target.lower().endswith(".jsonl"):
            violations.append(f"{where}: link to .jsonl (never shipped): {target}")
            continue
        if not strict:
            continue
        if not destination.exists():
            violations.append(f"{where}: link target does not exist: {target}")
            continue
        if destination.is_relative_to(ROOT):
            relative = str(destination.relative_to(ROOT)).replace("\\", "/")
            if relative in ignored:
                violations.append(f"{where}: link target is gitignored: {target}")

    for violation in violations:
        print(violation)
    if violations:
        print(f"\n{len(violations)} broken evidence link(s) under {', '.join(scopes)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
