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
  point at a file git actually ships.  That is exactly what
  ``tests/seed/test_project_identity.py`` asserts, only earlier.
* the rest of ``plans`` is historical prose (2,400+ dead links accumulated
  since 2026-09) and is not retro-fixed.  Two rules are still absolute there:
  a ``.jsonl`` link can never resolve, because ``.gitignore`` excludes every
  ``*.jsonl`` in this repository, and a link to any other ``.gitignore``-hit
  path resolves only on the machine that produced the artifact.  Missing-file
  reporting, the noisy legacy class, stays limited to ``plans/active``.

The strict predicate is "tracked by git", **not** "exists on disk".  A working
tree carries untracked files, so an existence check passes here and fails in a
clean clone -- the same trap as byte equality across machines: the reading is
true only under undocumented local conditions.  The first CI run of this guard
found 46 such links that a local existence check had called fine.

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


def tracked_files() -> set[str]:
    """Every repository-relative path git ships in this commit."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return {name for name in listed.split("\0") if name}


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
        # Inline code is prose about a link, not a link: documents quote broken
        # link syntax on purpose (this file's own ledger entry does).  Drop the
        # code spans before matching so quoting a violation is not one.
        line = re.sub(r"`[^`]*`", "", line)
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
    tracked = tracked_files()

    violations: list[str] = []
    for path, number, target, destination in resolved:
        strict = path.is_relative_to(ROOT / STRICT_SCOPE)
        where = f"{path.relative_to(ROOT).as_posix()}:{number}"
        if target.lower().endswith(".jsonl"):
            violations.append(f"{where}: link to .jsonl (never shipped): {target}")
            continue
        # A gitignored target is the same broken promise in a different costume:
        # it resolves on the author's machine and nowhere else, so it is rejected
        # everywhere, not just in the strict scope.
        if destination.is_relative_to(ROOT):
            relative = str(destination.relative_to(ROOT)).replace("\\", "/")
            if relative in ignored:
                violations.append(f"{where}: link target is gitignored: {target}")
                continue
        if strict and not destination.is_relative_to(ROOT):
            continue
        if strict and not destination.is_dir():
            if not destination.exists():
                violations.append(f"{where}: link target does not exist: {target}")
            elif relative not in tracked:
                violations.append(
                    f"{where}: link target is untracked (exists only in this "
                    f"working tree): {target}"
                )

    for violation in violations:
        print(violation)
    if violations:
        print(f"\n{len(violations)} broken evidence link(s) under {', '.join(scopes)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
