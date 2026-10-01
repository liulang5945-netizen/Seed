"""Frozen source markers vs. the formatter: does `black` still respect them?

Why this module exists
----------------------
A *frozen marker* is a string that an audit script asserts still appears, **as
contiguous text**, inside some other Python file.  The N2 disposition audit does this
for its judgement sites (``JUDGEMENT_SITES``: one ``path`` plus one ``marker`` each).
Because the assertion is textual rather than structural, a frozen marker is an
implicit API surface of the file it points at -- and formatters are its natural enemy.

On 2026-09-30 a routine ``black`` pass split the call carrying site **J12**
(``"stop": "all_members_exhausted"})``) across ten lines.  The marker was no longer
contiguous, the audit could not find it, and ``review_checks_passed`` silently flipped
to ``False``.  Six contract tests went red on what looked like a cosmetic commit.

The mitigation was a ``[tool.black] extend-exclude`` entry.  That entry was itself
broken for a full round: ``extend-exclude`` is **one regex**, and the two alternatives
had been written on separate lines, which makes the first branch require a trailing
newline and therefore match nothing.  ``black --check scripts/training/`` still
reported the file as "would reformat" -- the protection was decorative, and CI would
have re-broken J12 exactly as before.

So this module re-runs the whole conflict check from the live tree, every time:

* discover every ``(path, marker)`` pair declared in any module-level frozen table;
* confirm each marker is present in its target today (a missing one is **drift**);
* run ``black`` **in memory** over the target and re-test the marker (a marker that
  survives today but not after formatting is **formatter-fragile**);
* require every fragile target to be excluded from black, using black's own matcher
  rather than a hand-rolled regex -- so a malformed pattern is caught, not admired.

Exit code is non-zero on drift or on an unprotected fragile marker.  The point is not
to report; it is to make "the formatter and the audit guard disagree" unforgivable.

Read-only: nothing on disk is formatted, rewritten, or deleted.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import black
from black.files import path_is_excluded

# Python 3.10 has no stdlib tomllib; tomli is its upstream implementation.  pyproject
# still declares requires-python >=3.10, so this script must keep running on 3.10 --
# but H19c (2026-09-30) dropped the CI 3.10 leg, so that path is no longer exercised by
# the gate this script backs.  Do not read a green run as proof the 3.10 branch works.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_frozen_marker_formatter_conflict.json"
FORMAT = "taiji-frozen-marker-formatter-conflict-v1"

#: Frozen markers only ever live in code; reports/ and output/ are evidence, not source.
SCAN_ROOTS: tuple[str, ...] = (
    "scripts",
    "taiji",
    "seed",
    "seed_platform",
    "api",
    "tests",
    "neuroplex",
    "instruments",
)
SKIP_DIR_NAMES: frozenset[str] = frozenset(
    {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}
)

#: A module-level UPPER_CASE table whose name implies a sealed review surface.
TABLE_NAME_HINTS: tuple[str, ...] = ("SITES", "MARKER", "EXPECTED", "FROZEN")


def _black_config() -> dict[str, Any]:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    config: dict[str, Any] = data.get("tool", {}).get("black", {})
    return config


def _is_scannable(path: Path) -> bool:
    parts = path.relative_to(PROJECT_ROOT).parts
    return not any(part in SKIP_DIR_NAMES or part.startswith("direct-") for part in parts)


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _pairs_from_table(node: ast.AST) -> Iterator[tuple[str, str]]:
    """Yield (path, marker) from dict literals that carry BOTH keys.

    Requiring both keys is what separates a real marker from prose: the disposition
    table also holds ``safe_because`` explanations, and those are long strings full of
    punctuation that are not assertions about source text.
    """
    for child in ast.walk(node):
        if not isinstance(child, ast.Dict):
            continue
        found: dict[str, str] = {}
        for key, value in zip(child.keys, child.values, strict=False):
            name = _literal_str(key) if key is not None else None
            text = _literal_str(value)
            if name in {"path", "marker"} and text is not None:
                found[name] = text
        if "path" in found and "marker" in found:
            yield found["path"], found["marker"]


def discover_frozen_markers() -> list[dict[str, str]]:
    """Every (declaring audit, target path, marker) triple in the live tree."""
    triples: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for root in SCAN_ROOTS:
        base = PROJECT_ROOT / root
        if not base.is_dir():
            continue
        for script in sorted(base.rglob("*.py")):
            if not _is_scannable(script):
                continue
            try:
                text = script.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "marker" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            for node in tree.body:
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                name = getattr(target, "id", "")
                if not name.isupper() or not any(hint in name for hint in TABLE_NAME_HINTS):
                    continue
                for marker_path, marker in _pairs_from_table(node.value):
                    if "\n" in marker:
                        continue
                    declared = script.relative_to(PROJECT_ROOT).as_posix()
                    key = (declared, marker_path, marker)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append(
                        {
                            "declared_in": declared,
                            "table": name,
                            "target": marker_path,
                            "marker": marker,
                        }
                    )
    return triples


def black_mode() -> black.Mode:
    """The black configuration the project actually uses, read from pyproject.toml."""
    config = _black_config()
    line_length = int(config.get("line-length", 88))
    target_versions = {
        black.TargetVersion[str(version).upper()]
        for version in config.get("target-version", [])
        if str(version).upper() in black.TargetVersion.__members__
    }
    return black.Mode(line_length=line_length, target_versions=target_versions)


def formatted_source(source: str, mode: black.Mode) -> str | None:
    """What black would write, without writing it.  None when black cannot parse."""
    try:
        return black.format_str(source, mode=mode)
    except black.NothingChanged:
        return source
    except Exception:
        return None


def exclusion_patterns() -> list[re.Pattern[str]]:
    """Compiled exclude / extend-exclude / force-exclude, in black's own order."""
    config = _black_config()
    patterns: list[re.Pattern[str]] = []
    for key in ("exclude", "extend-exclude", "force-exclude"):
        raw = config.get(key)
        if raw:
            patterns.append(re.compile(str(raw)))
    return patterns


def excluded_from_black(path: Path) -> bool:
    """Ask black's own matcher whether it would skip this file.

    Deliberately not a hand-eyeballed ``re.search``: the round this guard exists for
    was caused by a pattern that looked right and matched nothing.  Black matches
    against ``"/" + path.relative_to(root).as_posix()``, so that is what we hand it.
    """
    root_relative = "/" + path.relative_to(PROJECT_ROOT).as_posix()
    return any(path_is_excluded(root_relative, pattern) for pattern in exclusion_patterns())


def classify(triples: Iterable[dict[str, str]], mode: black.Mode) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for triple in triples:
        target = PROJECT_ROOT / triple["target"]
        record = dict(triple)
        if not target.is_file():
            record["status"] = "missing_target"
            records.append(record)
            continue
        source = target.read_text(encoding="utf-8")
        record["excluded_from_black"] = excluded_from_black(target)
        if triple["marker"] not in source:
            record["status"] = "drift"
            records.append(record)
            continue
        formatted = formatted_source(source, mode)
        if formatted is None:
            record["status"] = "black_cannot_parse"
            records.append(record)
            continue
        record["marker_survives_formatting"] = triple["marker"] in formatted
        if record["marker_survives_formatting"]:
            record["status"] = "ok"
        elif record["excluded_from_black"]:
            record["status"] = "fragile_but_excluded"
        else:
            record["status"] = "fragile_unprotected"
        records.append(record)

    counts: dict[str, int] = {}
    for record in records:
        status = str(record["status"])
        counts[status] = counts.get(status, 0) + 1

    failing = [
        r for r in records if r["status"] in {"drift", "fragile_unprotected", "missing_target"}
    ]
    return {
        "format": FORMAT,
        "black_version": black.__version__,
        "black_line_length": mode.line_length,
        "exclude_pattern_count": len(exclusion_patterns()),
        "records": records,
        "counts": counts,
        "total_markers": len(records),
        "passed": not failing,
        "failures": failing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    report = classify(discover_frozen_markers(), black_mode())
    if not args.no_write:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "total_markers": report["total_markers"],
                "counts": report["counts"],
                "passed": report["passed"],
                "failures": [
                    {
                        "declared_in": r.get("declared_in"),
                        "target": r.get("target"),
                        "status": r.get("status"),
                        "marker": str(r.get("marker"))[:60],
                    }
                    for r in report["failures"]
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
