"""The two style gates must cover the same trees, and the fork must cover neither.

Why this test exists
--------------------
Commit ``82042a2f6`` ("G1 fork 落地") dropped the entire upstream ``dsh`` repository into
``taiji-harness/`` -- 13,137 tracked files, 32 of them ``.py``.  Nothing was said to ruff
or black about it, and the fork ships no ``pyproject.toml`` or ``ruff.toml`` of its own,
so our rules applied to it verbatim.  Both CI steps silently changed meaning:

* ``ruff check .``  -> 50 errors, **all 50 inside the fork**, zero in our own code;
* ``black --check .`` -> 28 files "would reformat", 27 of them inside the fork.

Nobody noticed for a round, because the tool output named real files and looked like
ordinary debt rather than a boundary that had never been drawn.  Reformatting a fork we
intend to keep syncing with upstream would also guarantee a conflict on every merge.

The invariant enforced here is that ruff and black **exclude exactly the same top-level
trees**.  A tree can be ours (excluded by neither) or declared third-party/artifact
(excluded by both); what is not allowed is one gate absorbing it while the other does
not, because that is the shape this incident took -- and the shape of every future
"why is only one of the two gates red" question.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

# Python 3.10 has no stdlib tomllib; tomli is its upstream implementation.  Two things
# need this guard.  (1) Import placement: pyproject sets ruff target-version = "py310",
# so ruff classifies tomllib as third-party and an unguarded `import tomllib` inside the
# stdlib block trips I001 -- that alone made this file fail `ruff check .`.  (2) Runtime:
# requires-python is still >=3.10, so collection would ImportError there.  H19c
# (2026-09-30) dropped the CI 3.10 leg, so (2) is no longer covered by CI.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def _tracked_files() -> list[str]:
    return (
        subprocess.run(
            ["git", "ls-files"],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        .stdout.replace("\\", "/")
        .splitlines()
    )


def _ruff_excludes() -> list[str]:
    return [str(item) for item in _config()["tool"]["ruff"]["exclude"]]


def _black_pattern() -> str:
    return str(_config()["tool"]["black"]["extend-exclude"])


def _ruff_excludes_name(name: str) -> bool:
    """Does ruff's exclude list skip this top-level directory?"""
    return any(fnmatch.fnmatch(name, entry.rstrip("/")) for entry in _ruff_excludes())


def _black_excludes_name(name: str) -> bool:
    """Does black's regex skip this top-level directory?

    Uses black's own matching shape: it searches ``"/" + relative_posix_path``, and a
    directory is reported as ``"/" + relative + "/"``.
    """
    return bool(re.search(_black_pattern(), f"/{name}/"))


def test_the_two_style_gates_exclude_the_same_top_level_trees() -> None:
    names = sorted({line.split("/", 1)[0] for line in _tracked_files() if "/" in line})
    disagreeing = [
        f"{name} (ruff={_ruff_excludes_name(name)}, black={_black_excludes_name(name)})"
        for name in names
        if _ruff_excludes_name(name) != _black_excludes_name(name)
    ]
    assert not disagreeing, (
        "ruff and black must absorb the same trees; a tree one gate lints and the "
        "other formats is how taiji-harness silently reddened both CI steps: "
        + ", ".join(disagreeing)
    )


def test_the_fork_is_declared_to_both_style_gates() -> None:
    """The concrete regression from 82042a2f6, pinned rather than inferred."""
    assert _ruff_excludes_name("taiji-harness"), "the vendored fork is inside `ruff check .`"
    assert _black_excludes_name("taiji-harness"), "the vendored fork is inside `black --check .`"


def test_first_party_packages_stay_inside_both_style_gates() -> None:
    """Declaring a vendored tree must not widen a pattern until our own code falls out.

    Guards the opposite direction from the fork test: a exclude regex broadened to
    swallow `taiji-harness` must not also swallow `taiji`.
    """
    for package in ("taiji", "seed", "seed_platform", "api", "neuroplex", "scripts", "tests"):
        assert not _ruff_excludes_name(package), package
        assert not _black_excludes_name(package), package


def test_black_extend_exclude_is_a_single_line_regex() -> None:
    """`extend-exclude` is one regex; a multi-line value breaks every branch silently.

    The J12 protection was written across three lines, which made the first alternative
    require a trailing newline and therefore match no path at all -- the file stayed in
    scope for `black --check .` and the guard comment describing it was a fiction.
    """
    assert "\n" not in _black_pattern().strip()


def test_no_stray_python_files_at_repo_root() -> None:
    """A root-level .py is invisible to subtree gates and breaks the disk/git face test.

    ``tests/seed/test_platform_boundary.py`` compares a filesystem walk against the git
    face and tolerates extras only under a skipped dir or ``output/``; a root-level .py
    fails it even when gitignored.  Scratch scripts belong in ``output/`` or nowhere.
    """
    strays = {line for line in _tracked_files() if "/" not in line and line.endswith(".py")}
    assert strays <= {"conftest.py", "setup.py"}, strays
