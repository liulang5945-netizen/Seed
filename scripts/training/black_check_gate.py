"""In-process black gate for the local pre-commit hook.

Black's process pool can hang after printing its result in this managed
environment.  This checker uses the same Mode as pyproject.toml
(line-length=100, py312) and exits non-zero if any file would be reformatted.
"""

import sys
from pathlib import Path

import black
from black import TargetVersion

MODE = black.Mode(line_length=100, target_versions={TargetVersion.PY312})


def main(paths: list[str]) -> int:
    bad: list[str] = []
    for raw in paths:
        path = Path(raw)
        source = path.read_text(encoding="utf-8")
        try:
            black.format_file_contents(source, fast=True, mode=MODE)
        except black.NothingChanged:
            continue
        except Exception as exc:  # pragma: no cover - reported as a gate failure
            print(f"black error: {path}: {exc}")
            bad.append(str(path))
            continue
        print(f"would reformat {path}")
        bad.append(str(path))
    if bad:
        return 1
    print("black in-process check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
