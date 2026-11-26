"""Shared report writer for instruments — DEBT-G115 修法③.

Every instrument used to write its own `write_text(...)`, and on this platform a missing
`newline=` silently produced CRLF artifacts, so a byte comparison between an `output/` original
and its `reports/` twin read as a full-file rewrite. One writer, line ending as a required-by-
presence parameter, atomic replace preserved from the previous call sites.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_text_report(
    path: Path, text: str, *, newline: str = "\n", overwrite: bool = False
) -> dict[str, int]:
    """Write `text` to `path` through a temporary file and report the resulting byte shape.

    Refuses an existing target unless `overwrite=True`; the refusal leaves the old bytes intact.
    """

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace an artifact you own"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline=newline)
    temporary.replace(path)
    raw = path.read_bytes()
    return {"bytes": len(raw), "cr": raw.count(b"\r"), "lf": raw.count(b"\n")}


def write_json_report(
    path: Path,
    payload: dict[str, Any],
    *,
    newline: str = "\n",
    overwrite: bool = False,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> dict[str, int]:
    """Serialize `payload` with a trailing newline, then hand the bytes to `write_text_report`."""

    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent) + "\n"
    return write_text_report(path, text, newline=newline, overwrite=overwrite)
