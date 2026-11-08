"""Process precheck: refuse to start a runner while a hung ``python -m black`` lives.

D16-13 recurrence (2026-11-08): thirteen ``python -m black`` processes had been
accumulating since 2026-11-07T22:08 and had burned about 236,931 CPU seconds,
all stuck after printing their result.  A runner that formats its own inputs
re-creates that pile while looking like "the probe is slow", so the runner now
refuses to start instead of silently adding to it.

Allowed formatter path: the in-process ``scripts/training/black_check_gate.py``.
``python -m black`` is banned outright (write, ``--check``, or "tidy up after").
"""

from __future__ import annotations

import json
import os
import subprocess

_BANNED = "-m black"


def _parse_ps(output: str) -> list[dict]:
    text = (output or "").strip()
    if not text:
        return []
    data = json.loads(text)
    if isinstance(data, dict):
        data = [data]
    return [
        {"pid": int(item["ProcessId"]), "command": item.get("CommandLine") or ""}
        for item in data
        if int(item["ProcessId"]) != os.getpid()
    ]


def black_processes() -> list[dict]:
    """Return ``[{"pid", "command"}]`` for live ``python -m black`` processes."""
    if os.name != "nt":
        out = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=False
        )
        rows = []
        for line in (out.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and _BANNED in " ".join(parts[1:]):
                rows.append({"pid": int(parts[0]), "command": " ".join(parts[1:])})
        return rows
    script = (
        "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m\\s+black' } "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    return _parse_ps(out.stdout)


def assert_no_black_processes(context: str = "runner") -> dict:
    """Raise when a banned formatter process is still alive, before any work starts."""
    found = black_processes()
    if found:
        listing = "; ".join(f"pid {row['pid']}: {row['command'][:120]}" for row in found)
        raise RuntimeError(
            f"{context} refused to start: {len(found)} `python -m black` process(es) still alive "
            f"(D16-13). Terminate them, then use scripts/training/black_check_gate.py. {listing}"
        )
    return {"checked": _BANNED, "found": 0}
