"""Watch a running R2 alignment pilot without touching the training entry.

`train_taiji_r2_aligned.py` prints nothing until it finishes (it only writes the
report at the end), so a background run is otherwise unobservable.  This watcher
gathers every signal that *is* available from outside the process:

* the live ``python.exe`` processes and their resident memory;
* the redirected log file (size + tail);
* the run's output directory (checkpoints / preflight / report appearing).

Usage:
    python -X utf8 scripts/training/watch_p3b_v2_pilot.py
    python -X utf8 scripts/training/watch_p3b_v2_pilot.py --loop --interval 20
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = PROJECT_ROOT / "logs" / "p3b_v2_pilot_20260917.log"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "p3b_v2_pilot_20260917"


def _python_processes() -> list[dict[str, Any]]:
    """Live python.exe processes (PID + resident memory) via tasklist."""

    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    processes: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        parts = [cell.strip('"') for cell in line.split('","')]
        if len(parts) < 5 or not parts[0].lower().startswith("python"):
            continue
        processes.append({"pid": parts[1], "memory": parts[4]})
    return processes


def _tail(path: Path, limit: int = 700) -> str:
    if not path.is_file():
        return "(log not created yet)"
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return "(log is empty -- the training entry prints nothing until it finishes)"
    return text[-limit:]


def snapshot(log: Path = DEFAULT_LOG, output_dir: Path = DEFAULT_OUTPUT_DIR) -> str:
    lines = [f"=== pilot snapshot @ {datetime.datetime.now().strftime('%H:%M:%S')} ==="]

    processes = _python_processes()
    if processes:
        lines.append(f"python processes: {len(processes)}")
        for proc in processes:
            lines.append(f"  pid {proc['pid']:>8}  mem {proc['memory']}")
    else:
        lines.append("python processes: NONE  <-- the run has finished or died")

    if log.is_file():
        stat = log.stat()
        lines.append(
            f"log: {log.name}  {stat.st_size} B  (mtime {datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')})"
        )
        lines.append("  tail: " + _tail(log).replace("\n", " | ")[:400])
    else:
        lines.append(f"log: {log}  (missing)")

    lines.append(f"output dir: {output_dir}")
    if output_dir.exists():
        files = sorted(p for p in output_dir.rglob("*") if p.is_file())
        if files:
            for path in files:
                stat = path.stat()
                lines.append(
                    f"  {datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%H:%M:%S')}"
                    f"  {stat.st_size:>10} B  {path.relative_to(output_dir)}"
                )
        else:
            lines.append("  (empty -- still in the preflight / training phase)")
    else:
        lines.append("  (not created yet -- the report is only written at the very end)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch the running P3b-v2 pilot")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--loop", action="store_true", help="keep printing snapshots")
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args(argv)

    while True:
        print(snapshot(args.log, args.output_dir))
        sys.stdout.flush()
        if not args.loop:
            return 0
        time.sleep(max(5.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
