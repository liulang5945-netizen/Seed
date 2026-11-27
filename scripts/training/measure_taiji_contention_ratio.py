"""竞争核对照比例尺（DEBT-G94 的结清前置，只读、不更新权重、零产品码）。

**为什么要有它**：本仓的墙钟数一直在两种完全不同的机器状态下被并列引用，而这两件事之前没有一枚数
把它们分开过——㊵-670 那份整族 1464.67 s 是在"两枚外来空转探针各钉住一枚核"的状态下测的。
G94 要求的结清前置是：**同一支命令在两种负载档下各测一次并出版差值**。本器做的就是那"加负载"的
一半（"减负载"那一半要把外来探针杀掉，归 owner）：同一支探针在「现状负载」与「现状＋一枚竞争核」
两种档下各跑 `--repeats` 次，出版逐次秒数、两档中位数与比值。

**探针默认走真实仪器**（`audit_taiji_console_glyph_encodability.py`，纯 AST、单线程、约 40 秒），
而不是另造一段合成忙循环——本仓要校准的就是"跑一支只读仪器要多久"这句话。

**负载档不是我这侧造的形容词**：每一趟探针前后各起一次 `read_taiji_machine_load.py` 的**真子进程**
取数（复用现成仪器，不在这里重抄它的取法），把它出版的 python 枚数／真在烧 CPU 的枚数／系统占用
并排进读数件。

**两条不许的写法**：①把「现状负载」那一档叫"无负载"——本机那两枚外来探针还在，本器测的是
**+1 枚竞争核的边际效应**，不是从 0 起；②拿比值去"修正"任何已入库的历史秒数（这是一枚敏感度，
不是一条校准曲线，只能用来决定"要不要在同集上比"）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROBE = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_console_glyph_encodability.py"
READER = PROJECT_ROOT / "scripts" / "training" / "read_taiji_machine_load.py"
MIN_REPEATS = 2
FORMAT = "taiji-contention-ratio-v1"
#: 竞争核的存活时长要盖住一整趟探针；给足上限，探针跑完就提前终止它（不靠睡满）。
COMPETITOR_LIFETIME_SECONDS = 900
COMPETITOR_CODE = (
    "import time\n"
    f"end = time.time() + {COMPETITOR_LIFETIME_SECONDS}\n"
    "while time.time() < end:\n"
    "    pass\n"
)


def _load_face() -> dict[str, Any]:
    """起一次现成的负载读数器（**不复用它的代码**：同一台仪器的取法只能有一处）。"""

    proc = subprocess.run(
        [sys.executable, str(READER), "--top", "1"],
        capture_output=True,
        check=False,
        cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"load reader rc={proc.returncode}: {proc.stderr[-200:]!r}")
    payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    return {
        "logical_cpus": payload["logical_cpus"],
        "system_cpu_percent": payload["system_cpu_percent"],
        "python_process_count": payload["python_process_count"],
        "python_processes_burning_cpu": payload["python_processes_burning_cpu"],
    }


def _probe_once(probe: Path, probe_args: list[str], scratch: Path) -> float:
    out = scratch / f"probe-{time.time_ns()}.json"
    cmd = [sys.executable, str(probe), *probe_args, "--out-report", str(out)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, check=False, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - started
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"probe rc={proc.returncode} artifact_exists={out.exists()}: {proc.stderr[-200:]!r}"
        )
    out.unlink()
    return elapsed


def measure(probe: Path, probe_args: list[str], repeats: int) -> dict[str, Any]:
    if repeats < MIN_REPEATS:
        raise ValueError(f"repeats must be >= {MIN_REPEATS}, got {repeats}")
    scratch = Path(tempfile.mkdtemp(prefix="taiji-contention-"))
    quiet: list[float] = []
    plus: list[float] = []
    loads: dict[str, list[dict[str, Any]]] = {"quiet": [], "plus_one": []}
    competitor_pids: list[int] = []
    try:
        for _ in range(repeats):
            before = _load_face()
            seconds = _probe_once(probe, probe_args, scratch)
            after = _load_face()
            quiet.append(seconds)
            loads["quiet"] += [before, after]
        for _ in range(repeats):
            #: 竞争核必须**被证实起过**：只记下 PID 不够，还要它在探针期间活着。
            proc = subprocess.Popen(
                [sys.executable, "-c", COMPETITOR_CODE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            competitor_pids.append(proc.pid)
            try:
                before = _load_face()
                seconds = _probe_once(probe, probe_args, scratch)
                after = _load_face()
                plus.append(seconds)
                loads["plus_one"] += [before, after]
                alive = proc.poll() is None
            finally:
                proc.terminate()
                proc.wait(timeout=30)
            if not alive:
                raise RuntimeError(f"competitor pid={proc.pid} died before the probe finished")
        median_quiet = statistics.median(quiet)
        median_plus = statistics.median(plus)
        if median_quiet <= 0.0:
            raise RuntimeError(f"probe measured zero/negative wall time: {median_quiet}")
        return {
            "format": FORMAT,
            "status": "ok",
            "probe": str(probe),
            "probe_args": probe_args,
            "repeats": repeats,
            "quiet_seconds": [round(one, 3) for one in quiet],
            "plus_one_seconds": [round(one, 3) for one in plus],
            "median_quiet_seconds": round(median_quiet, 3),
            "median_plus_one_seconds": round(median_plus, 3),
            "ratio_plus_over_quiet": round(median_plus / median_quiet, 4),
            "competitor_pids": competitor_pids,
            "competitor_lifetime_seconds": COMPETITOR_LIFETIME_SECONDS,
            "load_face_by_run": loads,
            "reading_limit": (
                "the quiet face is NOT an idle machine: foreign processes were still burning cores "
                "during it (see load_face_by_run). This is the marginal effect of adding one "
                "competing core to whatever was already running, on one single-threaded AST-bound "
                "probe only. It is a sensitivity, not a calibration curve, so it must not be used "
                "to restate any wall-clock number already in the ledger."
            ),
        }
    finally:
        for leftover in scratch.glob("probe-*.json"):
            leftover.unlink()
        scratch.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="contention ratio for one CPU-bound read-only probe"
    )
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE, help="script to time")
    parser.add_argument(
        "--probe-arg",
        action="append",
        default=[],
        dest="probe_args",
        help="extra argument for the probe (repeatable)",
    )
    parser.add_argument("--repeats", type=int, default=3, help="runs per face (>= 2)")
    parser.add_argument("--out-report", type=Path, required=True, help="where to write the JSON")
    args = parser.parse_args(argv)

    try:
        payload = measure(args.probe, list(args.probe_args), args.repeats)
    except (ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "measurement_failed", "error": str(error)}, ensure_ascii=True))
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "repeats": payload["repeats"],
                "median_quiet_seconds": payload["median_quiet_seconds"],
                "median_plus_one_seconds": payload["median_plus_one_seconds"],
                "ratio_plus_over_quiet": payload["ratio_plus_over_quiet"],
                "competitor_pids": payload["competitor_pids"],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
