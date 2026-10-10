"""机器负载读数器（DEBT-G94 修法①，只读、零算力、随墙钟报价出版）。

**为什么要有它**：本仓的墙钟数（训练档 `elapsed_seconds`、CAP 面单张 `seconds`、整族总秒数）一直
**裸报**，而同一条命令在"有人在抢核"与"没人抢核"的两种状态下不可能互比。㊵-668 在本机现读到
71 枚 python 进程，其中两枚命令行是 `python -` 的一次性探针正在空转（一枚自 2026-10-04 起累计
CPU 578,000 秒）⇒ 从这一格起，任何墙钟报价旁边都要有一枚**同轮**的负载读数，而这条纪律只有
"取它只要一条命令"时才真的会被执行。

**出版的是原始数，不是判断**：本器不给"负载高／低"的结论，也不设阈值（阈值住在仪器默认里是本仓
已登记的债，见 DEBT-G90）。它只并排出版四件事：系统 CPU 占用、python 进程枚数、其中**这一采样窗口内
真在烧 CPU** 的枚数、以及累计 CPU 秒数最高的几枚的身份（PID／秒数／启动时刻／命令行头）。
"该不该杀它们"归 owner，本器只把名字交出来。

**一枚 psutil 的坑已按实测绕开**：`proc.cpu_percent(None)` 在**首次调用**时返回的是"自进程创建以来"
的平均值，不是此刻占用——直接拿它当"此刻"会一次性把 71 枚全部报成在烧（假读数）。所以本器先
**priming** 一轮、再用同一次 `psutil.cpu_percent(interval=…)` 的阻塞窗口做间隔、第二轮取值，
两个数因此同源且真的是"窗口内"。

**自证在场**：`reader_counted_itself` 必须为 true——本器自己就是一枚 python 进程，连自己都数不到
就说明那枚 `python_process_count` 是假零（缺席类结论要先枚举同类全集，本仓的老雷）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = "taiji-machine-load-v1"
DEFAULT_SAMPLE_SECONDS = 0.5
DEFAULT_TOP = 3
CMDLINE_HEAD_LIMIT = 120
PYTHON_NAMES = ("python.exe", "python")


def _candidate_processes() -> list[Any]:
    import psutil

    out: list[Any] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = (proc.info["name"] or "").lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name in PYTHON_NAMES:
            out.append(proc)
    return out


def _read(sample_seconds: float, top: int) -> dict[str, Any]:
    import psutil

    if sample_seconds <= 0.0:
        raise ValueError(f"sample_seconds must be positive, got {sample_seconds}")

    procs = _candidate_processes()
    for proc in procs:
        #: 进程可能在两次枚举之间就没了——priming 失败不影响后面那轮取值，静默跳过即可。
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            proc.cpu_percent(None)

    #: 系统占用与逐进程取值共用**同一个**阻塞窗口 ⇒ 两个数同源，不是一前一后两次时刻。
    system_cpu = psutil.cpu_percent(interval=sample_seconds)
    logical_cpus = psutil.cpu_count(logical=True) or 0
    reader_pid = os.getpid()

    rows: list[dict[str, Any]] = []
    for proc in procs:
        pid = int(proc.pid)
        try:
            busy = float(proc.cpu_percent(None))
            cpu_seconds = float(sum(proc.cpu_times()[:2]))
            info = proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        cmdline = info["cmdline"] or []
        rows.append(
            {
                "pid": pid,
                "cpu_seconds": round(cpu_seconds, 1),
                "cpu_percent": round(busy, 1),
                "created": datetime.fromtimestamp(info["create_time"]).isoformat(
                    timespec="seconds"
                ),
                "cmdline_head": " ".join(cmdline)[:CMDLINE_HEAD_LIMIT],
                "is_reader": pid == reader_pid,
            }
        )
    rows.sort(key=lambda one: (-one["cpu_seconds"], one["pid"]))
    return {
        "format": FORMAT,
        "status": "ok",
        "sampled_at": datetime.now().isoformat(timespec="seconds"),
        "sample_seconds": sample_seconds,
        "logical_cpus": logical_cpus,
        "system_cpu_percent": system_cpu,
        "python_process_count": len(rows),
        "python_processes_burning_cpu": sum(1 for one in rows if one["cpu_percent"] > 0.0),
        "top_python_by_cpu_seconds": rows[:top],
        "reader_counted_itself": any(one["is_reader"] for one in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="machine load reader (read-only)")
    parser.add_argument("--out", type=Path, default=None, help="optional path for the JSON")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=DEFAULT_SAMPLE_SECONDS,
        help="sampling window in seconds (positive)",
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="how many hot pids to list")
    args = parser.parse_args(argv)

    try:
        payload = _read(args.sample_seconds, args.top)
    except (ImportError, ValueError, OSError) as error:
        #: 窗口给负数由 psutil 与本器当场拒绝；缺 psutil 也是**响亮拒绝**，不许降级成"读不到＝0"。
        print(json.dumps({"status": "load_read_failed", "error": str(error)}, ensure_ascii=True))
        return 2

    #: 只剩一枚 python 进程 ⇒ 要么这台机器真的只有本器，要么取法坏了；两种都不许当"负载很轻"引用。
    payload["disclosure"] = (
        "single_process_face" if payload["python_process_count"] == 1 else "ok"
    )
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
