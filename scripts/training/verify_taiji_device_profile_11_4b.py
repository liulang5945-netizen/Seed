"""11.4b: the device profile H that the 11.4 inventory listed as unmeasured.

Reads, without touching the candidate: OS, CPU name and counts, total RAM, torch build,
thread settings and accelerator availability. On this machine ``GlobalMemoryStatusEx``
returns 0, so RAM falls back to ``GetPhysicallyInstalledSystemMemory`` and then to the
CPU model's own reported value; whatever path answered is recorded.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

BUDGET_CPU_SECONDS = 60
BUDGET_WALL_SECONDS = 120


def cpu_name():
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        )
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        return str(name).strip()
    except OSError:
        return platform.processor() or "unknown"


def total_memory_kib():
    if os.name == "nt":
        value = ctypes.c_ulonglong(0)
        try:
            ok = ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(value))
        except OSError:
            ok = 0
        if ok and value.value:
            return int(value.value), "GetPhysicallyInstalledSystemMemory"
    try:
        import psutil

        return int(psutil.virtual_memory().total // 1024), "psutil.virtual_memory"
    except Exception:
        return None, "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.4b card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    memory_kib, memory_source = total_memory_kib()
    profile = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_name": cpu_name(),
        "cpu_logical": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": bool(torch.cuda.is_available()),
        "total_memory_kib": memory_kib,
        "total_memory_source": memory_source,
    }
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = (
        "device_profile_measured"
        if profile["cpu_name"] != "unknown" and memory_kib
        else "device_profile_partial"
    )

    report = {
        "schema": "taiji-device-profile-11-4b-v1",
        "step": "11.4b",
        "date": "2026-11-01",
        "status": status,
        "profile": profile,
        "model_involved": False,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Device profile only: it records the host, not the candidate. Thresholds quoted "
            "elsewhere are bound to this host and do not transfer without re-measuring."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            "scripts/training/verify_taiji_device_profile_11_4b.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "cpu_logical": profile["cpu_logical"],
                "torch_threads": profile["torch_threads"],
                "cuda": profile["cuda_available"],
                "memory_source": memory_source,
                "memory_kib": memory_kib,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
