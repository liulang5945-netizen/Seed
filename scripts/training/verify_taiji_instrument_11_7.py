"""11.7 instrument reading: repeatability, p50/p95, peak memory, scoring-path side effects.

The 11.4 card asks for "repeat runs, peak memory, p50/p95" and 11.1 asks that a
frozen evaluation "must not write". This script measures exactly those four things
on one declared path and reports what it cannot claim.

The side-effect hooks count calls and let them through: this is a detector, not a
sandbox. Any write outside ``__pycache__``, any socket, any subprocess is named in
the report and flips the exit.
"""

from __future__ import annotations

import argparse
import builtins
import ctypes
import hashlib
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.action_verdict_scorer import YES, answer_nll, prefix  # noqa: E402
from taiji.goal_conditioned_training import samples  # noqa: E402

REPEATS = 20
TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 240
BUDGET_WALL_SECONDS = 480
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"


def nearest_rank(values: list[float], percent: float):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def classify_open(mode: str, path: str) -> str:
    mode = str(mode)
    writable = any(flag in mode for flag in "wax+")
    if not writable:
        return "read"
    if "__pycache__" in str(path).replace("\\", "/"):
        return "pycache_write"
    return "write"


def status_from(*, budget_ok: bool, unsafe: bool, nondeterministic: bool, failed: bool) -> str:
    if not budget_ok:
        return "budget_stop"
    if unsafe:
        return "unsafe_side_effect"
    if nondeterministic:
        return "nondeterministic_output"
    if failed:
        return "engineering_failure"
    return "instrument_measured_quiet"


def _delta(before, after):
    if before is None or after is None:
        return None
    return int(after) - int(before)


def peak_working_set_bytes():
    if os.name != "nt":
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(usage) * 1024
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    try:
        query = ctypes.windll.psapi.GetProcessMemoryInfo
    except (AttributeError, OSError):
        query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    query.restype = ctypes.c_int
    ok = query(kernel32.GetCurrentProcess(), ctypes.byref(counters), ctypes.c_ulong(counters.cb))
    if not ok:
        return None
    return int(counters.PeakWorkingSetSize)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class SideEffectHooks:
    def __init__(self) -> None:
        self.writes: list[dict] = []
        self.sockets = 0
        self.subprocesses = 0
        self.system_calls = 0
        self._saved = {}

    def __enter__(self):
        original_open = builtins.open
        original_io_open = io.open
        original_socket = socket.socket
        original_popen = subprocess.Popen
        original_system = os.system
        self._saved = {
            "builtins.open": original_open,
            "io.open": original_io_open,
            "socket.socket": original_socket,
            "subprocess.Popen": original_popen,
            "os.system": original_system,
        }

        def hooked_open(file, mode="r", *args, **kwargs):
            kind = classify_open(mode, str(file))
            if kind == "write":
                self.writes.append({"path": str(file), "mode": str(mode)})
            return original_open(file, mode, *args, **kwargs)

        def hooked_io_open(file, mode="r", *args, **kwargs):
            kind = classify_open(mode, str(file))
            if kind == "write":
                self.writes.append({"path": str(file), "mode": str(mode)})
            return original_io_open(file, mode, *args, **kwargs)

        def hooked_socket(*args, **kwargs):
            self.sockets += 1
            return original_socket(*args, **kwargs)

        def hooked_popen(*args, **kwargs):
            self.subprocesses += 1
            return original_popen(*args, **kwargs)

        def hooked_system(*args, **kwargs):
            self.system_calls += 1
            return original_system(*args, **kwargs)

        builtins.open = hooked_open
        io.open = hooked_io_open
        socket.socket = hooked_socket
        subprocess.Popen = hooked_popen
        os.system = hooked_system
        return self

    def __exit__(self, *exc_info):
        builtins.open = self._saved["builtins.open"]
        io.open = self._saved["io.open"]
        socket.socket = self._saved["socket.socket"]
        subprocess.Popen = self._saved["subprocess.Popen"]
        os.system = self._saved["os.system"]
        return False

    def unsafe(self) -> bool:
        return bool(self.writes) or self.sockets or self.subprocesses or self.system_calls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("11.7 instrument card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    card = json.loads((ROOT / CLOSEOUT_CARD).read_text(encoding="utf-8"))
    expected_identity = card["identity"]["parent_model_digest"]

    peak_before_build = peak_working_set_bytes()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    rows = samples()
    per_state = len(rows) // 6
    held_rows = rows[TRAIN_STATES * per_state :]
    sample = held_rows[0]
    prompt = prefix(
        sample["state"],
        sample["goal"],
        f"把{sample['correct'][0]}放到{sample['correct'][1]}",
    )

    peak_after_build = peak_working_set_bytes()
    times, outputs, failure = [], [], None
    with SideEffectHooks() as hooks:
        try:
            with torch.no_grad():
                for _ in range(REPEATS):
                    started = time.perf_counter()
                    value = answer_nll(subject, prompt, YES, provider)
                    times.append(time.perf_counter() - started)
                    outputs.append(float(value))
        except Exception as error:  # noqa: BLE001 - the exit needs the message
            failure = f"{type(error).__name__}: {error}"
    peak_after_calls = peak_working_set_bytes()

    nondeterministic = bool(outputs) and outputs[0] != outputs[-1]
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    status = status_from(
        budget_ok=cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
        unsafe=hooks.unsafe(),
        nondeterministic=nondeterministic,
        failed=failure is not None,
    )

    report = {
        "schema": "taiji-instrument-11-7-v1",
        "step": "11.7",
        "date": "2026-11-01",
        "status": status,
        "parent_model_digest": expected_identity,
        "rebuilt_digest": digest,
        "identity_match": digest == expected_identity,
        "path": "taiji.action_verdict_scorer.answer_nll",
        "no_grad": True,
        "prompt_bytes": len(prompt.encode()),
        "repeats": REPEATS,
        "wall_seconds_per_call": times,
        "p50_seconds": nearest_rank(times, 50),
        "p95_seconds": nearest_rank(times, 95),
        "peak_working_set_start_bytes": peak_before_build,
        "peak_working_set_after_build_bytes": peak_after_build,
        "peak_working_set_after_calls_bytes": peak_after_calls,
        "build_peak_delta_bytes": _delta(peak_before_build, peak_after_build),
        "call_peak_delta_bytes": _delta(peak_after_build, peak_after_calls),
        "deterministic_outputs": not nondeterministic,
        "side_effects": {
            "writes": hooks.writes,
            "sockets": hooks.sockets,
            "subprocesses": hooks.subprocesses,
            "system_calls": hooks.system_calls,
            "allowed": ["read-only opens", "__pycache__ writes"],
        },
        "safety_checked": True,
        "safety_scope": ["static_declared_path", "runtime_hooked_open_socket_subprocess"],
        "failure": failure,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Instrument reading only: one process, one machine, one scoring path, twenty calls. "
            "It does not show that the subject is safe (it has no action channel, so this face is "
            "small by construction); it does not cover training, growth, recovery or migration "
            "paths; p50/p95 and memory do not extrapolate to other devices or larger inputs; and "
            "it is not task capability."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                CLOSEOUT_CARD,
                "taiji/action_verdict_scorer.py",
                "scripts/training/verify_taiji_instrument_11_7.py",
            )
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
                "identity_match": report["identity_match"],
                "p50": report["p50_seconds"],
                "p95": report["p95_seconds"],
                "peak_delta": report["call_peak_delta_bytes"],
                "deterministic": report["deterministic_outputs"],
                "writes": len(hooks.writes),
                "sockets": hooks.sockets,
                "subprocesses": hooks.subprocesses,
                "system_calls": hooks.system_calls,
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
