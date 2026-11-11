"""Decide whether two rebuilds of the 09.8 bundle read out the same.

Ledger ``D-20261111-03`` replaces the byte-equality self-proof with this
determinism gate.  One job builds the bundle twice, in two separate processes,
and feeds each build through the same readout consumers.  The two readout sets
must agree; the sha256 and byte count of either build are recorded as
diagnostics and are never the criterion.

Why the gate is shaped this way: identity with the sealed asset is a property of
the host that built it (the same two commits reproduce the sealed file on the
maintenance machine while the Windows runner does not), so a byte gate measures
the host, not the artifact.  Two rebuilds in the same job measure what the CI
job actually claims: that this build procedure is deterministic.

Exit names and codes (fixed in ``D-20261111-03``)::

    equivalent      0   every applicable layer agrees
    not_equivalent  1   structure, numbers or readouts disagree
    indeterminate   2   a required input is missing (build failed, consumer
                        crashed, runtime conditions not recorded)

Historical published reports are deliberately *not* consulted here.  Comparing
against them is a reviewed, manual procedure because the published readouts bind
the code of their own commit, not just the bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.core_09_bundle import (  # noqa: E402
    BUNDLE_RELATIVE_PATH,
    EXPECTED_SHA256,
    READOUT_MODE_ENV,
    sha256_of_file,
)

MATERIALIZER = ROOT / "scripts/training/materialize_taiji_core_09_bundle.py"

#: Fields that are toolchain, timing or provenance rather than a reading.  They
#: are compared nowhere and printed in the report instead.  The published-report
#: audit excludes the same family, so the two procedures agree on what a reading
#: is.
DIAGNOSTIC_KEYS = frozenset(
    {
        "cost",
        "identity",
        "manifest",
        "report",
        "path",
        "payload_dir",
        "module_dir",
        "prior_attempts",
        "date",
        "started_at",
        "finished_at",
        "elapsed",
        "elapsed_seconds",
        "cpu_seconds",
        "wall_seconds",
        "main_cpu_seconds",
        "worker_cpu_seconds",
        "total_cpu_seconds",
        "parent_wait_cpu_seconds",
        "peak_ram_mb",
        "max_live_bytes",
        "max_live_product_bytes",
        "cumulative_product_bytes",
        "bundle_bytes",
        "recorded_bundle_bytes",
        "bytes",
        "sha256",
        "grad_fn",
        "type_check",
        "lint",
    }
)

#: 诊断字段不只出现在顶层：消费者会把计时嵌在 `arms[*]`、`checks[*].observed` 里，
#: 键名带后缀（`eval_score_cpu`、`bptt_cpu`、`b_goal_grad_fn`、`estimated_wall_seconds`）。
#: 逐个列名跟不上，所以按后缀判：计时、内存、字节、sha、路径、autograd 节点都是诊断。
#: 本地自检（D-20261111-02 实施顺序 ④）抓到的 88 条全部落在这族里，没有一条是读数。
DIAGNOSTIC_SUFFIXES = (
    "_cpu",
    "_seconds",
    "_mb",
    "_bytes",
    "_sha256",
    "_grad_fn",
    "_path",
    "_dir",
)

#: 同理，`elapsed*` / `peak_ram*` 这类前缀也一并排除。
DIAGNOSTIC_PREFIXES = (
    "elapsed",
    "cpu_seconds",
    "started_at",
    "finished_at",
    "peak_ram",
    "max_live",
    "cumulative_product",
    "recorded_bundle",
)


def is_diagnostic(key: str) -> bool:
    """A key is a diagnostic when it is a timing, size, provenance or autograd node."""
    return (
        key in DIAGNOSTIC_KEYS
        or key.endswith(DIAGNOSTIC_SUFFIXES)
        or key.startswith(DIAGNOSTIC_PREFIXES)
    )


#: Relative width of one unit in the last place for the bundle's floating types.
ULP_RELATIVE = {
    "torch.float64": 2.0**-52,
    "torch.float32": 2.0**-23,
    "torch.float16": 2.0**-10,
}

READOUT_RELATIVE = 1e-12


def readouts() -> dict[str, dict[str, Any]]:
    """The consumers the gate replays, and how to invoke each one."""
    return {
        "10_21_1": {
            "script": "scripts/training/verify_taiji_10_21_1.py",
            "report": "10_21_1.json",
            "args": ("--output", "{report}"),
            "timeout": 900,
        },
        "composition_graph": {
            "script": "scripts/training/verify_taiji_composition_10_21.py",
            "report": "composition_graph.json",
            "args": ("--stage", "graph", "--output", "{report}"),
            "timeout": 1200,
        },
        "14_5_price": {
            "script": "scripts/training/verify_taiji_record_relation_14_5.py",
            "report": "14_5.json",
            "args": (
                "--stage",
                "price",
                "--report",
                "{report}",
                "--manifest",
                "{manifest}",
                "--payload-dir",
                "{payload}",
            ),
            "timeout": 1800,
        },
        "14_6_preflight": {
            "script": "scripts/training/verify_taiji_record_relation_14_6_preflight.py",
            "report": "14_6_preflight.json",
            "args": (
                "--stage",
                "preflight",
                "--report",
                "{report}",
                "--manifest",
                "{manifest}",
                "--payload-dir",
                "{payload}",
            ),
            "timeout": 1800,
        },
    }


def runtime_conditions() -> dict[str, Any]:
    """Everything a repeat of this comparison would have to hold fixed."""
    import torch

    config = torch.__config__.show()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "isa": getattr(torch.backends.cpu, "get_cpu_capability", lambda: None)(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "blas": config,
    }


def missing_conditions(conditions: dict[str, Any]) -> list[str]:
    required = (
        "python",
        "platform",
        "torch",
        "torch_threads",
        "torch_interop_threads",
        "isa",
        "blas",
    )
    return [name for name in required if not conditions.get(name)]


def build_once(work: Path, label: str) -> dict[str, Any]:
    """Run the materialiser in its own process; the point is cross-process equality."""
    directory = work / f"rebuild_{label}"
    command = [sys.executable, str(MATERIALIZER), "--bundle-dir", str(directory), "--force"]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    seconds = time.perf_counter() - started
    path = directory / Path(BUNDLE_RELATIVE_PATH).name
    row: dict[str, Any] = {
        "label": label,
        "exit": completed.returncode,
        "seconds": round(seconds, 3),
        "path": str(path),
        "stdout": completed.stdout.strip().splitlines()[-3:],
        "stderr": completed.stderr.strip().splitlines()[-3:],
    }
    if path.is_file():
        row["sha256"] = sha256_of_file(path)
        row["bytes"] = path.stat().st_size
        row["sha256_matches_sealed"] = row["sha256"] == EXPECTED_SHA256
    return row


def load_bundle(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} is not a mapping: {type(payload).__name__}")
    return payload


def compare_structure(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """Keys, list lengths, tensor dtypes and shapes must be identical."""
    import torch

    problems: list[dict[str, Any]] = []
    if type(left) is not type(right):
        return [
            {
                "path": path,
                "issue": "type",
                "left": type(left).__name__,
                "right": type(right).__name__,
            }
        ]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                problems.append(
                    {
                        "path": f"{path}/{key}",
                        "issue": "missing",
                        "left": key in left,
                        "right": key in right,
                    }
                )
                continue
            problems.extend(compare_structure(left[key], right[key], f"{path}/{key}"))
    elif isinstance(left, list):
        if len(left) != len(right):
            problems.append(
                {"path": path, "issue": "length", "left": len(left), "right": len(right)}
            )
        for index, (one, other) in enumerate(zip(left, right, strict=False)):
            problems.extend(compare_structure(one, other, f"{path}[{index}]"))
    elif isinstance(left, torch.Tensor):
        if left.dtype != right.dtype:
            problems.append(
                {"path": path, "issue": "dtype", "left": str(left.dtype), "right": str(right.dtype)}
            )
        elif tuple(left.shape) != tuple(right.shape):
            problems.append(
                {
                    "path": path,
                    "issue": "shape",
                    "left": list(left.shape),
                    "right": list(right.shape),
                }
            )
    return problems


def compare_numbers(
    left: Any, right: Any, path: str = "", worst: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Floating tensors may differ by at most one unit in the last place."""
    import torch

    summary = worst if worst is not None else {"max_relative": 0.0, "path": None, "failures": []}
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            compare_numbers(left[key], right[key], f"{path}/{key}", summary)
    elif isinstance(left, list) and isinstance(right, list):
        for index, (one, other) in enumerate(zip(left, right, strict=False)):
            compare_numbers(one, other, f"{path}[{index}]", summary)
    elif isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return summary
        if left.is_floating_point():
            one = left.double().ravel()
            other = right.double().ravel()
            scale = torch.maximum(one.abs(), other.abs()).clamp_min(1.0)
            relative = ((one - other).abs() / scale).max().item() if one.numel() else 0.0
            if relative > summary["max_relative"]:
                summary["max_relative"] = relative
                summary["path"] = path
            limit = ULP_RELATIVE.get(str(left.dtype), 2.0**-52)
            if relative > limit:
                summary["failures"].append({"path": path, "relative": relative, "limit": limit})
        elif not torch.equal(left, right):
            summary["failures"].append({"path": path, "issue": "integer mismatch"})
    return summary


def readout_command(spec: dict[str, Any], out_dir: Path) -> list[str]:
    report = out_dir / spec["report"]
    manifest = out_dir / f"{report.stem}_manifest.json"
    payload = out_dir / f"{report.stem}_payload"
    values = {"report": str(report), "manifest": str(manifest), "payload": str(payload)}
    return [
        sys.executable,
        str(ROOT / spec["script"]),
        *(arg.format(**values) for arg in spec["args"]),
    ]


def reset_readout_dir(out_dir: Path) -> Path:
    """Empty ``out_dir`` and recreate it.

    The consumers are write-once (10.21.1 refuses a report that already exists),
    so a leftover report from an earlier attempt would be read as this attempt's
    reading.  This is the same defect family as the ``SUPERSEDED`` reports: an
    old artifact standing in for a new observation.  Clearing is therefore a
    precondition of the run, not a cleanup detail, and it is pinned by
    ``test_readouts_clear_stale_reports_before_consumers_run``.
    """
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_readouts(bundle_path: Path, label: str, work: Path) -> dict[str, Any]:
    """Replay every consumer against one build, in readout mode."""
    out_dir = reset_readout_dir(work / f"readouts_{label}")
    environment = os.environ.copy()
    environment[READOUT_MODE_ENV] = "1"
    environment["TAIJI_BUNDLE_PATH"] = str(bundle_path)
    environment["PYTHONUTF8"] = "1"
    results: dict[str, Any] = {}
    for name, spec in readouts().items():
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                readout_command(spec, out_dir),
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=spec["timeout"],
                env=environment,
                check=False,
            )
            exit_code: int | None = completed.returncode
            tail = completed.stderr.strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            exit_code = None
            tail = ["timeout"]
        report = out_dir / spec["report"]
        results[name] = {
            "exit": exit_code,
            "seconds": round(time.perf_counter() - started, 3),
            "report": str(report),
            "report_present": report.is_file(),
            "stderr_tail": tail,
        }
    return results


def compare_readouts(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    """Compare two consumer reports; diagnostics and timings never take part."""
    problems: list[dict[str, Any]] = []
    if type(left) is not type(right):
        return [
            {
                "path": path,
                "issue": "type",
                "left": type(left).__name__,
                "right": type(right).__name__,
            }
        ]
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if is_diagnostic(key):
                continue
            if key not in left or key not in right:
                problems.append(
                    {
                        "path": f"{path}/{key}",
                        "issue": "missing",
                        "left": key in left,
                        "right": key in right,
                    }
                )
                continue
            problems.extend(compare_readouts(left[key], right[key], f"{path}/{key}"))
    elif isinstance(left, list):
        if len(left) != len(right):
            problems.append(
                {"path": path, "issue": "length", "left": len(left), "right": len(right)}
            )
        for index, (one, other) in enumerate(zip(left, right, strict=False)):
            problems.extend(compare_readouts(one, other, f"{path}[{index}]"))
    elif isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return problems
        tolerance = READOUT_RELATIVE * max(abs(left), abs(right), 1.0)
        if abs(left - right) > tolerance:
            problems.append({"path": path, "issue": "float", "left": left, "right": right})
    elif left != right:
        problems.append({"path": path, "issue": "value", "left": left, "right": right})
    return problems


def summarise(builds: list[dict[str, Any]], layer: dict[str, Any]) -> str:
    if any(row["exit"] != 0 or "sha256" not in row for row in builds):
        return "indeterminate"
    if layer["structure"]["problems"] or layer["numbers"]["failures"]:
        return "not_equivalent"
    if layer["readouts"]["indeterminate"]:
        return "indeterminate"
    if layer["readouts"]["problems"]:
        return "not_equivalent"
    return "equivalent"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "output" / "ci_verify")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "ci_verify" / "bundle_rebuild_determinism.json",
        help="where the machine-readable verdict is written",
    )
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    conditions = runtime_conditions()
    missing = missing_conditions(conditions)
    builds = [build_once(work, "a"), build_once(work, "b")]

    layer: dict[str, Any] = {
        "structure": {"problems": []},
        "numbers": {"max_relative": 0.0, "path": None, "failures": []},
        "readouts": {"indeterminate": False, "problems": [], "runs": {}},
    }
    if all(row["exit"] == 0 and "sha256" in row for row in builds):
        left = load_bundle(Path(builds[0]["path"]))
        right = load_bundle(Path(builds[1]["path"]))
        layer["structure"]["problems"] = compare_structure(left, right)
        layer["numbers"] = compare_numbers(left, right)
        del left, right
        official = Path(ROOT) / BUNDLE_RELATIVE_PATH
        layer["sealed_reference"] = {
            "present": official.is_file(),
            "sha256": sha256_of_file(official) if official.is_file() else None,
            "rebuilt_a_matches": official.is_file()
            and sha256_of_file(official) == builds[0]["sha256"],
        }
        for index, row in enumerate(builds):
            label = "a" if index == 0 else "b"
            layer["readouts"]["runs"][label] = run_readouts(Path(row["path"]), label, work)
        a_runs = layer["readouts"]["runs"]["a"]
        b_runs = layer["readouts"]["runs"]["b"]
        for name in readouts():
            one, other = a_runs[name], b_runs[name]
            if not (one["report_present"] and other["report_present"]):
                layer["readouts"]["indeterminate"] = True
                continue
            first = json.loads(Path(one["report"]).read_text(encoding="utf-8"))
            second = json.loads(Path(other["report"]).read_text(encoding="utf-8"))
            for problem in compare_readouts(first, second, f"/{name}"):
                layer["readouts"]["problems"].append(problem)

    verdict = summarise(builds, layer)
    if missing:
        verdict = "indeterminate"

    payload = {
        "schema": "taiji-bundle-rebuild-determinism-v1",
        "verdict": verdict,
        "exit_code": {"equivalent": 0, "not_equivalent": 1, "indeterminate": 2}[verdict],
        "decided_by": "D-20261111-03",
        "runtime_conditions": conditions,
        "missing_conditions": missing,
        "builds": builds,
        "layers": layer,
        "readout_fields_excluded_as_diagnostics": sorted(DIAGNOSTIC_KEYS),
        "readout_diagnostic_suffixes": list(DIAGNOSTIC_SUFFIXES),
        "readout_diagnostic_prefixes": list(DIAGNOSTIC_PREFIXES),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"verdict: {verdict}")
    for row in builds:
        seal = "sealed" if row.get("sha256_matches_sealed") else "not-sealed"
        print(
            f"  rebuild {row['label']}: exit {row['exit']} sha256={row.get('sha256', 'missing')} "
            f"bytes={row.get('bytes', 'missing')} ({seal}) {row['seconds']}s"
        )
    if missing:
        print(f"  missing runtime conditions: {', '.join(missing)}")
    if layer["structure"]["problems"]:
        print(f"  structure problems: {len(layer['structure']['problems'])}")
    if layer["numbers"]["failures"]:
        print(f"  numeric problems: {len(layer['numbers']['failures'])}")
    if layer["readouts"]["problems"]:
        print(f"  readout problems: {len(layer['readouts']['problems'])}")
        for problem in layer["readouts"]["problems"][:5]:
            print(f"    {problem}")
    if layer["readouts"]["indeterminate"]:
        print("  at least one consumer produced no report")
    print(f"report: {args.out}")
    return payload["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
