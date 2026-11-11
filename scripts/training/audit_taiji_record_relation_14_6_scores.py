"""14.6 score reconstruction and closeout correction.

The formal 14.6 run completed all 1,800 declared updates and all six
candidates.  Its v1 card was written as ``engineering_failure`` because two
instrument bugs corrupted the aggregation, not because the model failed:

1. ``score_generation`` compared the decoded ``bytes`` object against the
   fixture's integer tuple.  Every condition that carried the fixture target
   (step0, real, reverse, no-source, wrong-record, wrong-position, rescue)
   therefore had ``completed``/``full_sentence``/``entity_exact`` false even
   when the generated text was the exact canonical action.  The permuted
   condition already carried bytes and scored correctly, which exposes the
   mismatch.
2. ``collect_rows`` looked for step0 rows under a non-existent ``real`` key
   and returned all four structural conditions for each structural count.

This script does not run the model, does not call Adam and does not change a
weight.  It reconstructs every row's raw bytes from the stored generated text
(valid UTF-8 or the stored hex), re-scores the exact flags against the
canonical action bytes, recomputes the declared gates with the same bars,
bootstrap seed and fixed set, verifies the 12 retained bundles, and writes the
corrected card and manifest.  The v1 card and manifest remain under their
historical ``_v1_engineering_failure`` names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

V1_REPORT = "reports/taiji_core_record_relation_14_6_20261102_v1_engineering_failure.json"
V1_MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102_v1_engineering_failure.json"
REPORT = "reports/taiji_core_record_relation_14_6_20261102.json"
MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102.json"
OUTPUT_DIR = "output/taiji_core_record_relation_14_6"
AUDIT_SOURCE = "scripts/training/audit_taiji_record_relation_14_6_scores.py"

ARM_FULL = "A_full"
ARM_LOCAL = "B_local"
ARM_FLAT = "C_flat"
ARMS = (ARM_FULL, ARM_LOCAL, ARM_FLAT)
SEEDS = (27031, 27032)
STRUCTURAL = ("no_source", "wrong_record", "wrong_position", "rescue")

TASK_TOTAL_MIN = 0.80
TASK_LAYER_MIN = 0.60
BOTH_SIDES_MIN = 0.60
REVERSE_MIN = 0.95
STRUCTURE_GAIN_MIN = 0.10
CONTENT_DROP_MIN = 0.10
BOOTSTRAP_SEED = 57031
BOOTSTRAP_ROUNDS = 2000

FORMAL_CPU_SECONDS = 14400
FORMAL_WALL_SECONDS = 28800
FORMAL_ADAM_CALLS = 1800
FORMAL_PRODUCT_BYTES = 4 * 1024**3

EXPECTED_STEP0_REAL = 960
EXPECTED_STEP300_REAL = 960
EXPECTED_PERMUTED = 960
EXPECTED_REVERSE = 960
EXPECTED_STRUCTURAL = 48
EXPECTED_NLL = 1920
EXPECTED_GENERATIONS = 4032
EXPECTED_BUNDLES = 12

INVALID_PREFIX = "<invalid-utf8:"
INVALID_SUFFIX = ">"

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class InstrumentError(RuntimeError):
    """A v1 row, bundle or gate invariant failed during the reconstruction."""


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed,
    note: str = "",
    *,
    required: bool = True,
) -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "required": bool(required),
            "observed": observed,
            "note": note,
        }
    )


def raw_from_row(row: dict) -> tuple[bytes, str]:
    text = row["generated_text"]
    if row["utf8"]:
        raw = text.encode("utf-8")
        mode = "utf8"
    elif text.startswith(INVALID_PREFIX) and text.endswith(INVALID_SUFFIX):
        raw = bytes.fromhex(text[len(INVALID_PREFIX) : -len(INVALID_SUFFIX)])
        mode = "hex"
    else:
        raise InstrumentError(f"row {row.get('name')} has no stored raw form")
    if len(raw) != int(row["bytes"]):
        raise InstrumentError(
            f"row {row.get('name')} raw length {len(raw)} != stored {row['bytes']}"
        )
    return raw, mode


def rescore_row(row: dict) -> tuple[dict, str]:
    raw, mode = raw_from_row(row)
    expected = row["expected"].encode("utf-8")
    if len(expected) != 27:
        raise InstrumentError(f"row {row.get('name')} expected is not 27 bytes")
    full = raw == expected
    object_exact = len(raw) == 27 and raw[3:15] == expected[3:15]
    place_exact = len(raw) == 27 and raw[21:27] == expected[21:27]
    updated = dict(row)
    updated.update(
        {
            "object_exact": bool(object_exact),
            "place_exact": bool(place_exact),
            "entity_exact": bool(object_exact and place_exact),
            "full_sentence": bool(full),
            "completed": bool(full and row["terminated"]),
            "raw_mode": mode,
        }
    )
    return updated, mode


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "completed": 0.0,
            "full_sentence": 0.0,
            "entity_exact": 0.0,
            "eos": 0.0,
            "utf8": 0.0,
            "source_top1": None,
            "layers": {},
        }
    source_rows = [row for row in rows if row.get("source_correct") is not None]

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / len(rows)

    layers = {}
    for layer in ("seen_combination", "unseen_combination", "unseen_entity"):
        selected = [row for row in rows if row["layer"] == layer]
        if selected:
            layers[layer] = {
                "n": len(selected),
                "completed": sum(1 for row in selected if row["completed"]) / len(selected),
            }
    return {
        "n": len(rows),
        "completed": rate("completed"),
        "full_sentence": rate("full_sentence"),
        "entity_exact": rate("entity_exact"),
        "eos": rate("terminated"),
        "utf8": rate("utf8"),
        "source_top1": (
            None
            if not source_rows
            else sum(1 for row in source_rows if row["source_correct"]) / len(source_rows)
        ),
        "layers": layers,
    }


def accuracy(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row["completed"]) / len(rows)


def mean_float(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.fmean(values))


def bootstrap_lower(diffs: list[float], *, seed: int, rounds: int) -> dict:
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(rounds):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return {
        "rounds": rounds,
        "seed": seed,
        "mean": statistics.fmean(diffs),
        "lower_95": means[int(0.025 * rounds)],
        "upper_95": means[int(0.975 * rounds) - 1],
    }


def collect(
    candidates: list[dict],
    *,
    phase: str,
    condition: str,
    arm: str | None = None,
    seed: int | None = None,
) -> list[dict]:
    rows = []
    for candidate in candidates:
        if arm is not None and candidate["arm"] != arm:
            continue
        if seed is not None and candidate["seed"] != seed:
            continue
        if phase == "step0":
            if condition == "real":
                rows.extend(candidate["step0"]["rows"])
            continue
        if condition in ("real", "permuted", "reverse"):
            rows.extend(candidate["step300"][condition]["rows"])
        elif condition in STRUCTURAL:
            rows.extend(candidate["step300"]["structural"]["conditions"][condition]["rows"])
        else:
            raise InstrumentError(f"unknown phase/condition {phase}/{condition}")
    return rows


def gate_task(candidates: list[dict]) -> dict:
    real = collect(candidates, phase="step300", condition="real", arm=ARM_FULL)
    permuted = collect(candidates, phase="step300", condition="permuted", arm=ARM_FULL)
    reverse = collect(candidates, phase="step300", condition="reverse", arm=ARM_FULL)
    total = accuracy(real)
    unseen_combination = accuracy([row for row in real if row["layer"] == "unseen_combination"])
    unseen_entity = accuracy([row for row in real if row["layer"] == "unseen_entity"])
    real_by_key = {(row["seed"], row["name"]): row for row in real}
    perm_by_key = {(row["seed"], row["name"]): row for row in permuted}
    keys = sorted(set(real_by_key) & set(perm_by_key))
    both = [real_by_key[key]["completed"] and perm_by_key[key]["completed"] for key in keys]
    both_sides = sum(1 for value in both if value) / len(both) if both else 0.0
    reverse_rate = accuracy(reverse)
    passed = (
        total >= TASK_TOTAL_MIN
        and unseen_combination >= TASK_LAYER_MIN
        and unseen_entity >= TASK_LAYER_MIN
        and both_sides >= BOTH_SIDES_MIN
        and reverse_rate >= REVERSE_MIN
        and len(real) == 320
        and len(permuted) == 320
        and len(reverse) == 320
        and len(keys) == 320
    )
    return {
        "passed": passed,
        "bars": {
            "total": TASK_TOTAL_MIN,
            "unseen_combination": TASK_LAYER_MIN,
            "unseen_entity": TASK_LAYER_MIN,
            "both_sides": BOTH_SIDES_MIN,
            "reverse": REVERSE_MIN,
        },
        "observed": {
            "total": total,
            "total_n": len(real),
            "unseen_combination": unseen_combination,
            "unseen_entity": unseen_entity,
            "both_sides": both_sides,
            "both_sides_n": len(both),
            "reverse": reverse_rate,
            "reverse_n": len(reverse),
        },
    }


def gate_structure(candidates: list[dict]) -> dict:
    rows = []
    for seed in SEEDS:
        a_rows = collect(candidates, phase="step300", condition="real", arm=ARM_FULL, seed=seed)
        b_rows = collect(candidates, phase="step300", condition="real", arm=ARM_LOCAL, seed=seed)
        c_rows = collect(candidates, phase="step300", condition="real", arm=ARM_FLAT, seed=seed)
        a_acc = accuracy(a_rows)
        b_acc = accuracy(b_rows)
        c_acc = accuracy(c_rows)
        saturated = a_acc >= 0.95 and b_acc >= 0.95 and c_acc >= 0.95
        worlds = sorted({row["world"] for row in a_rows})
        complete_worlds = worlds == list(range(8))
        for name, other, other_acc in (
            ("A_minus_B", ARM_LOCAL, b_acc),
            ("A_minus_C", ARM_FLAT, c_acc),
        ):
            diffs = []
            for world in worlds:
                a_world = [row for row in a_rows if row["world"] == world]
                other_world = [
                    row
                    for row in (b_rows if other == ARM_LOCAL else c_rows)
                    if row["world"] == world
                ]
                diffs.append(accuracy(a_world) - accuracy(other_world))
            bootstrap = (
                bootstrap_lower(diffs, seed=BOOTSTRAP_SEED, rounds=BOOTSTRAP_ROUNDS)
                if diffs
                else {
                    "rounds": 0,
                    "seed": BOOTSTRAP_SEED,
                    "mean": 0.0,
                    "lower_95": 0.0,
                    "upper_95": 0.0,
                }
            )
            gain = a_acc - other_acc
            passed = complete_worlds and gain >= STRUCTURE_GAIN_MIN and bootstrap["lower_95"] > 0.0
            rows.append(
                {
                    "seed": seed,
                    "comparison": name,
                    "gain": gain,
                    "bootstrap": bootstrap,
                    "passed": bool(passed),
                    "saturated": bool(saturated),
                    "complete_worlds": bool(complete_worlds),
                    "a_accuracy": a_acc,
                    "other_accuracy": other_acc,
                    "world_differences": diffs,
                }
            )
    passed = all(row["passed"] for row in rows) and not any(row["saturated"] for row in rows)
    return {
        "passed": passed,
        "bar": STRUCTURE_GAIN_MIN,
        "rows": rows,
        "reason": (
            "saturated_unresolved" if any(row["saturated"] for row in rows) and not passed else None
        ),
    }


def gate_content(candidates: list[dict], fixed_names: tuple[str, ...]) -> dict:
    real = [
        row
        for row in collect(candidates, phase="step300", condition="real", arm=ARM_FULL)
        if row["name"] in fixed_names
    ]
    conditions = {}
    for name in ("no_source", "wrong_record", "wrong_position"):
        rows = [
            row
            for row in collect(candidates, phase="step300", condition=name, arm=ARM_FULL)
            if row["name"] in fixed_names
        ]
        real_acc = accuracy(real)
        condition_acc = accuracy(rows)
        drop = real_acc - condition_acc
        conditions[name] = {
            "n": len(rows),
            "real_accuracy": real_acc,
            "condition_accuracy": condition_acc,
            "drop": drop,
            "passed": drop >= CONTENT_DROP_MIN,
        }
    rescue = [
        row
        for row in collect(candidates, phase="step300", condition="rescue", arm=ARM_FULL)
        if row["name"] in fixed_names
    ]
    return {
        "passed": all(condition["passed"] for condition in conditions.values()),
        "bar": CONTENT_DROP_MIN,
        "conditions": conditions,
        "rescue": {
            "n": len(rescue),
            "real_accuracy": accuracy(real),
            "rescue_accuracy": accuracy(rescue),
            "delta": accuracy(rescue) - accuracy(real),
        },
        "fixed_names": list(fixed_names),
    }


def gate_retention(candidates: list[dict]) -> dict:
    step0 = collect(candidates, phase="step0", condition="real", arm=ARM_FULL)
    step300 = collect(candidates, phase="step300", condition="real", arm=ARM_FULL)
    step0_nll = [
        item["nll"]
        for candidate in candidates
        if candidate["arm"] == ARM_FULL
        for item in candidate["step0"]["nll"]["rows"]
    ]
    step300_nll = [
        item["nll"]
        for candidate in candidates
        if candidate["arm"] == ARM_FULL
        for item in candidate["step300"]["real"]["nll"]["rows"]
    ]
    return {
        "status": "retention_not_admissible",
        "reason": "no_qualified_parent_capability_set",
        "step0_accuracy": accuracy(step0),
        "step300_accuracy": accuracy(step300),
        "step0_mean_nll": mean_float(step0_nll),
        "step300_mean_nll": mean_float(step300_nll),
        "signed_nll_change": (
            mean_float(step300_nll) - mean_float(step0_nll) if step0_nll and step300_nll else None
        ),
        "note": (
            "Old behaviour is measured at both endpoints but not claimed as retained; "
            "there is no qualified parent capability set in this package."
        ),
    }


def closeout(
    *,
    engineering_ok: bool,
    task: dict,
    structure: dict,
    content: dict,
    within: dict,
) -> tuple[str, bool]:
    if not engineering_ok:
        return "engineering_failure", False
    if not (within["cpu"] and within["wall"] and within["adam"] and within["products"]):
        return "budget_stop", False
    if not task["passed"]:
        return "native_byte_task_failed", False
    if not content["passed"]:
        return "content_path_not_necessary", True
    if not structure["passed"]:
        return "task_supported_component_gain_unresolved", True
    return "task_and_component_supported_pending_confirm", True


def run_type_check(paths: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        *paths,
    ]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        errors = 0
        for line in result.stdout.splitlines():
            if line.startswith("Found ") and " error" in line:
                errors = int(line.split()[1])
        return {
            "returncode": result.returncode,
            "errors": errors,
            "output": result.stdout.strip(),
            "passed": result.returncode == 0 and errors == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": None,
            "errors": None,
            "output": f"{type(exc).__name__}: {exc}",
            "passed": False,
        }


def run_lint(paths: list[str]) -> dict:
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    ruff_format = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "ruff_format": {
            "returncode": ruff_format.returncode,
            "output": ruff_format.stdout.strip(),
        },
        "black": {
            "returncode": None,
            "status": "not_run",
            "reason": (
                "the installed black 26.5.1 does not terminate on the stage runners "
                "in this environment; ruff format is the declared formatter check"
            ),
        },
        "passed": ruff.returncode == 0 and ruff_format.returncode == 0,
    }


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"path": str(path), "sha256": sha256_text(path), "bytes": path.stat().st_size}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-report", default=V1_REPORT)
    parser.add_argument("--v1-manifest", default=V1_MANIFEST)
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--manifest", default=MANIFEST)
    args = parser.parse_args()
    v1_report_path = ROOT / args.v1_report
    v1_manifest_path = ROOT / args.v1_manifest
    report_path = ROOT / args.report
    manifest_path = ROOT / args.manifest
    if report_path.exists() or manifest_path.exists():
        raise FileExistsError("the corrected 14.6 canonical cards are write-once")
    v1 = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v1_manifest = json.loads(v1_manifest_path.read_text(encoding="utf-8"))
    if v1.get("status") != "engineering_failure":
        raise InstrumentError("the v1 card is not the declared engineering_failure")
    if "evaluation_complete" not in (v1.get("failed_required") or []):
        raise InstrumentError("the v1 card does not fail on evaluation_complete")
    if v1_manifest.get("status") != "engineering_failure":
        raise InstrumentError("the v1 manifest is not the declared engineering_failure")
    if len(v1.get("candidates", [])) != 6:
        raise InstrumentError("the v1 card does not carry six candidates")
    if v1["cost"]["adam_calls_total"] != FORMAL_ADAM_CALLS:
        raise InstrumentError("the v1 card does not carry the 1,800 formal updates")
    if not all(v1["within"][key] for key in ("cpu", "wall", "adam", "products")):
        raise InstrumentError("the v1 card is outside the declared budget")

    audit_sha = sha256_text(ROOT / AUDIT_SOURCE)
    v1_sha = sha256_text(v1_report_path)
    v1_manifest_sha = sha256_text(v1_manifest_path)

    # Deep copy the v1 candidate rows, then re-score every exact flag from the
    # stored raw form.  No model object is built in this script.
    candidates = json.loads(json.dumps(v1["candidates"]))
    mode_counts: dict[str, int] = {"utf8": 0, "hex": 0}
    permuted_mismatch = 0
    tuple_stored_true = 0
    reconstruction_rows = 0
    tuple_rows = 0
    permuted_rows = 0
    for candidate in candidates:
        for row in candidate["step0"]["rows"]:
            stored_true = bool(row["completed"] or row["full_sentence"] or row["entity_exact"])
            updated, mode = rescore_row(row)
            row.clear()
            row.update(updated)
            mode_counts[mode] += 1
            reconstruction_rows += 1
            tuple_rows += 1
            if stored_true:
                tuple_stored_true += 1
        candidate["step0"]["summary"] = summarize(candidate["step0"]["rows"])
        for condition in ("real", "permuted", "reverse"):
            for row in candidate["step300"][condition]["rows"]:
                updated, mode = rescore_row(row)
                if condition == "permuted":
                    if (
                        bool(row["completed"]) != bool(updated["completed"])
                        or bool(row["full_sentence"]) != bool(updated["full_sentence"])
                        or bool(row["entity_exact"]) != bool(updated["entity_exact"])
                    ):
                        permuted_mismatch += 1
                    permuted_rows += 1
                else:
                    tuple_rows += 1
                    if row["completed"] or row["full_sentence"] or row["entity_exact"]:
                        tuple_stored_true += 1
                row.clear()
                row.update(updated)
                mode_counts[mode] += 1
                reconstruction_rows += 1
            candidate["step300"][condition]["summary"] = summarize(
                candidate["step300"][condition]["rows"]
            )
        for _name, block in candidate["step300"]["structural"]["conditions"].items():
            for row in block["rows"]:
                updated, mode = rescore_row(row)
                tuple_rows += 1
                if row["completed"] or row["full_sentence"] or row["entity_exact"]:
                    tuple_stored_true += 1
                row.clear()
                row.update(updated)
                mode_counts[mode] += 1
                reconstruction_rows += 1
            block["summary"] = summarize(block["rows"])

    counts = {
        "step0_real": len(collect(candidates, phase="step0", condition="real")),
        "step300_real": len(collect(candidates, phase="step300", condition="real")),
        "permuted": len(collect(candidates, phase="step300", condition="permuted")),
        "reverse": len(collect(candidates, phase="step300", condition="reverse")),
        "no_source": len(collect(candidates, phase="step300", condition="no_source")),
        "wrong_record": len(collect(candidates, phase="step300", condition="wrong_record")),
        "wrong_position": len(collect(candidates, phase="step300", condition="wrong_position")),
        "rescue": len(collect(candidates, phase="step300", condition="rescue")),
        "nll": sum(
            (
                len(candidate[phase]["real"]["nll"]["rows"])
                if phase == "step300"
                else len(candidate["step0"]["nll"]["rows"])
            )
            for candidate in candidates
            for phase in ("step0", "step300")
        ),
    }
    counts["generations"] = (
        counts["step0_real"]
        + counts["step300_real"]
        + counts["permuted"]
        + counts["reverse"]
        + counts["no_source"]
        + counts["wrong_record"]
        + counts["wrong_position"]
        + counts["rescue"]
    )
    evaluation_ok = (
        counts["step0_real"] == EXPECTED_STEP0_REAL
        and counts["step300_real"] == EXPECTED_STEP300_REAL
        and counts["permuted"] == EXPECTED_PERMUTED
        and counts["reverse"] == EXPECTED_REVERSE
        and all(counts[name] == EXPECTED_STRUCTURAL for name in STRUCTURAL)
        and counts["nll"] == EXPECTED_NLL
        and counts["generations"] == EXPECTED_GENERATIONS
        and reconstruction_rows == EXPECTED_GENERATIONS
        and tuple_rows == EXPECTED_GENERATIONS - EXPECTED_PERMUTED
        and permuted_rows == EXPECTED_PERMUTED
        and permuted_mismatch == 0
        and tuple_stored_true == 0
    )
    if not evaluation_ok:
        raise InstrumentError(
            "the reconstructed rows do not match the declared evaluation counts: "
            f"{counts}, rows={reconstruction_rows}, permuted_mismatch={permuted_mismatch}, "
            f"tuple_stored_true={tuple_stored_true}"
        )

    # Verify the retained bundles against the v1 manifest.
    retained_rows = []
    for row in v1_manifest.get("retained", []):
        path = ROOT / row["path"]
        actual_bytes = path.stat().st_size if path.exists() else None
        actual_sha = sha256_text(path) if path.exists() else None
        ok = path.exists() and actual_bytes == int(row["bytes"]) and actual_sha == row["sha256"]
        retained_rows.append(
            {
                "seed": row["seed"],
                "arm": row["arm"],
                "update_index": row["update_index"],
                "path": row["path"],
                "expected_bytes": int(row["bytes"]),
                "actual_bytes": actual_bytes,
                "expected_sha256": row["sha256"],
                "actual_sha256": actual_sha,
                "verified": bool(ok),
            }
        )
    retained_ok = len(retained_rows) == EXPECTED_BUNDLES and all(
        row["verified"] for row in retained_rows
    )
    if not retained_ok:
        raise InstrumentError("the retained 14.6 bundles did not verify")

    fixed_names = tuple(v1["gates"]["content"]["fixed_names"] if v1.get("gates") else ())
    if len(fixed_names) != 8:
        raise InstrumentError("the v1 card does not carry the fixed 8 names")

    task = gate_task(candidates)
    structure = gate_structure(candidates)
    content = gate_content(candidates, fixed_names)
    retention = gate_retention(candidates)

    audit_cpu = time.process_time() - STARTED_CPU
    audit_wall = time.perf_counter() - STARTED_WALL
    v1_cost = v1["cost"]
    total_cpu = float(v1_cost["total_cpu_seconds"]) + audit_cpu
    total_wall = float(v1_cost["wall_seconds"]) + audit_wall
    max_products = int(v1_cost["max_live_product_bytes"])
    within = {
        "cpu": total_cpu <= FORMAL_CPU_SECONDS,
        "wall": total_wall <= FORMAL_WALL_SECONDS,
        "adam": FORMAL_ADAM_CALLS <= FORMAL_ADAM_CALLS,
        "products": max_products <= FORMAL_PRODUCT_BYTES,
        "limits": v1["within"]["limits"],
        "observed": {
            "cpu_seconds": total_cpu,
            "wall_seconds": total_wall,
            "adam_calls": FORMAL_ADAM_CALLS,
            "max_live_product_bytes": max_products,
        },
        "v1_observed": v1["within"]["observed"],
    }
    cost = dict(v1_cost)
    cost.update(
        {
            "main_cpu_seconds": float(v1_cost["main_cpu_seconds"]) + audit_cpu,
            "total_cpu_seconds": total_cpu,
            "wall_seconds": total_wall,
            "peak_ram_mb": v1_cost["peak_ram_mb"],
            "audit_cpu_seconds": audit_cpu,
            "audit_wall_seconds": audit_wall,
            "v1_cost": v1_cost,
        }
    )

    checks: list[dict] = []
    for row in v1["checks"]:
        if row["id"] == "evaluation_complete":
            check(checks, "evaluation_complete", evaluation_ok, counts)
        elif row["id"] == "task_gate":
            check(checks, "task_gate", task["passed"], task["observed"], required=False)
        elif row["id"] == "structure_gate":
            check(
                checks,
                "structure_gate",
                structure["passed"],
                {
                    "bar": structure["bar"],
                    "rows": [
                        {
                            "seed": item["seed"],
                            "comparison": item["comparison"],
                            "gain": item["gain"],
                            "lower_95": item["bootstrap"]["lower_95"],
                            "passed": item["passed"],
                            "saturated": item["saturated"],
                        }
                        for item in structure["rows"]
                    ],
                    "reason": structure["reason"],
                },
                required=False,
            )
        elif row["id"] == "content_gate":
            check(
                checks,
                "content_gate",
                content["passed"],
                content["conditions"],
                required=False,
            )
        elif row["id"] == "retention":
            check(
                checks,
                "retention",
                retention["status"] == "retention_not_admissible",
                retention,
                required=False,
            )
        elif row["id"] == "stage_cost_within_budget":
            check(
                checks,
                "stage_cost_within_budget",
                within["cpu"] and within["wall"] and within["adam"] and within["products"],
                within,
            )
        elif row["id"] == "manifest_identity":
            continue
        else:
            checks.append(row)

    check(
        checks,
        "score_reconstruction",
        evaluation_ok and mode_counts["utf8"] + mode_counts["hex"] == EXPECTED_GENERATIONS,
        {
            "rows_reconstructed": reconstruction_rows,
            "expected_generations": EXPECTED_GENERATIONS,
            "utf8_rows": mode_counts["utf8"],
            "hex_rows": mode_counts["hex"],
            "tuple_expected_rows_rescored": tuple_rows,
            "permuted_rows_checked": permuted_rows,
            "permuted_stored_mismatches": permuted_mismatch,
            "tuple_expected_stored_true_before": tuple_stored_true,
            "counts": counts,
        },
        "Raw bytes are reconstructed from the stored generated text; every exact flag "
        "is re-scored against the canonical 27-byte action.",
    )
    check(
        checks,
        "retained_bundles",
        retained_ok,
        {"rows": retained_rows},
        "The 12 retained step0/step300 bundles match the v1 manifest size and sha.",
    )
    check(
        checks,
        "v1_failure_classification",
        True,
        {
            "v1_report": args.v1_report,
            "v1_report_sha256": v1_sha,
            "v1_manifest": args.v1_manifest,
            "v1_manifest_sha256": v1_manifest_sha,
            "v1_status": v1["status"],
            "v1_failed_required": v1["failed_required"],
            "bugs": [
                "score_generation compared bytes to the fixture's integer tuple",
                "collect_rows used the wrong step0 nesting and returned all structural conditions",
            ],
        },
        "The v1 card is preserved; this correction re-scores stored text without a model forward.",
    )
    audit_type = run_type_check([AUDIT_SOURCE])
    check(checks, "audit_type_check", audit_type["passed"], audit_type)
    audit_lint = run_lint([AUDIT_SOURCE])
    check(checks, "audit_lint", audit_lint["passed"], audit_lint)

    engineering_ok = all(row["passed"] for row in checks if row.get("required", True))
    status, release = closeout(
        engineering_ok=engineering_ok,
        task=task,
        structure=structure,
        content=content,
        within=within,
    )

    source_hashes = dict(v1["identity"]["source_hashes"])
    source_hashes[AUDIT_SOURCE] = audit_sha
    correction = {
        "v1_report": args.v1_report,
        "v1_report_sha256": v1_sha,
        "v1_manifest": args.v1_manifest,
        "v1_manifest_sha256": v1_manifest_sha,
        "v1_status": v1["status"],
        "v1_failed_required": v1["failed_required"],
        "bugs": [
            (
                "score_generation compared the raw bytes object to the fixture's "
                "integer tuple, so exact flags were false for every tuple-expected "
                "condition; the permuted condition already used bytes and scored correctly"
            ),
            (
                "collect_rows looked for step0 rows under a non-existent 'real' key "
                "and returned all four structural conditions for each structural count"
            ),
        ],
        "method": (
            "deterministic re-score of the stored generated text: raw bytes are "
            "reconstructed from valid UTF-8 or the stored hex, exact flags are re-scored "
            "against the canonical action bytes, and the declared gates are recomputed "
            "with the same bars, bootstrap seed and fixed 8 examples"
        ),
        "rows_reconstructed": reconstruction_rows,
        "nll_rows": counts["nll"],
        "tuple_expected_rows_rescored": tuple_rows,
        "permuted_rows_checked": permuted_rows,
        "retained_bundles_verified": len(retained_rows),
        "no_model_forward": True,
        "no_adam_call": True,
        "no_weight_change": True,
        "audit_source": AUDIT_SOURCE,
        "audit_source_sha256": audit_sha,
    }

    identity = dict(v1["identity"])
    identity.update(
        {
            "v1_report": args.v1_report,
            "v1_report_sha256": v1_sha,
            "v1_manifest": args.v1_manifest,
            "v1_manifest_sha256": v1_manifest_sha,
            "correction_source": AUDIT_SOURCE,
            "correction_source_sha256": audit_sha,
            "source_hashes": source_hashes,
        }
    )
    limits = (
        v1["limits"] + " The corrected exact flags are a deterministic re-score of the stored raw "
        "generated text; the correction ran no model forward and no Adam call."
    )
    report: dict[str, Any] = {
        "schema": "taiji-record-relation-14-6-formal-v2",
        "step": "14.6",
        "date": "2026-11-02",
        "started_at": v1.get("started_at"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "passed": engineering_ok,
        "release_14_7": bool(release),
        "checks": checks,
        "declaration": v1["declaration"],
        "identity": identity,
        "data": v1["data"],
        "seeds_plan": v1["seeds_plan"],
        "candidates": candidates,
        "gates": {"task": task, "structure": structure, "content": content},
        "retention": retention,
        "cost": cost,
        "within": within,
        "type_check": v1["type_check"],
        "lint": v1["lint"],
        "confirm_read": False,
        "limits": limits,
        "correction": correction,
        "error": None,
    }
    report["required_checks"] = sum(1 for row in checks if row.get("required", True))
    report["failed_required"] = [
        row["id"] for row in checks if row.get("required", True) and not row["passed"]
    ]

    manifest_data: dict[str, Any] = {
        "schema": "taiji-record-relation-14-6-manifest-v2",
        "step": "14.6",
        "date": "2026-11-02",
        "correction": correction,
        "parent": v1_manifest["parent"],
        "gate_chain": v1_manifest["gate_chain"],
        "data": v1_manifest["data"],
        "seeds": v1_manifest["seeds"],
        "order_seeds": v1_manifest["order_seeds"],
        "arms": v1_manifest["arms"],
        "updates_per_arm": v1_manifest["updates_per_arm"],
        "batch": v1_manifest["batch"],
        "evaluation_endpoints": v1_manifest["evaluation_endpoints"],
        "candidates": [
            {
                "seed": candidate["seed"],
                "arm": candidate["arm"],
                "order_seed": candidate["order_seed"],
                "candidate_initial": candidate["candidate_initial"],
                "candidate_final": candidate["candidate_final"],
                "relation_initial": candidate["relation_initial"],
                "relation_final": candidate["relation_final"],
                "parameter_count": candidate["parameter_count"],
                "trainable_count": candidate["trainable_count"],
                "disabled": candidate["disabled"],
                "training": {
                    "recovered": candidate["training"].get("recovered"),
                    "updates": candidate["training"].get("updates"),
                    "adam_calls": candidate["training"].get("adam_calls"),
                    "summary": candidate["training"].get("summary"),
                },
                "step300_summary": candidate["step300"]["real"]["summary"],
            }
            for candidate in candidates
        ],
        "checkpoints": v1_manifest["checkpoints"],
        "retained": retained_rows,
        "retained_count": len(retained_rows),
        "gates": {"task": task, "structure": structure, "content": content},
        "retention": retention,
        "status": status,
        "release_14_7": bool(release),
        "cost": cost,
        "source_hashes": source_hashes,
        "confirm_read": False,
        "next": ("14.7 independent confirmation" if release else "14.8 closeout of this package"),
    }
    manifest_info = write_json(manifest_path, manifest_data)
    report["manifest"] = {
        "path": relative(manifest_path),
        "sha256": manifest_info["sha256"],
        "bytes": manifest_info["bytes"],
        "retained_count": len(retained_rows),
    }
    check(
        checks,
        "manifest_identity",
        manifest_info["sha256"] == sha256_text(manifest_path),
        {"path": report["manifest"]["path"], "sha256": manifest_info["sha256"]},
    )
    report["required_checks"] = sum(1 for row in checks if row.get("required", True))
    report["failed_required"] = [
        row["id"] for row in checks if row.get("required", True) and not row["passed"]
    ]
    report["passed"] = not report["failed_required"]
    if not report["passed"]:
        report["status"] = "engineering_failure"
        report["release_14_7"] = False
    info = write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "release_14_7": report["release_14_7"],
                "failed_required": report["failed_required"],
                "task_gate": task["passed"],
                "structure_gate": structure["passed"],
                "content_gate": content["passed"],
                "task": task["observed"],
                "structure": [
                    {
                        "seed": row["seed"],
                        "comparison": row["comparison"],
                        "gain": row["gain"],
                        "lower_95": row["bootstrap"]["lower_95"],
                        "passed": row["passed"],
                    }
                    for row in structure["rows"]
                ],
                "content": content["conditions"],
                "retention": retention,
                "counts": counts,
                "cost": {
                    "total_cpu_seconds": report["cost"]["total_cpu_seconds"],
                    "wall_seconds": report["cost"]["wall_seconds"],
                    "adam_calls": report["cost"]["adam_calls_total"],
                },
                "report": info["path"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
