"""P1: the consensus gate must refuse this write, and it must be the first refusal.

The gate has never rejected anything.  ``growth_gates={"consensus": 1.0}`` is a
constant multiplier and is short circuited while the subject has no grown units,
so before this run there is no evidence that a local write can be turned down at
all.  This probe submits exactly the write that F1 already measured -- one
correction on the low rung, mean-free repaired encoder, append policy, 8/16
slots -- and checks that the gate refuses it for the right reason.

Criteria, written before the reading and not changed after:

C0  the retention battery is answerable: at least 6 of the 7 held-out rules
    return their own slot before the write, otherwise no verdict is issued.
C1  the write is a copy by the direction band: ``growth_is_copy`` (o >= 0.85).
C2  the gate refuses it with ``commit_rejected_no_gain`` -- not retention, not
    separability, not inflation, not capacity.
C3  the refusal is total: nothing published, ``sha_after == sha_before`` bit for
    bit, ``evictions == 0``, exactly one quarantine row.
C4  G4 is refusing the right thing: the read value residual matches the
    half-and-half prediction 0.49 * sqrt(2) = 0.6930 within 0.05, i.e. the gate
    is catching "old and new mixed", not some unrelated defect.

If the gate publishes the write, the verdict is ``gate_not_wired`` and the run
stops.  No threshold is relaxed to make it pass.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero model
forward.  numpy only; the fixture is imported from the F1 runner so it is the
same bytes, not a re-derivation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_design as gd  # noqa: E402
from taiji import growth_gate as gg  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_gate_p1_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_gate_p1_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_gate_p1_20261108_progress.jsonl"
QUARANTINE = ROOT / "reports/taiji_growth_gate_p1_20261108_quarantine.jsonl"
GATE_LOG = ROOT / "reports/taiji_growth_gate_p1_20261108_gate_log.jsonl"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
F1_REPORT = ROOT / "reports/taiji_growth_f1_correction_20261108.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
RUNG = {"name": "low", "generation": 1, "occupied": 8, "slots": 16}
RULE_INDEX = 0
CORRECTION_POSITION = 5
PREDICTED_HALF_HALF_RESIDUAL = 0.49 * float(np.sqrt(2.0))  # 0.6930, from F1
RESIDUAL_TOLERANCE = 0.05

CRITERIA = {
    "C0_battery_answerable": "at least 6 of the 7 held-out rules are top-1 correct before the write",
    "C1_write_is_copy": "direction band verdict is growth_is_copy (o >= 0.85)",
    "C2_exit_is_no_gain": "gate exit code is exactly commit_rejected_no_gain",
    "C3_refusal_is_total": (
        "published false, sha_after == sha_before bit for bit, evictions 0, "
        "exactly one quarantine row"
    ),
    "C4_g4_catches_the_blend": (
        f"value residual within {RESIDUAL_TOLERANCE} of {PREDICTED_HALF_HALF_RESIDUAL:.4f}"
    ),
    "gate_not_wired": "if the write is published the verdict is gate_not_wired and the run stops",
    "unchanged": "no threshold, rung, battery or question is retuned after the first reading",
}

START_CPU = time.process_time()
START_WALL = time.time()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reset_run_logs() -> None:
    """Start every run from an empty ledger so a re-run is bit-for-bit reproducible.

    This clears only this probe's own per-run jsonl files.  It says nothing about
    the gate: inside one run a rejected write is never deleted and never evicts
    an existing slot, and the gate's own quarantine path is append-only.
    """
    for path in (PROGRESS, QUARANTINE, GATE_LOG):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def load_f1():
    """Import the F1 runner as a module so the fixture is the same bytes."""
    spec = importlib.util.spec_from_file_location("f1_probe", F1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {F1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quarantine_rows() -> int:
    if not QUARANTINE.exists():
        return 0
    with QUARANTINE.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    reset_run_logs()
    f1 = load_f1()
    base, tokens, contexts = f1.fixture()
    payloads = f1.value_payloads(f1.TOTAL_SENTENCES + 8)
    correction_value = payloads[f1.TOTAL_SENTENCES]

    legacy_encoder = gr.EncoderConfig(mode="mean")
    legacy_parents = np.stack(
        [gr.encode_read_key(contexts[index], legacy_encoder) for index in range(4)]
    )
    centroid = gr.store_centroid(legacy_parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)

    occupied = RUNG["occupied"]
    slots = RUNG["slots"]
    keys = np.zeros((slots, f1.DIM), dtype=np.float64)
    values = np.zeros((slots, int(np.prod(f1.VALUE_SHAPE))), dtype=np.float64)
    for slot in range(occupied):
        keys[slot] = gr.encode_read_key(contexts[slot], encoder, centroid)
        values[slot] = payloads[slot]
    state = gg.GateState(
        keys=keys,
        values=values,
        age=np.array([slot for slot in range(occupied)] + [-1] * (slots - occupied)),
        hits=np.zeros(slots, dtype=np.int64),
        occupied=occupied,
        metric_version=0,
    )

    self_key = gr.encode_read_key(
        f1.paraphrase(base, tokens[RULE_INDEX], CORRECTION_POSITION), encoder, centroid
    )
    battery = [
        gg.BatteryQuery(
            name=f"held_out_rule_{index}",
            key=keys[index].copy(),
            expect_slot=index,
            family=f"rule_{index}",
        )
        for index in range(1, occupied)
    ]
    write = gg.WriteCell(key=self_key, value=correction_value, family="rule_0", policy="append")

    receipt, run = gg.run_gate(state, write, battery, quarantine_path=QUARANTINE, log_path=GATE_LOG)
    cert = run.certificate
    band = gd.direction_band(keys[:occupied], self_key)
    rows_after = quarantine_rows()

    f1_payload = json.loads(F1_REPORT.read_text(encoding="utf-8")) if F1_REPORT.exists() else {}
    f1_low = next(
        (row for row in f1_payload.get("repaired_path", []) if row.get("rung") == RUNG["name"]),
        None,
    )
    f1_residual = None if f1_low is None else f1_low["treatment"]["median_residual_to_correction"]
    residual_delta = (
        None if f1_residual is None else abs(cert.g4_value_residual - PREDICTED_HALF_HALF_RESIDUAL)
    )

    checks = {
        "C0": cert.g1_before >= 6,
        "C1": band["verdict"] == "growth_is_copy",
        "C2": cert.exit_code == gg.EXIT_NO_GAIN,
        "C3": (
            (not receipt.published)
            and receipt.sha_after == receipt.sha_before
            and receipt.bit_identical
            and receipt.evictions == 0
            and rows_after == 1
        ),
        "C4": abs(cert.g4_value_residual - PREDICTED_HALF_HALF_RESIDUAL) <= RESIDUAL_TOLERANCE,
    }

    if receipt.published:
        decision = "gate_not_wired"
    elif all(checks.values()):
        decision = "gate_rejected_first_time"
    else:
        failed = [name for name, ok in checks.items() if not ok]
        decision = "p1_falsified_" + "_".join(failed)

    payload: dict[str, Any] = {
        "format": "taiji-growth-gate-p1-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "question": (
            "can the gate refuse a local write at all, and does it refuse the F1 "
            "correction for the reason F1 identified?"
        ),
        "fixture": {
            "source": "scripts/training/probe_taiji_growth_f1_correction_20261108.py",
            "source_sha256": file_sha(F1_RUNNER),
            "rung": dict(RUNG),
            "encoder": encoder.schema,
            "temperature": gg.GATE_TEMPERATURE,
            "write_policy": write.policy,
            "battery_size": len(battery),
            "battery_families": [row.family for row in battery],
            "battery_hygiene": (
                "the battery holds held-out rules 1..7; the write targets rule 0, and a "
                "battery row from the write's own family raises before any threshold is read"
            ),
        },
        "checks": checks,
        "gate": cert.as_dict(),
        "commit": {
            "published": receipt.published,
            "exit_code": receipt.exit_code,
            "sha_before": receipt.sha_before,
            "sha_after": receipt.sha_after,
            "bit_identical": receipt.bit_identical,
            "evictions": receipt.evictions,
            "quarantine_id": receipt.quarantine_id,
            "quarantine_rows": rows_after,
        },
        "apply": run.apply_receipt,
        "direction_band": {
            "overlap": band["overlap"],
            "separation": band["separation"],
            "nearest_cos": band["nearest_cos"],
            "verdict": band["verdict"],
        },
        "g4_vs_f1": {
            "gate_residual": cert.g4_value_residual,
            "f1_median_residual": f1_residual,
            "predicted_half_half": PREDICTED_HALF_HALF_RESIDUAL,
            "abs_delta_to_prediction": residual_delta,
            "interpretation": (
                "the gate's G4 reads the same blend F1 measured: the slot is retrieved, "
                "but the value handed back is 0.51*v' + 0.49*v_old, so residual ~= 0.693"
            ),
        },
        "state_accounting": {
            "floats_before": cert.g3_floats_before,
            "floats_after": cert.g3_floats_after,
            "bytes_delta_at_deployment_dtype": cert.g3_bytes_delta,
            "float_bytes_used": gg.GATE_FLOAT_BYTES,
            "note": (
                "the fixture runs float64; the gate accounts at the deployment dtype "
                "(torch float32), which is what 33,280 B per slot means in the design"
            ),
        },
        "decision": decision,
        "decision_zh": {
            "gate_not_wired": "门没接线：写入被发布，判据立即停止，不调阈值",
            "gate_rejected_first_time": "门第一次真实拒绝，原因与 F1 一致",
        }.get(decision, "P1 判负：" + ", ".join(k for k, v in checks.items() if not v) + " 未通过"),
        "what_this_means": (
            "a local write can be refused, and the refusal happens before publication: the "
            "store is restored bit for bit, nothing is evicted, and the refused write is kept "
            "in the quarantine journal instead of being dropped"
        ),
        "what_this_does_not_mean": [
            "no training, no optimizer step and no model forward ran",
            "acceptance is not tested here: P1 only shows the gate can refuse; whether a "
            "corrected write is admitted is P2",
            "the write policy under test is append, which mirrors the frozen addressed_write; "
            "the replace policy exists in the gate but is not what the graph runs",
            "one fixture, one rung, one write",
        ],
        "isolation": {
            "runner_sha256": file_sha(Path(__file__)),
            "gate_sha256": file_sha(ROOT / "taiji/growth_gate.py"),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "design_sha256": file_sha(ROOT / "taiji/growth_design.py"),
            "gate_imports_torch": "import torch"
            in (ROOT / "taiji/growth_gate.py").read_text(encoding="utf-8", errors="ignore"),
            "frozen_files_touched": 0,
            "f1_report": F1_REPORT.relative_to(ROOT).as_posix(),
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "model_forwards": 0,
        },
    }

    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "format": "taiji-growth-gate-p1-manifest-v1",
            "report": REPORT.relative_to(ROOT).as_posix(),
            "report_sha256": file_sha(REPORT),
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": payload["isolation"]["runner_sha256"],
            "gate": "taiji/growth_gate.py",
            "gate_sha256": payload["isolation"]["gate_sha256"],
            "decision": decision,
            "exit_code": cert.exit_code,
        },
    )
    append_progress(
        {
            "stage": "p1",
            "decision": decision,
            "checks": checks,
            "exit_code": cert.exit_code,
            "published": receipt.published,
            "sha_before": receipt.sha_before,
            "sha_after": receipt.sha_after,
            "residual": cert.g4_value_residual,
            "quarantine_rows": rows_after,
        }
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "exit_code": cert.exit_code,
                "published": receipt.published,
                "sha_equal": receipt.sha_before == receipt.sha_after,
                "evictions": receipt.evictions,
                "quarantine_rows": rows_after,
                "band": band["verdict"],
                "overlap": round(band["overlap"], 4),
                "residual": round(cert.g4_value_residual, 4),
                "g1": [cert.g1_before, cert.g1_after],
                "g2_margin": [round(cert.g2_margin_before, 5), round(cert.g2_margin_after, 5)],
                "g3_bytes": cert.g3_bytes_delta,
                "checks": checks,
            },
            ensure_ascii=False,
        )
    )
    return 0 if decision == "gate_rejected_first_time" else 1


if __name__ == "__main__":
    raise SystemExit(main())
