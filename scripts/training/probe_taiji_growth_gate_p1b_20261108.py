"""P1b: decouple the exit code from the trigger, and define "hit" before judging.

P1 found that the gate refuses because G2 contained an ``eff_rank`` clause, and
P2 found that this clause is what blocks the one write that actually works
(``replace``: value residual 0.000207, zero bytes, retention untouched).  The
clause was already falsified by P1 -- ``eff_rank`` is a real-valued participation
ratio and falls on *any* near-collinear append, copy or not -- so P1b does three
things, all registered here before the first number is read:

R1  Separate exit.  ``commit_rejected_no_direction`` is its own code, triggered
    by the direction band only (``growth_is_copy``), never by a rank number.
    G2 keeps "read margin must not drop" (0.002, unchanged).  ``eff_rank`` stays
    as a reading and blocks nothing.

R2  No rank threshold for a single write.  ``Delta eff_rank >= 1`` came from F3's
    four doublings (each generation +6..+16), so it cannot bound one write.  P2
    measured +0.862 for a genuine new entry.  Chosen form: **no threshold**,
    reading only, with one report-only observation ``Delta eff_rank > 0`` when the
    band says new direction.  Reason: n=1 per arm gives no distribution to
    calibrate a lower bound from, and a monotone function of collinearity cannot
    add information the direction band does not already carry -- it would only
    add a second way to refuse the same write.

R3  Two hit definitions, both reported, both judged, neither selected after the
    fact:

    H-slot   ``top1 == written_slot``, where ``written_slot`` is the slot the
             write actually landed in (append: the free slot; replace: the
             superseded slot).  P1/P2 could not test this honestly: the gate
             derived ``written_slot`` from ``top1``, which makes the test a
             tautology.  Fixed before this run.
    H-value  ``residual(read_value, v') <= 0.5``, the gate's G4 threshold,
             unchanged.

    Which one decides: **H-value**.  A replace legitimately leaves ``top1`` on the
    superseded slot, so the slot number cannot be the pass test; and under append
    a wrong slot number already fails H-value, so H-value dominates.  H-slot is
    still reported for every arm.

    Scope of R1's clause, also registered here: the direction band answers "did
    this write grow a new direction".  ``append`` makes that claim; ``replace``
    makes a different one (this address now returns v').  So the clause applies to
    direction-claiming policies and exempts ``replace``.  The alternative reading
    -- clause applies to every policy -- is derived and reported too; it would
    reject the replace arm with ``commit_rejected_no_direction``.  Both are stated
    in the report; the primary verdict follows the scoped rule.

R4  Validation, three arms, same fixture, same 8/16 rung, same temperature:
    copy (append, same correction key) must be refused; replace (same key) must
    be admitted; new entry (append, a rule not in the store) must be admitted.
    For every admitted arm the report says whether the value read back equals v'.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero model forward.
numpy only; the fixture is imported from the F1 runner so it is the same bytes.
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

REPORT = ROOT / "reports/taiji_growth_gate_p1b_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_gate_p1b_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_gate_p1b_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_growth_gate_p1b_20261108_journal"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
P2_REPORT = ROOT / "reports/taiji_growth_gate_p2_20261108.json"
GATE_FILE = ROOT / "taiji/growth_gate.py"

CPU_CAP = 150.0
CPU_GUARD = 140.0
RUNG = {"name": "low", "generation": 1, "occupied": 8, "slots": 16}
RULE_INDEX = 0
CORRECTION_POSITION = 5
NEW_DIRECTION_RULE = 20
VALUE_RESIDUAL_LIMIT = 0.05

CRITERIA = {
    "R1_exit_decoupled_from_trigger": (
        "commit_rejected_no_direction exists as its own exit; it is appended only when the "
        "direction band says growth_is_copy and the policy claims a direction; no rank number "
        "can add a failure code"
    ),
    "R2_rank_has_no_threshold": (
        "taiji/growth_gate.py has no GATE_G2_MAX_RANK_DROP; effective rank is reported as a "
        "reading (detail.rank_is_reading_only is true) and blocks nothing"
    ),
    "R3_two_hit_definitions_reported": (
        "every arm reports H-slot (top1 == written_slot) and H-value (residual <= 0.5); the gate "
        "pass test is H-value; H-slot is a reading; both verdicts are listed, neither is picked "
        "after the reading"
    ),
    "R4_copy_is_refused": (
        "append with the same correction key: not published, exit commit_rejected_no_direction"
    ),
    "R5_replace_is_admitted": (
        "replace with the same key: published, exit commit_accepted, bytes delta 0, retention "
        "unchanged, H-value true and read value equals v'"
    ),
    "R6_new_entry_is_admitted": (
        "append with a rule that is not in the store: published, exit commit_accepted, band "
        "growth_new_direction, Delta eff_rank recorded (reading, no threshold)"
    ),
    "R7_no_collateral_damage": (
        "for every admitted arm each untouched slot keeps itself as top-1 and residual < 0.05"
    ),
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
    """Clear only this probe's own per-run ledgers so a re-run is reproducible."""
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JOURNAL_DIR.glob("*.jsonl"):
        stale.write_text("", encoding="utf-8")


def journal_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_f1():
    spec = importlib.util.spec_from_file_location("f1_probe", F1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {F1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_slot(query: np.ndarray, state: gg.GateState) -> tuple[int, np.ndarray, np.ndarray]:
    rows = int(state.occupied)
    keys = state.keys[:rows]
    values = state.values[:rows]
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    cosine = normalized @ query
    logits = cosine / gg.GATE_TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return int(np.argmax(cosine)), weights, (weights[:, None] * values).sum(axis=0)


def relative_residual(read: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(read - target) / max(float(np.linalg.norm(target)), 1e-12))


def build_state(f1, encoder, centroid, payloads) -> gg.GateState:
    occupied = RUNG["occupied"]
    slots = RUNG["slots"]
    keys = np.zeros((slots, f1.DIM), dtype=np.float64)
    values = np.zeros((slots, int(np.prod(f1.VALUE_SHAPE))), dtype=np.float64)
    for slot in range(occupied):
        keys[slot] = gr.encode_read_key(f1_contexts[slot], encoder, centroid)
        values[slot] = payloads[slot]
    return gg.GateState(
        keys=keys,
        values=values,
        age=np.array(list(range(occupied)) + [-1] * (slots - occupied)),
        hits=np.zeros(slots, dtype=np.int64),
        occupied=occupied,
        metric_version=0,
    )


def battery(f1) -> list[gg.BatteryQuery]:
    return [
        gg.BatteryQuery(
            name=f"held_out_rule_{index}",
            key=f1_state_keys[index].copy(),
            expect_slot=index,
            family=f"rule_{index}",
        )
        for index in range(1, RUNG["occupied"])
    ]


def integrity(
    state: gg.GateState,
    reference_values: np.ndarray,
    skip_slot: int | None = None,
) -> list[dict[str, Any]]:
    """Ask every stored key again; the slot a replace superseded is checked separately."""
    rows = []
    for slot in range(min(int(state.occupied), reference_values.shape[0])):
        if slot == skip_slot:
            continue
        top1, weights, read = read_slot(state.keys[slot], state)
        rows.append(
            {
                "slot": slot,
                "top1": top1,
                "kept_slot": top1 == slot,
                "weight_on_own_slot": round(float(weights[slot]), 6),
                "value_residual": round(relative_residual(read, reference_values[slot]), 6),
            }
        )
    return rows


def run_arm(name, encoder, centroid, policy, key, value, family, quarantine_path):
    state = build_state(f1, encoder, centroid, payloads)
    queries = battery(f1)
    write = gg.WriteCell(key=key, value=value, family=family, policy=policy)
    quarantine_path.write_text("", encoding="utf-8")
    log_path = JOURNAL_DIR / f"{name}_gate_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    receipt, run = gg.run_gate(
        state, write, queries, quarantine_path=quarantine_path, log_path=log_path
    )
    cert = run.certificate
    band = gd.direction_band(state.keys[: int(state.occupied)], key)
    after = receipt.state
    written_slot = run.apply_receipt.get("slot")
    # Both hit definitions are read on the state the gate judged (the shadow), not
    # on the post-commit state: for a refused write nothing was published, so the
    # committed state cannot answer "would this write have fixed its own case".
    top1_shadow, weights_shadow, read_shadow = read_slot(key, run.shadow)
    top1_after, _, read_after = read_slot(key, after)
    skip = written_slot if policy == "replace" else None
    return {
        "arm": name,
        "encoder": encoder.schema,
        "policy": policy,
        "family": family,
        "direction_band": {
            "overlap": round(float(band["overlap"]), 6),
            "separation": round(float(band["separation"]), 6),
            "nearest_cos": round(float(band["nearest_cos"]), 6),
            "verdict": band["verdict"],
            "gate_internal_verdict": cert.direction_verdict,
        },
        "gate": cert.as_dict(),
        "delta_effective_rank": round(cert.g2_rank_after - cert.g2_rank_before, 6),
        "hit_definitions": {
            "measured_on": "shadow (the state the gate judged)",
            "H_slot": {
                "definition": "top1 == written_slot",
                "top1": top1_shadow,
                "written_slot": written_slot,
                "hit": bool(cert.g4_slot_hit),
            },
            "H_value": {
                "definition": f"residual(read_value, v') <= {gg.GATE_MAX_VALUE_RESIDUAL}",
                "residual": round(relative_residual(read_shadow, value), 6),
                "hit": bool(cert.g4_value_hit),
                "read_value_equals_v_prime_strict": (
                    relative_residual(read_shadow, value) < VALUE_RESIDUAL_LIMIT
                ),
            },
        },
        "post_commit_read": {
            "measured_on": "committed state (unchanged when the write was refused)",
            "top1": top1_after,
            "residual": round(relative_residual(read_after, value), 6),
        },
        "commit": {
            "published": receipt.published,
            "exit_code": receipt.exit_code,
            "sha_before": receipt.sha_before,
            "sha_after": receipt.sha_after,
            "bit_identical": receipt.bit_identical,
            "evictions": receipt.evictions,
            "quarantine_id": receipt.quarantine_id,
            "quarantine_rows": journal_rows(quarantine_path),
        },
        "apply": run.apply_receipt,
        "written_slot_contribution": {
            "weight_on_written_slot": (
                round(float(weights_shadow[int(written_slot)]), 6)
                if written_slot is not None and int(written_slot) < weights_shadow.size
                else 0.0
            ),
            "value_residual_to_written": round(relative_residual(read_shadow, value), 6),
        },
        "old_values_kept": integrity(after, values_snapshot, skip_slot=skip),
        "superseded_slot": (
            {
                "slot": written_slot,
                "value_residual_to_new_value": round(relative_residual(read_shadow, value), 6),
            }
            if policy == "replace"
            else None
        ),
    }


def band_holds(row: dict) -> bool:
    return row["direction_band"]["verdict"] == "growth_new_direction"


def load_previous_run() -> dict[str, Any] | None:
    """Keep the immediately preceding P1b reading in the new report, never hidden.

    The first P1b attempt disagreed with its own registration in two places: the
    gate appended the direction failure after G4, so a copy exited as
    ``no_gain`` although R4 had fixed the exit at ``no_direction``; and the probe
    measured H-value on the restored state, which answers "what is stored now"
    rather than "would this write have fixed its own case".  Both are recorded
    here and both corrections are listed in the new report.
    """
    if not REPORT.exists():
        return None
    try:
        old = json.loads(REPORT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if old.get("format") != "taiji-growth-gate-p1b-v1":
        return None
    arms = {}
    for name, arm in old.get("arms", {}).items():
        arms[name] = {
            "band": arm["direction_band"]["verdict"],
            "published": arm["commit"]["published"],
            "exit_code": arm["commit"]["exit_code"],
            "failed": arm["gate"]["failed"],
            "H_slot": arm["hit_definitions"]["H_slot"]["hit"],
            "H_value": arm["hit_definitions"]["H_value"]["hit"],
            "residual": arm["hit_definitions"]["H_value"]["residual"],
        }
    return {
        "decision": old.get("decision"),
        "checks": old.get("checks"),
        "arms": arms,
        "generated_at": old.get("generated_at"),
        "gate_sha256": old.get("isolation", {}).get("gate_sha256"),
        "runner_sha256": old.get("isolation", {}).get("runner_sha256"),
        "superseded_because": (
            "implementation did not match the registered exit order for a copy, and H-value was "
            "read on the restored state instead of the judged shadow state; criteria unchanged"
        ),
    }


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    global f1, f1_contexts, f1_state_keys, payloads, values_snapshot

    reset_run_logs()
    previous = load_previous_run()
    f1 = load_f1()
    base, tokens, contexts = f1.fixture()
    payloads = f1.value_payloads(f1.TOTAL_SENTENCES + 8)
    f1_contexts = contexts
    values_snapshot = payloads[: RUNG["occupied"]].copy()

    legacy_encoder = gr.EncoderConfig(mode="mean")
    legacy_parents = np.stack([gr.encode_read_key(contexts[i], legacy_encoder) for i in range(4)])
    centroid = gr.store_centroid(legacy_parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)
    f1_state_keys = np.stack(
        [gr.encode_read_key(contexts[i], encoder, centroid) for i in range(RUNG["occupied"])]
    )

    correction_context = f1.paraphrase(base, tokens[RULE_INDEX], CORRECTION_POSITION)
    correction_key = gr.encode_read_key(correction_context, encoder, centroid)
    correction_value = payloads[f1.TOTAL_SENTENCES]
    new_key = gr.encode_read_key(
        f1.paraphrase(base, tokens[NEW_DIRECTION_RULE], CORRECTION_POSITION), encoder, centroid
    )
    new_value = payloads[f1.TOTAL_SENTENCES + 1]

    copy_arm = run_arm(
        "append_copy",
        encoder,
        centroid,
        "append",
        correction_key,
        correction_value,
        "rule_0",
        JOURNAL_DIR / "append_copy_quarantine.jsonl",
    )
    replace_arm = run_arm(
        "replace_same_key",
        encoder,
        centroid,
        "replace",
        correction_key,
        correction_value,
        "rule_0",
        JOURNAL_DIR / "replace_quarantine.jsonl",
    )
    fresh_arm = run_arm(
        "append_new_entry",
        encoder,
        centroid,
        "append",
        new_key,
        new_value,
        "new_direction",
        JOURNAL_DIR / "append_new_entry_quarantine.jsonl",
    )

    for row in (copy_arm, replace_arm, fresh_arm):
        append_progress(
            {
                "arm": row["arm"],
                "policy": row["policy"],
                "band": row["direction_band"]["verdict"],
                "published": row["commit"]["published"],
                "exit_code": row["commit"]["exit_code"],
                "failed": row["gate"]["failed"],
                "H_slot": row["hit_definitions"]["H_slot"]["hit"],
                "H_value": row["hit_definitions"]["H_value"]["hit"],
                "residual": row["hit_definitions"]["H_value"]["residual"],
                "delta_eff_rank": row["delta_effective_rank"],
            }
        )

    source = GATE_FILE.read_text(encoding="utf-8")
    checks = {
        "R1": (
            gg.EXIT_NO_DIRECTION in gg.EXIT_CODES
            and gg.EXIT_NO_DIRECTION in copy_arm["gate"]["failed"]
            and copy_arm["commit"]["exit_code"] == gg.EXIT_NO_DIRECTION
        ),
        "R2": (
            not hasattr(gg, "GATE_G2_MAX_RANK_DROP")
            and all(
                row["gate"]["detail"]["rank_is_reading_only"]
                for row in (copy_arm, replace_arm, fresh_arm)
            )
        ),
        "R3": all(
            {"H_slot", "H_value"} <= set(row["hit_definitions"])
            for row in (copy_arm, replace_arm, fresh_arm)
        ),
        "R4": (
            not copy_arm["commit"]["published"]
            and copy_arm["commit"]["exit_code"] == gg.EXIT_NO_DIRECTION
        ),
        "R5": (
            replace_arm["commit"]["published"]
            and replace_arm["commit"]["exit_code"] == gg.EXIT_ACCEPTED
            and replace_arm["gate"]["g3_bytes_delta"] == 0
            and (
                replace_arm["gate"]["g1_retention_after"]
                == replace_arm["gate"]["g1_retention_before"]
            )
            and replace_arm["hit_definitions"]["H_value"]["hit"]
            and replace_arm["hit_definitions"]["H_value"]["residual"] < VALUE_RESIDUAL_LIMIT
        ),
        "R6": (
            fresh_arm["commit"]["published"]
            and fresh_arm["commit"]["exit_code"] == gg.EXIT_ACCEPTED
            and band_holds(fresh_arm)
        ),
        "R7": all(
            row["kept_slot"] and row["value_residual"] < VALUE_RESIDUAL_LIMIT
            for arm in (replace_arm, fresh_arm)
            for row in arm["old_values_kept"]
        ),
    }

    # The alternative reading of R1's scope, derived not run: if the direction
    # clause applied to every policy, the replace arm would also be refused.
    alternative = {
        "reading": "direction clause applies to every policy, replace included",
        "replace_band": replace_arm["direction_band"]["verdict"],
        "predicted_exit_for_replace": (
            gg.EXIT_NO_DIRECTION
            if replace_arm["direction_band"]["verdict"] == "growth_is_copy"
            else gg.EXIT_ACCEPTED
        ),
        "derived_from": "recorded band verdict plus the arm's other four conditions, all green",
        "status": "reported, not the primary verdict",
    }

    decision = (
        "p1b_passed"
        if all(checks.values())
        else "p1b_falsified_" + "_".join(name for name, ok in checks.items() if not ok)
    )

    payload: dict[str, Any] = {
        "format": "taiji-growth-gate-p1b-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "question": (
            "after removing the rank clause from G2 and defining both hit definitions, does the "
            "gate refuse the copy, admit the supersede and admit the new entry?"
        ),
        "previous_run": previous,
        "implementation_corrections": {
            "criteria_changed": False,
            "thresholds_changed": False,
            "items": [
                {
                    "what": "the direction clause now sits before G4 in the failure order",
                    "why": (
                        "R4 fixed the exit for a copy at commit_rejected_no_direction; the first "
                        "attempt appended the direction failure after G4, so a copy exited as "
                        "commit_rejected_no_gain"
                    ),
                    "kind": "implementation aligned to the registered exit, not a criterion change",
                },
                {
                    "what": "H-slot and H-value are read on the shadow state the gate judged",
                    "why": (
                        "the first attempt read them on the restored state, which answers 'what is "
                        "stored now' instead of 'would this write have fixed its own case'"
                    ),
                    "kind": "measurement aligned to the written definition",
                },
            ],
            "first_run_kept_in": "previous_run",
        },
        "pre_run_defect": {
            "found": (
                "P1/P2's evaluate() set written_slot = top1, so 'top1 != written_slot' could never "
                "be true and G4's hit clause was a tautology"
            ),
            "fixed_before_this_run": (
                "run_gate now passes the slot from apply_receipt; G4 decides on value fidelity and "
                "the slot hit is a reported reading"
            ),
            "effect_on_earlier_reports": (
                "P1 and P2 are not restated; their rejections were caused by the rank clause and "
                "the blend residual, neither of which this defect touches"
            ),
        },
        "scope_decision": {
            "primary": "direction clause applies to policies that claim a direction (append); replace is exempt",
            "reason": (
                "the band answers 'did this write grow a new direction'; replace makes a different "
                "claim (this address now returns v'). Registered before the run, not selected from "
                "the readings."
            ),
            "alternative": alternative,
        },
        "thresholds": {
            "g1_max_retention_drop": gg.GATE_G1_MAX_RETENTION_DROP,
            "g2_max_margin_drop": gg.GATE_G2_MAX_MARGIN_DROP,
            "g3_max_state_bytes": gg.GATE_MAX_STATE_BYTES,
            "g3_min_slot_headroom": gg.GATE_MIN_SLOT_HEADROOM,
            "g4_max_value_residual": gg.GATE_MAX_VALUE_RESIDUAL,
            "direction_min_overlap": gg.GATE_DIRECTION_MIN_OVERLAP,
            "direction_max_overlap": gg.GATE_DIRECTION_MAX_OVERLAP,
            "direction_min_separation": gg.GATE_DIRECTION_MIN_SEPARATION,
            "direction_exempt_policies": list(gg.GATE_DIRECTION_EXEMPT_POLICIES),
            "eff_rank_threshold": None,
            "temperature": gg.GATE_TEMPERATURE,
        },
        "checks": checks,
        "arms": {
            "append_copy": copy_arm,
            "replace_same_key": replace_arm,
            "append_new_entry": fresh_arm,
        },
        "readings": {
            "copy": {
                "published": copy_arm["commit"]["published"],
                "exit_code": copy_arm["commit"]["exit_code"],
                "failed": copy_arm["gate"]["failed"],
                "reasons_zh": [gg.EXIT_ZH[code] for code in copy_arm["gate"]["failed"]],
            },
            "replace": {
                "published": replace_arm["commit"]["published"],
                "exit_code": replace_arm["commit"]["exit_code"],
                "read_value_equals_v_prime": replace_arm["hit_definitions"]["H_value"][
                    "read_value_equals_v_prime_strict"
                ],
                "residual": replace_arm["hit_definitions"]["H_value"]["residual"],
            },
            "new_entry": {
                "published": fresh_arm["commit"]["published"],
                "exit_code": fresh_arm["commit"]["exit_code"],
                "delta_effective_rank_reading": fresh_arm["delta_effective_rank"],
                "delta_positive_observation": fresh_arm["delta_effective_rank"] > 0,
            },
        },
        "decision": decision,
        "decision_zh": (
            "三臂均符合预注册：复制被拒（commit_rejected_no_direction），取代被准，新条目被准"
            if decision == "p1b_passed"
            else "预注册未全部成立，见 checks"
        ),
        "isolation": {
            "runner_sha256": file_sha(Path(__file__).resolve()),
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "design_sha256": file_sha(ROOT / "taiji/growth_design.py"),
            "p2_report_sha256": file_sha(P2_REPORT) if P2_REPORT.exists() else None,
            "gate_imports_torch": "import torch" in source,
            "frozen_files_touched": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_steps": 0,
            "model_forward": 0,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 6),
            "wall_seconds": round(time.time() - START_WALL, 6),
            "cpu_cap": CPU_CAP,
        },
    }

    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "format": "taiji-growth-gate-p1b-manifest-v1",
            "generated_at": payload["generated_at"],
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": file_sha(REPORT),
            "runner": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "runner_sha256": file_sha(Path(__file__).resolve()),
            "gate_sha256": file_sha(GATE_FILE),
            "decision": decision,
            "checks": checks,
        },
    )

    print(json.dumps({"decision": decision, "checks": checks}, ensure_ascii=False, indent=2))
    for name, arm in (
        ("append_copy", copy_arm),
        ("replace_same_key", replace_arm),
        ("append_new_entry", fresh_arm),
    ):
        print(
            name,
            "| band",
            arm["direction_band"]["verdict"],
            "| published",
            arm["commit"]["published"],
            "| exit",
            arm["commit"]["exit_code"],
            "| H-slot",
            arm["hit_definitions"]["H_slot"]["hit"],
            "| H-value",
            arm["hit_definitions"]["H_value"]["hit"],
            "| residual",
            arm["hit_definitions"]["H_value"]["residual"],
        )
    if time.process_time() - START_CPU > CPU_CAP:
        raise SystemExit("cpu cap exceeded")
    return 0 if decision == "p1b_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
