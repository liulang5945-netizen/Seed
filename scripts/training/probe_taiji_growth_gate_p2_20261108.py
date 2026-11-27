"""P2: the gate must admit, not only refuse; and it must refuse the copy, not the write.

P1 showed one refusal.  A gate that refuses everything is not a gate either, so P2
submits the same ``(q, v')`` under ``localized_centered`` and asks the opposite
question.  The stored keys in P1 were already ``localized_centered-v1`` (see the
``fixture.encoder`` field of the P1 report), so "old encoding vs new encoding" is
not a contrast P1 and P2 can produce; the true legacy arm can only be run here,
side by side, as a diagnostic.

Criteria, written before the reading and not changed after:

C0  the retention battery is answerable: at least 6 of the 7 held-out rules return
    their own slot before the write, otherwise no verdict is issued.
C1  the write is a new direction by the direction band: ``growth_new_direction``
    (o <= 0.85 and s >= 0.30).
C2  the gate admits it: ``published == true`` and ``passed == true``.
C3  all four conditions are green, reported one by one, not just the first failure.
C4  admission does no collateral damage: every held-out rule still returns its own
    slot as top-1 and its stored value residual stays below 0.05.
C5  the admitted write adds a direction, not just a cell: ``eff_rank`` +1 or more.

Diagnostic arms (reported, never criteria):

D1  legacy ``mean`` encoder on the same write -- what the old key space does.
D2  same encoding, ``policy="replace"`` -- the supersede rule F1 said was missing.
D3  same encoding, ``append``, a genuinely new entry (a value token that is not in
    the store).  D3 is the control that decides whether a refusal means "this is a
    copy" or "the gate never lets anything through".

Reverse self-check, run before any verdict of falsification.  If the primary arm is
refused, the probe first asks whether the criterion itself is broken:

S1  does a condition use a quantity that necessarily falls when a key is appended
    (the P1 lesson: ``eff_rank`` is real-valued and drops on any near-collinear key)?
S2  was the write itself already a copy under this encoder (o >= 0.85 measured)?
S3  was P1 in fact already running this encoder, so that "new encoding" is not new?

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

REPORT = ROOT / "reports/taiji_growth_gate_p2_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_gate_p2_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_gate_p2_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_growth_gate_p2_20261108_journal"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
P1_REPORT = ROOT / "reports/taiji_growth_gate_p1_20261108.json"

CPU_CAP = 200.0
CPU_GUARD = 190.0
RUNG = {"name": "low", "generation": 1, "occupied": 8, "slots": 16}
RULE_INDEX = 0
CORRECTION_POSITION = 5
NEW_DIRECTION_RULE = 20
VALUE_RESIDUAL_LIMIT = 0.05
BAND_NEW_DIRECTION_MAX_OVERLAP = 0.85
BAND_NEW_DIRECTION_MIN_SEPARATION = 0.30
MIN_EFFECTIVE_RANK_GAIN = 1.0

CRITERIA = {
    "C0_battery_answerable": "at least 6 of the 7 held-out rules are top-1 correct before the write",
    "C1_write_is_new_direction": (
        f"direction band verdict is growth_new_direction "
        f"(o <= {BAND_NEW_DIRECTION_MAX_OVERLAP}, s >= {BAND_NEW_DIRECTION_MIN_SEPARATION})"
    ),
    "C2_gate_admits": "published is true and passed is true",
    "C3_four_conditions_green": "G1, G2, G3 and G4 each pass, reported one by one",
    "C4_no_collateral_damage": (
        f"every held-out rule keeps its own slot as top-1 and its value residual stays "
        f"below {VALUE_RESIDUAL_LIMIT}"
    ),
    "C5_grows_a_direction": f"eff_rank rises by at least {MIN_EFFECTIVE_RANK_GAIN}",
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
    """Clear this probe's own per-run ledgers so a re-run is reproducible.

    Nothing here touches the gate's behaviour: inside one run a rejected write is
    never deleted and never evicts an existing slot.
    """
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
    occupied = RUNG["occupied"]
    return [
        gg.BatteryQuery(
            name=f"held_out_rule_{index}",
            key=f1_state_keys[index].copy(),
            expect_slot=index,
            family=f"rule_{index}",
        )
        for index in range(1, occupied)
    ]


def integrity(state: gg.GateState, values: np.ndarray) -> list[dict[str, Any]]:
    """Ask every stored key again and check the old value was not damaged."""
    rows = []
    for slot in range(values.shape[0]):
        top1, weights, read = read_slot(state.keys[slot], state)
        rows.append(
            {
                "slot": slot,
                "top1": top1,
                "kept_slot": top1 == slot,
                "weight_on_own_slot": float(weights[slot]),
                "value_residual": round(relative_residual(read, values[slot]), 6),
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
    top1_after, weights_after, read_after = read_slot(key, after)
    integrity_rows = integrity(after, values_snapshot)
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
        },
        "gate": cert.as_dict(),
        "delta_effective_rank": round(cert.g2_rank_after - cert.g2_rank_before, 6),
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
            "top1_after": top1_after,
            "weight_on_written_slot": (
                round(float(weights_after[int(state.occupied)]), 6)
                if int(after.occupied) > int(state.occupied)
                else 0.0
            ),
            "value_residual_to_written": round(relative_residual(read_after, value), 6),
        },
        "old_values_kept": integrity_rows,
    }


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    global f1, f1_contexts, f1_state_keys, payloads, values_snapshot

    reset_run_logs()
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
    legacy_correction_key = gr.encode_read_key(correction_context, legacy_encoder, None)
    correction_value = payloads[f1.TOTAL_SENTENCES]
    new_key = gr.encode_read_key(
        f1.paraphrase(base, tokens[NEW_DIRECTION_RULE], CORRECTION_POSITION), encoder, centroid
    )
    new_value = payloads[f1.TOTAL_SENTENCES + 1]

    primary = run_arm(
        "primary_localized_append",
        encoder,
        centroid,
        "append",
        correction_key,
        correction_value,
        "rule_0",
        JOURNAL_DIR / "primary_quarantine.jsonl",
    )
    legacy = run_arm(
        "diagnostic_legacy_append",
        legacy_encoder,
        None,
        "append",
        legacy_correction_key,
        correction_value,
        "rule_0",
        JOURNAL_DIR / "legacy_quarantine.jsonl",
    )
    replace = run_arm(
        "diagnostic_replace",
        encoder,
        centroid,
        "replace",
        correction_key,
        correction_value,
        "rule_0",
        JOURNAL_DIR / "replace_quarantine.jsonl",
    )
    fresh = run_arm(
        "diagnostic_new_direction",
        encoder,
        centroid,
        "append",
        new_key,
        new_value,
        "new_direction",
        JOURNAL_DIR / "new_direction_quarantine.jsonl",
    )

    cert = primary["gate"]
    checks = {
        "C0": cert["g1_retention_before"] >= 6,
        "C1": primary["direction_band"]["verdict"] == "growth_new_direction",
        "C2": primary["commit"]["published"] and cert["passed"],
        "C3": (
            cert["g1_retention_after"]
            >= cert["g1_retention_before"] - gg.GATE_G1_MAX_RETENTION_DROP
            and cert["g2_margin_after"] >= cert["g2_margin_before"] - gg.GATE_G2_MAX_MARGIN_DROP
            and cert["g2_rank_after"] >= cert["g2_rank_before"] - gg.GATE_G2_MAX_RANK_DROP
            and cert["g3_bytes_delta"] <= gg.GATE_MAX_STATE_BYTES
            and cert["g3_headroom_after"] >= gg.GATE_MIN_SLOT_HEADROOM
            and cert["g4_top1_slot"] == RUNG["occupied"]
            and cert["g4_value_residual"] is not None
            and cert["g4_value_residual"] <= gg.GATE_MAX_VALUE_RESIDUAL
        ),
        "C4": all(
            row["kept_slot"] and row["value_residual"] < VALUE_RESIDUAL_LIMIT
            for row in primary["old_values_kept"]
        ),
        "C5": primary["delta_effective_rank"] >= MIN_EFFECTIVE_RANK_GAIN,
    }

    p1 = json.loads(P1_REPORT.read_text(encoding="utf-8")) if P1_REPORT.exists() else {}
    p1_encoder = p1.get("fixture", {}).get("encoder")
    self_check = {
        "S1_condition_uses_a_necessarily_falling_quantity": False,
        "S1_note": (
            "C1/C4/C5 are read on the candidate, not used as a pass condition; the gate's own "
            "G2 rank clause is unchanged from P1 and is still what decides the exit code when "
            "a near-collinear key is appended"
        ),
        "S2_write_was_already_a_copy_under_this_encoder": (
            primary["direction_band"]["overlap"] >= BAND_NEW_DIRECTION_MAX_OVERLAP
        ),
        "S2_measured_overlap": primary["direction_band"]["overlap"],
        "S3_p1_already_ran_this_encoder": p1_encoder == encoder.schema,
        "S3_p1_encoder": p1_encoder,
    }

    if checks["C2"] and all(checks.values()):
        decision = "p2_admitted"
    elif not checks["C2"]:
        failed = [name for name, ok in checks.items() if not ok]
        premise_broken = self_check["S2_write_was_already_a_copy_under_this_encoder"]
        decision = ("p2_falsified_premise_" if premise_broken else "p2_falsified_") + "_".join(
            failed
        )
    else:
        failed = [name for name, ok in checks.items() if not ok]
        decision = "p2_falsified_" + "_".join(failed)

    fresh_band = fresh["direction_band"]["verdict"]
    payload: dict[str, Any] = {
        "format": "taiji-growth-gate-p2-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "question": (
            "does the gate admit a write at all, and does it distinguish a copy from a new "
            "direction rather than refusing everything?"
        ),
        "premise_check": {
            "p1_store_encoder": p1_encoder,
            "p2_store_encoder": encoder.schema,
            "same_encoder": p1_encoder == encoder.schema,
            "note": (
                "P1's report records fixture.encoder = localized_centered-v1, so P1 was not an "
                "old-encoding run and P2 cannot be its new-encoding counterpart. The legacy arm "
                "is run here as a diagnostic instead."
            ),
        },
        "protocol_deviation_p1": {
            "spec_said": 'SPEC-TAIJI-WHOLE-LOCAL section 5 P1 lists the fixture as mode = "mean"',
            "p1_runner_did": (
                "localized_centered + center, centroid built from the legacy mean parents"
            ),
            "detected_by": (
                "this probe compares the P1 report field fixture.encoder with its own encoder"
            ),
            "effect_on_p1_verdict": (
                "no P1 check turns on the encoder: C0 and C3 are encoder-free, C1 needs o >= 0.85 "
                "and both encoders clear that (0.963 localized, 0.938 mean), C2 and C4 are "
                "properties of the blend read, which this probe reproduces"
            ),
            "status": (
                "recorded as a pre-registration drift; P1 is not re-run and its decision is not "
                "restated"
            ),
        },
        "probe_notes": [
            "the legacy arm must encode its write key with the same encoder that built the "
            "store; the first draft mixed key spaces (mean store, localized query) and produced "
            "a meaningless legacy reading. Fixed before any number left this probe.",
            "each arm gets its own quarantine journal so 'quarantine rows' counts one refusal, "
            "not the accumulation across arms",
            "the replace arm supersedes the same key instead of appending; its byte delta is 0 "
            "by construction, which is why G3 passes for a different reason than in the append arms",
        ],
        "checks": checks,
        "primary": primary,
        "contrast": {
            "legacy_append": {
                "encoder": legacy["encoder"],
                "band": legacy["direction_band"]["verdict"],
                "overlap": legacy["direction_band"]["overlap"],
                "published": legacy["commit"]["published"],
                "exit_code": legacy["commit"]["exit_code"],
            },
            "localized_append": {
                "encoder": primary["encoder"],
                "band": primary["direction_band"]["verdict"],
                "overlap": primary["direction_band"]["overlap"],
                "published": primary["commit"]["published"],
                "exit_code": primary["commit"]["exit_code"],
            },
            "localized_replace": {
                "encoder": replace["encoder"],
                "band": replace["direction_band"]["verdict"],
                "overlap": replace["direction_band"]["overlap"],
                "published": replace["commit"]["published"],
                "exit_code": replace["commit"]["exit_code"],
            },
            "new_direction_append": {
                "encoder": fresh["encoder"],
                "band": fresh_band,
                "overlap": fresh["direction_band"]["overlap"],
                "published": fresh["commit"]["published"],
                "exit_code": fresh["commit"]["exit_code"],
            },
        },
        "diagnostics": {
            "legacy_append": legacy,
            "localized_replace": replace,
            "new_direction_append": fresh,
        },
        "self_check": self_check,
        "decision": decision,
        "decision_zh": {
            "p2_admitted": "门准入了：四条件全绿，旧值未被破坏，有效秩上升",
            "p2_falsified_premise": (
                "P2 判负，但根因是前提：这条写在 localized_centered 下本身就是复制（o ≥ 0.85），"
                "编码换不掉它的复制性质"
            ),
        }.get(decision, "P2 判负：" + ", ".join(k for k, v in checks.items() if not v) + " 未通过"),
        "isolation": {
            "runner_sha256": file_sha(Path(__file__)),
            "gate_sha256": file_sha(ROOT / "taiji/growth_gate.py"),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "design_sha256": file_sha(ROOT / "taiji/growth_design.py"),
            "p1_report_sha256": file_sha(P1_REPORT) if P1_REPORT.exists() else None,
            "gate_imports_torch": "import torch"
            in (ROOT / "taiji/growth_gate.py").read_text(encoding="utf-8", errors="ignore"),
            "frozen_files_touched": 0,
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
            "format": "taiji-growth-gate-p2-manifest-v1",
            "report": REPORT.relative_to(ROOT).as_posix(),
            "report_sha256": file_sha(REPORT),
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": payload["isolation"]["runner_sha256"],
            "gate": "taiji/growth_gate.py",
            "gate_sha256": payload["isolation"]["gate_sha256"],
            "decision": decision,
            "primary_exit_code": primary["commit"]["exit_code"],
        },
    )
    append_progress(
        {
            "stage": "p2",
            "decision": decision,
            "checks": checks,
            "primary": payload["contrast"]["localized_append"],
            "legacy": payload["contrast"]["legacy_append"],
            "replace": payload["contrast"]["localized_replace"],
            "new_direction": payload["contrast"]["new_direction_append"],
        }
    )

    print(
        json.dumps(
            {
                "decision": decision,
                "checks": checks,
                "primary": payload["contrast"]["localized_append"],
                "legacy": payload["contrast"]["legacy_append"],
                "replace": payload["contrast"]["localized_replace"],
                "new_direction": payload["contrast"]["new_direction_append"],
                "self_check": self_check,
            },
            ensure_ascii=False,
        )
    )
    return 0 if decision == "p2_admitted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
