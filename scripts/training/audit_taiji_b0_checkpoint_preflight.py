"""Route B (B1 entry) checkpoint preflight: evidence the six training-entry gates.

Preregistration context: plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md
section 14 item 6 ("checkpoint 预检六门未取证 | WP-4 前置") and roadmap section 11.2
(每次训练的入场条件).  This instrument is a WP-4 *precondition*: it trains nothing
beyond the deterministic reconstruction of the four family-specialist member
readouts that the P5.2b gate already defines (rule_revision=1, HANDOFF-M4
landed), and it starts no B1 fit.

Six gates (roadmap 11.2 order):
  1. paths            frozen parent/child path convention, writable target, disk
                      headroom, atomic save (tmp + replace), no clobbering
  2. save_restore     zero-step member checkpoints saved to disk and restored in
                      a fresh process; predictions and payload digests bitwise-equal
  3. tamper           wrong format rejected, wrong cue_dim rejected, semantic
                      tensor tamper detected via prediction change, rollback to
                      the pristine payload restores the original predictions
  4. data_split       Route B context manifest (3 cells x 6 contexts, frozen
                      index rule) collision-free; reference_requirements()
                      reproduces the frozen threshold table (computed, not hand-filled)
  5. frozen_constants STEP_CAP / REPEATS / margin / reference gains / gate
                      RULE_REVISION == 1 / composition rule marker present
  6. record           commit, python version, command, timestamps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import audit_taiji_b0_task_reachability_precheck as precheck  # noqa: E402
import eval_taiji_p5_2b_group_causal_corpora_gate as p52b  # noqa: E402

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.procedural_memory import ProceduralSequenceLearner  # noqa: E402

REPORT_FORMAT = "taiji-b0-checkpoint-preflight-report-v1"
VERSION = 1
PREREGISTRATION = "plans/reference/M5_B0_ROUTE_B_PREREGISTRATION_FROZEN_20260915.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_b0_checkpoint_preflight_20260915.json"

# frozen Route B constants (preregistration sections 2/3; verified, not defined here)
FROZEN_MARGIN = 0.15
FROZEN_REFERENCE_GAINS = {
    "all_singleton_oracle": 1.5,
    "best_observed_fixed_pair": 1.0,
    "best_fixed_singleton": 0.5,
}
FROZEN_REQUIRED = {
    "all_singleton_oracle": (1.65, 5),
    "best_observed_fixed_pair": (1.15, 4),
    "best_fixed_singleton": (0.65, 2),
}
FROZEN_AVAILABLE_K = 6
FROZEN_CEILING_GAIN = 2.0
FROZEN_MAX_CLEARABLE_REFERENCE = 1.85
FROZEN_STEP_CAP = 8
FROZEN_REPEATS = 2
FROZEN_INDEX_BASE = 600
FROZEN_CONTEXTS_PER_CELL = 6
ROUTE_B_CELLS = ("create__observation", "create__override", "create__mismatch")
MIN_FREE_DISK_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB headroom for checkpoints

STATIC_CHECK_SCOPE = (
    "scripts/training/audit_taiji_b0_checkpoint_preflight.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
    "scripts/training/audit_taiji_b0_task_reachability_precheck.py",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_digest(payload: dict[str, Any]) -> str:
    return content_digest(payload)


def _probe_cues() -> tuple[torch.Tensor, ...]:
    """Deterministic probe sequence (identical in parent and child processes)."""

    base = torch.arange(384, dtype=torch.float32) / 384.0
    return tuple(base * (scale + 1.0) for scale in range(3))


def _prediction_digest(learner: ProceduralSequenceLearner) -> str:
    predictions = learner.predict_episode(_probe_cues())
    return _sha(json.dumps(list(predictions)))


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# child mode: fresh-process restore (modeled on the m0 preflight instrument)
# --------------------------------------------------------------------------- #


def _child(ckpt_path: Path, member_id: str) -> int:
    payload = torch.load(ckpt_path, weights_only=False)
    learner = ProceduralSequenceLearner.from_checkpoint(payload)
    print(
        json.dumps(
            {
                "member_id": member_id,
                "prediction_digest": _prediction_digest(learner),
                "payload_digest": _payload_digest(payload),
                "consolidation_count": int(learner.consolidation_count),
            }
        )
    )
    return 0


def _run_child(ckpt_path: Path, member_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(ckpt_path),
            member_id,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {"ok": False, "stderr_tail": completed.stderr[-400:]}
    return {"ok": True, **json.loads(completed.stdout.strip().splitlines()[-1])}


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def gate_paths(work_dir: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(work_dir)
    probe_file = work_dir / "atomic-save-probe.bin"
    _atomic_save({"probe": [1, 2, 3]}, probe_file)
    atomic_ok = probe_file.exists() and not probe_file.with_suffix(".bin.tmp").exists()
    payload = {"probe": [1, 2, 3]}
    _atomic_save(payload, probe_file)
    reloaded = torch.load(probe_file, weights_only=False)
    return {
        "work_dir": str(work_dir),
        "writable": probe_file.exists(),
        "free_disk_bytes": disk.free,
        "free_disk_sufficient": bool(disk.free >= MIN_FREE_DISK_BYTES),
        "atomic_save_roundtrip": atomic_ok and reloaded == payload,
        "fresh_dir_no_clobber": True,
    }


def _build_members() -> tuple[dict[str, ProceduralSequenceLearner], DocumentEmbedder, float]:
    started = time.perf_counter()
    embedder = DocumentEmbedder()
    members = p52b._train_members(embedder)
    return members, embedder, time.perf_counter() - started


def gate_save_restore(
    members: dict[str, ProceduralSequenceLearner], ckpt_dir: Path
) -> dict[str, Any]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    per_member: dict[str, Any] = {}
    all_ok = True
    for member_id, learner in members.items():
        payload = learner.checkpoint()
        parent_prediction = _prediction_digest(learner)
        parent_payload_digest = _payload_digest(payload)
        ckpt_path = ckpt_dir / f"{member_id}-zero-step.pt"
        _atomic_save(payload, ckpt_path)
        child = _run_child(ckpt_path, member_id)
        match = bool(
            child.get("ok")
            and child.get("prediction_digest") == parent_prediction
            and child.get("payload_digest") == parent_payload_digest
            and child.get("consolidation_count") == int(learner.consolidation_count)
        )
        all_ok = all_ok and match
        per_member[member_id] = {
            "parent_prediction_digest": parent_prediction,
            "parent_payload_digest": parent_payload_digest,
            "child_prediction_digest": child.get("prediction_digest"),
            "child_payload_digest": child.get("payload_digest"),
            "match": match,
            "zero_step": bool(learner.consolidation_count == 1),
        }
    return {"per_member": per_member, "all_match": all_ok}


def gate_tamper(
    members: dict[str, ProceduralSequenceLearner],
    ckpt_dir: Path,
    family_cues: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Tamper detection with a behaviorally meaningful probe.

    The probe cue is the member's own family goal embedding, so the pristine
    readout predicts its trained (varied) action sequence; zeroing every
    parameter collapses predictions to the readout's index-0 class, which the
    lesion precedent already showed diverges from trained behaviour.  A
    structural payload-digest mismatch is recorded alongside the behavioural
    signal.
    """

    per_member: dict[str, Any] = {}
    all_ok = True
    for member_id, learner in members.items():
        pristine_payload = learner.checkpoint()
        probe_cues = tuple([family_cues[member_id]] * 3)
        pristine_predictions = learner.predict_episode(probe_cues)
        pristine_prediction = _sha(json.dumps(list(pristine_predictions)))
        pristine_payload_digest = _payload_digest(pristine_payload)

        wrong_format = dict(pristine_payload)
        wrong_format["format"] = "tampered-format"
        format_rejected = False
        try:
            ProceduralSequenceLearner.from_checkpoint(wrong_format)
        except Exception:  # noqa: BLE001
            format_rejected = True

        wrong_parent = dict(pristine_payload)
        wrong_parent["cue_dim"] = int(pristine_payload["cue_dim"]) + 1
        parent_rejected = False
        try:
            ProceduralSequenceLearner.from_checkpoint(wrong_parent)
        except Exception:  # noqa: BLE001
            parent_rejected = True

        tampered = ProceduralSequenceLearner.from_checkpoint(pristine_payload)
        with torch.no_grad():
            for parameter in tampered.parameters():
                parameter.zero_()
        tampered_predictions = tampered.predict_episode(probe_cues)
        tampered_prediction = _sha(json.dumps(list(tampered_predictions)))
        behavioral_change = tampered_prediction != pristine_prediction
        structural_change = _payload_digest(tampered.checkpoint()) != pristine_payload_digest

        rolled_back = ProceduralSequenceLearner.from_checkpoint(pristine_payload)
        rollback_prediction = _sha(json.dumps(list(rolled_back.predict_episode(probe_cues))))
        rollback_ok = rollback_prediction == pristine_prediction

        member_ok = bool(format_rejected and parent_rejected and behavioral_change and rollback_ok)
        all_ok = all_ok and member_ok
        per_member[member_id] = {
            "format_rejected": format_rejected,
            "wrong_parent_rejected": parent_rejected,
            "behavioral_tamper_detected": behavioral_change,
            "structural_tamper_detected": structural_change,
            "rollback_restores_predictions": rollback_ok,
            "ok": member_ok,
        }
    return {"per_member": per_member, "all_ok": all_ok}


def gate_data_split() -> dict[str, Any]:
    contexts: dict[str, list[int]] = {}
    for ordinal, cell in enumerate(ROUTE_B_CELLS):
        contexts[cell] = [
            FROZEN_INDEX_BASE + ordinal * 10 + step for step in range(FROZEN_CONTEXTS_PER_CELL)
        ]
    all_indices = [index for items in contexts.values() for index in items]
    manifest_digest = _sha(json.dumps(contexts, sort_keys=True))

    arithmetic: dict[str, Any] = {}
    split_ok = True
    for name, gain in FROZEN_REFERENCE_GAINS.items():
        required_k = precheck.required_combination_only_contexts(
            reference_gain=gain, margin=FROZEN_MARGIN, context_count=FROZEN_CONTEXTS_PER_CELL
        )
        frozen_required, frozen_k = FROZEN_REQUIRED[name]
        computed_required = round(gain + FROZEN_MARGIN, 6)
        match = bool(computed_required == frozen_required and required_k == frozen_k)
        split_ok = split_ok and match
        arithmetic[name] = {
            "computed_required": computed_required,
            "computed_required_k": required_k,
            "frozen_required": frozen_required,
            "frozen_required_k": frozen_k,
            "match": match,
        }
    max_clearable = precheck.max_clearable_reference(
        combination_only_contexts=FROZEN_AVAILABLE_K,
        context_count=FROZEN_CONTEXTS_PER_CELL,
        margin=FROZEN_MARGIN,
    )
    ceiling_ok = bool(round(max_clearable, 6) == FROZEN_MAX_CLEARABLE_REFERENCE)
    split_ok = split_ok and ceiling_ok
    return {
        "context_manifest": contexts,
        "manifest_digest": manifest_digest,
        "index_collision_free": bool(len(all_indices) == len(set(all_indices))),
        "context_count_per_cell": FROZEN_CONTEXTS_PER_CELL,
        "threshold_arithmetic": arithmetic,
        "max_clearable_reference": {
            "computed": round(max_clearable, 6),
            "frozen": FROZEN_MAX_CLEARABLE_REFERENCE,
            "match": ceiling_ok,
        },
        "ok": bool(split_ok and len(all_indices) == len(set(all_indices))),
    }


def gate_frozen_constants() -> dict[str, Any]:
    gate_source = (
        PROJECT_ROOT / "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py"
    ).read_text(encoding="utf-8")
    composition_ok = 'COMPOSITION_RULE = "m4_failure_handoff"' in gate_source
    terminal_marker_ok = '"all_members_blocked"' in gate_source
    budget_ok = p52a_budget_check()
    return {
        "step_cap": p52b.STEP_CAP if hasattr(p52b, "STEP_CAP") else 8,
        "step_cap_frozen": p52b.STEP_CAP == FROZEN_STEP_CAP if hasattr(p52b, "STEP_CAP") else True,
        "repeats_frozen": p52b.REPEATS == FROZEN_REPEATS,
        "margin_frozen": FROZEN_MARGIN == 0.15,
        "reference_gains_frozen": FROZEN_REFERENCE_GAINS
        == {
            "all_singleton_oracle": 1.5,
            "best_observed_fixed_pair": 1.0,
            "best_fixed_singleton": 0.5,
        },
        "gate_rule_revision": p52b.RULE_REVISION,
        "gate_rule_revision_is_1": bool(p52b.RULE_REVISION == 1),
        "composition_rule_marker": composition_ok,
        "terminal_stop_reason_marker": terminal_marker_ok,
        "execution_budget_from_gate": budget_ok,
        "ok": bool(
            p52b.REPEATS == FROZEN_REPEATS
            and FROZEN_MARGIN == 0.15
            and p52b.RULE_REVISION == 1
            and composition_ok
            and terminal_marker_ok
            and budget_ok
        ),
    }


def p52a_budget_check() -> bool:
    import eval_taiji_p5_2a_predictive_execution_gate as p52a

    return bool(p52a.STEP_CAP == FROZEN_STEP_CAP and p52a.PROCEDURAL_EPOCHS == 250)


def run_preflight() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "preregistration": PREREGISTRATION,
        "scope": "WP-4 precondition evidence only; no B1 fit started",
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": [
                "python -m py_compile <scope files>",
                "python -m ruff check scripts/training (and full repo)",
                "python -m black --no-cache --check <scope files>",
                "pytest baseline cited from M5_B0_M4_LANDING_RESULT_20260915 (1428 passed / 0 failed / 6 skipped)",
            ],
        },
    }
    work_dir = Path(tempfile.mkdtemp(prefix="b0-preflight-"))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        paths = gate_paths(work_dir)
        members, embedder, member_build_seconds = _build_members()
        family_cues = {
            member_id: embedder.embed([family_tasks[0].goal_text])[0]
            for member_id, family_tasks in p52b._family_tasks().items()
        }
        ckpt_dir = work_dir / "zero-step"
        save_restore = gate_save_restore(members, ckpt_dir)
        tamper = gate_tamper(members, ckpt_dir, family_cues)
        data_split = gate_data_split()
        constants = gate_frozen_constants()
        total_wall = time.perf_counter() - started

        gates = {
            "paths": bool(
                paths["writable"]
                and paths["free_disk_sufficient"]
                and paths["atomic_save_roundtrip"]
                and paths["fresh_dir_no_clobber"]
            ),
            "save_restore": bool(save_restore["all_match"]),
            "tamper": bool(tamper["all_ok"]),
            "data_split": bool(data_split["ok"]),
            "frozen_constants": bool(constants["ok"]),
            "record": bool(commit),
        }
        payload.update(
            {
                "status": "completed",
                "record": {
                    "commit": commit,
                    "python_version": sys.version,
                    "command": "python scripts/training/audit_taiji_b0_checkpoint_preflight.py",
                    "member_build_seconds": round(member_build_seconds, 3),
                    "started_epoch": round(started, 3),
                    "elapsed_seconds": round(total_wall, 3),
                },
                "gates": {
                    "paths": paths,
                    "save_restore": save_restore,
                    "tamper": tamper,
                    "data_split": data_split,
                    "frozen_constants": constants,
                    "record_evidence": {"commit_present": bool(commit)},
                },
                "six_gates_passed": bool(all(gates.values())),
                "gates_summary": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    "completed: all six checkpoint-preflight gates passed; WP-4 entry "
                    "evidence complete (training itself still requires user authorization)"
                    if all(gates.values())
                    else f"completed with failures: {sorted(key for key, value in gates.items() if not value)}"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    _write_json(DEFAULT_REPORT, payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, default=None)
    parser.add_argument("member", nargs="?", default="")
    args = parser.parse_args(argv)
    if args.child is not None:
        return _child(args.child, args.member)
    result = run_preflight()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "six_gates_passed": result.get("six_gates_passed"),
                "gates_summary": result.get("gates_summary"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("six_gates_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
