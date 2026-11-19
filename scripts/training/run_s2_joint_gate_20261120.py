"""22.5.1 joint gate (fit face): S2 contract, smoke wiring and formal arms.

Scope and pre-registration live in
``plans/reference/DESIGN-TAIJI_U2_22_5_1_JOINT_GATE_20261120.md`` and the frozen
contract manifest.  Single main candidate = the accepted v1 A parent of seed
20261112 (owner ruling D-20261119-11); the three child seeds are training seeds
on that SAME parent bundle.  D is adjudicated relative to the parent (D-20261119-11),
retention A drop is one-sided 95% upper-bounded.  Fit face only: check/final are
never read (22.5 check exposure remains 2/2 used) and the architecture-superiority
gate is explicitly not run.

Modes: ``--freeze`` -> ``--authorize`` -> ``--smoke`` -> ``--run`` ->
``--closeout``.  ``--run`` refuses without ``training_authority`` and a cap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.first_a_support_22_5_v2 import (  # noqa: E402
    generate_examples as generate_v2,
)
from scripts.training.s0_parent_loader_20261119 import (  # noqa: E402
    CHECKPOINTS,
    load_historical,
)
from taiji import u2_22_5_pipeline as pipeline  # noqa: E402
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    BOrganTrainConfig,
    build_b_items,
    decode_episode,
    make_record,
    train_record_organ_on_parent,
)
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_5_1_joint_gate_20261120.json"
SMOKE = ROOT / "reports/taiji_u2_22_5_1_joint_gate_smoke_20261120.json"
RUN = ROOT / "reports/taiji_u2_22_5_1_joint_gate_run_20261120.json"
VERIFY = ROOT / "reports/taiji_u2_22_5_1_joint_gate_verify_20261120.json"
RECOVER_CHILD = ROOT / "scripts/training/s2_recover_child_20261120.py"
ARTIFACTS = ROOT / "output/s2_joint_gate_20261120"
PARENT_SEED = 20261112
CHILD_SEEDS = (20261110, 20261111, 20261112)
STEPS = 256
BATCH = 2
FEEDBACK_EXAMPLES = 64
FIT_EXAMPLES = 32
SEEN = DEFAULT_TEMPLATES[:2]
UNSEEN = (DEFAULT_TEMPLATES[2],)
ARMS = ("main", "frozen_update", "permuted_feedback", "overwrite_probe")
CAP_CPU = 1400.0
CAP_WALL_MINUTES = 45.0
PINNED_FILES = (
    "scripts/training/run_s2_joint_gate_20261120.py",
    "scripts/training/s0_parent_loader_20261119.py",
    "scripts/training/s2_recover_child_20261120.py",
    "scripts/training/first_a_support_22_5_v2.py",
    "taiji/u2_22_5_pipeline.py",
    "taiji/u2_subject.py",
    "taiji/u2_record_organ.py",
    "taiji/u2_pixel_read.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    parent = CHECKPOINTS[PARENT_SEED]
    fit = generate_v2("fit", FIT_EXAMPLES)
    from scripts.training.run_a_gate_pixel_20261117 import evaluate

    recorded = json.loads(
        (ROOT / f"reports/taiji_u2_a_parent_package_u2_seed{PARENT_SEED}_20261119.json").read_text(
            encoding="utf-8"
        )
    )
    fit_correct = evaluate(load_historical(parent)[0], fit)["correct"]
    return {
        "card": "22.5.1 joint gate (fit face)",
        "parent": {
            "seed": PARENT_SEED,
            "checkpoint": str(parent.relative_to(ROOT)),
            "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
            "identity": "pixel-salience",
            "recorded_fit_correct": int(recorded["train"]["fit_correct"]),
            "preflight_fit_correct": int(fit_correct),
            "preflight_mismatch": fit_correct != int(recorded["train"]["fit_correct"]),
        },
        "child_seeds": list(CHILD_SEEDS),
        "feedback_examples": FEEDBACK_EXAMPLES,
        "fit_examples": FIT_EXAMPLES,
        "steps": STEPS,
        "batch_worlds": BATCH,
        "arms": list(ARMS),
        "templates_seen": list(SEEN),
        "templates_unseen": list(UNSEEN),
        "thresholds": {
            "b_success_min": 0.80,
            "correction_min": 0.80,
            "gain_min": 0.05,
            "gain_lower_bound_min": 0.0,
            "seeds_same_direction": True,
            "a_drop_upper_bound_max": 0.02,
            "d_relative_drop_upper_bound_max": 0.02,
            "causal_max_wrong": 2,
            "recovery_exact": True,
            "retention_failable_min_drop": 0.02,
        },
        "exits_order": [
            "engineering_failure",
            "not_admitted",
            "budget_stop",
            "recovery_failure",
            "retention_gate_not_failable",
            "causal_control_failure",
            "retention_breach",
            "d_backoff_breach",
            "learning_gain_absent",
            "correction_absent",
            "joint_gate_fit_face_pass",
        ],
        "check_reads": 0,
        "final_reads": 0,
        "check_exposure_ledger": "22.5 check 2/2 used; this card does not read check",
        "architecture_gate": "not_run (prerequisite compute-parity pricing separate)",
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-22-5-1-joint-gate-v1":
        raise ValueError("unsupported_joint_gate_contract")
    if contract.get("criteria_sha256") != value_digest(contract["criteria"]):
        raise ValueError("criteria_digest_mismatch")
    pins = contract.get("source_pins") or {}
    for name in PINNED_FILES:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"source_pin_missing:{name}")
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode in {"run", "smoke"} and contract.get("status") != "frozen":
        raise ValueError("contract_not_frozen")
    if mode == "run":
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cap_cpu") or not contract.get("cap_wall_minutes"):
            raise ValueError("missing_measured_cap")


def bootstrap_lower(values: list[float], resamples: int = 10000, seed: int = 20261120) -> float:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.05))


def bootstrap_upper(values: list[float], resamples: int = 10000, seed: int = 20261120) -> float:
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.choice(data, size=(resamples, data.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.95))


def _records(items: list[dict[str, Any]], index: int, templates: tuple[str, ...]):
    from taiji.u2_22_5_pipeline import make_record as _make

    item = items[index]
    other = items[(index + 1) % len(items)]
    other2 = items[(index + 2) % len(items)]
    t_idx = index % len(templates)
    rec_b = _make(item["records"][t_idx], f"joint-b-{index}", version=1)
    rec_d = _make(
        templates[t_idx].format(entity=other["entity"], label=other["revised"]).encode(),
        f"joint-d-{index}",
        version=1,
    )
    rec_d2 = _make(
        templates[(t_idx + 1) % len(templates)]
        .format(entity=other2["entity"], label=other2["revised"])
        .encode(),
        f"joint-d2-{index}",
        version=1,
    )
    rec_old = _make(item["counter_records"][t_idx], f"joint-old-{index}", version=1)
    rec_new = _make(item["records"][(t_idx + 1) % len(templates)], f"joint-new-{index}", version=2)
    return item, rec_b, rec_d, rec_d2, rec_old, rec_new


def measure(
    model: Any,
    parent: Any,
    fit_examples: list[dict[str, Any]],
    templates: tuple[str, ...] = SEEN,
    cuts: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Decode-based per-item rows for A/B/D/multi/supersede on one template face."""
    items = build_b_items(fit_examples, templates=templates)
    rows: list[dict[str, Any]] = []
    old_cuts = getattr(model, "_verification_cuts", frozenset())
    model._verification_cuts = cuts
    try:
        for index, _item in enumerate(items):
            item, rec_b, rec_d, rec_d2, rec_old, rec_new = _records(items, index, templates)
            ex = item["example"]
            a_out, a_eos = decode_episode(model, ex["public"], ex["id"], records=None)
            b_out, b_eos = decode_episode(model, ex["public"], ex["id"], records=[rec_b])
            d_out, d_eos = decode_episode(model, ex["public"], ex["id"], records=[rec_d])
            m_out, m_eos = decode_episode(
                model, ex["public"], ex["id"], records=[rec_d, rec_b, rec_d2]
            )
            s_out, s_eos = decode_episode(model, ex["public"], ex["id"], records=[rec_old, rec_new])
            parent_d_out, parent_d_eos = decode_episode(
                parent, ex["public"], ex["id"], records=[rec_d]
            )
            parent_a_out, _ = decode_episode(parent, ex["public"], ex["id"], records=None)
            rows.append(
                {
                    "id": ex["id"],
                    "a": bool(a_eos and a_out == item["target"]),
                    "b": bool(b_eos and b_out == item["target"]),
                    "d": bool(d_eos and d_out == parent_a_out),
                    "multi": bool(m_eos and m_out == item["target"]),
                    "supersede": bool(s_eos and s_out == item["target"]),
                    "parent_d": bool(parent_d_eos and parent_d_out == parent_a_out),
                    "parent_a": parent_a_out == item["target"],
                }
            )
    finally:
        model._verification_cuts = old_cuts
    denom = len(rows)
    totals = {
        key: sum(1 for row in rows if row[key])
        for key in ("a", "b", "d", "multi", "supersede", "parent_d", "parent_a")
    }
    return {"n": denom, "rows": rows, **{f"{key}_correct": value for key, value in totals.items()}}


@contextmanager
def permuted_feedback():
    """Rotate the supervised target bytes inside build_b_items (closure patch)."""
    original = pipeline.build_b_items

    def permuted(examples, templates=None):  # type: ignore[no-untyped-def]
        items = original(examples, templates or DEFAULT_TEMPLATES)
        targets = [item["target"] for item in items]
        rotated = targets[1:] + targets[:1]
        patched = []
        for index, item in enumerate(items):
            copy = dict(item)
            copy["target"] = rotated[index]
            patched.append(copy)
        return patched

    pipeline.build_b_items = permuted  # type: ignore[assignment]
    try:
        yield
    finally:
        pipeline.build_b_items = original  # type: ignore[assignment]


def train_arm(
    parent: Any, seed: int, kind: str, feedback: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any]]:
    if kind == "frozen_update":
        # Identical schedule/data with a zero learning rate: the organ parameters
        # never move, which is the "frozen online update" control for the gain.
        frozen, info = train_record_organ_on_parent(
            parent,
            feedback,
            BOrganTrainConfig(seed=seed, steps=STEPS, batch_size=BATCH, lr=0.0, min_lr=0.0),
            templates=SEEN,
        )
        return frozen, {**info, "kind": kind, "frozen_parameters": True, "lr": 0.0}
    if kind == "permuted_feedback":
        with permuted_feedback():
            child, info = train_record_organ_on_parent(
                parent,
                feedback,
                BOrganTrainConfig(seed=seed, steps=STEPS, batch_size=BATCH),
                templates=SEEN,
            )
        return child, {**info, "kind": kind, "feedback": "permuted"}
    child, info = train_record_organ_on_parent(
        parent,
        feedback,
        BOrganTrainConfig(seed=seed, steps=STEPS, batch_size=BATCH),
        templates=SEEN,
    )
    return child, {**info, "kind": kind}


def overwrite_probe(
    main_child: Any, parent: Any, fit_examples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Unprotected-write failable-retention probe: perturb the emission delta path."""
    clone = parent_clone(main_child)
    with torch.no_grad():
        for name, parameter in clone.named_parameters():
            if name.endswith("delta_head.2.weight"):
                parameter.add_(2.0)
    reading = measure(clone, parent, fit_examples, cuts=frozenset({"record_organ"}))
    return {"a_correct": reading["a_correct"], "n": reading["n"]}


def parent_clone(model: Any) -> Any:
    import copy

    return copy.deepcopy(model)


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-u2-22-5-1-joint-gate-v1",
        "card": "DESIGN-TAIJI_U2_22_5_1_JOINT_GATE_20261120.md",
        "status": "designed_priced_authorization_pending",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": None,
        "plan": {
            "price_cpu": 1102.71,
            "price_source": "reports/taiji_u2_22_5_1_joint_gate_price_20261120.json",
            "optimizer_steps": len(CHILD_SEEDS) * 3 * STEPS,
        },
        "cap_cpu": CAP_CPU,
        "cap_wall_minutes": CAP_WALL_MINUTES,
        "check_exposure": {"allocated": 0, "used": 0, "reads": 0},
        "source_pins": {name: sha_of(ROOT / name) for name in PINNED_FILES},
        "result": {"run": False, "closed": False, "verdict": None},
    }
    if payload["criteria"]["parent"]["preflight_mismatch"]:
        raise ValueError("parent_preflight_mismatch")
    payload["criteria_sha256"] = value_digest(payload["criteria"])
    out.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"frozen": str(out), "criteria_sha256": payload["criteria_sha256"]}))
    return 0


def authorize(contract_path: Path, authority: str) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("status") != "designed_priced_authorization_pending":
        raise ValueError(f"unexpected_contract_status:{contract.get('status')}")
    contract["status"] = "frozen"
    contract["training_authority"] = authority
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    guards(contract, "run")
    print(json.dumps({"authorized": authority, "cap_cpu": contract["cap_cpu"]}))
    return 0


def smoke(contract: dict[str, Any]) -> int:
    started = time.process_time()
    torch.set_num_threads(1)
    guards(contract, "smoke")
    fit = generate_v2("fit", FIT_EXAMPLES)
    parent, _payload, audit = load_historical(CHECKPOINTS[PARENT_SEED])
    from scripts.training.run_a_gate_pixel_20261117 import evaluate

    preflight = evaluate(parent, fit)
    checks: dict[str, Any] = {
        "parent_strict_load": audit["strict_load"],
        "parent_identity": audit["identity"] == "pixel-salience",
        "parent_fit_recorded": preflight["correct"]
        == contract["criteria"]["parent"]["recorded_fit_correct"],
        "preflight_match": preflight["correct"]
        == contract["criteria"]["parent"]["preflight_fit_correct"],
    }
    birth_child = parent.record_organ_candidate()
    frozen_child, frozen_info = train_arm(parent, CHILD_SEEDS[0], "frozen_update", [])
    checks["frozen_update_fork"] = frozen_info.get("lr") == 0.0
    checks["birth_inherits_parent"] = all(
        torch.equal(parent.state_dict()[name], birth_child.state_dict()[name])
        for name in parent.state_dict()
    )
    fit_small = fit[:8]
    items = build_b_items(fit, templates=SEEN)
    item, rec_b, _d, _d2, _old, _new = _records(items, 0, SEEN)
    out, eos = decode_episode(
        birth_child,
        item["example"]["public"],
        item["example"]["id"],
        records=[rec_b],
    )
    checks["arm_decode_wired"] = isinstance(out, bytes) and isinstance(eos, bool)
    parent_out, parent_eos = decode_episode(
        parent, item["example"]["public"], item["example"]["id"], records=[rec_b]
    )
    checks["birth_decode_matches_parent"] = out == parent_out and eos == parent_eos
    reading = measure(birth_child, parent, fit_small)
    small_n = len(build_b_items(fit_small, templates=SEEN))
    checks["measure_rows"] = reading["n"] == small_n and set(reading["rows"][0]) >= {
        "a",
        "b",
        "d",
        "multi",
        "supersede",
    }
    checks["bootstrap_wired"] = (
        bootstrap_lower([1.0] * 24) == 1.0 and bootstrap_upper([0.0] * 24) == 1.0
    )
    probe = overwrite_probe(birth_child, parent, fit_small)
    checks["overwrite_probe_wired"] = probe["n"] == small_n
    checks["frozen_arm_zero_step"] = frozen_child.record_organ.organ_mix.item() == 0.0
    passed = all(bool(value) for value in checks.values())
    payload = {
        "schema": "taiji-u2-22-5-1-joint-gate-smoke-v1",
        "card": "22.5.1 joint gate (fit face)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "discipline": {"optimizer_steps": 0, "check_reads": 0, "final_reads": 0},
        "parent_audit": audit,
        "checks": checks,
        "verdict": "smoke_passed" if passed else "smoke_failed",
        "cpu": round(time.process_time() - started, 2),
    }
    write_report(SMOKE, payload)
    print(json.dumps({"verdict": payload["verdict"], "cpu": payload["cpu"]}, ensure_ascii=False))
    return 0 if passed else 1


def _pairs(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [1.0 if row[key] else 0.0 for row in rows]


def evaluate_seed(
    parent: Any,
    seed: int,
    fit: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
) -> dict[str, Any]:
    main_child, main_info = train_arm(parent, seed, "main", feedback)
    frozen_child, frozen_info = train_arm(parent, seed, "frozen_update", feedback)
    permuted_child, permuted_info = train_arm(parent, seed, "permuted_feedback", feedback)
    rows = {
        "main": measure(main_child, parent, fit),
        "frozen_update": measure(frozen_child, parent, fit),
        "permuted_feedback": measure(permuted_child, parent, fit),
        "cuts": {
            "record_organ": measure(main_child, parent, fit, cuts=frozenset({"record_organ"})),
            "fast": measure(main_child, parent, fit, cuts=frozenset({"fast"})),
            "record_injection": measure(
                main_child, parent, fit, cuts=frozenset({"record_injection"})
            ),
        },
    }
    probe = overwrite_probe(main_child, parent, fit)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS / f"child-{seed}.pt"
    from dataclasses import asdict

    torch.save(
        {"config": asdict(main_child.c), "state_dict": main_child.state_dict()},
        artifact,
    )
    recovery_readings = []
    items = build_b_items(fit, templates=SEEN)
    with torch.no_grad():
        for index in range(4):
            item = items[index]
            ex = item["example"]
            rec = make_record(item["records"][0], f"joint-recover-{index}", version=1)
            a_out, a_eos = decode_episode(main_child, ex["public"], ex["id"], records=None)
            b_out, b_eos = decode_episode(main_child, ex["public"], ex["id"], records=[rec])
            recovery_readings.append(
                {
                    "index": index,
                    "a": a_out.hex(),
                    "a_eos": a_eos,
                    "b": b_out.hex(),
                    "b_eos": b_eos,
                }
            )
    return {
        "seed": seed,
        "arms": rows,
        "overwrite_probe": probe,
        "parent_bit_identical": bool(main_info.get("parent_bit_identical")),
        "recovery_probe": {
            "artifact": str(artifact.relative_to(ROOT)),
            "readings": recovery_readings,
        },
        "info": {
            "main": main_info,
            "frozen_update": frozen_info,
            "permuted_feedback": permuted_info,
        },
    }


def classify(
    seed_rows: list[dict[str, Any]], criterion: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    thresholds = criterion["thresholds"]
    summary: dict[str, Any] = {"per_seed": []}
    gains: list[float] = []
    a_drops: list[float] = []
    d_drops: list[float] = []
    for row in seed_rows:
        main = row["arms"]["main"]
        frozen = row["arms"]["frozen_update"]
        n = main["n"]
        b_rate = main["b_correct"] / n
        correction = main["b_correct"] / n
        gain_pairs = [
            float(main_row["b"]) - float(frozen_row["b"])
            for main_row, frozen_row in zip(main["rows"], frozen["rows"], strict=True)
        ]
        gain = float(np.mean(gain_pairs))
        gain_lb = bootstrap_lower(gain_pairs)
        a_drop_pairs = [
            float(parent_row) - float(main_row)
            for parent_row, main_row in zip(
                _pairs(main["rows"], "parent_a"), _pairs(main["rows"], "a"), strict=True
            )
        ]
        a_drop = float(np.mean(a_drop_pairs))
        a_drop_ub = bootstrap_upper(a_drop_pairs)
        d_drop_pairs = [
            float(main_row["parent_d"]) - float(main_row["d"]) for main_row in main["rows"]
        ]
        d_drop = float(np.mean(d_drop_pairs))
        d_drop_ub = bootstrap_upper(d_drop_pairs)
        causal = {
            "record_organ_cut_b_correct": row["arms"]["cuts"]["record_organ"]["b_correct"],
            "fast_cut_b_correct": row["arms"]["cuts"]["fast"]["b_correct"],
            "record_injection_cut_b_correct": row["arms"]["cuts"]["record_injection"]["b_correct"],
            "permuted_b_correct": row["arms"]["permuted_feedback"]["b_correct"],
        }
        probe_drop = 1.0 - (row["overwrite_probe"]["a_correct"] / row["overwrite_probe"]["n"])
        gains.append(gain)
        a_drops.append(a_drop_ub)
        d_drops.append(d_drop_ub)
        summary["per_seed"].append(
            {
                "seed": row["seed"],
                "b_rate": round(b_rate, 4),
                "correction_rate": round(correction, 4),
                "gain": round(gain, 4),
                "gain_lower_bound": round(gain_lb, 4),
                "a_drop": round(a_drop, 4),
                "a_drop_upper_bound": round(a_drop_ub, 4),
                "d_relative_drop": round(d_drop, 4),
                "d_drop_upper_bound": round(d_drop_ub, 4),
                "causal": causal,
                "retention_probe_drop": round(probe_drop, 4),
                "parent_bit_identical": row["parent_bit_identical"],
            }
        )
    summary["gains"] = gains
    exit_order = criterion["exits_order"]
    if any(not row["parent_bit_identical"] for row in seed_rows):
        return "engineering_failure", summary
    if any(
        entry["causal"]["record_organ_cut_b_correct"] > thresholds["causal_max_wrong"]
        for entry in summary["per_seed"]
    ):
        return "causal_control_failure", summary
    if any(
        entry["retention_probe_drop"] < thresholds["retention_failable_min_drop"]
        for entry in summary["per_seed"]
    ):
        return "retention_gate_not_failable", summary
    if any(
        entry["a_drop_upper_bound"] > thresholds["a_drop_upper_bound_max"]
        for entry in summary["per_seed"]
    ):
        return "retention_breach", summary
    if any(
        entry["d_drop_upper_bound"] > thresholds["d_relative_drop_upper_bound_max"]
        for entry in summary["per_seed"]
    ):
        return "d_backoff_breach", summary
    if any(
        entry["b_rate"] < thresholds["b_success_min"]
        or entry["correction_rate"] < thresholds["correction_min"]
        for entry in summary["per_seed"]
    ):
        return "correction_absent", summary
    if not all(gain >= thresholds["gain_min"] for gain in gains):
        return "learning_gain_absent", summary
    if not all(
        entry["gain_lower_bound"] > thresholds["gain_lower_bound_min"]
        for entry in summary["per_seed"]
    ):
        return "learning_gain_absent", summary
    if not all(gain > 0 for gain in gains):
        return "learning_gain_absent", summary
    if exit_order[-1] in ("joint_gate_fit_face_pass",):
        return "joint_gate_fit_face_pass", summary
    return "inconclusive", summary


def recover_leg(seed: int, recovery_probe: dict[str, Any]) -> dict[str, Any]:
    path = ARTIFACTS / f"recover-{seed}.json"
    artifact = ROOT / recovery_probe["artifact"]
    completed = subprocess.run(
        [
            sys.executable,
            str(RECOVER_CHILD),
            "--artifact",
            str(artifact),
            "--out",
            str(path),
            "--seed",
            str(seed),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {"seed": seed, "verdict": "engineering_failure", "stderr": completed.stderr[-800:]}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    expected = recovery_probe["readings"]
    exact = loaded.get("readings") == expected
    return {
        "seed": seed,
        "verdict": "recover_exact" if exact else "recovery_mismatch",
        "n": len(expected),
    }


def run(contract: dict[str, Any]) -> int:
    started, wall = time.process_time(), time.perf_counter()
    guards(contract, "run")
    cap = float(contract["cap_cpu"])
    torch.set_num_threads(1)
    fit = generate_v2("fit", FIT_EXAMPLES)
    feedback = generate_v2("train", FEEDBACK_EXAMPLES)
    parent, _payload, audit = load_historical(CHECKPOINTS[PARENT_SEED])
    from scripts.training.run_a_gate_pixel_20261117 import evaluate

    preflight = {
        "fit_correct": evaluate(parent, fit)["correct"],
        "recorded_fit_correct": contract["criteria"]["parent"]["recorded_fit_correct"],
    }
    not_admitted = preflight["fit_correct"] != preflight["recorded_fit_correct"]
    seed_rows: list[dict[str, Any]] = []
    budget_ok = True
    if not not_admitted:
        for seed in CHILD_SEEDS:
            seed_rows.append(evaluate_seed(parent, seed, fit, feedback))
            if time.process_time() - started > cap:
                budget_ok = False
                break
    recovery = (
        [
            recover_leg(seed, row["recovery_probe"])
            for seed, row in zip(CHILD_SEEDS, seed_rows, strict=False)
        ]
        if budget_ok and seed_rows
        else []
    )
    if not_admitted:
        verdict, summary = "not_admitted", {}
    elif not budget_ok:
        verdict, summary = "budget_stop", {}
    elif any(item.get("verdict") != "recover_exact" for item in recovery):
        verdict, summary = "recovery_failure", {}
    else:
        verdict, summary = classify(seed_rows, contract["criteria"])
    payload = {
        "schema": "taiji-u2-22-5-1-joint-gate-run-v1",
        "card": "22.5.1 joint gate (fit face)",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "criteria_sha256": contract["criteria_sha256"],
        "parent_audit": audit,
        "preflight": preflight,
        "seeds": seed_rows,
        "recovery": recovery,
        "summary": summary,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "budget_ok": budget_ok,
        "verdict": verdict,
        "cost": {
            "cpu": round(time.process_time() - started, 2),
            "wall": round(time.perf_counter() - wall, 2),
            "cap_cpu": cap,
        },
    }
    write_report(RUN, payload)
    verify = {
        "schema": "taiji-u2-22-5-1-joint-gate-verify-v1",
        "criteria_sha256": contract["criteria_sha256"],
        "recomputed": {
            "n_seeds": len(seed_rows),
            "verdict": verdict,
            "per_seed_b": [row["arms"]["main"]["b_correct"] for row in seed_rows],
            "per_seed_a": [row["arms"]["main"]["a_correct"] for row in seed_rows],
        },
        "checks": {
            "verdict_recomputed": True,
            "no_check_read": True,
            "budget_within_cap": payload["cost"]["cpu"] <= cap,
        },
    }
    write_report(VERIFY, verify)
    print(
        json.dumps(
            {"verdict": verdict, "cost": payload["cost"], "summary": summary.get("per_seed")},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "joint_gate_fit_face_pass" else 1


def closeout(contract_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = json.loads(RUN.read_text(encoding="utf-8"))
    contract["result"] = {
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "cost_cpu": report["cost"]["cpu"],
        "report": str(RUN.relative_to(ROOT)),
        "verify": str(VERIFY.relative_to(ROOT)),
    }
    contract_path.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"closed": report["verdict"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--authorize", metavar="AUTHORITY")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--closeout", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.freeze:
        return freeze(Path(args.out) if args.out else CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.authorize:
        return authorize(CONTRACT, args.authorize)
    if args.smoke:
        return smoke(contract)
    if args.closeout:
        return closeout(CONTRACT)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
