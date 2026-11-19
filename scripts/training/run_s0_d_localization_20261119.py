"""22.5.1/S0 D-side localization package: organ routing operating point only.

Scope (see ``plans/reference/DESIGN-TAIJI_U2_S0_D_LOCALIZATION_20261119.md``):
two strong accepted v1 A parents, organ-only 300-step training on the v2 train
fixture, then an inference-time routing-bias sweep.  The ONLY variable is the
logit offset ``delta`` applied to the routing gate; it is a closure patch (zero
module-byte change), never trained, never written back.

Modes (order of use): ``--freeze`` (pins + criteria, after committing code) ->
owner authorization -> ``--authorize`` -> ``--smoke`` -> ``--run`` ->
``--closeout``.  ``--run`` refuses unless the contract is frozen, carries
``training_authority`` and a measured cap.  Zero check/final reads in every mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import MethodType
from typing import Any

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
from taiji.u2_22_5_pipeline import (  # noqa: E402
    DEFAULT_TEMPLATES,
    BOrganTrainConfig,
    build_b_items,
    decode_episode,
    make_record,
    prepare_episode,
    train_record_organ_on_parent,
)
from taiji.u2_history import value_digest  # noqa: E402

CONTRACT = ROOT / "plans/manifests/taiji_u2_s0_d_localization_20261119.json"
SMOKE = ROOT / "reports/taiji_u2_s0_d_localization_smoke_20261119.json"
RUN = ROOT / "reports/taiji_u2_s0_d_localization_run_20261119.json"
SEEDS = (20261110, 20261112)
SEEN_TEMPLATES = DEFAULT_TEMPLATES[:2]
UNSEEN_TEMPLATES = (DEFAULT_TEMPLATES[2],)
DELTA_GRID = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
MAX_STEPS = 300
CAP_CPU = 154.7
CAP_WALL_MINUTES = 20.0
PINNED_FILES = (
    "scripts/training/run_s0_d_localization_20261119.py",
    "scripts/training/s0_parent_loader_20261119.py",
    "scripts/training/first_a_support_22_5_v2.py",
    "scripts/training/first_a_support_22_5.py",
    "taiji/u2_22_5_pipeline.py",
    "taiji/u2_subject.py",
    "taiji/u2_record_organ.py",
    "taiji/u2_pixel_read.py",
)


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def criteria() -> dict[str, Any]:
    return {
        "card": "22.5.1/S0 D-side localization",
        "seeds": list(SEEDS),
        "parents": {
            str(seed): {
                "checkpoint": str(CHECKPOINTS[seed].relative_to(ROOT)),
                "identity": "pixel-salience",
                "recorded_fit_correct": 30 if seed == 20261110 else 31,
            }
            for seed in SEEDS
        },
        "excluded_seeds": {
            "20261111": "accepted parent fit 14/32; D gate >=20/24 is structurally unreachable"
        },
        "train_examples": 768,
        "fit_examples": 32,
        "organ_steps": MAX_STEPS,
        "organ_batch": 4,
        "seen_templates": list(SEEN_TEMPLATES),
        "unseen_templates": list(UNSEEN_TEMPLATES),
        "delta_grid": list(DELTA_GRID),
        "decision_gate": {
            "b_hit_min": 23,
            "d_hit_min": 21,
            "denominator": 24,
            "faces": ["seen", "unseen"],
            "seeds": list(SEEDS),
        },
        "decode_gate": {"b_correct": 24, "d_min": 20, "faces": ["seen", "unseen"]},
        "exits": [
            "engineering_failure",
            "budget_stop",
            "operating_point_insufficient",
            "operating_point_decision_only",
            "operating_point_only",
        ],
        "optimizer_steps_max": len(SEEDS) * MAX_STEPS,
        "check_reads": 0,
        "final_reads": 0,
        "training_authority": None,
    }


def guards(contract: dict[str, Any], mode: str) -> None:
    if contract.get("schema") != "taiji-u2-s0-d-localization-v1":
        raise ValueError("unsupported_localization_contract")
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


def route_prob_at(match_prob: torch.Tensor, delta: float, active: torch.Tensor) -> torch.Tensor:
    """Inference-time routing gate offset; ``delta=0`` reproduces the native gate."""
    return torch.sigmoid(12.0 * (match_prob - 0.5) + float(delta)) * active


def patch_plan_delta(organ: Any, delta: float) -> None:
    """Closure-patch ``plan`` to apply ``delta``; zero module-byte change."""
    base_plan = type(organ).plan

    def plan(self: Any, request: bytes, records: Any) -> dict[str, Any]:
        result = base_plan(self, request, records)
        if result.get("entries"):
            result["route_prob"] = route_prob_at(
                result["match_prob"], delta, self.active_strength()
            )
        return result

    organ.plan = MethodType(plan, organ)


def selected_window(plan: dict[str, Any]) -> bytes | None:
    entries = plan.get("entries") or ()
    if not entries:
        return None
    weights = plan["record_weights"].detach()
    if weights.numel() == 0:
        return None
    entry = entries[int(weights.argmax())]
    span_probs = entry["span_probs"]
    if span_probs.numel() == 0:
        return None
    best = int(span_probs.detach().argmax())
    return bytes(entry["raw"][entry["span_starts"][best] : entry["span_ends"][best]])


def distractor_records(
    items: list[dict[str, Any]], index: int, templates: tuple[str, ...]
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    item = items[index]
    other = items[(index + 1) % len(items)]
    other2 = items[(index + 2) % len(items)]
    t_idx = index % len(templates)
    rec_b = make_record(item["records"][t_idx], f"loc-b-{index}", version=1)
    rec_d = make_record(
        templates[t_idx].format(entity=other["entity"], label=other["revised"]).encode(),
        f"loc-d-{index}",
        version=1,
    )
    rec_d2 = make_record(
        templates[(t_idx + 1) % len(templates)]
        .format(entity=other2["entity"], label=other2["revised"])
        .encode(),
        f"loc-d2-{index}",
        version=1,
    )
    return item, rec_b, rec_d, rec_d2


def _request_bytes(model: Any, example: dict[str, Any]) -> bytes:
    state = prepare_episode(model, example["public"], example["id"], records=None)
    return model._organ_request_bytes(state)


def decision_readings(
    model: Any, fit_examples: list[dict[str, Any]], delta: float
) -> dict[str, dict[str, int]]:
    """Per-face B/D/multi decision hits for one delta (no decoding)."""
    organ = model.record_organ
    patch_plan_delta(organ, delta)
    readings = {
        "seen": {"b": 0, "d": 0, "multi": 0, "n": 0},
        "unseen": {"b": 0, "d": 0, "multi": 0, "n": 0},
    }
    for face, templates in (("seen", SEEN_TEMPLATES), ("unseen", UNSEEN_TEMPLATES)):
        items = build_b_items(fit_examples, templates=templates)
        for index, _raw_item in enumerate(items):
            item, rec_b, rec_d, rec_d2 = distractor_records(items, index, templates)
            ex = item["example"]
            request = _request_bytes(model, ex)
            target = item["target"]
            with torch.no_grad():
                plan_b = organ.plan(request, [rec_b])
                plan_d = organ.plan(request, [rec_d])
                plan_m = organ.plan(request, [rec_d, rec_b, rec_d2])
            b_hit = (
                float(plan_b["route_prob"].detach()) >= 0.5 and selected_window(plan_b) == target
            )
            d_hit = float(plan_d["route_prob"].detach()) < 0.5
            multi_hit = (
                float(plan_m["route_prob"].detach()) >= 0.5 and selected_window(plan_m) == target
            )
            readings[face]["b"] += int(b_hit)
            readings[face]["d"] += int(d_hit)
            readings[face]["multi"] += int(multi_hit)
            readings[face]["n"] += 1
    return readings


def decode_readings(model: Any, fit_examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Full greedy-decode B/D counts for the currently installed delta."""
    readings: dict[str, dict[str, int]] = {}
    for face, templates in (("seen", SEEN_TEMPLATES), ("unseen", UNSEEN_TEMPLATES)):
        items = build_b_items(fit_examples, templates=templates)
        b_ok = d_ok = 0
        for index, _raw_item in enumerate(items):
            item, rec_b, rec_d, _rec_d2 = distractor_records(items, index, templates)
            ex = item["example"]
            out_b, eos_b = decode_episode(model, ex["public"], ex["id"], records=[rec_b])
            b_ok += int(eos_b and out_b == item["target"])
            out_d, eos_d = decode_episode(model, ex["public"], ex["id"], records=[rec_d])
            d_ok += int(eos_d and out_d == ex["target"])
        readings[face] = {"b": b_ok, "d": d_ok, "n": len(items)}
    return readings


def classify(
    decision: dict[str, dict[str, dict[str, int]]],
    delta: int | None,
    decode: dict[str, dict[str, dict[str, int]]] | None,
) -> str:
    if delta is None:
        return "operating_point_insufficient"
    if decode is None:
        return "operating_point_decision_only"
    gate = criteria()["decode_gate"]
    confirmed = all(
        row["b"] == gate["b_correct"] and row["d"] >= gate["d_min"]
        for per_face in decode.values()
        for row in per_face.values()
    )
    return "operating_point_only" if confirmed else "operating_point_decision_only"


def freeze(out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "taiji-u2-s0-d-localization-v1",
        "card": "DESIGN-TAIJI_U2_S0_D_LOCALIZATION_20261119.md",
        "status": "designed_priced_authorization_pending",
        "training_authority": None,
        "criteria": criteria(),
        "criteria_sha256": value_digest(criteria()),
        "plan": {
            "price_cpu": 123.76,
            "price_source": "reports/taiji_u2_s0_decision_options_price_20261119.json",
            "optimizer_steps": len(SEEDS) * MAX_STEPS,
        },
        "cap_cpu": CAP_CPU,
        "cap_wall_minutes": CAP_WALL_MINUTES,
        "check_exposure": {"allocated": 0, "used": 0, "reads": 0},
        "source_pins": {name: sha_of(ROOT / name) for name in PINNED_FILES},
        "result": {"run": False, "closed": False, "verdict": None},
    }
    out.write_bytes(json.dumps(contract, ensure_ascii=False, indent=2).encode() + b"\n")
    print(json.dumps({"frozen": str(out), "criteria_sha256": contract["criteria_sha256"]}))
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
    fit_examples = generate_v2("fit", 32)
    checks: dict[str, Any] = {}
    parent, payload, audit = load_historical(CHECKPOINTS[SEEDS[0]])
    checks["parent_identity"] = audit["identity"] == "pixel-salience"
    checks["strict_load"] = audit["strict_load"]
    checks["pins_disclosed"] = audit["source_pin_mismatches"] == ["taiji/u2_subject.py"]
    parent_state = parent.state_dict()
    child = parent.record_organ_candidate()
    child_state = child.state_dict()
    inherited = [key for key in parent_state if not key.startswith("record_organ.")]
    checks["birth_bit_identical"] = all(
        torch.equal(parent_state[key], child_state[key]) for key in inherited
    )
    checks["organ_keys_added"] = any(key.startswith("record_organ.") for key in child_state)
    example = fit_examples[0]
    parent_reading = decode_episode(parent, example["public"], example["id"], records=None)
    child_reading = decode_episode(child, example["public"], example["id"], records=None)
    checks["birth_decode_identical"] = parent_reading == child_reading
    organ = child.record_organ
    active = torch.tensor(0.8)
    route_zero = float(route_prob_at(torch.tensor(0.9), 0.0, active))
    route_up = float(route_prob_at(torch.tensor(0.9), 4.0, active))
    route_down = float(route_prob_at(torch.tensor(0.9), -4.0, active))
    expected_zero = float(torch.sigmoid(torch.tensor(12.0 * (0.9 - 0.5))) * active)
    checks["delta_formula"] = math.isclose(route_zero, expected_zero, rel_tol=1e-6)
    checks["delta_monotone"] = route_down < route_zero < route_up
    items = build_b_items(fit_examples, templates=SEEN_TEMPLATES)
    item, rec_b, _rec_d, _rec_d2 = distractor_records(items, 0, SEEN_TEMPLATES)
    request = _request_bytes(child, item["example"])
    organ.zero_grad(set_to_none=True)
    with torch.no_grad():
        plan = organ.plan(request, [rec_b])
    checks["plan_route_prob_scalar"] = tuple(plan["route_prob"].shape) == ()
    checks["plan_record_weights"] = tuple(plan["record_weights"].shape) == (len(plan["entries"]),)
    emit_route, emit_probs = organ.emit(plan, 0)
    checks["emit_shapes"] = tuple(emit_probs.shape) == (257,) and tuple(emit_route.shape) == ()
    isolated = child.record_organ_candidate()
    patch_plan_delta(isolated.record_organ, 2.0)
    with torch.no_grad():
        patched = isolated.record_organ.plan(request, [rec_b])
    checks["delta_patch_installed"] = "route_prob" in patched
    passed = all(bool(value) for value in checks.values())
    payload = {
        "schema": "taiji-u2-s0-d-localization-smoke-v1",
        "card": "22.5.1/S0 D-side localization",
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
    print(
        json.dumps(
            {"verdict": payload["verdict"], "checks": checks, "cpu": payload["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


def run(contract: dict[str, Any]) -> int:
    started, wall = time.process_time(), time.perf_counter()
    guards(contract, "run")
    cap_cpu = float(contract["cap_cpu"])
    torch.set_num_threads(1)
    fit_examples = generate_v2("fit", 32)
    train_examples = generate_v2("train", 768)
    rows: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    budget_ok = True
    for seed in SEEDS:
        seed_started = time.process_time()
        parent, _payload, audit = load_historical(CHECKPOINTS[seed])
        child, organ_info = train_record_organ_on_parent(
            parent,
            train_examples,
            BOrganTrainConfig(seed=seed, steps=MAX_STEPS, batch_size=4),
            templates=SEEN_TEMPLATES,
        )
        models[seed] = child
        decision = {
            str(delta): decision_readings(child, fit_examples, float(delta)) for delta in DELTA_GRID
        }
        rows.append(
            {
                "seed": seed,
                "parent_audit": audit,
                "organ": organ_info,
                "decision": decision,
                "cost_cpu": round(time.process_time() - seed_started, 2),
            }
        )
        if time.process_time() - started > cap_cpu:
            budget_ok = False
            break
    gate = criteria()["decision_gate"]
    delta_star: int | None = None
    if budget_ok:
        for delta in DELTA_GRID:
            if all(
                row["decision"][str(delta)][face]["b"] >= gate["b_hit_min"]
                and row["decision"][str(delta)][face]["d"] >= gate["d_hit_min"]
                for row in rows
                for face in ("seen", "unseen")
            ):
                delta_star = delta
                break
    decode: dict[str, dict[str, dict[str, int]]] = {}
    if delta_star is not None:
        for seed, child in models.items():
            patch_plan_delta(child.record_organ, float(delta_star))
            decode[str(seed)] = decode_readings(child, fit_examples)
            if time.process_time() - started > cap_cpu:
                budget_ok = False
                break
    verdict = (
        "budget_stop"
        if not budget_ok
        else classify(
            {str(row["seed"]): row["decision"] for row in rows}, delta_star, decode or None
        )
    )
    payload = {
        "schema": "taiji-u2-s0-d-localization-run-v1",
        "card": "22.5.1/S0 D-side localization",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "criteria_sha256": contract["criteria_sha256"],
        "training_authority": contract["training_authority"],
        "rows": rows,
        "delta_star": delta_star,
        "decode": decode or None,
        "budget_ok": budget_ok,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "verdict": verdict,
        "cost": {
            "cpu": round(time.process_time() - started, 2),
            "wall": round(time.perf_counter() - wall, 2),
            "cap_cpu": cap_cpu,
        },
    }
    write_report(RUN, payload)
    print(json.dumps({"verdict": verdict, "cost": payload["cost"]}, ensure_ascii=False))
    return 0 if verdict == "operating_point_only" else 1


def closeout(contract_path: Path, report_path: Path) -> int:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("criteria_sha256") != contract.get("criteria_sha256"):
        raise ValueError("criteria_digest_mismatch")
    contract["result"] = {
        "run": True,
        "closed": True,
        "verdict": report["verdict"],
        "delta_star": report.get("delta_star"),
        "cost_cpu": report.get("cost", {}).get("cpu"),
        "report": str(report_path.relative_to(ROOT)),
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
        return closeout(CONTRACT, RUN)
    return run(contract)


if __name__ == "__main__":
    raise SystemExit(main())
