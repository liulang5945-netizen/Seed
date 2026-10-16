"""PLAN-N1-90: bind the nine final-cycle-decay endpoints and run the frozen v5 face once.

This module replaces only the binding and identity half of PLAN-N1-85's confirmation
executor. The measurement half (checkpoint load, evidence tape, teacher logits,
generation, per-item session cleanup, fingerprint neutrality, the four-gate
adjudication and the one-shot attempt discipline) is reused from the frozen 85
evaluator, so nothing in the generation chain is re-typed. Because that reused chain
writes its own session prefix, confirmation sessions stay named ``confirmation85-``
even though the nine endpoints come from three batches; PLAN-N1-90 section 2 records
that naming difference instead of pretending the 84 single-family receipt ran.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    _taiji_n1_transition_factorial_training_receipt as receipt,
)
from scripts.training import (  # noqa: E402
    build_taiji_n1_fresh_paired_admission as face,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation as base85,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_lr_decay_run88 as lr88,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_lr_decay_run89 as lr89,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_paired as train,
)

ORIGINAL_LOAD = train.load
REPORT = ROOT / "reports/taiji_n1_transition_factorial_confirmation_run89.json"
ATTEMPT = ROOT / "reports/taiji_n1_transition_factorial_confirmation_run89_attempt.json"
PLAN = ROOT / "plans/reference/PLAN-N1-90_three_family_binding_v5_confirmation_20261017.md"
V5_SHA = "22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d"
FROZEN_RUNNER_SHA256 = lr88.FROZEN_RUNNER_SHA256
CPU_LIMIT_SECONDS = 180
SCHEDULE = {
    "base_lr": 0.001,
    "final_cycle_start": 18432,
    "final_cycle_lr": 0.00025,
    "adam_moments": "continuous_unmodified",
}
FAMILIES = {
    "position": (
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88c.json",
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run88c_batch_verification.json",
    ),
    "terminal": (
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run89a.json",
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run89a_batch_verification.json",
    ),
    "cut": (
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run89b.json",
        ROOT / "reports/taiji_n1_transition_factorial_lr_decay_run89b_batch_verification.json",
    ),
}
ARM_ORDER = ("position", "terminal", "cut")


class Refuse(Exception):
    """Loud rejection before any checkpoint load or confirmation query."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha(path: Path) -> str | None:
    return lr88.sha256(Path(path)) if Path(path).is_file() else None


def read_json(path: Path) -> dict:
    return lr88.read_json(Path(path))


def protocol() -> dict:
    value = deepcopy(base85.PROTOCOL)
    value.update(
        schema="n1-transition-factorial-three-family-v5-confirmation-v1",
        batch_reports={arm: sha(report) for arm, (report, _) in FAMILIES.items()},
        batch_verifications={arm: sha(verified) for arm, (_, verified) in FAMILIES.items()},
        v5_manifest_sha256=V5_SHA,
        learning_rate_schedule=dict(SCHEDULE),
        frozen_runner_sha256=FROZEN_RUNNER_SHA256,
        cpu_limit_seconds=CPU_LIMIT_SECONDS,
        retry_allowed=False,
    )
    return value


def identity() -> dict:
    return dict(
        protocol=protocol(),
        binding_runner=sha(Path(__file__).resolve()),
        base_evaluator=sha(Path(base85.__file__)),
        run88_controller=sha(Path(lr88.__file__)),
        run89_controller=sha(Path(lr89.__file__)),
        plan_sha256=sha(PLAN),
        training=train.identity(),
        face_builder=face.identity(),
        receipt_validator=sha(Path(receipt.__file__)),
    )


def family_receipts() -> dict:
    receipts = {}
    for arm, (report_path, verification_path) in FAMILIES.items():
        require(report_path.is_file(), f"{arm} batch report is missing: {report_path.name}")
        require(
            verification_path.is_file(),
            f"{arm} batch verification is missing: {verification_path.name}",
        )
        report = read_json(report_path)
        verification = read_json(verification_path)
        require(
            report.get("status") == "complete"
            and report.get("complete") is True
            and report.get("can_promote") is False
            and report.get("sources_unchanged") is True
            and report.get("code_unchanged") is True
            and report.get("cpu_accounting_complete") is True
            and "train_sha" not in report,
            f"{arm} batch report is not an engineering-complete sealed batch",
        )
        require(
            all(
                report.get(key) == 0
                for key in ("dev_queries", "holdout_queries", "confirmation_queries")
            ),
            f"{arm} batch already touched an unseen face",
        )
        require(
            verification.get("complete") is True
            and verification["execution"]["model_loads"] == 0
            and verification["execution"]["parameter_updates"] == 0
            and verification["execution"]["can_promote"] is False,
            f"{arm} batch verification is not a zero-load sealed reading",
        )
        schedule = report["protocol"].get("learning_rate_schedule", {})
        require(
            all(schedule.get(key) == value for key, value in SCHEDULE.items()),
            f"{arm} batch does not carry the frozen final-cycle decay schedule",
        )
        require(
            report["code_identity"].get("frozen_runner_sha256") == FROZEN_RUNNER_SHA256,
            f"{arm} batch does not reference the frozen run84 runner",
        )
        require(
            len(report["models"]) == 3 and all(m["arm"] == arm for m in report["models"]),
            f"{arm} batch must hold exactly three models of that arm",
        )
        receipts[arm] = dict(
            report=report,
            report_sha256=sha(report_path),
            verification=verification,
            verification_sha256=sha(verification_path),
        )
    control = {
        arm: receipts[arm]["report"]["code_identity"].get("control_report_sha256")
        for arm in ARM_ORDER
    }
    require(len(set(control.values())) == 1, "families differ on their sealed control report")
    for arm in ("terminal", "cut"):
        bound = receipts[arm]["verification"].get("bound_position_batch", {})
        require(
            bound.get("report_sha256") == receipts["position"]["report_sha256"]
            and bound.get("verification_sha256") == receipts["position"]["verification_sha256"],
            f"{arm} batch is not bound to the adjudicated position batch bytes",
        )
    return receipts


def cross_family_guard_conflicts(receipts: dict) -> dict:
    seen: dict[str, set] = {}
    for arm in ARM_ORDER:
        for path, digest in receipts[arm]["report"]["source_guards"].items():
            seen.setdefault(path, set()).add(digest)
    return {path: digests for path, digests in seen.items() if len(digests) > 1}


def endpoints(receipts: dict) -> tuple[dict, dict]:
    paths: dict[tuple[int, str], Path] = {}
    guards: dict[str, str] = {}
    for arm in ARM_ORDER:
        report = receipts[arm]["report"]
        for raw_path, digest in report["source_guards"].items():
            guards[raw_path] = digest
        for model in report["models"]:
            finals = [
                check for check in model["restore_checks"] if check["path"].endswith("_final.pt")
            ]
            require(
                len(finals) == 1, f"{arm} seed {model['seed']} needs exactly one final endpoint"
            )
            check = finals[0]
            path = Path(check["path"])
            require(
                path.is_file() and check["equal"] is True and sha(path) == check["sha256"],
                f"{arm} endpoint bytes differ from its sealed receipt: {path.name}",
            )
            key = (model["seed"], arm)
            require(key not in paths, f"duplicate frozen endpoint for {key}")
            paths[key] = path
            guards[str(path)] = check["sha256"]
    expected = {(seed, arm) for seed in lr88.SEEDS for arm in ARM_ORDER}
    require(set(paths) == expected, "nine frozen final endpoints required")
    return paths, guards


def readiness() -> tuple[dict, dict, dict, list]:
    started = time.process_time()
    receipts = family_receipts()
    paths, guards = endpoints(receipts)
    conflicts = cross_family_guard_conflicts(receipts)
    require(not conflicts, f"families disagree on a shared guarded input: {sorted(conflicts)[:3]}")
    admission = read_json(face.REPORT)
    seal = admission["manifest"]
    require(
        seal["sha256"] == V5_SHA and sha(face.MANIFEST) == seal["sha256"],
        "exact frozen v5 bytes required",
    )
    manifest = read_json(face.MANIFEST)
    face.validate(manifest)
    for path, digest in guards.items():
        candidate = Path(path)
        if candidate.is_file() and lr88.guard_sha256(candidate) != digest:
            raise Refuse(f"bound input changed since its batch sealed: {path}")
    seals = {
        "v5": seal,
        "readiness_cpu_seconds": round(time.process_time() - started, 6),
        "batches": {
            arm: {
                "report_sha256": receipts[arm]["report_sha256"],
                "verification_sha256": receipts[arm]["verification_sha256"],
            }
            for arm in ARM_ORDER
        },
    }
    return guards, seals, paths, manifest["episodes"]


def load_with_family_protocol(payload):
    """Pin the consumed interface: each family sealed its own protocol object."""
    recorded = train.PROTOCOL
    if payload.get("protocol") == recorded:
        return ORIGINAL_LOAD(payload)
    try:
        train.PROTOCOL = payload["protocol"]
        return ORIGINAL_LOAD(payload)
    finally:
        train.PROTOCOL = recorded


def install() -> None:
    base85.REPORT = REPORT
    base85.ATTEMPT = ATTEMPT
    base85.PROTOCOL = protocol()
    base85.identity = identity
    base85.readiness = readiness
    train.load = load_with_family_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-only", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    arguments = parser.parse_args()
    train.torch.set_num_threads(1)
    install()
    if arguments.readiness_only:
        guards, seals, paths, episodes = readiness()
        print(
            json.dumps(
                {
                    "complete": True,
                    "endpoints": len(paths),
                    "episodes": len(episodes),
                    "guarded_inputs": len(guards),
                    "v5_sha256": seals["v5"]["sha256"],
                    "readiness_cpu_seconds": seals["readiness_cpu_seconds"],
                    "model_loads": 0,
                    "confirmation_queries": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if not arguments.execute_approved:
        print(json.dumps(dict(read_only=True, protocol=protocol()), sort_keys=True))
        return 0
    result = base85.audit()
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("complete", "passed", "aggregate_exact", "gates", "cpu_seconds", "error")
            },
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
