"""10 closeout evidence card: read the four probe reports and the admission gate.

This step measures nothing. Every check names the file and field it read, so a
reader can re-run the check instead of trusting the verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUDGET_CPU_SECONDS = 120
BUDGET_WALL_SECONDS = 240
RETENTION_GATE = 0.030
LOSS_RATIO_FLOOR = 0.70
ALTERNATING = ("FTFTFT", "TFTFTF")

PROBE_REPORTS = {
    "10.4": "reports/taiji_core_subject_selection_10_4_20261101.json",
    "10.5": "reports/taiji_core_goal_conditioned_10_5_20261101.json",
    "10.6": "reports/taiji_core_action_verdict_10_6_20261101.json",
    "10.7": "reports/taiji_core_goal_training_10_7_20261101.json",
}
EXPECTED_EXITS = {
    "10.4": "model_below_random",
    "10.5": "contrast_below_floor",
    "10.6": "action_gap_below_floor",
    "10.7": "training_below_chance_plus",
}
CAPABILITY_FIELDS = {
    "10.4": "capability_passed",
    "10.5": "capability_claimed",
    "10.6": "capability_claimed",
    "10.7": "capability_claimed",
}
ADMISSION_REPORT = "reports/taiji_core_l3_admission_11_20261101.json"
CLOSEOUT_CARD = "reports/taiji_core_09_closeout_20261101.json"
ADMISSION_EXIT = "step11_not_admitted"
ADMISSION_PRECONDITION = "P2_one_capability_by_the_model"

FAILURE_STATUS = {
    "identity": "identity_mismatch",
    "exits": "exit_not_recorded",
    "capability": "capability_claim_forbidden",
    "mechanism": "mechanism_not_reproduced",
    "retention": "mechanism_not_reproduced",
    "loss": "mechanism_not_reproduced",
    "weights": "weights_published_forbidden",
    "admission": "admission_regressed",
    "budget": "budget_stop",
}
CHECK_ORDER = tuple(FAILURE_STATUS)


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def path_get(node, *keys):
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def hit_sequence(rows) -> str:
    if not isinstance(rows, list):
        return ""
    return "".join("T" if row.get("hit") else "F" for row in rows)


def lr_run(runs, target: float):
    for run in runs:
        lr = run.get("lr")
        if isinstance(lr, (int, float)) and abs(lr - target) <= 1e-12:
            return run
    return None


def mechanism_summary(training: dict) -> str:
    untrained = hit_sequence(path_get(training, "untrained_held_out", "rows"))
    runs = training.get("runs") if isinstance(training.get("runs"), list) else []
    arms = [f"lr={run.get('lr')} {hit_sequence(run.get('held_out'))}" for run in runs]
    return " / ".join([f"untrained {untrained}", *arms])


def audit(probes: dict, admission: dict, card: dict) -> list[dict]:
    found: dict[str, dict] = {}

    digests = {sid: report.get("parent_model_digest") for sid, report in probes.items()}
    card_digest = path_get(card, "identity", "parent_model_digest")
    identity_ok = (
        isinstance(card_digest, str)
        and all(isinstance(value, str) for value in digests.values())
        and len(set(digests.values())) == 1
        and set(digests.values()) == {card_digest}
    )
    found["identity"] = {
        "passed": identity_ok,
        "observed": {"closeout_card": card_digest, "probes": digests},
    }

    exits = {sid: report.get("status") for sid, report in probes.items()}
    found["exits"] = {"passed": exits == EXPECTED_EXITS, "observed": exits}

    claims = {}
    capability_ok = True
    for sid, field in CAPABILITY_FIELDS.items():
        value = probes[sid].get(field)
        admitted = probes[sid].get("training_admitted")
        claims[sid] = {field: value, "training_admitted": admitted}
        capability_ok = capability_ok and value is False and admitted is False
    found["capability"] = {"passed": capability_ok, "observed": claims}

    training = probes["10.7"]
    untrained = {
        "sequence": hit_sequence(path_get(training, "untrained_held_out", "rows")),
        "hits": path_get(training, "untrained_held_out", "hits"),
    }
    runs = training.get("runs") if isinstance(training.get("runs"), list) else []
    arms = [
        {
            "lr": run.get("lr"),
            "sequence": hit_sequence(run.get("held_out")),
            "hits": run.get("held_out_hits"),
        }
        for run in runs
    ]
    mechanism_ok = (
        untrained["sequence"] in ALTERNATING
        and untrained["hits"] == 3
        and len(arms) == 2
        and all(arm["sequence"] in ALTERNATING and arm["hits"] == 3 for arm in arms)
        and arms[0]["sequence"] != arms[1]["sequence"]
    )
    found["mechanism"] = {
        "passed": mechanism_ok,
        "observed": {"untrained": untrained, "arms": arms},
    }

    slow = lr_run(runs, 1e-5)
    fast = lr_run(runs, 1e-4)
    slow_drift = slow.get("retention_drift") if slow else None
    fast_drift = fast.get("retention_drift") if fast else None
    retention_ok = (
        isinstance(slow_drift, (int, float))
        and isinstance(fast_drift, (int, float))
        and fast_drift > RETENTION_GATE
        and slow_drift <= RETENTION_GATE
    )
    found["retention"] = {
        "passed": retention_ok,
        "observed": {"lr_1e-5": slow_drift, "lr_1e-4": fast_drift, "gate": RETENTION_GATE},
    }

    first = fast.get("first_loss") if fast else None
    last = fast.get("last_loss") if fast else None
    ratio = None
    if isinstance(first, (int, float)) and isinstance(last, (int, float)) and first:
        ratio = last / first
    loss_ok = isinstance(ratio, (int, float)) and ratio <= LOSS_RATIO_FLOOR
    found["loss"] = {
        "passed": loss_ok,
        "observed": {"first": first, "last": last, "ratio": ratio, "floor": LOSS_RATIO_FLOOR},
    }

    weights_ok = (
        training.get("weights_published") is False
        and training.get("default_model_replaced") is False
    )
    found["weights"] = {
        "passed": weights_ok,
        "observed": {
            "weights_published": training.get("weights_published"),
            "default_model_replaced": training.get("default_model_replaced"),
        },
    }

    preconditions = (
        admission.get("preconditions") if isinstance(admission.get("preconditions"), list) else []
    )
    p2 = next((row for row in preconditions if row.get("name") == ADMISSION_PRECONDITION), None)
    admission_ok = (
        admission.get("status") == ADMISSION_EXIT
        and admission.get("training_admitted") is False
        and admission.get("l3_scope_started") is False
        and isinstance(p2, dict)
        and p2.get("passed") is False
    )
    found["admission"] = {
        "passed": admission_ok,
        "observed": {
            "status": admission.get("status"),
            "training_admitted": admission.get("training_admitted"),
            "l3_scope_started": admission.get("l3_scope_started"),
            "P2_passed": p2.get("passed") if isinstance(p2, dict) else None,
        },
    }

    budgets = {}
    budget_ok = True
    for sid, report in probes.items():
        cpu = report.get("cpu_seconds")
        cap = path_get(report, "budget", "cpu_seconds")
        budgets[sid] = {"cpu_seconds": cpu, "cap": cap}
        budget_ok = (
            budget_ok
            and isinstance(cpu, (int, float))
            and isinstance(cap, (int, float))
            and cpu <= cap
        )
    found["budget"] = {"passed": budget_ok, "observed": budgets}

    return [{"id": check_id, **found[check_id]} for check_id in CHECK_ORDER]


def status_from(checks: list[dict]) -> str:
    by_id = {check["id"]: check["passed"] for check in checks}
    for check_id, status in FAILURE_STATUS.items():
        if by_id.get(check_id) is not True:
            return status
    return "closeout_10_evidence_complete"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10 closeout card is write-once")

    total_cpu, total_wall = time.process_time(), time.perf_counter()
    probes = {sid: load(relative) for sid, relative in PROBE_REPORTS.items()}
    admission = load(ADMISSION_REPORT)
    card = load(CLOSEOUT_CARD)

    checks = audit(probes, admission, card)
    status = status_from(checks)
    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"

    exits = {sid: probes[sid].get("status") for sid in probes}
    card_digest = path_get(card, "identity", "parent_model_digest")
    report = {
        "schema": "taiji-10-closeout-v1",
        "step": "10",
        "date": "2026-11-01",
        "status": status,
        "stage_exits": exits,
        "p2_satisfied": any(str(value).startswith("passed") for value in exits.values()),
        "step11_status": admission.get("status"),
        "stage_close_decision": "owner_pending",
        "mechanism": mechanism_summary(probes["10.7"]),
        "identity": {
            "closeout_card": card_digest,
            "probes": {sid: probes[sid].get("parent_model_digest") for sid in probes},
        },
        "checks": checks,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Evidence card only. A pass means the four pre-declared exits are all negative, the "
            "readings recompute and no capability is claimed. It does not show that the subject can "
            "select, plan or bind; it is not task capability; it does not approve closing step 10 "
            "(that decision is the owner's) and it authorises no training."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                *PROBE_REPORTS.values(),
                ADMISSION_REPORT,
                CLOSEOUT_CARD,
                "scripts/training/verify_taiji_closeout_10.py",
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
                "checks": {check["id"]: check["passed"] for check in checks},
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
