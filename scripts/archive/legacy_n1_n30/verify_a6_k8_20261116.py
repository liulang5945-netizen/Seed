"""Guard for the k=8 continuation arm: re-derive both readings from its bundle.

The k=8 arm is the one that resolved the A.5/A.6 gap, so it gets the stronger
check: its saved bundle is loaded in a fresh process, the 32 A rounds and the 24
task-B rounds are re-run, and both published numbers must come back exactly.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_a5_22_5 as a5
from scripts.training import run_a6_budget_20261116 as a6
from taiji.u2_history import atomic_load

MANIFEST = ROOT / "plans/manifests/taiji_u2_a6_budget_20261116.json"
REPORT = ROOT / "reports/taiji_u2_a6_k8_run_20261116.json"
GUARD_REPORT = ROOT / "reports/taiji_u2_a6_k8_guard_20261116.json"
RUNNER = ROOT / "scripts/training/run_a6_budget_20261116.py"


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def evaluate(payload: dict, contract: dict, derived: dict | None) -> dict[str, bool]:
    arm = payload["arm"]
    checks = {
        "runner_pin_matches": sha_of(RUNNER) == contract["source_pins"]["runner"],
        "continuation_authorized": bool(contract["continuation"].get("reply"))
        and float(contract["continuation"]["cap_cpu"]) > 0,
        "bundle_sha_matches": hashlib.sha256((ROOT / payload["bundle"]).read_bytes()).hexdigest()
        == payload["bundle_sha256"],
        "updates_match_budget": arm["updates"] == int(arm["budget"]) * payload["denominator"],
        "damage_is_a_difference": abs(
            arm["damage"] - (payload["baseline_a_fit"] - arm["a_fit_accuracy"])
        )
        < 1e-12,
        "correct_matches_rate": abs(
            arm["second_pass_accuracy"] - arm["correct"] / payload["denominator"]
        )
        < 1e-12,
        "cap_respected": payload["cost"]["cpu"]
        <= float(contract["continuation"]["cap_cpu"]) + 1e-6,
        "discipline_zero": payload["discipline"]["capability_claims"] == 0
        and payload["discipline"]["accepted_parent"] is False,
    }
    if derived is not None:
        checks["b_second_pass_reproduced"] = derived["correct"] == arm["correct"]
        checks["a_fit_reproduced"] = derived["a_fit_accuracy"] == arm["a_fit_accuracy"]
        checks["a_fit_digest_reproduced"] = derived["a_fit_digest"] == arm["a_fit_digest"]
    return checks


def drills(payload: dict, contract: dict) -> dict[str, bool]:
    out = {}

    def fires(mutated: dict, key: str) -> bool:
        return not evaluate(mutated, contract, None)[key]

    mutated = json.loads(json.dumps(payload))
    mutated["arm"]["correct"] = 24
    out["claim_every_error_was_corrected"] = fires(mutated, "correct_matches_rate")

    mutated = json.loads(json.dumps(payload))
    mutated["arm"]["a_fit_accuracy"] = 1.0
    out["hide_the_damage"] = fires(mutated, "damage_is_a_difference")

    mutated = json.loads(json.dumps(payload))
    mutated["arm"]["updates"] = 24
    out["understate_the_updates"] = fires(mutated, "updates_match_budget")

    mutated = json.loads(json.dumps(payload))
    mutated["cost"]["cpu"] = 1000.0
    out["exceed_the_cap"] = fires(mutated, "cap_respected")
    return out


def main() -> int:
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    contract = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    data = atomic_load(ROOT / contract["data_root"])
    items, _ = a5.task_b_examples(data, contract)
    template_of = lambda index: index % len(contract["task_b"]["templates"])  # noqa: E731
    base = a6.load_start(contract, int(contract["plan"]["seeds"][0]))
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    codes = a2.code_by_id(base, head, prior, data["train"] + data["fit"], contract)

    model = a6.load_start(contract, int(contract["plan"]["seeds"][0]))
    model.load_state_dict(torch.load(ROOT / payload["bundle"], weights_only=False)["model"])
    model.eval()
    a_rows = [a5.a_round(model, example, codes[example["id"]]) for example in data["fit"]]
    b_rows = a5.b_pass(model, items, codes, template_of)
    derived = {
        "a_fit_accuracy": a5.accuracy(a_rows),
        "a_fit_digest": a5.rows_digest(a_rows),
        "correct": sum(int(row["correct"]) for row in b_rows),
    }
    checks = evaluate(payload, contract, derived)
    fired = drills(payload, contract)
    passed = all(checks.values()) and all(fired.values())
    report = {
        "schema": "taiji-u2-a6-k8-guard-v1",
        "card": "22.5.A.6",
        "date": "2026-11-16",
        "verdict": "guard_passed" if passed else "guard_failed",
        "checks": checks,
        "drills": fired,
        "rederived_from_bundle": derived,
        "published": {
            "correct": payload["arm"]["correct"],
            "a_fit_accuracy": payload["arm"]["a_fit_accuracy"],
            "a_fit_digest": payload["arm"]["a_fit_digest"],
        },
        "measured": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
        },
    }
    GUARD_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "checks": f"{sum(checks.values())}/{len(checks)}",
                "drills": f"{sum(fired.values())}/{len(fired)}",
                "cpu": report["measured"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
