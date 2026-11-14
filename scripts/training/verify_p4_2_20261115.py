"""P4.2 guard: contract, cache replay, emission support, and red directions.

Runs at zero training. It proves the frozen card is internally consistent and
that the two machinery pieces the run depends on are live before any budget is
spent: the column layout is the identity on pixel columns, and the Stage-2 cache
replays the body's own log-probability exactly. The run-report checks activate
only after the run writes its receipt; until then the receipt line says
``run_not_present``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_p4_2_22_5 as p4_2
from taiji.u2_observation_columns import column_values

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_2_byte_path_20261115.json"
CENSUS = ROOT / "reports/taiji_u2_p4_2_census_20261115.json"
PRICE = ROOT / "reports/taiji_u2_p4_2_price_20261115.json"
RUN = ROOT / "reports/taiji_u2_p4_2_byte_path_20261115.json"
REPORT = ROOT / "reports/taiji_u2_p4_2_guard_20261115.json"
GATES = {
    "full_answer_round_fit",
    "correction",
    "feedback_gain",
    "recovery",
    "world_paired_95_lower",
    "three_seed_direction",
    "a_retention",
    "check2",
}


def check(name: str, passed: bool, evidence) -> dict:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def pins_match(contract: dict) -> bool:
    pins = contract["source_pins"]
    paths = {
        "runner": Path(p4_2.__file__),
        "support": ROOT / "scripts/training/first_a_support_22_5.py",
        "column_pathway": ROOT / "taiji/u2_observation_columns.py",
        "subject": ROOT / "taiji/u2_subject.py",
        "p4_1_runner": ROOT / "scripts/training/run_p4_1_22_5.py",
    }
    return all(pins.get(name) == p4_2.sha_of(path) for name, path in paths.items())


def criteria_match(contract: dict) -> bool:
    raw = json.dumps(contract["criteria"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest() == contract["criteria_sha256"]


def layout_ok(values: torch.Tensor, expected: torch.Tensor) -> bool:
    return values.shape == (16, 48) and bool(torch.equal(values, expected))


def support_ok(contract: dict, example: dict, names: list[str]) -> tuple[bool, dict]:
    """The declared set must equal byte-side support minus everything the state needs."""
    byte_side, upstream = p4_2.gradient_support(contract, example)
    expected = sorted(set(byte_side) - set(upstream))
    listed = all(name in byte_side and name not in set(upstream) for name in names)
    return expected == sorted(names) and listed, {
        "expected": expected,
        "declared": sorted(names),
        "missing_from_declared": sorted(set(expected) - set(names)),
        "extra_in_declared": sorted(set(names) - set(expected)),
    }


def claim_check(contract: dict) -> bool:
    return contract["criteria"].get("accepted_parent") is False


def main() -> int:
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    price = json.loads(PRICE.read_text(encoding="utf-8"))
    data = p4_2.atomic_load(Path(contract["data_root"]))
    example = data["train"][0]

    checks = []
    checks.append(
        check("contract_status_frozen", contract["status"] == "frozen", contract["status"])
    )
    checks.append(
        check(
            "caps_positive",
            contract["cpu_cap"] > 0 and contract["wall_cap"] > 0,
            [contract["cpu_cap"], contract["wall_cap"]],
        )
    )
    checks.append(check("source_pins_match", pins_match(contract), contract["source_pins"]))
    checks.append(
        check("criteria_sha256_matches", criteria_match(contract), contract["criteria_sha256"])
    )
    checks.append(
        check(
            "census_matches_manifest",
            census["emission_parameters"] == contract["emission_parameters"],
            {
                "census": len(census["emission_parameters"]),
                "manifest": len(contract["emission_parameters"]),
            },
        )
    )
    checks.append(
        check(
            "census_support_identity",
            sorted(set(census["byte_side_support"]) - set(census["upstream_support"]))
            == census["emission_parameters"],
            census["excluded_overlap"][:5],
        )
    )
    checks.append(
        check(
            "data_digest_matches",
            p4_2.value_digest(data) == contract["data_digest"],
            contract["data_digest"],
        )
    )
    checks.append(
        check(
            "split_sizes",
            len(data["train"]) == contract["splits"]["train"] == 256
            and len(data["fit"]) == contract["splits"]["fit"] == 32,
            [len(data["train"]), len(data["fit"])],
        )
    )
    checks.append(
        check(
            "price_matches_cap",
            price["estimate"]["cap_cpu"] <= contract["cpu_cap"],
            [price["estimate"]["cap_cpu"], contract["cpu_cap"]],
        )
    )
    checks.append(
        check(
            "no_check_or_final_reads",
            price["discipline"]["check_reads"] == 0
            and contract["splits"]["check"] == "not read"
            and contract["splits"]["final"] == "not read",
            "price and contract both report zero sealed reads",
        )
    )
    checks.append(
        check(
            "run_blocked_pending_authority",
            contract["training_authority"] is None,
            "no owner authority is recorded, so run mode cannot start",
        )
    )
    checks.append(
        check(
            "verdict_table_defined",
            contract["criteria"]["main_gate"] == 0.8
            and contract["criteria"]["correction_gate"] == 0.8
            and contract["criteria"]["feedback_gain_gate"] == 0.05
            and contract["criteria"]["a_retention"].startswith("not_applicable")
            and contract["criteria"]["three_seed"].startswith("not_in_scope")
            and contract["criteria"]["check2"].startswith("not_read"),
            {
                "main": contract["criteria"]["main_gate"],
                "correction": contract["criteria"]["correction_gate"],
                "gain": contract["criteria"]["feedback_gain_gate"],
                "a_retention": contract["criteria"]["a_retention"],
            },
        )
    )
    checks.append(
        check(
            "accepted_parent_not_claimed",
            claim_check(contract)
            and contract["criteria"]["a_retention"].startswith("not_applicable"),
            contract["criteria"]["a_retention"],
        )
    )

    image = torch.arange(16 * 16 * 3, dtype=torch.float32).reshape(16, 16, 3)
    patches = image.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(16, 48)
    expected = image.permute(1, 0, 2).reshape(16, 48)
    values = column_values(patches)
    checks.append(
        check(
            "column_layout_round_trip",
            layout_ok(values, expected),
            float((values - expected).abs().max()),
        )
    )

    model = p4_2.build(contract)
    positions = p4_2.cache_positions(model, [p4_2.column_example(example)])
    sample = positions[:: max(1, len(positions) // 3)][:3]
    replay = []
    for item in sample:
        body = p4_2.body_log_probability(model, example, item["position"], item["symbol"])
        cached = -float(p4_2.emission_loss(model, [item]).detach())
        replay.append({"position": item["position"], "body": body, "cached": cached})
    checks.append(
        check(
            "cache_replay_equals_body",
            all(abs(row["body"] - row["cached"]) <= 1e-6 for row in replay),
            replay,
        )
    )

    ok, detail = support_ok(contract, p4_2.column_example(example), contract["emission_parameters"])
    checks.append(check("emission_support_exact", ok, detail))

    red = []
    wrong = copy.deepcopy(sample[0])
    wrong["symbol"] = (wrong["symbol"] + 1) % 257
    body = p4_2.body_log_probability(model, example, sample[0]["position"], sample[0]["symbol"])
    replayed = -float(p4_2.emission_loss(model, [wrong]).detach())
    red.append(
        {
            "name": "wrong_symbol_replay_mismatch",
            "triggered": abs(body - replayed) > 1e-6,
            "gap": abs(body - replayed),
        }
    )
    stale = p4_2.build(contract)
    with torch.no_grad():
        stale.encoder["rgb"].weight.add_(0.5)
    cached_stale = -float(p4_2.emission_loss(stale, [sample[0]]).detach())
    body_stale = p4_2.body_log_probability(
        stale, example, sample[0]["position"], sample[0]["symbol"]
    )
    red.append(
        {
            "name": "stale_state_replay_mismatch",
            "triggered": abs(body_stale - cached_stale) > 1e-6,
            "gap": abs(body_stale - cached_stale),
        }
    )
    permuted = values[torch.arange(15, -1, -1)]
    red.append(
        {
            "name": "permuted_layout_fails_round_trip",
            "triggered": not layout_ok(permuted, expected),
        }
    )
    claimed = copy.deepcopy(contract)
    claimed["criteria"]["accepted_parent"] = True
    red.append({"name": "parent_claim_fails", "triggered": not claim_check(claimed)})
    tampered = copy.deepcopy(contract)
    tampered["source_pins"]["subject"] = "0" * 64
    red.append({"name": "pin_tamper_fails", "triggered": not pins_match(tampered)})

    run_state = "run_not_present"
    run_checks = []
    if RUN.exists():
        report = json.loads(RUN.read_text(encoding="utf-8"))
        rows = report.get("feedback", {})
        run_checks.append(
            check(
                "run_denominator_all_32",
                rows.get("first_pass_correct", -1) + rows.get("first_pass_errors", -1) == 32,
                [rows.get("first_pass_correct"), rows.get("first_pass_errors")],
            )
        )
        run_checks.append(
            check(
                "run_control_has_zero_updates",
                rows.get("control_updates", 1) == 0 and rows.get("control_state_unchanged") is True,
                [rows.get("control_updates"), rows.get("control_state_unchanged")],
            )
        )
        run_checks.append(
            check(
                "run_does_not_claim_parent",
                report["discipline"]["accepted_parent"] is False
                and report["discipline"]["capability_claims"] == 0,
                report["discipline"],
            )
        )
        run_checks.append(
            check(
                "run_cost_within_cap",
                report["cost"]["total_cpu"] <= report["cost"]["cap_cpu"],
                [report["cost"]["total_cpu"], report["cost"]["cap_cpu"]],
            )
        )
        run_checks.append(
            check(
                "run_gate_table_complete",
                set(report["gate_table"]) == GATES,
                sorted(report["gate_table"]),
            )
        )
        run_state = "run_present"
    payload = {
        "schema": "taiji-u2-p4-2-guard-v1",
        "card": "22.5.P4.2",
        "date": "2026-11-15",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "checks": checks,
        "run_state": run_state,
        "run_checks": run_checks,
        "red_directions": red,
        "summary": {
            "checks": len(checks),
            "green": sum(row["passed"] for row in checks),
            "red_defined": len(red),
            "red_triggered": sum(row["triggered"] for row in red),
            "run_checks": len(run_checks),
        },
    }
    p4_2.write_report(REPORT, payload)
    print(json.dumps(payload["summary"]))
    return 0 if all(row["passed"] for row in checks) and all(row["triggered"] for row in red) else 1


if __name__ == "__main__":
    raise SystemExit(main())
