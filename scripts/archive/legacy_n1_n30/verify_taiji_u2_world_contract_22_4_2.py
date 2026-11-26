"""Audit the frozen world design and local algebra without a model or rollout."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_world_contract_20261112.json"
REPORT = ROOT / "reports/taiji_u2_world_contract_22_4_2_20261112.json"


def require(condition, identity):
    if not condition:
        raise ValueError(identity)


def validate(c, pins=True):
    rows = []

    def record(identity, condition, detail):
        require(condition, identity)
        rows.append({"id": identity, "passed": True, "detail": detail})

    record("W01", c["status"] == "frozen_design", "design scope")
    if pins:
        for name, expected in c["source_pins"].items():
            actual = hashlib.sha256(
                (ROOT / name).read_text(encoding="utf-8-sig").encode()
            ).hexdigest()
            require(actual == expected, "W01 source " + name)
    p, a, camera, audio = c["physics"], c["action"], c["camera"], c["audio"]
    record(
        "W02",
        a["width"] == 15
        and a["world_dt"] == {"act": 0.05, "inspect": 0.05, "ask": 0}
        and a["duration_semantics"] == "within_tick_force_or_exposure",
        "world dt and pulse separated",
    )
    record(
        "W03",
        min(p["masses"]) > 0
        and min(p["drag_rates"]) > 0
        and 0 <= p["restitution"] <= 1
        and p["substep_max"] <= 0.001,
        "finite positive physical domains",
    )
    maximum_error = 0.0
    for m1 in p["masses"]:
        for m2 in p["masses"]:
            for speed in (0.01, 1, 10):
                v1, v2 = speed, -speed
                before = 0.5 * (m1 * v1 * v1 + m2 * v2 * v2)
                relative = v2 - v1
                impulse = -(1 + p["restitution"]) * relative / (1 / m1 + 1 / m2)
                after = 0.5 * (m1 * (v1 - impulse / m1) ** 2 + m2 * (v2 + impulse / m2) ** 2)
                predicted = -(1 - p["restitution"] ** 2) * relative**2 / (2 * (1 / m1 + 1 / m2))
                maximum_error = max(maximum_error, abs(after - before - predicted))
                require(after <= before + 1e-10 and abs(after - before - predicted) <= 1e-10, "W04")
    record("W04", True, {"local_collision_energy_max_error": maximum_error})
    for gamma in p["drag_rates"]:
        require((1 / (1 + gamma * p["substep_max"])) ** 2 <= 1, "W05")
    for tau in (0, 1e-10, 0.0075, 0.025, 0.05):
        integral = sum(max(0, min(1, (tau - i * 0.001) / 0.001)) * 0.001 for i in range(50))
        require(abs(integral - tau) < 1e-14, "W05")
    record("W05", True, "drag and fractional pulse local identities")
    for intensity in (0, 0.5, 1):
        half = 1 - 0.75 * intensity
        for target in (-1, 0, 1):
            center = (1 - half) * target
            require(center - half >= -1 and center + half <= 1, "W06")
    record(
        "W06",
        camera["side"] == 16 and camera["patch"] == 4,
        {"rgb_shape": [16, 48], "crop_inside_arena": True},
    )
    frames = (audio["samples"] - audio["fft"]) // audio["hop"] + 1
    record(
        "W07",
        frames == 7
        and audio["fft"] // 2 + 1 == 33
        and not audio["center"]
        and audio["periodic_hann"]
        and math.log1p(audio["fft"] / 2) < 6,
        "audio shape and magnitude bound",
    )
    record(
        "W08",
        len(c["proprio"]["channels"]) == 8
        and c["proprio"]["valid_bits"] == 8
        and c["proprio"]["coordinates"] == "all_zero",
        "proprio 1x16 and fixed coordinates",
    )
    rights = c["permissions"]
    record(
        "W09",
        not set(rights["public"]) & set(rights["private"])
        and "mass" not in rights["public"]
        and "score_only" not in rights["public"]
        and rights["outcome_mode"] == "execution_neutral"
        and rights["outcome"] == [0, 1, 0, 0, 1, 1]
        and rights["arrival_clock"] == "host_process_epoch",
        "public fields and feedback clock",
    )
    interface = c["interface"]
    record(
        "W10",
        {"rng_states", "transactions", "audio_bank", "audio_ring", "bodies"}
        <= set(interface["snapshot_fields"])
        and interface["duplicate_same_command"] == "cached"
        and interface["duplicate_changed_command"] == "reject"
        and interface["unknown_authority"] == "wait",
        "transaction and restore obligations",
    )
    history = c["history"]
    reserve = (
        history["B_rounds"] * history["reserved_operations_per_round"]
        + history["initial_operations_reserve"]
    )
    record(
        "W11",
        history["page_entries"] * history["max_pages"] >= reserve
        and history["active_operations"] == 32
        and history["implementation_status"] == "required_pending",
        {
            "existing_entry_limit": history["current_entries"],
            "B_minimum": 48 * 4,
            "B_reserve": reserve,
            "segmented_capacity": history["page_entries"] * history["max_pages"],
        },
    )
    record(
        "W12",
        c["future_gates"]["actual_16_round_reachability_required"]
        and c["future_gates"]["legal_history_oracle_min"] == 0.95
        and c["future_gates"]["blank_history_max"] == 0.60,
        "future task gates retained",
    )
    record(
        "W13",
        all(
            c["scope"][key] == 0
            for key in (
                "model_forwards",
                "optimizer_steps",
                "world_rollouts",
                "project_data_reads",
                "check_reads",
                "final_reads",
            )
        ),
        "static and local algebra scope",
    )
    return rows


def main():
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rows = validate(c)
    mutations = [
        ("W02", ("action", "world_dt", "act"), 0.06),
        ("W03", ("physics", "restitution"), 1.2),
        ("W07", ("audio", "center"), True),
        ("W09", ("permissions", "public"), c["permissions"]["public"] + ["mass"]),
        ("W09", ("permissions", "arrival_clock"), "world_time"),
        ("W10", ("interface", "duplicate_changed_command"), "execute"),
        ("W11", ("history", "max_pages"), 1),
        ("W13", ("scope", "optimizer_steps"), 1),
    ]
    negative = []
    for expected, path, value in mutations:
        mutant = copy.deepcopy(c)
        node = mutant
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        try:
            validate(mutant, pins=False)
        except ValueError as error:
            require(str(error).startswith(expected), "wrong negative control cause")
            negative.append({"path": list(path), "caught_by": expected})
        else:
            raise AssertionError(f"Negative control passed: {path}")
    cpu, wall = time.process_time(), time.monotonic() - STARTED
    require(cpu <= c["budget"]["cpu_cap"] and wall <= c["budget"]["wall_cap"], "budget_stop")
    report = {
        "schema": "taiji-u2-world-design-verification-v1",
        "verdict": "frozen_design_contract_verified",
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "checks": rows,
        "negative_controls": negative,
        "scope": c["scope"],
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "run_conditions": {"device": "cpu", "arithmetic": "python_float64", "python": sys.version},
        "pending": [
            "world implementation and actual trajectories",
            "float32/64 codec and subject graph",
            "executor dispatch and exogenous intent",
            "segmented history and full recovery",
            "legal-history oracle and task behavior",
        ],
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "checks": len(rows),
                "negative_controls": len(negative),
                "cpu_seconds": cpu,
                "wall_seconds": wall,
            }
        )
    )


if __name__ == "__main__":
    main()
