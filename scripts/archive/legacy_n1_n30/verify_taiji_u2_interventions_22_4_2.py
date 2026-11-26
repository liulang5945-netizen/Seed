"""22.4.2.5 zero-training fixed intervention matrix verification.

The matrix and gates are fixed before the reading. Position arms use the frozen
task contract; audio/proprio arms use a local cross-modal scorer schema. No model
forward, no optimizer, no project data. Raw media are recorded as SHA-256 digests
and actions as compact tuples so every trajectory is regenerable.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.u2_task_interventions import INCONSISTENT, UNKNOWN, VERIFY, build_matrix

REPORT = ROOT / "reports/taiji_u2_intervention_matrix_22_4_2_20261117.json"
SEEDS = (124, 125, 126)
CPU_CAP = 120.0
WALL_CAP = 240.0

CRITERIA = {
    "card": "22.4.2.5",
    "seeds": list(SEEDS),
    "arms": [
        "action_causality",
        "history_necessity",
        "appearance_dynamics",
        "occlusion",
        "audio_missing",
        "audio_conflict",
        "proprio_missing",
        "proprio_conflict",
    ],
    "position_gate": "oracle public history distinguishes the pair",
    "history_gate": "same last frame, different first frame, previous target differs",
    "occlusion_gate": "visible target answer, hidden target unknown",
    "cross_modal_gate": "present target answer, missing unknown, conflict verify",
    "unknown_target": UNKNOWN,
    "verify_target": VERIFY,
    "inconsistent_target": INCONSISTENT,
    "audio_frequencies": [400, 700, 1000],
    "audio_energy_min": 0.05,
    "audio_tolerance_hz": 100.0,
    "cpu_cap": CPU_CAP,
    "wall_cap": WALL_CAP,
    "verdict_order": [
        "budget_stop",
        "instrument_invalid",
        "fixture_invalid",
        "intervention_matrix_valid",
    ],
}


def _criteria_digest() -> str:
    body = json.dumps(CRITERIA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _simple(checks: dict) -> bool:
    return bool(checks["targets_differ"] and checks["legal_inputs_differ"])


def _status_ok(cases: list[dict], statuses: tuple[str, str]) -> bool:
    return tuple(case["status"] for case in cases) == statuses


def _seed_gates(matrix: dict) -> dict:
    arms = matrix["arms"]
    action = arms["action_causality"]["checks"]
    history = arms["history_necessity"]["checks"]
    dynamics = arms["appearance_dynamics"]["checks"]
    occlusion = arms["occlusion"]["checks"]
    audio_missing = arms["audio_missing"]["checks"]
    audio_conflict = arms["audio_conflict"]["checks"]
    proprio_missing = arms["proprio_missing"]["checks"]
    proprio_conflict = arms["proprio_conflict"]["checks"]

    action_ok = bool(
        _simple(action)
        and action["first_frames_equal"]
        and not action["last_frames_equal"]
        and not action["actions_equal"]
        and action["current_view_only_distinguishable"]
    )
    history_ok = bool(
        _simple(history)
        and history["last_frames_equal"]
        and not history["first_frames_equal"]
        and not history["actions_equal"]
        and not history["current_view_only_distinguishable"]
        and len(set(history["previous_view_targets"])) == 2
    )
    dynamics_ok = bool(
        _simple(dynamics)
        and dynamics["first_frames_equal"]
        and dynamics["actions_equal"]
        and not dynamics["last_frames_equal"]
        and dynamics["current_view_only_distinguishable"]
    )
    occlusion_ok = bool(
        _simple(occlusion)
        and occlusion["first_frames_equal"]
        and not occlusion["last_frames_equal"]
        and occlusion["current_view_only_distinguishable"]
        and _status_ok(arms["occlusion"]["cases"], ("answer", "unknown"))
    )
    audio_missing_ok = bool(
        _simple(audio_missing)
        and _status_ok(arms["audio_missing"]["cases"], ("answer", "unknown"))
        and arms["audio_missing"]["cases"][0]["target"] == matrix["name"]
    )
    audio_conflict_ok = bool(
        _simple(audio_conflict)
        and _status_ok(arms["audio_conflict"]["cases"], ("answer", "verify"))
        and arms["audio_conflict"]["cases"][0]["target"] == matrix["name"]
        and arms["audio_conflict"]["cases"][1]["target"] == VERIFY
    )
    proprio_missing_ok = bool(
        _simple(proprio_missing)
        and _status_ok(arms["proprio_missing"]["cases"], ("answer", "unknown"))
    )
    proprio_conflict_ok = bool(
        _simple(proprio_conflict)
        and _status_ok(arms["proprio_conflict"]["cases"], ("answer", "verify"))
        and arms["proprio_conflict"]["cases"][1]["target"] == INCONSISTENT
    )

    grouped: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for arm in arms.values():
        for case in arm["cases"]:
            grouped.setdefault((case["question"], case["legal_digest"]), set()).add(
                (case["target"], case["status"])
            )
    identical_inputs_consistent = all(len(values) == 1 for values in grouped.values())

    return {
        "action_causality": action_ok,
        "history_necessity": history_ok,
        "appearance_dynamics": dynamics_ok,
        "occlusion": occlusion_ok,
        "audio_missing": audio_missing_ok,
        "audio_conflict": audio_conflict_ok,
        "proprio_missing": proprio_missing_ok,
        "proprio_conflict": proprio_conflict_ok,
        "no_identical_inputs_different_targets": identical_inputs_consistent,
        "all_arms": all(
            [
                action_ok,
                history_ok,
                dynamics_ok,
                occlusion_ok,
                audio_missing_ok,
                audio_conflict_ok,
                proprio_missing_ok,
                proprio_conflict_ok,
                identical_inputs_consistent,
            ]
        ),
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    criteria_digest = _criteria_digest()
    matrices = {}
    instrument = {"audio_detections": [], "history_frame_equalities": []}
    seed_gates = {}
    for seed in SEEDS:
        matrix = build_matrix(seed)
        matrices[str(seed)] = matrix
        seed_gates[str(seed)] = _seed_gates(matrix)
        audio_checks = matrix["arms"]["audio_missing"]["checks"]
        instrument["audio_detections"].append(
            {
                "seed": seed,
                "detected": audio_checks["audio_detected"],
                "energy": audio_checks["audio_energy"],
                "body_frequency": matrix["arms"]["audio_missing"]["cases"][0]["receipt"][
                    "body_frequency"
                ],
            }
        )
        instrument["history_frame_equalities"].append(
            {
                "seed": seed,
                "last_frames_equal": matrix["arms"]["history_necessity"]["checks"][
                    "last_frames_equal"
                ],
                "first_frames_equal": matrix["arms"]["history_necessity"]["checks"][
                    "first_frames_equal"
                ],
            }
        )
    all_gates = all(gates["all_arms"] for gates in seed_gates.values())
    audio_detected = all(item["detected"] is not None for item in instrument["audio_detections"])
    cpu = round(time.process_time() - started_cpu, 4)
    wall = round(time.monotonic() - started_wall, 4)
    if cpu > CPU_CAP or wall > WALL_CAP:
        verdict = "budget_stop"
    elif not audio_detected:
        verdict = "instrument_invalid"
    elif not all_gates:
        verdict = "fixture_invalid"
    else:
        verdict = "intervention_matrix_valid"
    report = {
        "schema": "taiji-u2-22-4-2-intervention-matrix-v1",
        "card": "22.4.2.5",
        "phase": "zero_training_intervention_matrix",
        "verdict": verdict,
        "criteria": CRITERIA,
        "criteria_digest": criteria_digest,
        "seeds": list(SEEDS),
        "matrix": matrices,
        "seed_gates": seed_gates,
        "instrument": instrument,
        "discipline": {
            "model_forwards": 0,
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {"cpu": cpu, "wall": wall, "cpu_cap": CPU_CAP, "wall_cap": WALL_CAP},
    }
    if REPORT.exists():
        raise FileExistsError(str(REPORT))
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8") + b"\n")
    print(
        "intervention verdict="
        + verdict
        + " seeds="
        + str(len(SEEDS))
        + " gates="
        + str(all_gates)
        + " audio="
        + str(audio_detected)
        + " cpu="
        + str(cpu)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
