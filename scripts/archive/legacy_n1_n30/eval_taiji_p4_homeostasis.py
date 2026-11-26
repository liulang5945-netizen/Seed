"""Evaluate event-driven homeostasis and sleep/play lesions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    HomeostaticConfig,
    HomeostaticController,
    HomeostaticState,
    TSKV8Adapter,
)

MANIFEST_FORMAT = "taiji-p4-homeostasis-manifest-v1"
REPORT_FORMAT = "taiji-p4-homeostasis-v1"


def evaluate() -> dict[str, object]:
    controller = HomeostaticController()
    start = HomeostaticState(tick=0)
    signal = controller.update(
        start,
        prediction_error=0.9,
        novelty=0.9,
        reward=-1.0,
        resource_cost=0.8,
        mode="wake",
    )
    adaptive_mode = controller.select_mode(signal)
    fixed_mode = "wake"
    random_mode = random.Random(3).choice(("wake", "sleep", "play"))
    sleep_state = signal
    for _ in range(4):
        sleep_state = controller.update(sleep_state, mode="sleep")
    sleep_lesion_state = signal
    for _ in range(4):
        sleep_lesion_state = controller.update(sleep_lesion_state, mode="wake")

    play_start = HomeostaticState(tick=0, curiosity=0.8, fatigue=0.1, stress=0.5)
    play_state = controller.update(
        play_start,
        prediction_error=0.1,
        novelty=0.8,
        resource_cost=0.1,
        mode="play",
    )
    play_lesion_state = controller.update(
        play_start,
        prediction_error=0.1,
        novelty=0.8,
        resource_cost=0.1,
        mode="wake",
    )
    no_modulator = HomeostaticController(
        HomeostaticConfig(
            curiosity_gain=0.0,
            curiosity_decay=0.0,
            fatigue_gain=0.0,
            fatigue_recovery=0.0,
            stress_gain=0.0,
            stress_recovery=0.0,
            reward_relief=0.0,
            sleep_recovery=0.0,
            sleep_stress_recovery=0.0,
            play_fatigue_cost=0.0,
            play_stress_relief=0.0,
        )
    )
    no_modulator_state = no_modulator.update(
        start,
        prediction_error=0.9,
        novelty=0.9,
        reward=-1.0,
        resource_cost=0.8,
    )

    runtime = TSKV8Adapter()
    runtime.attach_homeostatic_controller(controller)
    runtime.observe(97, learn=False)
    before_outcome = runtime.cognitive_snapshot().homeostasis
    runtime.act((0, 1), sample=False)
    runtime.settle_action(-1.0, learn=False)
    after_outcome = runtime.cognitive_snapshot().homeostasis
    restored = TSKV8Adapter.from_native_checkpoint(runtime.native_checkpoint())
    restored_state = restored.cognitive_snapshot().homeostasis
    gate_passed = bool(
        signal.curiosity > start.curiosity
        and signal.fatigue > start.fatigue
        and signal.stress > start.stress
        and adaptive_mode == "sleep"
        and fixed_mode != adaptive_mode
        and random_mode != adaptive_mode
        and sleep_state.fatigue < sleep_lesion_state.fatigue
        and sleep_state.stress < sleep_lesion_state.stress
        and play_state.stress < play_lesion_state.stress
        and no_modulator_state.curiosity == 0.0
        and no_modulator_state.fatigue == 0.0
        and no_modulator_state.stress == 0.0
        and after_outcome.stress > before_outcome.stress
        and restored_state == after_outcome
    )
    return {
        "format": REPORT_FORMAT,
        "metrics": {
            "signal_state": {
                "curiosity": signal.curiosity,
                "fatigue": signal.fatigue,
                "stress": signal.stress,
            },
            "adaptive_mode": adaptive_mode,
            "fixed_schedule_lesion_mode": fixed_mode,
            "random_drive_lesion_mode": random_mode,
            "sleep_fatigue": sleep_state.fatigue,
            "sleep_lesion_fatigue": sleep_lesion_state.fatigue,
            "sleep_stress": sleep_state.stress,
            "sleep_lesion_stress": sleep_lesion_state.stress,
            "play_stress": play_state.stress,
            "play_lesion_stress": play_lesion_state.stress,
            "no_modulator_state": {
                "curiosity": no_modulator_state.curiosity,
                "fatigue": no_modulator_state.fatigue,
                "stress": no_modulator_state.stress,
            },
            "runtime_checkpoint_restored": restored_state == after_outcome,
        },
        "gate": {
            "passed": gate_passed,
            "criterion": "internal drives respond to event signals, select rest/play, and sleep/play/no-modulator lesions cause measurable loss",
        },
    }


def build_manifest() -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "drive wake, sleep, and play from prediction error, novelty, reward, and resource cost",
        "signals": ["prediction_error", "novelty", "reward", "resource_cost"],
        "modes": ["wake", "sleep", "play"],
        "lesions": ["fixed_schedule", "random_drive", "no_modulator", "sleep", "play"],
        "boundary": "homeostatic state regulation and runtime checkpoint only; not a complete life system",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_homeostasis_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_p4_homeostasis_baseline_20260825.json",
    )
    args = parser.parse_args()
    report = evaluate()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
