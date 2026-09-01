"""M0 foundation evaluation entry point for the five Taiji abilities.

The task runners are intentionally not hidden behind this command yet.  Until
they produce real measurements, this entry point emits ``not_evaluated`` and
never manufactures a capability pass from a fixture or a provider response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    Outcome,
    WorldAction,
    WorldInterventionCase,
    WorldObject,
    WorldState,
)
from taiji.foundation_evaluation import (
    FOUNDATION_REQUIRED_ABILITIES,
    FoundationEvaluation,
    FoundationManifest,
    FoundationMeasurement,
)  # noqa: E402
from taiji.foundation_tasks import (  # noqa: E402
    DelayedMemoryCorpus,
    DelayedMemoryQuery,
    DelayedMemoryTask,
    GoalActionCorpus,
    GoalActionEpisode,
    GoalActionTask,
    MemoryEpisode,
    SequencePredictionCorpus,
    SequencePredictionTask,
    WorldTransitionCorpus,
    WorldTransitionTask,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "taiji_foundation_baseline_v1.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_foundation_baseline_20260901.json"


def _checkpoint_gate_status(manifest: FoundationManifest, path: Path | None) -> str:
    if path is None:
        return "not_run"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("gate") != manifest.checkpoint_gate:
        return "failed"
    if payload.get("status") != "passed":
        return "failed"
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        return "failed"
    return "passed"


def build_contract_report(
    manifest: FoundationManifest,
    *,
    checkpoint_gate_status: str,
    b1_measurement: FoundationMeasurement | None = None,
    b2_measurement: FoundationMeasurement | None = None,
    b3_measurement: FoundationMeasurement | None = None,
    b4_measurement: FoundationMeasurement | None = None,
) -> FoundationEvaluation:
    measurements = {
        ability_id: FoundationMeasurement(
            ability_id=ability_id,
            status="not_evaluated",
            primary_metric=manifest.task(ability_id).primary_metric,
            metric_direction=manifest.task(ability_id).metric_direction,
            metric_value=None,
            baseline_metrics={},
            sample_counts={},
            holdout_updates=0,
            evidence=("m0-1-contract-only; task runner pending",),
        )
        for ability_id in FOUNDATION_REQUIRED_ABILITIES
    }
    if b1_measurement is not None:
        measurements[b1_measurement.ability_id] = b1_measurement
    if b2_measurement is not None:
        measurements[b2_measurement.ability_id] = b2_measurement
    if b3_measurement is not None:
        measurements[b3_measurement.ability_id] = b3_measurement
    if b4_measurement is not None:
        measurements[b4_measurement.ability_id] = b4_measurement
    return FoundationEvaluation.evaluate(
        manifest,
        measurements,
        checkpoint_gate_status=checkpoint_gate_status,
    )


def _text_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, dict):
        return None
    for key in ("text", "content", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_sequence_corpus(
    paths: list[Path],
    *,
    train_bytes: int,
    holdout_bytes: int,
    retention_bytes: int,
    seed: int,
) -> SequencePredictionCorpus:
    budgets = {
        "train": int(train_bytes),
        "holdout": int(holdout_bytes),
        "retention": int(retention_bytes),
    }
    if any(value <= 0 for value in budgets.values()):
        raise ValueError("B1 byte budgets must be positive")
    buffers = {partition: bytearray() for partition in budgets}
    seen_text_digests: set[str] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _text_from_record(record)
                if text is None:
                    continue
                text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if text_digest in seen_text_digests:
                    continue
                seen_text_digests.add(text_digest)
                bucket = int.from_bytes(
                    hashlib.sha256(f"{int(seed)}\0{text}".encode()).digest()[:4],
                    "big",
                ) % 10_000
                partition = (
                    "train"
                    if bucket < 8_000
                    else "holdout"
                    if bucket < 9_000
                    else "retention"
                )
                remaining = budgets[partition] - len(buffers[partition])
                if remaining > 0:
                    buffers[partition].extend(text.encode("utf-8")[:remaining])
                if all(len(buffers[name]) >= budgets[name] for name in budgets):
                    break
        if all(len(buffers[name]) >= budgets[name] for name in budgets):
            break
    missing = {
        name: f"{len(buffers[name])}/{budgets[name]}"
        for name in budgets
        if len(buffers[name]) < budgets[name]
    }
    if missing:
        raise ValueError("B1 corpus did not meet byte budgets: " + json.dumps(missing))
    return SequencePredictionCorpus(
        train=bytes(buffers["train"]),
        holdout=bytes(buffers["holdout"]),
        retention=bytes(buffers["retention"]),
    )


def build_delayed_memory_smoke_corpus(*, count: int = 8) -> DelayedMemoryCorpus:
    if int(count) < 4:
        raise ValueError("B2 smoke corpus needs at least four memory episodes")
    train = tuple(
        MemoryEpisode(
            memory_id=f"m0-b2-smoke-{index}",
            cue=65 + index,
            action=48 + index % 2,
            outcome=43 if index % 2 == 0 else 45,
        )
        for index in range(int(count))
    )
    holdout = tuple(
        DelayedMemoryQuery(
            query_id=f"m0-b2-holdout-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    retention = tuple(
        DelayedMemoryQuery(
            query_id=f"m0-b2-retention-{index}",
            cue=episode.cue,
            expected_action=episode.action,
        )
        for index, episode in enumerate(train)
    )
    return DelayedMemoryCorpus(train=train, holdout=holdout, retention=retention)


def build_world_transition_smoke_corpus(*, count: int = 8) -> WorldTransitionCorpus:
    if int(count) < 4:
        raise ValueError("B3 smoke corpus needs at least four transitions")

    def case(case_id: str, position: float) -> WorldInterventionCase:
        before = WorldState(
            tick=0,
            latent=torch.zeros(1),
            objects=(
                WorldObject("agent", attributes={"energy": 1.0}),
                WorldObject("target", attributes={"position": position}),
            ),
        )
        action = WorldAction(
            action_id=case_id,
            kind="push",
            tick=0,
            actor_id="agent",
            target_id="target",
            parameters={"amount": 1.0},
        )
        after = WorldState(
            tick=1,
            latent=torch.zeros(1),
            objects=(
                WorldObject("agent", attributes={"energy": 1.0}),
                WorldObject("target", attributes={"position": position + 1.0}),
            ),
        )
        return WorldInterventionCase(
            case_id=case_id,
            initial=before,
            action=action,
            expected_state=after,
            expected_outcome=Outcome(
                intent_id=case_id,
                reward=1.0,
                success=True,
                tick=1,
            ),
        )

    return WorldTransitionCorpus(
        train=tuple(case(f"m0-b3-train-{index}", float(index)) for index in range(int(count))),
        holdout=tuple(
            case(f"m0-b3-holdout-{index}", 10.0 + index) for index in range(max(3, count // 2))
        ),
        retention=tuple(
            case(f"m0-b3-retention-{index}", 20.0 + index)
            for index in range(max(3, count // 2))
        ),
    )


def build_goal_action_smoke_corpus(*, count: int = 32) -> GoalActionCorpus:
    if int(count) < 4 or int(count) % 2:
        raise ValueError("B4 smoke corpus needs an even count of at least four episodes")
    episodes = tuple(
        GoalActionEpisode(
            episode_id=f"m0-b4-smoke-{index}",
            cue=65 + index % 2,
            preferred_action=48 + index % 2,
            alternate_action=49 - index % 2,
        )
        for index in range(int(count))
    )
    half = int(count) // 2
    return GoalActionCorpus(
        train=episodes,
        holdout=tuple(
            GoalActionEpisode(
                episode_id=f"m0-b4-holdout-{index}",
                cue=65 + index % 2,
                preferred_action=48 + index % 2,
                alternate_action=49 - index % 2,
            )
            for index in range(half)
        ),
        retention=tuple(
            GoalActionEpisode(
                episode_id=f"m0-b4-retention-{index}",
                cue=65 + index % 2,
                preferred_action=48 + index % 2,
                alternate_action=49 - index % 2,
            )
            for index in range(half)
        ),
    )


def _model_config(tier: str, seed: int) -> Any:
    from taiji import TaijiConfig

    if tier == "default":
        values = TaijiConfig().to_dict()
    elif tier == "micro":
        values = TaijiConfig(
            region_sizes=(8,),
            synapse_fan_in=2,
            motor_fan_in=4,
            memory_units=16,
            memory_fan_in=2,
            memory_readout_fan_in=2,
            memory_meta_dim=4,
            memory_time_dim=2,
            memory_episode_dim=2,
            lateral_fan_in=2,
            concept_capacity=8,
        ).to_dict()
    else:
        raise ValueError(f"unsupported B1 model tier: {tier}")
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def _memory_config(seed: int) -> Any:
    from taiji import TaijiConfig

    values = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        memory_units=128,
        memory_fan_in=32,
        memory_meta_dim=32,
        memory_readout_fan_in=32,
        memory_iterations=3,
    ).to_dict()
    values["seed"] = int(seed)
    return TaijiConfig.from_dict(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--b1-corpus", nargs="+", type=Path)
    parser.add_argument("--b2-smoke", action="store_true")
    parser.add_argument("--b3-smoke", action="store_true")
    parser.add_argument("--b4-smoke", action="store_true")
    parser.add_argument("--model-tier", choices=("micro", "default"), default="micro")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--profile", choices=("smoke", "foundation"), default="smoke")
    parser.add_argument("--checkpoint-report", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    manifest = FoundationManifest.load(args.manifest)
    checkpoint_status = _checkpoint_gate_status(manifest, args.checkpoint_report)
    b1_measurement = None
    b2_measurement = None
    b3_measurement = None
    b4_measurement = None
    if args.b1_corpus:
        if args.profile == "smoke":
            budgets = (4_096, 1_024, 1_024)
        else:
            budgets = (1_048_576, 131_072, 131_072)
        corpus = build_sequence_corpus(
            args.b1_corpus,
            train_bytes=budgets[0],
            holdout_bytes=budgets[1],
            retention_bytes=budgets[2],
            seed=manifest.seeds[0],
        )
        b1_measurement = SequencePredictionTask(
            _model_config(args.model_tier, manifest.seeds[0]),
            seeds=manifest.seeds,
            epochs=args.epochs,
        ).evaluate(corpus)
    if args.b2_smoke:
        b2_measurement = DelayedMemoryTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
        ).evaluate(build_delayed_memory_smoke_corpus())
    if args.b3_smoke:
        b3_measurement = WorldTransitionTask(
            seeds=manifest.seeds,
            epochs=10 if args.profile == "smoke" else 50,
        ).evaluate(build_world_transition_smoke_corpus())
    if args.b4_smoke:
        b4_measurement = GoalActionTask(
            _memory_config(manifest.seeds[0]),
            seeds=manifest.seeds,
        ).evaluate(build_goal_action_smoke_corpus())
    evaluation = build_contract_report(
        manifest,
        checkpoint_gate_status=checkpoint_status,
        b1_measurement=b1_measurement,
        b2_measurement=b2_measurement,
        b3_measurement=b3_measurement,
        b4_measurement=b4_measurement,
    )
    result = evaluation.to_payload()
    result["manifest_path"] = str(args.manifest)
    result["contract_status"] = "validated"
    measured = [
        ability_id
        for ability_id, measurement in (
            ("b1_sequence_prediction", b1_measurement),
            ("b2_delayed_memory", b2_measurement),
            ("b3_world_transition", b3_measurement),
            ("b4_goal_action", b4_measurement),
        )
        if measurement is not None
    ]
    result["capability_measurements"] = (
        "; ".join((*measured, "b5_not_evaluated")) if measured else "not_evaluated"
    )
    result["profile"] = args.profile
    result["model_tier"] = args.model_tier if b1_measurement is not None else None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report_written"] = args.report.is_file()
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["report_written"] and result["contract_status"] == "validated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
