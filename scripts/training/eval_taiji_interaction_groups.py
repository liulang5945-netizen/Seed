"""Evaluate trace-grounded interaction groups on a deterministic S0 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionGroupEvaluatorConfig,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
    InteractionTraceEvent,
    Observation,
    TaijiConfig,
    TSKV8Adapter,
    project_native_adapter_episode,
)


def _episode(
    *,
    split: str,
    episode_id: str,
    context_id: str,
    owner_ids: tuple[str, ...],
    outcome: float,
    recovery_effect: float,
) -> InteractionTraceEpisode:
    full_episode_id = f"{split}-{episode_id}"
    outcome_id = f"{full_episode_id}:outcome"
    return InteractionTraceEpisode(
        episode_id=full_episode_id,
        checkpoint_revision=7,
        outcome_id=outcome_id,
        events=tuple(
            InteractionTraceEvent(
                event_id=f"{full_episode_id}:event:{index}",
                owner_id=owner_id,
                episode_id=full_episode_id,
                checkpoint_revision=7,
                outcome_id=outcome_id,
                resource_cost=0.4 + 0.1 * index,
            )
            for index, owner_id in enumerate(owner_ids)
        ),
        outcome=outcome,
        recovery_effect=recovery_effect,
        context_id=context_id,
    )


def build_corpus() -> InteractionTraceCorpus:
    """Build two unseen task contexts without semantic role labels."""

    train = (
        _episode(
            split="train",
            episode_id="ab-none",
            context_id="task-family-ab",
            owner_ids=(),
            outcome=0.0,
            recovery_effect=0.0,
        ),
        _episode(
            split="train",
            episode_id="ab-a",
            context_id="task-family-ab",
            owner_ids=("surface-a",),
            outcome=0.2,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="ab-b",
            context_id="task-family-ab",
            owner_ids=("surface-b",),
            outcome=0.3,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="ab-both",
            context_id="task-family-ab",
            owner_ids=("surface-a", "surface-b"),
            outcome=1.0,
            recovery_effect=0.7,
        ),
        _episode(
            split="train",
            episode_id="cd-none",
            context_id="task-family-cd",
            owner_ids=(),
            outcome=0.0,
            recovery_effect=0.0,
        ),
        _episode(
            split="train",
            episode_id="cd-c",
            context_id="task-family-cd",
            owner_ids=("surface-c",),
            outcome=0.6,
            recovery_effect=0.2,
        ),
        _episode(
            split="train",
            episode_id="cd-d",
            context_id="task-family-cd",
            owner_ids=("surface-d",),
            outcome=0.2,
            recovery_effect=0.1,
        ),
        _episode(
            split="train",
            episode_id="cd-both",
            context_id="task-family-cd",
            owner_ids=("surface-c", "surface-d"),
            outcome=-0.2,
            recovery_effect=0.05,
        ),
    )
    holdout = (
        _episode(
            split="holdout",
            episode_id="ab-none",
            context_id="unseen-family-ab",
            owner_ids=(),
            outcome=0.05,
            recovery_effect=0.0,
        ),
        _episode(
            split="holdout",
            episode_id="ab-a",
            context_id="unseen-family-ab",
            owner_ids=("surface-a",),
            outcome=0.25,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="ab-b",
            context_id="unseen-family-ab",
            owner_ids=("surface-b",),
            outcome=0.35,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="ab-both",
            context_id="unseen-family-ab",
            owner_ids=("surface-a", "surface-b"),
            outcome=1.1,
            recovery_effect=0.8,
        ),
        _episode(
            split="holdout",
            episode_id="cd-none",
            context_id="unseen-family-cd",
            owner_ids=(),
            outcome=0.05,
            recovery_effect=0.0,
        ),
        _episode(
            split="holdout",
            episode_id="cd-c",
            context_id="unseen-family-cd",
            owner_ids=("surface-c",),
            outcome=0.55,
            recovery_effect=0.2,
        ),
        _episode(
            split="holdout",
            episode_id="cd-d",
            context_id="unseen-family-cd",
            owner_ids=("surface-d",),
            outcome=0.15,
            recovery_effect=0.1,
        ),
        _episode(
            split="holdout",
            episode_id="cd-both",
            context_id="unseen-family-cd",
            owner_ids=("surface-c", "surface-d"),
            outcome=-0.3,
            recovery_effect=0.05,
        ),
    )
    return InteractionTraceCorpus(train=train, holdout=holdout)


def _native_config(seed: int) -> TaijiConfig:
    """Use a small deterministic native substrate for the replay Gate."""

    return TaijiConfig(
        region_sizes=(12, 8),
        synapse_fan_in=4,
        motor_fan_in=6,
        memory_units=16,
        memory_fan_in=4,
        memory_readout_fan_in=6,
        memory_meta_dim=6,
        memory_iterations=2,
        memory_time_dim=4,
        memory_episode_dim=4,
        lateral_fan_in=4,
        seed=seed,
    )


def _native_owner(symbol: int) -> str:
    """Create a stable opaque owner handle from the observed native surface."""

    digest = hashlib.sha256(f"text-byte:{int(symbol)}".encode()).hexdigest()[:24]
    return f"native-owner:{digest}"


def _run_native_episode(
    *,
    split: str,
    context_label: str,
    cell_label: str,
    active_symbols: tuple[int, ...],
    outcome: float,
    recovery_effect: float,
    seed: int,
) -> tuple[InteractionTraceEpisode, dict[str, object]]:
    """Run one factorial cell through the real native adapter and replay it."""

    episode_id = f"r2-s1-{split}-{context_label}-{cell_label}"
    adapter = TSKV8Adapter(_native_config(seed), episode_id=episode_id)
    factor_symbols = {97, 98, 99, 100}
    owner_id_by_event_id: dict[str, str | None] = {}
    resource_cost_by_event_id: dict[str, float] = {}
    for timestamp, symbol in enumerate((32, *active_symbols)):
        adapter.observe_event(
            Observation(
                modality="text-byte",
                value=symbol,
                timestamp=timestamp,
                source=f"r2-s1:{split}:{context_label}",
                provenance="deterministic-replay",
            ),
            learn=False,
        )
        event = adapter.cognitive_snapshot().events[-1]
        owner_id_by_event_id[event.event_id] = (
            _native_owner(symbol) if symbol in factor_symbols else None
        )
        resource_cost_by_event_id[event.event_id] = float(event.end_tick - event.start_tick + 1)

    adapter.act((0, 1), sample=False)
    adapter.settle_action(outcome, learn=False, terminal=True)
    checkpoint = adapter.native_checkpoint()
    episode = project_native_adapter_episode(
        adapter,
        context_id=f"{split}-native-{context_label}",
        owner_id_by_event_id=owner_id_by_event_id,
        recovery_effect=recovery_effect,
        resource_cost_by_event_id=resource_cost_by_event_id,
        checkpoint=checkpoint,
    )
    restored = TSKV8Adapter.from_native_checkpoint(checkpoint)
    replayed = project_native_adapter_episode(
        restored,
        context_id=f"{split}-native-{context_label}",
        owner_id_by_event_id=owner_id_by_event_id,
        recovery_effect=recovery_effect,
        resource_cost_by_event_id=resource_cost_by_event_id,
        checkpoint=checkpoint,
    )
    return episode, {
        "episode_id": episode.episode_id,
        "native_event_count": len(adapter.cognitive_snapshot().events),
        "projected_event_count": len(episode.events),
        "checkpoint_format": checkpoint["format"],
        "checkpoint_revision": checkpoint["version"],
        "replay_equal": episode.to_payload() == replayed.to_payload(),
        "event_ids": [event.event_id for event in episode.events],
        "outcome_id": episode.outcome_id,
    }


def build_native_corpus() -> tuple[InteractionTraceCorpus, list[dict[str, object]]]:
    """Build independent factorial cells from actual native adapter episodes."""

    contexts = (
        (
            "ab",
            (97, 98),
            (("none", (), 0.0, 0.0), ("first", (97,), 0.2, 0.1),
             ("second", (98,), 0.3, 0.1), ("pair", (97, 98), 1.0, 0.7)),
        ),
        (
            "cd",
            (99, 100),
            (("none", (), 0.0, 0.0), ("first", (99,), 0.6, 0.2),
             ("second", (100,), 0.2, 0.1), ("pair", (99, 100), -0.2, 0.05)),
        ),
    )
    episodes: dict[str, list[InteractionTraceEpisode]] = {"train": [], "holdout": []}
    replay_records: list[dict[str, object]] = []
    for split, seed in (("train", 7), ("holdout", 17)):
        for context_label, _, cells in contexts:
            for cell_label, active_symbols, outcome, recovery_effect in cells:
                episode, replay_record = _run_native_episode(
                    split=split,
                    context_label=context_label,
                    cell_label=cell_label,
                    active_symbols=active_symbols,
                    outcome=outcome + (0.05 if split == "holdout" else 0.0),
                    recovery_effect=recovery_effect + (0.0 if split == "train" else 0.05),
                    seed=seed,
                )
                episodes[split].append(episode)
                replay_records.append(replay_record)
    return InteractionTraceCorpus(
        train=tuple(episodes["train"]),
        holdout=tuple(episodes["holdout"]),
    ), replay_records


def evaluate() -> dict[str, object]:
    corpus = build_corpus()
    result = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    ).evaluate(corpus)
    report = result.to_report()
    report["task"] = "deterministic trace-grounded pair interaction attribution"
    report["input"] = {
        "train_episode_count": len(corpus.train),
        "holdout_episode_count": len(corpus.holdout),
        "train_checkpoint_revisions": sorted(corpus.train_checkpoint_revisions),
        "train_trace_digest": corpus.train_trace_digest,
        "holdout_trace_digest": corpus.holdout_trace_digest,
        "owner_ids_are_opaque": True,
        "semantic_role_labels": 0,
    }
    report["boundary"] = {
        "policy_mutation": False,
        "tool_selection": False,
        "provider_selection": False,
        "high_order_search": False,
        "holdout_can_add_evidence": False,
    }
    return report


def evaluate_native() -> dict[str, object]:
    corpus, replay_records = build_native_corpus()
    result = InteractionGroupEvaluator(
        InteractionGroupEvaluatorConfig(
            minimum_interaction=0.1,
            maximum_uncertainty=0.12,
            maximum_group_cardinality=2,
            maximum_pairwise_candidates=32,
            maximum_resource_cost=10.0,
        )
    ).evaluate(corpus)
    report = result.to_report()
    report["stage"] = "S1"
    report["task"] = "native adapter trace projection and checkpoint replay"
    report["input"] = {
        "train_episode_count": len(corpus.train),
        "holdout_episode_count": len(corpus.holdout),
        "train_checkpoint_revisions": sorted(corpus.train_checkpoint_revisions),
        "train_trace_digest": corpus.train_trace_digest,
        "holdout_trace_digest": corpus.holdout_trace_digest,
        "native_adapter": TSKV8Adapter.ADAPTER_NAME,
        "native_checkpoint_format": "taiji-native-v1",
        "owner_ids_are_opaque": True,
        "semantic_role_labels": 0,
        "replay_records": replay_records,
    }
    report["boundary"] = {
        "policy_mutation": False,
        "tool_selection": False,
        "provider_selection": False,
        "high_order_search": False,
        "holdout_can_add_evidence": False,
        "native_adapter_is_source_of_event_and_outcome_identity": True,
    }
    report["gate"]["criterion"] = (
        "native adapter Event/Outcome projections must replay exactly through a "
        "taiji-native-v1 checkpoint before trace-grounded attribution is evaluated"
    )
    report["metrics"]["native_checkpoint_replay"] = all(
        bool(item["replay_equal"]) for item in replay_records
    )
    report["metrics"]["native_event_ids_preserved"] = all(
        bool(item["event_ids"]) == (item["projected_event_count"] > 0)
        for item in replay_records
    )
    report["metrics"]["gate_passed"] = bool(
        report["metrics"].get("gate_passed", False)
        and report["metrics"]["native_checkpoint_replay"]
        and report["metrics"]["native_event_ids_preserved"]
    )
    report["gate"]["passed"] = report["metrics"]["gate_passed"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("s0", "s1"), default="s0")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.stage == "s1":
        report = evaluate_native()
        output = args.output or PROJECT_ROOT / "reports" / "taiji_w7_r2_interaction_groups_s1_20260829.json"
    else:
        report = evaluate()
        output = args.output or PROJECT_ROOT / "reports" / "taiji_w7_r2_interaction_groups_20260829.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": report["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
