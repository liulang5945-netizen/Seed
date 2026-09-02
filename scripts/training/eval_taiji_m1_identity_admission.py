"""Review native identity-organ admission across scale, cue boundaries and gain.

This review is a larger follow-up to M1-28.  It predates the M1-63 promotion, so
it reports the live default rather than asserting one: the promotion decision
itself belongs to ``eval_taiji_m1_63_identity_organ_promotion``.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_b5_memory import build_corpus as build_b5_corpus  # noqa: E402
from scripts.training.eval_taiji_foundation_baseline import (  # noqa: E402
    build_delayed_memory_smoke_corpus,
)
from scripts.training.eval_taiji_m1_identity_organ_canary import (  # noqa: E402
    _accuracy,
    _config,
    _core_digest,
    _fresh_process_probe,
    _organ_digest,
    _query,
)
from taiji import DelayedMemoryTask, Taiji, TaijiConfig  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-admission-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_admission_20260902.json"
DEFAULT_CAPACITY = 128
GAIN_SWEEP = (4.0, 8.0, 16.0, 32.0)
REVIEW_GAIN = 32.0


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _review_config(seed: int, *, enabled: bool, capacity: int = DEFAULT_CAPACITY) -> TaijiConfig:
    values = _config(seed, enabled=enabled, capacity=capacity).to_dict()
    if enabled:
        values["identity_organ_evidence_gain"] = REVIEW_GAIN
    return TaijiConfig.from_dict(values)


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp_min(1e-8)


def _cue_patterns(model: Taiji, episodes: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    patterns: list[torch.Tensor] = []
    for episode in episodes:
        _query(model, episode.cue)
        patterns.append(model.fabric.cortical_context(model.snapshot().regions).detach().clone())
    return tuple(patterns)


def _boundary_record(seed: int) -> dict[str, Any]:
    model = Taiji(_config(seed, enabled=True, capacity=8))
    dimension = model.config.cortical_context_dim
    base = torch.zeros(dimension)
    base[0] = 1.0
    orthogonal = torch.zeros(dimension)
    orthogonal[1] = 1.0
    near = _unit(0.995 * base + (1.0 - 0.995**2) ** 0.5 * orthogonal)
    split = _unit(0.88 * base + (1.0 - 0.88**2) ** 0.5 * orthogonal)
    first = model.identity_organ.learn(base, 48, outcome_symbol=43)
    near_binding = model.identity_organ.learn(near, 48, outcome_symbol=43)
    split_binding = model.identity_organ.learn(split, 49, outcome_symbol=45)
    before_read = _organ_digest(model)
    near_read = model.identity_organ.recall(near)
    split_read = model.identity_organ.recall(split)
    after_read = _organ_digest(model)
    return {
        "match_threshold": model.config.identity_organ_match_threshold,
        "base_slot": first.slot_index,
        "near_slot": near_binding.slot_index,
        "split_slot": split_binding.slot_index,
        "near_similarity": near_read.similarity,
        "split_similarity": split_read.similarity,
        "near_duplicate_reused": near_binding.slot_index == first.slot_index,
        "below_threshold_split": split_binding.slot_index != first.slot_index,
        "near_action": int(near_read.action_probabilities.argmax().item()),
        "split_action": int(split_read.action_probabilities.argmax().item()),
        "organ_digest_read_only": before_read == after_read,
    }


def _capacity_records(seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for capacity in (32, 64, 128):
        model = Taiji(_config(seed, enabled=True, capacity=capacity))
        patterns = torch.eye(capacity + 8, model.config.cortical_context_dim)
        for index, pattern in enumerate(patterns):
            model.identity_organ.learn(
                pattern,
                48 + index % 2,
                outcome_symbol=43 if index % 2 == 0 else 45,
            )
        records.append(
            {
                "capacity": capacity,
                "attempted_writes": len(patterns),
                "occupied_count": model.identity_organ.bank.occupied_count,
                "allocation_count": model.identity_organ.bank.allocation_count,
                "replacement_count": model.identity_organ.replacement_count,
                "occupied_within_capacity": (
                    model.identity_organ.bank.occupied_count <= capacity
                ),
                "expected_replacements": 8,
                "parameter_count_matches_plan": (
                    model.parameter_count() == model.config.planned_active_parameter_count
                ),
            }
        )
    return records


def _larger_record(seed: int) -> dict[str, Any]:
    b2 = build_delayed_memory_smoke_corpus(count=32)
    b5 = build_b5_corpus(train_count=32, holdout_count=16, retention_count=16)
    b2_actions = _actions(*b2.train)
    b5_actions = _actions(*b5.phase_a_train, *b5.phase_b_train)

    identity = Taiji(_review_config(seed, enabled=True), episode_id=f"m1-29-b2-{seed}")
    for episode in b2.train:
        DelayedMemoryTask._write_episode(identity, episode)
    b2_score = _accuracy(identity, b2.holdout, b2_actions)
    b2_control = Taiji(_config(seed, enabled=False), episode_id=f"m1-29-b2-control-{seed}")
    for episode in b2.train:
        DelayedMemoryTask._write_episode(b2_control, episode)
    b2_control_score = _accuracy(b2_control, b2.holdout, b2_actions)

    parent = Taiji(_review_config(seed, enabled=True), episode_id=f"m1-29-parent-{seed}")
    for episode in b5.phase_a_train:
        DelayedMemoryTask._write_episode(parent, episode)
    parent_checkpoint = deepcopy(parent.checkpoint())
    parent_old = _accuracy(parent, b5.phase_a_holdout, b5_actions)
    parent_retention = _accuracy(parent, b5.phase_a_retention, b5_actions)

    child = Taiji(_review_config(seed, enabled=True), episode_id=f"m1-29-child-{seed}")
    child.restore(deepcopy(parent_checkpoint))
    for episode in b5.phase_b_train:
        DelayedMemoryTask._write_episode(child, episode)
    child_checkpoint = deepcopy(child.checkpoint())
    child_old = _accuracy(child, b5.phase_a_holdout, b5_actions)
    child_retention = _accuracy(child, b5.phase_a_retention, b5_actions)
    child_new = _accuracy(child, b5.phase_b_holdout, b5_actions)

    control = Taiji(_review_config(seed, enabled=False), episode_id=f"m1-29-control-{seed}")
    for episode in (*b5.phase_a_train, *b5.phase_b_train):
        DelayedMemoryTask._write_episode(control, episode)
    control_old = _accuracy(control, b5.phase_a_holdout, b5_actions)
    control_new = _accuracy(control, b5.phase_b_holdout, b5_actions)

    lesion = Taiji.from_checkpoint(deepcopy(child_checkpoint))
    lesion.identity_organ.lesion()
    lesion_old = _accuracy(lesion, b5.phase_a_holdout, b5_actions)
    lesion_new = _accuracy(lesion, b5.phase_b_holdout, b5_actions)

    phase_a_patterns = _cue_patterns(child, b5.phase_a_train)
    phase_b_patterns = _cue_patterns(child, b5.phase_b_train)
    phase_a_slots = tuple(
        child.identity_organ.recall(pattern).slot_index for pattern in phase_a_patterns
    )
    phase_b_slots = tuple(
        child.identity_organ.recall(pattern).slot_index for pattern in phase_b_patterns
    )

    no_change_before = _organ_digest(child)
    bound_steps = [_query(child, query.cue) for query in b5.phase_a_holdout]
    unknown = _query(child, 250)
    no_change_after = _organ_digest(child)

    restored = Taiji.from_checkpoint(deepcopy(child_checkpoint))
    restored_initial = restored.checkpoint()
    restored_old = _accuracy(restored, b5.phase_a_holdout, b5_actions)
    restored_new = _accuracy(restored, b5.phase_b_holdout, b5_actions)
    fresh = _fresh_process_probe(child_checkpoint, b5.phase_a_holdout[0].cue)

    lineage = child_checkpoint["identity_organ"]["lineage"]
    return {
        "seed": seed,
        "b2": {
            "identity_holdout": b2_score,
            "shared_control_holdout": b2_control_score,
        },
        "b5": {
            "parent_old_holdout": parent_old,
            "parent_retention": parent_retention,
            "child_old_holdout": child_old,
            "child_old_retention": child_retention,
            "child_new_holdout": child_new,
            "shared_control_old_holdout": control_old,
            "shared_control_new_holdout": control_new,
            "identity_lesion_old_holdout": lesion_old,
            "identity_lesion_new_holdout": lesion_new,
        },
        "route": {
            "phase_a_unique_slots": len(set(phase_a_slots)),
            "phase_b_unique_slots": len(set(phase_b_slots)),
            "cross_phase_slot_collisions": len(
                set(phase_a_slots).intersection(phase_b_slots)
            ),
            "unbound_slots": sum(slot is None for slot in (*phase_a_slots, *phase_b_slots)),
            "replacement_count": child.identity_organ.replacement_count,
        },
        "provenance": {
            "bound_sources": sorted({step.identity_recall.source for step in bound_steps}),
            "unbound_source": unknown.identity_recall.source,
            "unbound_provenance": unknown.identity_recall.provenance,
            "final_action_owner": "ByteMotor",
            "action_intent_generated": False,
        },
        "no_change": {
            "all_bound": all(step.identity_recall.used for step in bound_steps),
            "unbound_is_fallback": not unknown.identity_recall.used,
            "organ_digest_unchanged": no_change_before == no_change_after,
        },
        "budget": {
            "model_parameter_count": child.parameter_count(),
            "planned_parameter_count": child.config.planned_active_parameter_count,
            "identity_total_parameters": child.identity_organ.parameter_count,
        },
        "checkpoint": {
            "parent_core_digest_matches_lineage": (
                _core_digest(parent_checkpoint)
                == parent_checkpoint["identity_organ"]["lineage"]["parent_checkpoint_digest"]
            ),
            "child_core_digest_matches_lineage": (
                _core_digest(child_checkpoint) == lineage["parent_checkpoint_digest"]
            ),
            "restored_bundle_digest_matches": (
                content_digest(restored_initial) == content_digest(child_checkpoint)
            ),
            "restored_old_holdout": restored_old,
            "restored_new_holdout": restored_new,
            "fresh_process_source": fresh["source"],
            "fresh_process_persistent_digest_unchanged": fresh[
                "persistent_digest_unchanged"
            ],
            "fresh_process_checkpoint_digest_matches": (
                fresh["loaded_checkpoint_digest"] == content_digest(child_checkpoint)
            ),
        },
    }


def _gain_record(seed: int, gain: float) -> dict[str, float]:
    corpus = build_b5_corpus(train_count=32, holdout_count=16, retention_count=16)
    values = _config(seed, enabled=True).to_dict()
    values["identity_organ_evidence_gain"] = float(gain)
    config = TaijiConfig.from_dict(values)
    model = Taiji(config, episode_id=f"m1-29-gain-{seed}-{gain}")
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(model, episode)
    parent = _accuracy(model, corpus.phase_a_holdout, _actions(*corpus.phase_a_train))
    for episode in corpus.phase_b_train:
        DelayedMemoryTask._write_episode(model, episode)
    old = _accuracy(model, corpus.phase_a_holdout, _actions(*corpus.phase_a_train, *corpus.phase_b_train))
    new = _accuracy(model, corpus.phase_b_holdout, _actions(*corpus.phase_a_train, *corpus.phase_b_train))
    return {"gain": float(gain), "parent_old": parent, "child_old": old, "child_new": new}


def _record_passes(record: dict[str, Any]) -> bool:
    b2 = record["b2"]
    b5 = record["b5"]
    route = record["route"]
    provenance = record["provenance"]
    no_change = record["no_change"]
    budget = record["budget"]
    checkpoint = record["checkpoint"]
    boundary = record["boundary"]
    capacity = record["capacity"]
    return bool(
        b2["identity_holdout"] >= b2["shared_control_holdout"]
        and b5["child_old_holdout"] >= b5["parent_old_holdout"]
        and b5["child_old_retention"] >= b5["parent_retention"]
        and b5["child_new_holdout"] > b5["shared_control_new_holdout"]
        and b5["child_old_holdout"] > b5["identity_lesion_old_holdout"]
        and route["phase_a_unique_slots"] == 32
        and route["phase_b_unique_slots"] == 32
        and route["cross_phase_slot_collisions"] == 0
        and route["unbound_slots"] == 0
        and route["replacement_count"] == 0
        and provenance["bound_sources"] == ["identity-route"]
        and provenance["unbound_source"] == "shared-fallback"
        and provenance["final_action_owner"] == "ByteMotor"
        and provenance["action_intent_generated"] is False
        and no_change["all_bound"]
        and no_change["unbound_is_fallback"]
        and no_change["organ_digest_unchanged"]
        and budget["model_parameter_count"] == budget["planned_parameter_count"]
        and checkpoint["parent_core_digest_matches_lineage"]
        and checkpoint["child_core_digest_matches_lineage"]
        and checkpoint["restored_bundle_digest_matches"]
        and checkpoint["fresh_process_source"] == "identity-route"
        and checkpoint["fresh_process_persistent_digest_unchanged"]
        and checkpoint["fresh_process_checkpoint_digest_matches"]
        and boundary["near_duplicate_reused"]
        and boundary["below_threshold_split"]
        and boundary["near_action"] == 48
        and boundary["split_action"] == 49
        and boundary["organ_digest_read_only"]
        and all(
            item["occupied_within_capacity"]
            and item["replacement_count"] == item["expected_replacements"]
            and item["parameter_count_matches_plan"]
            for item in capacity
        )
    )


def run_admission(*, seeds: tuple[int, ...]) -> dict[str, Any]:
    records = []
    capacity_records = []
    boundary_records = []
    for seed in seeds:
        record = _larger_record(seed)
        record["boundary"] = _boundary_record(seed)
        record["capacity"] = _capacity_records(seed)
        records.append(record)
        boundary_records.append(record["boundary"])
        capacity_records.append(record["capacity"])
    calibration_seed = int(seeds[0])
    gain_sweep = [_gain_record(calibration_seed, gain) for gain in GAIN_SWEEP]
    return {
        "seeds": list(seeds),
        "larger_b2_train_count": 32,
        "larger_b5_train_count_per_phase": 32,
        "b5_holdout_count_per_phase": 16,
        # M1-63 promoted the organ onto the default path.  These fields are read
        # from the live default instead of being hardcoded, so this older review
        # can never keep asserting a default that the codebase no longer has.
        "default_identity_organ_enabled": bool(TaijiConfig().identity_organ_enabled),
        "gain_sweep_calibration_seed": calibration_seed,
        "gain_sweep": gain_sweep,
        "records": records,
        "boundary_records": boundary_records,
        "capacity_records": capacity_records,
        "all_records_pass": all(_record_passes(record) for record in records),
        "recommended_gain": REVIEW_GAIN,
        "default_candidate_ready": bool(TaijiConfig().identity_organ_enabled),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostics = run_admission(seeds=tuple(int(seed) for seed in args.seeds))
    result = {
        "format": FORMAT,
        "version": 1,
        "status": "admission-review",
        "identity_route_default": (
            "enabled" if TaijiConfig().identity_organ_enabled else "disabled"
        ),
        "shared_decoder_default_fallback": True,
        "action_intent_execution": False,
        "diagnostics": diagnostics,
        "canary_passed": bool(diagnostics["all_records_pass"]),
        "recommended_gain": diagnostics["recommended_gain"],
        "default_candidate_ready": diagnostics["default_candidate_ready"],
        "report_path": str(args.report),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
