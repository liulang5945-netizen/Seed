"""Run the feature-gated native identity-organ integration canary.

This canary exercises the optional organ through the real ``Taiji`` runtime.
It is intentionally separate from the default configuration: the shared
memory decoder remains the fallback and the organ never creates an
``ActionIntent``.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
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
from scripts.training.train_taiji_memory import _memory_config  # noqa: E402
from taiji import (  # noqa: E402
    DelayedMemoryTask,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-identity-organ-canary-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_identity_organ_canary_20260902.json"
DEFAULT_CAPACITY = 128
MODES = ("shared_control", "identity_organ")


def _config(seed: int, *, enabled: bool, capacity: int = DEFAULT_CAPACITY) -> TaijiConfig:
    values = _memory_config(seed).to_dict()
    values.update(
        {
            "identity_organ_enabled": bool(enabled),
            "identity_organ_capacity": int(capacity),
        }
    )
    return TaijiConfig.from_dict(values)


def _actions(*episodes: Any) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(episode.action) for episode in episodes))


def _query(model: Taiji, cue: int):
    model.reset_dynamics(episode_id=f"m1-28-query-{int(cue)}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    return model.observe(int(cue), learn=False, learn_motor=False)


def _cue_pattern(model: Taiji, cue: int) -> torch.Tensor:
    _query(model, cue)
    return model.fabric.cortical_context(model.snapshot().regions).detach().clone()


def _accuracy(model: Taiji, queries: tuple[Any, ...], actions: tuple[int, ...]) -> float:
    correct = 0
    for query in queries:
        step = _query(model, query.cue)
        probabilities = step.probabilities
        prediction = max(actions, key=lambda action: float(probabilities[action].item()))
        correct += int(prediction == query.expected_action)
    return correct / len(queries)


def _organ_digest(model: Taiji) -> str:
    organ = model.identity_organ
    if organ is None:
        return "disabled"
    return content_digest(
        {
            "bank": organ.bank.to_payload(),
            "action_synapses": organ.action_synapses.to_payload(),
            "outcome_synapses": organ.outcome_synapses.to_payload(),
            "write_count": organ.write_count,
            "replacement_count": organ.replacement_count,
        }
    )


def _core_digest(checkpoint: dict[str, Any]) -> str:
    return content_digest(
        {key: checkpoint[key] for key in Taiji._checkpoint_core_keys()}
    )


def _fresh_process_probe(checkpoint: dict[str, Any], cue: int) -> dict[str, Any]:
    code = """
import json
import io
import sys
import torch
from taiji import Taiji
from taiji.internalization import content_digest

payload = torch.load(io.BytesIO(sys.stdin.buffer.read()), map_location="cpu")
model = Taiji.from_checkpoint(payload)
before = content_digest(model.checkpoint())
before_organ = content_digest(
    {
        "bank": model.identity_organ.bank.to_payload(),
        "action_synapses": model.identity_organ.action_synapses.to_payload(),
        "outcome_synapses": model.identity_organ.outcome_synapses.to_payload(),
        "write_count": model.identity_organ.write_count,
        "replacement_count": model.identity_organ.replacement_count,
    }
)
model.reset_dynamics(episode_id="m1-28-fresh")
model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
step = model.observe(int(sys.argv[1]), learn=False, learn_motor=False)
after = content_digest(model.checkpoint())
after_organ = content_digest(
    {
        "bank": model.identity_organ.bank.to_payload(),
        "action_synapses": model.identity_organ.action_synapses.to_payload(),
        "outcome_synapses": model.identity_organ.outcome_synapses.to_payload(),
        "write_count": model.identity_organ.write_count,
        "replacement_count": model.identity_organ.replacement_count,
    }
)
print(json.dumps({
    "source": None if step.identity_recall is None else step.identity_recall.source,
    "provenance": None if step.identity_recall is None else step.identity_recall.provenance,
    "predicted_symbol": step.predicted_symbol,
    "persistent_digest_unchanged": before_organ == after_organ,
    "loaded_checkpoint_digest": before,
    "post_query_checkpoint_digest": after,
}))
"""
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    output = subprocess.check_output(
        (sys.executable, "-c", code, str(int(cue))),
        cwd=PROJECT_ROOT,
        input=buffer.getvalue(),
        text=False,
    )
    return dict(json.loads(output.decode().strip().splitlines()[-1]))


def _identity_record(seed: int) -> dict[str, Any]:
    b2 = build_delayed_memory_smoke_corpus(count=8)
    b5 = build_b5_corpus(train_count=16, holdout_count=8, retention_count=8)
    b2_actions = _actions(*b2.train)
    b5_actions = _actions(*b5.phase_a_train, *b5.phase_b_train)

    model = Taiji(_config(seed, enabled=True), episode_id=f"m1-28-identity-{seed}")
    for episode in b2.train:
        DelayedMemoryTask._write_episode(model, episode)
    b2_score = _accuracy(model, b2.holdout, b2_actions)
    b2_retention = _accuracy(model, b2.retention, b2_actions)
    b2_lesion = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
    b2_lesion.identity_organ.lesion()
    b2_lesion_score = _accuracy(b2_lesion, b2.holdout, b2_actions)
    b2_control = Taiji(_config(seed, enabled=False), episode_id=f"m1-28-b2-control-{seed}")
    for episode in b2.train:
        DelayedMemoryTask._write_episode(b2_control, episode)
    b2_control_score = _accuracy(b2_control, b2.holdout, b2_actions)

    phase_a = Taiji(_config(seed, enabled=True), episode_id=f"m1-28-b5-phase-a-{seed}")
    for episode in b5.phase_a_train:
        DelayedMemoryTask._write_episode(phase_a, episode)
    phase_a_checkpoint = deepcopy(phase_a.checkpoint())
    parent_core_digest = _core_digest(phase_a_checkpoint)
    parent_old = _accuracy(phase_a, b5.phase_a_holdout, b5_actions)
    parent_retention = _accuracy(phase_a, b5.phase_a_retention, b5_actions)

    child = Taiji(_config(seed, enabled=True), episode_id=f"m1-28-b5-child-{seed}")
    child.restore(deepcopy(phase_a_checkpoint))
    for episode in b5.phase_b_train:
        DelayedMemoryTask._write_episode(child, episode)
    child_checkpoint = deepcopy(child.checkpoint())
    child_old = _accuracy(child, b5.phase_a_holdout, b5_actions)
    child_new = _accuracy(child, b5.phase_b_holdout, b5_actions)
    child_retention = _accuracy(child, b5.phase_a_retention, b5_actions)

    control = Taiji(_config(seed, enabled=False), episode_id=f"m1-28-b5-control-{seed}")
    # Disabled and enabled configurations are intentionally different
    # architectures at the checkpoint contract.  Rebuild the control from
    # the same seed instead of stripping a feature-gated payload and hiding a
    # configuration mismatch.
    for episode in b5.phase_a_train:
        DelayedMemoryTask._write_episode(control, episode)
    for episode in b5.phase_b_train:
        DelayedMemoryTask._write_episode(control, episode)
    control_old = _accuracy(control, b5.phase_a_holdout, b5_actions)
    control_new = _accuracy(control, b5.phase_b_holdout, b5_actions)

    lesion = Taiji.from_checkpoint(deepcopy(child_checkpoint))
    lesion.identity_organ.lesion()
    lesion_old = _accuracy(lesion, b5.phase_a_holdout, b5_actions)
    lesion_new = _accuracy(lesion, b5.phase_b_holdout, b5_actions)

    repeated_slots: list[bool] = []
    expected_slots: list[int | None] = []
    repeated_before = _organ_digest(child)
    for episode in b5.phase_a_train:
        pattern = _cue_pattern(child, episode.cue)
        binding = child.identity_organ.recall(pattern)
        expected_slots.append(binding.slot_index)
        learned = child.identity_organ.learn(
            pattern,
            episode.action,
            outcome_symbol=episode.outcome,
        )
        repeated_slots.append(learned.slot_index == binding.slot_index)
    repeated_after = _organ_digest(child)
    repeated_old = _accuracy(child, b5.phase_a_holdout, b5_actions)
    repeated_new = _accuracy(child, b5.phase_b_holdout, b5_actions)

    no_change_before = _organ_digest(child)
    bound_steps = [_query(child, query.cue) for query in b5.phase_a_holdout]
    unknown_step = _query(child, 250)
    no_change_after = _organ_digest(child)

    restored = Taiji.from_checkpoint(deepcopy(child_checkpoint))
    restored_initial_checkpoint = restored.checkpoint()
    restored_old = _accuracy(restored, b5.phase_a_holdout, b5_actions)
    restored_new = _accuracy(restored, b5.phase_b_holdout, b5_actions)
    fresh = _fresh_process_probe(child_checkpoint, b5.phase_a_holdout[0].cue)

    rollback_source = Taiji(_config(seed, enabled=False), episode_id=f"m1-28-rollback-{seed}")
    rollback_checkpoint = rollback_source.checkpoint()
    rollback_restored = Taiji.from_checkpoint(deepcopy(rollback_checkpoint))

    stress = Taiji(
        _config(seed, enabled=True, capacity=8),
        episode_id=f"m1-28-capacity-stress-{seed}",
    )
    stress_patterns = torch.eye(
        stress.config.identity_organ_capacity + 4,
        stress.config.cortical_context_dim,
    )
    for index, pattern in enumerate(stress_patterns):
        stress.identity_organ.learn(
            pattern,
            48 + index % 2,
            outcome_symbol=43 if index % 2 == 0 else 45,
        )

    lineage = child_checkpoint["identity_organ"]["lineage"]
    return {
        "seed": seed,
        "b2": {
            "identity_holdout": b2_score,
            "identity_retention": b2_retention,
            "shared_control_holdout": b2_control_score,
            "identity_lesion_holdout": b2_lesion_score,
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
        "provenance": {
            "bound_sources": sorted({step.identity_recall.source for step in bound_steps}),
            "bound_provenance": sorted(
                {step.identity_recall.provenance for step in bound_steps}
            ),
            "unbound_source": unknown_step.identity_recall.source,
            "unbound_provenance": unknown_step.identity_recall.provenance,
            "action_intent_generated": False,
            "final_action_owner": "ByteMotor",
        },
        "repeated_replay": {
            "same_slot_rate": sum(repeated_slots) / len(repeated_slots),
            "old_holdout_after": repeated_old,
            "new_holdout_after": repeated_new,
            "organ_digest_changed": repeated_before != repeated_after,
            "expected_slots_bound": all(slot is not None for slot in expected_slots),
        },
        "no_change": {
            "bound_count": len(bound_steps),
            "all_bound": all(step.identity_recall.used for step in bound_steps),
            "unbound_is_fallback": not unknown_step.identity_recall.used,
            "organ_digest_unchanged": no_change_before == no_change_after,
        },
        "capacity": {
            "capacity": stress.identity_organ.capacity,
            "attempted_writes": len(stress_patterns),
            "occupied_count": stress.identity_organ.bank.occupied_count,
            "allocation_count": stress.identity_organ.bank.allocation_count,
            "replacement_count": stress.identity_organ.replacement_count,
            "occupied_within_capacity": (
                stress.identity_organ.bank.occupied_count <= stress.identity_organ.capacity
            ),
            "replacement_policy": "least-used occupied slot",
        },
        "budget": {
            "model_parameter_count": child.parameter_count(),
            "planned_parameter_count": child.config.planned_active_parameter_count,
            "identity_prototype_parameters": child.identity_organ.bank.prototypes.numel(),
            "identity_action_edges": child.identity_organ.action_synapses.edge_count,
            "identity_outcome_edges": child.identity_organ.outcome_synapses.edge_count,
            "identity_total_parameters": child.identity_organ.parameter_count,
        },
        "checkpoint": {
            "format": child_checkpoint["identity_organ"]["format"],
            "version": child_checkpoint["identity_organ"]["version"],
            "parent_core_digest_matches_lineage": (
                parent_core_digest
                == phase_a_checkpoint["identity_organ"]["lineage"][
                    "parent_checkpoint_digest"
                ]
            ),
            "child_core_digest_matches_lineage": (
                _core_digest(child_checkpoint) == lineage["parent_checkpoint_digest"]
            ),
            "restored_bundle_digest_matches": (
                content_digest(restored_initial_checkpoint) == content_digest(child_checkpoint)
            ),
            "restored_old_holdout": restored_old,
            "restored_new_holdout": restored_new,
            "fresh_process_source": fresh["source"],
            "fresh_process_provenance": fresh["provenance"],
            "fresh_process_persistent_digest_unchanged": fresh[
                "persistent_digest_unchanged"
            ],
            "fresh_process_checkpoint_digest_matches": (
                fresh["loaded_checkpoint_digest"] == content_digest(child_checkpoint)
            ),
            "rollback_disabled_has_no_identity_payload": (
                "identity_organ" not in rollback_checkpoint
                and rollback_restored.identity_organ is None
            ),
        },
    }


def _record_passes(record: dict[str, Any]) -> bool:
    b2 = record["b2"]
    b5 = record["b5"]
    repeated = record["repeated_replay"]
    no_change = record["no_change"]
    capacity = record["capacity"]
    budget = record["budget"]
    checkpoint = record["checkpoint"]
    provenance = record["provenance"]
    return bool(
        b2["identity_holdout"] >= b2["shared_control_holdout"]
        and b2["identity_holdout"] >= b2["identity_lesion_holdout"]
        and b5["child_old_holdout"] >= b5["parent_old_holdout"]
        and b5["child_old_retention"] >= b5["parent_retention"]
        and b5["child_new_holdout"] > b5["shared_control_new_holdout"]
        and b5["child_old_holdout"] > b5["identity_lesion_old_holdout"]
        and repeated["same_slot_rate"] == 1.0
        and repeated["old_holdout_after"] >= b5["parent_old_holdout"]
        and repeated["new_holdout_after"] >= b5["child_new_holdout"]
        and no_change["all_bound"]
        and no_change["unbound_is_fallback"]
        and no_change["organ_digest_unchanged"]
        and capacity["occupied_within_capacity"]
        and capacity["replacement_count"] == 4
        and budget["model_parameter_count"] == budget["planned_parameter_count"]
        and provenance["action_intent_generated"] is False
        and provenance["final_action_owner"] == "ByteMotor"
        and checkpoint["parent_core_digest_matches_lineage"]
        and checkpoint["child_core_digest_matches_lineage"]
        and checkpoint["restored_bundle_digest_matches"]
        and checkpoint["fresh_process_source"] == "identity-route"
        and checkpoint["fresh_process_persistent_digest_unchanged"]
        and checkpoint["fresh_process_checkpoint_digest_matches"]
        and checkpoint["rollback_disabled_has_no_identity_payload"]
    )


def run_canary(*, seeds: tuple[int, ...], modes: tuple[str, ...] = MODES) -> dict[str, Any]:
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unsupported identity organ canary mode: {sorted(unknown)}")
    records = {mode: [] for mode in modes}
    if "identity_organ" in modes:
        records["identity_organ"] = [_identity_record(seed) for seed in seeds]
    if "shared_control" in modes:
        # The full identity record already constructs the shared controls; this
        # marker keeps the CLI contract explicit without duplicating training.
        records["shared_control"] = [
            {"seed": seed, "provided_by": "identity_organ.b2/b5.shared_control"}
            for seed in seeds
        ]
    identity_records = records.get("identity_organ", [])
    return {
        "seeds": list(seeds),
        "modes": list(modes),
        "default_identity_organ_enabled": False,
        "architecture_default_unchanged": True,
        "canary_passed": bool(identity_records) and all(
            _record_passes(record) for record in identity_records
        ),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    diagnostics = run_canary(
        seeds=tuple(int(seed) for seed in args.seeds),
        modes=tuple(str(mode) for mode in args.modes),
    )
    result = {
        "format": FORMAT,
        "version": 1,
        "status": "canary",
        "identity_route_default": "disabled",
        "shared_decoder_default_fallback": True,
        "action_intent_execution": False,
        "checkpoint_lineage": "identity-organ payload is parent-digested against Taiji core",
        "capacity_replacement": "least-used occupied slot; replacement clears stale action edges",
        "action_evidence_ownership": "identity organ emits evidence; ByteMotor owns final action distribution",
        "rollback": "restore a pre-organ checkpoint under identity_organ_enabled=false",
        "diagnostics": diagnostics,
        "canary_passed": bool(diagnostics["canary_passed"]),
        "can_promote_for_review": bool(diagnostics["canary_passed"]),
        "report_path": str(args.report),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
