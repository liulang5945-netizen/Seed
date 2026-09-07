"""Evaluate runtime ownership and checkpoint boundaries for semantic transitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r3_r3_multistep import (  # noqa: E402
    _event,
    _world,
    build_corpus,
)
from taiji import (  # noqa: E402
    StructuredSemanticTransitionLearner,
    TSKV8Adapter,
    WorldState,
    content_digest,
    semantic_fact_key,
)

REPORT_FORMAT = "taiji-m2r3-r4-runtime-transition-owner-v1"
REPORT_VERSION = 1


def _sequence(
    adapter: TSKV8Adapter,
    seed: int,
    initial: WorldState,
) -> tuple[tuple[dict[str, Any], ...], WorldState]:
    current = initial
    results: list[dict[str, Any]] = []
    for tick, kind in enumerate(("release", "move", "block"), 1):
        result = adapter.infer_structured_transition(
            current,
            _event(seed, kind, tick),
        )
        results.append(result.to_payload())
        if result.world is None:
            break
        current = result.world
    return tuple(results), current


def _fact_keys(world: WorldState) -> set[str]:
    return {semantic_fact_key(*relation) for relation in world.relations}


def _run_seed(seed: int) -> dict[str, Any]:
    corpus = build_corpus(seed)
    learner = StructuredSemanticTransitionLearner(corpus)
    learner.fit(corpus.train, epochs=240, learning_rate=0.5)
    adapter = TSKV8Adapter()
    plain_checkpoint = adapter.native_checkpoint()
    before_state = content_digest(adapter.cognitive_snapshot().to_payload())
    adapter.attach_structured_semantic_transition_learner(learner)
    attached_checkpoint = adapter.native_checkpoint()
    initial = _world("alpha", "pending", "cache", 0)
    direct_results, direct_final = _sequence(adapter, seed, initial)
    after_state = content_digest(adapter.cognitive_snapshot().to_payload())
    restored = TSKV8Adapter.from_native_checkpoint(attached_checkpoint)
    restored_checkpoint = restored.native_checkpoint()
    restored_results, restored_final = _sequence(restored, seed, initial)

    lesion = TSKV8Adapter.from_native_checkpoint(attached_checkpoint)
    lesion_learner = lesion.structured_semantic_transition_learner
    if lesion_learner is None:
        raise AssertionError("transition owner was not restored for lesion")
    lesion_before, lesion_after = lesion_learner.zero_transition_head()
    lesion_result = lesion.infer_structured_transition(initial, _event(seed, "release", 1))

    detached = TSKV8Adapter.from_native_checkpoint(attached_checkpoint)
    detached.attach_structured_semantic_transition_learner(None)
    expected_final = {
        semantic_fact_key("agent", "tracks", "alpha"),
        semantic_fact_key("agent", "state", "blocked"),
        semantic_fact_key("agent", "holds", "queue"),
    }
    checks = {
        "plain_owner_optional": (
            "structured_semantic_transition" not in plain_checkpoint["components"]
            and TSKV8Adapter.from_native_checkpoint(
                plain_checkpoint
            ).structured_semantic_transition_learner
            is None
        ),
        "attached_component_present": (
            "structured_semantic_transition" in attached_checkpoint["components"]
        ),
        "sequence_persistence": (
            len(direct_results) == 3 and _fact_keys(direct_final) == expected_final
        ),
        "checkpoint_roundtrip": (
            len(direct_results) == len(restored_results)
            and content_digest(direct_results) == content_digest(restored_results)
            and content_digest(direct_final.to_payload())
            == content_digest(restored_final.to_payload())
            and restored.last_structured_semantic_transition_result is not None
            and content_digest(restored.last_structured_semantic_transition_result.to_payload())
            == content_digest(restored_results[-1])
        ),
        "source_digest_preserved": (
            restored.structured_semantic_transition_learner is not None
            and restored.structured_semantic_transition_learner.source_digest
            == learner.source_digest
        ),
        "cognitive_state_read_only": before_state == after_state,
        "no_execution_side_effect": (
            adapter.last_task_interpretation is None
            and adapter.last_task_decomposition is None
            and adapter.last_semantic_provider_evidence is None
        ),
        "transition_lesion_effective": (
            lesion_before != lesion_after
            and lesion_result.world is not None
            and _fact_keys(lesion_result.world) != {
                semantic_fact_key("agent", "tracks", "alpha"),
                semantic_fact_key("agent", "state", "ready"),
                semantic_fact_key("agent", "holds", "cache"),
            }
        ),
        "detach_removes_component": (
            detached.structured_semantic_transition_learner is None
            and "structured_semantic_transition" not in detached.native_checkpoint()["components"]
        ),
        "restored_checkpoint_stable": (
            content_digest(attached_checkpoint) == content_digest(restored_checkpoint)
        ),
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "seed": seed,
        "status": "passed" if passed else "failed",
        "training": {"epochs": 240, "learning_rate": 0.5},
        "checkpoint": {
            "transition_format": attached_checkpoint["components"][
                "structured_semantic_transition"
            ]["learner"]["format"],
            "parameter_count": learner.parameter_count,
            "component_present": "structured_semantic_transition"
            in attached_checkpoint["components"],
        },
        "sequence": {
            "steps": len(direct_results),
            "final_world_digest": content_digest(direct_final.to_payload()),
        },
        "gate": {"passed": passed, "checks": checks},
    }


def evaluate(seeds: tuple[int, ...] = (11, 29, 47)) -> dict[str, Any]:
    runs = tuple(_run_seed(int(seed)) for seed in seeds)
    passed = all(run["status"] == "passed" for run in runs)
    return {
        "format": REPORT_FORMAT,
        "version": REPORT_VERSION,
        "status": "passed" if passed else "failed",
        "can_promote": False,
        "seeds": list(seeds),
        "runs": list(runs),
        "gate": {"passed": passed, "criterion": "every runtime owner seed passes every check"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m2r3_r4_runtime_transition_owner_20260907.json",
    )
    args = parser.parse_args()
    report = evaluate(tuple(args.seeds))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
