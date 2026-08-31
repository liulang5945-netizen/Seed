from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.evolution_experience import EvolutionExperience  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.procedural_evolution import NativeProceduralMemoryTrainer  # noqa: E402


def _experience(
    experience_id: str,
    *,
    partition: str,
    capability_id: str,
    success: bool = True,
) -> EvolutionExperience:
    return EvolutionExperience(
        experience_id=experience_id,
        source_kind="workbench",
        source_id="seed.workbench.procedure",
        source_version="1",
        source_digest=content_digest({"source": "seed.workbench.procedure", "version": "1"}),
        parent_checkpoint_digest="a" * 64,
        partition=partition,
        status="success" if success else "error",
        success=success,
        capability_id=capability_id,
        capability_snapshot_id="b" * 64,
        episode_id=f"episode-{experience_id}",
        tick=1,
        result_digest=content_digest({"result": experience_id}),
    )


def run_gate() -> dict[str, object]:
    trainer = NativeProceduralMemoryTrainer(cue_dim=64, epochs=120)
    train = (
        _experience("train-editor", partition="train", capability_id="editor.open"),
        _experience("train-mcp", partition="train", capability_id="mcp.list"),
        _experience("train-failed", partition="train", capability_id="terminal.run", success=False),
    )
    holdout = (_experience("holdout-mcp", partition="holdout", capability_id="mcp.list"),)
    retention = (_experience("retention-editor", partition="retention", capability_id="editor.open"),)
    report = trainer.consolidate(
        train,
        holdout_experiences=holdout,
        retention_experiences=retention,
    )
    restored = NativeProceduralMemoryTrainer.from_checkpoint(trainer.checkpoint())
    checks = {
        "native_update_admitted": report.admitted,
        "holdout_improves_over_frozen": report.native_holdout_accuracy > report.frozen_holdout_accuracy,
        "replay_only_matches_frozen": report.replay_only_holdout_accuracy == report.frozen_holdout_accuracy,
        "retention_preserved": report.native_retention_accuracy >= report.frozen_retention_accuracy,
        "failed_train_excluded": report.excluded_experience_ids == ("train-failed",),
        "failed_train_not_consumed": "train-failed" not in trainer.consumed_experience_ids,
        "checkpoint_roundtrip": content_digest(restored.checkpoint()) == content_digest(trainer.checkpoint()),
        "dynamic_action_discovery": trainer.learner.action_kinds == ("editor.open", "mcp.list"),
    }
    return {
        "gate": "taiji-e3-2-procedural-memory",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "report": report.to_payload(),
        "trainer_revision": trainer.revision,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e3_2_procedural_memory_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
