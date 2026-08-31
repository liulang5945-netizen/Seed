from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.evolution_experience import EvolutionExperience  # noqa: E402
from taiji.evolution_training import NativeEvolutionTrainer  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _experience(
    experience_id: str,
    *,
    partition: str,
    capability_id: str,
    reward: float,
    success: bool = True,
) -> EvolutionExperience:
    return EvolutionExperience(
        experience_id=experience_id,
        source_kind="workbench",
        source_id="seed.workbench.route",
        source_version="1",
        source_digest=content_digest({"source": "seed.workbench.route", "version": "1"}),
        parent_checkpoint_digest="a" * 64,
        partition=partition,
        status="success" if success else "error",
        success=success,
        capability_id=capability_id,
        capability_snapshot_id="b" * 64,
        reward_components={"quality": reward},
        result_digest=content_digest({"result": experience_id}),
    )


def run_gate() -> dict[str, object]:
    trainer = NativeEvolutionTrainer(feature_dim=64)
    train = (
        _experience("train-editor-1", partition="train", capability_id="editor.open", reward=1.0),
        _experience("train-editor-2", partition="train", capability_id="editor.open", reward=1.0),
    )
    holdout = (
        _experience("holdout-editor", partition="holdout", capability_id="editor.open", reward=1.0),
    )
    retention = (
        _experience("retention-mcp", partition="retention", capability_id="mcp.list", reward=0.0),
    )
    report = trainer.consolidate(
        train,
        holdout_experiences=holdout,
        retention_experiences=retention,
    )
    restored = NativeEvolutionTrainer.from_checkpoint(trainer.checkpoint())
    holdout_example = restored.example(holdout[0])
    checks = {
        "native_update_admitted": report.admitted,
        "holdout_improves_over_frozen": report.native_holdout_loss < report.frozen_holdout_loss,
        "replay_only_matches_frozen": abs(report.replay_only_holdout_loss - report.frozen_holdout_loss) < 1e-12,
        "retention_within_gate": report.native_retention_loss_after <= report.native_retention_loss_before + 0.05,
        "holdout_not_consumed": "holdout-editor" not in trainer.consumed_experience_ids,
        "train_cursor_consumed": trainer.consumed_experience_ids == ("train-editor-1", "train-editor-2"),
        "checkpoint_roundtrip": content_digest(restored.checkpoint())
        == content_digest(trainer.checkpoint()),
        "restored_score_matches": abs(
            restored.learner.score(holdout_example) - trainer.learner.score(trainer.example(holdout[0]))
        )
        < 1e-12,
    }
    return {
        "gate": "taiji-e3-1-native-route-credit",
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
        default=PROJECT_ROOT / "reports" / "taiji_w7_e3_1_native_route_credit_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
