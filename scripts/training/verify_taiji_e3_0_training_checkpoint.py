"""E3-0 Gate: prove training checkpoints are writable before learning."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.training_checkpoint import NativeTrainingCheckpoint  # noqa: E402
from taiji import Taiji, TaijiConfig  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact, EvolutionExperience  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _config() -> TaijiConfig:
    return TaijiConfig(
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
        seed=19,
    )


def _ledger() -> EvolutionExperienceLedger:
    ledger = EvolutionExperienceLedger()
    corpus = EvolutionCorpusArtifact(
        corpus_id="fixture:e3-training",
        source_kind="skill_artifact",
        source_id="skill.e3.training",
        source_version="1",
        source_digest=content_digest({"source": "skill.e3.training"}),
        unit_kind="knowledge",
        content={"title": "e3 training fixture"},
    )
    ledger.add_corpus(corpus)
    ledger.admit_corpus(corpus.artifact_digest, admission_revision="e3-admission")
    ledger.append(
        EvolutionExperience(
            experience_id="e3-training-1",
            source_kind="skill",
            source_id="skill.e3.training",
            source_version="1",
            source_digest=corpus.source_digest,
            parent_checkpoint_digest="a" * 64,
            partition="train",
            status="success",
            success=True,
            episode_id="e3-episode-1",
            tick=1,
            result_digest=content_digest({"status": "ok"}),
            reward_components={"quality": 1.0},
        )
    )
    return ledger


def run_gate(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Taiji(_config())
    ledger = _ledger()
    parent = NativeTrainingCheckpoint.create(
        model,
        ledger,
        checkpoint_kind="parent",
        checkpoint_id="e3-parent",
        learner_state={"updates": 0, "phase": "preflight"},
        resource_ledger={"cpu_ms": 0.0},
    )
    parent_path = parent.save(output_dir / "parent.pt")
    loaded_parent = NativeTrainingCheckpoint.load(parent_path)
    restored_model = Taiji(_config(), episode_id="restored")
    restored_ledger = loaded_parent.restore_into(restored_model, ledger)

    trial = NativeTrainingCheckpoint.create(
        restored_model,
        restored_ledger,
        checkpoint_kind="trial",
        checkpoint_id="e3-trial",
        parent_checkpoint_digest=loaded_parent.checkpoint_digest,
        learner_state={"updates": 0, "phase": "trial"},
        resource_ledger={"cpu_ms": 1.0},
    )
    trial_path = trial.save(output_dir / "trial.pt")
    loaded_trial = NativeTrainingCheckpoint.load(trial_path)
    admitted = NativeTrainingCheckpoint.create(
        restored_model,
        restored_ledger,
        checkpoint_kind="admitted",
        checkpoint_id="e3-admitted",
        parent_checkpoint_digest=loaded_trial.checkpoint_digest,
        learner_state={"updates": 0, "phase": "admitted-preflight"},
        resource_ledger={"cpu_ms": 1.0},
    )
    admitted_path = admitted.save(output_dir / "admitted.pt")
    loaded_admitted = NativeTrainingCheckpoint.load(admitted_path)

    tampered = deepcopy(loaded_admitted.to_payload())
    tampered["learner_state"]["updates"] = 1
    try:
        NativeTrainingCheckpoint.from_payload(tampered)
    except ValueError as exc:
        tamper_rejected = "checkpoint digest mismatch" in str(exc)
    else:  # pragma: no cover - the Gate must fail if mutation is accepted
        tamper_rejected = False

    drifted = EvolutionExperienceLedger.from_checkpoint(restored_ledger.checkpoint())
    drifted.append(
        EvolutionExperience(
            experience_id="e3-training-2",
            source_kind="skill",
            source_id="skill.e3.training",
            source_version="1",
            source_digest="b" * 64,
            parent_checkpoint_digest="a" * 64,
            partition="train",
            status="success",
            success=True,
            episode_id="e3-episode-2",
            tick=2,
            result_digest=content_digest({"status": "ok-2"}),
        )
    )
    before_drift_restore = content_digest(restored_model.checkpoint())
    try:
        loaded_admitted.restore_into(restored_model, drifted)
    except ValueError as exc:
        drift_rejected = "ledger drift" in str(exc)
    else:  # pragma: no cover - the Gate must fail if drift is accepted
        drift_rejected = False

    checks = {
        "parent_written_and_loaded": parent_path.is_file()
        and loaded_parent.checkpoint_digest == parent.checkpoint_digest,
        "trial_written_and_lineage_bound": trial_path.is_file()
        and loaded_trial.parent_checkpoint_digest == loaded_parent.checkpoint_digest,
        "admitted_written_and_lineage_bound": admitted_path.is_file()
        and loaded_admitted.parent_checkpoint_digest == loaded_trial.checkpoint_digest,
        "native_model_roundtrip": content_digest(restored_model.checkpoint()) == parent.model_digest,
        "ledger_cursor_roundtrip": loaded_parent.ledger_cursor == parent.ledger_cursor,
        "partition_manifest_roundtrip": loaded_parent.partition_manifest == parent.partition_manifest,
        "tamper_rejected": tamper_rejected,
        "dataset_drift_rejected_before_model_restore": drift_rejected
        and content_digest(restored_model.checkpoint()) == before_drift_restore,
    }
    return {
        "gate": "taiji-e3-0-training-checkpoint",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "checkpoint_paths": [str(parent_path), str(trial_path), str(admitted_path)],
        "parent_digest": parent.checkpoint_digest,
        "trial_digest": trial.checkpoint_digest,
        "admitted_digest": admitted.checkpoint_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-e3-0-checkpoints",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e3_0_training_checkpoint_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate(args.output_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
