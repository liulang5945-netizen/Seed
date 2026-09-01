"""M0-0: prove a Taiji training checkpoint survives a fresh process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.training_checkpoint import NativeTrainingCheckpoint  # noqa: E402
from taiji import Taiji, TaijiConfig  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact, EvolutionExperience  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _config() -> TaijiConfig:
    """Use a small deterministic model; M0 validates the envelope, not scale."""

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
        seed=23,
    )


def _ledger() -> EvolutionExperienceLedger:
    ledger = EvolutionExperienceLedger()
    corpus = EvolutionCorpusArtifact(
        corpus_id="fixture:m0-checkpoint",
        source_kind="skill_artifact",
        source_id="skill.m0.checkpoint",
        source_version="1",
        source_digest=content_digest({"source": "skill.m0.checkpoint"}),
        unit_kind="knowledge",
        content={"title": "m0 checkpoint fixture"},
    )
    ledger.add_corpus(corpus)
    ledger.admit_corpus(corpus.artifact_digest, admission_revision="m0-checkpoint-admission")
    ledger.append(
        EvolutionExperience(
            experience_id="m0-checkpoint-episode",
            source_kind="skill",
            source_id="skill.m0.checkpoint",
            source_version="1",
            source_digest=corpus.source_digest,
            parent_checkpoint_digest="a" * 64,
            partition="train",
            status="success",
            success=True,
            episode_id="m0-checkpoint-episode",
            tick=1,
            result_digest=content_digest({"status": "ok"}),
            reward_components={"quality": 1.0},
        )
    )
    return ledger


def _run_fresh_process(parent_path: Path, symbol: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(parent_path),
            "--symbol",
            str(int(symbol)),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _child(parent_path: Path, symbol: int) -> int:
    record = NativeTrainingCheckpoint.load(parent_path)
    config = TaijiConfig.from_dict(dict(record.model_checkpoint["config"]))
    model = Taiji(config, episode_id="m0-fresh-child")
    ledger = EvolutionExperienceLedger.from_checkpoint(record.ledger_checkpoint)
    record.restore_into(model, ledger)

    step = model.observe(int(symbol), learn=True, learn_motor=False)
    child = NativeTrainingCheckpoint.create(
        model,
        ledger,
        checkpoint_kind="trial",
        checkpoint_id="m0-fresh-child",
        parent_checkpoint_digest=record.checkpoint_digest,
        learner_state={"updates": 1, "phase": "fresh-process-continuation"},
        resource_ledger={"fresh_process": 1.0},
    )
    child_path = parent_path.with_name("child-after-fresh-restore.pt")
    child.save(child_path)
    print(
        json.dumps(
            {
                "predicted_symbol": int(step.predicted_symbol),
                "model_digest": content_digest(model.checkpoint()),
                "child_checkpoint_digest": child.checkpoint_digest,
                "child_path": str(child_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


def run_gate(output_dir: Path, report_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model = Taiji(_config(), episode_id="m0-parent")
    ledger = _ledger()
    parent = NativeTrainingCheckpoint.create(
        model,
        ledger,
        checkpoint_kind="parent",
        checkpoint_id="m0-parent",
        learner_state={"updates": 0, "phase": "m0-preflight"},
        resource_ledger={"fresh_process": 0.0},
    )
    parent_path = parent.save(output_dir / "parent.pt")

    expected_model = Taiji(_config(), episode_id="m0-expected-child")
    expected_ledger = EvolutionExperienceLedger.from_checkpoint(parent.ledger_checkpoint)
    parent.restore_into(expected_model, expected_ledger)
    expected_step = expected_model.observe(1, learn=True, learn_motor=False)
    expected_model_digest = content_digest(expected_model.checkpoint())

    process = _run_fresh_process(parent_path, 1)
    child_payload: dict[str, Any] = {}
    if process.returncode == 0:
        try:
            child_payload = json.loads(process.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            child_payload = {}

    child_path = Path(str(child_payload.get("child_path", "")))
    checks = {
        "parent_saved": parent_path.is_file(),
        "child_saved_after_fresh_restore": child_path.is_file(),
        "fresh_process_completed": process.returncode == 0,
        "fresh_process_next_step_matches": (
            int(child_payload.get("predicted_symbol", -1)) == int(expected_step.predicted_symbol)
        ),
        "fresh_process_checkpoint_matches": (
            child_payload.get("model_digest") == expected_model_digest
        ),
        "report_written": False,
    }
    result: dict[str, Any] = {
        "gate": "taiji-m0-checkpoint-preflight",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "parent_checkpoint": {
            "path": str(parent_path),
            "digest": parent.checkpoint_digest,
        },
        "child_checkpoint": child_payload,
        "fresh_process": {
            "returncode": process.returncode,
            "stderr": process.stderr[-2000:],
        },
    }
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks["report_written"] = report_path.is_file()
    result["status"] = "passed" if all(checks.values()) else "failed"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", type=Path)
    parser.add_argument("--symbol", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "taiji-m0-checkpoint-preflight",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m0_checkpoint_preflight_20260901.json",
    )
    args = parser.parse_args()
    if args.child is not None:
        return _child(args.child, args.symbol)
    result = run_gate(args.output_dir, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
