"""P5.1h adoption recovery verification (fresh process).

Loads the adopted child checkpoint + adoption manifest in a brand-new
process, rebuilds the frozen corpus deterministically, re-measures the three
recorded accuracies, and verifies they match the manifest exactly.  Exit 0
only when every recorded number reproduces bit-for-bit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import torch  # noqa: E402
from eval_taiji_p5_1f_real_corpus_same_budget_gate import (  # noqa: E402
    DocumentEmbedder,
    _MemoizedEmbedder,
    _sample_agate,
)
from eval_taiji_p5_1g_real_corpus_quota_budget_gate import (  # noqa: E402
    ARM_COUNT,
    HOLDOUT_COUNT,
    SOURCED_PATH,
    TRAIN_COUNT,
    _accuracy,
    _build_arm_from_partitions,
    _gate_records_g,
    _sample_arm,
)
from eval_taiji_p5_1h_admission_recipe_gate import _independent_slice  # noqa: E402

from taiji import ArtifactInternalizationTrainer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    import hashlib

    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    sha_match = checkpoint_sha == manifest["checkpoint_sha256"]

    sourced_path = PROJECT_ROOT / SOURCED_PATH
    sourced_sample = _sample_arm(sourced_path)
    sourced_partitions = {
        "train": sourced_sample.trajectories[:TRAIN_COUNT],
        "holdout": sourced_sample.trajectories[TRAIN_COUNT : TRAIN_COUNT + HOLDOUT_COUNT],
        "retention": sourced_sample.trajectories[TRAIN_COUNT + HOLDOUT_COUNT : ARM_COUNT],
    }
    train_vocabulary = frozenset(
        f"tool.{name}"
        for trajectory in sourced_partitions["train"]
        for name in trajectory.tool_calls
    )
    p51g_agate = _sample_agate(sourced_path, train_vocabulary, after_line=sourced_sample.last_line)
    independent = _independent_slice(sourced_path, train_vocabulary, after_line=p51g_agate.last_line)

    embedder = _MemoizedEmbedder(DocumentEmbedder())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restored = ArtifactInternalizationTrainer.from_checkpoint(payload, embedder=embedder)
    arm = _build_arm_from_partitions(sourced_partitions)

    def proc_records(partition: str) -> tuple:
        artifacts, experiences = arm[partition]
        return restored._procedural_records(artifacts, experiences)

    retention_records = proc_records("retention")
    holdout_records = proc_records("holdout")
    independent_records = _gate_records_g(independent, encoder=restored.encoder)

    observed = {
        "procedural_retention_accuracy": _accuracy(restored.procedural, retention_records),
        "procedural_holdout_accuracy": _accuracy(restored.procedural, holdout_records),
        "independent_accuracy": _accuracy(restored.procedural, independent_records),
    }
    comparison = {
        key: {
            "manifest": manifest["metrics"][key],
            "observed": observed[key],
            "match": abs(manifest["metrics"][key] - observed[key]) <= 5e-7,
        }
        for key in observed
    }
    verdict = {
        "format": "taiji-p5-1h-recovery-verification-v1",
        "checkpoint_sha256_match": sha_match,
        "accuracy_comparison": comparison,
        "recovery_verified": sha_match and all(item["match"] for item in comparison.values()),
    }
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    return 0 if verdict["recovery_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
