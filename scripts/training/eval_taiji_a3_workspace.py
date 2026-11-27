"""Build and evaluate the Taiji A3 workspace composition benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    WorkspaceCandidate,
    WorkspaceCollaborationEvaluator,
    WorkspaceCompositionSample,
)

MANIFEST_FORMAT = "taiji-a3-workspace-composition-manifest-v1"


def _sample(index: int, generator: torch.Generator) -> WorkspaceCompositionSample:
    left = float(torch.randn((), generator=generator))
    right = float(torch.randn((), generator=generator))
    source_a = torch.tensor([left, 0.0, 1.0, 1.0])
    source_b = torch.tensor([0.0, right, 1.0, 1.0])
    distractor_a = torch.cat((torch.randn(2, generator=generator) * 2.0, torch.zeros(2)))
    distractor_b = torch.cat((torch.randn(2, generator=generator) * 2.0, torch.zeros(2)))
    candidates = (
        WorkspaceCandidate(f"sample-{index}:a", source_a, source="expert-a"),
        WorkspaceCandidate(f"sample-{index}:b", source_b, source="expert-b"),
        WorkspaceCandidate(f"sample-{index}:d0", distractor_a, source="distractor"),
        WorkspaceCandidate(f"sample-{index}:d1", distractor_b, source="distractor"),
    )
    order = torch.randperm(len(candidates), generator=generator).tolist()
    return WorkspaceCompositionSample(
        candidates=tuple(candidates[position] for position in order),
        target=(source_a[:2] + source_b[:2]) / 2.0,
        relevant_ids=(f"sample-{index}:a", f"sample-{index}:b"),
        tick=index,
    )


def build_corpus(
    *,
    seed: int = 20260825,
    train_count: int = 64,
    holdout_count: int = 32,
) -> tuple[tuple[WorkspaceCompositionSample, ...], tuple[WorkspaceCompositionSample, ...]]:
    if train_count <= 0 or holdout_count <= 0:
        raise ValueError("A3 train_count and holdout_count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train = tuple(_sample(index, generator) for index in range(train_count))
    holdout = tuple(_sample(train_count + index, generator) for index in range(holdout_count))
    return train, holdout


def build_manifest(*, train_count: int = 64, holdout_count: int = 32) -> dict[str, object]:
    return {
        "format": MANIFEST_FORMAT,
        "task": "recover the mean content of two relevant expert candidates",
        "feature_dim": 4,
        "content_dim": 2,
        "capacity": 2,
        "candidate_sources": ["expert-a", "expert-b", "distractor", "distractor"],
        "relevant_candidate_count": 2,
        "distractor_content": "independent Gaussian noise with scale 2",
        "train_count": train_count,
        "holdout_count": holdout_count,
        "controls": ["strongest_single", "dense", "fixed", "random", "none"],
        "split_constraint": "holdout uses new sampled combinations and independent distractor noise",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a3_workspace_manifest_20260825.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_a3_workspace_baseline_20260825.json",
    )
    args = parser.parse_args()
    train, holdout = build_corpus()
    report = WorkspaceCollaborationEvaluator(content_dim=2).evaluate(
        train,
        holdout,
        capacity=2,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(build_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
