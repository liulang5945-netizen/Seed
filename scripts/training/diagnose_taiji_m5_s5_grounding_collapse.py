"""M5.S5 diagnostic: isolate why the grounding lesion collapses.

S3/S4 both drive holdout loss AND the grounding-lesion loss to ~0 at once.
``score(grounding_enabled=False)`` returns the bias alone, so both being ~0
implies bias≈target and weights·features≈0: the outcome is fully predicted
by the bias and the grounded features carry no information the learner
needs.  This micro-experiment isolates the mechanism from the embedding
pipeline: it holds the learner fixed and varies only (a) whether the target
is constant across examples and (b) the feature geometry, then reports the
grounding-lesion margin.  It trains no Taiji model and touches no corpus.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import GroundedOutcomeEvidence, Outcome, WorldAffordance, content_digest  # noqa: E402
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

DIAGNOSTIC_FORMAT = "taiji-m5-s5-grounding-collapse-diagnostic-v1"
DIM = 384
N = 2000
PASSES = 8


def _example(features: torch.Tensor, reward: float, tag: str) -> Any:
    affordance = WorldAffordance(
        affordance_id=f"affordance:{tag}",
        action_kind="diagnostic",
        actor_id="diagnostic",
        target_id=f"target:{tag}",
        features=features,
        feature_provenance="world-state-grounding",
        grounding_lineage=(f"world-state:{tag}",),
    )
    return GroundedOutcomeEvidence(
        evidence_id=f"evidence:{tag}",
        outcome_id=f"outcome:{tag}",
        outcome=Outcome(
            intent_id=f"intent:{tag}",
            reward=reward,
            success=reward >= 0.0,
            tick=1,
        ),
        affordance=affordance,
        capability_snapshot_digest="capability-sha256:s5",
        parent_checkpoint_id="checkpoint:s5-parent",
        owner_id="taiji:diagnostic",
        reward_terms={"outcome": reward},
        world_digest=f"world-sha256:{tag}",
    )


def _run_case(
    *,
    generator: torch.Generator,
    constant_target: bool,
    normalize_features: bool,
) -> dict[str, float]:
    train_features = torch.randn(N, DIM, generator=generator)
    holdout_features = torch.randn(N, DIM, generator=generator)
    if normalize_features:
        train_features = torch.nn.functional.normalize(train_features, p=2, dim=1)
        holdout_features = torch.nn.functional.normalize(holdout_features, p=2, dim=1)
    if constant_target:
        train_targets = torch.ones(N)
        holdout_targets = torch.ones(N)
    else:
        # Target is a fixed linear readout of the features: the grounded
        # features genuinely carry outcome information.
        w_true = torch.randn(DIM, generator=generator)
        w_true = torch.nn.functional.normalize(w_true, p=2, dim=0)
        train_targets = (train_features @ w_true).clamp(-1.0, 1.0)
        holdout_targets = (holdout_features @ w_true).clamp(-1.0, 1.0)

    learner = InternalizedFeatureLearner(feature_dim=DIM, learning_rate=0.5)
    from taiji.internalization import InternalizationConverter

    converter = InternalizationConverter(seed=17, replay_budget=N)
    train_examples = tuple(
        converter.convert(_example(train_features[i], float(train_targets[i]), f"t{i}")).example
        for i in range(N)
    )
    holdout_examples = tuple(
        converter.convert(_example(holdout_features[i], float(holdout_targets[i]), f"h{i}")).example
        for i in range(N)
    )
    train_examples = tuple(e for e in train_examples if e is not None)
    holdout_examples = tuple(e for e in holdout_examples if e is not None)
    report = learner.consolidate(
        train_examples,
        holdout_examples=holdout_examples,
        retention_examples=train_examples[:200],
        replay_digest=content_digest({"case": [constant_target, normalize_features]}),
        passes=PASSES,
    )
    return {
        "holdout_loss_after": report.holdout_loss_after,
        "grounding_lesion_loss": report.holdout_grounding_lesion_loss,
        "grounding_lesion_margin": (
            report.holdout_grounding_lesion_loss - report.holdout_loss_after
        ),
        "learned_bias": float(learner.bias),
        "weight_norm": float(torch.linalg.norm(learner.weights)),
    }


def main() -> int:
    started = time.perf_counter()
    cases: dict[str, Any] = {}
    for constant in (True, False):
        for normalize in (True, False):
            key = f"constant_target={constant},normalized_features={normalize}"
            gen = torch.Generator().manual_seed(9)
            cases[key] = _run_case(
                generator=gen,
                constant_target=constant,
                normalize_features=normalize,
            )
    payload = {
        "format": DIAGNOSTIC_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "can_promote": False,
        "hypothesis": (
            "the grounding lesion collapses to ~0 whenever the target is "
            "constant, because score(grounding=False)=bias and a constant "
            "outcome is fully predicted by the bias; feature geometry and "
            "holdout heterogeneity are secondary"
        ),
        "cases": cases,
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
    }
    report_path = (
        PROJECT_ROOT / "reports" / "taiji_m5_s5_grounding_collapse_diagnostic_20260909.json"
    )
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cases, ensure_ascii=False, indent=2))
    print("report:", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
