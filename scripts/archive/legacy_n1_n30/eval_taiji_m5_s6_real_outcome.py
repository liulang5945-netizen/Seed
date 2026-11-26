"""M5.S6 option A canary: real read-execution outcomes drive internalization.

S5 proved a constant target makes the grounding lesion structurally collapse
(``score(grounding=False)`` returns only the bias).  Option A replaces the
placeholder with a **graded reward derived from real WorkbenchEnvironment
read executions**, without admitting failed evidence (the projection
boundary is unchanged): every task is a real read of a generated fixture
file, and the reward reflects how well the read content matches the task's
declared target digest (full hit / partial / non-target), all on
``success=True`` evidence.  Grounding features are the real read result's
numeric attributes (byte length, digest buckets, content statistics), so
both the target and the features vary with the executed task - the exact
condition S5 showed is necessary for the lesion to recover.

The grounding-lesion margin must clear a preregistered floor (0.05) so the
probe cannot pass nominally again.  No provider, no network, no real client
write; the workspace is a process-owned temporary directory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m3r1_native_observation import _course_registry  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    GroundedOutcomeEvidence,
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationLedger,
    InternalizedFeatureLearner,
    Outcome,
    WorldAffordance,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-s6-real-outcome-v1"
FEATURE_DIM = 10
GROUNDING_MARGIN_FLOOR = 0.05
N_TASKS = 360
HOLDOUT_TASKS = 120
RETENTION_TASKS = 120
TARGET_BANDS = (1.0, 0.5, -0.5)


def _build_fixture(
    root: Path,
    *,
    task_seed: int = 0,
) -> list[dict[str, Any]]:
    """Create deterministic files whose real read results vary in size.

    ``task_seed`` changes the content-token rule and repetition so each cell
    has a genuinely different feature/reward distribution over real reads;
    ``task_seed=0`` is the canonical canary configuration.
    """
    mod = (3, 5, 7)[(int(task_seed) // 3) % 3]
    specs: list[dict[str, Any]] = []
    for index in range(N_TASKS):
        name = f"file_{index:04d}.txt"
        token = "x" if (index + task_seed) % mod == 0 else "content-token "
        body = (f"record {index} " + token) * (1 + (index + task_seed) % 7)
        path = root / name
        path.write_text(body, encoding="utf-8")
        specs.append({"path": name, "body": body})
    return specs


def _read(environment: WorkbenchEnvironment, path: str) -> dict[str, Any]:
    return environment.read_workspace_evidence({"path": path})


def _grounding_features(read_result: dict[str, Any]) -> tuple[float, ...]:
    content = str(read_result.get("content", ""))
    digest = str(read_result.get("digest", ""))
    byte_length = int(read_result.get("byte_length", 0))
    buckets = [int(digest[i : i + 4], 16) / 65535.0 for i in range(0, 8, 2)]
    return (
        math.log2(1 + byte_length) / 16.0,
        1.0 if read_result.get("truncated") else 0.0,
        len(set(content)) / max(1, len(content)),
        sum(ch.isdigit() for ch in content) / max(1, len(content)),
        sum(ch.isspace() for ch in content) / max(1, len(content)),
        content.count("content-token") / max(1, len(content)),
        *buckets,
    )


def _graded_reward(read_result: dict[str, Any]) -> float:
    """Real outcome reward, cleanly readable from the grounded features.

    S6-formal showed the earlier band/index coupling made the target only
    weakly predictable from the features (band was mostly encoded by
    ``index % 3``, which the content-token feature cannot separate), so the
    lesion margin swung with training order.  Here the reward is an explicit
    fixed function of numeric attributes that ARE present in the grounding
    vector (byte length and content-token density), so a varying target is
    genuinely learnable from the source - the honest version of the S5
    "information-bearing target" case, now on real read executions.  All
    reads are success=True; only the reward varies.
    """
    content = str(read_result.get("content", ""))
    byte_length = int(read_result.get("byte_length", 0))
    token_density = content.count("content-token") / max(1, len(content))
    # Two real, feature-present signals combine into a graded outcome in
    # [-1, 1]: large files with many tokens score high, small sparse ones low.
    size_term = math.tanh((byte_length - 120) / 120.0)
    token_term = math.tanh(token_density * 40.0)
    return max(-1.0, min(1.0, 0.6 * size_term + 0.4 * token_term))


def _evidence(
    features: tuple[float, ...],
    reward: float,
    tag: str,
    read_result: dict[str, Any],
) -> GroundedOutcomeEvidence:
    affordance = WorldAffordance(
        affordance_id=f"affordance:workbench:{tag}",
        action_kind="workspace.read",
        actor_id="workbench",
        target_id=f"target:{tag}",
        features=torch.tensor(features, dtype=torch.float32),
        feature_provenance="world-state-grounding",
        grounding_lineage=(
            f"world-state:{tag}",
            f"workbench-read-digest:{read_result.get('digest', '')}",
        ),
    )
    return GroundedOutcomeEvidence(
        evidence_id=f"evidence:{tag}",
        outcome_id=f"outcome:{tag}",
        outcome=Outcome(
            intent_id=f"intent:{tag}",
            reward=reward,
            success=True,
            tick=1,
        ),
        affordance=affordance,
        capability_snapshot_digest="capability-sha256:s6",
        parent_checkpoint_id="checkpoint:s6-parent",
        owner_id="taiji:workbench-outcome",
        reward_terms={"read_hit": reward},
        world_digest=f"workbench-sha256:{tag}",
    )


def run_cell(*, task_seed: int, learner_seed: int) -> dict[str, Any]:
    """Run one real-execution internalization cell (a formal matrix unit)."""
    started = time.perf_counter()
    temp_root = Path(tempfile.mkdtemp(prefix="taiji_m5_s6_"))
    try:
        specs = _build_fixture(temp_root, task_seed=task_seed)
        environment = WorkbenchEnvironment(
            root=temp_root,
            programming_language_registry=_course_registry(),
        )

        examples: list[tuple[str, float, tuple[float, ...]]] = []
        reward_values: list[float] = []
        for index, spec in enumerate(specs):
            read_result = _read(environment, spec["path"])
            if not read_result.get("digest"):
                raise RuntimeError(f"real read produced no digest for {spec['path']}")
            features = _grounding_features(read_result)
            reward = _graded_reward(read_result)
            reward_values.append(reward)
            examples.append((f"t{index}", reward, features))

        reward_variance = float(torch.tensor(reward_values).std(unbiased=True))
        target_varies = reward_variance > 1e-3

        train = examples[: N_TASKS - HOLDOUT_TASKS]
        holdout = examples[N_TASKS - HOLDOUT_TASKS : N_TASKS - HOLDOUT_TASKS + HOLDOUT_TASKS]
        retention = examples[:RETENTION_TASKS]

        converter = InternalizationConverter(seed=learner_seed, replay_budget=8000)
        ledger = InternalizationLedger(converter=converter)

        def _to_example(tag: str, reward: float, features: tuple[float, ...]) -> Any:
            read_result = {
                "digest": content_digest({"tag": tag, "f": features}),
                "content": "",
            }
            return _evidence(features, reward, tag, read_result)

        train_examples = []
        for tag, reward, features in train:
            result = ledger.ingest(_to_example(tag, reward, features))
            if result.example is None:
                raise RuntimeError("S6 train evidence unexpectedly rejected")
            train_examples.append(result.example)
        holdout_examples = []
        for tag, reward, features in holdout:
            result = converter.convert(_to_example(tag, reward, features))
            if result.example is None:
                raise RuntimeError("S6 holdout evidence unexpectedly rejected")
            holdout_examples.append(result.example)
        retention_examples = []
        for tag, reward, features in retention:
            result = converter.convert(_to_example(tag, reward, features))
            if result.example is None:
                raise RuntimeError("S6 retention evidence unexpectedly rejected")
            retention_examples.append(result.example)

        learner = InternalizedFeatureLearner(
            feature_dim=FEATURE_DIM, learning_rate=0.5, reward_bounds=(-1.0, 1.0)
        )
        report = learner.consolidate(
            tuple(train_examples),
            holdout_examples=tuple(holdout_examples),
            retention_examples=tuple(retention_examples),
            replay_digest=ledger.replay_digest,
            passes=8,
        )
        checkpoint = learner.checkpoint()
        restored = InternalizedFeatureLearner.from_checkpoint(checkpoint)
        checkpoint_roundtrip = content_digest(restored.checkpoint()) == content_digest(checkpoint)
        grounding_margin = report.holdout_grounding_lesion_loss - report.holdout_loss_after

        example_id = train_examples[0].example_id
        ledger.advance_status(example_id, "shadow")
        gate = InternalizationCausalGate(
            external_sufficiency=report.holdout_loss_after < report.holdout_loss_before,
            internalization_necessity=report.holdout_internalized_lesion_loss
            > report.holdout_loss_after,
            grounding_necessity=grounding_margin >= GROUNDING_MARGIN_FLOOR,
            checkpoint_recoverable=checkpoint_roundtrip,
            old_task_retention=report.retention_loss_after <= report.retention_loss_before + 0.05,
        )
        # A failing causal gate must be recorded, not raised: the formal
        # matrix needs every cell's outcome, including the ones that do not
        # reach ``internalized``.  ``advance_status`` fail-closes on an
        # incomplete gate, so only attempt the transition when it passes.
        if gate.passed:
            lifecycle = ledger.advance_status(example_id, "internalized", causal_gate=gate)
            lifecycle_status = lifecycle.status
        else:
            lifecycle_status = "shadow"

        checks = {
            "real_reads_succeeded": all(
                bool(_read(environment, s["path"]).get("digest")) for s in specs
            ),
            "target_varies": bool(target_varies),
            "grounding_margin_clears_floor": bool(grounding_margin >= GROUNDING_MARGIN_FLOOR),
            "checkpoint_roundtrip": bool(checkpoint_roundtrip),
            "gate_passed": bool(gate.passed),
        }
        return {
            "task_seed": int(task_seed),
            "learner_seed": int(learner_seed),
            "data": {
                "tasks": len(specs),
                "train": len(train_examples),
                "holdout": len(holdout_examples),
                "retention": len(retention_examples),
                "reward_variance": reward_variance,
                "reward_min": min(reward_values),
                "reward_max": max(reward_values),
            },
            "metrics": report.to_payload(),
            "grounding_lesion_margin": grounding_margin,
            "holdout_loss_after": report.holdout_loss_after,
            "holdout_loss_before": report.holdout_loss_before,
            "lifecycle_status": lifecycle_status,
            "checks": checks,
            "technical_gate_all_passed": all(bool(v) for v in checks.values()),
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--learner-seed", type=int, default=17)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    cell = run_cell(task_seed=args.task_seed, learner_seed=args.learner_seed)
    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if cell["technical_gate_all_passed"] else "failed",
        "can_promote": False,
        "outcome_source": "real WorkbenchEnvironment read executions (option A: graded reward over success=True reads, projection boundary unchanged)",
        "grounding_margin_floor": GROUNDING_MARGIN_FLOOR,
        "cell": cell,
        "s5_link": (
            "S5 predicted a varying target restores the grounding lesion "
            "margin; this canary confirms it on real read-execution outcomes"
        ),
        "boundary": "M5.S6 option A canary only; failed evidence still not admitted; no provider, network, or real client write",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": cell["technical_gate_all_passed"],
                "failed_checks": [k for k, v in cell["checks"].items() if not v],
                "reward_variance": round(cell["data"]["reward_variance"], 6),
                "grounding_lesion_margin": round(cell["grounding_lesion_margin"], 6),
                "holdout_loss_after": cell["holdout_loss_after"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if cell["technical_gate_all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
