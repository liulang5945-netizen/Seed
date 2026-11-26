"""M5.S6B canary: real mixed success/failure executions drive internalization.

Implements the preregistered failure-admission boundary change
(``plans/reference/M5_S6B_FAILURE_ADMISSION_PREREGISTRATION_20260909.md``)
end to end: a single SeedRuntime executes real ``workspace.read`` intents in
an isolated temporary workspace - some succeed (graded reward from the real
read content), some fail (missing paths and directory reads, negative
reward, admitted under the new policy) - and every outcome is projected
through the runtime's internalization boundary into the native learner.  The
grounding lesion must clear the preregistered floor on the unseen holdout
tasks.
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

from api.seed_runtime import SeedRuntime  # noqa: E402
from seed import Seed  # noqa: E402
from seed.config import SeedConfig  # noqa: E402
from seed_platform.workbench import WorkbenchEnvironment  # noqa: E402
from taiji import (  # noqa: E402
    ActionIntent,
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationLedger,
    InternalizedFeatureLearner,
    TaijiConfig,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-s6b-failure-admission-v1"
FEATURE_DIM = 17
GROUNDING_MARGIN_FLOOR = 0.05
N_SUCCESS = 240
N_FAILURE = 120
HOLDOUT_TASKS = 120
RETENTION_TASKS = 120


def _build_fixture(
    root: Path, *, task_seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Real files (success tasks) and real failing targets (failure tasks)."""
    mod = (3, 5, 7)[task_seed % 3]
    valid: list[dict[str, str]] = []
    for index in range(N_SUCCESS):
        name = f"file_{index:04d}.txt"
        token = "x" if (index + task_seed) % mod == 0 else "content-token "
        body = (f"record {index} " + token) * (1 + (index + task_seed) % 7)
        (root / name).write_text(body, encoding="utf-8")
        valid.append({"path": name, "body": body})
    missing: list[dict[str, str]] = []
    for index in range(N_FAILURE):
        if index % 2 == 0:
            missing.append({"path": f"missing_{index:04d}.txt", "kind": "not_found"})
        else:
            missing.append({"path": ".", "kind": "not_a_file"})
    return valid, missing


def _graded_reward(read_result: dict[str, Any]) -> float:
    """Graded reward from the real read content (same semantics as S6-A)."""
    content = str(read_result.get("content", ""))
    byte_length = int(read_result.get("byte_length", 0))
    token_density = content.count("content-token") / max(1, len(content))
    size_term = math.tanh((byte_length - 120) / 120.0)
    token_term = math.tanh(token_density * 40.0)
    return max(-1.0, min(1.0, 0.6 * size_term + 0.4 * token_term))


def run_cell(*, task_seed: int, learner_seed: int) -> dict[str, Any]:
    """One full-runtime cell: real mixed executions -> internalization."""
    started = time.perf_counter()
    temp_root = Path(tempfile.mkdtemp(prefix="taiji_m5_s6b_"))
    try:
        import seed_platform.workbench as workbench_module

        original_get_setting = workbench_module.get_setting
        workbench_module.get_setting = lambda key, default=None: (
            str(temp_root) if key == "workspace_path" else default
        )
        try:
            valid, missing = _build_fixture(temp_root, task_seed=task_seed)
            runtime = SeedRuntime(
                Seed(
                    SeedConfig(taiji=TaijiConfig(seed=learner_seed)),
                    episode_id=f"m5-s6b-cell-{task_seed}-{learner_seed}",
                )
            )
            environment = WorkbenchEnvironment(root=temp_root)
            runtime._workbench_environment = environment
            converter = InternalizationConverter(seed=learner_seed, replay_budget=8000)
            ledger = InternalizationLedger(converter=converter)

            failure_admitted_count = 0
            reward_values: list[float] = []

            def _execute_read(path: str, tag: str) -> None:
                snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
                runtime.execute_workbench_intent(
                    ActionIntent(
                        intent_id=f"intent:{tag}",
                        kind="workspace.read",
                        parameters={"path": path},
                        confidence=1.0,
                        tick=runtime.model.tick,
                    ),
                    snapshot_id=snapshot_id,
                    learn=False,
                )

            def _project(reward: float, tag: str, affordance_id: str) -> Any:
                snapshot_id = runtime.workbench_environment.capability_snapshot.snapshot_id
                return runtime.project_workbench_outcome_for_internalization(
                    snapshot_id=snapshot_id,
                    affordance_id=affordance_id,
                    reward=reward,
                    reward_terms={"read_hit": reward},
                    parent_checkpoint_id="checkpoint:s6b-parent",
                )

            def _admit(source: Any, *, collect: list[Any] | None) -> None:
                nonlocal failure_admitted_count
                result = ledger.ingest(source)
                if result.example is None:
                    raise RuntimeError(f"S6B evidence rejected: {result.reason}")
                if "failure_admitted" in result.lifecycle.events:
                    failure_admitted_count += 1
                if collect is not None:
                    collect.append(result.example)

            train_examples: list[Any] = []

            # Success tasks: real reads, graded reward from real content.
            for index, spec in enumerate(valid):
                tag = f"s{index}"
                read_result = environment.read_workspace_evidence({"path": spec["path"]})
                reward = _graded_reward(read_result)
                reward_values.append(reward)
                _execute_read(spec["path"], tag)
                reprojected = runtime.reproject_workbench_from_latest_evidence(
                    snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id
                )
                affordance_id = reprojected["affordances"][0]["affordance_id"]
                source = _project(reward, tag, affordance_id)
                _admit(source, collect=train_examples)

            # Failure tasks: real failed reads, negative reward, new policy.
            for index, spec in enumerate(missing):
                tag = f"f{index}"
                reward = -0.75
                reward_values.append(reward)
                _execute_read(spec["path"], tag)
                source = _project(reward, tag, "workbench-failed:auto")
                _admit(source, collect=train_examples)

            # Holdout: fresh unseen tasks from the same generative rule.
            holdout_sources: list[Any] = []
            for index in range(N_SUCCESS, N_SUCCESS + HOLDOUT_TASKS // 2):
                name = f"holdout_file_{index:04d}.txt"
                body = (f"record {index} content-token ") * (1 + (index + task_seed) % 7)
                (temp_root / name).write_text(body, encoding="utf-8")
                tag = f"hs{index}"
                read_result = environment.read_workspace_evidence({"path": name})
                reward = _graded_reward(read_result)
                reward_values.append(reward)
                _execute_read(name, tag)
                reprojected = runtime.reproject_workbench_from_latest_evidence(
                    snapshot_id=runtime.workbench_environment.capability_snapshot.snapshot_id
                )
                affordance_id = reprojected["affordances"][0]["affordance_id"]
                holdout_sources.append(_project(reward, tag, affordance_id))
            for index in range(N_FAILURE, N_FAILURE + HOLDOUT_TASKS // 2):
                tag = f"hf{index}"
                reward = -0.75
                reward_values.append(reward)
                _execute_read(f"missing_holdout_{index:04d}.txt", tag)
                holdout_sources.append(_project(reward, tag, "workbench-failed:auto"))

            reward_variance = float(torch.tensor(reward_values).std(unbiased=True))
            holdout_examples = []
            for source in holdout_sources:
                result = converter.convert(source)
                if result.example is None:
                    raise RuntimeError(f"S6B holdout rejected: {result.reason}")
                holdout_examples.append(result.example)
            retention_examples = train_examples[-RETENTION_TASKS:]

            learner = InternalizedFeatureLearner(
                feature_dim=FEATURE_DIM, learning_rate=0.5, reward_bounds=(-1.0, 1.0)
            )
            report = learner.consolidate(
                tuple(train_examples[:-RETENTION_TASKS]),
                holdout_examples=tuple(holdout_examples),
                retention_examples=tuple(retention_examples),
                replay_digest=ledger.replay_digest,
                passes=8,
            )
            checkpoint = learner.checkpoint()
            restored = InternalizedFeatureLearner.from_checkpoint(checkpoint)
            checkpoint_roundtrip = content_digest(restored.checkpoint()) == content_digest(
                checkpoint
            )
            grounding_margin = report.holdout_grounding_lesion_loss - report.holdout_loss_after

            example_id = train_examples[0].example_id
            ledger.advance_status(example_id, "shadow")
            gate = InternalizationCausalGate(
                external_sufficiency=report.holdout_loss_after < report.holdout_loss_before,
                internalization_necessity=report.holdout_internalized_lesion_loss
                > report.holdout_loss_after,
                grounding_necessity=grounding_margin >= GROUNDING_MARGIN_FLOOR,
                checkpoint_recoverable=checkpoint_roundtrip,
                old_task_retention=report.retention_loss_after
                <= report.retention_loss_before + 0.05,
            )
            if gate.passed:
                lifecycle_status = ledger.advance_status(
                    example_id, "internalized", causal_gate=gate
                ).status
            else:
                lifecycle_status = "shadow"

            checks = {
                "failure_evidence_admitted": failure_admitted_count >= N_FAILURE,
                "target_varies": bool(reward_variance > 1e-3),
                "grounding_margin_clears_floor": bool(grounding_margin >= GROUNDING_MARGIN_FLOOR),
                "checkpoint_roundtrip": bool(checkpoint_roundtrip),
                "gate_passed": bool(gate.passed),
            }
            return {
                "task_seed": int(task_seed),
                "learner_seed": int(learner_seed),
                "data": {
                    "success_tasks": N_SUCCESS,
                    "failure_tasks": N_FAILURE,
                    "holdout_tasks": HOLDOUT_TASKS,
                    "train_examples": len(train_examples) - RETENTION_TASKS,
                    "failure_admitted_count": failure_admitted_count,
                    "reward_variance": reward_variance,
                },
                "metrics": report.to_payload(),
                "grounding_lesion_margin": grounding_margin,
                "holdout_loss_after": report.holdout_loss_after,
                "lifecycle_status": lifecycle_status,
                "checks": checks,
                "technical_gate_all_passed": all(bool(v) for v in checks.values()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        finally:
            workbench_module.get_setting = original_get_setting
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
        "outcome_source": (
            "real mixed success/failure WorkbenchEnvironment executions via the "
            "SeedRuntime internalization projection (S6B failure-admission policy)"
        ),
        "grounding_margin_floor": GROUNDING_MARGIN_FLOOR,
        "cell": cell,
        "boundary": (
            "M5.S6B canary only; failures enter with non-positive reward only; "
            "no provider, network, or real client write"
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": cell["technical_gate_all_passed"],
                "failed_checks": [k for k, v in cell["checks"].items() if not v],
                "failure_admitted_count": cell["data"]["failure_admitted_count"],
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
