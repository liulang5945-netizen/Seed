from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.training import run_taiji_r2_daily_dialogue_warmstart_matched as matched
from taiji.internalization import content_digest

# 2026-10-25 owner 裁定：data/ 下 checkpoints/语料不入 git——产物在位才跑，
# CI 上跳过（本机仍全量守护）。准入出处：roadmap/12 批 C。
_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/dialogue_context_final.pt"
)
_ARTIFACT_PRESENT = _ARTIFACT.is_file()
_ARTIFACT_REASON = (
    "consumes data/ artifacts that are not git-stored "
    "(owner 2026-10-25: checkpoints/corpus stay out of git)"
)


def test_matched_arms_use_disjoint_write_once_paths() -> None:
    warm = matched._arm_paths("warm_start")
    fresh = matched._arm_paths("fresh_same_capacity")
    assert warm["checkpoint_dir"] != fresh["checkpoint_dir"]
    assert warm["readiness"] != fresh["readiness"]
    assert warm["attempt"] != fresh["attempt"]
    assert warm["progress"] != fresh["progress"]
    assert warm["result"] != fresh["result"]
    assert all(
        path.is_relative_to(matched.CHECKPOINT_ROOT)
        for path in (warm["checkpoint_dir"], fresh["checkpoint_dir"])
    )


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_warm_and_fresh_models_share_shape_but_not_initial_weights() -> None:
    payload = torch.load(matched.base.BASELINE_CHECKPOINT, map_location="cpu", weights_only=True)
    warm = matched._new_model("warm_start", payload)
    fresh = matched._new_model("fresh_same_capacity", payload)

    assert warm.parameter_count() == fresh.parameter_count()
    assert (
        warm.checkpoint()["config"]["region_sizes"] == fresh.checkpoint()["config"]["region_sizes"]
    )
    assert (
        warm.checkpoint()["config"]["synapse_fan_in"]
        == fresh.checkpoint()["config"]["synapse_fan_in"]
    )
    assert warm.checkpoint()["config"]["seed"] != fresh.checkpoint()["config"]["seed"]
    warm_weights = {key: value for key, value in warm.checkpoint().items() if key != "rng_state"}
    fresh_weights = {key: value for key, value in fresh.checkpoint().items() if key != "rng_state"}
    assert content_digest(warm_weights) != content_digest(fresh_weights)


def test_parallel_resource_ceiling_matches_frozen_plan() -> None:
    assert matched.CPU_LIMIT_PER_ARM == 120_000.0
    assert matched.MIN_FREE_MEMORY_GIB == 8.0
    assert matched.ARM_NAMES == ("warm_start", "fresh_same_capacity")
    assert matched._sha256(matched.PLAN) == matched.PLAN_SHA256
