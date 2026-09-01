"""E7 脑—客户端协同选择器的可执行合同。

这些测试把 plans/active/roadmap/04_EXECUTION_PLAN.md 第 17 节 E7 的
Gate 从"文档约定"变成"机器强制约束"：选择器只能在六类候选中输出唯一一个，
且不得在已有能力可解决时向 Seed 客户端申请新插件。
"""

from __future__ import annotations

import pytest

from taiji.evolution_credit import (
    EVOLUTION_CREDIT_CANDIDATE_KINDS,
    BrainClientCreditDecision,
    BrainClientCreditSelector,
)
from taiji.evolution_experience import EvolutionExperience


def _experience(**overrides: object) -> EvolutionExperience:
    payload: dict[str, object] = {
        "experience_id": "episode-1",
        "source_kind": "workbench",
        "source_id": "workbench.filesystem.read",
        "source_version": "1",
        "source_digest": "a" * 64,
        "parent_checkpoint_digest": "b" * 64,
        "partition": "train",
        "status": "success",
        "success": True,
        "capability_id": "workbench.filesystem.read",
        "reward_components": {"task": 0.4},
        "resource_usage": {"latency_ms": 2.0},
    }
    payload.update(overrides)
    return EvolutionExperience(**payload)  # type: ignore[arg-type]


def test_registered_capability_is_learned_instead_of_requesting_a_client_plugin() -> None:
    """Gate：已有能力可解决时不申请插件。

    能力已注册意味着执行通路存在，收益只能来自学习既有通路；
    若此时仍产出 client_capability_candidate，等于用装插件掩盖学习不足。
    """
    selector = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )

    decision = selector.select(_experience())

    assert decision.candidate_kind in ("weight_update", "memory_consolidation")
    assert "capability_already_registered" in decision.reasons


def test_missing_affordance_asks_the_client_instead_of_growing_structure() -> None:
    """Gate：缺少 affordance 时不靠增加突触伪造执行器。

    即使容量门已经批准结构增长，没有执行通路时长出神经元也变不出执行器；
    唯一诚实的输出是向客户端申请能力。
    """
    selector = BrainClientCreditSelector(registered_capability_ids=())

    decision = selector.select(_experience(), growth_permitted=True)

    assert decision.candidate_kind == "client_capability_candidate"
    assert "capability_not_registered" in decision.reasons


def test_language_failure_does_not_trigger_structural_growth() -> None:
    """Gate：语言失败不触发结构增长。

    语言器官（provider）的失败属于语言支线，把它当成容量不足的证据
    会让本体无限长大来补偿一个不相干的故障源。
    """
    selector = BrainClientCreditSelector(registered_capability_ids=("provider.qwen",))

    decision = selector.select(
        _experience(
            source_kind="provider",
            source_id="provider.qwen",
            capability_id="provider.qwen",
            status="error",
            success=False,
            error_code="taiji_execution_error",
        ),
        growth_permitted=True,
    )

    assert decision.candidate_kind == "clarify_or_stop"
    assert "language_failure_is_not_structural_evidence" in decision.reasons


def test_exhausted_resources_stop_instead_of_growing_without_bound() -> None:
    """Gate：资源不足时降级/停止而不是无限增长。"""
    selector = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )

    decision = selector.select(
        _experience(), growth_permitted=True, resources_exhausted=True
    )

    assert decision.candidate_kind == "clarify_or_stop"
    assert "resource_budget_exhausted" in decision.reasons


def test_every_candidate_kind_is_reachable_and_exactly_one_is_emitted() -> None:
    """六类输出互斥：每个场景只产出一类，且六类都真实可达。

    若某一类永远不可达，它就是文档里的死条目；若一个场景能产出多类，
    收益归属就无法解释。
    """
    registered = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read", "provider.qwen")
    )
    unregistered = BrainClientCreditSelector(registered_capability_ids=())

    observed = {
        registered.select(_experience()).candidate_kind,
        registered.select(_experience(partition="retention")).candidate_kind,
        registered.select(
            _experience(status="error", success=False, error_code="artifact_rejected")
        ).candidate_kind,
        registered.select(_experience(), growth_permitted=True).candidate_kind,
        unregistered.select(_experience()).candidate_kind,
        unregistered.select(_experience(capability_id="")).candidate_kind,
    }

    assert observed == set(EVOLUTION_CREDIT_CANDIDATE_KINDS)
    assert len(set(EVOLUTION_CREDIT_CANDIDATE_KINDS)) == len(
        EVOLUTION_CREDIT_CANDIDATE_KINDS
    )


def test_tampered_decision_payload_is_rejected() -> None:
    """决策必须内容寻址，否则事后无法证明收益归属没有被改写。"""
    selector = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )
    payload = selector.select(_experience()).to_payload()

    assert BrainClientCreditDecision.from_payload(payload).candidate_kind == (
        payload["candidate_kind"]
    )

    payload["candidate_kind"] = "structure_candidate"
    with pytest.raises(ValueError, match="digest mismatch"):
        BrainClientCreditDecision.from_payload(payload)
