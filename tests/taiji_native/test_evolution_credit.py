"""E7 脑—客户端协同选择器的可执行合同。

这些测试把 plans/active/roadmap/04_EXECUTION_PLAN.md 第 17 节 E7 的
Gate 从"文档约定"变成"机器强制约束"：选择器只能在六类候选中输出唯一一个，
且不得在已有能力可解决时向 Seed 客户端申请新插件。
"""

from __future__ import annotations

import pytest

from taiji.evolution_credit import (
    EVOLUTION_CREDIT_ARMS,
    EVOLUTION_CREDIT_CANDIDATE_KINDS,
    BrainClientAblationAttribution,
    BrainClientCreditDecision,
    BrainClientCreditSelector,
    attribute_brain_client_ablation,
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


def _ablation_experiences() -> tuple[EvolutionExperience, ...]:
    return (
        _experience(experience_id="episode-1"),
        _experience(experience_id="episode-2", partition="retention"),
        _experience(experience_id="episode-3", capability_id=""),
    )


def test_brain_only_arm_credits_learning_and_never_the_client_plugin() -> None:
    """Gate：brain-only 对照的收益只能归给脑内学习。

    第 16 节要求每个变更都要有 brain 侧与 client-plugin-only 侧对照。
    能力已注册这一臂根本没有产生插件申请，若归属结果里出现
    client_plugin_only 收益，就是把不存在的干预算进了账。
    """
    registered = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )
    experiences = _ablation_experiences()

    attribution = attribute_brain_client_ablation(
        registered.select(item) for item in experiences
    )

    assert attribution.brain_only == ("episode-1", "episode-2")
    assert attribution.client_plugin_only == ()
    assert attribution.unattributed == ("episode-3",)


def test_client_plugin_only_arm_credit_is_not_absorbed_into_self_evolution() -> None:
    """Gate：client-plugin-only 对照的收益不得笼统归于"自进化"。

    这是第 16 节明令禁止的失真：同一批经验在能力未注册时，收益完全来自
    "向客户端申请能力"，此时 brain_only 必须为空，否则脑就白拿了客户端的功劳。
    """
    unregistered = BrainClientCreditSelector(registered_capability_ids=())
    experiences = _ablation_experiences()

    attribution = attribute_brain_client_ablation(
        unregistered.select(item) for item in experiences
    )

    assert attribution.client_plugin_only == ("episode-1", "episode-2")
    assert attribution.brain_only == ()
    assert attribution.unattributed == ("episode-3",)


def test_two_arms_explain_the_same_delta_on_the_same_experience_set() -> None:
    """Gate：两臂对照能解释收益归属。

    同一经验集只改 registry 一个变量，归属差值必须整体从一侧移到另一侧，
    且两臂对"无法归属"的判断一致——否则差值就无法被解释为该变量造成的。
    """
    experiences = _ablation_experiences()
    brain = attribute_brain_client_ablation(
        BrainClientCreditSelector(
            registered_capability_ids=("workbench.filesystem.read",)
        ).select(item)
        for item in experiences
    )
    client = attribute_brain_client_ablation(
        BrainClientCreditSelector(registered_capability_ids=()).select(item)
        for item in experiences
    )

    assert brain.brain_only == client.client_plugin_only
    assert brain.unattributed == client.unattributed
    for attribution in (brain, client):
        credited = attribution.brain_only + attribution.client_plugin_only
        assert set(credited).isdisjoint(attribution.unattributed)
        assert len(credited) + len(attribution.unattributed) == len(experiences)


def test_every_candidate_kind_maps_to_exactly_one_arm() -> None:
    """每一类候选都必须落在唯一一臂，否则归属存在死角或重复计数。"""
    assert set(EVOLUTION_CREDIT_ARMS) == {
        "brain_only",
        "client_plugin_only",
        "unattributed",
    }

    registered = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read", "provider.qwen")
    )
    unregistered = BrainClientCreditSelector(registered_capability_ids=())
    decisions = (
        registered.select(_experience(experience_id="k1")),
        registered.select(_experience(experience_id="k2", partition="retention")),
        registered.select(
            _experience(
                experience_id="k3",
                status="error",
                success=False,
                error_code="artifact_rejected",
            )
        ),
        registered.select(_experience(experience_id="k4"), growth_permitted=True),
        unregistered.select(_experience(experience_id="k5")),
        unregistered.select(_experience(experience_id="k6", capability_id="")),
    )
    assert {item.candidate_kind for item in decisions} == set(
        EVOLUTION_CREDIT_CANDIDATE_KINDS
    )

    attribution = attribute_brain_client_ablation(decisions)

    assert attribution.brain_only == ("k1", "k2", "k3", "k4")
    assert attribution.client_plugin_only == ("k5",)
    assert attribution.unattributed == ("k6",)


def test_the_same_episode_cannot_be_counted_twice() -> None:
    """同一 episode 重复入账会凭空放大收益，必须 fail-closed。"""
    registered = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )
    decision = registered.select(_experience())

    with pytest.raises(ValueError, match="duplicate"):
        attribute_brain_client_ablation((decision, decision))


def test_tampered_attribution_payload_is_rejected() -> None:
    """归属结论同样内容寻址：事后把收益搬到另一臂必须被发现。"""
    registered = BrainClientCreditSelector(
        registered_capability_ids=("workbench.filesystem.read",)
    )
    payload = attribute_brain_client_ablation(
        registered.select(item) for item in _ablation_experiences()
    ).to_payload()

    restored = BrainClientAblationAttribution.from_payload(payload)
    assert restored.brain_only == ("episode-1", "episode-2")

    payload["client_plugin_only"] = ["episode-1"]
    with pytest.raises(ValueError, match="digest mismatch"):
        BrainClientAblationAttribution.from_payload(payload)
