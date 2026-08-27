from dataclasses import replace

from taiji import (
    RecoveryArchiveEntry,
    RecoveryReaderContribution,
    RecoveryReaderDependencyGraph,
    RecoveryReaderInteraction,
    RecoveryReaderInteractionGroup,
    RecoveryStrategyLedger,
)


def _entry(
    rollout_id: str,
    *,
    resource_cost: float,
    evidence_count: int,
    outcome_consistency: float,
) -> RecoveryArchiveEntry:
    return RecoveryArchiveEntry(
        portfolio_id="portfolio",
        rollout_id=rollout_id,
        source_episode_id="episode",
        goal_id="goal",
        lifecycle="completed",
        action_kinds=("idle",),
        resource_cost=resource_cost,
        outcome_reward=1.0,
        outcome_success=True,
        terminal=True,
        evidence_count=evidence_count,
        outcome_consistency=outcome_consistency,
    )


def _interaction(
    first: str,
    second: str,
    *,
    residual: float = 0.0,
    order_delta: float = 0.0,
    order_invariant: bool = True,
) -> RecoveryReaderInteraction:
    return RecoveryReaderInteraction(
        reader_kind="semantic",
        strategy_rollout_ids=(first, second),
        memory_ids=(f"memory-{first}", f"memory-{second}"),
        pair_effect_l2=1.0 + residual,
        additive_effect_l2=1.0,
        interaction_delta_l2=residual,
        interaction_residual_l2=residual,
        order_delta_l2=order_delta,
        order_invariant=order_invariant,
        replay_epochs=1,
        replay_learning_rate=0.1,
    )


def _interaction_group(
    *rollout_ids: str,
    higher_order_residual: float = 0.5,
    order_delta: float = 0.0,
) -> RecoveryReaderInteractionGroup:
    return RecoveryReaderInteractionGroup(
        reader_kind="semantic",
        strategy_rollout_ids=tuple(rollout_ids),
        memory_ids=tuple(f"memory-{rollout_id}" for rollout_id in rollout_ids),
        group_effect_l2=3.0 + higher_order_residual,
        additive_effect_l2=3.0,
        pairwise_interaction_delta_l2=0.0,
        pairwise_predicted_effect_l2=3.0,
        higher_order_delta_l2=higher_order_residual,
        higher_order_residual_l2=higher_order_residual,
        order_delta_l2=order_delta,
        order_invariant=order_delta <= 1e-7,
        replay_epochs=1,
        replay_learning_rate=0.1,
    )


def test_recovery_strategy_higher_order_group_is_atomic_under_budget() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=0.5)
    for rollout_id in ("first", "second", "third"):
        ledger = ledger.admit(
            _entry(rollout_id, resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
            memory_id=f"memory-{rollout_id}",
        )

    pairs = (
        _interaction("first", "second"),
        _interaction("first", "third"),
        _interaction("second", "third"),
    )
    selected = ledger.select_with_interaction_audit(
        pairs,
        groups=(_interaction_group("first", "second", "third"),),
        audit_available=True,
        residual_tolerance=1e-7,
        order_tolerance=1e-7,
    )

    assert selected == ()


def test_recovery_reader_higher_order_group_roundtrip_and_revocation() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=1.0)
    for rollout_id in ("first", "second", "third"):
        ledger = ledger.admit(
            _entry(rollout_id, resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
            memory_id=f"memory-{rollout_id}",
        )
    pairs = (
        _interaction("first", "second"),
        _interaction("first", "third"),
        _interaction("second", "third"),
    )
    group = _interaction_group("first", "second", "third")
    policy_ledger = ledger.record_interaction_audit(pairs, groups=(group,), audit_available=True)
    assert policy_ledger.selected_rollout_ids == ("first", "second", "third")
    assert RecoveryStrategyLedger.from_payload(policy_ledger.to_payload()) == policy_ledger
    graph = RecoveryReaderDependencyGraph().bind(
        "semantic",
        policy_ledger.selected_approvals,
        interactions=pairs,
        interaction_groups=(group,),
        base_checkpoint={"value": 1},
        base_checkpoint_digest="baseline-digest",
    )
    restored = RecoveryReaderDependencyGraph.from_payload(graph.to_payload())
    assert restored == graph
    dependency = restored.dependency_for("semantic")
    assert dependency is not None
    assert dependency.interaction_groups == (group,)

    revoked = policy_ledger.revoke("first")
    updated = restored.retain_selected(revoked.selected_approvals)
    dependency = updated.dependency_for("semantic")
    assert dependency is not None
    assert dependency.interaction_groups == ()


def test_recovery_strategy_interaction_policy_keeps_atomic_pair_together() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=0.7)
    ledger = ledger.admit(
        _entry("primary", resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
        memory_id="memory-primary",
    )
    ledger = ledger.admit(
        _entry("survivor", resource_cost=0.4, evidence_count=2, outcome_consistency=0.9),
        memory_id="memory-survivor",
    )
    ledger = ledger.admit(
        _entry("independent", resource_cost=0.2, evidence_count=2, outcome_consistency=0.4),
        memory_id="memory-independent",
    )

    selected = ledger.select_with_interaction_audit(
        (
            _interaction(
                "primary", "survivor", residual=0.5, order_delta=0.2, order_invariant=False
            ),
            _interaction("primary", "independent"),
            _interaction("survivor", "independent"),
        ),
        audit_available=True,
        residual_tolerance=1e-7,
        order_tolerance=1e-7,
    )

    assert tuple(approval.rollout_id for approval in selected) == ("primary", "survivor")


def test_recovery_strategy_interaction_policy_is_fail_closed_for_unknown_pairs() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=0.5)
    for rollout_id in ("primary", "survivor", "unknown"):
        ledger = ledger.admit(
            _entry(rollout_id, resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
            memory_id=f"memory-{rollout_id}",
        )

    selected = ledger.select_with_interaction_audit(
        (_interaction("primary", "survivor"),),
        audit_available=True,
        residual_tolerance=1e-7,
        order_tolerance=1e-7,
    )

    assert selected == ()


def test_recovery_strategy_competition_respects_budget_and_survivor() -> None:
    ledger = RecoveryStrategyLedger(evidence_threshold=2, memory_budget=0.7)
    primary = _entry(
        "primary",
        resource_cost=0.2,
        evidence_count=2,
        outcome_consistency=1.0,
    )
    survivor = _entry(
        "survivor",
        resource_cost=0.4,
        evidence_count=3,
        outcome_consistency=0.9,
    )
    budget_limited = _entry(
        "budget-limited",
        resource_cost=0.5,
        evidence_count=2,
        outcome_consistency=0.4,
    )
    ledger = ledger.admit(primary, memory_id="memory-primary")
    ledger = ledger.admit(survivor, memory_id="memory-survivor")
    ledger = ledger.admit(budget_limited, memory_id="memory-budget-limited")

    assert set(ledger.selected_rollout_ids) == {"primary", "survivor"}
    assert ledger.selected_memory_ids == ("memory-primary", "memory-survivor")
    assert set(ledger.approved_memory_ids) == {
        "memory-primary",
        "memory-survivor",
        "memory-budget-limited",
    }

    ledger = ledger.revoke("primary")
    assert ledger.selected_rollout_ids == ("survivor",)
    assert ledger.selected_memory_ids == ("memory-survivor",)
    assert "memory-primary" in ledger.revoked_memory_ids


def test_recovery_strategy_competition_checkpoint_preserves_policy() -> None:
    ledger = RecoveryStrategyLedger(
        evidence_threshold=3,
        memory_budget=0.8,
        evidence_weight=0.4,
        consistency_weight=0.4,
        resource_weight=0.2,
    ).admit(
        replace(
            _entry(
                "strategy",
                resource_cost=0.2,
                evidence_count=3,
                outcome_consistency=0.8,
            ),
            outcome_reward=0.7,
        ),
        memory_id="memory-strategy",
    )

    restored = RecoveryStrategyLedger.from_payload(ledger.to_payload())
    assert restored == ledger
    assert restored.selected_rollout_ids == ("strategy",)
    assert restored.memory_budget == 0.8
    assert restored.evidence_weight == 0.4
    assert restored.consistency_weight == 0.4
    assert restored.resource_weight == 0.2


def test_recovery_reader_dependencies_retain_unaffected_reader() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=1.0)
    primary = ledger.admit(
        _entry(
            "primary",
            resource_cost=0.2,
            evidence_count=2,
            outcome_consistency=1.0,
        ),
        memory_id="memory-primary",
    )
    survivor = primary.admit(
        _entry(
            "survivor",
            resource_cost=0.2,
            evidence_count=2,
            outcome_consistency=1.0,
        ),
        memory_id="memory-survivor",
    )
    graph = RecoveryReaderDependencyGraph().bind("semantic", survivor.selected_approvals)
    graph = graph.bind("audit", (survivor.active_approvals()[1],))

    revoked = survivor.revoke("primary")
    updated = graph.retain_selected(revoked.selected_approvals)

    assert updated.dependency_for("semantic") is not None
    assert updated.dependency_for("semantic").strategy_rollout_ids == ("survivor",)
    assert updated.dependency_for("audit") is not None
    assert updated.dependency_for("audit").strategy_rollout_ids == ("survivor",)
    assert "audit" not in graph.reader_kinds_for_rollout("primary")


def test_recovery_reader_contribution_roundtrip_and_revocation() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=1.0)
    admitted = ledger.admit(
        _entry(
            "primary",
            resource_cost=0.2,
            evidence_count=2,
            outcome_consistency=1.0,
        ),
        memory_id="memory-primary",
    )
    admitted = admitted.admit(
        _entry(
            "survivor",
            resource_cost=0.2,
            evidence_count=2,
            outcome_consistency=1.0,
        ),
        memory_id="memory-survivor",
    )
    selected = admitted.selected_approvals
    contributions = tuple(
        RecoveryReaderContribution(
            reader_kind="semantic",
            strategy_rollout_id=approval.rollout_id,
            memory_id=approval.memory_id,
            effect_delta_l2=float(index + 1),
            credit=float(index + 1) / sum(range(1, len(selected) + 1)),
            replay_epochs=3,
            replay_learning_rate=0.05,
        )
        for index, approval in enumerate(selected)
    )
    graph = RecoveryReaderDependencyGraph().bind(
        "semantic",
        selected,
        contributions=contributions,
        base_checkpoint={"value": 1},
        base_checkpoint_digest="baseline-digest",
    )
    restored = RecoveryReaderDependencyGraph.from_payload(graph.to_payload())
    assert restored == graph
    assert restored.dependency_for("semantic") is not None
    assert len(restored.dependency_for("semantic").contributions) == len(selected)

    revoked = admitted.revoke("primary")
    updated = restored.retain_selected(revoked.selected_approvals)
    dependency = updated.dependency_for("semantic")
    assert dependency is not None
    assert dependency.strategy_rollout_ids == ("survivor",)
    assert dependency.contributions[0].strategy_rollout_id == "survivor"
    assert dependency.base_checkpoint_digest == "baseline-digest"


def test_recovery_reader_interaction_roundtrip_and_revocation() -> None:
    ledger = RecoveryStrategyLedger(memory_budget=1.0)
    admitted = ledger.admit(
        _entry("primary", resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
        memory_id="memory-primary",
    )
    admitted = admitted.admit(
        _entry("survivor", resource_cost=0.2, evidence_count=2, outcome_consistency=1.0),
        memory_id="memory-survivor",
    )
    selected = admitted.selected_approvals
    interaction = RecoveryReaderInteraction(
        reader_kind="semantic",
        strategy_rollout_ids=("primary", "survivor"),
        memory_ids=("memory-primary", "memory-survivor"),
        pair_effect_l2=1.5,
        additive_effect_l2=1.0,
        interaction_delta_l2=0.5,
        interaction_residual_l2=0.5,
        order_delta_l2=0.0,
        order_invariant=True,
        replay_epochs=3,
        replay_learning_rate=0.05,
    )
    policy_ledger = admitted.record_interaction_audit((interaction,), audit_available=True)
    assert policy_ledger.selected_rollout_ids == ("primary", "survivor")
    assert RecoveryStrategyLedger.from_payload(policy_ledger.to_payload()) == policy_ledger
    graph = RecoveryReaderDependencyGraph().bind(
        "semantic",
        selected,
        interactions=(interaction,),
        base_checkpoint={"value": 1},
        base_checkpoint_digest="baseline-digest",
    )
    restored = RecoveryReaderDependencyGraph.from_payload(graph.to_payload())
    assert restored == graph
    dependency = restored.dependency_for("semantic")
    assert dependency is not None
    assert dependency.interactions == (interaction,)
    assert dependency.interaction_audit_complete is True

    revoked = admitted.revoke("primary")
    updated = restored.retain_selected(revoked.selected_approvals)
    dependency = updated.dependency_for("semantic")
    assert dependency is not None
    assert dependency.interactions == ()
    assert dependency.interaction_audit_complete is True
