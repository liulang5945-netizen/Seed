from dataclasses import replace

from taiji import (
    RecoveryArchiveEntry,
    RecoveryReaderContribution,
    RecoveryReaderDependencyGraph,
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
