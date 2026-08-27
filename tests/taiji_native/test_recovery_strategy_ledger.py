from dataclasses import replace

from taiji import RecoveryArchiveEntry, RecoveryStrategyLedger


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
