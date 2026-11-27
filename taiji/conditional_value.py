"""Conditional Value Head & Bounded Subgoal Proposal (`taiji-conditional-value-v1`, VISION §3.2/VX2).

Implements explicit state x candidate Hadamard interaction `MLP([z_a, z_s, z_a * P(z_s)])`
so different task/goal contexts flip relative action preferences instead of cancelling out
as a shared state bias, plus hard budget/completion wait guards and evidence-grounded subgoals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

SCHEMA = "taiji-conditional-value-v1"

CANONICAL_ACTIONS: tuple[str, ...] = (
    "continue_task",
    "verify_evidence",
    "explore_novel",
    "replay_rest",
    "express_finish",
    "ask_wait",
)


@dataclass(frozen=True)
class TaskDecisionContext:
    goal_id: str
    goal_active: bool
    goal_completed: bool
    budget_remaining: float
    uncertainty: float
    state_vector: torch.Tensor
    policy_version: str = SCHEMA


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    action_kind: str
    cost: float
    precondition_met: bool
    feature_vector: torch.Tensor
    parent_goal_id: str
    reason: str = ""


@dataclass(frozen=True)
class SubgoalProposal:
    subgoal_id: str
    parent_goal_id: str
    evidence_ref: str
    expected_gain: float
    estimated_cost: float
    stop_condition: str


@dataclass(frozen=True)
class DecisionTrace:
    schema: str
    decision_id: str
    selected_candidate_id: str
    selected_action: str
    scores: dict[str, float]
    uncertainties: dict[str, float]
    policy_version: str
    waited_due_to_guard: bool
    reason: str


class ConditionalValueHead(nn.Module):
    """State-conditioned bilinear/Hadamard value & uncertainty estimator (VISION §3.2)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)

        self.state_proj = nn.Linear(self.state_dim, self.action_dim, bias=True)
        in_dim = self.action_dim + self.state_dim + self.action_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 2),  # [value, log_uncertainty]
        )

    def forward_pair(
        self, state_vec: torch.Tensor, action_vecs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score `(K, action_dim)` candidates against a single `(state_dim,)` vector."""
        if state_vec.ndim != 1 or state_vec.shape[0] != self.state_dim:
            raise ValueError(f"Expected state_vec shape ({self.state_dim},)")
        if action_vecs.ndim != 2 or action_vecs.shape[1] != self.action_dim:
            raise ValueError(f"Expected action_vecs shape (K, {self.action_dim})")

        k = action_vecs.shape[0]
        z_s = state_vec.unsqueeze(0).expand(k, -1)
        p_s = torch.tanh(self.state_proj(z_s))
        interaction = action_vecs * p_s
        joint = torch.cat([action_vecs, z_s, interaction], dim=-1)
        out = self.mlp(joint)
        values = out[:, 0]
        uncertainties = torch.nn.functional.softplus(out[:, 1])
        return values, uncertainties

    @torch.no_grad()
    def score_candidates(
        self,
        ctx: TaskDecisionContext,
        candidates: Sequence[ActionCandidate],
        *,
        decision_id: str = "dec_0",
    ) -> DecisionTrace:
        """Select highest-value legal candidate or fail-safe to `ask_wait`."""
        if not candidates:
            return DecisionTrace(
                schema=SCHEMA,
                decision_id=decision_id,
                selected_candidate_id="guard_wait",
                selected_action="ask_wait",
                scores={},
                uncertainties={},
                policy_version=ctx.policy_version,
                waited_due_to_guard=True,
                reason="no_candidates",
            )

        if not ctx.goal_active or ctx.goal_completed:
            wait_cand = next((c for c in candidates if c.action_kind == "ask_wait"), None)
            cid = wait_cand.candidate_id if wait_cand else "guard_wait"
            return DecisionTrace(
                schema=SCHEMA,
                decision_id=decision_id,
                selected_candidate_id=cid,
                selected_action="ask_wait",
                scores={},
                uncertainties={},
                policy_version=ctx.policy_version,
                waited_due_to_guard=True,
                reason="goal_inactive_or_completed",
            )

        legal = [
            c
            for c in candidates
            if c.precondition_met
            and c.cost <= ctx.budget_remaining
            and c.action_kind in CANONICAL_ACTIONS
        ]
        if not legal:
            return DecisionTrace(
                schema=SCHEMA,
                decision_id=decision_id,
                selected_candidate_id="guard_wait",
                selected_action="ask_wait",
                scores={},
                uncertainties={},
                policy_version=ctx.policy_version,
                waited_due_to_guard=True,
                reason="no_legal_candidate_within_budget",
            )

        a_mat = torch.stack([c.feature_vector for c in legal], dim=0)
        vals, uncs = self.forward_pair(ctx.state_vector, a_mat)
        scores = {c.candidate_id: float(vals[i].item()) for i, c in enumerate(legal)}
        unc_map = {c.candidate_id: float(uncs[i].item()) for i, c in enumerate(legal)}
        best_idx = int(torch.argmax(vals).item())
        chosen = legal[best_idx]
        return DecisionTrace(
            schema=SCHEMA,
            decision_id=decision_id,
            selected_candidate_id=chosen.candidate_id,
            selected_action=chosen.action_kind,
            scores=scores,
            uncertainties=unc_map,
            policy_version=ctx.policy_version,
            waited_due_to_guard=False,
            reason=chosen.reason or "max_conditional_value",
        )

    def propose_subgoals(
        self,
        ctx: TaskDecisionContext,
        *,
        gap_evidence: Sequence[dict[str, Any]] = (),
        max_subgoals: int = 2,
    ) -> list[SubgoalProposal]:
        """Propose evidence-backed bounded subgoals tied strictly to an active parent goal."""
        if not ctx.goal_active or ctx.goal_completed or ctx.budget_remaining <= 0.0:
            return []
        proposals: list[SubgoalProposal] = []
        remaining = float(ctx.budget_remaining)
        for idx, gap in enumerate(gap_evidence):
            if len(proposals) >= max_subgoals:
                break
            cost = float(gap.get("estimated_cost", 1.0))
            gain = float(gap.get("expected_gain", 0.0))
            ev_ref = str(gap.get("evidence_ref", "")).strip()
            if not ev_ref or gain <= 0.0 or cost <= 0.0 or cost > remaining:
                continue
            proposals.append(
                SubgoalProposal(
                    subgoal_id=f"{ctx.goal_id}/sub_{idx}",
                    parent_goal_id=ctx.goal_id,
                    evidence_ref=ev_ref,
                    expected_gain=gain,
                    estimated_cost=cost,
                    stop_condition=str(gap.get("stop_condition", "verified_outcome_received")),
                )
            )
            remaining -= cost
        return proposals
