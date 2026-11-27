"""Unified 态极 Native Cognitive Runtime (`taiji-native-runtime-v1`).

Wires the authoritative native modules into a single persistent, fail-closed,
crash-recoverable runtime without touching legacy closed-vocab bundles:
  * `TaijiNativeCognitiveBackbone` (pure UTF-8 byte-group dynamics, FP1 KKT, SVD birth, U2 §7 copy)
  * `NativeGrowthController` (pressure-triggered, pre-birth matured, shadow-verified growth)
  * `DynamicByteChunkEncoder` + `EntropyBoundaryChunker` (causal compute-allocation signal)
  * `ConditionalValueHead` (state x candidate Hadamard interaction + wait guards)
  * `ActionLifecycleLedger` (9-state fail-closed external action & 4-boundary crash recovery)
  * `OutcomeRouter` (5-plane isolation & provenance/authorization/retention gates)
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from taiji.action_lifecycle import ActionLifecycleLedger
from taiji.conditional_value import (
    ActionCandidate,
    ConditionalValueHead,
    DecisionTrace,
    TaskDecisionContext,
)
from taiji.development_learning import (
    LearningPermissionPolicy,
    OutcomeRouter,
    VerifiedOutcomeEvent,
)
from taiji.dynamic_byte_chunker import (
    BOUNDARY_ROLE,
    DynamicByteChunkEncoder,
    EntropyBoundaryChunker,
)
from taiji.native_cognitive_backbone import (
    BOS_BYTE,
    NativeBackboneConfig,
    NativeSubjectState,
    StatefulNeuralGroup,
    TaijiNativeCognitiveBackbone,
)
from taiji.native_growth_controller import (
    NativeGrowthController,
    NativeGrowthPolicy,
)

SCHEMA = "taiji-native-runtime-v1"


class TaijiNativeRuntime:
    """Unified persistent runtime over `TaijiNativeCognitiveBackbone` and governance planes."""

    def __init__(
        self,
        backbone: TaijiNativeCognitiveBackbone,
        *,
        state_dir: str | Path | None = None,
        growth_policy: NativeGrowthPolicy | None = None,
        learning_policy: LearningPermissionPolicy | None = None,
        boundary_entropy_threshold: float = 2.0,
    ) -> None:
        self.backbone = backbone
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self.subject_state: NativeSubjectState = backbone.init_state(1)
        self.growth_controller = NativeGrowthController(growth_policy or NativeGrowthPolicy())
        self.chunk_encoder = DynamicByteChunkEncoder(
            backbone.cfg.d_model, backbone.cfg.byte_emb_dim
        )
        self.chunker = EntropyBoundaryChunker(entropy_threshold=boundary_entropy_threshold)
        self.value_head = ConditionalValueHead(
            state_dim=backbone.cfg.d_model, action_dim=backbone.cfg.d_model
        )
        ledger_path = (
            (self.state_dir / "action_ledger.json") if self.state_dir is not None else None
        )
        self.action_ledger = (
            ActionLifecycleLedger.open(ledger_path)
            if ledger_path is not None and ledger_path.exists()
            else ActionLifecycleLedger(ledger_path)
        )
        self.outcome_router = OutcomeRouter(learning_policy)

    @classmethod
    def from_preset(
        cls,
        cfg: NativeBackboneConfig,
        *,
        state_dir: str | Path | None = None,
        growth_policy: NativeGrowthPolicy | None = None,
        learning_policy: LearningPermissionPolicy | None = None,
    ) -> TaijiNativeRuntime:
        return cls(
            TaijiNativeCognitiveBackbone(cfg),
            state_dir=state_dir,
            growth_policy=growth_policy,
            learning_policy=learning_policy,
        )

    @torch.no_grad()
    def _prompt_entropy_stats(self, prompt: str) -> dict[str, Any]:
        raw = list(prompt.encode("utf-8")) or [BOS_BYTE]
        dev = self.backbone.encoder.byte_emb.weight.device
        x = torch.tensor([raw[-self.backbone.cfg.max_seq_bytes :]], dtype=torch.long, device=dev)
        self.backbone.eval()
        logits, _, _ = self.backbone(x, state=self.subject_state, p_allowed=0.0)
        probs = F.softmax(logits[0].float(), dim=-1).clamp(min=1e-9)
        ent = -(probs * probs.log()).sum(dim=-1)
        spans = self.chunker.segment([float(v) for v in ent.tolist()])
        lengths = [end - start for start, end in spans]
        return {
            "boundary_role": BOUNDARY_ROLE,
            "n_bytes": len(raw),
            "n_chunks": len(spans),
            "mean_chunk_len": round(sum(lengths) / max(1, len(lengths)), 3),
            "boundary_ratio": round(len(spans) / max(1, len(raw)), 4),
        }

    @torch.no_grad()
    def observe_and_respond(
        self,
        prompt: str,
        *,
        experience_records: Sequence[bytes | str] | None = None,
        max_new_bytes: int = 64,
        temperature: float = 0.8,
        revoked_indices: Sequence[int] = (),
    ) -> dict[str, Any]:
        """Run native UTF-8 rollout with causal chunk diagnostics and optional U2 §7 copy head."""
        boundary_stats = self._prompt_entropy_stats(prompt)
        copy_meta: dict[str, Any] | None = None
        if experience_records:
            text, next_state, copy_meta = self.backbone.generate_utf8_with_records(
                prompt,
                experience_records,
                max_new_bytes=max_new_bytes,
                state=self.subject_state,
                temperature=temperature,
                revoked_indices=revoked_indices,
            )
            used_copy = True
        else:
            text, next_state = self.backbone.generate_utf8(
                prompt,
                max_new_bytes=max_new_bytes,
                state=self.subject_state,
                temperature=temperature,
            )
            used_copy = False
        self.subject_state = next_state.detach()
        return {
            "schema": SCHEMA,
            "text": text,
            "used_copy_head": used_copy,
            "copy_meta": copy_meta,
            "boundary_stats": boundary_stats,
            "topology": self.backbone.topology_summary(),
        }

    def current_state_vector(self) -> torch.Tensor:
        return torch.stack(self.subject_state.h_states, dim=0).mean(dim=0)[0].detach().float()

    def evaluate_goal_action(
        self,
        candidates: Sequence[ActionCandidate],
        *,
        ctx: TaskDecisionContext | None = None,
        goal_id: str = "active_goal",
        decision_id: str = "dec_0",
        wait_budget_ok: bool = True,
        goal_active: bool = True,
        goal_completed: bool = False,
        budget_remaining: float = 5.0,
    ) -> DecisionTrace:
        """Score action candidates via `ConditionalValueHead` against live subject state."""
        decision_ctx = ctx or TaskDecisionContext(
            goal_id=goal_id,
            goal_active=goal_active,
            goal_completed=goal_completed,
            budget_remaining=float(budget_remaining) if wait_budget_ok else 0.0,
            uncertainty=0.0,
            state_vector=self.current_state_vector(),
        )
        return self.value_head.score_candidates(decision_ctx, candidates, decision_id=decision_id)

    def propose_and_execute_action(
        self,
        action_id: str,
        *,
        idempotency_key: str,
        intent_digest: str,
        executor_fn: Callable[[], dict[str, Any]],
        goal_revision: int = 1,
        belief_revision: int = 1,
        reward: float = 0.0,
        prediction_error: float = 0.0,
        cost: float = 0.0,
        eval_old_fn: Callable[[], float] | None = None,
        eval_new_fn: Callable[[], float] | None = None,
        apply_learning_fn: Callable[[list[str]], Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an external action through 9-state ledger and route verified learning."""
        rec = self.action_ledger.propose(
            action_id,
            idempotency_key=idempotency_key,
            intent_digest=intent_digest,
            goal_revision=goal_revision,
            belief_revision=belief_revision,
        )
        if rec.settled_once:
            return {
                "schema": SCHEMA,
                "action_id": rec.action_id,
                "state": rec.state,
                "idempotent_replay": True,
                "outcome_payload": rec.outcome_payload,
                "learning_receipt": None,
            }

        if rec.state == "proposed":
            rec = self.action_ledger.prepare(rec.action_id)
        if rec.state == "prepared":
            rec = self.action_ledger.dispatch(rec.action_id)

        try:
            payload = dict(executor_fn())
        except Exception as exc:
            self.action_ledger.mark_unknown(rec.action_id, reason=f"dispatch_error:{exc}")
            return {
                "schema": SCHEMA,
                "action_id": rec.action_id,
                "state": "unknown",
                "error": str(exc),
                "learning_receipt": None,
            }

        receipt_id = str(payload.get("receipt_id", f"rc_{rec.action_id}"))
        self.action_ledger.observe(rec.action_id, receipt_id=receipt_id, outcome_payload=payload)
        self.action_ledger.settle(rec.action_id)

        event = VerifiedOutcomeEvent(
            event_id=f"ext:{rec.action_id}:{receipt_id}",
            action_ref=rec.action_id,
            provenance="real",
            authorized=bool(payload.get("authorized", True)),
            reward=float(reward),
            prediction_error=float(prediction_error),
            cost=float(cost),
        )
        learning_receipt = self.outcome_router.route_event(
            event,
            apply_fn=apply_learning_fn or (lambda _planes: None),
            eval_old_fn=eval_old_fn or (lambda: 1.0),
            eval_new_fn=eval_new_fn or (lambda: 1.0 + max(0.0, float(reward))),
        )
        return {
            "schema": SCHEMA,
            "action_id": rec.action_id,
            "state": self.action_ledger.records[rec.action_id].state,
            "idempotent_replay": False,
            "outcome_payload": payload,
            "learning_receipt": learning_receipt,
        }

    def _sync_subject_state_after_growth(self, new_backbone: TaijiNativeCognitiveBackbone) -> None:
        bsz = self.subject_state.h_states[0].shape[0]
        fresh = new_backbone.init_state(bsz)
        for i, h in enumerate(self.subject_state.h_states):
            if i < len(fresh.h_states):
                fresh.h_states[i].copy_(h.to(dtype=fresh.h_states[i].dtype))
        for i, f in enumerate(self.subject_state.fast_memories):
            if i < len(fresh.fast_memories):
                fresh.fast_memories[i].copy_(f.to(dtype=fresh.fast_memories[i].dtype))
        for i, grp_m in enumerate(new_backbone.groups):
            grp = grp_m
            if isinstance(grp, StatefulNeuralGroup) and i < len(self.subject_state.unit_z):
                old_z = self.subject_state.unit_z[i]
                old_a = self.subject_state.unit_a[i]
                keep = min(int(old_z.shape[1]), int(fresh.unit_z[i].shape[1]))
                fresh.unit_z[i][:, :keep].copy_(old_z[:, :keep].to(dtype=fresh.unit_z[i].dtype))
                fresh.unit_a[i][:, :keep].copy_(old_a[:, :keep].to(dtype=fresh.unit_a[i].dtype))
        self.subject_state = fresh.detach()

    def governed_growth_step(
        self,
        val_loss: float,
        step: int,
        *,
        adapt_fn: Callable[[TaijiNativeCognitiveBackbone], None],
        eval_old_fn: Callable[[TaijiNativeCognitiveBackbone], float],
        eval_new_fn: Callable[[TaijiNativeCognitiveBackbone], float],
        sample_input: torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        policy_path: str | Path | None = None,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Propose, shadow-verify, route via `OutcomeRouter`, and commit iff gates pass."""
        if policy_path is not None:
            self.growth_controller.refresh_policy(policy_path)
        proposal = self.growth_controller.propose_action(self.backbone, val_loss, step=step)
        old_handles = self.backbone.snapshot_param_handles() if optimizer is not None else None
        next_backbone, receipt = self.growth_controller.run_shadow_growth_cycle(
            self.backbone,
            proposal,
            adapt_fn=adapt_fn,
            eval_old_fn=eval_old_fn,
            eval_new_fn=eval_new_fn,
            sample_input=sample_input,
            outcome_router=self.outcome_router,
            seed=seed,
        )
        if receipt.get("accepted"):
            self.backbone = next_backbone
            self._sync_subject_state_after_growth(next_backbone)
            if optimizer is not None and old_handles is not None:
                receipt["optimizer_migration"] = next_backbone.migrate_adamw_state_(
                    optimizer, old_handles
                )
        else:
            self.action_ledger.record_model_rollback(
                str(receipt.get("reject_reason", "shadow_rejected"))
            )
        return receipt

    def save(self, ckpt_path: str | Path) -> Path:
        """Atomically persist backbone capsule, subject state, policy, and router state."""
        target = Path(ckpt_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "capsule": self.backbone.export_capsule(),
            "subject_state": self.subject_state.to_dict(),
            "growth_policy": self.growth_controller.policy.to_dict(),
            "settled_event_ids": sorted(self.outcome_router.settled_event_ids),
            "value_head": {
                k: v.detach().cpu().clone() for k, v in self.value_head.state_dict().items()
            },
        }
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)
        return target

    @classmethod
    def load(
        cls,
        ckpt_path: str | Path,
        *,
        state_dir: str | Path | None = None,
    ) -> TaijiNativeRuntime:
        """Restore `TaijiNativeRuntime` (`weights_only=True`) preserving identity & state."""
        payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported runtime checkpoint schema: {payload.get('schema')}")
        backbone = TaijiNativeCognitiveBackbone.from_capsule(payload["capsule"])
        policy = NativeGrowthPolicy.from_dict(payload.get("growth_policy", {}))
        rt = cls(backbone, state_dir=state_dir, growth_policy=policy)
        if isinstance(payload.get("subject_state"), dict):
            rt.subject_state = NativeSubjectState.from_dict(payload["subject_state"])
        if isinstance(payload.get("value_head"), dict):
            rt.value_head.load_state_dict(payload["value_head"], strict=True)
        rt.outcome_router.settled_event_ids.update(payload.get("settled_event_ids", []))
        return rt
