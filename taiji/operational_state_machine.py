"""Bounded N37 Operational State Machine (`taiji-n37-ops-fsm-v1`, §15).

Layers the 8-state post-delivery operational protocol, 6-step daily shadow
upgrade loop, 3-tier change routing, Restricted health/resource guard, and
retirement archival verification atop `taiji.bounded_delivery_runtime`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from taiji.bounded_delivery_runtime import (
    BOUNDARIES,
    SCOPE,
    BoundedDeliveryRuntime,
    _atomic_write_json,
)
from taiji.credibility_learning import SOURCES as ECL_SOURCES

SCHEMA = "taiji-n37-ops-fsm-v1"

STATES: tuple[str, ...] = (
    "Qualified",
    "Trial",
    "Running",
    "Waiting",
    "Shadow",
    "Restricted",
    "Paused",
    "Archived",
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "Qualified": frozenset({"Trial"}),
    "Trial": frozenset({"Running", "Paused"}),
    "Running": frozenset({"Waiting", "Shadow", "Restricted", "Paused", "Archived"}),
    "Waiting": frozenset({"Running", "Paused"}),
    "Shadow": frozenset({"Running", "Paused"}),
    "Restricted": frozenset({"Paused", "Running"}),
    "Paused": frozenset({"Trial", "Running", "Archived"}),
    "Archived": frozenset(),
}

_TIER1_OPS = frozenset({"ops_refactor", "logging", "cli"})
_TIER2_SHADOW = frozenset(
    {
        "ecl_calibration",
        "unc_recalibration",
        "phy_bounded_step",
        "growth_proposal",
        "harmful_phy_corrupt",
    }
)
_TIER3_CONTRACT: dict[str, list[str]] = {
    "readout_class": ["N31", "N32", "N35", "N36"],
    "objective_change": ["N32", "N33", "N35", "N36"],
    "dynamics_rule": ["N34", "N35", "N36"],
    "modality_change": ["N32", "N34", "N35", "N36"],
    "rights_boundary": ["N35", "N36", "N37"],
}


class OperationalTransitionError(RuntimeError):
    """Raised when an FSM transition violates the §15 state diagram."""


class OutOfContractChangeError(RuntimeError):
    """Raised when a Tier-3 change attempts to run inside daily Shadow."""


class ShadowBudgetError(RuntimeError):
    """Raised when shadow upgrade budget is exhausted."""


def classify_change(change_kind: str) -> dict[str, Any]:
    """Route a proposed change into Tier-1 ops, Tier-2 Shadow, or Tier-3 research reopen."""
    if change_kind in _TIER1_OPS:
        return {
            "change_kind": change_kind,
            "tier": "tier1_ops_only",
            "route": "engineering_regression",
            "allowed_in_shadow": False,
            "reopen_research_required": False,
            "impacted_stages": [],
        }
    if change_kind in _TIER2_SHADOW:
        return {
            "change_kind": change_kind,
            "tier": "tier2_in_contract_shadow",
            "route": "daily_shadow",
            "allowed_in_shadow": True,
            "reopen_research_required": False,
            "impacted_stages": [],
        }
    if change_kind in _TIER3_CONTRACT:
        return {
            "change_kind": change_kind,
            "tier": "tier3_contract_or_architecture",
            "route": "reopen_research_required",
            "allowed_in_shadow": False,
            "reopen_research_required": True,
            "impacted_stages": list(_TIER3_CONTRACT[change_kind]),
        }
    raise OutOfContractChangeError(
        f"reopen_research_required: unknown/out-of-contract change_kind {change_kind!r}"
    )


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


def _evaluate_retention_and_ecl(rt: BoundedDeliveryRuntime) -> dict[str, float]:
    """Compute bounded retention probes (phy + ch) and ecl forum posterior."""
    phy_res = rt.predict_physics("rough_collision", seed=101)
    phy_recall = float(phy_res["metrics"]["collision_recall"])
    ch_res = rt.interact_chinese(["请把红色杯子放到桌上。", "再去检查一下它。"], persist=False)
    ch_conf = sum(float(t["confidence"]) for t in ch_res["turns"]) / max(1, len(ch_res["turns"]))
    alpha = rt.record.capsule["ecl.alpha"]
    beta = rt.record.capsule["ecl.beta"]
    f_idx = ECL_SOURCES.index("forum")
    forum_post = float(alpha[f_idx] / (alpha[f_idx] + beta[f_idx]))
    return {
        "phy_collision_recall": phy_recall,
        "ch_mean_confidence": ch_conf,
        "ecl_forum_posterior": forum_post,
    }


def _capsule_all_finite(rt: BoundedDeliveryRuntime) -> bool:
    for val in rt.record.capsule.values():
        if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
            return False
    return True


class OperationalStateMachine:
    """§15 Operational State Machine layered atop BoundedDeliveryRuntime."""

    def __init__(
        self, bundle_dir: Path, rt: BoundedDeliveryRuntime, ops_state: dict[str, Any]
    ) -> None:
        self.bundle_dir = bundle_dir
        self.rt = rt
        self.ops_state = ops_state

    @property
    def state(self) -> str:
        return str(self.ops_state["state"])

    @classmethod
    def initialize(
        cls,
        bundle_dir: str | Path,
        *,
        initial_state: str = "Qualified",
        shadow_budget: int = 4,
        max_consecutive_rejects: int = 2,
    ) -> OperationalStateMachine:
        if initial_state not in STATES:
            raise OperationalTransitionError(f"invalid initial_state {initial_state!r}")
        root = Path(bundle_dir)
        rt = BoundedDeliveryRuntime.open(root)
        ops_state: dict[str, Any] = {
            "schema": SCHEMA,
            "scope": SCOPE,
            "state": initial_state,
            "trial_passed": initial_state == "Running",
            "shadow_budget": int(shadow_budget),
            "shadow_runs_used": 0,
            "max_consecutive_rejects": int(max_consecutive_rejects),
            "consecutive_rejects": 0,
            "restricted_reason": None,
            "archived_to": None,
        }
        _atomic_write_json(root / "ops_state.json", ops_state)
        rt._sync_manifest(lifecycle_state=initial_state)
        _append_jsonl(
            root / "ops_transition_log.jsonl",
            {
                "ts_utc": datetime.now(UTC).isoformat(),
                "from": None,
                "to": initial_state,
                "reason": "initialize_ops_fsm",
            },
        )
        return cls(root, rt, ops_state)

    @classmethod
    def open(cls, bundle_dir: str | Path) -> OperationalStateMachine:
        root = Path(bundle_dir)
        rt = BoundedDeliveryRuntime.open(root)
        spath = root / "ops_state.json"
        if not spath.exists():
            raise OperationalTransitionError(f"missing ops_state.json in {root}")
        ops_state = json.loads(spath.read_text(encoding="utf-8"))
        if ops_state.get("schema") != SCHEMA:
            raise OperationalTransitionError(f"invalid ops_state schema: {ops_state.get('schema')}")
        return cls(root, rt, ops_state)

    def _save_ops_state(self) -> None:
        _atomic_write_json(self.bundle_dir / "ops_state.json", self.ops_state)

    def transition(self, target_state: str, *, reason: str = "") -> dict[str, Any]:
        cur = self.state
        if target_state not in STATES:
            raise OperationalTransitionError(f"unknown target_state {target_state!r}")
        if target_state not in ALLOWED_TRANSITIONS[cur]:
            raise OperationalTransitionError(f"illegal transition {cur!r} -> {target_state!r}")
        if cur == "Paused" and target_state == "Running" and not self.ops_state.get("trial_passed"):
            raise OperationalTransitionError(
                "illegal transition 'Paused' -> 'Running': trial has not passed yet"
            )
        if cur == "Trial" and target_state == "Running":
            self.ops_state["trial_passed"] = True
        self.ops_state["state"] = target_state
        if target_state != "Restricted":
            self.ops_state["restricted_reason"] = None
        self._save_ops_state()
        self.rt._sync_manifest(lifecycle_state=target_state)
        entry = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "from": cur,
            "to": target_state,
            "reason": reason,
        }
        _append_jsonl(self.bundle_dir / "ops_transition_log.jsonl", entry)
        return entry

    def enter_restricted(self, reason: str) -> dict[str, Any]:
        self.ops_state["restricted_reason"] = reason
        return self.transition("Restricted", reason=reason)

    def recover_from_restricted(self, *, health_ok: bool, reason: str = "") -> dict[str, Any]:
        if self.state != "Restricted":
            raise OperationalTransitionError(
                f"recover_from_restricted requires Restricted state, current={self.state!r}"
            )
        if health_ok:
            if not self.rt.harness.verify_chain():
                raise OperationalTransitionError("cannot recover to Running: audit chain invalid")
            return self.transition("Running", reason=reason or "restricted_health_recovered")
        return self.transition("Paused", reason=reason or "restricted_cannot_continue_safely")

    def wake_event(self, event_note: str) -> dict[str, Any]:
        if self.state != "Waiting":
            raise OperationalTransitionError(
                f"wake_event requires Waiting state, current={self.state!r}"
            )
        return self.transition("Running", reason=f"wake:{event_note}")

    def run_shadow_upgrade(
        self,
        candidate_kind: str,
        *,
        min_gain: float = 0.0005,
        max_retention_drop: float = 0.02,
        external_actions_taken: list[str] | None = None,
        auto_wait_on_stop: bool = True,
    ) -> dict[str, Any]:
        """Execute §15.2 6-step daily shadow upgrade loop."""
        if self.state != "Running":
            raise OperationalTransitionError(
                f"shadow upgrade requires Running state, current={self.state!r}"
            )
        route = classify_change(candidate_kind)
        if not route["allowed_in_shadow"]:
            raise OutOfContractChangeError(
                f"reopen_research_required: {candidate_kind!r} routed to {route['route']} "
                f"(impacted={route['impacted_stages']})"
            )
        used = int(self.ops_state["shadow_runs_used"])
        budget = int(self.ops_state["shadow_budget"])
        if used >= budget:
            raise ShadowBudgetError(f"shadow budget exhausted ({used}/{budget})")

        # Step 1 & 2: Enter Shadow and construct isolated shadow copy bound to parent digest
        parent_digest = str(self.rt.manifest["record_digest"])
        parent_chain = str(self.rt.manifest["chain_head"])
        parent_metrics = _evaluate_retention_and_ecl(self.rt)
        parent_active_units = self.rt.grow_net.active_units()
        self.transition("Shadow", reason=f"shadow_start:{candidate_kind}")

        run_id = used + 1
        sid = str(self.rt.manifest["subject_id"])
        accepted = False
        reject_reason = ""
        gain = 0.0
        retention_drop = 0.0

        with tempfile.TemporaryDirectory() as tmp:
            shadow_dir = Path(tmp) / "shadow_bundle"
            shutil.copytree(self.bundle_dir, shadow_dir)
            shadow_rt = BoundedDeliveryRuntime.open(shadow_dir)
            shadow_rt._sync_manifest(lifecycle_state="Running")

            # Step 3: Execute bounded candidate update in isolated shadow
            if candidate_kind == "ecl_calibration":
                for _ in range(15):
                    shadow_rt.update_credibility("forum", 1)
            elif candidate_kind == "growth_proposal":

                def _grow_fn(rec: Any, _s: int) -> None:
                    before_units = shadow_rt.grow_net.active_units()
                    shadow_rt.grow_net.grow_neutral_(k=2, seed=run_id)
                    after_units = shadow_rt.grow_net.active_units()
                    if after_units <= before_units:
                        raise RuntimeError("growth_proposal failed to expand active units")
                    for k, v in shadow_rt.grow_net.to_session_record("grow").capsule.items():
                        if k == "meta":
                            events = int(rec.capsule["grow.meta"][4].item())
                            rec.capsule["grow.meta"] = torch.cat(
                                [v, torch.tensor([events], dtype=torch.long)]
                            )
                        else:
                            rec.capsule[f"grow.{k}"] = v

                shadow_rt.harness.run_session(_grow_fn, crash_before_commit=False)
                shadow_rt._sync_manifest()
            elif candidate_kind == "harmful_phy_corrupt":

                def _corrupt_fn(rec: Any, _s: int) -> None:
                    for k in list(rec.capsule.keys()):
                        if k.startswith("phy.") and isinstance(rec.capsule[k], torch.Tensor):
                            rec.capsule[k] = torch.zeros_like(rec.capsule[k])

                shadow_rt.harness.run_session(_corrupt_fn, crash_before_commit=False)
                shadow_rt._sync_manifest()
            else:
                shadow_rt.update_credibility("news", 1)

            shadow_rt._reload_state()
            shadow_metrics = _evaluate_retention_and_ecl(shadow_rt)
            phy_drop = max(
                0.0, parent_metrics["phy_collision_recall"] - shadow_metrics["phy_collision_recall"]
            )
            ch_drop = max(
                0.0, parent_metrics["ch_mean_confidence"] - shadow_metrics["ch_mean_confidence"]
            )
            retention_drop = round(max(phy_drop, ch_drop), 6)
            if candidate_kind == "growth_proposal":
                gain = round(
                    max(0, shadow_rt.grow_net.active_units() - parent_active_units) * 0.01,
                    6,
                )
            else:
                gain = round(
                    shadow_metrics["ecl_forum_posterior"] - parent_metrics["ecl_forum_posterior"],
                    6,
                )
            finite_ok = _capsule_all_finite(shadow_rt)
            chain_ok = shadow_rt.harness.verify_chain()

            if not finite_ok or not chain_ok:
                reject_reason = "integrity_or_nonfinite_failure"
            elif retention_drop > max_retention_drop:
                reject_reason = f"retention_breach:{retention_drop:.4f}>{max_retention_drop:.4f}"
            elif gain < min_gain:
                reject_reason = f"insufficient_gain:{gain:.4f}<{min_gain:.4f}"
            else:
                accepted = True

            # Step 4: Single-writer atomic commit OR clean reject leaving parent untouched
            self.ops_state["shadow_runs_used"] = run_id
            if accepted:
                self.rt.snapshot(f"pre_shadow_{run_id}")
                for fname in (f"{sid}.session", f"{sid}.audit.jsonl", "manifest.json"):
                    tmp_dst = self.bundle_dir / f".{fname}.{os.getpid()}.tmp"
                    shutil.copy2(shadow_dir / fname, tmp_dst)
                    os.replace(tmp_dst, self.bundle_dir / fname)
                self.rt._reload_state()
                self.ops_state["consecutive_rejects"] = 0
                self.transition("Running", reason=f"shadow_commit:{candidate_kind}")
            else:
                self.ops_state["consecutive_rejects"] = (
                    int(self.ops_state["consecutive_rejects"]) + 1
                )
                self.transition("Running", reason=f"shadow_reject:{reject_reason}")

        # Step 5: Recalibrate isolated consequence ledger
        ext_actions = list(external_actions_taken or [])
        ledger_entry = {
            "ts_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "candidate_kind": candidate_kind,
            "parent_digest": parent_digest,
            "parent_chain": parent_chain,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "gain": gain,
            "retention_drop": retention_drop,
            "external_actions_not_pseudo_rolled_back": ext_actions,
            "result_digest": str(self.rt.manifest["record_digest"]),
        }
        _append_jsonl(self.bundle_dir / "ops_consequence_ledger.jsonl", ledger_entry)

        # Step 6: Auto-enter Waiting when no beneficial activity / stop window triggers
        auto_waited = False
        if (
            auto_wait_on_stop
            and not accepted
            and (
                int(self.ops_state["consecutive_rejects"])
                >= int(self.ops_state["max_consecutive_rejects"])
                or int(self.ops_state["shadow_runs_used"]) >= int(self.ops_state["shadow_budget"])
            )
        ):
            self.transition("Waiting", reason="shadow_stop_window_to_waiting")
            auto_waited = True

        return {
            "run_id": run_id,
            "candidate_kind": candidate_kind,
            "accepted": accepted,
            "reject_reason": reject_reason,
            "gain": gain,
            "retention_drop": retention_drop,
            "state": self.state,
            "auto_waited": auto_waited,
            "record_digest": self.rt.manifest["record_digest"],
            "chain_head": self.rt.manifest["chain_head"],
        }

    def retire_and_archive(
        self,
        archive_dir: str | Path,
        *,
        reason: str = "explicit_retirement",
    ) -> dict[str, Any]:
        """§15.3: Settle/export pending items, verify archive readable, then revoke write rights."""
        if self.state not in ("Running", "Paused"):
            raise OperationalTransitionError(
                f"retire_and_archive requires Running or Paused state, current={self.state!r}"
            )
        pending_items = list(self.rt.record.pending)
        if self.state == "Running" and pending_items:
            for item in pending_items:
                self.rt.resolve_pending(str(item["id"]), "settled_at_retirement")

        dst = Path(archive_dir)
        dst.mkdir(parents=True, exist_ok=True)
        sid = str(self.rt.manifest["subject_id"])
        for fname in (
            f"{sid}.session",
            f"{sid}.audit.jsonl",
            "manifest.json",
            "ops_state.json",
            "ops_transition_log.jsonl",
            "ops_consequence_ledger.jsonl",
        ):
            src_f = self.bundle_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, dst / fname)

        verified_rt = BoundedDeliveryRuntime.open(dst)
        if verified_rt.manifest["record_digest"] != self.rt.manifest["record_digest"]:
            raise OperationalTransitionError("archive verification failed: record_digest mismatch")
        if verified_rt.manifest["chain_head"] != self.rt.manifest["chain_head"]:
            raise OperationalTransitionError("archive verification failed: chain_head mismatch")

        _atomic_write_json(
            dst / "archive_receipt.json",
            {
                "schema": SCHEMA,
                "scope": SCOPE,
                "reason": reason,
                "pending_settled_or_exported": pending_items,
                "record_digest": verified_rt.manifest["record_digest"],
                "chain_head": verified_rt.manifest["chain_head"],
                "verified_readable": True,
            },
        )

        self.ops_state["archived_to"] = str(dst)
        self.transition("Archived", reason=reason)
        return {
            "state": self.state,
            "archive_dir": str(dst),
            "verified_readable": True,
            "pending_count_exported": len(pending_items),
            "record_digest": self.rt.manifest["record_digest"],
        }

    def status(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scope": SCOPE,
            "boundaries": dict(BOUNDARIES),
            "state": self.state,
            "trial_passed": bool(self.ops_state.get("trial_passed")),
            "shadow_runs_used": int(self.ops_state["shadow_runs_used"]),
            "shadow_budget": int(self.ops_state["shadow_budget"]),
            "consecutive_rejects": int(self.ops_state["consecutive_rejects"]),
            "restricted_reason": self.ops_state.get("restricted_reason"),
            "archived_to": self.ops_state.get("archived_to"),
            "transitions_count": _count_jsonl(self.bundle_dir / "ops_transition_log.jsonl"),
            "consequence_ledger_count": _count_jsonl(
                self.bundle_dir / "ops_consequence_ledger.jsonl"
            ),
            "runtime": self.rt.status(),
        }
