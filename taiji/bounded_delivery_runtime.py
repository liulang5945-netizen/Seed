"""Bounded N36 delivery & user-takeover runtime (`taiji-n36-bounded-delivery-v1`).

Packages the confirmed bounded capabilities (`N34.5.90` + `N34.9.90` + `N35.1.91` 8/9)
into a single user-operable, fail-closed, auditable, snapshot/rollback-capable runtime:
1. `ch.*`: `ChineseTurnPolicy(recurrent=True)` over closed-vocab `taiji.chinese_task_env_v2`
   (refuses out-of-bounded-vocab input fail-closed rather than faking open-vocab chat);
2. `phy.*`: `ConsequenceNetV2` over two synthetic physics families (`ramp_bumper` /
   `rough_collision`) with optional barrier ablation;
3. `grow.*`: `PlasticGrowthNet` + `GrowthPressureController` capsule state;
4. `ecl.*`: `BetaCredibility(forgetting=0.995)` online source credibility tracking;
5. `unc.*`: `LearnedConfidence` calibrated confidence + `VERBAL_LEVELS` Chinese verbalisation;
6. `SessionStore` + `LongRunHarness` sha256 audit chain + `SealedSplitGuard` (`0 check/final`).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from taiji.chinese_task_env_v2 import ACTIONS as CH_ACTIONS
from taiji.chinese_task_env_v2 import MAX_TOKENS, PAD_ID, VOCAB, VOCAB_TO_ID
from taiji.chinese_turn_policy import ChineseTurnPolicy
from taiji.credibility_learning import SOURCES as ECL_SOURCES
from taiji.long_run_harness import LongRunHarness
from taiji.long_run_trained_subject import (
    RunConfig,
    _confidence_features,
    _load_prefixed_state,
    _unpack_growth_net,
    run_long_run_subject,
)
from taiji.multimodal_physics_env_v2 import (
    OUTCOMES as PHY_OUTCOMES,
)
from taiji.multimodal_physics_env_v2 import (
    ConsequenceNetV2,
    ablate_barrier_collision_recall,
    collect_v2,
    eval_metrics_v2,
)
from taiji.ops_contract import BOUNDARIES, SCOPE
from taiji.plastic_growth import GrowthPressureController, PlasticGrowthNet
from taiji.session_store import SessionRecord
from taiji.uncertainty_calibration import VERBAL_LEVELS

SCHEMA = "taiji-n36-bounded-delivery-v1"
CONFIRMED_EVIDENCE_REFS: tuple[str, ...] = (
    "N34.5.90 (confirm_positive)",
    "N34.9.90 (confirm_all_passed)",
    "N35.1.91 (confirm_partial 8/9)",
    "N33.2.90 / N33.3.91 / N33.4.90",
)


class BundleIntegrityError(RuntimeError):
    """Raised when a delivery bundle is missing, tampered, or fails sha256 verification."""


class LifecycleError(RuntimeError):
    """Raised when an operation is invalid for the current lifecycle state."""


def tokenize_closed_chinese(text: str) -> list[int]:
    """Tokenize closed-vocab Chinese text greedily; raise ValueError on any OOV char."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("out_of_bounded_vocab: empty utterance")
    tokens_sorted = sorted(VOCAB[1:], key=len, reverse=True)
    ids: list[int] = []
    pos = 0
    while pos < len(cleaned):
        if cleaned[pos].isspace():
            pos += 1
            continue
        matched = None
        for tok in tokens_sorted:
            if cleaned.startswith(tok, pos):
                matched = tok
                break
        if matched is None:
            raise ValueError(
                f"out_of_bounded_vocab: unsupported character {cleaned[pos]!r} at pos={pos}"
            )
        ids.append(VOCAB_TO_ID[matched])
        pos += len(matched)
    if not ids:
        raise ValueError("out_of_bounded_vocab: no valid tokens")
    return ids[:MAX_TOKENS]


def verbalize_confidence(prob: float) -> str:
    """Map calibrated probability to canonical Chinese confidence level."""
    p = max(0.0, min(1.0, float(prob)))
    for thresh, label in VERBAL_LEVELS:
        if p >= thresh:
            return label
    return VERBAL_LEVELS[-1][1]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def build_delivery_bundle(
    bundle_dir: str | Path,
    *,
    subject_id: str = "taiji-n36-bounded",
    seed: int = 6801,
    cfg: RunConfig | None = None,
    cap_cpu: float = 90.0,
) -> dict[str, Any]:
    """Train and export a self-contained bounded N36 delivery bundle."""
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    deadline = time.process_time() + float(cap_cpu)
    summary = run_long_run_subject(root, int(seed), deadline, cfg=cfg, subject_id=subject_id)

    harness = LongRunHarness(root, subject_id)
    record = harness.store.load(subject_id)
    param_count = sum(
        int(t.numel())
        for k, t in record.capsule.items()
        if k.startswith(("ch.", "phy.", "grow.fc_"))
    )
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "scope": SCOPE,
        "subject_id": subject_id,
        "seed": int(seed),
        "lifecycle_state": "Running",
        "record_digest": record.digest(),
        "chain_head": harness.chain[-1],
        "sessions": harness.session_index,
        "parameter_count": param_count,
        "capsule_keys": sorted(record.capsule.keys()),
        "confirmed_evidence_refs": list(CONFIRMED_EVIDENCE_REFS),
        "boundaries": dict(BOUNDARIES),
        "training_summary": summary,
    }
    _atomic_write_json(root / "manifest.json", manifest)
    rt = BoundedDeliveryRuntime.open(root)
    rt.snapshot("initial")
    return manifest


class BoundedDeliveryRuntime:
    """Fail-closed user-takeover runtime over a persisted bounded N36 bundle."""

    def __init__(
        self,
        bundle_dir: Path,
        manifest: dict[str, Any],
        harness: LongRunHarness,
        record: SessionRecord,
    ) -> None:
        self.bundle_dir = bundle_dir
        self.manifest = manifest
        self.harness = harness
        self.record = record
        self.ch_policy = ChineseTurnPolicy(int(manifest["seed"]), recurrent=True)
        self.ch_policy.load_state_dict(_load_prefixed_state(record.capsule, "ch"), strict=True)
        self.ch_policy.eval()

        self.phy_net = ConsequenceNetV2(int(manifest["seed"]))
        self.phy_net.load_state_dict(_load_prefixed_state(record.capsule, "phy"), strict=True)
        self.phy_net.eval()

        self.grow_net: PlasticGrowthNet
        self.grow_ctrl: GrowthPressureController
        self.grow_net, self.grow_ctrl, _ = _unpack_growth_net(record.capsule)
        self.grow_net.eval()

        self.unc_w = record.capsule["unc.w"].clone()
        self.unc_b = record.capsule["unc.b"].clone()

    @classmethod
    def open(cls, bundle_dir: str | Path) -> BoundedDeliveryRuntime:
        root = Path(bundle_dir)
        mpath = root / "manifest.json"
        if not mpath.exists():
            raise BundleIntegrityError(f"missing manifest.json in {root}")
        try:
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as exc:
            raise BundleIntegrityError(f"corrupt manifest.json: {exc}") from exc
        if manifest.get("schema") != SCHEMA or manifest.get("scope") != SCOPE:
            raise BundleIntegrityError(
                f"invalid schema/scope in manifest: {manifest.get('schema')}"
            )
        subject_id = str(manifest["subject_id"])
        try:
            harness = LongRunHarness(root, subject_id)
            if not harness.chain:
                raise BundleIntegrityError("empty audit chain")
            record = harness.store.load(subject_id)
        except Exception as exc:
            raise BundleIntegrityError(f"failed integrity check: {exc}") from exc
        if record.digest() != manifest.get("record_digest"):
            raise BundleIntegrityError("record_digest mismatch against manifest.json")
        if harness.chain[-1] != manifest.get("chain_head"):
            raise BundleIntegrityError("chain_head mismatch against manifest.json")
        return cls(root, manifest, harness, record)

    def _reload_state(self) -> None:
        fresh = BoundedDeliveryRuntime.open(self.bundle_dir)
        self.manifest = fresh.manifest
        self.harness = fresh.harness
        self.record = fresh.record
        self.ch_policy = fresh.ch_policy
        self.phy_net = fresh.phy_net
        self.grow_net = fresh.grow_net
        self.grow_ctrl = fresh.grow_ctrl
        self.unc_w = fresh.unc_w
        self.unc_b = fresh.unc_b

    def _require_mutable(self) -> None:
        state = str(self.manifest["lifecycle_state"])
        if state != "Running":
            raise LifecycleError(f"operation requires Running state, current={state!r}")

    def _sync_manifest(self, *, lifecycle_state: str | None = None) -> None:
        if lifecycle_state is not None:
            self.manifest["lifecycle_state"] = lifecycle_state
        self.record = self.harness.store.load(str(self.manifest["subject_id"]))
        self.manifest["record_digest"] = self.record.digest()
        self.manifest["chain_head"] = self.harness.chain[-1]
        self.manifest["sessions"] = self.harness.session_index
        _atomic_write_json(self.bundle_dir / "manifest.json", self.manifest)

    def _calibrated_conf(self, logits: torch.Tensor) -> float:
        feats = _confidence_features(logits)
        with torch.no_grad():
            z = (feats @ self.unc_w + self.unc_b).squeeze(0)
            return float(torch.sigmoid(z).item())

    def status(self) -> dict[str, Any]:
        alpha = self.record.capsule["ecl.alpha"]
        beta = self.record.capsule["ecl.beta"]
        cred = {
            src: round(float(alpha[i] / (alpha[i] + beta[i])), 4)
            for i, src in enumerate(ECL_SOURCES)
        }
        snap_dir = self.bundle_dir / "snapshots"
        snaps = (
            sorted(p.name for p in snap_dir.iterdir() if p.is_dir()) if snap_dir.exists() else []
        )
        return {
            "schema": SCHEMA,
            "scope": SCOPE,
            "boundaries": dict(BOUNDARIES),
            "subject_id": self.manifest["subject_id"],
            "lifecycle_state": self.manifest["lifecycle_state"],
            "sessions": self.harness.session_index,
            "record_digest": self.manifest["record_digest"],
            "chain_head": self.manifest["chain_head"],
            "chain_valid": self.harness.verify_chain(),
            "parameter_count": self.manifest["parameter_count"],
            "pending_count": len(self.record.pending),
            "credibility_posteriors": cred,
            "snapshots": snaps,
        }

    def interact_chinese(self, utterances: list[str], *, persist: bool = True) -> dict[str, Any]:
        if persist:
            self._require_mutable()
        tokenized = [tokenize_closed_chinese(u) for u in utterances]
        turn_outputs: list[dict[str, Any]] = []
        with torch.no_grad():
            h: torch.Tensor | None = None
            for raw, tok_ids in zip(utterances, tokenized, strict=True):
                x = torch.full((1, MAX_TOKENS), PAD_ID, dtype=torch.long)
                x[0, : len(tok_ids)] = torch.tensor(tok_ids, dtype=torch.long)
                logits_1d, _, h = self.ch_policy.step_hidden(x[0], h)
                logits = logits_1d.unsqueeze(0)
                pred_id = int(logits.argmax(dim=-1).item())
                conf = self._calibrated_conf(logits)
                turn_outputs.append(
                    {
                        "utterance": raw,
                        "action": CH_ACTIONS[pred_id],
                        "action_id": pred_id,
                        "confidence": round(conf, 4),
                        "verbal_confidence": verbalize_confidence(conf),
                    }
                )
        if persist:

            def _fn(rec: SessionRecord, s_idx: int) -> None:
                rec.append_experience(
                    "n36_chinese_interaction",
                    {"session": s_idx, "turns": len(turn_outputs), "last": turn_outputs[-1]},
                    source="n36_takeover",
                )

            self.harness.run_session(_fn, crash_before_commit=False)
            self._sync_manifest()
        return {
            "scope": SCOPE,
            "turns": turn_outputs,
            "persisted": bool(persist),
            "sessions": self.harness.session_index,
        }

    def predict_physics(
        self,
        family: str = "rough_collision",
        *,
        seed: int = 101,
        ablate_barrier: bool = False,
    ) -> dict[str, Any]:
        x, y = collect_v2(int(seed), family, 20)
        metrics = eval_metrics_v2(self.phy_net, x, y)
        abl = ablate_barrier_collision_recall(self.phy_net, x, y) if ablate_barrier else None
        with torch.no_grad():
            sample_logits = self.phy_net(x[:1])
            pred_id = int(sample_logits.argmax(dim=-1).item())
        conf = self._calibrated_conf(sample_logits)
        return {
            "scope": SCOPE,
            "family": family,
            "metrics": metrics,
            "sample_prediction": PHY_OUTCOMES[pred_id],
            "sample_confidence": round(conf, 4),
            "verbal_confidence": verbalize_confidence(conf),
            "barrier_ablation": abl,
        }

    def update_credibility(
        self,
        source: str,
        correct: int,
        *,
        crash_before_commit: bool = False,
    ) -> dict[str, Any]:
        self._require_mutable()
        if source not in ECL_SOURCES:
            raise ValueError(f"unknown source {source!r}; expected one of {ECL_SOURCES}")
        if correct not in (0, 1):
            raise ValueError(f"correct must be 0 or 1, got {correct!r}")
        idx = ECL_SOURCES.index(source)

        def _fn(rec: SessionRecord, s_idx: int) -> None:
            alpha = rec.capsule["ecl.alpha"].clone()
            beta = rec.capsule["ecl.beta"].clone()
            alpha[idx] = 1.0 + 0.995 * (alpha[idx] - 1.0) + float(correct)
            beta[idx] = 1.0 + 0.995 * (beta[idx] - 1.0) + float(1 - correct)
            rec.capsule["ecl.alpha"] = alpha
            rec.capsule["ecl.beta"] = beta
            rec.append_experience(
                "n36_credibility_update",
                {"session": s_idx, "source": source, "correct": int(correct)},
                source="n36_takeover",
            )

        self.harness.run_session(_fn, crash_before_commit=crash_before_commit)
        self._sync_manifest()
        return self.status()

    def add_pending(self, item_id: str, note: str) -> dict[str, Any]:
        self._require_mutable()

        def _fn(rec: SessionRecord, _s: int) -> None:
            rec.add_pending({"id": item_id, "note": note})

        self.harness.run_session(_fn, crash_before_commit=False)
        self._sync_manifest()
        return {
            "pending_count": len(self.record.pending),
            "record_digest": self.manifest["record_digest"],
        }

    def resolve_pending(self, item_id: str, outcome: str) -> dict[str, Any]:
        self._require_mutable()

        def _fn(rec: SessionRecord, _s: int) -> None:
            rec.resolve_pending(item_id, outcome)

        self.harness.run_session(_fn, crash_before_commit=False)
        self._sync_manifest()
        return {
            "pending_count": len(self.record.pending),
            "record_digest": self.manifest["record_digest"],
        }

    def pause(self) -> dict[str, Any]:
        if self.manifest["lifecycle_state"] == "Archived":
            raise LifecycleError("cannot pause Archived bundle")
        self._sync_manifest(lifecycle_state="Paused")
        return {"lifecycle_state": self.manifest["lifecycle_state"]}

    def resume(self) -> dict[str, Any]:
        if self.manifest["lifecycle_state"] == "Archived":
            raise LifecycleError("cannot resume Archived bundle")
        self._sync_manifest(lifecycle_state="Running")
        return {"lifecycle_state": self.manifest["lifecycle_state"]}

    def archive(self) -> dict[str, Any]:
        self._sync_manifest(lifecycle_state="Archived")
        return {"lifecycle_state": self.manifest["lifecycle_state"]}

    def snapshot(self, name: str) -> dict[str, Any]:
        if not name or any(c in name for c in ("/", "\\", "..")):
            raise ValueError(f"invalid snapshot name {name!r}")
        snap_dir = self.bundle_dir / "snapshots" / name
        snap_dir.mkdir(parents=True, exist_ok=True)
        sid = str(self.manifest["subject_id"])
        shutil.copy2(self.bundle_dir / f"{sid}.session", snap_dir / f"{sid}.session")
        shutil.copy2(self.bundle_dir / f"{sid}.audit.jsonl", snap_dir / f"{sid}.audit.jsonl")
        shutil.copy2(self.bundle_dir / "manifest.json", snap_dir / "manifest.json")
        return {"snapshot": name, "record_digest": self.manifest["record_digest"]}

    def rollback(self, name: str) -> dict[str, Any]:
        if self.manifest["lifecycle_state"] == "Archived":
            raise LifecycleError("cannot rollback Archived bundle")
        snap_dir = self.bundle_dir / "snapshots" / name
        if not snap_dir.exists():
            raise FileNotFoundError(f"unknown snapshot {name!r}")
        sid = str(self.manifest["subject_id"])
        shutil.copy2(snap_dir / f"{sid}.session", self.bundle_dir / f"{sid}.session")
        shutil.copy2(snap_dir / f"{sid}.audit.jsonl", self.bundle_dir / f"{sid}.audit.jsonl")
        shutil.copy2(snap_dir / "manifest.json", self.bundle_dir / "manifest.json")
        self._reload_state()
        return {
            "rolled_back_to": name,
            "record_digest": self.manifest["record_digest"],
            "sessions": self.harness.session_index,
        }

    def read_split(self, split: str) -> None:
        self.harness.read_split(split)
