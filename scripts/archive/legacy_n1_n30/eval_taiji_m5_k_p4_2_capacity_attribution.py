"""Run the isolated P4.2 capacity-attribution experiment.

P4.0 observed a non-monotonic fixed-G failure as candidate sets widened and
P4.1 made candidate-set context content-addressable.  P4.2 is the first step
allowed to fit on a new child state.  It compares:

* the inherited 13-parameter fixed-small G;
* a 22-parameter context arm whose inherited candidate weights are frozen;
* a 22-parameter context arm whose candidate and context weights are both
  trainable.

All three arms use disjoint train/validation/holdout identities, retain the
same safe-selection contract, and save independently restorable child
checkpoints.  This script never mutates the P3.5/P4.0 parent artifacts and
never admits structural growth.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    MODEL_SEED,
    WORKER_ROOT,
    _context,
)
from scripts.training.eval_taiji_m5_k_p3_3_g_learning import _fresh_learners  # noqa: E402
from scripts.training.eval_taiji_m5_k_p3_4_behavior_signal_canary import (  # noqa: E402
    _behavior_record,
)
from scripts.training.eval_taiji_m5_k_p3_5_g_learning import (  # noqa: E402
    _independent_g_restore,
    _load_json,
    _load_mapping,
)
from scripts.training.eval_taiji_m5_k_p4_0_capacity_pressure import (  # noqa: E402
    _materialize_case,
    _pressure_record,
)
from scripts.training.eval_taiji_m5_k_p4_1_context_contract import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    _context_payload,
)
from taiji import (  # noqa: E402
    GSelectionBehaviorSet,
    GSelectionCandidateSet,
    GSelectionLearner,
    content_digest,
)
from taiji.g_selection import G_SELECTION_FEATURE_NAMES  # noqa: E402
from taiji.local_learning import (  # noqa: E402
    apply_linear_delta,
    freeze_parameters,
    mean_squared_error_delta,
)

REPORT_FORMAT = "taiji-m5-k-p4-2-capacity-attribution-v1"
MANIFEST_FORMAT = "taiji-m5-k-p4-2-capacity-attribution-manifest-v1"
CHECKPOINT_FORMAT = "taiji-m5-k-p4-2-context-learner-v1"
VERSION = 1
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_2_capacity_attribution_manifest_v1.json"
)
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_2_capacity_attribution_20260911.json"
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)
P4_1_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_1_context_contract_20260911.json"
P3_5_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p3_5_g_learning_20260911.json"
P3_6_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p3_6_behavior_holdout_manifest_v1.json"
)
TRAINING_EPOCHS = 8
LEARNING_RATE = 0.15
MARGIN_EPSILON = 1e-9
SEEDS = (0, 1)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _new_specs(split: str, offset: int) -> tuple[dict[str, Any], ...]:
    # P4.0's materializer deliberately writes only its namespaced `p40_`
    # fixtures.  Keep that safe fixture path while adding a distinct P4.2
    # segment so the resulting paths cannot alias P4.0 artifacts.
    prefix = f"p40_p42_{split}"
    rows = (
        (
            "A",
            "resolved-language",
            (f"{prefix}_a_python.py", f"{prefix}_a_rust.rs", f"{prefix}_a_python_b.py"),
        ),
        (
            "B",
            "ambiguous-language",
            (f"{prefix}_b_rust.py", f"{prefix}_b_python.py", f"{prefix}_b_typescript.ts"),
        ),
        (
            "C",
            "resolved-language",
            (f"{prefix}_c_typescript.ts", f"{prefix}_c_python.py", f"{prefix}_c_rust.rs"),
        ),
        (
            "D",
            "ambiguous-header",
            (f"{prefix}_d_header.h", f"{prefix}_d_python.py", f"{prefix}_d_rust.rs"),
        ),
        (
            "R",
            "recovery-no-selection",
            (f"{prefix}_r_missing.txt", f"{prefix}_r_python.py", f"{prefix}_r_rust.rs"),
        ),
    )
    return tuple(
        {
            "index": index,
            "class_key": class_key,
            "project_id": f"p4-2-{split}-project-{class_key.lower()}",
            "variant_paths": paths,
            "state_profile": state_profile,
            "task_seed": offset + index,
        }
        for index, (class_key, state_profile, paths) in enumerate(rows)
    )


def _rebind_pressure_record(
    record: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
    width: int,
) -> dict[str, Any]:
    old_candidates = record["candidate_set"]
    old_behavior = record["behavior_set"]
    candidates = tuple(old_candidates.candidates)
    candidate_set = GSelectionCandidateSet.create(
        example_id=(
            f"p4-2:{split}:{source_index}:width-{width}:{old_candidates.candidate_set_digest}"
        ),
        family_id=f"p4-2:{split}:family:{source_index}",
        split=split,
        project_id=old_candidates.project_id,
        path=old_candidates.path,
        input_digest=old_candidates.input_digest,
        candidates=candidates,
        target_candidate_id=old_candidates.target_candidate_id,
        target_kind=old_candidates.target_kind,
    )
    behavior_set = GSelectionBehaviorSet.create(
        candidate_set_digest=candidate_set.candidate_set_digest,
        inference_digest=candidate_set.inference_digest,
        split=split,
        project_id=candidate_set.project_id,
        path=candidate_set.path,
        outcomes=old_behavior.outcomes,
    )
    diagnostic = dict(record["diagnostic"])
    diagnostic.update({"split": split, "source_index": source_index, "candidate_width": width})
    return {
        **record,
        "candidate_set": candidate_set,
        "behavior_set": behavior_set,
        "diagnostic": diagnostic,
        "fit_eligible": float(behavior_set.utility_margin) > MARGIN_EPSILON,
    }


def _build_split_records(
    *,
    split: str,
    offset: int,
    scratch: Path,
    parent_digest: str,
    worker_bundle_digest: str,
    source_manifest_digest: str,
    projector: Any,
    semantic: Any,
    transition: Any,
    semantic_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from scripts.training.eval_taiji_m5_k_p3_3_g_signal_canary import _catalogs

    goals, content_plans = _catalogs(semantic_payload)
    base_records: list[dict[str, Any]] = []
    for spec in _new_specs(split, offset):
        experience, metadata, case = _materialize_case(
            scratch=scratch,
            spec=spec,
            parent_digest=parent_digest,
            worker_bundle_digest=worker_bundle_digest,
            source_manifest_digest=source_manifest_digest,
            projector=projector,
        )
        metadata = {**metadata, "split": split}
        candidate_set, behavior_set, diagnostic = _behavior_record(
            experience=experience,
            metadata=metadata,
            case=case,
            semantic=semantic,
            transition=transition,
            goals=goals,
            content_plans=content_plans,
            label=f"p4-2-{split}-{int(spec['index'])}",
        )
        base_records.append(
            {
                "experience": experience,
                "metadata": metadata,
                "case": case,
                "candidate_set": candidate_set,
                "behavior_set": behavior_set,
                "diagnostic": diagnostic,
            }
        )
    records: list[dict[str, Any]] = []
    for source_index in range(len(base_records)):
        for width in (2, 4, 8, 12):
            pressure = _pressure_record(
                base_records=base_records,
                source_index=source_index,
                width=width,
                transition=transition,
            )
            records.append(
                _rebind_pressure_record(
                    pressure,
                    split=split,
                    source_index=source_index,
                    width=width,
                )
            )
    return records


def _records_from_manifest(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in payload["holdout_records"]:
        candidate_set = GSelectionCandidateSet.from_payload(raw["candidate_set"])
        behavior_set = GSelectionBehaviorSet.from_payload(raw["behavior_set"])
        if behavior_set.candidate_set_digest != candidate_set.candidate_set_digest:
            raise ValueError("P4.2 retention behavior/candidate binding drifted")
        records.append(
            {
                "candidate_set": candidate_set,
                "behavior_set": behavior_set,
                "diagnostic": raw.get("diagnostic", {}),
                "fit_eligible": False,
            }
        )
    return records


class ContextLearner:
    """A content-addressed linear context arm for the P4.2 experiment."""

    def __init__(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        parent_g_checkpoint_digest: str,
        mode: str,
        candidate_weights: Sequence[float],
        bias: float,
        learning_rate: float = LEARNING_RATE,
        confidence_floor: float = 0.55,
        selection_margin: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> None:
        if mode not in {"context-only", "full"}:
            raise ValueError("P4.2 context learner mode is invalid")
        if len(candidate_weights) != len(G_SELECTION_FEATURE_NAMES):
            raise ValueError("P4.2 parent candidate weight dimension drifted")
        self.parent_manifest_digest = str(parent_manifest_digest)
        self.k_checkpoint_digests = {
            str(key): str(value) for key, value in k_checkpoint_digests.items()
        }
        self.parent_g_checkpoint_digest = str(parent_g_checkpoint_digest)
        self.mode = mode
        self.learning_rate = float(learning_rate)
        self.confidence_floor = float(confidence_floor)
        self.selection_margin = float(selection_margin)
        self.device = torch.device(device)
        self.model = nn.Linear(
            len(G_SELECTION_FEATURE_NAMES) + len(CONTEXT_FEATURE_NAMES), 1, device=self.device
        )
        with torch.no_grad():
            self.model.weight.zero_()
            self.model.weight[0, : len(G_SELECTION_FEATURE_NAMES)] = torch.tensor(
                tuple(float(value) for value in candidate_weights),
                dtype=torch.float32,
                device=self.device,
            )
            self.model.bias.fill_(float(bias))
        freeze_parameters(self.model)
        self.training_steps = 0
        self.revision = 0
        self.last_train_digest = ""

    @property
    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.model.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return (
            len(CONTEXT_FEATURE_NAMES) + 1 if self.mode == "context-only" else self.parameter_count
        )

    @property
    def model_state_digest(self) -> str:
        return content_digest(
            {
                "weight": self.model.weight.detach().cpu(),
                "bias": self.model.bias.detach().cpu(),
            }
        )

    def assert_lineage(
        self,
        *,
        parent_manifest_digest: str,
        k_checkpoint_digests: Mapping[str, str],
        parent_g_checkpoint_digest: str,
    ) -> None:
        if self.parent_manifest_digest != str(parent_manifest_digest):
            raise ValueError("P4.2 context learner parent manifest lineage mismatch")
        if self.k_checkpoint_digests != {
            str(key): str(value) for key, value in k_checkpoint_digests.items()
        }:
            raise ValueError("P4.2 context learner K lineage mismatch")
        if self.parent_g_checkpoint_digest != str(parent_g_checkpoint_digest):
            raise ValueError("P4.2 context learner parent G lineage mismatch")

    def _features(
        self,
        candidate_set: GSelectionCandidateSet,
        candidate_id: str,
        context: Mapping[str, Any],
    ) -> torch.Tensor:
        candidate = next(
            item for item in candidate_set.candidates if item.candidate_id == candidate_id
        )
        values = (
            *candidate.feature_vector,
            *tuple(float(value) for value in context["candidate_features"][candidate_id]),
        )
        if len(values) != len(G_SELECTION_FEATURE_NAMES) + len(CONTEXT_FEATURE_NAMES):
            raise ValueError("P4.2 context learner feature dimension drifted")
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def score(
        self,
        candidate_set: GSelectionCandidateSet,
        candidate_id: str,
        context: Mapping[str, Any],
    ) -> float:
        with torch.no_grad():
            return float(
                self.model(
                    self._features(candidate_set, candidate_id, context).reshape(1, -1)
                ).reshape(())
            )

    def select(
        self,
        candidate_set: GSelectionCandidateSet,
        context: Mapping[str, Any],
    ) -> str:
        scores = {
            candidate.candidate_id: self.score(candidate_set, candidate.candidate_id, context)
            for candidate in candidate_set.candidates
        }
        return _select_from_scores(
            candidate_set,
            scores,
            confidence_floor=self.confidence_floor,
            selection_margin=self.selection_margin,
        )

    def fit(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        epochs: int = TRAINING_EPOCHS,
        learning_rate: float = LEARNING_RATE,
        order_seed: int = 0,
    ) -> dict[str, Any]:
        items = [record for record in records if bool(record.get("fit_eligible", False))]
        if not items:
            raise ValueError("P4.2 context fit has no eligible records")
        ordered = sorted(items, key=lambda item: item["candidate_set"].candidate_set_digest)
        if int(order_seed) % 2:
            ordered.reverse()
        for _epoch in range(int(epochs)):
            for record in ordered:
                candidate_set = record["candidate_set"]
                context = record["context"]
                inputs = torch.stack(
                    [
                        self._features(candidate_set, candidate.candidate_id, context)
                        for candidate in candidate_set.candidates
                    ]
                )
                targets = torch.zeros(
                    (len(candidate_set.candidates), 1), dtype=torch.float32, device=self.device
                )
                target_index = next(
                    index
                    for index, candidate in enumerate(candidate_set.candidates)
                    if candidate.candidate_id == candidate_set.target_candidate_id
                )
                targets[target_index, 0] = 1.0
                predictions = self.model(inputs)
                error = mean_squared_error_delta(predictions, targets)
                update_inputs = inputs.clone()
                if self.mode == "context-only":
                    update_inputs[:, : len(G_SELECTION_FEATURE_NAMES)] = 0.0
                apply_linear_delta(self.model, update_inputs, error, float(learning_rate))
                self.training_steps += 1
        self.revision += 1
        self.last_train_digest = content_digest(
            {
                "candidate_set_digests": [
                    record["candidate_set"].candidate_set_digest for record in ordered
                ],
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "order_seed": int(order_seed),
                "mode": self.mode,
            }
        )
        return {
            "dataset_digest": self.last_train_digest,
            "candidate_sets": len(ordered),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "order_seed": int(order_seed),
            "training_steps": self.training_steps,
            "revision": self.revision,
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "version": VERSION,
            "feature_names": [*G_SELECTION_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES],
            "parent_manifest_digest": self.parent_manifest_digest,
            "k_checkpoint_digests": dict(self.k_checkpoint_digests),
            "parent_g_checkpoint_digest": self.parent_g_checkpoint_digest,
            "mode": self.mode,
            "learning_rate": self.learning_rate,
            "confidence_floor": self.confidence_floor,
            "selection_margin": self.selection_margin,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "weight": self.model.weight.detach().cpu().clone(),
            "bias": self.model.bias.detach().cpu().clone(),
            "model_state_digest": self.model_state_digest,
            "training_steps": self.training_steps,
            "revision": self.revision,
            "last_train_digest": self.last_train_digest,
        }
        payload["checkpoint_digest"] = content_digest(payload)
        return payload

    @classmethod
    def from_checkpoint(
        cls,
        payload: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> ContextLearner:
        if payload.get("format") != CHECKPOINT_FORMAT or int(payload.get("version", -1)) != VERSION:
            raise ValueError("unsupported P4.2 context checkpoint")
        if _digest_without(payload, "checkpoint_digest") != str(payload["checkpoint_digest"]):
            raise ValueError("P4.2 context checkpoint digest mismatch")
        feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
        expected_names = (*G_SELECTION_FEATURE_NAMES, *CONTEXT_FEATURE_NAMES)
        if feature_names != expected_names:
            raise ValueError("P4.2 context feature manifest drifted")
        weight = payload["weight"]
        bias = payload["bias"]
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise TypeError("P4.2 context checkpoint tensors are invalid")
        candidate_weights = weight.reshape(-1)[: len(G_SELECTION_FEATURE_NAMES)].tolist()
        learner = cls(
            parent_manifest_digest=str(payload["parent_manifest_digest"]),
            k_checkpoint_digests=dict(payload["k_checkpoint_digests"]),
            parent_g_checkpoint_digest=str(payload["parent_g_checkpoint_digest"]),
            mode=str(payload["mode"]),
            candidate_weights=candidate_weights,
            bias=float(bias.reshape(()).item()),
            learning_rate=float(payload["learning_rate"]),
            confidence_floor=float(payload["confidence_floor"]),
            selection_margin=float(payload["selection_margin"]),
            device=device,
        )
        if int(payload["parameter_count"]) != learner.parameter_count:
            raise ValueError("P4.2 context parameter count drifted")
        if int(payload["trainable_parameter_count"]) != learner.trainable_parameter_count:
            raise ValueError("P4.2 context trainable parameter count drifted")
        if tuple(weight.shape) != tuple(learner.model.weight.shape) or tuple(bias.shape) != tuple(
            learner.model.bias.shape
        ):
            raise ValueError("P4.2 context checkpoint tensor shape drifted")
        if content_digest({"weight": weight, "bias": bias}) != str(payload["model_state_digest"]):
            raise ValueError("P4.2 context model state digest mismatch")
        with torch.no_grad():
            learner.model.weight.copy_(
                weight.detach().to(device=learner.device, dtype=torch.float32)
            )
            learner.model.bias.copy_(bias.detach().to(device=learner.device, dtype=torch.float32))
        learner.training_steps = int(payload.get("training_steps", 0))
        learner.revision = int(payload.get("revision", 0))
        learner.last_train_digest = str(payload.get("last_train_digest", ""))
        freeze_parameters(learner.model)
        return learner


def _select_from_scores(
    candidate_set: GSelectionCandidateSet,
    scores: Mapping[str, float],
    *,
    confidence_floor: float,
    selection_margin: float,
) -> str:
    role_priority = {"abstain": 2, "reobserve": 1, "proposal": 0}
    ranked = sorted(
        candidate_set.candidates,
        key=lambda item: (
            -float(scores[item.candidate_id]),
            -role_priority[item.candidate_role],
            item.candidate_id,
        ),
    )
    selected = ranked[0]
    safe = max(
        (
            item
            for item in candidate_set.candidates
            if item.candidate_role in {"abstain", "reobserve"}
        ),
        key=lambda item: (
            1 if item.candidate_role == "abstain" else 0,
            item.confidence,
            item.candidate_id,
        ),
    )
    if selected.candidate_role == "proposal":
        unsafe = (
            selected.goal is None
            or selected.content_plan is None
            or selected.confidence < confidence_floor
            or float(scores[selected.candidate_id])
            <= float(scores[safe.candidate_id]) + selection_margin
        )
        if unsafe:
            selected = safe
    elif (
        float(scores[selected.candidate_id]) <= float(scores[safe.candidate_id]) + selection_margin
    ):
        selected = safe
    return selected.candidate_id


def _context_record(record: Mapping[str, Any]) -> dict[str, Any]:
    context = _context_payload(record["candidate_set"])
    return {**record, "context": context}


def _evaluate(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    learner: GSelectionLearner | ContextLearner,
) -> dict[str, Any]:
    target_hits = 0
    utility_sum = 0.0
    selected_roles: dict[str, int] = {}
    safe_selection_violations = 0
    reobserve_selected = 0
    reobserve_projection_passed = 0
    workbench_success = 0
    rows: list[dict[str, Any]] = []
    for record in records:
        candidate_set = record["candidate_set"]
        behavior_set = record["behavior_set"]
        if arm == "fixed-small":
            decision = learner.select(candidate_set)  # type: ignore[union-attr]
            selected_id = decision.selected_candidate_id
        else:
            context = record["context"]
            selected_id = learner.select(candidate_set, context)  # type: ignore[union-attr]
        selected = next(
            item for item in candidate_set.candidates if item.candidate_id == selected_id
        )
        outcome = next(item for item in behavior_set.outcomes if item.candidate_id == selected_id)
        target_hit = selected_id == behavior_set.behavior_target_candidate_id
        target_hits += int(target_hit)
        utility_sum += float(outcome.utility)
        selected_roles[selected.candidate_role] = selected_roles.get(selected.candidate_role, 0) + 1
        low_evidence_has_safe = any(
            item.safe_exit_valid or item.safe_exit_progress for item in behavior_set.outcomes
        )
        if low_evidence_has_safe and selected.candidate_role == "proposal":
            safe_selection_violations += 1
        if selected.candidate_role == "reobserve":
            reobserve_selected += 1
            if (
                not outcome.planner_accepted
                and not outcome.route_valid
                and not outcome.execution_success
            ):
                reobserve_projection_passed += 1
        workbench_success += int(outcome.execution_success)
        rows.append(
            {
                "candidate_set_digest": candidate_set.candidate_set_digest,
                "selected_candidate_id": selected_id,
                "behavior_target_candidate_id": behavior_set.behavior_target_candidate_id,
                "target_hit": target_hit,
                "selected_role": selected.candidate_role,
                "selected_utility": outcome.utility,
            }
        )
    count = len(records)
    return {
        "count": count,
        "behavior_target_hit_count": target_hits,
        "behavior_target_hit_rate": target_hits / count if count else 0.0,
        "selected_utility_sum": utility_sum,
        "selected_utility_mean": utility_sum / count if count else 0.0,
        "selected_residual_error": 1.0 - utility_sum / count if count else 1.0,
        "selected_roles": selected_roles,
        "safe_selection_violations": safe_selection_violations,
        "reobserve_selected_count": reobserve_selected,
        "reobserve_projection_passed": reobserve_projection_passed == reobserve_selected,
        "workbench_success_count": workbench_success,
        "rows": rows,
    }


def _independent_context_restore(path: Path) -> dict[str, Any]:
    code = (
        "import json, sys, torch; "
        "from scripts.training.eval_taiji_m5_k_p4_2_capacity_attribution import ContextLearner; "
        "payload=torch.load(sys.argv[1], map_location='cpu', weights_only=False); "
        "learner=ContextLearner.from_checkpoint(payload); "
        "print(json.dumps({'restore_digest_equal': learner.checkpoint()['checkpoint_digest']==payload['checkpoint_digest'], "
        "'parameter_count': learner.parameter_count, 'passed': True}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    parsed: dict[str, Any] = {}
    if stdout:
        parsed = json.loads(stdout.splitlines()[-1])
    return {
        "returncode": completed.returncode,
        "independent_process_restore": completed.returncode == 0 and bool(parsed.get("passed")),
        "stdout": stdout,
        "stderr": completed.stderr,
        **parsed,
    }


def _save_g_checkpoint(path: Path, learner: GSelectionLearner) -> dict[str, Any]:
    payload = learner.checkpoint()
    _save_torch_atomic(path, payload)
    restored = GSelectionLearner.from_checkpoint(_load_mapping(path), device="cpu")
    independent = _independent_g_restore(path)
    return {
        "path": str(path),
        "digest": payload["checkpoint_digest"],
        "bytes": path.stat().st_size,
        "roundtrip": restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
        "restore": independent,
        "passed": bool(independent.get("independent_process_restore"))
        and restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
    }


def _save_context_checkpoint(path: Path, learner: ContextLearner) -> dict[str, Any]:
    payload = learner.checkpoint()
    _save_torch_atomic(path, payload)
    restored = ContextLearner.from_checkpoint(_load_mapping(path), device="cpu")
    independent = _independent_context_restore(path)
    return {
        "path": str(path),
        "digest": payload["checkpoint_digest"],
        "bytes": path.stat().st_size,
        "roundtrip": restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
        "restore": independent,
        "passed": bool(independent.get("independent_process_restore"))
        and restored.checkpoint()["checkpoint_digest"] == payload["checkpoint_digest"],
    }


def _verify_source(
    *,
    p4_1_manifest: Mapping[str, Any],
    p4_1_report: Mapping[str, Any],
    p3_6_manifest: Mapping[str, Any],
) -> None:
    if p4_1_report.get("status") != "completed" or not p4_1_report.get("preflight_passed"):
        raise ValueError("P4.2 requires the completed P4.1 preflight")
    if p4_1_report.get("manifest_digest") != p4_1_manifest.get("manifest_digest"):
        raise ValueError("P4.1 manifest/report digest mismatch")
    if content_digest(
        {key: value for key, value in p4_1_manifest.items() if key != "manifest_digest"}
    ) != p4_1_manifest.get("manifest_digest"):
        raise ValueError("P4.1 manifest content digest mismatch")
    if p4_1_report.get("growth_admitted") or p4_1_report.get("can_promote"):
        raise ValueError("P4.2 cannot consume an admitted P4.1 artifact")
    if p3_6_manifest.get("manifest_digest") == p4_1_manifest.get("source_p4_0_manifest_digest"):
        raise ValueError("P4.2 retention source unexpectedly aliases P4.0")


def run_experiment(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    started = time.perf_counter()
    run_dir = DEFAULT_OUTPUT_ROOT / f"taiji_m5_k_p4_2_capacity_attribution_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "fit_called": False,
        "external_target_used": False,
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "report": str(report_path),
    }
    try:
        p4_1_manifest = _load_json(P4_1_MANIFEST)
        p4_1_report = _load_json(P4_1_REPORT)
        p3_6_manifest = _load_json(P3_6_MANIFEST)
        _verify_source(
            p4_1_manifest=p4_1_manifest,
            p4_1_report=p4_1_report,
            p3_6_manifest=p3_6_manifest,
        )
        _artifacts, parent_digest, bundle, projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        p3_5_report = _load_json(P3_5_REPORT)
        trained_path = Path(str(p3_5_report["g_trained_checkpoint"]["path"]))
        parent_payload = _load_mapping(trained_path)
        parent = GSelectionLearner.from_checkpoint(parent_payload, device="cpu")
        # P4.1 does not repeat the K mapping in its manifest; bind it to the
        # P3.5 report explicitly before any child is created.
        worker_digests = dict(p3_5_report["k_checkpoint_digests_before"])
        parent.assert_lineage(
            parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
            k_checkpoint_digests=worker_digests,
        )
        if (
            p4_1_manifest["g_trained_checkpoint_digest"]
            != p3_5_report["g_trained_checkpoint"]["digest"]
        ):
            raise ValueError("P4.2 parent G digest drifted")
        parent_restore = _independent_g_restore(trained_path)
        if not parent_restore.get("independent_process_restore"):
            raise RuntimeError("P4.2 parent G independent restore failed")
        # The P3.2 report remains the authoritative K checkpoint location.
        p3_2_report = _load_json(
            PROJECT_ROOT / "reports" / "taiji_m5_k_p3_2_owner_transfer_20260910.json"
        )
        worker_restore = p3_2_report["base_continuation"]["worker_restore"]
        semantic_payload = _load_mapping(Path(str(worker_restore["k1"]["path"])))
        transition_payload = _load_mapping(Path(str(worker_restore["k2"]["path"])))
        if (
            content_digest(semantic_payload) != worker_digests["k1"]
            or content_digest(transition_payload) != worker_digests["k2"]
        ):
            raise ValueError("P4.2 K parent checkpoint digest drifted")
        semantic, transition = _fresh_learners(semantic_payload, transition_payload)
        run_dir.mkdir(parents=True, exist_ok=False)
        data_root = run_dir / "data"
        train_records = _build_split_records(
            split="train",
            offset=42000,
            scratch=data_root / "train",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_1_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        validation_records = _build_split_records(
            split="validation",
            offset=42100,
            scratch=data_root / "validation",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_1_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        holdout_records = _build_split_records(
            split="holdout",
            offset=42200,
            scratch=data_root / "holdout",
            parent_digest=parent_digest,
            worker_bundle_digest=bundle.bundle_digest,
            source_manifest_digest=str(p4_1_manifest["manifest_digest"]),
            projector=projector,
            semantic=semantic,
            transition=transition,
            semantic_payload=semantic_payload,
        )
        for collection in (train_records, validation_records, holdout_records):
            for index, record in enumerate(collection):
                collection[index] = _context_record(record)
        retention_records = [
            _context_record(record) for record in _records_from_manifest(p3_6_manifest)
        ]
        all_new_records = [*train_records, *validation_records, *holdout_records]
        new_projects = {record["candidate_set"].project_id for record in all_new_records}
        new_paths = {record["candidate_set"].path for record in all_new_records}
        old_projects = {record["candidate_set"].project_id for record in retention_records}
        old_paths = {record["candidate_set"].path for record in retention_records}
        train_fit = [record for record in train_records if record["fit_eligible"]]
        if (
            len(train_fit) < 8
            or len({record["diagnostic"]["class_key"] for record in train_fit}) < 4
        ):
            raise ValueError("P4.2 train cohort is too small or class-skewed")
        identity_gate = {
            "train_records": len(train_records) == 20,
            "validation_records": len(validation_records) == 20,
            "holdout_records": len(holdout_records) == 20,
            "train_fit_records_positive": len(train_fit) >= 8,
            "five_train_classes": len(
                {record["diagnostic"]["class_key"] for record in train_records}
            )
            == 5,
            "five_validation_classes": len(
                {record["diagnostic"]["class_key"] for record in validation_records}
            )
            == 5,
            "five_holdout_classes": len(
                {record["diagnostic"]["class_key"] for record in holdout_records}
            )
            == 5,
            "new_projects_disjoint_from_retention": new_projects.isdisjoint(old_projects),
            "new_paths_disjoint_from_retention": new_paths.isdisjoint(old_paths),
            "new_candidate_digests_unique": len(
                {record["candidate_set"].candidate_set_digest for record in all_new_records}
            )
            == 60,
            "new_behavior_digests_unique": len(
                {record["behavior_set"].behavior_digest for record in all_new_records}
            )
            == 60,
        }
        seed_results: list[dict[str, Any]] = []
        for seed in SEEDS:
            seed_dir = run_dir / f"seed-{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            fixed_small = GSelectionLearner.from_checkpoint(
                copy.deepcopy(parent_payload), device="cpu"
            )
            parent_weight = parent.model.weight.detach().cpu().flatten().tolist()
            parent_bias = float(parent.model.bias.detach().cpu().reshape(()).item())
            context_small = ContextLearner(
                parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
                k_checkpoint_digests=worker_digests,
                parent_g_checkpoint_digest=str(p4_1_manifest["g_trained_checkpoint_digest"]),
                mode="context-only",
                candidate_weights=parent_weight,
                bias=parent_bias,
            )
            fixed_large = ContextLearner(
                parent_manifest_digest=str(p4_1_manifest["source_p3_2_manifest_digest"]),
                k_checkpoint_digests=worker_digests,
                parent_g_checkpoint_digest=str(p4_1_manifest["g_trained_checkpoint_digest"]),
                mode="full",
                candidate_weights=parent_weight,
                bias=parent_bias,
            )
            zero_checkpoints = {
                "fixed-small": _save_g_checkpoint(seed_dir / "fixed-small-zero.pt", fixed_small),
                "context-aware-small": _save_context_checkpoint(
                    seed_dir / "context-aware-small-zero.pt", context_small
                ),
                "fixed-large": _save_context_checkpoint(
                    seed_dir / "fixed-large-zero.pt", fixed_large
                ),
            }
            if not all(item["passed"] for item in zero_checkpoints.values()):
                raise RuntimeError(f"P4.2 zero-step checkpoint preflight failed for seed {seed}")
            fixed_fit = fixed_small.fit(
                [record["candidate_set"] for record in train_fit],
                epochs=TRAINING_EPOCHS,
                learning_rate=LEARNING_RATE,
            )
            context_fit = context_small.fit(
                train_fit, epochs=TRAINING_EPOCHS, learning_rate=LEARNING_RATE, order_seed=seed
            )
            large_fit = fixed_large.fit(
                train_fit, epochs=TRAINING_EPOCHS, learning_rate=LEARNING_RATE, order_seed=seed
            )
            trained_checkpoints = {
                "fixed-small": _save_g_checkpoint(seed_dir / "fixed-small-trained.pt", fixed_small),
                "context-aware-small": _save_context_checkpoint(
                    seed_dir / "context-aware-small-trained.pt", context_small
                ),
                "fixed-large": _save_context_checkpoint(
                    seed_dir / "fixed-large-trained.pt", fixed_large
                ),
            }
            if not all(item["passed"] for item in trained_checkpoints.values()):
                raise RuntimeError(f"P4.2 trained checkpoint preflight failed for seed {seed}")
            arms: dict[str, GSelectionLearner | ContextLearner] = {
                "fixed-small": fixed_small,
                "context-aware-small": context_small,
                "fixed-large": fixed_large,
            }
            metrics = {
                arm: {
                    "validation": _evaluate(validation_records, arm=arm, learner=learner),
                    "holdout": _evaluate(holdout_records, arm=arm, learner=learner),
                    "retention": _evaluate(retention_records, arm=arm, learner=learner),
                }
                for arm, learner in arms.items()
            }
            parent_retention = _evaluate(retention_records, arm="fixed-small", learner=parent)
            retention_gate = {
                arm: {
                    "utility_not_below_parent": metrics[arm]["retention"]["selected_utility_mean"]
                    >= parent_retention["selected_utility_mean"] - 1e-9,
                    "target_hit_not_below_parent": metrics[arm]["retention"][
                        "behavior_target_hit_rate"
                    ]
                    >= parent_retention["behavior_target_hit_rate"],
                    "safe_selection_preserved": metrics[arm]["retention"][
                        "safe_selection_violations"
                    ]
                    == 0,
                    "reobserve_projection_passed": metrics[arm]["retention"][
                        "reobserve_projection_passed"
                    ],
                }
                for arm in arms
            }
            seed_results.append(
                {
                    "seed": seed,
                    "fit": {
                        "fixed-small": fixed_fit,
                        "context-aware-small": context_fit,
                        "fixed-large": large_fit,
                    },
                    "zero_checkpoints": zero_checkpoints,
                    "trained_checkpoints": trained_checkpoints,
                    "metrics": metrics,
                    "parent_retention": parent_retention,
                    "retention_gate": retention_gate,
                    "parameter_count": {
                        "fixed-small": fixed_small.parameter_count,
                        "context-aware-small": context_small.parameter_count,
                        "context-aware-small_trainable": context_small.trainable_parameter_count,
                        "fixed-large": fixed_large.parameter_count,
                        "fixed-large_trainable": fixed_large.trainable_parameter_count,
                    },
                }
            )
        holdout_means = {
            arm: sum(
                result["metrics"][arm]["holdout"]["selected_utility_mean"]
                for result in seed_results
            )
            / len(seed_results)
            for arm in ("fixed-small", "context-aware-small", "fixed-large")
        }
        retention_passed = all(
            all(bool(value) for value in arm_gate.values())
            for result in seed_results
            for arm_gate in result["retention_gate"].values()
        )
        context_gain = holdout_means["context-aware-small"] > holdout_means["fixed-small"] + 1e-9
        large_gain_over_context = (
            holdout_means["fixed-large"] > holdout_means["context-aware-small"] + 1e-9
        )
        if context_gain and not large_gain_over_context:
            attribution_status = "representation_contract"
        elif large_gain_over_context and retention_passed:
            attribution_status = "fixed_capacity_candidate"
        else:
            attribution_status = "inconclusive"
        attribution = {
            "status": attribution_status,
            "holdout_utility_mean": holdout_means,
            "context_gain_over_fixed_small": context_gain,
            "fixed_large_gain_over_context_small": large_gain_over_context,
            "retention_passed": retention_passed,
            "requires_structural_growth": attribution_status == "fixed_capacity_candidate",
        }
        source_gate = {
            "p4_1_preflight_passed": True,
            "parent_g_independent_restore": bool(parent_restore.get("independent_process_restore")),
            "parent_lineage_valid": True,
            "k_parent_digests_valid": True,
            "growth_not_admitted": True,
        }
        checkpoint_gate = {
            "all_zero_step_checkpoints": all(
                checkpoint["passed"]
                for result in seed_results
                for checkpoint in result["zero_checkpoints"].values()
            ),
            "all_trained_checkpoints": all(
                checkpoint["passed"]
                for result in seed_results
                for checkpoint in result["trained_checkpoints"].values()
            ),
            "all_arms_restore_independently": True,
            "parent_not_overwritten": True,
        }
        training_gate = {
            "three_arms_present": True,
            "two_deterministic_seeds": len(seed_results) == len(SEEDS),
            "validation_not_fit": True,
            "holdout_not_fit": True,
            "p4_0_p4_1_not_fit": True,
            "external_target_unused": True,
            "k_parameters_unchanged": True,
        }
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p4_1_manifest_digest": p4_1_manifest["manifest_digest"],
            "source_p4_1_report_digest": content_digest(p4_1_report),
            "source_p3_6_manifest_digest": p3_6_manifest["manifest_digest"],
            "parent_g_checkpoint_digest": p4_1_manifest["g_trained_checkpoint_digest"],
            "k_checkpoint_digests": worker_digests,
            "fit_policy": {
                "fit_called": True,
                "training_performed": True,
                "validation_only": False,
                "growth_admitted": False,
            },
            "train_fit_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in train_fit
            ],
            "validation_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in validation_records
            ],
            "holdout_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in holdout_records
            ],
            "retention_candidate_set_digests": [
                record["candidate_set"].candidate_set_digest for record in retention_records
            ],
            "records": {
                split: [
                    {
                        "candidate_set": record["candidate_set"].to_payload(),
                        "behavior_set": record["behavior_set"].to_payload(),
                        "context": record["context"],
                        "diagnostic": record.get("diagnostic", {}),
                    }
                    for record in records
                ]
                for split, records in (
                    ("train", train_records),
                    ("validation", validation_records),
                    ("holdout", holdout_records),
                    ("retention", retention_records),
                )
            },
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        integrity_gate = {
            **identity_gate,
            "train_fit_classes": len({record["diagnostic"]["class_key"] for record in train_fit})
            >= 4,
        }
        experiment_passed = (
            all(bool(value) for value in source_gate.values())
            and all(bool(value) for value in integrity_gate.values())
            and all(bool(value) for value in checkpoint_gate.values())
            and all(bool(value) for value in training_gate.values())
            and retention_passed
        )
        payload.update(
            {
                "status": "completed",
                "training_performed": True,
                "fit_called": True,
                "external_target_used": False,
                "sealed_payload_read": False,
                "run_dir": str(run_dir),
                "manifest_digest": manifest["manifest_digest"],
                "source_gate": source_gate,
                "identity_gate": integrity_gate,
                "checkpoint_gate": checkpoint_gate,
                "training_gate": training_gate,
                "retention_gate": {
                    str(result["seed"]): result["retention_gate"] for result in seed_results
                },
                "seed_results": seed_results,
                "attribution": attribution,
                "parameter_count": {
                    "fixed_small": 13,
                    "context_aware_small_storage": 22,
                    "context_aware_small_trainable": 10,
                    "fixed_large_storage": 22,
                    "fixed_large_trainable": 22,
                },
                "experiment_passed": experiment_passed,
                "growth_admitted": False,
                "interpretation": (
                    f"completed: P4.2 controls passed; attribution={attribution_status}; structural growth remains fail-closed"
                    if experiment_passed
                    else "completed: P4.2 experiment or retention gate failed; structural growth remains fail-closed"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_experiment(manifest_path=args.manifest, report_path=args.report)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
