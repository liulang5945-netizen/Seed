"""Build and audit the P2.3 recovery-continuation data contract.

The existing P1 R rows stop at a missing target.  This contract adds the
next observable event without pretending that the host recovery route is a
learned action:

``missing target -> typed abstention -> workspace.list(path='.') ->
candidate observation -> read-only inspection``

The first recovery step is host policy and is explicitly non-trainable.  The
candidate observation is a separate, high-evidence K1/K2 example and is the
only part marked fit-eligible.  This script only builds and audits JSON
artifacts; it never calls ``fit`` and never reads sealed payloads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.build_taiji_m5_k_p1_data import (  # noqa: E402
    _observations,
    _prepare_workspace,
)
from scripts.training.eval_taiji_m5_k1_skill_composition import (  # noqa: E402
    _schema,
    _semantic_example,
    _transition_examples,
    _world,
)
from scripts.training.eval_taiji_m5_k_p2_validation_pilot import (  # noqa: E402
    MODEL_SEED,
    P1_MANIFEST,
    WORKER_ROOT,
    _context,
)
from taiji import (  # noqa: E402
    StructuredSemanticExample,
    StructuredSemanticTransitionExample,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-k-p2-3-recovery-continuation-contract-v1"
MANIFEST_FORMAT = "taiji-m5-k-p2-3-recovery-continuation-manifest-v1"
VERSION = 1
CONFIDENCE_FLOOR = 0.55
RECOVERY_ROUTE = "workspace.list"
RECOVERY_PATH = "."
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p2_3_recovery_continuation_manifest_v1.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_m5_k_p2_3_recovery_continuation_contract_20260910.json"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"

TRAIN_SPECS = (
    (0, "missing_01.txt", "python_00.py"),
    (1, "missing_02.txt", "python_01.py"),
    (2, "missing_01.txt", "python_00.py"),
    (3, "missing_02.txt", "python_01.py"),
    (4, "missing_01.txt", "python_00.py"),
    (5, "missing_02.txt", "python_01.py"),
)
VALIDATION_SPECS = (
    (0, "missing_validation.txt", "python_04.py"),
    (1, "missing_validation.txt", "python_05.py"),
)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"__tensor__": value.detach().cpu().tolist()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _restore_tensors(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__tensor__"}:
            return torch.tensor(value["__tensor__"], dtype=torch.float32)
        return {key: _restore_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_tensors(item) for item in value]
    return value


def _build_record(
    *,
    scratch: Path,
    split: str,
    index: int,
    missing_path: str,
    candidate_path: str,
    project_id: str,
    task_seed: int,
    source_manifest_digest: str,
    parent_digest: str,
    worker_bundle_digest: str,
) -> dict[str, Any]:
    root = scratch / f"{split}-{index:04d}"
    root.mkdir(parents=True, exist_ok=False)
    _prepare_workspace(
        root,
        task_seed=task_seed,
        first_path=missing_path,
        state_profile="recovery-no-selection",
    )
    schema = _schema()
    initial_observations = _observations(
        root,
        paths=(missing_path, "python_00.py", "rust_00.rs"),
        split=f"p2-3-{split}-{index:04d}-initial",
        project_id=project_id,
        state_profile="recovery-no-selection",
        schema=schema,
    )
    candidate_observations = _observations(
        root,
        paths=(candidate_path, "rust_00.rs", "python_01.py"),
        split=f"p2-3-{split}-{index:04d}-candidate",
        project_id=project_id,
        state_profile="resolved-language",
        schema=schema,
    )
    anchor = initial_observations[0]
    initial = initial_observations[1]
    candidate = candidate_observations[1]
    transition = _transition_examples(
        (anchor, initial, candidate),
        split=f"p2-3-{split}-{index:04d}",
    )[1]
    candidate_semantic = _semantic_example(
        candidate,
        split=f"p2-3-{split}-{index:04d}",
        tick=2,
    )
    if transition.after.tick != candidate_semantic.percept.observation_tick:
        raise ValueError("recovery continuation candidate ticks are not aligned")
    if candidate_semantic.content.content_id != transition.content.content_id:
        raise ValueError("K1 and K2 candidate content targets diverge")
    initial_payload = _jsonable(initial.to_payload())
    candidate_payload = _jsonable(candidate.to_payload())
    semantic_payload = _jsonable(candidate_semantic.to_payload())
    transition_payload = _jsonable(transition.to_payload())
    record = {
        "record_id": f"p2-3-recovery-{split}-{index:04d}",
        "split": split,
        "index": int(index),
        "class_key": "R",
        "project_id": project_id,
        "template_family_id": content_digest(
            {
                "format": "taiji-m5-k-p2-3-recovery-template-v1",
                "split": split,
                "missing_path": missing_path,
                "candidate_path": candidate_path,
            }
        ),
        "source_manifest_digest": source_manifest_digest,
        "parent_checkpoint_digest": parent_digest,
        "worker_bundle_digest": worker_bundle_digest,
        "stages": [
            "missing_target",
            "typed_abstention",
            "workspace.list",
            "candidate_observation",
            "read_only_inspection",
        ],
        "host_recovery": {
            "owner": "host_policy",
            "model_owned": False,
            "oracle_control": False,
            "capability": RECOVERY_ROUTE,
            "parameters": {"path": RECOVERY_PATH},
            "expected_next_step": "candidate_observation",
        },
        "initial": {
            "fit_eligible": False,
            "reason_code": "missing_target",
            "next_step": RECOVERY_ROUTE,
            "observation": initial_payload,
            "world": _jsonable(_world(initial, tick=1).to_payload()),
            "percept": _jsonable(initial.to_percept_event(tick=1).to_payload()),
        },
        "candidate": {
            "fit_eligible": True,
            "observation": candidate_payload,
            "world": _jsonable(_world(candidate, tick=2).to_payload()),
            "semantic_example": semantic_payload,
            "transition_example": transition_payload,
            "k1_input_digest": candidate_semantic.input_digest,
            "k2_input_digest": transition.input_digest,
        },
    }
    record["record_digest"] = content_digest(record)
    return record


def _audit_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_split: str,
    expected_projects: set[str],
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    candidate_inputs: list[str] = []
    candidate_templates: list[str] = []
    candidate_paths: list[str] = []
    for record in records:
        record_id = str(record.get("record_id"))
        record_without_digest = dict(record)
        record_digest = record_without_digest.pop("record_digest", None)
        if record_digest != content_digest(record_without_digest):
            errors.append({"record_id": record_id, "reason": "record_digest_mismatch"})
        initial = record.get("initial")
        candidate = record.get("candidate")
        host_recovery = record.get("host_recovery")
        if not isinstance(initial, Mapping) or not isinstance(candidate, Mapping):
            errors.append({"record_id": record_id, "reason": "missing_stage_payload"})
            continue
        initial_observation = initial.get("observation")
        candidate_observation = candidate.get("observation")
        initial_percept = initial.get("percept")
        candidate_semantic_payload = candidate.get("semantic_example")
        if not isinstance(initial_observation, Mapping) or not isinstance(
            candidate_observation, Mapping
        ):
            errors.append({"record_id": record_id, "reason": "missing_observation_payload"})
            continue
        if not isinstance(initial_percept, Mapping) or not isinstance(
            candidate_semantic_payload, Mapping
        ):
            errors.append({"record_id": record_id, "reason": "missing_percept_payload"})
            continue
        candidate_percept = candidate_semantic_payload.get("percept")
        if not isinstance(candidate_percept, Mapping):
            errors.append({"record_id": record_id, "reason": "missing_candidate_percept"})
            continue
        if str(record.get("split")) != expected_split:
            errors.append({"record_id": record_id, "reason": "split_mismatch"})
        if str(record.get("project_id")) not in expected_projects:
            errors.append({"record_id": record_id, "reason": "project_mismatch"})
        if initial_observation.get("read_success") is not False:
            errors.append({"record_id": record_id, "reason": "initial_target_is_not_missing"})
        if float(initial_percept.get("confidence", 1.0)) >= CONFIDENCE_FLOOR:
            errors.append({"record_id": record_id, "reason": "initial_confidence_not_low"})
        if initial.get("fit_eligible") is not False:
            errors.append({"record_id": record_id, "reason": "initial_marked_fit_eligible"})
        if not isinstance(host_recovery, Mapping):
            errors.append({"record_id": record_id, "reason": "missing_host_recovery"})
        else:
            if host_recovery.get("model_owned") is not False:
                errors.append({"record_id": record_id, "reason": "recovery_marked_model_owned"})
            if host_recovery.get("capability") != RECOVERY_ROUTE:
                errors.append({"record_id": record_id, "reason": "recovery_route_mismatch"})
            if host_recovery.get("parameters") != {"path": RECOVERY_PATH}:
                errors.append({"record_id": record_id, "reason": "recovery_scope_mismatch"})
        if candidate.get("fit_eligible") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_not_fit_eligible"})
        if candidate_observation.get("read_success") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_read_not_successful"})
        if candidate_observation.get("file_is_file") is not True:
            errors.append({"record_id": record_id, "reason": "candidate_is_not_file"})
        if float(candidate_percept.get("confidence", 0.0)) < CONFIDENCE_FLOOR:
            errors.append({"record_id": record_id, "reason": "candidate_confidence_below_floor"})
        if candidate_observation.get("selection_state") != "resolved":
            errors.append({"record_id": record_id, "reason": "candidate_language_not_resolved"})
        if initial_observation.get("observation_digest") == candidate_observation.get(
            "observation_digest"
        ):
            errors.append({"record_id": record_id, "reason": "stage_observation_digest_reused"})
        try:
            semantic = StructuredSemanticExample.from_payload(
                _restore_tensors(candidate["semantic_example"])
            )
            transition = StructuredSemanticTransitionExample.from_payload(
                _restore_tensors(candidate["transition_example"])
            )
            if semantic.input_digest != candidate.get("k1_input_digest"):
                errors.append({"record_id": record_id, "reason": "k1_input_digest_mismatch"})
            if transition.input_digest != candidate.get("k2_input_digest"):
                errors.append({"record_id": record_id, "reason": "k2_input_digest_mismatch"})
            if semantic.content.content_id != "content:inspect-language":
                errors.append({"record_id": record_id, "reason": "candidate_k1_target_mismatch"})
            if transition.content.content_id != "content:inspect-language":
                errors.append({"record_id": record_id, "reason": "candidate_k2_target_mismatch"})
            if transition.before.tick != 1 or transition.after.tick != 2:
                errors.append({"record_id": record_id, "reason": "transition_tick_mismatch"})
            candidate_inputs.extend([semantic.input_digest, transition.input_digest])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(
                {"record_id": record_id, "reason": f"example_roundtrip_failed:{type(exc).__name__}"}
            )
        candidate_templates.append(str(record.get("template_family_id")))
        candidate_paths.append(str(candidate_observation.get("path")))
    return {
        "expected_split": expected_split,
        "record_count": len(records),
        "project_ids": sorted({str(record.get("project_id")) for record in records}),
        "unique_candidate_input_digest_count": len(set(candidate_inputs)),
        "unique_template_family_count": len(set(candidate_templates)),
        "unique_candidate_path_count": len(set(candidate_paths)),
        "errors": errors,
        "passed": not errors and bool(records),
    }


def run_contract(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    report_path: Path = DEFAULT_REPORT,
    p1_manifest_path: Path = P1_MANIFEST,
) -> dict[str, Any]:
    started = time.perf_counter()
    scratch = DEFAULT_OUTPUT_ROOT / f"_taiji_m5_k_p2_3_contract_{uuid4().hex}"
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "training_performed": False,
        "sealed_payload_read": False,
        "can_promote": False,
        "manifest": str(manifest_path),
        "source_p1_manifest": str(p1_manifest_path),
    }
    try:
        p1_manifest = json.loads(p1_manifest_path.read_text(encoding="utf-8"))
        if p1_manifest.get("format") != "taiji-m5-k-p1-data-manifest-v2":
            raise ValueError("P2.3 requires the passed P1 v2 manifest")
        _artifacts, parent_digest, bundle, _projector = _context(
            worker_root=WORKER_ROOT,
            model_seed=MODEL_SEED,
        )
        scratch.mkdir(parents=True, exist_ok=False)
        train_records = [
            _build_record(
                scratch=scratch / "train",
                split="train",
                index=index,
                missing_path=missing_path,
                candidate_path=candidate_path,
                project_id="p2-3-recovery-train-project",
                task_seed=3100 + index,
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
            )
            for index, missing_path, candidate_path in TRAIN_SPECS
        ]
        validation_records = [
            _build_record(
                scratch=scratch / "validation",
                split="validation",
                index=index,
                missing_path=missing_path,
                candidate_path=candidate_path,
                project_id="p2-3-recovery-validation-project",
                task_seed=4100 + index,
                source_manifest_digest=str(p1_manifest["manifest_digest"]),
                parent_digest=parent_digest,
                worker_bundle_digest=bundle.bundle_digest,
            )
            for index, missing_path, candidate_path in VALIDATION_SPECS
        ]
        train_audit = _audit_records(
            train_records,
            expected_split="train",
            expected_projects={"p2-3-recovery-train-project"},
        )
        validation_audit = _audit_records(
            validation_records,
            expected_split="validation",
            expected_projects={"p2-3-recovery-validation-project"},
        )
        train_paths = {
            str(record["candidate"]["observation"]["path"]) for record in train_records
        }
        validation_paths = {
            str(record["candidate"]["observation"]["path"]) for record in validation_records
        }
        split_isolation = {
            "project_disjoint": set(train_audit["project_ids"]).isdisjoint(
                validation_audit["project_ids"]
            ),
            "candidate_path_disjoint": train_paths.isdisjoint(validation_paths),
            "template_disjoint": set(
                str(record["template_family_id"]) for record in train_records
            ).isdisjoint(
                str(record["template_family_id"]) for record in validation_records
            ),
        }
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": VERSION,
            "source_p1_manifest_digest": str(p1_manifest["manifest_digest"]),
            "parent_checkpoint_digest": parent_digest,
            "worker_bundle_digest": bundle.bundle_digest,
            "confidence_floor": CONFIDENCE_FLOOR,
            "host_recovery": {
                "capability": RECOVERY_ROUTE,
                "parameters": {"path": RECOVERY_PATH},
                "model_owned": False,
            },
            "train_records": train_records,
            "validation_records": validation_records,
        }
        manifest["manifest_digest"] = content_digest(manifest)
        _write_json_atomic(manifest_path, manifest)
        payload.update(
            {
                "status": "completed",
                "manifest_digest": manifest["manifest_digest"],
                "train_audit": train_audit,
                "validation_audit": validation_audit,
                "split_isolation": split_isolation,
                "checks": {
                    "manifest_format": manifest["format"] == MANIFEST_FORMAT,
                    "source_p1_manifest": p1_manifest["manifest_digest"]
                    == manifest["source_p1_manifest_digest"],
                    "train_contract": bool(train_audit["passed"]),
                    "validation_contract": bool(validation_audit["passed"]),
                    "project_isolation": bool(split_isolation["project_disjoint"]),
                    "candidate_path_isolation": bool(split_isolation["candidate_path_disjoint"]),
                    "template_isolation": bool(split_isolation["template_disjoint"]),
                    "no_fit": True,
                    "no_sealed_payload": True,
                },
                "interpretation": (
                    "data-contract-only; initial recovery is host policy and excluded from fit; "
                    "candidate K1/K2 examples are fit-eligible but were not trained in this run"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        payload["can_promote"] = False
    except Exception as exc:
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    _write_json_atomic(report_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--p1-manifest", type=Path, default=P1_MANIFEST)
    args = parser.parse_args()
    payload = run_contract(
        manifest_path=args.manifest,
        report_path=args.report,
        p1_manifest_path=args.p1_manifest,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "report": str(args.report),
                "status": payload["status"],
                "training_performed": payload["training_performed"],
                "can_promote": payload["can_promote"],
                "error": payload.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
