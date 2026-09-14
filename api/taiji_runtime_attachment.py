"""Default-runtime K/G state attachment (frozen consumption contract).

Runtime-owned implementation of the ``taiji-default-runtime-rollout-attachment-v1``
contract validated by the M5 default-runtime rollout review.  It loads the
P4.14 joint-course artifacts (post-K K1/K2 workers + per-cell projected
extended G head) into the product runtime through an explicitly opt-in,
fail-closed attachment.  This module never trains, never writes any
artifact or default checkpoint, and never overrides the embedded safety
invariants - it reads them from the checkpoints and verifies them against
the frozen preregistration pins.

Load order is frozen (review preregistration section 2): manifest
self-digest -> P4.14 lineage -> per-artifact digest -> embedded invariants
-> independent-process restore -> tamper / wrong-cell-mixing rejection.
Preregistration:
``plans/reference/M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import (  # noqa: E402
    StructuredSemanticLearner,
    StructuredSemanticTransitionLearner,
    content_digest,
)
from taiji.g_selection_extended import ExtendedGSelectionLearner  # noqa: E402

ATTACHMENT_FORMAT = "taiji-default-runtime-rollout-attachment-v1"
ATTACHMENT_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_default_runtime_rollout_attachment_v1.json"
)
P4_14_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_p4_14_joint_course_20260911.json"
P4_14_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_14_joint_course_manifest_v1.json"
)
P4_1_MANIFEST = (
    PROJECT_ROOT / "plans" / "manifests" / "taiji_m5_k_p4_1_context_contract_manifest_v1.json"
)

K_PARAMETER_TOTAL = 5648
G_PARAMETER_COUNT = 17
G_SELECTION_MARGIN = 0.05
# Frozen embedded invariants (preregistration section 2/3): the K2 transition
# learner uses a lower fact threshold than the K1 semantic learner.
K_INVARIANTS: dict[str, dict[str, float]] = {
    "k1": {"confidence_floor": 0.55, "fact_threshold": 0.65, "ambiguity_ceiling": 0.12},
    "k2": {"confidence_floor": 0.55, "fact_threshold": 0.55, "ambiguity_ceiling": 0.12},
}
G_CONFIDENCE_FLOOR = 0.55
ROLES = ("k1", "k2", "g")


class AttachmentRefused(ValueError):
    """Raised when any frozen consumption-contract check fails."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AttachmentRefused(f"expected a JSON object at {path}")
    return payload


def _load_mapping(path: Path) -> dict[str, Any]:
    import torch

    # Digest validation after deserialization cannot make arbitrary pickle safe.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise AttachmentRefused(f"expected a mapping checkpoint at {path}")
    return {str(key): value for key, value in payload.items()}


def _digest_without(payload: Mapping[str, Any], key: str) -> str:
    return content_digest({name: value for name, value in payload.items() if name != key})


def _artifact_digest(role: str, payload: Mapping[str, Any]) -> str:
    """Digest in the convention that produced the attachment pin.

    K worker checkpoints carry no self-digest key, so the pin is the full
    content digest.  The extended G checkpoint embeds its own
    ``checkpoint_digest`` (the digest of the payload without that key), and
    the P4.14 report pinned exactly that value.
    """

    if role == "g":
        return _digest_without(payload, "checkpoint_digest")
    return content_digest(payload)


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("format") != ATTACHMENT_FORMAT:
        raise AttachmentRefused(f"unsupported attachment manifest format: {manifest.get('format')}")
    if _digest_without(manifest, "manifest_digest") != manifest.get("manifest_digest"):
        raise AttachmentRefused("attachment manifest self-digest mismatch")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise AttachmentRefused("attachment manifest has no cells")
    return manifest


def _verify_lineage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    p4_14_report = _load_json(P4_14_REPORT)
    if content_digest(p4_14_report) != manifest.get("source_p4_14_report_digest"):
        raise AttachmentRefused("P4.14 report digest drifted from the manifest")
    p4_14_manifest = _load_json(P4_14_MANIFEST)
    if _digest_without(p4_14_manifest, "manifest_digest") != manifest.get(
        "source_p4_14_manifest_digest"
    ):
        raise AttachmentRefused("P4.14 manifest digest drifted from the manifest")
    p4_1_manifest = _load_json(P4_1_MANIFEST)
    if _digest_without(p4_1_manifest, "manifest_digest") != manifest.get(
        "source_p4_1_manifest_digest"
    ):
        raise AttachmentRefused("P4.1 manifest digest drifted from the manifest")
    return {
        "p4_14_outcome": str(p4_14_report.get("outcome")),
        "p4_14_passing_cells": int(p4_14_report.get("passing_cells", 0)),
        "g_parent_manifest_digest_expected": str(p4_1_manifest["source_p3_2_manifest_digest"]),
    }


def _cell_artifacts(manifest: Mapping[str, Any], cell_index: int) -> dict[str, Any]:
    cells = manifest["cells"]
    if not 0 <= int(cell_index) < len(cells):
        raise AttachmentRefused(f"cell index out of range: {cell_index}")
    artifacts: dict[str, Any] = cells[int(cell_index)]["artifacts"]
    return artifacts


def _load_cell_payloads(manifest: Mapping[str, Any], cell_index: int) -> dict[str, dict[str, Any]]:
    artifacts = _cell_artifacts(manifest, cell_index)
    shared = manifest["consumption_contract"]["post_k_k_digests_shared"]
    payloads: dict[str, dict[str, Any]] = {}
    verified: dict[str, dict[str, str]] = {}
    for role in ROLES:
        artifact = artifacts[role]
        path = Path(str(artifact["path"]))
        if not path.is_file():
            raise AttachmentRefused(f"cell {cell_index} artifact {role} missing: {path}")
        payload = _load_mapping(path)
        digest = _artifact_digest(role, payload)
        if digest != str(artifact["digest"]):
            raise AttachmentRefused(
                f"cell {cell_index} artifact {role} digest mismatch: "
                f"{digest} != {artifact['digest']}"
            )
        payloads[role] = payload
        verified[role] = {"path": str(path), "digest": digest}
    if verified["k1"]["digest"] != str(shared["k1"]) or verified["k2"]["digest"] != str(
        shared["k2"]
    ):
        raise AttachmentRefused("cell post-K workers drift from the shared anchors")
    return {"payloads": payloads, "verified": verified}


def _invariant_checks(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for role in ("k1", "k2"):
        for name, expected in K_INVARIANTS[role].items():
            actual = payloads[role].get(name)
            checks[f"{role}_{name}"] = actual is not None and float(actual) == float(expected)
    g_payload = payloads["g"]
    checks["g_confidence_floor"] = float(g_payload["confidence_floor"]) == float(G_CONFIDENCE_FLOOR)
    checks["g_selection_margin"] = float(g_payload["selection_margin"]) == G_SELECTION_MARGIN
    checks["g_parameter_count"] = int(g_payload["parameter_count"]) == G_PARAMETER_COUNT
    return checks


def _build_learners(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[
    StructuredSemanticLearner, StructuredSemanticTransitionLearner, ExtendedGSelectionLearner
]:
    semantic = StructuredSemanticLearner.from_checkpoint(
        copy.deepcopy(dict(payloads["k1"])), device="cpu"
    )
    transition = StructuredSemanticTransitionLearner.from_checkpoint(
        copy.deepcopy(dict(payloads["k2"])), device="cpu"
    )
    g_learner = ExtendedGSelectionLearner.from_checkpoint(
        copy.deepcopy(dict(payloads["g"])), device="cpu"
    )
    return semantic, transition, g_learner


def _learner_checks(
    semantic: StructuredSemanticLearner,
    transition: StructuredSemanticTransitionLearner,
    g_learner: ExtendedGSelectionLearner,
) -> dict[str, bool]:
    return {
        "k_parameter_total": int(semantic.parameter_count + transition.parameter_count)
        == K_PARAMETER_TOTAL,
        "g_parameter_count_learner": int(g_learner.parameter_count) == G_PARAMETER_COUNT,
    }


def _tamper_probes_rejected(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for role in ("k1", "k2"):
        tampered = copy.deepcopy(dict(payloads[role]))
        tampered["source_digest"] = "tampered-source-digest"
        try:
            if role == "k1":
                StructuredSemanticLearner.from_checkpoint(tampered, device="cpu")
            else:
                StructuredSemanticTransitionLearner.from_checkpoint(tampered, device="cpu")
        except ValueError:
            results[f"{role}_tamper_rejected"] = True
        else:
            results[f"{role}_tamper_rejected"] = False
    tampered_g = copy.deepcopy(dict(payloads["g"]))
    tampered_g["revision"] = int(tampered_g.get("revision", 0)) + 1
    try:
        ExtendedGSelectionLearner.from_checkpoint(tampered_g, device="cpu")
    except ValueError:
        results["g_tamper_rejected"] = True
    else:
        results["g_tamper_rejected"] = False
    return results


def _mixing_probe_rejected(manifest: Mapping[str, Any]) -> bool:
    if len(manifest["cells"]) < 2:
        return False
    mixed = copy.deepcopy(dict(manifest))
    mixed["cells"] = copy.deepcopy(manifest["cells"])
    mixed["cells"][0]["artifacts"]["g"]["digest"] = mixed["cells"][1]["artifacts"]["g"]["digest"]
    try:
        _load_cell_payloads(mixed, 0)
    except AttachmentRefused:
        return True
    return False


def verify_attachment(manifest_path: Path, cell_index: int) -> dict[str, Any]:
    """Full fail-closed preflight of one cell without keeping learner state."""

    manifest = _load_manifest(manifest_path)
    lineage = _verify_lineage(manifest)
    cell = _load_cell_payloads(manifest, cell_index)
    payloads = cell["payloads"]
    invariant_checks = _invariant_checks(payloads)
    if not all(invariant_checks.values()):
        raise AttachmentRefused(
            f"cell {cell_index} embedded invariant check failed: "
            f"{[name for name, ok in invariant_checks.items() if not ok]}"
        )
    semantic, transition, g_learner = _build_learners(payloads)
    learner_checks = _learner_checks(semantic, transition, g_learner)
    if not all(learner_checks.values()):
        raise AttachmentRefused("learner parameter-count check failed")
    g_payload = payloads["g"]
    if str(g_payload["parent_manifest_digest"]) != lineage["g_parent_manifest_digest_expected"]:
        raise AttachmentRefused("G parent manifest digest drifted from P4.1 lineage")
    tamper = _tamper_probes_rejected(payloads)
    if not all(tamper.values()):
        raise AttachmentRefused("tamper probe was not rejected")
    mixing = _mixing_probe_rejected(manifest)
    if not mixing:
        raise AttachmentRefused("wrong-cell mixing probe was not rejected")
    return {
        "passed": True,
        "cell_index": int(cell_index),
        "manifest_digest": str(manifest["manifest_digest"]),
        "lineage": lineage,
        "artifact_verification": cell["verified"],
        "invariant_checks": invariant_checks,
        "learner_checks": learner_checks,
        "tamper_rejection": tamper,
        "wrong_cell_mixing_rejected": mixing,
        "feature_source_state_digest": str(g_payload["feature_source_state_digest"]),
    }


def _independent_restore(manifest_path: Path, cell_index: int) -> dict[str, Any]:
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--verify-attachment",
            str(manifest_path),
            str(int(cell_index)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, Any] = {
        "returncode": int(child.returncode),
        "independent_process_restore": False,
        "stdout": child.stdout[-2000:],
        "stderr": child.stderr[-2000:],
    }
    if child.returncode != 0:
        return result
    try:
        decoded = json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
        return result
    result.update(decoded)
    result["independent_process_restore"] = bool(decoded.get("passed"))
    return result


class AttachedKGState:
    """Runtime-owned consumer state loaded through the frozen contract."""

    def __init__(
        self,
        *,
        cell_index: int,
        manifest_digest: str,
        manifest_path: Path,
        verification: dict[str, Any],
        semantic: StructuredSemanticLearner,
        transition: StructuredSemanticTransitionLearner,
        g_learner: ExtendedGSelectionLearner,
    ) -> None:
        self.cell_index = int(cell_index)
        self.manifest_digest = str(manifest_digest)
        self.manifest_path = Path(manifest_path)
        self.verification = verification
        self.semantic = semantic
        self.transition = transition
        self.g_learner = g_learner
        self.attached_at_epoch = int(time.time())

    # -- typed readout surface (the product-facing API) --------------------

    def k1_predict(self, percept: Any) -> Any:
        return self.semantic.predict(percept)

    def k2_predict(self, world: Any, event: Any) -> Any:
        return self.transition.predict(world, event)

    def g_select(self, candidate_set: Any) -> Any:
        return self.g_learner.select(candidate_set)

    # -- status ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "attached": True,
            "cell_index": self.cell_index,
            "manifest_digest": self.manifest_digest,
            "manifest_path": str(self.manifest_path),
            "artifact_digests": {
                role: info["digest"]
                for role, info in self.verification["artifact_verification"].items()
            },
            "invariant_checks": dict(self.verification["invariant_checks"]),
            "parameter_counts": {
                "k_total": K_PARAMETER_TOTAL,
                "g_extended": G_PARAMETER_COUNT,
            },
            "feature_source_state_digest": self.verification["feature_source_state_digest"],
            "readout_surface": ["k1_predict", "k2_predict", "g_select"],
            "fit_called": False,
            "attached_at_epoch": self.attached_at_epoch,
        }


def attach_cell(manifest_path: Path, cell_index: int) -> AttachedKGState:
    """Fail-closed, atomic attachment of one cell's artifacts.

    Raises :class:`AttachmentRefused` on any frozen-contract failure; the
    runtime stays unattached unless every check passes.
    """

    manifest = _load_manifest(manifest_path)
    lineage = _verify_lineage(manifest)
    cell = _load_cell_payloads(manifest, cell_index)
    payloads = cell["payloads"]
    invariant_checks = _invariant_checks(payloads)
    if not all(invariant_checks.values()):
        raise AttachmentRefused(
            f"cell {cell_index} embedded invariant check failed: "
            f"{[name for name, ok in invariant_checks.items() if not ok]}"
        )
    semantic, transition, g_learner = _build_learners(payloads)
    learner_checks = _learner_checks(semantic, transition, g_learner)
    if not all(learner_checks.values()):
        raise AttachmentRefused("learner parameter-count check failed")
    g_payload = payloads["g"]
    if str(g_payload["parent_manifest_digest"]) != lineage["g_parent_manifest_digest_expected"]:
        raise AttachmentRefused("G parent manifest digest drifted from P4.1 lineage")
    tamper = _tamper_probes_rejected(payloads)
    if not all(tamper.values()):
        raise AttachmentRefused("tamper probe was not rejected")
    if not _mixing_probe_rejected(manifest):
        raise AttachmentRefused("wrong-cell mixing probe was not rejected")
    restore = _independent_restore(manifest_path, cell_index)
    if not restore.get("independent_process_restore"):
        raise AttachmentRefused(
            f"independent-process restore failed: {restore.get('stderr', '')[-300:]}"
        )
    verification = {
        "passed": True,
        "cell_index": int(cell_index),
        "manifest_digest": str(manifest["manifest_digest"]),
        "lineage": lineage,
        "artifact_verification": cell["verified"],
        "invariant_checks": invariant_checks,
        "learner_checks": learner_checks,
        "tamper_rejection": tamper,
        "wrong_cell_mixing_rejected": True,
        "independent_restore": restore,
        "feature_source_state_digest": str(g_payload["feature_source_state_digest"]),
    }
    return AttachedKGState(
        cell_index=int(cell_index),
        manifest_digest=str(manifest["manifest_digest"]),
        manifest_path=manifest_path,
        verification=verification,
        semantic=semantic,
        transition=transition,
        g_learner=g_learner,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-attachment", nargs=2, metavar=("MANIFEST", "CELL"))
    args = parser.parse_args(argv)
    if args.verify_attachment is not None:
        manifest_path = Path(args.verify_attachment[0])
        cell_index = int(args.verify_attachment[1])
        try:
            result = verify_attachment(manifest_path, cell_index)
        except AttachmentRefused as exc:
            print(
                json.dumps(
                    {"passed": False, "refused": True, "reason": str(exc)},
                    ensure_ascii=False,
                )
            )
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    parser.print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
