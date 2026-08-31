"""Taiji-owned grounding from semantic Workbench steps to live bindings.

The provider may describe an operation, a file, or an edit request, but it
cannot supply the final editor language, workspace digest, or executable
patch.  This module is deliberately runtime-agnostic: it receives only a
Workbench environment and a semantic step, then derives the binding from
current sensors and returns the evidence needed by the executor.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def ground_natural_language_workbench_step(
    environment: Any,
    step: Any,
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    str,
    dict[str, Any] | None,
    str,
]:
    """Ground one semantic step against current Workbench evidence.

    The returned tuple is ``(bindings, error, evidence, evidence_key)``.
    Empty ``error`` means that the bindings are safe for Taiji's planner to
    turn into an executive intent.  All executable values are derived here
    from live Workbench sensors; semantic provider evidence remains declarative.
    """

    grounded = environment.capability_snapshot.ground_semantic_step(
        step.semantic_slots,
        allow_reversible_ui=True,
        allow_controlled_write=True,
    )
    if not grounded:
        return {}, "semantic_grounding_unresolved", None, ""
    if len(grounded) > 1:
        return {}, "semantic_grounding_ambiguous", None, ""

    normalized = {
        str(capability_id): tuple(dict(parameters) for parameters in bindings)
        for capability_id, bindings in grounded.items()
    }
    language_evidence: dict[str, Any] | None = None
    language_bindings = normalized.get("editor.set_language")
    if language_bindings is not None:
        rebound: list[dict[str, Any]] = []
        for binding in language_bindings:
            path = str(binding.get("path", "")).strip()
            if not path:
                return {}, "semantic_grounding_unresolved", None, ""
            assessment = environment.resolve_programming_language_evidence({"path": path})
            selection_state = str(assessment.get("selection_state", "unknown"))
            if selection_state == "user_override" and not bool(
                binding.get("user_override", False)
            ):
                return {}, "user_override_has_priority", assessment, "language_evidence"
            if selection_state in {"ambiguous", "unknown"}:
                return {}, "language_evidence_ambiguous", assessment, "language_evidence"
            language_id = str(assessment.get("programming_language_id", "")).strip()
            if not language_id:
                return {}, "language_evidence_unresolved", assessment, "language_evidence"
            rebound.append(
                {
                    **binding,
                    "programming_language_id": language_id,
                }
            )
            language_evidence = assessment
        normalized["editor.set_language"] = tuple(rebound)
        return normalized, "", language_evidence, "language_evidence"

    patch_bindings = normalized.get("workspace.apply_patch")
    if patch_bindings is None:
        return normalized, "", None, ""
    if len(patch_bindings) != 1:
        return {}, "semantic_grounding_ambiguous", None, ""
    edit = step.semantic_slots.get("edit")
    if not isinstance(edit, Mapping):
        return {}, "edit_evidence_unresolved", None, ""
    if str(edit.get("kind", "")).strip() != "replace_text":
        return {}, "edit_kind_unsupported", None, ""
    find_text = edit.get("find")
    replacement_text = edit.get("replace", "")
    if not isinstance(find_text, str) or not find_text:
        return {}, "edit_target_unresolved", None, ""
    if not isinstance(replacement_text, str):
        return {}, "edit_replacement_invalid", None, ""
    path = str(patch_bindings[0].get("path", "")).strip()
    if not path:
        return {}, "semantic_grounding_unresolved", None, ""
    current = environment.read_workspace_evidence({"path": path})
    if bool(current.get("truncated", False)):
        return {}, "edit_source_truncated", current, "patch_evidence"
    content = current.get("content")
    if not isinstance(content, str):
        return {}, "edit_source_not_text", current, "patch_evidence"
    match_count = content.count(find_text)
    patch_evidence = {
        "path": path,
        "before_digest": str(current["digest"]),
        "match_count": match_count,
        "edit_kind": "replace_text",
    }
    if match_count == 0:
        return {}, "edit_target_not_found", patch_evidence, "patch_evidence"
    if match_count != 1:
        return {}, "edit_target_ambiguous", patch_evidence, "patch_evidence"
    start = content.index(find_text)
    updated = content[:start] + replacement_text + content[start + len(find_text) :]
    expected_after_digest = hashlib.sha256(updated.encode("utf-8")).hexdigest()
    normalized["workspace.apply_patch"] = (
        {
            **patch_bindings[0],
            "before_digest": str(current["digest"]),
            "patch": {
                "kind": "text_replace",
                "operations": [
                    {
                        "start": start,
                        "end": start + len(find_text),
                        "text": replacement_text,
                    }
                ],
            },
            "expected_after_digest": expected_after_digest,
        },
    )
    return (
        normalized,
        "",
        {
            **patch_evidence,
            "expected_after_digest": expected_after_digest,
        },
        "patch_evidence",
    )
