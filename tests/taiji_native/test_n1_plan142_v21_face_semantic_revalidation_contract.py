from __future__ import annotations

import pytest

from scripts.training import revalidate_taiji_n1_v21_face_semantic_audit as revalidation


def test_revalidation_rejects_replay_changes() -> None:
    prior = {
        "complete": True,
        "face_sha256": "face",
        "source_exclusion_sha256": "manifest",
        "semantic_checks": {"items": 160},
    }
    current = {"items": 159}

    with pytest.raises(ValueError, match="differs from the original"):
        revalidation._require_replay_matches(prior, current, "face", "manifest")


def test_revalidation_accepts_byte_bound_semantic_replay() -> None:
    semantic = {"items": 160, "unknown": 40}
    prior = {
        "complete": True,
        "face_sha256": "face",
        "source_exclusion_sha256": "manifest",
        "semantic_checks": semantic,
    }

    assert revalidation._require_replay_matches(prior, semantic, "face", "manifest") is True
