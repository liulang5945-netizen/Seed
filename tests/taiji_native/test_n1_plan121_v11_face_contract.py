from __future__ import annotations

from collections import Counter

from scripts.training import build_taiji_n1_v11_after_n120 as builder


def _synthetic_face() -> dict:
    return builder.build_v11_face(train_items=[], excluded_names=set(), excluded_words=set())


def test_v11_face_synthetic_contract_preserves_seed_balance_and_split() -> None:
    face = _synthetic_face()
    assert face["schema"] == "n1-hard-neighbor-v11-confirmation-v1"
    assert face["seed"] == 111
    assert len(face["episodes"]) == 40
    assert face["subset_sizes"] == {"sacrifice": 20, "confirm": 20}

    for kind in ("lookup", "revision", "transition", "unknown"):
        items = [episode for episode in face["episodes"] if episode["kind"] == kind]
        assert len(items) == 10
        assert Counter(item["subset"] for item in items) == {"sacrifice": 5, "confirm": 5}
        assert all(len(item["observations"]) == 12 for item in items)
    for kind in ("lookup", "revision"):
        items = [episode for episode in face["episodes"] if episode["kind"] == kind]
        assert Counter(len(item["answer"]) for item in items) == {6: 5, 7: 5}
        for subset, expected in (("sacrifice", {6: 3, 7: 2}), ("confirm", {6: 2, 7: 3})):
            selected = [item for item in items if item["subset"] == subset]
            assert Counter(len(item["answer"]) for item in selected) == expected
        if kind == "revision":
            for item in items:
                subject = item["entities"][0]
                targets = [row for row in item["observations"] if row.startswith(subject + " ")]
                assert len(targets) == 2
                assert len(targets[0].removesuffix(".").split()[-1]) == len(item["answer"])
                assert len(targets[1].removesuffix(".").split()[-1]) == len(item["answer"])

    checks = builder.audit_v11_face(face, train_items=[], names=set(), words=set())
    assert checks["accepted"] is True


def test_v11_face_audit_rejects_revision_stale_length_mismatch() -> None:
    face = _synthetic_face()
    item = next(
        item for item in face["episodes"] if item["kind"] == "revision" and len(item["answer"]) == 7
    )
    subject = item["entities"][0]
    target_index = next(
        index
        for index, row in enumerate(item["observations"])
        if row.startswith(subject + " uses ")
    )
    row = item["observations"][target_index]
    old_value = row.removesuffix(".").split()[-1]
    item["observations"][target_index] = f"{subject} uses {old_value[:6]}."

    checks = builder.audit_v11_face(face, train_items=[], names=set(), words=set())

    assert checks["checks"]["sacrifice_confirm_length_split"] is False
    assert checks["accepted"] is False
