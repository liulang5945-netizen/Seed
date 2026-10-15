"""Offline attribution must preserve paired identity and the exact training schedule."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_relation_seed_drift as audit


def rows():
    result = []
    for step in audit.STEPS:
        for index, kind in enumerate(("lookup", "revision")):
            exact = (step != 6144) if index == 0 else (step == 6144)
            result.append(
                dict(
                    seed=13,
                    arm="position",
                    evaluation_set="rename",
                    completed_updates=step,
                    id=f"train-{index}",
                    kind=kind,
                    target_symbols=[97, 256],
                    free_exact=exact,
                    teacher_exact=exact,
                    teacher_ce=0.1 if exact else 1.0,
                    raw_hex="61" if exact else "62",
                    generated_steps=2,
                    stop_reason="boundary",
                )
            )
    return result


def test_offline_flips_preserve_direction_and_teacher_evidence():
    result = audit.trajectory(rows(), 13, "position", "rename", 2)
    late = result["transitions"][-1]
    assert late["counts"] == {"lost": 1, "gained": 1}
    lost = next(r for r in late["flips"] if r["direction"] == "lost")
    assert lost["kind"] == "lookup" and lost["teacher_before"] and not lost["teacher_after"]
    assert lost["ce_after"] > lost["ce_before"]
    assert [s["free_exact"] for s in result["milestones"]] == [1, 1, 1]


def test_offline_trajectory_rejects_duplicates_and_changed_targets():
    original = rows()
    with pytest.raises(ValueError, match="unique"):
        audit.trajectory(original + [original[0]], 13, "position", "rename", 2)
    changed = deepcopy(original)
    changed[-1]["target_symbols"] = [98, 256]
    with pytest.raises(ValueError, match="target changed"):
        audit.trajectory(changed, 13, "position", "rename", 2)


def test_loss_stream_maps_even_odd_views_and_all_qa_pairs():
    train = [dict(kind=audit.KINDS[i % 4]) for i in range(192)]
    losses = [
        dict(
            total=1.0,
            qa=0.75,
            observation=1.0,
            qa_sequences=2 if train[(step // 2) % 192]["kind"] in ("lookup", "unknown") else 1,
        )
        for step in range(6144)
    ]
    windows = audit.loss_windows(losses, train)
    assert len(windows) == 6 and windows[-1]["end"] == 6144
    assert sum(g["items"] for w in windows for g in w["groups"].values()) == 6144
    assert windows[0]["groups"]["renamed/lookup"]["mean_qa"] == 0.75
    losses[0]["qa_sequences"] = 1
    with pytest.raises(ValueError, match="schedule"):
        audit.loss_windows(losses, train)
