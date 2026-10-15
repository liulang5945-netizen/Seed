"""Dataset integrity and non-copy transition semantics for PLAN-N1-14."""

from copy import deepcopy

import pytest

from scripts.training.build_taiji_n1_evidence_curriculum import build, validate


def test_manifest_reproducible_and_entity_leak_rejected():
    manifest = build()
    assert manifest == build()
    counts = validate(manifest)
    assert counts["holdout"] == {"lookup": 8, "revision": 8, "transition": 8, "unknown": 8}
    contaminated = deepcopy(manifest)
    contaminated["episodes"][-1]["entities"][0] = manifest["episodes"][0]["entities"][0]
    with pytest.raises(AssertionError):
        validate(contaminated)
    assert all(len(item["observations"]) == 2 for item in manifest["episodes"])
    shortcut = deepcopy(manifest)
    shortcut["episodes"][0]["entities"][0] = "tunit000"
    with pytest.raises(AssertionError):
        validate(shortcut)


def test_transition_semantics_and_answer_absence():
    right = {"north": "east", "east": "south", "south": "west", "west": "north"}
    left = {value: key for key, value in right.items()}
    for item in build()["episodes"]:
        if item["kind"] != "transition":
            continue
        initial = item["observations"][0].removesuffix(".").split()[-1]
        turn = item["observations"][1].removesuffix(".").split()[-1]
        expected = (right if turn == "right" else left)[initial]
        assert item["answer"] == expected
        assert expected not in " ".join(item["observations"])


def test_revision_labels_follow_later_fact_and_unknown_lacks_entity_fact():
    for item in build()["episodes"]:
        if item["kind"] == "revision":
            assert item["observations"][-1].endswith(f"uses {item['answer']}.")
            assert not item["observations"][0].endswith(f"uses {item['answer']}.")
        if item["kind"] == "unknown":
            queried = item["question"].removeprefix("material of ").removesuffix("?")
            assert all(queried not in observation for observation in item["observations"])
            assert item["answer"] == "unknown"
