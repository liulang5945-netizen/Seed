"""Material interventions preserve other input information and reject heldout use."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_train_content_read as diagnostic


@pytest.mark.parametrize("kind,target", [("lookup", 0), ("revision", 1)])
def test_material_interventions_change_only_selected_value(kind, target):
    observations = ["entity uses abcdef.", "other uses ghijkl."]
    if kind == "revision":
        observations = ["entity uses abcdef.", "entity now uses ghijkl."]
    item = {
        "split": "train",
        "kind": kind,
        "observations": observations,
        "answer": ("abcdef", "ghijkl")[target],
        "question": "material of entity?",
    }
    before = deepcopy(item)
    source, variants = diagnostic.variants(item, ("mnopqr", "stuvwx"))
    assert source == target and item == before
    assert variants["target"]["observations"][1 - target] == item["observations"][1 - target]
    assert variants["target"]["question"] == item["question"]
    assert variants["target"]["answer"] == "mnopqr"
    assert variants["distractor"]["answer"] == item["answer"]
    assert variants["distractor"]["observations"][target] == item["observations"][target]
    assert [len(text) for text in variants["target"]["observations"]] == [
        len(text) for text in observations
    ]


def test_heldout_diagnostic_rejected_before_model_access():
    with pytest.raises(ValueError, match="dev or holdout"):
        diagnostic.infer(None, {"split": "holdout"}, "forbidden")
    with pytest.raises(ValueError, match="training episode"):
        diagnostic.variants({"split": "dev"}, ("abcdef", "ghijkl"))
