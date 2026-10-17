from __future__ import annotations

import copy
from collections import Counter
from types import SimpleNamespace

import pytest

from scripts.training import build_taiji_n1_input_factorial_diagnostic as builder
from scripts.training import eval_taiji_n1_input_factorial_diagnostic as evaluator
from scripts.training import run_taiji_n1_hard_neighbor_training as training


@pytest.fixture
def synthetic_face():
    face = builder.build_face(
        excluded_names={"blockedname"},
        excluded_words={"blocked"},
    )
    return face


def test_face_has_exact_full_position_length_factorial_and_d1_distractors(synthetic_face):
    report = builder.audit_face(
        synthetic_face,
        excluded_names={"blockedname"},
        excluded_words={"blocked"},
    )
    assert report["accepted"] is True, report["errors"]
    assert report["item_count"] == 248
    assert report["kind_counts"] == builder.EXPECTED_ITEMS
    assert report["unique_d1_distractors"] == 364
    assert report["d1_distractor_occurrences"] == 2616
    assert report["unique_target_templates"] == 12
    assert report["checks"]["full_factorial_cells"] is True
    assert report["checks"]["transition_position_cells"] is True


def test_lookup_and_revision_cells_pair_each_template_across_every_position_and_length(
    synthetic_face,
):
    cells = Counter(
        (
            episode["kind"],
            episode["target_row_position_1based"],
            episode["material_answer_length_chars"],
        )
        for episode in synthetic_face["episodes"]
        if episode["kind"] in {"lookup", "revision"}
    )
    assert len(cells) == 46
    assert set(cells.values()) == {4}
    assert all(episode["split"] == "diagnostic" for episode in synthetic_face["episodes"])


def test_template_identity_and_answer_values_are_reused_only_within_crossed_cells(
    synthetic_face,
):
    lookup = [
        episode
        for episode in synthetic_face["episodes"]
        if episode["kind"] == "lookup" and episode["template_index"] == 0
    ]
    assert len(lookup) == 24
    assert len({episode["entities"][0] for episode in lookup}) == 1
    assert len({episode["question"] for episode in lookup}) == 1
    assert all(episode["entities"] == lookup[0]["entities"] for episode in lookup)

    def fillers(episode):
        subject = episode["entities"][0]
        return tuple(row for row in episode["observations"] if not row.startswith(subject + " "))

    assert all(fillers(episode) == fillers(lookup[0]) for episode in lookup)
    for length in builder.ANSWER_LENGTHS:
        selected = [item for item in lookup if item["material_answer_length_chars"] == length]
        assert len(selected) == 12
        assert len({item["answer"] for item in selected}) == 1
    other_templates = [
        episode["answer"]
        for episode in synthetic_face["episodes"]
        if episode["kind"] == "lookup" and episode["target_row_position_1based"] == 1
    ]
    assert len(other_templates) == len(set(other_templates))


def test_audit_rejects_source_overlap_and_misaligned_target_position(synthetic_face):
    subject = next(
        episode["entities"][0]
        for episode in synthetic_face["episodes"]
        if episode["kind"] == "lookup"
    )
    blocked_name = builder.audit_face(
        synthetic_face,
        excluded_names={subject},
        excluded_words=set(),
    )
    assert blocked_name["accepted"] is False
    assert any(error.startswith("excluded_entity:") for error in blocked_name["errors"])

    altered = copy.deepcopy(synthetic_face)
    lookup = next(episode for episode in altered["episodes"] if episode["kind"] == "lookup")
    lookup["target_row_position_1based"] += 1
    shifted = builder.audit_face(altered, excluded_names=set(), excluded_words=set())
    assert shifted["accepted"] is False
    assert any("target_position_mismatch" in error for error in shifted["errors"])


def test_audit_rejects_a_semantically_reversed_revision():
    face = builder.build_face(excluded_names=set(), excluded_words=set())
    revision = next(episode for episode in face["episodes"] if episode["kind"] == "revision")
    position = revision["target_row_position_1based"] - 1
    revision["observations"][position], revision["observations"][position + 1] = (
        revision["observations"][position + 1],
        revision["observations"][position],
    )
    result = builder.audit_face(face, excluded_names=set(), excluded_words=set())
    assert result["accepted"] is False
    assert any("revision_target_mismatch" in error for error in result["errors"])


def test_query_count_is_frozen_and_diagnostic_has_no_existing_gate_fields():
    total = sum(builder.EXPECTED_ITEMS.values()) * len(training.SEEDS) * len(training.ARMS)
    assert total == 2232
    face = builder.build_face(excluded_names=set(), excluded_words=set())
    assert "passed" not in face
    assert "gates" not in face


def test_evaluation_summary_preserves_full_factorial_denominators_and_paired_contrasts(
    synthetic_face,
):
    rows = []
    for seed in evaluator.SEEDS:
        for arm in evaluator.ARMS:
            for item in synthetic_face["episodes"]:
                rows.append(
                    {
                        "id": item["id"],
                        "kind": item["kind"],
                        "seed": seed,
                        "arm": arm,
                        "free_exact": (
                            arm == "position"
                            and item["kind"] == "lookup"
                            and item["material_answer_length_chars"] == 7
                        ),
                    }
                )

    result = evaluator._summarize(rows, synthetic_face["episodes"])
    assert result["endpoint_count"] == 9
    assert result["items_per_endpoint"] == 248
    assert result["rows_analyzed"] == 2232
    assert len(result["feature_cells"]) == 58
    assert len(result["paired_length_contrasts"]) == 828
    assert len(result["paired_position_profiles"]) == 180
    assert all(
        cell["denominator"] == 4 for cell in result["feature_cells"] if cell["kind"] != "unknown"
    )
    unknown = next(cell for cell in result["feature_cells"] if cell["kind"] == "unknown")
    assert unknown["denominator"] == 20
    assert all(contrast["denominator"] == 1 for contrast in result["paired_length_contrasts"])
    assert all(
        profile["denominator"] == len(profile["free_exact_by_position"])
        for profile in result["paired_position_profiles"]
    )
    assert "gates" not in result and "passed" not in result


def test_evaluation_summary_rejects_missing_or_duplicate_endpoint_items(synthetic_face):
    rows = [
        {
            "id": item["id"],
            "kind": item["kind"],
            "seed": seed,
            "arm": arm,
            "free_exact": False,
        }
        for seed in evaluator.SEEDS
        for arm in evaluator.ARMS
        for item in synthetic_face["episodes"]
    ]
    with pytest.raises(ValueError, match="exactly one result"):
        evaluator._summarize(rows[:-1], synthetic_face["episodes"])
    with pytest.raises(ValueError, match="invalid or duplicate"):
        evaluator._summarize(rows + [rows[0]], synthetic_face["episodes"])


def test_diagnostic_scorer_refuses_to_relabel_holdout_items():
    with pytest.raises(ValueError, match="diagnostic split"):
        evaluator.measure_diagnostic(None, {"split": "holdout"}, "position", "session")


def test_restore_preflight_accepts_frozen_training_history_without_session_collisions(
    tmp_path, monkeypatch
):
    class FakeModel:
        def __init__(self):
            self._observations = {"training-session": {"step": 24}}

        def bundle(self, optimizer):
            return {"observations": dict(self._observations), "weight": 17}

    paths = {}
    for seed in evaluator.SEEDS:
        for arm in evaluator.ARMS:
            path = tmp_path / f"{seed}-{arm}.pt"
            path.write_bytes(f"{seed}/{arm}".encode())
            paths[(seed, arm)] = path

    monkeypatch.setattr(
        evaluator.v9_evaluator,
        "_new_model",
        lambda g, path, seed, arm: (FakeModel(), object()),
    )
    result = evaluator._restore_endpoints(
        {"paired": SimpleNamespace(fingerprint=lambda bundle: str(bundle))},
        paths,
        [{"id": "item-01"}],
    )
    assert len(result) == 9
    assert all(item["full_bundle_equal"] is True for item in result)
    assert all(item["initial_history_entries"] == 1 for item in result)
    assert all(item["diagnostic_session_collisions"] == 0 for item in result)


def test_restore_preflight_rejects_a_diagnostic_session_collision(tmp_path, monkeypatch):
    class FakeModel:
        _observations = {"n1-v10-diagnostic-11-position-item-01": {"step": 1}}

        def bundle(self, optimizer):
            return {"observations": dict(self._observations)}

    paths = {}
    for seed in evaluator.SEEDS:
        for arm in evaluator.ARMS:
            path = tmp_path / f"{seed}-{arm}.pt"
            path.write_bytes(b"endpoint")
            paths[(seed, arm)] = path
    monkeypatch.setattr(
        evaluator.v9_evaluator,
        "_new_model",
        lambda g, path, seed, arm: (FakeModel(), object()),
    )
    with pytest.raises(ValueError, match="diagnostic session"):
        evaluator._restore_endpoints(
            {"paired": SimpleNamespace(fingerprint=lambda bundle: str(bundle))},
            paths,
            [{"id": "item-01"}],
        )


def test_endpoint_scorer_removes_each_session_and_checks_the_full_bundle(monkeypatch):
    class FakeModel:
        def __init__(self):
            self._observations = {"training-session": {"step": 24}}
            self.weight = 17

        def bundle(self, optimizer):
            return {"weight": self.weight, "observations": dict(self._observations)}

    model = FakeModel()
    optimizer = object()

    def fake_measure(current_model, item, arm, session):
        current_model._observations[session] = {"item": item["id"]}
        return {"id": item["id"], "kind": item["kind"], "free_exact": False}

    monkeypatch.setattr(evaluator, "measure_diagnostic", fake_measure)
    accounting = {"model_queries": 0}
    rows = evaluator.score_endpoint(
        model,
        optimizer,
        11,
        "position",
        [{"id": "one", "kind": "lookup"}, {"id": "two", "kind": "lookup"}],
        fingerprint=lambda bundle: str(bundle),
        budget_guard=lambda: None,
        accounting=accounting,
    )
    assert len(rows) == accounting["model_queries"] == 2
    assert model._observations == {"training-session": {"step": 24}}


def test_endpoint_scorer_fails_if_a_query_mutates_the_bundle(monkeypatch):
    class FakeModel:
        def __init__(self):
            self._observations = {}
            self.weight = 17

        def bundle(self, optimizer):
            return {"weight": self.weight, "observations": dict(self._observations)}

    model = FakeModel()

    def mutating_measure(current_model, item, arm, session):
        current_model._observations[session] = {"item": item["id"]}
        current_model.weight += 1
        return {"id": item["id"], "kind": item["kind"], "free_exact": False}

    monkeypatch.setattr(evaluator, "measure_diagnostic", mutating_measure)
    with pytest.raises(ValueError, match="changed model"):
        evaluator.score_endpoint(
            model,
            object(),
            11,
            "position",
            [{"id": "one", "kind": "lookup"}],
            fingerprint=lambda bundle: str(bundle),
            budget_guard=lambda: None,
            accounting={"model_queries": 0},
        )
    assert model._observations == {}
