"""R3b contracts for the corrected old baseline: hard gate, time, bias and scoring."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from taiji.conditional_language_v3 import TaskCoreConfigV3
from taiji.corrected_old_baseline import (
    BaselinePredictionV1,
    CorrectedOldBundleV1,
    parse_query,
    score_fixed_denominator,
)

ROOT = Path(__file__).resolve().parents[2]
QUERY = "雨伞的颜色是什么？".encode()


def _config() -> TaskCoreConfigV3:
    return TaskCoreConfigV3(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        seed=21,
        position_extension_seed=22,
    )


def _bundle() -> CorrectedOldBundleV1:
    return CorrectedOldBundleV1.create(_config())


def _ingest(
    bundle: CorrectedOldBundleV1,
    *,
    text: str,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    turn: int,
    revision: int,
    owner: str = "u",
    scope: str = "w",
    session: str = "s",
):
    return bundle.ingest(
        source_text=text,
        entity=entity,
        attribute=attribute,
        value=value,
        source_input_id=f"src:{turn}:{revision}",
        owner=owner,
        scope=scope,
        session=session,
        turn_index=turn,
        revision=revision,
    )


def _logits(bundle: CorrectedOldBundleV1, *, time_intent: str = "current", target: bytes = b"ab"):
    logits, selection = bundle.score_target(
        query=QUERY,
        target=target,
        time_intent=time_intent,
        owner="u",
        scope="w",
        session="s",
    )
    return logits, selection


def test_version_chain_and_hard_gate():
    bundle = _bundle()
    first = _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    second = _ingest(
        bundle,
        text="雨伞的颜色是蓝色。",
        entity="雨伞",
        attribute="颜色",
        value="蓝色",
        turn=1,
        revision=1,
    )
    assert first is not None and second is not None
    current = bundle.store.current("雨伞", "颜色")
    previous = bundle.store.previous("雨伞", "颜色")
    assert current is not None and current.value == "蓝色"
    assert previous is not None and previous.value == "红色"
    superseded = next(item for item in bundle.store.facts if item.fact_id == first.fact_id)
    assert superseded.valid_to == 1
    assert superseded.superseded_by == second.fact_id
    assert (
        _ingest(
            bundle,
            text="可能是红",
            entity=None,
            attribute=None,
            value=None,
            turn=2,
            revision=2,
        )
        is None
    )
    assert bundle.store.current("雨伞", "颜色") is not None


def test_query_parse_handles_location_and_owner_paraphrase():
    location = parse_query("水杯的位置在哪里？".encode(), time_intent="current")
    assert location.entity == "水杯" and location.attribute == "位置"
    owner = parse_query("台灯归谁？".encode(), time_intent="current")
    assert owner.entity == "台灯" and owner.attribute == "归属"
    colour = parse_query("耳机的颜色是什么？".encode(), time_intent="current")
    assert colour.entity == "耳机" and colour.attribute == "颜色"
    failed = parse_query("那个东西是什么？".encode(), time_intent="current")
    assert not failed.parsed


def test_only_value_bias_is_trainable():
    bundle = _bundle()
    assert bundle.model.trainable_parameters() == ("value_bias.weight",)
    frozen = [
        name for name, parameter in bundle.model.named_parameters() if name != "value_bias.weight"
    ]
    assert frozen
    assert all(
        not parameter.requires_grad
        for name, parameter in bundle.model.named_parameters()
        if name != "value_bias.weight"
    )


def test_first_target_byte_is_supervised_and_prefix_advances():
    bundle = _bundle()
    _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    left, _ = _logits(bundle, target=b"ab")
    right, _ = _logits(bundle, target=b"cb")
    assert left.shape == (2, 257)
    assert torch.allclose(left[0], right[0])
    assert not torch.allclose(left[1], right[1])


def test_time_intent_selects_current_and_previous():
    bundle = _bundle()
    _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    _ingest(
        bundle,
        text="雨伞的颜色是蓝色。",
        entity="雨伞",
        attribute="颜色",
        value="蓝色",
        turn=1,
        revision=1,
    )
    current_logits, current = _logits(bundle, time_intent="current")
    previous_logits, previous = _logits(bundle, time_intent="previous")
    assert current.value == "蓝色"
    assert previous.value == "红色"
    assert previous.answerable
    assert not torch.allclose(current_logits, previous_logits)
    unknown_logits, selection = _logits(bundle, time_intent="unknown")
    assert selection.answerable
    assert selection.reason == "unknown_fallback_current"
    assert torch.allclose(current_logits, unknown_logits)


def test_previous_without_predecessor_is_not_answerable():
    bundle = _bundle()
    _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    _, selection = _logits(bundle, time_intent="previous")
    assert selection.fact is None
    assert selection.answerable is False
    assert selection.reason == "no_fact"


def test_owner_scope_session_visibility():
    bundle = _bundle()
    _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    selection = bundle.condition(
        query=QUERY, time_intent="current", owner="other", scope="w", session="s"
    )
    assert selection.fact is None
    assert selection.reason == "not_visible"


def test_cut_changes_output_and_rescue_restores_identical():
    bundle = _bundle()
    fact = _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    assert fact is not None
    baseline, _ = _logits(bundle)
    saved = bundle.store
    bundle.store = saved.cut(fact.fact_id)
    cut, selection = _logits(bundle)
    assert selection.fact is None
    assert not torch.equal(baseline, cut)
    bundle.store = saved
    rescued, _ = _logits(bundle)
    assert torch.equal(baseline, rescued)


def test_fixed_denominator_counts_parse_failures_as_wrong():
    cases = [
        {"fixture_id": "a", "category": "current_query", "target": {"target_bytes_utf8": [1, 2]}},
        {"fixture_id": "b", "category": "previous_query", "target": {"target_bytes_utf8": [3, 4]}},
        {"fixture_id": "c", "category": "unknown_query", "target": {"target_bytes_utf8": [5, 6]}},
    ]
    predictions = {
        "a": BaselinePredictionV1(case_id="a", generated=bytes([1, 2]), parse_ok=True),
        "b": BaselinePredictionV1(case_id="b", generated=bytes([9]), parse_ok=True),
        "c": BaselinePredictionV1(case_id="c", generated=b"", parse_ok=False),
    }
    score = score_fixed_denominator(cases, predictions)
    assert score["scored_turns"] == 3
    assert score["exact"] == 1
    assert score["parse_failures"] == 1
    assert score["answer_exact"] == pytest.approx(1 / 3)
    assert score["by_category"]["unknown_query"]["parse_failures"] == 1


def test_zero_step_bundle_restores_in_process_and_fresh_process(tmp_path):
    bundle = _bundle()
    _ingest(
        bundle,
        text="雨伞的颜色是红色。",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
        owner="probe",
        scope="probe",
        session="probe",
    )
    _ingest(
        bundle,
        text="雨伞的颜色是蓝色。",
        entity="雨伞",
        attribute="颜色",
        value="蓝色",
        turn=1,
        revision=1,
        owner="probe",
        scope="probe",
        session="probe",
    )
    expected = bundle.probe_digest(query=QUERY)
    path = tmp_path / "corrected-old.pt"
    bundle.save(path)
    restored = CorrectedOldBundleV1.load(path)
    assert restored.probe_digest(query=QUERY) == expected
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.corrected_old_baseline import CorrectedOldBundleV1; "
        "b=CorrectedOldBundleV1.load(sys.argv[1]); "
        "print(b.probe_digest(query=sys.argv[2].encode()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), QUERY.decode()],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected
