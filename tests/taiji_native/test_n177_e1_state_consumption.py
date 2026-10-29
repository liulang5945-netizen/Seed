"""Contract tests for the E1 persistent content store and query selector."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from taiji.joint_state_consumption import (
    PersistentContentFeatureV1,
    PersistentContentStoreV1,
    QueryConditionedSelectorV1,
    ValueTokenFeatureEncoderV1,
    feature_from_parse,
)
from taiji.joint_state_consumption_course import E1_COLORS

_SOURCE = "a" * 64


def _feature(
    entity: str,
    value: str,
    *,
    revision: int,
    turn_index: int,
    input_id: str,
    continuous: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> PersistentContentFeatureV1:
    return PersistentContentFeatureV1.create(
        entity=entity,
        attribute="颜色",
        value=value,
        source_input_id=input_id,
        source_sha256=_SOURCE,
        turn_index=turn_index,
        revision=revision,
        valid_from=turn_index,
        confidence=0.9,
        continuous=continuous,
    )


def _store() -> tuple[PersistentContentStoreV1, PersistentContentFeatureV1]:
    old = _feature("钥匙", "黄色", revision=0, turn_index=0, input_id="fact-0")
    current = _feature("钥匙", "蓝色", revision=1, turn_index=3, input_id="fact-3")
    other = _feature("杯子", "黑色", revision=0, turn_index=5, input_id="fact-5")
    store = PersistentContentStoreV1(()).add(old).add(current).add(other)
    return store, current


def test_correction_supersedes_but_keeps_history() -> None:
    store, current = _store()
    assert store.current("钥匙", "颜色") == current
    history = store.history("钥匙", "颜色")
    assert len(history) == 2
    assert history[0].value == "黄色"
    assert history[0].superseded_by == current.feature_id
    assert history[1].value == "蓝色"


def test_cut_and_rescue_change_current_content() -> None:
    store, current = _store()
    cut = store.cut("钥匙", "颜色")
    assert cut.current("钥匙", "颜色") is None
    assert cut.current("杯子", "颜色") is not None
    rescued = cut.add(current)
    assert rescued.current("钥匙", "颜色") == current


def test_relevant_and_irrelevant_swap_at_selection() -> None:
    store, current = _store()
    selector = QueryConditionedSelectorV1()
    selected = selector.select(entity="钥匙", attribute="颜色", candidates=store.active_features())
    assert selected is not None
    assert selected.selected_feature == current

    relevant = _feature("钥匙", "绿色", revision=2, turn_index=6, input_id="fact-6")
    swapped = store.swap("钥匙", "颜色", relevant)
    selected = selector.select(
        entity="钥匙", attribute="颜色", candidates=swapped.active_features()
    )
    assert selected is not None
    assert selected.selected_feature == relevant

    irrelevant = _feature("杯子", "白色", revision=1, turn_index=7, input_id="fact-7")
    swapped_other = swapped.swap("杯子", "颜色", irrelevant)
    selected = selector.select(
        entity="钥匙", attribute="颜色", candidates=swapped_other.active_features()
    )
    assert selected is not None
    assert selected.selected_feature == relevant


def test_selector_returns_none_after_cut() -> None:
    store, _ = _store()
    selector = QueryConditionedSelectorV1()
    cut = store.cut("钥匙", "颜色")
    assert (
        selector.select(entity="钥匙", attribute="颜色", candidates=cut.active_features()) is None
    )


def test_selector_parameters_receive_gradients() -> None:
    store, _ = _store()
    selector = QueryConditionedSelectorV1()
    scores = selector.score_candidates(
        entity="钥匙", attribute="颜色", candidates=store.active_features()
    )
    loss = (scores**2).sum()
    loss.backward()
    gradients = [
        parameter.grad for parameter in selector.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_feature_from_parse_requires_all_three_roles() -> None:
    parse = SimpleNamespace(
        spans=(
            SimpleNamespace(role="entity", text="钥匙"),
            SimpleNamespace(role="attribute", text="颜色"),
            SimpleNamespace(role="value", text="蓝色"),
        ),
        semantic_features=(0.5, 1.0),
        confidence=0.9,
    )
    feature = feature_from_parse(
        parse, source_input_id="fact", source_sha256=_SOURCE, turn_index=3, revision=1
    )
    assert feature is not None
    assert (feature.entity, feature.attribute, feature.value) == ("钥匙", "颜色", "蓝色")
    assert feature.continuous == (0.5, 1.0)
    query = SimpleNamespace(
        spans=(
            SimpleNamespace(role="entity", text="钥匙"),
            SimpleNamespace(role="attribute", text="颜色"),
        ),
        semantic_features=(0.5, 1.0),
        confidence=0.9,
    )
    assert (
        feature_from_parse(
            query, source_input_id="query", source_sha256=_SOURCE, turn_index=4, revision=2
        )
        is None
    )


def test_value_token_feature_encoder_separates_colours() -> None:
    encoder = ValueTokenFeatureEncoderV1()
    rows = [encoder.encode(colour) for colour in E1_COLORS]
    assert len({tuple(row.tolist()) for row in rows}) == len(E1_COLORS)
    loss = torch.stack(rows).sum()
    loss.backward()
    assert encoder.embedding.weight.grad is not None
    assert float(encoder.embedding.weight.grad.abs().sum()) > 0


def test_value_token_feature_encoder_rejects_unknown_value() -> None:
    encoder = ValueTokenFeatureEncoderV1()
    with pytest.raises(ValueError):
        encoder.encode("灰色")


def test_store_round_trip() -> None:
    store, _ = _store()
    payload = store.to_payload()
    restored = PersistentContentStoreV1.from_payload(payload)
    assert restored == store


def test_duplicate_active_value_is_rejected() -> None:
    store, current = _store()
    with pytest.raises(ValueError):
        store.add(current)
    with pytest.raises(ValueError):
        store.add(_feature("钥匙", "绿色", revision=0, turn_index=9, input_id="fact-9"))
