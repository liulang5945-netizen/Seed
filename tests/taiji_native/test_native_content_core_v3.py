"""R5c contracts: pointer auxiliary CE, v3 bundle identity, schedule/credit inheritance."""

from __future__ import annotations

import pytest
import torch

from taiji.native_content_core import span_sha256
from taiji.native_content_core_v2 import NativeContentCoreV2
from taiji.native_content_core_v3 import (
    NATIVE_CORE_BUNDLE_FORMAT_V3,
    NativeContentBundleV3,
    NativeContentCoreV3,
)
from taiji.native_content_training_v3 import (
    POINTER_CE_WEIGHT,
    NativeContentV3Adapter,
    R5CRunTrainer,
)

QUERY = "雨伞是什么颜色？".encode()
SPAN = "雨伞的颜色是红色。"
TARGET = "红色<|end|>".encode()


def _config():
    from taiji.native_content_core import NativeContentCoreConfigV1

    return NativeContentCoreConfigV1(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        content_width=32,
        seed=11,
        position_extension_seed=12,
        read_seed=13,
    )


def _ingest(bundle, *, span: str, value: str, turn: int = 0) -> None:
    raw = span.encode()
    bundle.ingest(
        raw_span=raw,
        source_input_id=f"src:{turn}",
        source_sha256=span_sha256(raw),
        owner="u",
        scope="w",
        session="s",
        turn_index=turn,
        revision=0,
        entity="雨伞",
        attribute="颜色",
        value=value,
        status="resolved",
    )


def _episode() -> dict:
    return {
        "episode_id": "e:3",
        "turns": [
            {
                "index": 0,
                "role": "fact_v0",
                "scored": False,
                "text": SPAN,
                "entity": "雨伞",
                "attribute": "颜色",
                "value": "红色",
            },
            {
                "index": 1,
                "role": "query",
                "scored": True,
                "text": "雨伞是什么颜色？",
                "category": "current_query",
                "entity": "雨伞",
                "attribute": "颜色",
                "target": {"target_bytes_utf8": list(TARGET)},
            },
        ],
    }


def test_v3_architecture_is_byte_identical_to_v2() -> None:
    config = _config()
    core_v2 = NativeContentCoreV2(config)
    core_v3 = NativeContentCoreV3(config)
    v2_names = dict(core_v2.named_parameters())
    v3_names = dict(core_v3.named_parameters())
    assert set(v2_names) == set(v3_names)
    for name, parameter in v2_names.items():
        assert torch.equal(parameter, v3_names[name])
    assert core_v2.parameter_count() == core_v3.parameter_count()


def test_pointer_auxiliary_ce_is_finite_with_finite_gradients() -> None:
    bundle = NativeContentBundleV3.create(_config())
    _ingest(bundle, span=SPAN, value="红色")
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope="w", session="s"
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:v3", scope_id="w", session_id="s"
    )
    cache = bundle.model.content_read_cache(state, condition, store=bundle.store)
    auxiliary = bundle.model.pointer_auxiliary_ce(state, cache, "红色".encode())
    assert auxiliary is not None and bool(torch.isfinite(auxiliary))
    weighted = POINTER_CE_WEIGHT * auxiliary
    weighted.backward()
    parameters = dict(bundle.model.named_parameters())
    for name in ("copy_key.weight", "copy_query.weight"):
        gradient = parameters[name].grad
        assert gradient is not None and bool(torch.isfinite(gradient).all())


def test_pointer_auxiliary_excluded_without_single_occurrence() -> None:
    bundle = NativeContentBundleV3.create(_config())
    _ingest(bundle, span="雨伞在桌子上。", value="红色")
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope="w", session="s"
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe:v3", scope_id="w", session_id="s"
    )
    cache = bundle.model.content_read_cache(state, condition, store=bundle.store)
    assert bundle.model.pointer_auxiliary_ce(state, cache, "红色".encode()) is None


def test_bundle_v3_roundtrip(tmp_path) -> None:
    bundle = NativeContentBundleV3.create(_config())
    _ingest(bundle, span=SPAN, value="红色")
    digest = bundle.probe_digest(query=QUERY)
    path = tmp_path / "bundle-v3.pt"
    bundle.save(path)
    loaded = NativeContentBundleV3.load(path)
    assert loaded.FORMAT == NATIVE_CORE_BUNDLE_FORMAT_V3
    assert loaded.probe_digest(query=QUERY) == digest


def test_v3_adapter_loss_terms_with_auxiliary_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = NativeContentV3Adapter()
    model = adapter.create_model()
    episode = _episode()
    target_length = len(TARGET)

    adapter.schedule_update = 0
    terms, total = adapter.loss_terms(model, episode, owner="u", scope="w")
    assert total == target_length
    assert adapter.branch_counters["aux_turns"] == 1
    assert adapter.branch_counters["aux_loss_sum"] > 0.0
    assert len(terms) == 2

    monkeypatch.setattr(
        "taiji.native_content_training_v3.NativeContentV3Adapter._expected_value",
        lambda self, episode, turn: None,
    )
    adapter.reset_branch_counters()
    terms, total = adapter.loss_terms(model, episode, owner="u", scope="w")
    assert total == target_length
    assert adapter.branch_counters["aux_turns"] == 0
    assert len(terms) == 1


def test_v3_adapter_schedule_inherited_and_trainer_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from taiji.native_content_training import batch_schedule

    monkeypatch.setattr("taiji.native_content_training_v2.own_prefix_coin", lambda **kwargs: True)
    adapter = NativeContentV3Adapter()
    trainer = R5CRunTrainer.create(adapter)
    trainer.cursor.update({"run_id": "probe", "seed": 0})
    episodes = [_episode()]
    batches = batch_schedule(episodes, seed=0, updates=1, batch_size=1)

    trainer.schedule_position_override = 300
    metrics = trainer.update(batches[0].episodes)
    assert metrics["schedule_position"] == 300
    assert metrics["episodes_scheduled"] == 1
    assert metrics["aux_turns"] == 1
    assert metrics["aux_pointer_loss"] > 0.0
