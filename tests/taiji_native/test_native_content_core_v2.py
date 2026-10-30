"""R5b copy-track candidate contracts: store binding, gold rejection, degenerate equivalence, recovery."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from taiji.native_content_core import (
    NativeContentConditionV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
    span_sha256,
)
from taiji.native_content_core_v2 import (
    NATIVE_CORE_BUNDLE_FORMAT_V2,
    NativeContentBundleV2,
    NativeContentCoreV2,
)
from taiji.native_content_training_v2 import (
    SCHEDULE_OWN_PREFIX_START,
    NativeContentV2Adapter,
    own_prefix_coin,
)

ROOT = Path(__file__).resolve().parents[2]
QUERY = "雨伞是什么颜色？".encode()
SPAN = "雨伞的颜色是红色。"
TARGET = "红色<|end|>".encode()


def _config() -> NativeContentCoreConfigV1:
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


def _bundle() -> NativeContentBundleV2:
    bundle = NativeContentBundleV2.create(_config())
    bundle.cursor.update({"owner": "u", "scope": "w", "session": "s"})
    return bundle


def _ingest(bundle: NativeContentBundleV2, *, span: str, value: str, turn: int = 0) -> None:
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
        "episode_id": "e:1",
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
                "target": {"target_bytes_utf8": list(TARGET)},
            },
        ],
    }


def test_copy_track_reduces_to_v1_without_content() -> None:
    config = _config()
    core_v1 = NativeContentCoreV1(config)
    core_v2 = NativeContentCoreV2(config)
    condition = NativeContentConditionV1(query=QUERY, records=(), time_intent="current")
    state_v1 = core_v1.conditioned_state(
        condition, native_state_ref="probe", scope_id="u", session_id="s"
    )
    state_v2 = core_v2.conditioned_state(
        condition, native_state_ref="probe", scope_id="u", session_id="s"
    )
    cache_v1 = core_v1.content_read_cache(state_v1, condition)
    cache_v2 = core_v2.content_read_cache(state_v2, condition, store=NativeContentStoreV1(()))
    logits_v1, _ = core_v1.next_logits(state_v1, cache=cache_v1)
    logits_v2, _ = core_v2.next_logits(state_v2, cache=cache_v2)
    assert torch.equal(logits_v1, logits_v2)


def test_copy_track_math_is_the_declared_mixture() -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    core_v1 = NativeContentCoreV1(_config())
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope="w", session="s"
    )
    state_v2 = bundle.model.conditioned_state(
        condition, native_state_ref="probe", scope_id="w", session_id="s"
    )
    state_v1 = core_v1.conditioned_state(
        condition, native_state_ref="probe", scope_id="w", session_id="s"
    )
    cache = bundle.model.content_read_cache(state_v2, condition, store=bundle.store)
    logits_v2, updated = bundle.model.next_logits(state_v2, cache=cache)
    logits_v1, _ = core_v1.next_logits(
        state_v1, cache=core_v1.content_read_cache(state_v1, condition)
    )
    hidden = state_v2.hidden if state_v2.decoder_hidden is None else state_v2.decoder_hidden
    read, scores = bundle.model._read(cache, hidden)
    span = cache.content_spans[int(torch.argmax(scores))]
    tokens = torch.tensor(list(span), dtype=torch.long)
    keys = bundle.model.copy_key(core_v1.core.embedding(tokens))
    positions = keys @ bundle.model.copy_query(hidden) / (32**0.5)
    weights = torch.softmax(positions, dim=0)
    copy_distribution = torch.zeros_like(logits_v1)
    copy_distribution.index_add_(0, tokens, weights)
    gate = torch.sigmoid(bundle.model.copy_gate(torch.cat((updated, read))))
    expected = torch.log(
        ((1.0 - gate) * torch.softmax(logits_v1, dim=-1) + gate * copy_distribution).clamp_min(
            1e-12
        )
    )
    assert torch.allclose(logits_v2, expected, atol=1e-6)
    assert not torch.allclose(logits_v2, logits_v1)


def test_forged_condition_record_is_rejected() -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    foreign = NativeContentBundleV2.create(_config())
    _ingest(foreign, span="伞是红色的，颜色为红色。", value="红色")
    forged_record = foreign.store.records[0]
    condition = NativeContentConditionV1(
        query=QUERY, records=(forged_record,), time_intent="current"
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="probe", scope_id="w", session_id="s"
    )
    with pytest.raises(ValueError, match="not in the declared store"):
        bundle.model.content_read_cache(state, condition, store=bundle.store)


def test_cut_store_degenerates_to_v1() -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    core_v1 = NativeContentCoreV1(_config())
    cut_condition = NativeContentConditionV1(query=QUERY, records=(), time_intent="current")
    state_v2 = bundle.model.conditioned_state(
        cut_condition, native_state_ref="probe", scope_id="w", session_id="s"
    )
    state_v1 = core_v1.conditioned_state(
        cut_condition, native_state_ref="probe", scope_id="w", session_id="s"
    )
    cache_v2 = bundle.model.content_read_cache(
        state_v2, cut_condition, store=NativeContentStoreV1(())
    )
    cache_v1 = core_v1.content_read_cache(state_v1, cut_condition)
    logits_v2, _ = bundle.model.next_logits(state_v2, cache=cache_v2)
    logits_v1, _ = core_v1.next_logits(state_v1, cache=cache_v1)
    assert torch.equal(logits_v2, logits_v1)


def test_scheduled_forced_conditioned_is_deterministic_and_byte_exact() -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    condition = bundle.condition(
        query=QUERY, time_intent="current", owner="u", scope="w", session="s"
    )
    first, prefix = bundle.model.scheduled_forced_conditioned(
        condition, TARGET, own_prefix_len=2, scope_id="w", session_id="s", store=bundle.store
    )
    second, _ = bundle.model.scheduled_forced_conditioned(
        condition, TARGET, own_prefix_len=2, scope_id="w", session_id="s", store=bundle.store
    )
    assert prefix == 2
    assert first.shape[0] == len(TARGET)
    assert torch.equal(first, second)
    full = bundle.score_target(
        query=QUERY, target=TARGET, time_intent="current", owner="u", scope="w", session="s"
    )
    assert full.shape[0] == len(TARGET)


def test_zero_step_bundle_recovery_v2(tmp_path: Path) -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    digest = bundle.probe_digest(query=QUERY)
    payload = bundle.checkpoint()
    restored = NativeContentBundleV2.from_checkpoint(payload)
    assert restored.probe_digest(query=QUERY) == digest
    path = tmp_path / "bundle-v2.pt"
    bundle.save(path)
    loaded = NativeContentBundleV2.load(path)
    assert loaded.probe_digest(query=QUERY) == digest
    assert loaded.FORMAT == NATIVE_CORE_BUNDLE_FORMAT_V2


def test_fresh_process_restore_v2(tmp_path: Path) -> None:
    bundle = _bundle()
    _ingest(bundle, span=SPAN, value="红色")
    path = tmp_path / "bundle-v2-fresh.pt"
    bundle.save(path)
    code = (
        f"import json, sys; sys.path.insert(0, r'{str(ROOT)}'); "
        "from taiji.native_content_core_v2 import NativeContentBundleV2; "
        f"bundle = NativeContentBundleV2.load(r'{str(path)}'); "
        f"print(json.dumps({{'digest': bundle.probe_digest(query={QUERY!r})}}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    digest = json.loads(completed.stdout.strip().splitlines()[-1])["digest"]
    assert digest == bundle.probe_digest(query=QUERY)


def test_credit_stop_freezes_copy_parameters() -> None:
    bundle = _bundle()
    model = bundle.model
    groups = model.credit_groups()
    copy_names = ("copy_gate.weight", "copy_gate.bias", "copy_key.weight", "copy_query.weight")
    for name in copy_names:
        assert name in groups["decoder"]
    model.set_credit_group("decoder", False)
    try:
        parameters = dict(model.named_parameters())
        for name in copy_names:
            assert parameters[name].requires_grad is False
        assert parameters["decoder.weight_ih"].requires_grad is False
        assert parameters["core.embedding.weight"].requires_grad is True
    finally:
        model.set_credit_group("decoder", True)
        parameters = dict(model.named_parameters())
        for name in copy_names:
            assert parameters[name].requires_grad is True


def test_own_prefix_coin_is_pure_and_declared() -> None:
    kwargs = {"seed": 0, "update": SCHEDULE_OWN_PREFIX_START, "episode_id": "e:1"}
    assert own_prefix_coin(**kwargs) == own_prefix_coin(**kwargs)
    assert SCHEDULE_OWN_PREFIX_START == 257


def test_v2_adapter_loss_terms_byte_accounting(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = NativeContentV2Adapter()
    model = adapter.create_model()
    episode = _episode()
    target_length = len(TARGET)

    adapter.schedule_update = 0
    terms, total = adapter.loss_terms(model, episode, owner="u", scope="w")
    assert total == target_length
    assert all(bool(torch.isfinite(term)) for term in terms)
    assert adapter.branch_counters["episodes_scheduled"] == 0
    assert adapter.branch_counters["supervised_bytes"] == target_length

    monkeypatch.setattr("taiji.native_content_training_v2.own_prefix_coin", lambda **kwargs: True)
    adapter.schedule_update = 300
    terms, total = adapter.loss_terms(model, episode, owner="u", scope="w")
    assert total == target_length
    assert all(bool(torch.isfinite(term)) for term in terms)
    assert adapter.branch_counters["episodes_scheduled"] == 1
    assert adapter.branch_counters["supervised_bytes"] == 2 * target_length

    monkeypatch.setattr("taiji.native_content_training_v2.own_prefix_coin", lambda **kwargs: False)
    adapter.schedule_update = 300
    terms, total = adapter.loss_terms(model, episode, owner="u", scope="w")
    assert total == target_length
    assert adapter.branch_counters["episodes_scheduled"] == 1


def test_trainer_override_keeps_probe_on_declared_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from taiji.native_content_training import batch_schedule
    from taiji.native_content_training_v2 import R5BRunTrainer

    monkeypatch.setattr("taiji.native_content_training_v2.own_prefix_coin", lambda **kwargs: True)
    adapter = NativeContentV2Adapter()
    trainer = R5BRunTrainer.create(adapter)
    trainer.cursor.update({"run_id": "probe", "seed": 0})
    episodes = [_episode()]
    batches = batch_schedule(episodes, seed=0, updates=2, batch_size=1)

    trainer.schedule_position_override = 300
    rows = [trainer.update(batch.episodes) for batch in batches]
    assert all(row["schedule_position"] == 300 for row in rows)
    assert all(row["schedule_override_active"] is True for row in rows)
    assert all(row["episodes_scheduled"] == 1 for row in rows)
    assert all(row["scheduled_supervised_bytes"] == row["target_bytes"] for row in rows)

    trainer.schedule_position_override = None
    row = trainer.update(batches[0].episodes)
    assert row["schedule_position"] == 3
    assert row["schedule_override_active"] is False
    assert row["episodes_scheduled"] == 0


def test_replay_episode_supports_v2_adapter() -> None:
    from taiji.native_content_eval import replay_episode

    adapter = NativeContentV2Adapter()
    model = adapter.create_model()
    result = replay_episode(adapter, model, _episode())
    key = ("e:1", 1)
    assert key in result.predictions
    prediction = result.predictions[key]
    assert prediction.parse_ok is True
    assert isinstance(prediction.produced, bytes)
    assert result.selected_content[key] is not None
    assert result.selected_content[key][1] == "红色"
