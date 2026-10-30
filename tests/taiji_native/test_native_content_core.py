"""R3 contracts for the native content core: version store, time reading, credit, recovery."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from taiji.native_content_core import (
    NativeContentBundleV1,
    NativeContentCoreConfigV1,
    NativeContentStoreV1,
    span_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
QUERY = "雨伞是什么颜色？".encode()


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


def _bundle(*, owner: str = "u", scope: str = "w", session: str = "s") -> NativeContentBundleV1:
    bundle = NativeContentBundleV1.create(_config())
    bundle.cursor.update({"owner": owner, "scope": scope, "session": session})
    return bundle


def _ingest(
    bundle: NativeContentBundleV1,
    *,
    span: str,
    entity: str | None,
    attribute: str | None,
    value: str | None,
    turn: int,
    revision: int,
    status: str = "resolved",
    owner: str = "u",
    scope: str = "w",
    session: str = "s",
    source: str = "src",
):
    raw = span.encode()
    return bundle.ingest(
        raw_span=raw,
        source_input_id=f"{source}:{turn}:{revision}",
        source_sha256=span_sha256(raw),
        owner=owner,
        scope=scope,
        session=session,
        turn_index=turn,
        revision=revision,
        entity=entity,
        attribute=attribute,
        value=value,
        status=status,
        confidence=1.0,
    )


def _logits(bundle: NativeContentBundleV1, *, target: bytes = b"ab", time_intent: str = "current"):
    return bundle.score_target(
        query=QUERY,
        target=target,
        time_intent=time_intent,
        owner="u",
        scope="w",
        session="s",
    )


def test_record_and_store_payload_roundtrip():
    bundle = _bundle()
    record = _ingest(
        bundle,
        span="红色",
        entity="雨伞",
        attribute="颜色",
        value="红色",
        turn=0,
        revision=0,
    )
    payload = bundle.store.to_payload()
    restored = NativeContentStoreV1.from_payload(payload)
    assert restored == bundle.store
    assert restored.records[0].record_sha256 == record.record_sha256
    assert restored.records[0].source_byte_span == (0, len(record.raw_span))


def test_version_chain_keeps_history_and_previous_is_direct_predecessor():
    bundle = _bundle()
    first = _ingest(
        bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0
    )
    second = _ingest(
        bundle, span="蓝色", entity="雨伞", attribute="颜色", value="蓝色", turn=4, revision=1
    )
    current = bundle.store.current("雨伞", "颜色")
    previous = bundle.store.previous("雨伞", "颜色")
    assert current is not None and current.content_id == second.content_id
    assert previous is not None and previous.content_id == first.content_id
    superseded = next(item for item in bundle.store.records if item.content_id == first.content_id)
    assert superseded.valid_to == 4
    assert superseded.superseded_by == second.content_id
    same_value = _ingest(
        bundle, span="蓝色", entity="雨伞", attribute="颜色", value="蓝色", turn=6, revision=2
    )
    previous = bundle.store.previous("雨伞", "颜色")
    assert previous is not None
    assert previous.content_id == second.content_id
    assert previous.value == same_value.value
    with pytest.raises(ValueError):
        _ingest(
            bundle, span="绿色", entity="雨伞", attribute="颜色", value="绿色", turn=8, revision=2
        )


def test_unresolved_record_is_readable_but_never_current():
    bundle = _bundle()
    record = _ingest(
        bundle,
        span="可能是红",
        entity=None,
        attribute=None,
        value=None,
        turn=1,
        revision=0,
        status="unresolved",
    )
    assert bundle.store.current("雨伞", "颜色") is None
    readable = bundle.store.readable(owner="u", scope="w", session="s", time_intent="current")
    assert [item.content_id for item in readable] == [record.content_id]
    previous = bundle.store.readable(owner="u", scope="w", session="s", time_intent="previous")
    assert previous == ()


def test_owner_scope_and_session_are_hard_boundaries():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    assert bundle.store.readable(owner="other", scope="w", session="s", time_intent="current") == ()
    assert bundle.store.readable(owner="u", scope="other", session="s", time_intent="current") == ()
    cross = bundle.store.readable(owner="u", scope="w", session="s", time_intent="current")
    assert len(cross) == 1
    assert (
        bundle.store.readable(
            owner="u", scope="w", session="other", time_intent="current", cross_session=False
        )
        == ()
    )


def test_time_intent_changes_candidate_set_and_model_read():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    correction = _ingest(
        bundle, span="蓝色", entity="雨伞", attribute="颜色", value="蓝色", turn=4, revision=1
    )
    other = _ingest(
        bundle, span="桌上", entity="水杯", attribute="位置", value="桌上", turn=0, revision=0
    )
    current = bundle.store.readable(owner="u", scope="w", session="s", time_intent="current")
    previous = bundle.store.readable(owner="u", scope="w", session="s", time_intent="previous")
    assert {item.content_id for item in current} == {correction.content_id, other.content_id}
    assert [item.content_id for item in previous] == [
        next(
            item.content_id
            for item in bundle.store.records
            if item.revision == 0 and item.entity == "雨伞"
        )
    ]
    current_logits = _logits(bundle, time_intent="current")
    previous_logits = _logits(bundle, time_intent="previous")
    assert not torch.allclose(current_logits, previous_logits)


def test_read_depends_on_decoder_prefix():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    _ingest(bundle, span="抽屉", entity="水杯", attribute="位置", value="抽屉", turn=0, revision=0)
    condition = bundle.condition(query=QUERY, owner="u", scope="w", session="s")
    state = bundle.model.conditioned_state(
        condition, scope_id="w", session_id="s", native_state_ref="n"
    )
    cache = bundle.model.content_read_cache(state, condition)
    assert len(cache.content_ids) == 2
    with torch.no_grad():
        first, first_scores = bundle.model._read(cache, state.hidden)
        other_hidden = torch.ones_like(state.hidden)
        second, second_scores = bundle.model._read(cache, other_hidden)
    assert not torch.allclose(first_scores, second_scores)
    assert not torch.allclose(first, second)


def test_first_target_byte_is_supervised_and_prefix_advances():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    left = _logits(bundle, target=b"ab")
    right = _logits(bundle, target=b"cb")
    assert left.shape[0] == 2
    assert torch.allclose(left[0], right[0])
    assert not torch.allclose(left[1], right[1])
    loss = F.cross_entropy(left, torch.tensor([ord("a"), ord("b")], dtype=torch.long))
    assert bool(torch.isfinite(loss))


def test_credit_groups_cover_every_parameter_and_stop_is_observable():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    groups = bundle.model.credit_groups()
    names = {name for name, _ in bundle.model.named_parameters()}
    covered = [name for values in groups.values() for name in values]
    assert set(covered) == names
    assert len(covered) == len(names)
    loss = F.cross_entropy(_logits(bundle), torch.tensor([ord("a"), ord("b")], dtype=torch.long))
    loss.backward()
    has_grad = {
        group: any(
            parameter.grad is not None
            for name, parameter in bundle.model.named_parameters()
            if name in values
        )
        for group, values in groups.items()
    }
    assert has_grad["reader"] and has_grad["decoder"] and has_grad["core"]
    assert not has_grad["content_encoder"]
    bundle.model.zero_grad(set_to_none=True)
    reencoded = bundle.score_target(
        query=QUERY,
        target=b"ab",
        owner="u",
        scope="w",
        session="s",
        reencode=True,
    )
    F.cross_entropy(reencoded, torch.tensor([ord("a"), ord("b")], dtype=torch.long)).backward()
    assert any(
        parameter.grad is not None
        for name, parameter in bundle.model.named_parameters()
        if name in groups["content_encoder"]
    )
    bundle.model.zero_grad(set_to_none=True)
    bundle.model.set_credit_group("reader", False)
    assert all(
        not parameter.requires_grad
        for name, parameter in bundle.model.named_parameters()
        if name in groups["reader"]
    )
    F.cross_entropy(
        _logits(bundle), torch.tensor([ord("a"), ord("b")], dtype=torch.long)
    ).backward()
    assert all(
        parameter.grad is None
        for name, parameter in bundle.model.named_parameters()
        if name in groups["reader"]
    )
    bundle.model.set_credit_group("reader", True)


def test_latent_swap_changes_output_cut_changes_output_rescue_restores_identical():
    bundle = _bundle()
    record = _ingest(
        bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0
    )
    baseline = _logits(bundle)
    other_latent = bundle.model.encode_content("蓝色".encode()).detach().tolist()
    swapped = record.reencode(latent=other_latent, encoder_revision=bundle.model.parameter_revision)
    bundle.store = bundle.store.swap(swapped)
    assert not torch.allclose(baseline, _logits(bundle))
    bundle.store = bundle.store.swap(record)
    rescued = _logits(bundle)
    assert torch.equal(baseline, rescued)
    bundle.store = bundle.store.cut(record.content_id)
    cut = _logits(bundle)
    assert not torch.equal(baseline, cut)
    bundle.store = bundle.store.add(record)
    assert torch.equal(baseline, _logits(bundle))


def test_answer_path_never_reencodes_fact_bytes(monkeypatch):
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)

    def forbidden(_raw: bytes) -> torch.Tensor:
        raise AssertionError("fact bytes entered the answer path")

    monkeypatch.setattr(bundle.model, "encode_content", forbidden)
    answer = bundle.answer(query=QUERY, owner="u", scope="w", session="s", max_new_bytes=8)
    assert len(answer.generated) == 8
    assert answer.selected_ids[0] != ""


def test_zero_step_bundle_restores_in_process_and_fresh_process(tmp_path):
    bundle = _bundle(owner="probe", scope="probe", session="probe")
    _ingest(
        bundle,
        span="红色",
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
        span="蓝色",
        entity="雨伞",
        attribute="颜色",
        value="蓝色",
        turn=4,
        revision=1,
        owner="probe",
        scope="probe",
        session="probe",
    )
    expected = bundle.probe_digest(query=QUERY)
    path = tmp_path / "native-content-bundle.pt"
    bundle.save(path)
    restored = NativeContentBundleV1.load(path)
    assert restored.probe_digest(query=QUERY) == expected
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.native_content_core import NativeContentBundleV1; "
        "b=NativeContentBundleV1.load(sys.argv[1]); "
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


def test_encoder_revision_makes_latents_stale_until_reencoded():
    bundle = _bundle()
    _ingest(bundle, span="红色", entity="雨伞", attribute="颜色", value="红色", turn=0, revision=0)
    bundle.model.parameter_revision += 1
    assert bundle.stale_records()
    with pytest.raises(ValueError):
        _logits(bundle)
    bundle.reencode_store()
    assert not bundle.stale_records()
    assert _logits(bundle).shape[0] == 2


def test_teacher_forced_rejects_empty_target_and_empty_candidate_cache_is_safe():
    bundle = _bundle()
    with pytest.raises(ValueError):
        bundle.score_target(query=QUERY, target=b"", owner="u", scope="w", session="s")
    logits = bundle.score_target(query=QUERY, target=b"ab", owner="u", scope="w", session="s")
    assert torch.isnan(logits).sum() == 0
    answer = bundle.answer(query=QUERY, owner="u", scope="w", session="s", max_new_bytes=4)
    assert answer.selected_ids == ("", "", "", "")
