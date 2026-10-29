from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from taiji.adapter import TSKV8Adapter
from taiji.conditional_language_v3 import TaskCoreConfigV3
from taiji.conditional_language_v4 import (
    ConditionalLanguageCandidateV4,
    ConditionalLanguageTrainerV4,
)
from taiji.config import TaijiConfig
from taiji.contracts import NativeCheckpoint, WorldEvent, WorldObject
from taiji.input_boundary import InputFrame
from taiji.internalization import content_digest
from taiji.joint_input_producer import (
    ACT_NAMES,
    TAG_NAMES,
    JointInputProducerV1,
    JointSemanticParseV1,
    JointSemanticSpanV1,
    JointSemanticTrainingTargetV1,
)
from taiji.joint_t1_course import JointT1AnswerTrainingTargetV1
from taiji.joint_t1_trainer import JointT1TrainerV1, _mode_policy
from taiji.joint_task_state import JointTaskStateV1


def setup_joint(text: str, *, input_id: str = "producer-input", width: int = 8):
    adapter = TSKV8Adapter(
        TaijiConfig(region_sizes=(24, 16), synapse_fan_in=6, motor_fan_in=8, seed=71)
    )
    language_model = ConditionalLanguageCandidateV4(
        TaskCoreConfigV3(
            width=width,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=256,
            max_input_evidence_bytes=256,
            max_output_position=32,
            max_generation_bytes=32,
            seed=37,
        )
    )
    root_before = adapter.cognitive_snapshot()
    joint = JointTaskStateV1.begin(
        language_model,
        root_before,
        joint_bundle_id="joint-v1:producer-test",
        owner_id="user-1",
        scope_id="scope-1",
        session_id="session-producer",
    )
    frame = InputFrame(
        input_id=input_id,
        modality="text",
        payload=text.encode("utf-8"),
        source="test.client",
        timestamp=3,
    )
    trace = adapter.ingest_input(frame, learn=False)
    root_after = adapter.cognitive_snapshot()
    joint = joint.append_input(
        language_model,
        trace,
        root_after,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )
    return adapter, language_model, trace, joint


def byte_span(text: str, fragment: str) -> tuple[int, int]:
    char_start = text.index(fragment)
    prefix = text[:char_start].encode("utf-8")
    content = fragment.encode("utf-8")
    return len(prefix), len(prefix) + len(content)


def logits_for_fact(text: str, *, act: str = "assertion", value: str = "蓝色"):
    raw = text.encode("utf-8")
    tag_ids = [0] * len(raw)
    slots = [("ENTITY", "杯子"), ("ATTRIBUTE", "颜色")]
    if act != "query":
        slots.append(("VALUE", value))
    for role, fragment in slots:
        start, end = byte_span(text, fragment)
        tag_ids[start] = TAG_NAMES.index(f"B_{role}")
        for index in range(start + 1, end):
            tag_ids[index] = TAG_NAMES.index(f"I_{role}")
    slot_logits = torch.full((len(raw), len(TAG_NAMES)), -12.0)
    for index, tag_id in enumerate(tag_ids):
        slot_logits[index, tag_id] = 12.0
    act_logits = torch.full((len(ACT_NAMES),), -12.0)
    act_logits[ACT_NAMES.index(act)] = 12.0
    return slot_logits, act_logits


class FixedLogitProducer(JointInputProducerV1):
    """Test double for the span decoder; its fixed outputs are not capability evidence."""

    def __init__(
        self,
        *,
        trace,
        act="assertion",
        value="蓝色",
        negation=False,
        low_confidence=False,
        hidden_dim=8,
    ):
        super().__init__(
            percept_feature_dim=trace.percepts[0].features.numel(), hidden_dim=hidden_dim
        )
        self.test_act = act
        self.test_value = value
        self.test_negation = negation
        self.test_low_confidence = low_confidence

    def forward(self, trace):
        text = bytes(item.value for item in trace.observations).decode("utf-8")
        slot_logits, act_logits = logits_for_fact(text, act=self.test_act, value=self.test_value)
        if self.test_negation:
            start, end = byte_span(text, "是")
            slot_logits[start, TAG_NAMES.index("B_NEGATION")] = 30.0
            for index in range(start + 1, end):
                slot_logits[index, TAG_NAMES.index("I_NEGATION")] = 30.0
        if self.test_low_confidence:
            act_logits = torch.zeros_like(act_logits)
        encoded = torch.arange(1, self.hidden_dim + 1, dtype=torch.float32).expand(
            len(trace.observations), -1
        )
        return slot_logits, act_logits, encoded


def make_parse(producer, joint, trace):
    return producer.predict(joint, trace)


def test_native_span_parse_updates_world_and_v4_reads_the_exact_source():
    text = "杯子的颜色是蓝色。"
    _, language_model, trace, joint = setup_joint(text)
    producer = FixedLogitProducer(trace=trace)
    parse = make_parse(producer, joint, trace)
    assert parse.status == "resolved"
    assert {span.role for span in parse.spans} == {"entity", "attribute", "value"}
    assert all(span.input_id == trace.input_id for span in parse.spans)
    forged_fields = {
        field.name: getattr(parse, field.name)
        for field in dataclass_fields(parse)
        if field.name != "parse_sha256"
    }
    forged_fields["semantic_features"] = tuple(item + 1.0 for item in parse.semantic_features)
    forged = JointSemanticParseV1.create(**forged_fields)
    with pytest.raises(ValueError, match="differs from this producer's current model output"):
        producer.commit_to_joint_state(joint, trace, forged, language_model)
    updated = producer.commit_to_joint_state(joint, trace, parse, language_model)
    assert updated.joint_revision == joint.joint_revision + 2
    object_state = updated.cognitive_state.world.objects[0]
    assert object_state.attribute("颜色") == "蓝色"
    event = updated.cognitive_state.world.events[-1]
    assert event.provenance == trace.input_id and ("颜色", "蓝色") in event.attributes
    view = updated.content_view()
    assert view.selected_candidates[0].source_status == "source_linked"
    assert view.selected_neural_source_ids == (updated.input_bindings[0].neural_source_id,)
    logits, _ = language_model.next_logits_from_joint(updated)
    assert logits.shape[-1] == 257
    raw_only_neural = replace(
        updated.language_state,
        workspace=updated.language_state.workspace[:-1],
        evidence_refs=updated.language_state.evidence_refs[:-1],
    )
    raw_only = updated.with_language_state(raw_only_neural)
    raw_logits, _ = language_model.next_logits_from_joint(raw_only)
    assert not torch.allclose(logits, raw_logits)


@pytest.mark.parametrize(("producer_width", "workspace_width"), [(4, 8), (48, 256)])
def test_smaller_producer_features_reach_both_workspace_paths_without_truncation(
    producer_width, workspace_width
):
    text = "杯子的颜色是蓝色。"
    _, language_model, trace, joint = setup_joint(text, width=workspace_width)
    producer = FixedLogitProducer(trace=trace, hidden_dim=producer_width)
    parse = producer.predict(joint, trace)
    assert parse.status == "resolved"

    updated = producer.commit_to_joint_state(joint, trace, parse, language_model)
    expected = torch.cat(
        (
            torch.arange(1, producer_width + 1, dtype=torch.float32),
            torch.zeros(workspace_width - producer_width),
        )
    )
    assert torch.equal(updated.cognitive_state.workspace.candidates[-1].features, expected)
    assert torch.equal(updated.language_state.workspace[-1], expected)
    logits, _ = language_model.next_logits_from_joint(updated)
    without_semantic_row = updated.with_language_state(
        replace(
            updated.language_state,
            workspace=updated.language_state.workspace[:-1],
            evidence_refs=updated.language_state.evidence_refs[:-1],
        )
    )
    raw_logits, _ = language_model.next_logits_from_joint(without_semantic_row)
    assert not torch.allclose(logits, raw_logits)


def test_ambiguous_or_negated_candidate_never_writes_a_positive_fact():
    text = "杯子的颜色是蓝色。"
    _, language_model, trace, joint = setup_joint(text)
    producer = FixedLogitProducer(trace=trace, act="ambiguous")
    ambiguous = make_parse(producer, joint, trace)
    assert ambiguous.status == "ambiguous"
    assert producer.commit_to_joint_state(joint, trace, ambiguous, language_model) is joint

    negation_producer = FixedLogitProducer(trace=trace, negation=True)
    negated = make_parse(negation_producer, joint, trace)
    assert negated.status == "ambiguous"
    assert negation_producer.commit_to_joint_state(joint, trace, negated, language_model) is joint


def test_query_with_asserted_value_is_marked_ambiguous_before_state_commit():
    text = "杯子的颜色是蓝色。"
    _, language_model, trace, joint = setup_joint(text)

    class QueryWithValueProducer(FixedLogitProducer):
        def forward(self, trace):
            slot_logits, _, encoded = super().forward(trace)
            act_logits = torch.full((len(ACT_NAMES),), -12.0)
            act_logits[ACT_NAMES.index("query")] = 12.0
            return slot_logits, act_logits, encoded

    producer = QueryWithValueProducer(trace=trace)
    parse = producer.predict(joint, trace)
    assert parse.act == "query"
    assert {span.role for span in parse.spans} == {"entity", "attribute", "value"}
    assert parse.status == "ambiguous"
    assert producer.commit_to_joint_state(joint, trace, parse, language_model) is joint


def test_correction_updates_only_the_bound_entity_attribute_and_keeps_history():
    text = "杯子的颜色是蓝色。"
    adapter, language_model, trace, joint = setup_joint(text)
    producer = FixedLogitProducer(trace=trace, act="correction")
    cup_id = producer.entity_object_id(joint, "杯子")
    old_cup = WorldObject(cup_id, attributes=(("位置", "桌上"), ("颜色", "红色")))
    unrelated = WorldObject("native-entity:unrelated", attributes=(("颜色", "绿色"),))
    previous = replace(
        joint.cognitive_state,
        world=replace(
            joint.cognitive_state.world,
            entities=(cup_id, unrelated.object_id),
            objects=(old_cup, unrelated),
            events=(
                WorldEvent(
                    event_id="seeded-fact-event",
                    kind="observed_fact",
                    tick=joint.cognitive_state.world.tick,
                    subject_id=cup_id,
                    object_id=cup_id,
                    attributes=old_cup.attributes,
                    provenance=trace.input_id,
                ),
            ),
        ),
    )
    seeded = joint.advance_cognitive_state(previous)
    parse = make_parse(producer, seeded, trace)
    updated = producer.commit_to_joint_state(seeded, trace, parse, language_model)
    objects = {item.object_id: item for item in updated.cognitive_state.world.objects}
    assert objects[cup_id].attribute("颜色") == "蓝色"
    assert objects[cup_id].attribute("位置") == "桌上"
    assert objects[unrelated.object_id].attribute("颜色") == "绿色"
    assert updated.cognitive_state.world.events[-1].kind == "native_fact_correction"
    assert len(updated.cognitive_state.world.events) == len(seeded.cognitive_state.world.events) + 1
    synced = producer.bind_cognitive_owner_to_native_adapter_checkpoint(
        adapter.native_checkpoint(), updated
    )
    resumed_adapter = TSKV8Adapter.from_native_checkpoint(synced)
    correction_text = "杯子的颜色是红色。"
    correction_trace = resumed_adapter.ingest_input(
        InputFrame(
            input_id="producer-correction-input",
            modality="text",
            payload=correction_text.encode("utf-8"),
            source="test.client",
            timestamp=resumed_adapter.tick + 1,
        ),
        learn=False,
    )
    correction_root = resumed_adapter.cognitive_snapshot()
    correction_joint = updated.append_input(
        language_model,
        correction_trace,
        correction_root,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )
    correction_producer = FixedLogitProducer(trace=correction_trace, act="correction", value="红色")
    correction_parse = make_parse(correction_producer, correction_joint, correction_trace)
    corrected = correction_producer.commit_to_joint_state(
        correction_joint, correction_trace, correction_parse, language_model
    )
    corrected_objects = {item.object_id: item for item in corrected.cognitive_state.world.objects}
    assert corrected_objects[cup_id].attribute("颜色") == "红色"
    assert corrected_objects[cup_id].attribute("位置") == "桌上"
    assert corrected_objects[unrelated.object_id].attribute("颜色") == "绿色"
    corrected_logits, _ = language_model.next_logits_from_joint(corrected)
    cut_language_state = replace(
        corrected.language_state,
        workspace=corrected.language_state.workspace[:-1],
        evidence_refs=corrected.language_state.evidence_refs[:-1],
    )
    cut_joint = corrected.with_language_state(cut_language_state)
    cut_logits, _ = language_model.next_logits_from_joint(cut_joint)
    assert not torch.allclose(corrected_logits, cut_logits)
    query_adapter_checkpoint = producer.bind_cognitive_owner_to_native_adapter_checkpoint(
        resumed_adapter.native_checkpoint(), corrected
    )
    query_adapter = TSKV8Adapter.from_native_checkpoint(query_adapter_checkpoint)
    query_text = "杯子的颜色是什么？"
    query_trace = query_adapter.ingest_input(
        InputFrame(
            input_id="producer-query-input",
            modality="text",
            payload=query_text.encode("utf-8"),
            source="test.client",
            timestamp=query_adapter.tick + 1,
        ),
        learn=False,
    )
    query_root = query_adapter.cognitive_snapshot()
    query_joint = corrected.append_input(
        language_model,
        query_trace,
        query_root,
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 256},
    )
    query_producer = FixedLogitProducer(trace=query_trace, act="query")
    query_parse = make_parse(query_producer, query_joint, query_trace)
    assert query_parse.status == "resolved" and query_parse.act == "query"
    queried = query_producer.commit_to_joint_state(
        query_joint, query_trace, query_parse, language_model
    )
    queried_objects = {item.object_id: item for item in queried.cognitive_state.world.objects}
    assert queried_objects[cup_id].attribute("颜色") == "红色"
    query_view = queried.content_view()
    assert query_view.selected_candidates[0].source_status == "source_linked"
    assert set(query_view.selected_neural_source_ids) == {
        binding.neural_source_id for binding in queried.input_bindings
    }
    query_logits, _ = language_model.next_logits_from_joint(queried)
    assert query_logits.shape[-1] == 257
    query_cut_state = replace(
        queried.language_state,
        workspace=queried.language_state.workspace[:-1],
        evidence_refs=queried.language_state.evidence_refs[:-1],
    )
    query_cut = queried.with_language_state(query_cut_state)
    query_cut_logits, _ = language_model.next_logits_from_joint(query_cut)
    assert not torch.allclose(query_logits, query_cut_logits)


@torch.random.fork_rng(devices=[])
def test_separate_task_losses_reach_only_their_declared_parameter_groups(monkeypatch):
    """Exercise the real producer/answer paths without stepping an optimizer."""
    torch.manual_seed(37)
    text = "杯子的颜色是蓝色。"
    _, model, trace, joint = setup_joint(text)
    producer = JointInputProducerV1(
        percept_feature_dim=trace.percepts[0].features.numel(), hidden_dim=8
    )
    fixed_slot, fixed_act = logits_for_fact(text)
    original_forward = producer.forward

    def resolved_forward(current_trace):
        slot, act, encoded = original_forward(current_trace)
        return fixed_slot + slot * 0.001, fixed_act + act * 0.001, encoded

    monkeypatch.setattr(producer, "forward", resolved_forward)
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    semantic_target = JointSemanticTrainingTargetV1.create(
        input_id=trace.input_id,
        source_sha256=source_sha,
        family_id="synthetic-family",
        partition="train",
        tag_ids=tuple(int(item) for item in fixed_slot.argmax(dim=-1).tolist()),
        act_id=ACT_NAMES.index("assertion"),
    )
    frame = InputFrame(
        input_id=trace.input_id,
        modality="text",
        payload=text.encode("utf-8"),
        source="test.client",
        timestamp=3,
    )
    answer_target = JointT1AnswerTrainingTargetV1.create(
        input_id=trace.input_id,
        source_sha256=source_sha,
        family_id="synthetic-family",
        partition="train",
        target_bytes=b"x<|end|>",
    )
    turn = SimpleNamespace(
        frame=frame,
        family_id="synthetic-family",
        level="F1",
        semantic_target=semantic_target,
        answer_target=answer_target,
    )
    trainer = SimpleNamespace(producer=producer, model=model, mode="correct")
    before = (content_digest(producer.state_dict()), content_digest(model.state_dict()))
    parser_loss, components = JointT1TrainerV1._semantic_loss(trainer, joint, trace, turn)
    parse = producer.predict(joint, trace)
    assert parse.status == "resolved"
    committed = producer.commit_to_joint_state(joint, trace, parse, model)
    answer_loss, _, _ = JointT1TrainerV1._answer_loss(trainer, committed, turn)

    slot_parameters = tuple(producer.slot_head.parameters())
    act_parameters = tuple(producer.act_head.parameters())
    shared_parameters = tuple(producer.context.parameters())
    producer_parameters = tuple(producer.parameters())

    def has_signal(loss, parameters):
        if not loss.requires_grad:
            return False
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return any(
            g is not None and torch.isfinite(g).all() and torch.any(g != 0) for g in gradients
        )

    assert has_signal(components["slot"], slot_parameters)
    assert has_signal(components["slot"], shared_parameters)
    assert not has_signal(components["slot"], act_parameters)
    assert has_signal(components["act"], act_parameters)
    assert has_signal(components["act"], shared_parameters)
    assert not has_signal(components["act"], slot_parameters)
    assert not has_signal(components["slot"] * 0, slot_parameters)
    assert has_signal(parser_loss, producer_parameters)
    assert all(
        gradient is None
        for gradient in torch.autograd.grad(
            answer_loss, producer_parameters, retain_graph=True, allow_unused=True
        )
    )
    assert has_signal(answer_loss, tuple(model.output.parameters()))
    assert has_signal(answer_loss + producer.slot_head.weight.sum(), producer_parameters)

    committed_modes = []
    original_commit = producer.commit_to_joint_state

    def record_commit(*args, **kwargs):
        committed_modes.append(True)
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(producer, "commit_to_joint_state", record_commit)
    for mode, parser_enabled, language_enabled, writes in (
        ("correct", True, True, True),
        ("parser_permuted", True, True, True),
        ("no_update", False, False, False),
        ("readout_only", False, True, False),
        ("responsible_owner_frozen", False, True, True),
    ):
        harness = SimpleNamespace(
            producer=producer, model=model, mode=mode, policy=_mode_policy(mode)
        )
        harness._new_episode_state = lambda _episode_id: (None, joint)
        harness._append_turn = lambda _adapter, _joint, _turn: (trace, joint)
        harness._semantic_loss = MethodType(JointT1TrainerV1._semantic_loss, harness)
        harness._answer_loss = MethodType(JointT1TrainerV1._answer_loss, harness)
        committed_modes.clear()
        semantic, answer, _, _ = JointT1TrainerV1._run_episode(
            harness, SimpleNamespace(episode_id="synthetic-episode", turns=(turn,))
        )
        assert bool(committed_modes) is writes
        assert has_signal(semantic, producer_parameters) is parser_enabled
        assert has_signal(answer, tuple(model.output.parameters())) is language_enabled
        assert not has_signal(answer, producer_parameters)
        model.zero_grad(set_to_none=True)
        if language_enabled:
            named_parameters = tuple(model.named_parameters())
            gradients = torch.autograd.grad(
                answer,
                tuple(parameter for _, parameter in named_parameters),
                retain_graph=True,
                allow_unused=True,
            )
            for (_, parameter), gradient in zip(named_parameters, gradients, strict=True):
                parameter.grad = None if gradient is None else gradient.detach().clone()
            JointT1TrainerV1._mask_frozen_language_gradients(harness)
            admitted = {name for name, parameter in named_parameters if parameter.grad is not None}
            assert any(name.startswith("output.") for name in admitted)
            if mode == "readout_only":
                assert all(name.startswith("output.") for name in admitted)
        model.zero_grad(set_to_none=True)

    assert before == (content_digest(producer.state_dict()), content_digest(model.state_dict()))


def test_fit_loss_is_source_bound_train_only_and_reaches_producer_parameters():
    text = "杯子的颜色是蓝色。"
    _, _, trace, joint = setup_joint(text)
    producer = JointInputProducerV1(
        percept_feature_dim=trace.percepts[0].features.numel(), hidden_dim=8
    )
    slot_logits, _ = logits_for_fact(text)
    tag_ids = tuple(int(item) for item in slot_logits.argmax(dim=-1).tolist())
    source_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    target = JointSemanticTrainingTargetV1.create(
        input_id=trace.input_id,
        source_sha256=source_sha,
        family_id="family-fact-1",
        partition="train",
        tag_ids=tag_ids,
        act_id=ACT_NAMES.index("assertion"),
    )
    assert producer.PARAMETER_OWNER == target.RECEIVER == "input_semantic_producer"
    assert JointSemanticTrainingTargetV1.from_payload(target.to_payload()) == target
    model_before = content_digest(producer.checkpoint())
    loss, components = producer.supervised_loss(joint, trace, target)
    assert torch.isfinite(loss) and set(components) == {"slot", "act"}
    gradients = torch.autograd.grad(loss, tuple(producer.parameters()))
    assert any(gradient is not None and bool(torch.any(gradient != 0)) for gradient in gradients)
    assert content_digest(producer.checkpoint()) == model_before
    support_target = JointSemanticTrainingTargetV1.create(
        input_id=trace.input_id,
        source_sha256=source_sha,
        family_id="family-fact-1",
        partition="support",
        tag_ids=tag_ids,
        act_id=ACT_NAMES.index("assertion"),
    )
    with pytest.raises(ValueError, match="support/query/holdout"):
        producer.supervised_loss(joint, trace, support_target)


def test_parse_spans_reject_utf8_splits_and_native_checkpoint_restores_ledger(tmp_path):
    text = "杯子的颜色是蓝色。"
    _, language_model, trace, joint = setup_joint(text)
    producer = FixedLogitProducer(trace=trace)
    parse = make_parse(producer, joint, trace)
    with pytest.raises(ValueError, match="UTF-8"):
        JointSemanticSpanV1(
            role="entity",
            input_id=trace.input_id,
            source_sha256=parse.source_sha256,
            start_byte=0,
            end_byte=1,
            span_bytes=b"\xe6",
            span_sha256=hashlib.sha256(b"\xe6").hexdigest(),
            percept_event_ids=(trace.percepts[0].event_id,),
            assembly_ids=(trace.percepts[0].assembly_id,),
            confidence=1.0,
        )

    joint_with_fact = producer.commit_to_joint_state(joint, trace, parse, language_model)
    trainer = ConditionalLanguageTrainerV4(language_model)
    lang_checkpoint = trainer.checkpoint(joint_with_fact.language_state, cursor={"step": 0})
    native = NativeCheckpoint(
        kernel={"format": "synthetic-task-kernel-v1"},
        cognitive_state=joint_with_fact.cognitive_state,
    )
    joint_envelope = joint_with_fact.attach_to_native_checkpoint(
        native.to_payload(), lang_checkpoint
    )
    full_envelope = producer.attach_to_native_checkpoint(
        joint_envelope, joint_with_fact, parses=(parse,)
    )
    restored_joint, _ = JointTaskStateV1.restore_native_checkpoint(
        full_envelope,
        joint_bundle_id=joint_with_fact.joint_bundle_id,
        owner_id=joint_with_fact.owner_id,
        scope_id=joint_with_fact.scope_id,
        session_id=joint_with_fact.session_id,
    )
    restored_producer, parses = JointInputProducerV1.restore_from_native_checkpoint(
        full_envelope, restored_joint
    )
    assert content_digest(restored_producer.checkpoint()) == content_digest(producer.checkpoint())
    assert len(parses) == 1 and parses[0].parse_sha256 == parse.parse_sha256
    checkpoint_path = tmp_path / "joint-input-producer.pt"
    torch.save(full_envelope, checkpoint_path)
    child = """
import hashlib, json, sys, torch
from taiji.joint_task_state import JointTaskStateV1
from taiji.joint_input_producer import JointInputProducerV1
payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
state, _ = JointTaskStateV1.restore_native_checkpoint(
    payload,
    joint_bundle_id='joint-v1:producer-test',
    owner_id='user-1',
    scope_id='scope-1',
    session_id='session-producer',
)
producer, parses = JointInputProducerV1.restore_from_native_checkpoint(payload, state)
print(json.dumps({
    'producer_bundle_id': producer.bundle_id,
    'parse_sha256': parses[0].parse_sha256,
    'attribute_sha256': hashlib.sha256(
        str(next(iter(state.cognitive_state.world.objects[0].attributes))[1]).encode('utf-8')
    ).hexdigest(),
}))
"""
    child_result = subprocess.run(
        [sys.executable, "-c", child, str(checkpoint_path)],
        cwd=str(tmp_path.parents[2]),
        capture_output=True,
        text=True,
    )
    assert child_result.returncode == 0, child_result.stderr
    assert json.loads(child_result.stdout.strip()) == {
        "producer_bundle_id": producer.bundle_id,
        "parse_sha256": parse.parse_sha256,
        "attribute_sha256": hashlib.sha256("蓝色".encode()).hexdigest(),
    }
