"""Directed tests for the 14.4 record-relation content interface.

These are unit/integration checks on the formal graph, not capability
evaluations.  They use the fixed 09.8 parent and the two public engineering
fixtures only.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from taiji.living_core import EOS, RawSource
from taiji.native_goal_glossary import glossary_records
from taiji.record_relation import (
    RecordRelationConfig,
    RecordRelationModule,
    expected_parameter_count,
)
from taiji.record_relation_core import (
    ARM_FLAT,
    ARM_FULL,
    ARM_LOCAL,
    DISABLED_NAMES_BY_ARM,
    PARAMETER_GROUPS,
    answer_step,
    build_candidate,
    decode_endpoint,
    forced_read,
    load_bundle,
    make_optimizer,
    observe_goal,
    prepare,
    save_bundle,
    score_example,
)

pytestmark = pytest.mark.skipif(
    not Path(__file__)
    .resolve()
    .parents[2]
    .joinpath("output/taiji_core_09/subject_bundle.pt")
    .exists(),
    reason="the fixed 09.8 parent bundle is not present",
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "output/taiji_core_09/subject_bundle.pt"


@pytest.fixture(scope="module")
def parent():
    torch.set_num_threads(1)
    return torch.load(BUNDLE, map_location="cpu", weights_only=True)


@pytest.fixture(scope="module")
def fixture() -> dict:
    records = glossary_records()
    return {
        "goal": "让圆口的器皿到四方木台。",
        "records": records,
        "target": tuple("把蓝色杯子放到木桌".encode()) + (EOS,),
    }


def test_parameter_count_matches_spec() -> None:
    module = RecordRelationModule(RecordRelationConfig())
    assert sum(parameter.numel() for parameter in module.parameters()) == 172801
    assert expected_parameter_count(128, 32, 64) == 172801


def test_arm_disabled_counts(parent) -> None:
    expected = {ARM_FULL: 0, ARM_LOCAL: 91008, ARM_FLAT: 9504}
    for arm, count in expected.items():
        candidate = build_candidate(parent, arm=arm)
        assert candidate.disabled_rows()["elements"] == count
        named = dict(candidate.named_parameters())
        for name in DISABLED_NAMES_BY_ARM[arm]:
            assert named[name].requires_grad is False


def test_record_relationship_full_and_local(parent, fixture) -> None:
    record = fixture["records"][0]
    early, late = 5, 20
    replacement = record.raw[:18] + "红色印章".encode() + record.raw[30:]
    assert len(replacement) == len(record.raw)
    rows = {}
    for arm in (ARM_FULL, ARM_LOCAL):
        candidate = build_candidate(parent, arm=arm)
        units = candidate.relation.encode_units(candidate.subject, record.raw)
        gradient = torch.autograd.grad(
            units[early].sum(), candidate.subject.encoder.embedding.weight, allow_unused=True
        )[0]
        derivative = float(gradient[record.raw[late]].abs().max())
        replaced = candidate.relation.encode_units(candidate.subject, replacement)
        delta = float((replaced[early].detach() - units[early].detach()).abs().max())
        view = candidate.relation.build_view(candidate.subject, fixture["records"])
        alone = candidate.relation.encode_units(candidate.subject, record.raw)
        other = float((view.units[0].detach() - alone.detach()).abs().max())
        rows[arm] = (derivative, delta, other)
    assert rows[ARM_FULL][0] > 1e-6
    assert rows[ARM_FULL][1] > 1e-6
    assert rows[ARM_LOCAL][0] == 0.0
    assert rows[ARM_LOCAL][1] == 0.0
    assert rows[ARM_FULL][2] == 0.0
    assert rows[ARM_LOCAL][2] == 0.0


def test_content_path_and_intervention(parent, fixture) -> None:
    candidate = build_candidate(parent)
    result = score_example(
        candidate,
        fixture["goal"].encode(),
        fixture["records"],
        fixture["target"][:-1],
    )
    prepared = result.prepared
    view = prepared.view
    state = result.observation_state
    coverage = result.observation_coverage
    unforced = answer_step(candidate, prepared, state, coverage, EOS, gate_index=0)
    total = float((unforced.read.copy_weights.sum() + unforced.read.alpha_null).detach())
    assert abs(total - 1.0) <= 1e-12
    assert abs(float(unforced.probabilities.sum().detach()) - 1.0) <= 1e-12
    assert float(unforced.read.copy_byte_mass[EOS].detach()) == 0.0
    correct_index = next(index for index, raw in enumerate(view.raw) if "蓝色杯子".encode() in raw)
    correct_offset = view.raw[correct_index].index("蓝色杯子".encode())
    correct_byte = view.raw[correct_index][correct_offset]
    correct = forced_read(
        candidate,
        prepared,
        state,
        coverage,
        EOS,
        record_index=correct_index,
        offset=correct_offset,
        gamma_override=0.0,
    )
    assert abs(float(correct.probabilities[correct_byte].detach()) - 1.0) <= 1e-12
    assert abs(float(correct.probabilities.sum().detach()) - 1.0) <= 1e-12
    assert abs(float(correct.probabilities[EOS].detach())) <= 1e-12
    wrong_index = next(index for index, raw in enumerate(view.raw) if "木桌".encode() in raw)
    wrong_offset = view.raw[wrong_index].index("木桌".encode())
    wrong_byte = view.raw[wrong_index][wrong_offset]
    wrong = forced_read(
        candidate,
        prepared,
        state,
        coverage,
        EOS,
        record_index=wrong_index,
        offset=wrong_offset,
        gamma_override=0.0,
    )
    assert abs(float(wrong.probabilities[wrong_byte].detach()) - 1.0) <= 1e-12
    assert abs(float(wrong.probabilities[correct_byte].detach())) <= 1e-12
    null = forced_read(
        candidate,
        prepared,
        state,
        coverage,
        EOS,
        record_index=None,
        offset=None,
        gamma_override=0.0,
    )
    assert torch.equal(null.probabilities, null.generated)


def test_label_and_identity_isolation(parent, fixture) -> None:
    candidate = build_candidate(parent)
    prefix = tuple(fixture["target"][:-1])
    wrong = tuple("把红色印章放到窗台".encode())
    base = score_example(candidate, fixture["goal"].encode(), fixture["records"], prefix)
    swapped = score_example(
        candidate, fixture["goal"].encode(), fixture["records"], prefix, labels=wrong
    )
    assert torch.equal(base.distributions, swapped.distributions)
    assert abs(float(base.nll.detach()) - float(swapped.nll.detach())) > 1e-9

    mapping = {record.source_id: f"r14-{index}" for index, record in enumerate(fixture["records"])}
    renamed = tuple(
        RawSource(
            source_id=mapping[record.source_id],
            raw=record.raw,
            observed_at=record.observed_at,
            version=record.version,
            owner=record.owner,
            scope=record.scope,
            real=record.real,
        )
        for record in fixture["records"]
    )
    renamed_state = base.prefix_state
    renamed_state = type(renamed_state)(
        subject_id=renamed_state.subject_id,
        revision=renamed_state.revision,
        step=renamed_state.step,
        h=renamed_state.h,
        fast=renamed_state.fast,
        nu_ema=renamed_state.nu_ema,
        keys=tuple(
            (mapping[source], version, offset) for source, version, offset in renamed_state.keys
        ),
        coverage=renamed_state.coverage,
        prior=renamed_state.prior,
    )
    prepared_renamed = prepare(candidate, renamed_state, fixture["goal"].encode(), renamed)
    original = answer_step(
        candidate,
        base.prepared,
        base.prefix_state,
        base.prefix_coverage,
        prefix[-1],
        gate_index=0,
    )
    restored = answer_step(
        candidate,
        prepared_renamed,
        renamed_state,
        renamed_state.coverage,
        prefix[-1],
        gate_index=0,
    )
    assert torch.equal(original.probabilities, restored.probabilities)


def test_output_endpoint_negative_control() -> None:
    wrong_byte = "红色印章".encode()[0]
    distribution = torch.zeros(257, dtype=torch.float64)
    distribution[wrong_byte] = 1.0
    endpoint = decode_endpoint(distribution)
    assert endpoint["byte"] == wrong_byte
    assert endpoint["eos"] is False
    assert torch.equal(endpoint["probabilities"], distribution)
    eos_distribution = torch.zeros(257, dtype=torch.float64)
    eos_distribution[EOS] = 1.0
    endpoint = decode_endpoint(eos_distribution)
    assert endpoint["byte"] == EOS
    assert endpoint["eos"] is True
    assert torch.equal(endpoint["probabilities"], eos_distribution)


def test_bundle_roundtrip(parent, fixture, tmp_path) -> None:
    candidate = build_candidate(parent)
    optimizer = make_optimizer(candidate)
    state = candidate.subject.initial_state("test-restore", revision=1)
    prepared = prepare(candidate, state, fixture["goal"].encode(), fixture["records"])
    observation, gate_index = observe_goal(candidate, prepared)
    probe = answer_step(
        candidate, prepared, observation, observation.coverage, EOS, gate_index=gate_index
    )
    bundle = tmp_path / "bundle.pt"
    save_bundle(
        bundle,
        profile="test",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=0,
        cursor={},
        parent_identity={},
    )
    restored, restored_state, _optimizer, *_rest = load_bundle(bundle, torch.float64)
    restored_prepared = prepare(
        restored, restored_state, fixture["goal"].encode(), fixture["records"]
    )
    restored_probe = answer_step(
        restored,
        restored_prepared,
        restored_state,
        restored_state.coverage,
        EOS,
        gate_index=gate_index,
    )
    assert torch.equal(probe.probabilities, restored_probe.probabilities)
    assert torch.equal(observation.h, restored_state.h)
    assert torch.equal(observation.coverage, restored_state.coverage)


def test_bundle_roundtrip_float32(parent, fixture, tmp_path) -> None:
    candidate = build_candidate(parent, dtype=torch.float32)
    optimizer = make_optimizer(candidate)
    state = candidate.subject.initial_state("test-restore-f32", revision=1)
    prepared = prepare(candidate, state, fixture["goal"].encode(), fixture["records"])
    observation, gate_index = observe_goal(candidate, prepared)
    probe = answer_step(
        candidate, prepared, observation, observation.coverage, EOS, gate_index=gate_index
    )
    bundle = tmp_path / "bundle32.pt"
    save_bundle(
        bundle,
        profile="test32",
        candidate=candidate,
        state=observation,
        optimizer=optimizer,
        update_index=0,
        cursor={},
        parent_identity={},
    )
    restored, restored_state, _optimizer, *_rest = load_bundle(bundle, torch.float32)
    restored_prepared = prepare(
        restored, restored_state, fixture["goal"].encode(), fixture["records"]
    )
    restored_probe = answer_step(
        restored,
        restored_prepared,
        restored_state,
        restored_state.coverage,
        EOS,
        gate_index=gate_index,
    )
    assert torch.equal(probe.probabilities, restored_probe.probabilities)


def test_groups_cover_all_trainable_parameters(parent) -> None:
    candidate = build_candidate(parent)
    named = dict(candidate.named_parameters())
    covered = {name for names in PARAMETER_GROUPS.values() for name in names}
    assert covered <= set(named)
    assert all(name in covered or not parameter.requires_grad for name, parameter in named.items())
