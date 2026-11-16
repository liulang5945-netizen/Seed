"""Zero-training contract tests for the C.1 retrieval representation runner.

C.1 adds no subject structure; these pin the runner's scaffolding and guards:
the decode head is zero-initialized (birth at chance), the dataset has the
frozen denominators with distinct distractor entities, the trainable set is
exactly the record channel with emission frozen, and the purity check detects
any drift outside the record channel.
"""

import json
from pathlib import Path

import pytest
import torch

from scripts.training import run_c1_retrieval_20261116 as c1
from taiji.u2_history import atomic_load

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    return json.loads(c1.CONTRACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data(contract):
    return atomic_load(ROOT / contract["data_root"])


def test_decode_head_zero_initialized():
    head = c1.make_decode_head(8)
    assert bool((head.last.weight == 0).all())
    assert bool((head.last.bias == 0).all())
    out = head(torch.zeros(2, 8))
    assert bool((out == 0).all())


def test_dataset_denominators_and_distinct_distractors(contract, data):
    ds = c1.build_dataset(data, contract)
    assert len(ds["decode"]["train"]) == 192
    assert len(ds["decode"]["fit"]) == 96
    assert len(ds["pairs"]) == 32
    for item in ds["decode"]["fit"]:
        assert item["label"] in c1.POSITIONS
    for pair in ds["pairs"]:
        assert pair["match_raw"] != pair["dist_raw"]


def test_trainable_set_is_record_channel_emission_frozen(contract, data):
    model = c1.b1.build_v4_reference(contract, 20261110)
    found = c1.set_trainable_c1(model, contract)
    assert found == sorted(contract["record_parameters"])
    params = dict(model.named_parameters())
    for name in contract["emission_parameters"]:
        assert params[name].requires_grad is False
    for name in contract["record_parameters"]:
        assert params[name].requires_grad is True


def test_purity_check_detects_non_record_drift(contract, data):
    model = c1.b1.build_v4_reference(contract, 20261110)
    start = torch.load(ROOT / contract["start_bundles"]["20261110"], weights_only=False)
    assert c1.purity_check_c1(model, start, contract)["pure"] is True
    with torch.no_grad():
        model.byte_head.weight.add_(0.1)
    report = c1.purity_check_c1(model, start, contract)
    assert report["pure"] is False
    assert "byte_head.weight" in report["drifted"]
