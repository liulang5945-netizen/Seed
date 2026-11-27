"""Counterfactual full-history identifiability and actual-output episode continuity."""

import copy

import pytest
import torch

from scripts.training import paired_episode_22_5 as paired
from scripts.training.first_a_support_22_5 import accept, make_model, output
from scripts.training.paired_world_22_5 import audit_partition, build_partition, public_example
from taiji.u2_evidence_subject import EvidenceSubject
from taiji.u2_history import value_digest


@pytest.fixture(scope="module")
def data():
    torch.set_num_threads(1)
    return {name: build_partition(name) for name in ("train", "dev")}


@pytest.mark.parametrize("partition", ["train", "dev"])
def test_full_support_oracle_and_partial_history_ceilings(data, partition):
    report = audit_partition(data[partition])
    assert report["worlds"] == 18 and report["episodes"] == 180
    assert report["rounds"] == report["oracle_correct"] == 360
    for counts in report["support"].values():
        assert counts["未知"] == 30
        assert all(counts[label] == 10 for label in ("左边", "中间", "右边"))
    assert all(value["accuracy"] == 0.5 for value in report["partial_input_ceilings"].values())


def test_world_and_family_partition_and_reproducibility(data):
    assert set(data["train"]["worlds"]).isdisjoint(data["dev"]["worlds"])
    train_seeds = {world["audit"]["seed"] for world in data["train"]["worlds"].values()}
    dev_seeds = {world["audit"]["seed"] for world in data["dev"]["worlds"].values()}
    assert len(train_seeds) == len(dev_seeds) == 3 and train_seeds.isdisjoint(dev_seeds)
    assert value_digest(data["train"]) == value_digest(build_partition("train"))
    with pytest.raises(ValueError, match="partition_not_registered"):
        build_partition("check")


def test_retired_cyclic_rule_has_a_full_history_shortcut(data):
    partition = data["train"]
    selected = [
        episode
        for episode in partition["episodes"]
        if episode["audit"]["absent"]
        == (episode["audit"]["visible"] + episode["audit"]["position"] + 1) % 6
    ]
    report = audit_partition({**partition, "episodes": selected})
    assert report["rounds"] == 72
    assert report["partial_input_ceilings"]["question_and_turn"]["accuracy"] == 0.5
    assert (
        report["partial_input_ceilings"]["public_without_rgb_and_question_history"]["accuracy"]
        == 2 / 3
    )


def test_non_rgb_identity_leak_and_unregistered_public_fields_rejected(data):
    partition = copy.deepcopy(data["train"])
    episode = partition["episodes"][0]
    world = partition["worlds"][episode["world_id"]]
    world["public"]["action"][0] = 1
    with pytest.raises(ValueError, match="non_rgb_family_leak"):
        audit_partition(partition)
    world["public"]["gold"] = episode["targets"]
    with pytest.raises(ValueError, match="unregistered_paired_public_fields"):
        public_example(partition, episode)


def test_targets_and_private_audit_do_not_change_public_inputs(data):
    episode = data["train"]["episodes"][0]
    changed = copy.deepcopy(episode)
    changed["targets"] = ("右边".encode(), "中间".encode())
    changed["audit"] = {"arbitrary": "changed"}
    assert value_digest(public_example(data["train"], episode)) == value_digest(
        public_example(data["train"], changed)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_actual_answer_state_survives_second_request_and_task_credit(data, dtype):
    model = EvidenceSubject.from_parent(make_model("native")).to(dtype=dtype)
    partition = copy.deepcopy(data["train"])
    for world in partition["worlds"].values():
        public = world["public"]
        for key in ("observations", "coordinates"):
            public[key] = {name: value.to(dtype=dtype) for name, value in public[key].items()}
        public["outcome"] = public["outcome"].to(dtype=dtype)
    episode = partition["episodes"][0]
    state = paired.start_episode(model, partition, episode)
    for symbol in [*"未知".encode(), 256]:
        _, _, details = output(model, state)
        state = accept(model, state, symbol, details)
    previous_h = state["H"]
    second = paired.next_request(model, state, episode, 1)
    assert second["subject_id"] == state["subject_id"]
    assert second["goal_raw"] == state["goal_raw"]
    assert second["world_time"] == state["world_time"]
    assert len(second["seen_packets"]) == len(second["evidence"]) == 1
    assert previous_h.grad_fn is not None and second["H"].grad_fn is not None
    alternative = paired.start_episode(model, partition, episode)
    for symbol in [*episode["targets"][0], 256]:
        _, _, details = output(model, alternative)
        alternative = accept(model, alternative, symbol, details)
    alternative = paired.next_request(model, alternative, episode, 1)
    assert not torch.equal(second["H"], alternative["H"])
    loss = paired.episode_loss(model, partition, episode)
    gradients = torch.autograd.grad(loss, list(model.evidence_reader.parameters()))
    assert all(bool(torch.isfinite(g).all()) and bool(g.any()) for g in gradients)


def test_wrong_first_generated_answer_is_carried_and_unfinished_round_blocks(monkeypatch, data):
    episode = data["train"]["episodes"][0]
    observed = []
    scripts = [list("未知".encode()) + [256], list("左边".encode()) + [256]]
    monkeypatch.setattr(paired, "start_episode", lambda *_: {"turn": 0, "prefix": b""})

    def second_request(_model, state, _episode, turn):
        observed.append(state["prefix"])
        return {"turn": turn, "prefix": b""}

    def emit(_model, state):
        symbol = scripts[state["turn"]][len(state["prefix"])]
        scores = torch.full((257,), -torch.inf)
        scores[symbol] = 0
        return scores, None, None

    def accept_symbol(_model, state, symbol, _details):
        return {**state, "prefix": state["prefix"] + (b"" if symbol == 256 else bytes([symbol]))}

    monkeypatch.setattr(paired, "next_request", second_request)
    monkeypatch.setattr(paired, "output", emit)
    monkeypatch.setattr(paired, "accept", accept_symbol)
    result = paired.rollout_episode(object(), data["train"], episode)
    assert observed == ["未知".encode()] and result["denominator"] == 2
    assert result["rows"][0]["correct"] is False
    monkeypatch.setattr(
        paired,
        "output",
        lambda *_: (torch.nn.functional.one_hot(torch.tensor(97), 257).float(), None, None),
    )
    blocked = paired.rollout_episode(object(), data["train"], episode)
    assert blocked["denominator"] == 2 and blocked["correct"] == 0
    assert len(bytes.fromhex(blocked["rows"][0]["output_hex"])) == 64
    assert blocked["rows"][1]["status"] == "blocked_by_prior_round"
