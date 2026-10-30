"""R4 pricing contracts: deterministic schedule, declared credit boundary, restore."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from taiji.internalization import content_digest
from taiji.native_content_training import (
    CREDIT_STOP_GROUPS,
    NativeRunTrainerV1,
    adapter_for,
    batch_schedule,
    credit_on_configure,
    credit_stop_configure,
)

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"


def _episodes(count: int = 1) -> list[dict]:
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    return payload["episodes"][:count]


def _fake_episodes(count: int = 12) -> list[dict]:
    return [
        {
            "episode_id": f"fake:{index}",
            "turns": [{"index": 0, "role": "delay", "scored": False, "text": "等待"}],
        }
        for index in range(count)
    ]


def test_batch_schedule_is_deterministic_and_seed_dependent():
    episodes = _fake_episodes()
    first = batch_schedule(episodes, seed=0, updates=4, batch_size=3)
    again = batch_schedule(episodes, seed=0, updates=4, batch_size=3)
    other = batch_schedule(episodes, seed=1, updates=4, batch_size=3)
    assert [batch.index for batch in first] == [0, 1, 2, 3]
    assert [[episode["episode_id"] for episode in batch.episodes] for batch in first] == [
        [episode["episode_id"] for episode in batch.episodes] for batch in again
    ]
    assert [[episode["episode_id"] for episode in batch.episodes] for batch in first] != [
        [episode["episode_id"] for episode in batch.episodes] for batch in other
    ]
    for batch in first:
        assert len(batch.episodes) == 3
        assert len({episode["episode_id"] for episode in batch.episodes}) == 3


def test_batch_schedule_cycles_epochs_in_order():
    episodes = _fake_episodes(5)
    schedule = batch_schedule(episodes, seed=0, updates=4, batch_size=4)
    seen = [episode["episode_id"] for batch in schedule for episode in batch.episodes]
    assert len(seen) == 16
    assert set(seen[:5]) == {episode["episode_id"] for episode in episodes}


def test_parameter_counts_match_the_declared_r4_contract():
    expected = {
        "new_core_v1": (144_355, 144_355),
        "corrected_old_v1": (123_438, 3_341),
        "ordinary_v1": (120_097, 120_097),
    }
    for name, (total, trainable) in expected.items():
        trainer = NativeRunTrainerV1.create(adapter_for(name))
        counts = trainer.parameter_counts()
        assert (counts["total"], counts["trainable"]) == (total, trainable)


def test_native_update_keeps_persistent_latent_boundary():
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"))
    episode = _episodes(1)[0]
    metrics = trainer.update([episode])
    assert metrics["step"] == 1
    assert metrics["loss_nats"] == pytest.approx(5.5, abs=0.5)
    assert metrics["target_bytes"] == sum(
        len(turn["target"]["target_bytes_utf8"]) for turn in episode["turns"] if turn["scored"]
    )
    assert "content_encoder" not in metrics["gradient_group_norms"]
    assert all(
        name.startswith("content_encoder.") for name in metrics["trainable_without_gradient"]
    )
    for group in ("reader", "decoder", "core", "output"):
        assert metrics["gradient_group_norms"].get(group, 0.0) > 0
    assert trainer.parameters_finite()
    assert trainer.model.parameter_revision == 0


def test_corrected_update_only_trains_the_value_bias():
    trainer = NativeRunTrainerV1.create(adapter_for("corrected_old_v1"))
    metrics = trainer.update(_episodes(1))
    assert metrics["step"] == 1
    assert metrics["loss_nats"] > 5.0
    assert set(metrics["gradient_group_norms"]) == {"value_bias"}
    assert metrics["trainable_without_gradient"] == []
    assert trainer.parameters_finite()


def test_ordinary_update_trains_every_parameter():
    trainer = NativeRunTrainerV1.create(adapter_for("ordinary_v1"))
    counts = trainer.parameter_counts()
    assert counts["trainable"] == counts["total"]
    metrics = trainer.update(_episodes(1))
    assert metrics["step"] == 1
    assert metrics["trainable_without_gradient"] == []
    assert metrics["loss_nats"] > 5.0


def test_zero_step_checkpoint_restores_probe_digest(tmp_path):
    for name in ("new_core_v1", "corrected_old_v1", "ordinary_v1"):
        trainer = NativeRunTrainerV1.create(adapter_for(name))
        digest = trainer.probe_digest()
        path = tmp_path / f"{name}-zero.pt"
        trainer.save(path)
        restored = NativeRunTrainerV1.load(path)
        assert restored.steps == 0
        assert restored.probe_digest() == digest


def test_midtraining_checkpoint_resumes_to_the_same_update(tmp_path):
    episode = _episodes(1)
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"))
    trainer.update(episode)
    path = tmp_path / "native-step1.pt"
    trainer.save(path)
    restored = NativeRunTrainerV1.load(path)
    assert restored.steps == 1
    left = trainer.update(episode)
    right = restored.update(episode)
    assert left["loss_nats"] == pytest.approx(right["loss_nats"], abs=1e-9)
    assert left["gradient_norm_before_clip"] == pytest.approx(
        right["gradient_norm_before_clip"], abs=1e-9
    )
    assert trainer.probe_digest() == restored.probe_digest()
    assert restored.steps == 2


def test_credit_on_keeps_reader_and_decoder_receiving_task_gradient():
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"), configure=credit_on_configure)
    metrics = trainer.update(_episodes(1))
    assert trainer.frozen_groups == ()
    assert metrics["gradient_group_norms"].get("reader", 0.0) > 0
    assert metrics["gradient_group_norms"].get("decoder", 0.0) > 0


def test_credit_stop_freezes_reader_and_decoder_before_adamw():
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"), configure=credit_stop_configure)
    assert trainer.frozen_groups == CREDIT_STOP_GROUPS
    groups = trainer.model.credit_groups()
    frozen = tuple(groups["reader"]) + tuple(groups["decoder"])
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if name in frozen
    }
    metrics = trainer.update(_episodes(1))
    assert "reader" not in metrics["gradient_group_norms"]
    assert "decoder" not in metrics["gradient_group_norms"]
    parameters = dict(trainer.model.named_parameters())
    assert all(torch.equal(value, parameters[name]) for name, value in before.items())
    assert metrics["gradient_group_norms"].get("core", 0.0) > 0


def test_credit_stop_checkpoint_restores_frozen_groups(tmp_path):
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"), configure=credit_stop_configure)
    digest = trainer.probe_digest()
    path = tmp_path / "credit-stop-zero.pt"
    trainer.save(path)
    restored = NativeRunTrainerV1.load(path)
    assert restored.frozen_groups == CREDIT_STOP_GROUPS
    assert len(restored.trainable) == len(trainer.trainable)
    assert restored.probe_digest() == digest


def test_version1_checkpoint_without_frozen_groups_still_loads(tmp_path):
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"))
    payload = trainer.checkpoint()
    payload.pop("frozen_groups")
    payload["version"] = 1
    payload["checkpoint_digest"] = content_digest(
        {name: value for name, value in payload.items() if name != "checkpoint_digest"}
    )
    path = tmp_path / "native-v1.pt"
    torch.save(payload, path)
    restored = NativeRunTrainerV1.load(path)
    assert restored.frozen_groups == ()
    assert restored.steps == 0


def test_fresh_process_restores_pricing_checkpoint(tmp_path):
    trainer = NativeRunTrainerV1.create(adapter_for("new_core_v1"))
    digest = trainer.probe_digest()
    path = tmp_path / "fresh.pt"
    trainer.save(path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.native_content_training import NativeRunTrainerV1; "
        "print(NativeRunTrainerV1.load(sys.argv[1]).probe_digest())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=str(ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip().splitlines()[-1] == digest
