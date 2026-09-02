from __future__ import annotations

from copy import deepcopy

from scripts.training.eval_taiji_m1_identity_organ_canary import run_canary
from scripts.training.train_taiji_memory import _memory_config
from taiji import Taiji, TaijiConfig
from taiji.internalization import content_digest


def _enabled_config(seed: int = 11) -> TaijiConfig:
    values = _memory_config(seed).to_dict()
    values.update(
        {
            "identity_organ_enabled": True,
            "identity_organ_capacity": 8,
        }
    )
    return TaijiConfig.from_dict(values)


def test_identity_organ_defaults_off_without_checkpoint_payload() -> None:
    model = Taiji(_memory_config(11))
    step = model.observe(65, learn=False, learn_motor=False)

    assert model.identity_organ is None
    assert step.identity_recall is None
    assert "identity_organ" not in model.checkpoint()


def test_identity_organ_checkpoint_lineage_and_roundtrip_are_exact() -> None:
    model = Taiji(_enabled_config())
    checkpoint = deepcopy(model.checkpoint())
    restored = Taiji.from_checkpoint(checkpoint)

    assert checkpoint["identity_organ"]["format"] == "taiji-native-identity-organ-v1"
    assert checkpoint["identity_organ"]["lineage"]["organ_id"] == "cue-identity-route"
    assert restored.identity_organ is not None
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)


def test_m1_identity_organ_canary_passes_one_seed() -> None:
    result = run_canary(seeds=(11,))
    assert result["canary_passed"] is True
    record = result["records"]["identity_organ"][0]
    assert record["provenance"]["final_action_owner"] == "ByteMotor"
    assert record["no_change"]["organ_digest_unchanged"] is True
    assert record["checkpoint"]["fresh_process_checkpoint_digest_matches"] is True
