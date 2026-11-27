"""F0b 四臂的判据册：臂间差异必须落在"特征从哪来、能不能反传"上，别的一律相同。"""

from __future__ import annotations

import torch

from scripts.training.run_taiji_n1_plan177_f_f0b import _Arm, _value_bytes

ITEM = {"value": "红色", "target_byte": int(list("红色".encode())[0])}


def test_credit_on_keeps_the_encoder_in_the_graph() -> None:
    feature = _Arm(arm="main_credit_on", seed=0).feature(
        {**ITEM, "value_bytes": _value_bytes("红色")}
    )
    assert feature.requires_grad is True


def test_credit_stop_detaches_it() -> None:
    feature = _Arm(arm="main_credit_stop", seed=0).feature(
        {**ITEM, "value_bytes": _value_bytes("红色")}
    )
    assert feature.requires_grad is False


def test_copy_only_is_a_one_hot_address_and_not_trainable() -> None:
    feature = _Arm(arm="copy_only", seed=0).feature(ITEM)
    assert feature.requires_grad is False
    assert int(feature.sum()) == 1
    assert float(feature.max()) == 1.0


def test_no_state_is_the_zero_vector() -> None:
    feature = _Arm(arm="no_state", seed=0).feature(ITEM)
    assert float(feature.abs().sum()) == 0.0


def test_only_the_main_arms_put_the_encoder_in_the_optimizer() -> None:
    for arm in ("main_credit_on", "main_credit_stop"):
        params = {
            id(p) for group in _Arm(arm=arm, seed=0).optimizer.param_groups for p in group["params"]
        }
        encoder = {id(p) for p in _Arm(arm=arm, seed=0).encoder.parameters()}
        assert encoder, arm
        assert params, arm
    frozen = _Arm(arm="copy_only", seed=0)
    trained = {id(p) for group in frozen.optimizer.param_groups for p in group["params"]}
    assert trained.isdisjoint({id(p) for p in frozen.encoder.parameters()})


def test_checkpoint_round_trip_restores_optimizer_and_cursor() -> None:
    arm = _Arm(arm="main_credit_on", seed=3)
    arm.cursor = 7
    payload = arm.checkpoint()
    restored = _Arm.from_checkpoint(payload)
    assert restored.cursor == 7
    assert restored.seed == 3
    assert torch.equal(restored.bias.linear.weight, arm.bias.linear.weight)
