"""E0 状态隔离实测里"声明 vs 实测"比较器的判据册：两支都要能为假。

`scope_ok` 只从 policy 的 scope 推约束，不认模式名：
①scope=output_only 时出现任何非 output 组、或没有 output 组、或 producer 收到梯度 ⇒ 假；
②scope=none 时出现任何语言组 ⇒ 假；
③producer_updates 与实测梯度有无不一致 ⇒ 假（两个方向都查）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = (
    PROJECT_ROOT / "scripts" / "training" / "probe_taiji_n1_plan177_6_t1_state_isolation.py"
)
GROUPS = {"output", "frame", "encoder"}


def _module():
    spec = importlib.util.spec_from_file_location("n177_6_state_isolation", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def _policy(scope: str, producer: bool) -> dict[str, object]:
    return {"language_parameter_scope": scope, "producer_updates": producer}


def test_declared_groups_follow_the_scope(instrument) -> None:
    assert instrument.declared_groups(_policy("all", True), GROUPS) == GROUPS
    assert instrument.declared_groups(_policy("output_only", True), GROUPS) == {"output"}
    assert instrument.declared_groups(_policy("none", False), GROUPS) == set()
    with pytest.raises(ValueError, match="unknown"):
        instrument.declared_groups(_policy("half", True), GROUPS)


def test_readout_only_accepts_exactly_the_output_group(instrument) -> None:
    policy = _policy("output_only", producer=False)
    assert instrument.scope_ok(policy, {"output": 1.0}, 0.0, GROUPS) is True
    assert instrument.scope_ok(policy, {"output": 1.0, "frame": 1.0}, 0.0, GROUPS) is False
    assert instrument.scope_ok(policy, {"frame": 1.0}, 0.0, GROUPS) is False
    assert instrument.scope_ok(policy, {"output": 1.0}, 0.5, GROUPS) is False


def test_no_update_accepts_no_language_gradient(instrument) -> None:
    policy = _policy("none", producer=False)
    assert instrument.scope_ok(policy, {}, 0.0, GROUPS) is True
    assert instrument.scope_ok(policy, {"output": 1.0}, 0.0, GROUPS) is False


def test_producer_expectation_is_two_sided(instrument) -> None:
    joint = _policy("all", producer=True)
    assert instrument.scope_ok(joint, {"output": 1.0, "frame": 0.5}, 2.0, GROUPS) is True
    assert instrument.scope_ok(joint, {"output": 1.0}, 0.0, GROUPS) is False
    frozen = _policy("all", producer=False)
    assert instrument.scope_ok(frozen, {"output": 1.0}, 0.0, GROUPS) is True
    assert instrument.scope_ok(frozen, {"output": 1.0}, 3.0, GROUPS) is False


def test_a_non_positive_norm_is_not_a_gradient(instrument) -> None:
    policy = _policy("all", producer=True)
    assert instrument.scope_ok(policy, {"output": 0.0}, 1.0, GROUPS) is False
