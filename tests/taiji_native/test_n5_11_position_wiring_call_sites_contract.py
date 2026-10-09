"""PLAN-N5-05 §2 J-N5e-2 与 §3 G-N5e-1／G-N5e-2 在甲落地后的**重钉**（实施格，2026-10-09）。

㊵-635 事先许诺的形状是：接通之后"该抛"那一支要**重钉成新形状，而不是删掉**。本册就按那句话办，
三件事各自钉、各自能为假：

① 旧闸（位置输入 × 发育 bundle 互斥）必须从产品源码里**消失**——两处调用点一起消失；留着它
   就等于甲没做，只做一半则是"消音"（J-N5e-2 明确否证的假修）。
② 新闸（一条没带位置类的重放事件 ⇒ 响亮拒绝）钉住定义点与调用点，并且**两支都走**：
   带位置类 ⇒ 放行且位置列真动；不带 ⇒ 抛错且位置列一动不动。
③ J-N5e-2 的正向证据出两枚机械读数（`position_path_delta > 0` ∧ `n_changed_units >= 1`），
   配一支"稀疏库学习关掉 ⇒ 两枚读数都为零"的反面对照——没有反面，这条判据就是恒真式。

另钉一条边界（**已登记为 DEBT-G76 的缺口，不是已实现的好处**）：甲**不新增张量**，位置列仍住在
读出器 payload 里，bundle 摘要因此不含它 ⇒ 发育层清 `fast_delta` 不撤销位置学习。
夹具用 `test_developmental_synapse.py` 同一份 tiny 配置，不写 `reports/` 与 `output/`。
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.developmental_synapse import DevelopmentalReplayEvent
from taiji.internalization import content_digest
from taiji.utf8_state import UTF8_POSITION_DIM

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "taiji" / "model.py"

#: 甲之前的闸名与闸话（本册第 1 支要求它们在产品源码里彻底消失）。
OLD_HELPER = "_reject_position_input_without_learning_path"
OLD_MESSAGE = "not wired to the developmental F1"
#: 甲之后的闸：重放事件没带位置类 ⇒ 不许按零学。
HELPER = "_reject_position_stateless_replay"
#: 现读：定义点 `taiji/model.py:1123`；唯一调用点在 `replay_developmental_f1` 的事件循环里（:1462）。
#: 钉行号是 ㊵-635 的规矩——数量对而位置漂＝有人把它挪进了别的分支。
EXPECTED_DEF_LINE = 1123
EXPECTED_CALL_SITES = [1462]
#: 钉着"该抛"的两册改动后必须仍钉着某个拒绝点，不许被顺手删成"没抛是默认正确"。
GUARD_FILES = (
    "tests/taiji_native/test_n3_04_developmental_flags_contract.py",
    "tests/taiji_native/test_readout_utf8_position.py",
)
MIGRATE_CALLERS = (
    "taiji/language_alignment.py",
    "scripts/training/check_taiji_m4v2_checkpoint_preflight.py",
    "scripts/training/eval_taiji_m4v2_r2_canary.py",
    "tests/taiji_native/test_developmental_synapse.py",
)

WARMUP = b"abba-caba"
NOVEL = b"caba-abba"


def _config(*, position: bool = False) -> TaijiConfig:
    base = TaijiConfig(
        region_sizes=(24,),
        synapse_fan_in=6,
        motor_fan_in=12,
        memory_units=24,
        memory_fan_in=6,
        memory_meta_dim=16,
        memory_readout_fan_in=12,
        seed=71,
    )
    if not position:
        return base
    return TaijiConfig.from_dict({**base.to_dict(), "readout_utf8_position_input": True})


def _mounted(*, position: bool, mode: str = "fast_slow") -> Taiji:
    """暖机（普通链）⇒ 迁移 ⇒ 写入模式。返回的模型已带一条被普通链训过的位置列。"""

    model = Taiji(_config(position=position), episode_id="n5e-wake")
    model.learn_bytes(WARMUP, epochs=1, learn_fabric=False)
    migration = model.migrate_f1_to_developmental_synapses()
    assert migration["format"] == "taiji-developmental-f1-migration-v1"
    model.set_developmental_f1_learning_mode(mode)
    return model


def _column_copy(model: Taiji) -> torch.Tensor | None:
    column = model.predictive_readout.position_weight
    return None if column is None else column.detach().cpu().clone()


def _position_evidence(model: Taiji, before: torch.Tensor) -> dict[str, float | int]:
    """J-N5e-2 的尺：只量位置列，逐元素取绝对差后求和／计数；不手算、不四舍五入。"""

    column = model.predictive_readout.position_weight
    assert column is not None, "位置列不在场时这把尺没有量程，必须换夹具而不是给零读数"
    moved = (column.detach().cpu() - before).abs()
    return {
        "position_path_delta": float(moved.sum().item()),
        "n_changed_units": int(torch.count_nonzero(moved).item()),
    }


def _event(model: Taiji, *, position_state: int | None) -> DevelopmentalReplayEvent:
    """手造一条**非零误差**的事件：不碰私有缓冲，形状取自配置本身。

    语境两枚给全零 ⇒ `replay_developmental_f1` 的语境支被跳过，本册只问读出侧。
    """

    alphabet = int(model.config.alphabet_size)
    context_dim = int(model.config.motor_context_dim)
    return DevelopmentalReplayEvent(
        event_id="n5e-hand-made",
        source="wake",
        tick=1,
        observed_symbol=alphabet - 1,
        predicted_probability=0.5,
        readout_error=torch.full((alphabet,), 0.25),
        readout_trace=torch.full((context_dim,), 0.5),
        context_feedback=torch.zeros(alphabet),
        context_trace=torch.zeros(context_dim),
        position_state=position_state,
    )


def _payload_keys(node: object) -> set[str]:
    if isinstance(node, Mapping):
        keys = {str(key) for key in node}
        for value in node.values():
            keys |= _payload_keys(value)
        return keys
    if isinstance(node, (list, tuple, set)):
        keys: set[str] = set()
        for value in node:
            keys |= _payload_keys(value)
        return keys
    return set()


# ---------------------------------------------------------------- ① 旧闸消失


def test_the_old_mutual_exclusion_gate_is_gone_from_product_source() -> None:
    """甲的实施就是把这道互斥拒绝换掉：它一旦还在，位置输入 × 发育通路仍是不可得形状。"""

    for rel in ("taiji/model.py", "taiji/organs.py", "taiji/developmental_synapse.py"):
        text = (REPO / rel).read_text(encoding="utf-8")
        assert OLD_HELPER not in text, rel
        assert OLD_MESSAGE not in text, rel


def test_position_input_and_developmental_migration_now_coexist() -> None:
    #: 反面：迁移若又拒绝，本测当场红——这一支就是 DEBT-G71 里"取数形状不可得"的那条。
    model = _mounted(position=True)
    assert model.developmental_f1_bundle is not None
    assert model.config.readout_utf8_position_input is True
    assert model.predictive_readout.position_input_enabled is True


# ---------------------------------------------------------------- ② 新闸的形状


def test_the_new_guard_is_defined_once_at_the_inventoried_site_and_still_raises() -> None:
    text = MODEL.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(MODEL))
    defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == HELPER
    ]
    assert len(defs) == 1, [d.lineno for d in defs]
    assert defs[0].lineno == EXPECTED_DEF_LINE, defs[0].lineno
    body = ast.dump(defs[0])
    assert "position_input_enabled" in body, "新闸不再看位置列是否在场＝它已经不是那道闸"
    assert "ValueError" in body, "新闸不再抛＝清单过期，须重钉而不是让本测静默放行"
    sites = [
        index
        for index, line in enumerate(text.split(chr(10)), start=1)
        if f"self.{HELPER}(" in line
    ]
    assert sites == EXPECTED_CALL_SITES, sites


def test_the_pinned_guards_still_reference_a_position_rejection() -> None:
    """钉着"该抛"的测不许被悄悄删——删了就等于让"没抛"成为默认正确。"""

    for rel in GUARD_FILES:
        path = REPO / rel
        assert path.is_file(), rel
        text = path.read_text(encoding="utf-8")
        assert HELPER in text or "carries no utf-8 position state" in text, rel
        assert "pytest.raises" in text, rel


# ---------------------------------------------------------------- ③ J-N5e-2 正反两支


def test_wake_with_position_on_moves_the_column_by_the_frozen_arithmetic() -> None:
    model = _mounted(position=True)
    before = _column_copy(model)
    assert before is not None

    model.learn_bytes(NOVEL, epochs=1, learn_fabric=False)

    out = _position_evidence(model, before)
    assert out["position_path_delta"] > 0.0, out
    assert out["n_changed_units"] >= 1, out
    assert model.predictive_readout.position_learn_steps > 0
    #: 位置列不进 bundle ⇒ 发育叠加层的 fast 清除不撤销它（DEBT-G76 点名的缺口，此处钉"确实不在"）。
    bundle_payload = model.checkpoint()[Taiji.DEVELOPMENTAL_F1_KEY]
    assert not [key for key in _payload_keys(bundle_payload) if "position" in key.lower()]


def test_wake_without_readout_learning_leaves_the_column_alone() -> None:
    #: 反面对照：稀疏库不学时位置列也不许动，否则 J-N5e-2 的两枚读数就是恒真式。
    model = _mounted(position=True)
    before = _column_copy(model)
    steps_before = model.predictive_readout.position_learn_steps
    assert before is not None

    model.learn_bytes(NOVEL, epochs=1, learn_fabric=False, learn_predictive_readout=False)

    out = _position_evidence(model, before)
    assert out["position_path_delta"] == 0.0, out
    assert out["n_changed_units"] == 0, out
    assert model.predictive_readout.position_learn_steps == steps_before


def test_position_off_keeps_the_overlay_shape_untouched() -> None:
    #: J-N5e-3② 的旧形状：位置关闭＋发育叠加层 ⇒ 没有列可学，bundle 照常在、照写 fast。
    model = _mounted(position=False)
    assert model.predictive_readout.position_input_enabled is False
    assert model.predictive_readout.position_weight is None
    readout_parent = content_digest(model.predictive_readout.to_payload())

    model.learn_bytes(NOVEL, epochs=1, learn_fabric=False)

    bundle = model.developmental_f1_bundle
    assert bundle is not None
    assert bundle.fast_is_zero is False
    assert content_digest(model.predictive_readout.to_payload()) == readout_parent


# ---------------------------------------------------------------- G-N5e-2 两支


def test_replay_of_a_stateless_event_raises_and_changes_nothing() -> None:
    #: 这支有实到入口：`pre-甲` 存档里的 replay 事件按 `from_payload` 取到 `None`。
    model = _mounted(position=True)
    before = _column_copy(model)
    steps_before = model.predictive_readout.position_learn_steps
    assert before is not None
    event = _event(model, position_state=None)

    with pytest.raises(ValueError, match="carries no utf-8 position state"):
        model.replay_developmental_f1([event])

    assert _position_evidence(model, before) == {
        "position_path_delta": 0.0,
        "n_changed_units": 0,
    }
    assert model.predictive_readout.position_learn_steps == steps_before


def test_replay_of_a_stateful_event_learns_and_publishes_both_counters() -> None:
    model = _mounted(position=True)
    before = _column_copy(model)
    assert before is not None
    event = _event(model, position_state=2)

    report = model.replay_developmental_f1([event], consolidate=False)

    assert report["position_learn_steps_after"] - report["position_learn_steps_before"] == 1
    out = _position_evidence(model, before)
    assert out["position_path_delta"] > 0.0, out
    assert out["n_changed_units"] >= 1, out


# ---------------------------------------------------------------- 事件与存档边界


def test_event_payload_adds_the_key_only_when_present_and_seals_it() -> None:
    model = _mounted(position=True)
    absent = _event(model, position_state=None)
    present = _event(model, position_state=3)

    #: 缺位 ⇒ payload 一字不加，旧摘要逐位不变（密封的 `event_digest` 不许被新键顶开）。
    assert "position_state" not in absent.to_payload()
    assert "position_state" in present.to_payload()
    assert DevelopmentalReplayEvent.from_payload(present.to_payload()).position_state == 3

    unsigned = dict(present.to_payload())
    unsigned.pop("event_digest")
    unsigned.pop("position_state")
    tampered = {**unsigned, "event_digest": content_digest(unsigned), "position_state": 1}
    with pytest.raises(ValueError, match="digest"):
        DevelopmentalReplayEvent.from_payload(tampered)

    #: 越界的位置类是外域错误，不是"关闭"的写法。
    with pytest.raises(ValueError, match="position_state must be in"):
        _event(model, position_state=UTF8_POSITION_DIM)


def test_replayed_event_from_a_pre_jia_payload_is_refused_not_assumed_zero() -> None:
    """`pre-甲` 的档里没有 `position_state` 这枚键 ⇒ 取回来是 `None`，重放必须响亮拒绝。

    这里**按旧形状重新封摘要**，证的正是"旧档能载回来"（向后兼容）＋"载回来之后不许按零学"
    （新闸）两件事同时成立。
    """

    model = _mounted(position=True)
    unsigned = dict(_event(model, position_state=2).to_payload())
    unsigned.pop("event_digest")
    unsigned.pop("position_state")
    legacy = {**unsigned, "event_digest": content_digest(unsigned)}
    restored = DevelopmentalReplayEvent.from_payload(legacy)
    assert restored.position_state is None

    with pytest.raises(ValueError, match="carries no utf-8 position state"):
        model.replay_developmental_f1([restored])


def test_every_inventoried_migrate_caller_still_exists() -> None:
    """migrate 的调用方清单：任何一处改名/删除都要先更新本件，而不是让甲的实施跳过它。"""

    for rel in MIGRATE_CALLERS:
        path = REPO / rel
        assert path.is_file(), rel
        assert "migrate_f1_to_developmental_synapses" in path.read_text(encoding="utf-8"), rel
