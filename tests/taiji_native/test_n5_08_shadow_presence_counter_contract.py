"""㊵-618：N5 影子在场计数器的合同（PLAN-N5-04 G-N5d-4 的实施侧对账）。

为什么这一格要测：㊵-613 查实两臂 `n5_shadow` 块里根本没有在场计数器（读回 `None`＝从没进过块），
而「影子确曾被喂过」一直靠 `candidate_gate`／`candidate_utility` 的**结果侧**读数支撑。

本件钉五件事，其中三条是反向守卫（守卫必须能为 false）：
① 两枚计数器初值为 0；② 自增只住在早退闸**之后**（闸关或被损时不许涨，否则"没通电"会被记成"被喂过"）；
③ 两枚读数在**导入后的类面**上必须仍是 `property`，且只带 `@property` 一枚装饰器——
   这条是上一格的自伤实录：我把属性插在 `@torch.no_grad()` 与 `set_gate` 之间，装饰器被劈走，
   读数退化成 method，直到训练器存盘时 `int(method)` 才 `TypeError`（两臂各崩一次，零权重改动）；
④ 训练器把三枚读数写进 `n5_shadow` 段，填充仍在 `atomic_save` 之前（㊵-593 那次顺序缺陷不许回退）；
⑤ `shadow_materialized` 那类恒真键不得出现。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from taiji.adaptive_residual_shadow import AdaptiveResidualShadow

REPO = Path(__file__).resolve().parents[2]
SHADOW_SRC = REPO / "taiji" / "adaptive_residual_shadow.py"
TRAINER_SRC = REPO / "scripts" / "training" / "train_seed_corpus.py"

FORWARD_BYPASS = "            return context.clone()\n"
LEARN_BYPASS = "        if self._gate == 0.0 or self._lesioned:\n            return\n"
COUNTER_NAMES = ("forward_hits", "learn_hits")


def _shadow_text() -> str:
    return SHADOW_SRC.read_text(encoding="utf-8")


def _trainer_text() -> str:
    return TRAINER_SRC.read_text(encoding="utf-8")


def _class_node() -> ast.ClassDef:
    tree = ast.parse(_shadow_text(), filename=str(SHADOW_SRC))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AdaptiveResidualShadow"
    )


def _decorator_names(func: ast.FunctionDef) -> list[str]:
    out: list[str] = []
    for dec in func.decorator_list:
        if isinstance(dec, ast.Name):
            out.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            out.append(dec.attr)
        else:
            out.append(ast.unparse(dec))
    return out


def test_counters_are_initialised_to_zero_on_the_construction_path() -> None:
    text = _shadow_text()
    assert text.count("self._forward_hits = 0") == 1
    assert text.count("self._learn_hits = 0") == 1


def test_increment_sites_live_only_on_the_powered_branch() -> None:
    """自增必须在早退之后；早退块**内部**出现自增就是把「没通电」记成「被喂过」。"""
    text = _shadow_text()
    assert text.count("if self._gate == 0.0 or self._lesioned:") == 2
    assert (FORWARD_BYPASS + "        self._forward_hits += 1\n") in text
    assert (LEARN_BYPASS + "        self._learn_hits += 1\n") in text
    assert "self._lesioned:\n            self._forward_hits" not in text
    assert "self._lesioned:\n            self._learn_hits" not in text


def test_both_readings_are_properties_on_the_imported_class() -> None:
    for name in COUNTER_NAMES:
        found = inspect.getattr_static(AdaptiveResidualShadow, name)
        assert isinstance(found, property), f"{name} 不是 property（实为 {type(found).__name__}）"


def test_the_two_properties_carry_only_the_property_decorator() -> None:
    """本格的自伤形状：属性被插到别人的装饰器与自己之间 ⇒ 装饰器挪走 ⇒ 读数变 method。"""
    seen: dict[str, list[str]] = {}
    for func in _class_node().body:
        if isinstance(func, ast.FunctionDef) and func.name in COUNTER_NAMES:
            seen[func.name] = _decorator_names(func)
    assert set(seen) == set(COUNTER_NAMES), seen
    for name, decorators in seen.items():
        assert decorators == ["property"], (name, decorators)


def test_set_gate_keeps_its_torch_no_grad_decorator() -> None:
    """反向：我搬动属性时不得把 `set_gate` 原有的 `@torch.no_grad()` 一起带走。

    `@torch.no_grad()` 在 AST 里是 **Call** 节点（`torch.no_grad()`），
    所以取到的是整串 `torch.no_grad()` 而不是属性名 `no_grad`——期望值按真实节点形状写。
    """
    set_gate = next(
        func
        for func in _class_node().body
        if isinstance(func, ast.FunctionDef) and func.name == "set_gate"
    )
    assert _decorator_names(set_gate) == ["torch.no_grad()"], _decorator_names(set_gate)


def test_trainer_publishes_three_presence_readings_before_the_atomic_save() -> None:
    text = _trainer_text()
    for key in ('"shadow_forward_hits"', '"shadow_learn_hits"', '"shadow_branch_hits"'):
        assert text.count(key) == 1, key
    fill = text.index('envelope["n5_shadow"] = {')
    save = text.index("atomic_save(envelope, checkpoint_path)")
    assert fill < save, "填充必须在落盘之前（㊵-593 的顺序缺陷不许回退）"
    for key in ('"shadow_forward_hits"', '"shadow_learn_hits"', '"shadow_branch_hits"'):
        assert fill < text.index(key) < save


def test_no_tautological_materialised_flag_is_published() -> None:
    """`shadow_materialized` 在这个形状下只能是真 ⇒ 不得成为证据（守卫自己也要能为 false）。"""
    assert '"shadow_materialized"' not in _trainer_text()
