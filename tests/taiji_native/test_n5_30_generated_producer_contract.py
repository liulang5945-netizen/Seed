"""`PLAN-N5-19` 的 producer 契约册（J-N5q-1／2／3／4＋G-N5q-1／2）。

producer 要答的是一句很小的话：**产品里有没有一条把自生成答案写进情节场、并把它标成 `generated` 的路径**。
㊵-734 的预检证明今天没有（全量搜该字面量命中 2 处，都在定义与拒载侧），所以本册不是给既有行为补断言，
而是**给一条新入口立规矩**，并且规矩里有两条是"不许扩"：

①默认不许接线（`J-N5q-3`）——入口存在但没有任何产品调用点，跑一段固定回合后计数必须仍为 0；
②不改外部回执的标签（`G-N5q-1`）——`provenance="generated"` 这个实参在产品码里只能出现 **1 次**，
  把工具执行的回执改标成自我产物是本件明确禁止的方向（与 VISION §19.6 相反）。

两条守卫都带**自造负对照**：喂一份含两处赋值／一处直调 `EpisodicField.write(` 的合成源码，谓词必须能数出来。
计数走运行时属性而**不进 payload**：payload 键受摘要与跨线精确相等约束（三条坑见 DEBT 记录），
`load_payload` 也不许从旧件里"回填"一个不存在的计数。
"""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest
import torch

from taiji import Taiji, TaijiConfig
from taiji.contracts import EpisodicMemoryRecord, Outcome

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENTRY = "record_generated_answer"
CUE = ord("Q")
ACTION = ord("1")
#: 情节写入是在**下一次观察**才落账（`settle_action` 只挂 pending），所以每条测都要走到这里。
OUTCOME = ord("+")
GENERATED_LITERAL = "generated"


def _config() -> TaijiConfig:
    return TaijiConfig(
        region_sizes=(16, 12),
        synapse_fan_in=4,
        motor_fan_in=8,
        memory_units=24,
        memory_fan_in=4,
        memory_meta_dim=8,
        memory_readout_fan_in=8,
        memory_iterations=2,
        seed=3,
        identity_organ_enabled=False,
    )


def _pending_generated_answer(model: Taiji) -> None:
    model.observe(256, learn=False, learn_motor=False)
    model.observe(CUE, learn=False, learn_motor=False)
    model.act((ACTION,), sample=False)


def _tracked_product_sources() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z", "taiji/*.py", "seed/*.py", "api/*.py"],
        capture_output=True,
        cwd=PROJECT_ROOT,
    )
    assert listing.returncode == 0, listing.stderr.decode("utf-8", "replace")[:120]
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    assert len(names) > 100, f"扫描面只有 {len(names)} 个文件，守卫不能静默为空"
    return names


def _generated_keyword_calls(source: str) -> int:
    """数「把 provenance 实参写成 generated」的调用点——比较式与定义都不算。"""

    hits = 0
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "provenance" and getattr(keyword.value, "value", None) in {
                GENERATED_LITERAL
            }:
                hits += 1
    return hits


def _entry_body(source: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == ENTRY:
            return node
    raise AssertionError(f"找不到 {ENTRY} 的定义，扫描面不能静默为空")


def test_recorded_generated_answer_lights_the_generated_row() -> None:
    """J-N5q-1：写一次就要在 `generated` 那一行拿到质量，且读侧那一行成为 argmax。"""

    model = Taiji(_config())
    _pending_generated_answer(model)
    assert model.memory.generated_write_count == 0

    outcome = model.record_generated_answer()
    model.observe(OUTCOME, learn=False, learn_motor=False)
    assert outcome.reward == 0.0
    assert model.memory.generated_write_count == 1
    assert model.memory.write_count == 1

    index = model.memory.PROVENANCE_KINDS.index(GENERATED_LITERAL)
    model.observe(256, learn=False, learn_motor=False)
    step = model.observe(CUE, learn=False, learn_motor=False)
    probabilities = step.memory_recall.provenance_probabilities
    assert float(probabilities[index]) > 0.0, probabilities.tolist()
    # 其余来源从未被写过 ⇒ 掩码给它们恰为 0.0，所以 argmax 只能是 generated 位。
    assert int(probabilities.argmax().item()) == index, probabilities.tolist()


def test_generated_answer_can_not_be_paid_a_reward() -> None:
    """J-N5q-2：入口没有 reward 参数，且合同对象那条非零即抛的支路仍然生效。"""

    assert "reward" not in inspect.signature(Taiji.record_generated_answer).parameters

    with pytest.raises(ValueError, match="non-zero reward"):
        EpisodicMemoryRecord(
            memory_id="m-generated",
            episode_id="episode-generated",
            tick=1,
            cue=torch.tensor([1.0, 0.0]),
            outcome=Outcome(intent_id="i-generated", reward=1.0, tick=1),
            provenance=GENERATED_LITERAL,
        )
    # 负对照的另一半：同一构造函数在零奖励下必须通过，否则上面那条红只是夹具坏了。
    record = EpisodicMemoryRecord(
        memory_id="m-generated-zero",
        episode_id="episode-generated",
        tick=1,
        cue=torch.tensor([1.0, 0.0]),
        outcome=Outcome(intent_id="i-generated", reward=0.0, tick=1),
        provenance=GENERATED_LITERAL,
    )
    assert record.provenance == GENERATED_LITERAL


def test_the_entry_is_not_wired_into_any_default_path() -> None:
    """J-N5q-3：默认中性——产品码里除了定义之外不许有调用点，跑一段固定回合计数仍为 0。"""

    callers = 0
    for name in _tracked_product_sources():
        source = (PROJECT_ROOT / name).read_text(encoding="utf-8", errors="replace")
        if f"{ENTRY}(" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call):
                func = node.func
                if getattr(func, "attr", None) == ENTRY:
                    callers += 1
    assert callers == 0, callers

    model = Taiji(_config())
    _pending_generated_answer(model)
    model.settle_action(1.0, learn_memory=True)
    model.observe(OUTCOME, learn=False, learn_motor=False)
    assert model.memory.generated_write_count == 0
    assert model.memory.write_count == 1


def test_the_counter_is_published_without_touching_the_payload() -> None:
    """J-N5q-4（收窄形）：计数活在运行时属性里；payload 不带它，载入也不回填。"""

    model = Taiji(_config())
    _pending_generated_answer(model)
    model.record_generated_answer()
    model.observe(OUTCOME, learn=False, learn_motor=False)
    payload = model.memory.to_payload()

    assert "generated_write_count" not in payload
    assert model.memory.generated_write_count == 1

    fresh = Taiji(_config())
    assert fresh.memory.generated_write_count == 0
    fresh.memory.load_payload(payload)
    assert fresh.memory.generated_write_count == 0


def test_the_entry_reuses_the_existing_settle_chain() -> None:
    """G-N5q-2：入口体内只许调 `settle_action`，不许出现第四份写入链（直调 `write(`）。"""

    source = Path(inspect.getsourcefile(Taiji) or "").read_text(encoding="utf-8")
    called = {
        node.func.attr
        for node in ast.walk(_entry_body(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "settle_action" in called
    assert "write" not in called

    fabricated = ast.parse(
        "class M:\n"
        "    def record_generated_answer(self):\n"
        "        return self.memory.write(0)\n"
    )
    negative = {
        node.func.attr
        for node in ast.walk(_entry_body(fabricated))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write" in negative and "settle_action" not in negative


def test_the_generated_label_has_exactly_one_write_side_home() -> None:
    """G-N5q-1：`provenance=generated` 作为实参在产品码里只能出现一次（就是这条入口）。"""

    total = 0
    homes = []
    for name in _tracked_product_sources():
        source = (PROJECT_ROOT / name).read_text(encoding="utf-8", errors="replace")
        hits = _generated_keyword_calls(source)
        if hits:
            homes.append(name)
        total += hits
    assert total == 1, homes

    fabricated = (
        "def a():\n"
        '    settle(0.0, provenance="generated")\n'
        "def b():\n"
        '    settle(0.0, provenance="generated")\n'
    )
    assert _generated_keyword_calls(fabricated) == 2
