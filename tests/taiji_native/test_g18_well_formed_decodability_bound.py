"""DEBT-G18 修法②：`well_formed ≤ utf8_decodable_rate × texts` 必须是 **G17 后判据**的性质。

为什么值得钉成一条守卫（不是为了绿）：`PLAN-A-30` §2at 曾用"可解码数"当 `well_formed` 的上限，把
a26_p1 的 241 与算出的 12 放进同一张表里比，得出"量级塌十倍"的说法；2026-10-02 全量重算后的真相是——
**这条不等式恰好就是 DEBT-G17 的判别式**：

* G17 之后那把判据（拒 U+FFFD／lone surrogate）跑出的件，`well_formed` 为真 ⇒ 整条文本可解码 ⇒ 不等式**成立**；
* G17 之前那批件里（"UTF-8 合法"那一道在 `str` 上恒真），241／100／288／156／40… 都在只有个位数～十几条
  可解码的情况下报出上百条"成句" ⇒ 不等式**被违反**，而那正是 DEBT-G17 记的症状本身。

所以本文件钉三件事，缺一即红：
1. 名字带 `_g17_` 的表层件逐臂满足不等式（正向面）；
2. 至少 3 条 G17 之前的件违反它（**能为 false** 的那一面——若哪天有人把 ①写成恒真式，②会先红）；
3. 实际核到的臂数有下限（字段被改名或筛成空集合 ⇒ 响亮失败，不是"没筛出东西所以通过"）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"

#: 判据范围＝这两台两臂表层仪器的产出族；不扩到整个 reports/
#: （那里有与本判据无关的件，例如 `reports/_m7_check.json` 根本不是 UTF-8）。
SURFACE_GLOB = "*surface*.json"
FIELDS = ("well_formed_texts", "utf8_decodable_rate", "texts")


def _arms(artifact: Path) -> list[tuple[str, dict]]:
    """取一件里双臂的 (名字, 计数字典)；不是本判据的读者就返回空。"""

    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    arms: list[tuple[str, dict]] = []
    control = payload.get("control_no_circuit")
    if isinstance(control, dict) and all(k in control for k in FIELDS):
        arms.append(("control_no_circuit", control))
    for arm in payload.get("treated_arms") or []:
        if isinstance(arm, dict) and all(k in arm for k in FIELDS):
            arms.append(("treated_with_circuit", arm))
    return arms


def _family(g17_only: bool) -> list[Path]:
    found = [f for f in sorted(REPORTS.glob(SURFACE_GLOB)) if ("_g17_" in f.name) == g17_only]
    if not found:
        pytest.skip(f"no {'g17' if g17_only else 'pre-g17'} surface artifacts on disk")
    return found


def test_g17_era_artifacts_satisfy_the_decodability_bound() -> None:
    """正向面：G17 之后的每一臂，成句数不得超过可解码文本数。"""

    checked = 0
    for artifact in _family(g17_only=True):
        for name, arm in _arms(artifact):
            cap = arm["utf8_decodable_rate"] * arm["texts"]
            assert arm["well_formed_texts"] <= cap + 1e-9, (
                f"{artifact.name}:{name} 成句 {arm['well_formed_texts']} > 可解码上限 {cap:.1f}"
                " ⇒ 产品表层判据又退回恒真那一道（DEBT-G17 复发）"
            )
            checked += 1
    assert checked >= 8, f"只核到 {checked} 条臂 ⇒ 判据面缩到没意义，先查字段是否改名"


def test_pre_g17_artifacts_do_violate_the_bound() -> None:
    """能为 false 的那一面：G17 之前的件必须违反这条不等式，否则上一条是恒真式。"""

    violations = 0
    for artifact in _family(g17_only=False):
        for _name, arm in _arms(artifact):
            if arm["well_formed_texts"] > arm["utf8_decodable_rate"] * arm["texts"] + 1e-9:
                violations += 1
    assert violations >= 3, (
        f"G17 之前的件里只有 {violations} 条违反不等式 ⇒ 要么这批件被重跑过、要么这条守卫的判别力没了"
        "（DEBT-G17 的历史证据不该被抹平）"
    )
