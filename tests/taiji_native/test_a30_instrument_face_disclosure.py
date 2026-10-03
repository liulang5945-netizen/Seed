"""A30 仪器必须自述"这张面到底是哪张"，且证据门要报**有效值**而不是 `config` 那一位。

来历（2026-09-30，我自己差点把它写错并入库）：`probe_taiji_a30_stop_failure.py` 的 `--circuit` 不给
**不等于**"出厂无回路面"——带回路的信封（如 `checkpoints/seed_beta_with_circuit.pt`）在
`SeedRuntime.load` 里就自动挂载（冒烟读数 `mount_route=envelope_auto_mount`）。
同理证据门的**有效值 ≠ `config` 那一位**：`taiji/config.py` 默认 `False`，
而 restore 的自动挂载分支会 `set_copy_evidence_utf8_gate(True)`（owner 裁定 (b)，`taiji/model.py:3497`）。
我第一版只报 `config`，读起来就是"装机面门关着"——**列名与语义都该被守住**，所以有本文件。

本守卫是**源码级**的（不跑生成，秒级）：只要求仪器把这三列写进件里，且**禁止**再用那个只报 config 的旧键名。
否证演示（改前码＝`72d81faf^`，同一套谓词跑它）：三列全缺 ⇒ 谓词返回 False，见本模块
`test_the_guard_rejects_the_pre_v4_instrument`。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = "scripts/training/probe_taiji_a30_stop_failure.py"

#: 件里必须存在的**自述列**（缺一即"这张面靠猜"）。
REQUIRED_FACE_COLUMNS = (
    '"mount_route"',
    '"copy_circuit_present_after_load"',
    '"copy_evidence_utf8_gate_effective"',
)
#: 有效值在 v39 之后的唯一合法来源：产品的公开状态出口（`DEBT-G30` 第一步建的）。
PUBLIC_ACCESSOR = "copy_evidence_utf8_gate_state()"
#: 有效值必须这样算：有 override 用 override，否则才落到 config。
EFFECTIVE_EXPRESSION = "_copy_evidence_utf8_gate_override"
#: 旧键名＝只报 config，会被读成"门关着"。
FORBIDDEN_LEGACY_KEY = '"copy_evidence_utf8_gate":'


def _source_from_git(rev: str = "HEAD") -> str:
    """按**版本控制面**取码（工作树可能被别的会话改动，守的是已入库那份）。"""
    return subprocess.run(
        ["git", "show", f"{rev}:{INSTRUMENT}"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")


def _face_columns_present(source: str) -> bool:
    return all(column in source for column in REQUIRED_FACE_COLUMNS)


def test_instrument_reports_which_assembly_it_ran():
    source = _source_from_git()
    missing = [c for c in REQUIRED_FACE_COLUMNS if c not in source]
    assert missing == [], f"仪器不再自述取数面，读数又得靠命令行猜：缺 {missing}"


def test_effective_gate_is_reported_from_the_public_accessor_not_config_only():
    """v39 起这条守卫**换了形状**（原来钉的是"探针里有 override 推导式"，v39 把它删掉了）。

    现在要求的是同一件事的**新写法**：三列必须都来自产品公开出口 `copy_evidence_utf8_gate_state()`，
    而不是这台仪器自己复制 `config-or-override` 那条式子。两向都能为假：
    ①回退成只报 config ⇒ 三个键里至少一个消失，第一条循环断言红；
    ②重新引入私有字段复制推导 ⇒ 第二条断言红（v38 及以前正是那种形状，`DEBT-G30` 第二步已清除）；
    ③又用只报 config 的旧键名 ⇒ 最后一条红（装机面会被读成"证据门关闭"）。
    """

    source = _source_from_git()
    assert PUBLIC_ACCESSOR in source, "有效值不再取自产品公开出口 ⇒ 这台仪器又开始自己复制那条式子"
    assert (
        EFFECTIVE_EXPRESSION not in source
    ), "私有字段 `_copy_evidence_utf8_gate_override` 又回到这台仪器里 ⇒ 同一判定出现两份实现（DEBT-G30 的成因）"
    for column in (
        '"copy_evidence_utf8_gate_effective"',
        '"copy_evidence_utf8_gate_config"',
        '"copy_evidence_utf8_gate_override"',
    ):
        assert column in source, f"三列自述少了 {column}"
    assert (
        FORBIDDEN_LEGACY_KEY not in source
    ), "又出现只报 config 的旧键名（装机面会被读成证据门关闭）"


def test_scan_face_is_the_tracked_file():
    """零命中的另一种成因是"扫错了面"：这里把范围本身也报出来。"""
    tracked = (
        subprocess.run(
            ["git", "ls-files", INSTRUMENT],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
        )
        .stdout.decode("utf-8")
        .strip()
    )
    assert (
        tracked == INSTRUMENT
    ), f"仪器不在版本控制面上（扫到 {tracked!r}），本守卫实际没在守任何东西"
    assert len(_source_from_git().splitlines()) > 100


def test_the_guard_rejects_the_pre_v4_instrument():
    """同一套谓词跑**改前的码**（`72d81faf^`）必须为 False——否则本守卫恒真、不能算守。"""
    pre_fix = _source_from_git(rev="72d81faf^")
    assert not _face_columns_present(pre_fix), "改前码竟已带这三列 ⇒ 谓词分不开两版，守卫无判别力"
    assert EFFECTIVE_EXPRESSION not in pre_fix or FORBIDDEN_LEGACY_KEY in pre_fix
