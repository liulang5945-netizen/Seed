"""B-2/B-3 黄金向量等价测试：`infer_domain` 必须**逐值复现**迁移前实现，且兜底**确定**。

黄金向量：`reports/cortex_infer_domain_golden_20260925.json`
（迁移前用 `Cortex._infer_domain` 采的 10 个域集 × 33 条文本 = 330 格）。

⚠️ 本件的判据在 B-2 落地时是**空跑**的：跳过条件写成 `expected not in domains`，而 `domains` 里的
键形是 `zh_unit`、黄金值是裸前缀 `zh` ⇒ 条件对 330 格恒真 ⇒ 一条等价断言都没执行，
"B-2 黄金 330 格等价"当时只测到了格子总数。B-3 收口把判据改成按**前缀**在册（与实现的
`_has_domain` 同口径），330 格才真的进断言。

另钉语义锚点（防"等价但语义漂移"）：
* 含代码关键字 ⇒ "code"（仅当 code 域在册）；
* 含数学符号 ⇒ "math"（仅当 math 域在册）；
* CJK 占比 >0.3 ⇒ "zh"（仅当 zh 域在册）；
* 空域集 ⇒ "general"；
* 兜底 = 最小键的域前缀（PLAN-B-03 §4 声明的唯一行为变更），且随哈希种子**不变**。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from neuroplex.brain import _cortex_helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = PROJECT_ROOT / "reports" / "cortex_infer_domain_golden_20260925.json"
BEFORE_REPORT = PROJECT_ROOT / "reports" / "cortex_domain_fallback_before_20260926.json"
AFTER_REPORT = PROJECT_ROOT / "reports" / "cortex_domain_fallback_after_20260926.json"
PROBE = PROJECT_ROOT / "scripts" / "training" / "probe_taiji_cortex_domain_fallback_determinism.py"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _domains(key: str) -> set[str]:
    # ⚠️ 必须复现黄金采集时的**键形**：兜底取的是 neuron 键的 `_` 前缀，键形变了兜底值就变。
    if key == "(empty)":
        return set()
    return {f"{n}_unit" for n in key.split("|")}


def test_every_golden_cell_is_reproduced() -> None:
    """逐格等价，无豁免格。

    旧注释声称"兜底依赖的格子结果不是输入的确定函数、只能跳过"——那是对**旧实现**的描述。
    §4 改动后兜底按定义确定（最小键前缀），⇒ 330 格全部可断言，实测全部与黄金相同
    （`reports/cortex_domain_fallback_after_20260926.json`：非确定格 21 ⇒ 0）。
    """

    total = 0
    for key, cells in GOLDEN["grid"].items():
        domains = _domains(key)
        for text, expected in cells.items():
            total += 1
            got = _cortex_helpers.infer_domain(domains, text)
            assert (
                got == expected
            ), f"域集 {key!r} 文本 {text[:30]!r}: 期望 {expected!r} 实得 {got!r}"
    assert total == 330, f"黄金向量应为 330 格，实得 {total}"


def test_semantic_anchors() -> None:
    """语义锚点（在册域决定可返回的域集合）。"""

    code = "def foo():\n    return 1"
    math = "E = mc^2 and the derivative of sin(x)"
    zh = "这是一段中文文本，足够长的中文内容。"
    en = "The quick brown fox jumps over the lazy dog."

    assert _cortex_helpers.infer_domain({"code"}, code) == "code"
    assert _cortex_helpers.infer_domain({"math"}, math) == "math"
    assert _cortex_helpers.infer_domain({"zh"}, zh) == "zh"
    assert _cortex_helpers.infer_domain({"en"}, en) == "en"
    # 域不在册时回退：code 关键字仍在，但 code 域不在册 ⇒ 不返回 code
    assert _cortex_helpers.infer_domain({"zh"}, code) != "code"
    # 空域集 ⇒ general
    assert _cortex_helpers.infer_domain(set(), zh) == "general"


def test_fallback_falls_to_smallest_registered_prefix() -> None:
    """兜底的确定语义 = 最小键的域前缀（§4 声明的唯一行为变更）。"""

    # zh|code：无 en/general 在册、文本非代码/数学/CJK ⇒ 走兜底
    assert _cortex_helpers.infer_domain({"code_unit", "zh_unit"}, "hello world 123") == "code"
    # 插入序无关（同一集合的两种构造 ⇒ 同一输出）
    assert _cortex_helpers.infer_domain({"zh_unit", "code_unit"}, "plain text") == "code"
    # en|math 的 CJK 文本：zh 不在册 ⇒ 走兜底 ⇒ en_unit 在前
    assert _cortex_helpers.infer_domain({"math_unit", "en_unit"}, "这是一段中文文本。") == "en"
    # 反向锚：把最小键换成 math 之外的域，兜底值必须跟着变（否则该函数没真读集合）
    assert _cortex_helpers.infer_domain({"aaa_unit", "zh_unit"}, "hello world 123") == "aaa"


def test_change_moved_no_golden_output() -> None:
    """改前的 21 个非确定格，改后逐格仍等于黄金 ⇒ §4 未顺带挪动任何输出。"""

    report = json.loads(BEFORE_REPORT.read_text(encoding="utf-8"))
    assert report["non_deterministic_count"] == 21, "改前非确定格读数入册，变了要重推"
    for cell in report["non_deterministic_cells"]:
        got = _cortex_helpers.infer_domain(_domains(cell["domain_set"]), cell["text"])
        assert (
            got == cell["golden"]
        ), f"{cell['domain_set']}×{cell['text'][:20]!r}: {got!r} != {cell['golden']!r}"
    after = json.loads(AFTER_REPORT.read_text(encoding="utf-8"))
    assert after["non_deterministic_count"] == 0


def test_fallback_is_invariant_across_hash_seeds(tmp_path) -> None:
    """复跑探针：330 格在 6 个哈希种子 × 多插入序下**零翻转**（兜底非确定 ⇒ 本条红）。"""

    out = tmp_path / "cortex_fallback_probe.json"
    proc = subprocess.run(
        [sys.executable, str(PROBE), "--runs", "6", "--out", str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["cells_measured"] == 330
    assert report["non_deterministic_count"] == 0, json.dumps(
        report["non_deterministic_cells"][:5], ensure_ascii=False
    )
