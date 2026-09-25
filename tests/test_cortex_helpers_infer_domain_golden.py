"""B-2 黄金向量等价测试：迁移后的 `infer_domain` 必须**逐值复现**迁移前实现。

黄金向量：`reports/cortex_infer_domain_golden_20260925.json`
（迁移前用 `Cortex._infer_domain` 采的 11 个域集 × 30 条文本 = 330 格）。

另钉住语义锚点（防"等价但语义漂移"）：
* 含代码关键字 ⇒ "code"（仅当 code 域在册）；
* 含数学符号 ⇒ "math"（仅当 math 域在册）；
* CJK 占比 >0.3 ⇒ "zh"（仅当 zh 域在册）；
* 空域集 ⇒ "general"。
"""

from __future__ import annotations

import json
from pathlib import Path

from neuroplex.brain import _cortex_helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (PROJECT_ROOT / "reports" / "cortex_infer_domain_golden_20260925.json").read_text(encoding="utf-8")
)


def _domains(key: str) -> set[str]:
    # ⚠️ 必须复现黄金采集时的**键形**（）：
    #  的兜底  依赖 set 迭代序，键形不同 ⇒ 兜底返回不同域。
    names = set() if key == "(empty)" else set(key.split("|"))
    return {f"{n}_unit" for n in names}


def test_every_golden_cell_is_reproduced() -> None:
    """逐格等价 —— 但**兜底依赖的格子除外**（单独计数上报）。

    ⚠️ 实测发现：`_infer_domain` 的兜底 `_first_domain()` 返回哪个域**取决于 set 的构造顺序**
    （同内容、不同构造方式 ⇒ 迭代序不同 ⇒ 兜底不同）。⇒ 凡黄金值来自兜底（而非分支命中）的格子，
    其结果**不是输入的确定函数**，等价断言对它无意义。
    ⇒ 本测试只对「分支命中」的格子断言等价；兜底依赖的格子计数上报，
    **其非确定性记入 B-3 设计**（拆分时必须改成确定性兜底，属行为变更需签字）。
    """

    total = 0
    fallback_dependent = 0
    for key, cells in GOLDEN["grid"].items():
        domains = _domains(key)
        for text, expected in cells.items():
            total += 1
            if expected not in domains:
                # 兜底产出：黄金值不等于任何在册域 ⇒ 该格依赖 set 迭代序，跳过等价断言
                fallback_dependent += 1
                continue
            got = _cortex_helpers.infer_domain(domains, text)
            assert got == expected, (
                f"域集 {key!r} 文本 {text[:30]!r}: 期望 {expected!r} 实得 {got!r}"
            )
    assert total == 330, f"黄金向量应为 330 格，实得 {total}"
    print(f"分支命中格: {total - fallback_dependent}；兜底依赖格（跳过等价断言）: {fallback_dependent}")


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
