"""DEBT-G46 修法③ 的契约测：预注册出口句的含糊用词扫描必须**两面都能判**。

背景（缺陷本体，不是假想）：PLAN-N1-01 的第三出口写"不过但 F1/F3 **大幅**改善 ⇒ 登记
'读取可修、但不充分'"，而"大幅"没有冻结数值 ⇒ 实测落进含糊区时两个出口都说得通，判据本身
无法裁决（㊵-485⑤a）；㊵-487⑦ 又抓到第二类实例（"恰好跌破 1 项"无定义）。

这里刻意做成两支都能红：
* **反支**＝把已知含糊的两件钉成"必须命中、且命中行号与条数不变"——扫描器一旦被阉掉就红；
* **正支**＝把当前干净的预注册钉成"必须零命中"——有人往判据句里塞口语词就红。

两条如实的边界（写成测，防止把这台仪器读成"判据不成立裁判"）：
1. rc=1 的语义是"判据句里有未钉数值的含糊词，要人看一眼"，**不是**判据判负；
   PLAN-N1-00 那两处命中其实是原文在**否定**这种说法（用 大幅 来声明它不构成判据），形状上是假阳性；
   冻结件不追改（DEBT-G46 原文口径），所以它们被钉成"必须仍然命中"而不是"必须已被修掉"。
2. 标记是按行匹配的：**任何**提到"判据/出口/J-"的行都算判据句，包括散文里转述这两个词的行。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/audit_taiji_prereg_exit_wording.py"
PREREG_DIR = REPO / "plans/reference"

#: 冻结件里已知的含糊行——行号与条数都是钉值；只有追改判据文本才会红，而本仓不追改。
#: **〔2026-10-09 ㊵-629 重钉〕** `PLAN-N1-00` 的第 24 行**不再命中**：那一行原文是
#: `J1（唯一判据）：R1 ≤ **118**（基线 236 的 50%）`，同行已把数值钉死，
#: 旧 `NUMERIC_PIN` 因粗体标记卡在算符与数字之间认不出而误报——红的是门的取法瞎，不是文档没钉。
#: 保留的第 7 行是假设陈述里的「显著」，同行确无数值 ⇒ 那是真含糊，不随本次修门而消失。
FLAGGED = {
    "PLAN-N1-00_s5_endpoint_falsification_prereg_20261007.md": [7],
    "PLAN-N1-01_s1_readout_prereg_20261007.md": [35],
}

#: 当前零命中的冻结件——往判据句里新增任何未钉数值的含糊词都会把这支打红。
CLEAN = (
    "PLAN-N1-02_attribution_prereg_20261007.md",
    "PLAN-N2-01_consolidation_powerup_prereg_20261007.md",
    "PLAN-N3-01_r4_hooks_prereg_20261007.md",
    "PLAN-N3-02_scaling_probe_prereg_20261007.md",
    "PLAN-N3-03_tau_definition_20261008.md",
    "PLAN-N3-04_developmental_assembly_prereg_20261008.md",
    "PLAN-N2-02_second_powerup_dosewindow_prereg_20261008.md",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_prereg_exit_wording", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_ambiguous_preregs_are_still_flagged_at_pinned_lines() -> None:
    """反支：扫描器必须还能抓到已知那两处；行号与条数都是钉值（阉掉扫描器就红）。"""

    module = _load_module()
    for name, expected_lines in FLAGGED.items():
        scan = module.scan_text((PREREG_DIR / name).read_text(encoding="utf-8"))
        assert scan["criterion_lines"] > 0, name
        assert [hit["line"] for hit in scan["hits"]] == expected_lines, name
        assert scan["vague_word_lines_unpinned"] == len(expected_lines), name


@pytest.mark.parametrize("name", CLEAN)
def test_clean_frozen_preregs_stay_unflagged(name: str) -> None:
    """正支：这五件现在零命中；将来谁把口语词写进判据句，这里当场红。"""

    module = _load_module()
    scan = module.scan_text((PREREG_DIR / name).read_text(encoding="utf-8"))
    assert scan["criterion_lines"] > 0, "没有判据句时零命中不代表干净"
    assert scan["hits"] == [], name


def test_numeric_pin_exempts_the_word_on_that_line_only() -> None:
    """数值豁免这条支路必须在起作用，且**只按行**豁免——所以 rc=1 是"看一眼"不是"判负"。

    同行只要有任一数值钉住，那行的含糊词就被豁免（本例里 `≤118` 就足以豁免 大幅）。
    这是刻意的粗粒度：判据句要求"这一句被人钉过数"，不要求每个子句各自带数。
    """

    module = _load_module()
    pinned = "- 出口二：主判据 ≤118 不成立，但 F1 大幅改善 ⇒ 走定位叙述。\n"
    assert module.scan_text(pinned)["vague_word_lines_unpinned"] == 0

    unpinned = module.scan_text("- 出口二：不过但 F1 高于基线，且改善大幅 ⇒ 走定位叙述。\n")
    assert unpinned["criterion_lines"] == 1
    assert unpinned["vague_word_lines_unpinned"] == 1
    assert unpinned["hits"][0]["words"] == ["大幅"]


def test_unpinned_vague_word_in_criterion_line_hits() -> None:
    module = _load_module()
    scan = module.scan_text("- 出口三：不过但 F1/F3 大幅改善 ⇒ 登记'可修但不充分'。\n")
    assert scan["criterion_lines"] == 1
    assert scan["vague_word_lines_unpinned"] == 1
    assert scan["hits"][0]["words"] == ["大幅"]


def test_doc_without_any_criterion_line_is_a_loud_refusal(tmp_path: Path, capsys) -> None:
    """没有可机检判据的预注册＝拒判（rc=2），不能读成"零命中所以干净"。"""

    module = _load_module()
    doc = tmp_path / "PLAN-DEMO-01_without_criteria_prereg.md"
    doc.write_text("这篇只写流程与背景，一句可机检的条件都没有。\n", encoding="utf-8", newline="\n")

    assert module.main(["--doc", str(doc)]) == 2
    capsys.readouterr()


def test_missing_doc_is_a_loud_refusal(tmp_path: Path, capsys) -> None:
    """缺件不是零命中：点名缺失并把 rc 定成响亮拒绝（rc=2）。"""

    module = _load_module()
    assert module.main(["--doc", str(tmp_path / "nope_prereg.md")]) == 2
    capsys.readouterr()
