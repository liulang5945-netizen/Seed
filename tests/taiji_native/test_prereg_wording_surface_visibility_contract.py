"""DEBT-G54 修法② 的契约测：判据段的**存在性**由标题机检，一份 `PLAN-*` 不能靠改叫法对门隐形。

撞出这条债的过程本身就是证据（㊵-506/508/513）：新写的定义件通篇用"规则/恒等式/处置"当标题，
措辞门第一次跑它是 **rc=2 `no_criterion_section`**——它知道自己扫不到东西是对的，但旧实现里
"扫不到"与"干净"只隔一个 rc，而修它的我是**改标题**让件进入扫描面，没有改扫描器。

三条测各自都能为假；夹具里两条**故意**踩我自己踩过的两个坑：
文档大标题里出现"判据"两个字**不算**一节判据段（只认二级及以下标题）；
非 `PLAN-*` 的台账/债务登记**不受这条新守卫约束**（否则历史文档要被集体判红）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/training/audit_taiji_prereg_exit_wording.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prereg_wording_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GATE = _load_module()

#: 通篇不提"判据/出口/验收"的预注册——旧实现只看有没有标记词，会被标题里那一句糊过去。
BLIND = """# PLAN-BLIND-01 · 一份不肯老实点名的预注册

判据这一节我偏不写成标题。

## 1. 口径与约定

1. 达标线取 p90 向上取整到 0.05 格。
2. 不满足则登记"不可冻"。
"""

SEEN = """# PLAN-SEEN-01 · 一份正常点名的预注册

## 2. 判据（先冻）

1. J-1：`tau >= 0.65` 才算达成。

## 5. 出口

* 达成 ⇒ 出版候选值；不达成 ⇒ 出版上界。
"""

#: 有一节正经判据标题、但**判据句本身**用词没钉数值 ⇒ 走的是 rc=1 那条老通道。
VAGUE = """# PLAN-VAGUE-01 · 命中通道的对照件

## 2. 判据（先冻）

1. J-2：若各项均不成立，则改走"大幅改善"那一支。
"""

LEDGER = """# 台账一枚（不是预注册）

【㊵-1 这条记录写着"判据"两个字，但它没有判据节。】
"""


def _run(tmp_path: Path, name: str, text: str) -> tuple[int, dict]:
    doc = tmp_path / name
    doc.write_text(text, encoding="utf-8")
    out = tmp_path / "report.json"
    rc = GATE.main(["--doc", str(doc), "--out-report", str(out)])
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_plan_doc_without_criterion_heading_is_loudly_invisible(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, "PLAN-BLIND-01_x.md", BLIND)
    entry = payload["results"][0]
    assert rc == 2, entry
    assert entry["status"] == "invisible_criterion_surface"
    #: 大标题里明明写着"判据"，但那只算一行普通文字——存在性检查看的是二级标题。
    assert entry["criterion_headings"] == 0
    assert entry["criterion_lines"] >= 1  # 旧通道本来会被那句散文骗过去


def test_plan_doc_with_criterion_heading_scans_normally(tmp_path: Path) -> None:
    rc, payload = _run(tmp_path, "PLAN-SEEN-01_x.md", SEEN)
    entry = payload["results"][0]
    assert rc == 0, entry
    assert entry["criterion_headings"] == 2
    assert entry["status"] == "ok"
    assert entry["vague_word_lines_unpinned"] == 0


def test_ambiguous_wording_still_hits_the_rc1_channel(tmp_path: Path) -> None:
    #: 新守卫不许把老通道顶掉：有判据节、但用词没钉数值 ⇒ 仍须 rc=1。
    rc, payload = _run(tmp_path, "PLAN-VAGUE-01_x.md", VAGUE)
    entry = payload["results"][0]
    assert rc == 1, entry
    assert entry["status"] == "ambiguous_exit_wording"
    assert entry["criterion_headings"] == 1
    assert entry["vague_word_lines_unpinned"] == 1


def test_non_plan_docs_are_exempt_from_the_heading_rule(tmp_path: Path) -> None:
    #: 台账/债务登记不是预注册：新守卫不加到它们头上（但旧通道照旧，这里正文含"判据"故为 ok）。
    rc, payload = _run(tmp_path, "08_UPSTREAM_SYNC_PLAYBOOK_x.md", LEDGER)
    entry = payload["results"][0]
    assert rc == 0, entry
    assert "criterion_headings" not in entry
    assert entry["status"] == "ok"
