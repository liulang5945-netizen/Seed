"""DEBT-G74③ 的门叶子：预注册语料的**全集**措辞审计棘轮（跑套件时就顺带检，不再只检动过的那份）。

为什么需要这条：㊵-622 第一次把 `plans/reference/PLAN-*.md` 全量跑了一遍，才发现 42 份里
有 8 份一直不过门——**不是因为它们是坏的，而是因为这道门从没被全集跑过**（过去只在
「本轮动过的那份」上跑）。一条只在改动时触发的守卫，等于没有守卫。

形状选择（三处都刻意如此）：
1. **棘轮而非清零**：已知违例冻成白名单，判定用「子集」而不是「相等」⇒
   新增违例当场红，**修掉违例不会红**（不许为了把数字做漂亮去批量改已冻预注册的措辞——
   那属于 G74②，要走升版，且归人逐份判）；
2. ** corpus 规模也钉**：扫到的份数必须等于盘上 `PLAN-*.md` 的份数——
   glob 深度写错会让「零违例」变成假绿（本仓记过：零命中要连范围一起报）；
3. **不起 42 个子进程**：一次调用传多个 `--doc`，快且可比。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_prereg_exit_wording.py"

#: ㊵-625 实测冻结（改后基线：42 份＝34 ok／4 invisible／4 ambiguous）。
#: 这两组是**已知待办清单**，不是"允许长期存在的红"：G74② 逐份修完就该往下缩。
FROZEN_INVISIBLE = {
    "PLAN-B-03_cortex-split_20260925.md",
    "PLAN-N3-01_r4_hooks_prereg_20261007.md",
    "PLAN-N3-03_tau_definition_20261008.md",
    "PLAN-N3-06_ADJUDICATION_20261008.md",
}
FROZEN_AMBIGUOUS = {
    "PLAN-N1-00_s5_endpoint_falsification_prereg_20261007.md",
    "PLAN-N1-01_ADJUDICATION_20261008.md",
    "PLAN-N1-01_s1_readout_prereg_20261007.md",
    "PLAN-N3-08_face_self_report_prereg_20261008.md",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("prereg_wording_corpus_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _corpus() -> list[Path]:
    return sorted((REPO / "plans" / "reference").glob("PLAN-*.md"))


def _audit(tmp_path: Path, docs: list[Path]) -> list[dict]:
    out = tmp_path / "corpus.json"
    argv: list[str] = ["--out-report", str(out)]
    for doc in docs:
        argv += ["--doc", str(doc)]
    rc = _load_module().main(argv)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["rc"] == rc, (payload["rc"], rc)
    return payload["results"]


def test_corpus_is_not_silently_narrowed(tmp_path: Path) -> None:
    #: 扫到的份数＝盘上的份数（防 glob 写错造成"全集=0 份⇒全绿"的假绿）。
    docs = _corpus()
    assert len(docs) >= 42, f"语料规模异常（{len(docs)} 份）⇒ 先证明 glob 还覆盖着全集"
    entries = _audit(tmp_path, docs)
    assert len(entries) == len(docs), (len(entries), len(docs))
    #: 每一份都得真出现（路径串对账），少一份就是取法坏了而不是文档好了。
    names = {Path(e["doc"]).name for e in entries}
    assert names == {d.name for d in docs}, names ^ {d.name for d in docs}


def test_no_new_invisible_docs_appear(tmp_path: Path) -> None:
    entries = _audit(tmp_path, _corpus())
    invisible = {
        Path(e["doc"]).name for e in entries if e["status"] == "invisible_criterion_surface"
    }
    new = invisible - FROZEN_INVISIBLE
    assert (
        not new
    ), f"新增对措辞门隐形的预注册：{sorted(new)}（修法：给判据段加一个含 判据/出口/验收 的二级标题）"


def test_no_new_ambiguous_docs_appear(tmp_path: Path) -> None:
    entries = _audit(tmp_path, _corpus())
    ambiguous = {Path(e["doc"]).name for e in entries if e["status"] == "ambiguous_exit_wording"}
    new = ambiguous - FROZEN_AMBIGUOUS
    assert (
        not new
    ), f"新增未钉数值的出口句：{sorted(new)}（修法：把「明显/大幅/显著/足够」换成数值合取）"
