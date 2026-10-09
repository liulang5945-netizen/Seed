"""DEBT-G54 修法③ 的契约测：`--min-criterion-lines` 的 rc 分档与多件聚合。

要防的是同一种静默：**"扫到 2 行"与"本该扫到 200 行"以前同码**，
于是"这份件干净"与"这份件几乎没被扫到"在读侧长得一模一样。
四支都必须走：薄面⇒rc=3、关掉分档⇒回到 0、含糊优先于薄面（rc=1 不被 3 洗掉）、
多件时后一件的干净不许把前一件的失败冲掉。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_prereg_exit_wording.py"

PLAN_THIN = "# PLAN-THIN-01_x\n\n" "## 3. 判据\n\n" "* J-T-1 末段均值高于基线 +0.02 即成立。\n"
PLAN_AMBIGUOUS_THIN = (
    "# PLAN-AMB-01_x\n\n"
    "## 3. 判据\n\n"
    "* J-A-1 过 ⇒ 判成立；不过但改善足够 ⇒ 登记不充分。\n"
    "* J-A-2 末段均值高于基线 +0.02 即成立。\n"
)
PLAN_THICK = (
    "# PLAN-THICK-01_x\n\n"
    "## 3. 判据\n\n"
    "* J-1 末段均值高于基线 +0.02 即成立。\n"
    "* J-2 斜率 > 0.03 即成立。\n"
    "* J-3 退化 ≤ +0.02 即通过。\n"
    "* J-4 覆盖率 ≥ 0.9 即通过。\n"
)


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("wording_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["wording_gate"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _run(tmp_path: Path, *docs: Path, floor: int | None = None) -> tuple[int, dict[str, Any]]:
    report = tmp_path / "out.json"
    argv = []
    for doc in docs:
        argv += ["--doc", str(doc)]
    argv += ["--out-report", str(report)]
    if floor is not None:
        argv += ["--min-criterion-lines", str(floor)]
    rc = MODULE.main(argv)
    return rc, json.loads(report.read_text(encoding="utf-8"))


def test_thin_surface_gets_its_own_rc_tier(tmp_path: Path) -> None:
    doc = _write(tmp_path, "PLAN-THIN-01_x.md", PLAN_THIN)
    rc, payload = _run(tmp_path, doc, floor=3)
    entry = payload["results"][0]
    assert rc == 3
    assert entry["status"] == "criterion_surface_too_thin"
    assert entry["criterion_lines"] == 2  # 标题行那句"判据"本身也算一行（门按行内标记挑句）
    assert entry["criterion_lines_floor"] == 3
    assert entry["vague_word_lines_unpinned"] == 0


def test_floor_off_keeps_the_old_answer(tmp_path: Path) -> None:
    #: DEBT-G54③ 之后"关"不再是缺省态：缺省是按件定档（`PLAN-*` 有下限），
    #: 所以这一支必须**显式**给 0 才回到旧答案；不给旗标的那一支由
    #: `test_prereg_wording_default_floor_contract.py` 钉成 rc=3。
    doc = _write(tmp_path, "PLAN-THIN-02_x.md", PLAN_THIN)
    rc, payload = _run(tmp_path, doc, floor=0)
    entry = payload["results"][0]
    assert rc == 0
    assert entry["status"] == "ok"
    assert entry["criterion_lines_floor"] == 0
    assert entry["floor_source"] == "explicit"


def test_ambiguous_wording_is_not_washed_out_by_the_thin_tier(tmp_path: Path) -> None:
    #: 同一份件既薄又有含糊出口 ⇒ 报 rc=1（内容缺陷优先于覆盖面），但薄面事实仍在件里。
    doc = _write(tmp_path, "PLAN-AMB-01_x.md", PLAN_AMBIGUOUS_THIN)
    rc, payload = _run(tmp_path, doc, floor=5)
    entry = payload["results"][0]
    assert rc == 1
    assert entry["status"] == "ambiguous_exit_wording"
    assert entry["vague_word_lines_unpinned"] == 1


def test_clean_doc_above_the_floor_is_zero(tmp_path: Path) -> None:
    doc = _write(tmp_path, "PLAN-THICK-01_x.md", PLAN_THICK)
    rc, payload = _run(tmp_path, doc, floor=3)
    assert rc == 0
    assert payload["results"][0]["criterion_lines"] >= 3


def test_multi_doc_rc_is_the_most_severe_not_the_last(tmp_path: Path) -> None:
    bad = _write(tmp_path, "PLAN-MB-01_x.md", PLAN_THIN)
    good = _write(tmp_path, "PLAN-MB-02_x.md", PLAN_THICK)
    rc, payload = _run(tmp_path, bad, good, floor=3)
    assert rc == 3, payload["results"]
    #: 后一件干净，不许把前一件的失败冲掉；但两件各自的读数都要在件里。
    assert [e["status"] for e in payload["results"]] == [
        "criterion_surface_too_thin",
        "ok",
    ]
