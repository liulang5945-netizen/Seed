"""DEBT-G54 修法③ 的"默认开"契约：判据面下限从 opt-in 变成按件定的缺省档。

旧行为（㊵-578 之后、本格之前）：`--min-criterion-lines` 默认 0 ⇒ 只有调用者主动给数才分档，
而实际调用几乎从不给 ⇒ "本该扫到 200 行却只扫到 2 行"这一档**结构上存在但默认不生效**。
现改成按件定档：`PLAN-*` 缺省下限 `PLAN_CRITERION_FLOOR`（实测值，见脚本注释），
其余件不设限；显式给整数＝对全部件用同一个下限（给 0＝关掉分档）。

五支都能为假，且覆盖两侧：

* 3 行的 PLAN 件（不给旗标）⇒ rc=3／`criterion_surface_too_thin`／`floor_source=plan_default`；
* **恰好 5 行**的 PLAN 件 ⇒ rc=0（边界：证明下限不是恒真也不是差一）；
* 非 PLAN 件 3 行 ⇒ rc=0／`floor=0`／`floor_source=none`（历史台账类不该被判据件的标准管）；
* 显式 `--min-criterion-lines 0` ⇒ rc=0（"关"现在必须是一个显式动作）；
* 显式 `--min-criterion-lines 9` 打在满足缺省档的件上 ⇒ rc=3／`floor_source=explicit`。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_prereg_exit_wording.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("wording_gate_floor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load()
FLOOR = int(GATE.PLAN_CRITERION_FLOOR)


#: 标题行本身含"判据"二字也会被旧通道计成一行判据句 ⇒ 一份 `items` 条目的夹具
#: 实际出版 `items + 1` 行（这一条由下面的 `criterion_lines` 断言钉住，不是猜的）。
def _doc(items: int) -> str:
    lines = ["# 下限契约夹具", "", "## 判据", ""]
    lines += [f"{n}. J-{n}：这一行的判据句只描述处置，不含含糊量词。" for n in range(1, items + 1)]
    return "\n".join(lines) + "\n"


def _run(tmp_path: Path, name: str, items: int, *extra: str) -> dict[str, Any]:
    doc = tmp_path / name
    doc.write_text(_doc(items), encoding="utf-8", newline="\n")
    report = tmp_path / "out.json"
    argv = ["--doc", str(doc), "--out-report", str(report), *extra]
    rc = GATE.main(argv)
    payload = json.loads(report.read_text(encoding="utf-8"))
    entry = dict(payload["results"][0])
    entry["_rc"] = rc
    return entry


def test_plan_doc_below_the_default_floor_is_tiered(tmp_path: Path) -> None:
    entry = _run(tmp_path, "PLAN-DF-THIN_x.md", 2)
    assert entry["_rc"] == 3, entry
    assert entry["status"] == "criterion_surface_too_thin"
    assert entry["criterion_lines"] == 3  # 2 条目＋1 行标题
    assert entry["criterion_lines_floor"] == FLOOR
    assert entry["floor_source"] == "plan_default"


def test_doc_exactly_at_the_floor_is_clean(tmp_path: Path) -> None:
    #: 边界探针：少一行必须红（上一条），正好等于下限必须绿。
    entry = _run(tmp_path, "PLAN-DF-EDGE_x.md", FLOOR - 1)
    assert entry["_rc"] == 0, entry
    assert entry["status"] == "ok"
    assert entry["criterion_lines"] == FLOOR  # 正好等于下限
    assert entry["criterion_lines_floor"] == FLOOR


def test_non_plan_doc_is_not_held_to_the_prereg_floor(tmp_path: Path) -> None:
    entry = _run(tmp_path, "REGISTER_NOTE_x.md", 2)
    assert entry["_rc"] == 0, entry
    assert entry["criterion_lines_floor"] == 0
    assert entry["floor_source"] == "none"


def test_turning_the_tier_off_is_now_an_explicit_act(tmp_path: Path) -> None:
    entry = _run(tmp_path, "PLAN-DF-OFF_x.md", 2, "--min-criterion-lines", "0")
    assert entry["_rc"] == 0, entry
    assert entry["criterion_lines_floor"] == 0
    assert entry["floor_source"] == "explicit"


def test_explicit_floor_overrides_the_per_kind_default(tmp_path: Path) -> None:
    #: 同一份满足缺省档的 PLAN 件，被显式抬到 9 之后必须转红——证明显式档真的接管了。
    entry = _run(tmp_path, "PLAN-DF-RAISE_x.md", FLOOR - 1, "--min-criterion-lines", "9")
    assert entry["_rc"] == 3, entry
    assert entry["criterion_lines_floor"] == 9
    assert entry["floor_source"] == "explicit"
