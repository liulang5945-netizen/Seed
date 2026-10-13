"""`audit_taiji_ledger_cited_reports_are_tracked.py`（DEBT-G134 载体）的契约册。

这台仪器的价值全在**分桶**上：把"还没跑的输出"或"省略形引用"混进"丢了的证据"里，
读数看着像红、其实什么都没测。所以四支合成测逐桶钉，另两支钉真面与拒绝分支。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_ledger_cited_reports_are_tracked.py"


def _load():
    spec = importlib.util.spec_from_file_location("n5_38_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CENSUS = _load()


def test_the_three_kinds_of_absence_are_split_into_three_buckets() -> None:
    cited = {
        "reports/gone.json": ["doc.md"],
        "reports/...json": ["doc.md"],
        "reports/future.json": ["doc.md"],
    }
    texts = {
        "doc.md": (
            "已入库读数见 `reports/gone.json`；省略形写作 reports/...json；"
            "下一步落盘 `--out-report reports/future.json`。"
        )
    }
    buckets = CENSUS._classify(cited, texts, lambda name: False)
    assert buckets["malformed"] == ["reports/...json"]
    assert buckets["planned_output"] == ["reports/future.json"]
    assert buckets["cited_as_reading"] == ["reports/gone.json"]


def test_a_name_appearing_both_as_reading_and_as_output_stays_a_reading(tmp_path: Path) -> None:
    cited = {"reports/x.json": ["a.md", "b.md"]}
    texts = {
        "a.md": "读数见 `reports/x.json`",
        "b.md": "--out-report reports/x.json",
    }
    buckets = CENSUS._classify(cited, texts, lambda name: False)
    #: 只要有任何一处是"当作已有读数"引用，就不许被"将来会落盘"那一桶赦免。
    assert buckets["cited_as_reading"] == ["reports/x.json"]
    assert buckets["planned_output"] == []


def test_the_report_flag_defers_only_a_file_that_is_not_there_yet() -> None:
    """㊵-791 v4 的两个方向都要钉住。

    交接文档会逐字写出**将来**要落的输出件（评测器用 `--report` 指写靶），只认 `--out-report`
    就会把"还没跑的输出"报成丢证据；但同一枚 `--report` 旗标也被用来**读**已有件——
    所以"件已在册"时必须照旧判，否则这条放宽就成了让普查仪闭嘴的后门。
    """

    cited = {"reports/soon.json": ["plan.md"]}
    texts = {"plan.md": "`python eval_x.py --report reports/soon.json` 之后再配对"}
    absent = CENSUS._classify(cited, texts, lambda name: False)
    assert absent["planned_output"] == ["reports/soon.json"]
    assert absent["cited_as_reading"] == []

    present = CENSUS._classify(cited, texts, lambda name: True)
    assert present["cited_as_reading"] == ["reports/soon.json"]
    assert present["planned_output"] == []


def test_citations_are_taken_from_the_named_text_and_not_from_a_reimplemented_scan(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "plans" / "active" / "roadmap"
    doc.mkdir(parents=True)
    (doc / "ledger.md").write_text(
        "见 reports/alpha_2026.json 与 reports/beta.json，"
        "还有 reports/skip.md 与 reports/truth.jsonl 都不算。\n",
        encoding="utf-8",
        newline="\n",
    )
    #: `_citations` 走 PROJECT_ROOT，所以这里把根指到 tmp 再测它取到的件名集合。
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(CENSUS, "PROJECT_ROOT", tmp_path)
        found = CENSUS._citations(["plans/active/roadmap/ledger.md"])
    assert sorted(found) == ["reports/alpha_2026.json", "reports/beta.json"]
    #: 扩展名边界要钉住：把 `.jsonl` 截断成 `.json`，一台「缺件」仪器就会报出根本不存在的缺席——
    #: v1 的真面正是这么错的（4 枚「盘上也没有」其实是 `.jsonl`，其中两枚 ㊵-575 当年逐字节复制进过 `reports/`）。
    assert not any(name.endswith("truth.json") for name in found)


def test_an_unreadable_face_is_a_red_not_an_empty_citation_set(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(CENSUS, "PROJECT_ROOT", tmp_path)
        monkey.setattr(CENSUS, "_tracked_files", lambda: set())
        monkey.setattr(CENSUS, "_reference_docs", lambda: [])
        payload = CENSUS.build()
    assert payload["verdicts"]["citation_face_scanned"] is False
    assert "citation_face_scanned" in payload["failing_verdicts"]


def test_a_verified_rename_moves_the_stale_name_out_of_the_red(tmp_path: Path) -> None:
    """v3（DEBT-G134 最后一格）：带核验的更名登记表把"陈旧名"与"证据丢失"分开。

    登记表本身不是消音器——它自己受 `supersede_map_fully_verified` 与"映射目标必须在索引面里"约束，
    所以下面那一支**不给表**的对照必须仍然红；两支方向相反，这一格不是仪式。
    """

    stale = "reports/taiji_n2_powerup_20261008.json"
    successor = "reports/taiji_n2b_powerup_20261008_n2c1.json"
    payload = CENSUS.build()
    assert payload["superseded_citations"].get(stale) == successor
    assert stale not in payload["not_tracked"]
    assert stale not in payload["missing_on_disk"]
    assert payload["verdicts"]["supersede_map_fully_verified"] is True
    assert payload["verdict"] == "holds"
    #: 反面对照：没有登记表（或表里没有这条）时，同一枚陈旧名必须回到 `not_tracked` 并让整件红。
    without = CENSUS.build("plans/manifests/this-map-does-not-exist.json")
    assert stale in without["not_tracked"]
    assert without["verdict"] == "violated"


def test_a_rename_pointing_at_an_untracked_successor_is_not_verified(tmp_path: Path) -> None:
    mapping, unverified = CENSUS._supersede_map({"reports/real.json"}, path="m.json")
    assert mapping == {} and unverified == []
    written = tmp_path / "m.json"
    body = {
        "entries": [
            {"superseded": "reports/old_a.json", "successor": "reports/real.json"},
            {"superseded": "reports/old_b.json", "successor": "reports/phantom.json"},
        ]
    }
    written.write_text(json.dumps(body), encoding="utf-8", newline="\n")
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(CENSUS, "PROJECT_ROOT", tmp_path)
        mapping, unverified = CENSUS._supersede_map({"reports/real.json"}, path="m.json")
    assert mapping == {"reports/old_a.json": "reports/real.json"}
    assert unverified == ["reports/old_b.json"]


def test_refuses_to_overwrite_and_writes_lf(tmp_path: Path) -> None:
    out = tmp_path / "census.json"
    assert CENSUS.main(["--out-report", str(out)]) == 0
    assert out.read_bytes().count(b"\r") == 0
    assert CENSUS.main(["--out-report", str(out)]) == 2
    assert CENSUS.main(["--out-report", str(out), "--overwrite"]) == 0
    payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "taiji-ledger-cited-reports-tracked-v4"
