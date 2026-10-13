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
    buckets = CENSUS._classify(cited, texts)
    assert buckets["malformed"] == ["reports/...json"]
    assert buckets["planned_output"] == ["reports/future.json"]
    assert buckets["cited_as_reading"] == ["reports/gone.json"]


def test_a_name_appearing_both_as_reading_and_as_output_stays_a_reading(tmp_path: Path) -> None:
    cited = {"reports/x.json": ["a.md", "b.md"]}
    texts = {
        "a.md": "读数见 `reports/x.json`",
        "b.md": "--out-report reports/x.json",
    }
    buckets = CENSUS._classify(cited, texts)
    #: 只要有任何一处是"当作已有读数"引用，就不许被"将来会落盘"那一桶赦免。
    assert buckets["cited_as_reading"] == ["reports/x.json"]
    assert buckets["planned_output"] == []


def test_citations_are_taken_from_the_named_text_and_not_from_a_reimplemented_scan(
    tmp_path: Path,
) -> None:
    doc = tmp_path / "plans" / "active" / "roadmap"
    doc.mkdir(parents=True)
    (doc / "ledger.md").write_text(
        "见 reports/alpha_2026.json 与 reports/beta.json，还有 reports/skip.md 不算。\n",
        encoding="utf-8",
        newline="\n",
    )
    #: `_citations` 走 PROJECT_ROOT，所以这里把根指到 tmp 再测它取到的件名集合。
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(CENSUS, "PROJECT_ROOT", tmp_path)
        found = CENSUS._citations(["plans/active/roadmap/ledger.md"])
    assert sorted(found) == ["reports/alpha_2026.json", "reports/beta.json"]


def test_an_unreadable_face_is_a_red_not_an_empty_citation_set(tmp_path: Path) -> None:
    with pytest.MonkeyPatch.context() as monkey:
        monkey.setattr(CENSUS, "PROJECT_ROOT", tmp_path)
        monkey.setattr(CENSUS, "_tracked_files", lambda: set())
        monkey.setattr(CENSUS, "_reference_docs", lambda: [])
        payload = CENSUS.build()
    assert payload["verdicts"]["citation_face_scanned"] is False
    assert "citation_face_scanned" in payload["failing_verdicts"]


def test_real_face_names_the_known_never_committed_reading(tmp_path: Path) -> None:
    #: 本行现读的债面（㊵-781）：台账 ㊵-565／567 引用的那枚 N2 乙档读数件从未入库、但仍在盘上。
    #: 这一支钉的是"仪器真的能看见这件事实"；修好之后要改成断言它**不在**名单里（那时它才会红一次）。
    payload = CENSUS.build()
    assert "reports/taiji_n2b_powerup_20261008_n2c1.json" in payload["not_tracked"]
    assert "reports/taiji_n2b_powerup_20261008_n2c1.json" not in payload["missing_on_disk"]
    assert payload["verdict"] == "violated"


def test_refuses_to_overwrite_and_writes_lf(tmp_path: Path) -> None:
    out = tmp_path / "census.json"
    assert CENSUS.main(["--out-report", str(out)]) == 1
    assert out.read_bytes().count(b"\r") == 0
    assert CENSUS.main(["--out-report", str(out)]) == 2
    assert CENSUS.main(["--out-report", str(out), "--overwrite"]) == 1
    payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert payload["format"] == "taiji-ledger-cited-reports-tracked-v1"
