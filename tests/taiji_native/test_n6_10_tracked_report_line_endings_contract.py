"""DEBT-G115 修法② 的契约册：入库 `reports/` 件的行尾必须有一道**能为假**的门。

本册不钉「所有件都是 LF」（那句话 ㊵-732 已经被证伪过一次，而且今天仍不成立），它钉四件事：
①取法只认**索引字节**，工作树那一面只出版不判（两面在同一批件上现读是 2 对 138，混用会得出相反答案）；
②三桶并列：`\r\n`／散 `\r`／`\r\r\n` 各数各的，一把尺不许把另一把摊平；
③基线只许减：豁免集之外的新件必须红，而**修好一枚不许变成红**（这条最容易被写反）；
④面为空必须响亮失败——零命中通常意味着 `--face` 指错了目录，不是「已规范化」。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_tracked_report_line_endings.py"


def _module():
    spec = importlib.util.spec_from_file_location("tracked_line_endings", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def _scan(module, contents: dict[str, bytes]):
    """用注入的读法扫一张假面：真面与假面走的是同一段判定代码，不是另写一条链。"""

    return module.scan_face(
        sorted(contents),
        lambda name: contents[name],
        lambda name: contents[name],
        "reports/",
    )


def test_a_crlf_pair_is_not_counted_as_a_lone_cr(instrument) -> None:
    """负对照第一支：`\r\n` 是一枚配对，散 `\r` 才是另一回事。"""

    assert instrument.classify_endings(b"a\r\nb\n") == {
        "crlf_pairs": 1,
        "lone_cr": 0,
        "cr_lf_pairs_after_cr": 0,
        "bytes": 5,
    }


def test_the_three_buckets_stay_separate(instrument) -> None:
    """纯 LF／散 CR／混档 `\r\r\n` 三形分开数——混档那形正是"按字节追加"最容易造出来的。"""

    assert instrument.classify_endings(b"a\nb\nc\n")["lone_cr"] == 0
    assert instrument.classify_endings(b"a\nb\nc\n")["crlf_pairs"] == 0
    assert instrument.classify_endings(b"a\rb\rc") == {
        "crlf_pairs": 0,
        "lone_cr": 2,
        "cr_lf_pairs_after_cr": 0,
        "bytes": 5,
    }
    mixed = instrument.classify_endings(b"a\r\r\n")
    assert (mixed["crlf_pairs"], mixed["lone_cr"], mixed["cr_lf_pairs_after_cr"]) == (
        1,
        1,
        1,
    ), mixed


def test_a_new_cr_bearing_file_fires_the_gate(instrument) -> None:
    """③的正形状：豁免集之外新写一枚带 CR 的件 ⇒ rc=1 并把名字点名。"""

    contents = {"reports/brand_new.json": b'{\n  "cr": 1\r\n}\n'}
    report = _scan(instrument, contents)
    assert report["verdict"] == "not_holds", report
    assert "no_new_cr_bearing_tracked_file" in report["failing_verdicts"]
    assert report["cr_bearing_names"] == ["reports/brand_new.json"]


def test_fixing_one_grandfathered_file_does_not_fire_the_gate(instrument) -> None:
    """③的反形状（最容易写反的一支）：基线**缩小**是好事，不能判成红。"""

    contents = {"reports/_m7_check.json": b"a\r\n"}
    report = _scan(instrument, contents)
    assert report["verdicts"]["no_new_cr_bearing_tracked_file"] is True
    assert report["verdict"] == "holds", report
    #: 「恰好还是那两枚」只作现状陈述，缩小之后它可以为假，但不许拖累判定。
    assert report["equals_current_baseline"] is False


def test_an_empty_face_fails_loudly(instrument) -> None:
    """零命中不是"已规范化"：面为空必须抛，而不是出版一个漂亮的 0。"""

    with pytest.raises(ValueError, match="reports/"):
        _scan(instrument, {})


def test_the_two_faces_disagree_on_the_real_repo(instrument, tmp_path) -> None:
    """真面：索引面两枚、工作树面 138 枚 ⇒ 判据必须只认前者。

    枚数不钉死（我下一批加一枚读数件就会推翻它），这里只断言"两面不相等"这件事本身——
    它才是"哪张面"必须写进件里的理由。
    """

    out = tmp_path / "lf.json"
    assert instrument.main(["--out-report", str(out)]) == 0
    report: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "holds", report["failing_verdicts"]
    assert report["cr_bearing_names"] == sorted(instrument.GRANDFATHERED)
    assert report["tracked_total"] == len(
        instrument.tracked_paths(PROJECT_ROOT, instrument.DEFAULT_FACE)
    )
    assert report["working_tree_cr_files_total"] != len(report["cr_bearing_names"])
    #: 不钉死枚数（下一批加一枚读数件就会推翻它）；钉的是"桶里的总数必须等于逐枚求和"与
    #: "索引面上没有一枚是 `\r\n`"——后者才是"两面必须分开"的实证。
    buckets = report["buckets_index_face"]
    #: 〔㊵-1106 取法修口〕原句 `crlf_pairs == 0` 把"那两枚豁免件都只是散 CR"这一**现场观察**
    #: 当成了判据；本轮新入册的封存件（`taiji_n1_fresh_paired_train_20261016.json`，SHA 被 45 枚
    #: 件引用、不可改写）里确有 1 对 `\r\n` ⇒ 换成同一条意图的版本化形式：
    #: **索引面上带 CRLF 对的那些件必须是豁免集的子集**（新来一枚带 CRLF 的照样当场红）。
    crlf_names = sorted(
        row["path"] for row in report["cr_bearing_tracked_files"] if row["crlf_pairs"]
    )
    assert set(crlf_names) <= set(instrument.GRANDFATHERED), crlf_names
    assert buckets["crlf_pairs"] == sum(
        row["crlf_pairs"] for row in report["cr_bearing_tracked_files"]
    )
    assert buckets["lone_cr"] == sum(row["lone_cr"] for row in report["cr_bearing_tracked_files"])
    assert buckets["lone_cr"] > 0


def test_the_instrument_refuses_to_overwrite_an_existing_report(instrument, tmp_path) -> None:
    out = tmp_path / "already.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    assert instrument.main(["--out-report", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_a_face_outside_the_repo_fails_closed(instrument, tmp_path) -> None:
    """`--face` 指到没有已跟踪件的目录 ⇒ rc=2 并点名那个面，不许返回"干净"。"""

    out = tmp_path / "nowhere.json"
    rc = instrument.main(["--face", "definitely-not-a-face/", "--out-report", str(out)])
    assert rc == 2
    assert not out.exists()


def test_the_help_face_survives_a_gbk_console_and_lists_every_flag() -> None:
    """DEBT-G81／G85 口径：真子进程＋`PYTHONIOENCODING=gbk`，capsys 不算；每个旗标名都要在帮助里。"""

    env = dict(os.environ, PYTHONIOENCODING="gbk")
    proc = subprocess.run(
        [sys.executable, str(INSTRUMENT), "--help"],
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    for flag in ("--root", "--face", "--out-report", "--overwrite"):
        assert flag in proc.stdout, flag
