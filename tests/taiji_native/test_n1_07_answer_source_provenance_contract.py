"""`audit_taiji_answer_source_provenance.py` 的契约册：三张面各自正反两支，外加「不许把探测当起跑」。

这台仪器的危险之处不是算错，而是**读自己的披露当证据**与**把"没有载体"读成"已经验证"**：
①仓里两枚审计件把 `answer_source` 当**数据**登着（`"answer_source": []` 是按键名归的桶）——
  裸按字段名扫就会把它们当成"一趟跑的自述"，于是 J-N1c-3 白得一个绿；
②真面今天 0 枚载体，如果"0 枚"走的是「没有矛盾 ⇒ 一致」那一支，就等于宣布判据成立。
所以这册把谓词、四支矛盾式、零载体出口、守卫被删时的响亮失败、以及一致组合拒探全部钉住。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_answer_source_provenance.py"

MINI_RUNNER = '''"""fake runner with the same guard shapes."""


def main():
    args = parse()
    if args.answer_source == "self" and args.answer_chunking != "per-answer":
        parser.error("--answer-source self only applies to --answer-chunking per-answer")
    if args.answer_source == "self" and not args.self_answers_path:
        parser.error("--answer-source self requires --self-answers <path>.")
    if args.answer_source == "corpus" and args.self_answers_path:
        parser.error("--self-answers given without --answer-source self.")
'''

AUDIT_STYLE = {"buckets": [{"key": "answer_source", "writer_sites": 3}], "answer_source": []}
ENVELOPE_OK = {
    "trainer": "train_seed_corpus",
    "ticks": 4000,
    "answer_source": "corpus",
    "self_answers_path": None,
    "answer_chunking": "stream",
}
ENVELOPE_CORPUS_WITH_TABLE = {**ENVELOPE_OK, "self_answers_path": "answers.jsonl"}
ENVELOPE_SELF_WITHOUT_TABLE = {
    **ENVELOPE_OK,
    "answer_source": "self",
    "answer_chunking": "per-answer",
}
ENVELOPE_SELF_ON_STREAM = {
    **ENVELOPE_OK,
    "answer_source": "self",
    "self_answers_path": "answers.jsonl",
    "answer_chunking": "stream",
}


def _load():
    spec = importlib.util.spec_from_file_location("n1_07_answer_source_provenance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, files: dict[str, object]) -> Path:
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):
            text = json.dumps(content, ensure_ascii=False, indent=2) + "\n"
        elif isinstance(content, list):
            text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content)
        else:
            text = str(content)
        target.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    return tmp_path


def _faces(tmp_path: Path, carrier_files: dict[str, object]) -> dict[str, object]:
    files = {"scripts/training/train_seed_corpus.py": MINI_RUNNER}
    files.update(carrier_files)
    module = _load()
    root = _fixture(tmp_path, files)
    return module.scan_carrier_face(root)


def test_audit_style_mention_is_not_a_carrier(tmp_path: Path) -> None:
    """仪器不许读自己的披露：按字段名扫会把它当载体，那 J-N1c-3 就是白得的绿。"""

    face = _faces(tmp_path, {"reports/an_audit.json": AUDIT_STYLE})
    assert face["carrier_count"] == 0, face
    assert face["name_mental_only_files"] == 1, face
    assert face["verdict"] == "no_carrier_outside_weight_artifacts", face


def test_envelope_record_is_a_carrier_and_consistent(tmp_path: Path) -> None:
    face = _faces(tmp_path, {"output/run/exit.json": ENVELOPE_OK})
    assert face["carrier_count"] == 1, face
    assert face["mismatches"] == [], face
    assert face["verdict"] == "consistent_within_carriers", face


def test_reference_labeled_as_model_side_is_mismatch(tmp_path: Path) -> None:
    """判据原文点名的那一支：`answer_source` 指向参考侧而产出路径是模型侧。"""

    face = _faces(tmp_path, {"output/run/exit.json": ENVELOPE_CORPUS_WITH_TABLE})
    assert face["verdict"] == "answer_source_mismatch", face
    assert any("模型侧标成参考侧" in row or "参考侧" in row for row in face["mismatches"]), face


def test_self_side_without_table_is_mismatch(tmp_path: Path) -> None:
    face = _faces(tmp_path, {"output/run/exit.json": ENVELOPE_SELF_WITHOUT_TABLE})
    assert face["verdict"] == "answer_source_mismatch", face
    assert len(face["mismatches"]) == 1, face["mismatches"]


def test_self_side_on_stream_feed_is_mismatch(tmp_path: Path) -> None:
    face = _faces(tmp_path, {"output/run/exit.json": ENVELOPE_SELF_ON_STREAM})
    assert face["verdict"] == "answer_source_mismatch", face
    assert any("连续流" in row for row in face["mismatches"]), face["mismatches"]


def test_zero_carriers_is_not_read_as_verified(tmp_path: Path) -> None:
    """空面必须走"没有载体"那一支，而不是"没有矛盾 ⇒ 一致"。"""

    face = _faces(tmp_path, {"output/run/other.json": {"ticks": 1, "exit_reason": "budget"}})
    assert face["carrier_count"] == 0, face
    assert face["verdict"] == "no_carrier_outside_weight_artifacts", face
    assert face["verdict"] != "consistent_within_carriers"


def test_jsonl_records_are_read_per_line(tmp_path: Path) -> None:
    """进度流是 JSONL：只解析 `.json` 会让"0 枚载体"成为取法造出来的假缺席。"""

    face = _faces(
        tmp_path,
        {"output/run/progress.jsonl": [ENVELOPE_OK, ENVELOPE_CORPUS_WITH_TABLE]},
    )
    assert face["carrier_count"] == 2, face
    assert len(face["mismatches"]) == 1, face["mismatches"]


def test_enforcement_face_fails_loud_when_guards_are_removed(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(tmp_path, {"scripts/training/train_seed_corpus.py": "def main():\n    pass\n"})
    try:
        module.scan_enforcement_face(root)
    except ValueError as exc:
        assert "零命中" in str(exc), exc
    else:
        raise AssertionError("守卫被删必须响亮失败，不许读成「没有可反写的形状」")


def test_enforcement_face_picks_the_three_guards(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(tmp_path, {"scripts/training/train_seed_corpus.py": MINI_RUNNER})
    face = module.scan_enforcement_face(root)
    assert face["guard_count"] == 3, face
    assert all("answer_source" in str(guard["condition"]) for guard in face["guards"]), face


def test_consistent_combo_is_refused_as_a_probe() -> None:
    """探测只许打非法组合：一致组合会真的开始一趟跑，那就不再是"只读已存在的件"。"""

    module = _load()
    for argv in (
        ["--answer-source", "self", "--self-answers", "t.jsonl", "--answer-chunking", "per-answer"],
        ["--answer-source", "corpus"],
    ):
        try:
            module._probe_argv(argv)
        except ValueError as exc:
            assert "一致" in str(exc), exc
        else:
            raise AssertionError(f"一致组合必须拒探：{argv}")
    for control in module.NEGATIVE_CONTROLS:
        argv = module._probe_argv(list(control["argv"]))
        assert "--checkpoint" in argv, argv


def test_negative_controls_are_actually_rejected(tmp_path: Path) -> None:
    """静态面要有一条真跑的对账：两支非法组合都在加载之前被关掉（rc=2）。"""

    module = _load()
    rows = module.run_negative_controls(PROJECT_ROOT)
    assert len(rows) == 2, rows
    for row in rows:
        assert row["rc"] == 2, row
        assert row["rejected"] is True, row
        assert row["fragment_seen"] is True, row
    assert not (PROJECT_ROOT / "output" / "_answer_source_probe_should_not_exist").exists()


def test_real_face_reproduces_today_reading() -> None:
    """真面锚点：0 枚 JSON 载体／审计式提及被谓词排除／3 支守卫／产品侧 0 写者／`.pt` 面不开。"""

    module = _load()
    report = module.build_report(PROJECT_ROOT, run_controls=False)
    carrier = report["carrier_face"]
    #: 这里**不钉 `carrier_count`**：载体数会随跟踪面与"有没有跑过一次训练器"变化——
    #: ㊵-708 封件时是 0，㊵-712 整族回归里某支测写了退出件就变成 1（`output/seed_corpus_smoke_progress_exit.json`）。
    #: 钉住的是**不变量**：不许出现矛盾、判级词必须在词典内、载体只能是跑道器自己的取值。
    assert carrier["mismatches"] == [], carrier
    assert carrier["verdict"] in module.CARRIER_VERDICTS, carrier
    for row in carrier["carriers"]:
        assert row["answer_source"] in module.VALID_SOURCES, row
        if row["answer_source"] == "self":
            assert row["self_answers_path"], row
    #: 能红的守卫：这一列必须是**全集**而不是样本——上一版仪器把它截成 12 枚，
    #: 于是下面那句成员断言按字母序失败，读起来却像"来源面今天少了一枚审计式提及"。
    assert len(carrier["name_mental_only_files_list"]) == carrier["name_mental_only_files"]
    mentions = set(carrier["name_mental_only_files_list"])
    assert {
        "reports/taiji_default_reachability_20261011.json",
        "reports/taiji_write_only_key_face_20261011.json",
    } <= mentions, carrier
    # 「提及数」不是载体数，而且它随**跟踪面**变化：载体面取 `git ls-files -- reports`，
    # 所以本仪器自己的落盘件一旦被 `git add` 就会多算一枚提及（㊵-712 整族复算实测已从 2 涨到 4，
    # 涨的是 `taiji_answer_source_provenance` 与 `taiji_n1_03_publication_claim_gate` 两件自己）。
    # 点名列表因此是必需的披露，不是装饰——只有它能解释这个数字为什么会长。
    assert carrier["name_mental_only_files"] == len(mentions), carrier
    assert report["enforcement_face"]["guard_count"] == 3, report["enforcement_face"]
    assert report["product_face"]["product_writer_files"] == [], report["product_face"]
    weight = report["weight_artifact_face"]
    assert weight["opened"] is False, weight
    #: 计数必须带面名：这里的下界是「`output`＋`checkpoints`」这一张面，不是全仓（全仓现取 867）。
    assert weight["scan_faces"] == ["checkpoints", "output"], weight
    assert weight["pt_count"] > 100, weight
    assert len(report["what_this_does_not_prove"]) >= 4, report["what_this_does_not_prove"]
    assert carrier["files_scanned"] > 100, carrier["files_scanned"]


def test_cli_refuses_to_overwrite_and_writes_lf(tmp_path: Path) -> None:
    out = tmp_path / "prov.json"
    out.write_text("{}", encoding="utf-8", newline="\n")
    first = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(PROJECT_ROOT),
            "--out-report",
            str(out),
            "--skip-controls",
        ],
        capture_output=True,
        check=False,
    )
    assert first.returncode == 2, first.stderr[:200]
    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(PROJECT_ROOT),
            "--out-report",
            str(out),
            "--skip-controls",
            "--overwrite",
        ],
        capture_output=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr[:300]
    assert b"\r\n" not in out.read_bytes()
