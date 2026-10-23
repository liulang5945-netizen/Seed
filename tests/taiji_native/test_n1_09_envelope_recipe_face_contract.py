"""`read_taiji_envelope_recipe_face.py` 的契约册：三态分开、覆盖必出版、抽样不许冒充全面。

这台仪器只有读没有写，所以它的危险不在改坏东西，而在**把话说过头**：
①「信封里没有 `answer_source`」＝无记录，既不是反写也不是合规——默认基座 13 枚全是这一类；
②只读了一部分时判级词必须带 `partial_` 前缀，否则抽样会被读成全面结论；
③批次选择跳过的枚数必须**带理由**出版，不然"这一面没记录"可能只是"我没读到"。
夹具一律用 `torch.save` 现造小信封，不碰仓里任何一枚真件。
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "read_taiji_envelope_recipe_face.py"
REPORT = PROJECT_ROOT / "reports" / "taiji_envelope_recipe_face_20261012.json"


def _load():
    spec = importlib.util.spec_from_file_location("n1_09_envelope_recipe_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_envelope(path: Path, metadata: dict | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"config": {"taiji": {}}, "substrate": {}}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, path)
    return path


def _root(tmp_path: Path, envelopes: dict[str, dict | None]) -> Path:
    for rel, metadata in envelopes.items():
        _write_envelope(tmp_path / "checkpoints" / rel, metadata)
    return tmp_path


RECIPE_OK = {
    "trainer": "train_seed_corpus",
    "answer_source": "corpus",
    "answer_chunking": "stream",
    "self_answers_path": None,
}
RECIPE_MISMATCH = {**RECIPE_OK, "answer_source": "corpus", "self_answers_path": "answers.jsonl"}
RECIPE_SELF_NO_TABLE = {**RECIPE_OK, "answer_source": "self", "self_answers_path": None}


def test_absent_recipe_is_recorded_as_no_record_not_as_violation() -> None:
    """①的核心：没记录不等于合规。三枚不同的缺失形状要能分开。"""

    module = _load()
    predicate = module._provenance_predicate()
    no_metadata = module.classify_envelope({"config": {}}, "a.pt", predicate)
    assert no_metadata["has_recipe"] is False and no_metadata["reason"] == "no metadata dict"
    no_key = module.classify_envelope({"metadata": {"trainer": "x"}}, "b.pt", predicate)
    assert no_key["reason"] == "metadata has no answer_source", no_key
    assert no_key["metadata_keys"] == ["trainer"], no_key
    assert not (set(no_metadata) & {"mismatches"} and no_metadata["mismatches"])


def test_present_recipe_is_judged_by_the_shared_predicate() -> None:
    """矛盾式必须**复用** `audit_taiji_answer_source_provenance`，不在这台里手写第二份。"""

    module = _load()
    predicate = module._provenance_predicate()
    ok = module.classify_envelope({"metadata": RECIPE_OK}, "ok.pt", predicate)
    bad = module.classify_envelope({"metadata": RECIPE_MISMATCH}, "bad.pt", predicate)
    self_no_table = module.classify_envelope({"metadata": RECIPE_SELF_NO_TABLE}, "s.pt", predicate)
    assert ok["has_recipe"] is True and ok["mismatches"] == [], ok
    assert bad["mismatches"] and "answer_source=corpus" in bad["mismatches"][0], bad
    assert self_no_table["mismatches"] and "self_answers_path" in self_no_table["mismatches"][0]


def test_empty_sample_refuses_to_publish_a_verdict(tmp_path: Path) -> None:
    module = _load()
    root = tmp_path / "empty"
    (root / "checkpoints").mkdir(parents=True)
    try:
        module.read_face(root, 8, 8, [], ["checkpoints"])
    except ValueError as exc:
        assert "批次为空" in str(exc), exc
    else:
        raise AssertionError("零样本必须拒绝，不许出版一个看起来像结论的判级词")


def test_batch_selection_publishes_why_files_were_skipped(tmp_path: Path) -> None:
    """字节上限与枚数上限各挡一批，两笔跳过数都要在场——否则「没读到」会伪装成「没有」。

    2026-10-12 自抓：全挡那半段原来写的是 `--max-mb 1`，而 1 是**一兆**预算，
    小信封根本挡不住 ⇒ 那半段是恒真式（当时被 git_head 的崩盖住，没人看见）。现在两个上限各自现证。
    """

    module = _load()
    root = _root(tmp_path, {"a.pt": RECIPE_OK, "b.pt": RECIPE_OK, "c.pt": RECIPE_OK})
    points = module.enumerate_face(root, ["checkpoints"])
    chosen, by_size, by_count = module.select_batch(points, 2, 1024, [])
    assert len(chosen) == 2 and by_count == 1 and by_size == 0, (len(chosen), by_size, by_count)
    blocked, blocked_size, blocked_count = module.select_batch(points, 8, 0, [])
    assert blocked == [] and blocked_size == 3 and blocked_count == 0, (blocked_size, blocked_count)
    for label, max_files, max_mb in (("max-mb=0", 8, 0), ("max-files=0", 0, 1024)):
        try:
            module.read_face(root, max_files, max_mb, [], ["checkpoints"])
        except ValueError as exc:
            assert "批次为空" in str(exc), (label, exc)
        else:
            raise AssertionError(f"{label} 把每一枚都挡住时必须拒绝出版判级词")


def test_only_flag_rejects_a_misnamed_file(tmp_path: Path) -> None:
    """点名的件不存在必须响亮失败——静默跳过会让"我扫了那三枚"变成假话。"""

    module = _load()
    root = _root(tmp_path, {"real.pt": RECIPE_OK})
    try:
        module.select_batch(module.enumerate_face(root, ["checkpoints"]), 4, 8, ["nope.pt"])
    except ValueError as exc:
        assert "不存在" in str(exc), exc
    else:
        raise AssertionError("--only 打空必须拒绝")


def test_only_mode_publishes_that_it_was_hand_named(tmp_path: Path) -> None:
    """`--only` 点名的枚数可以来自扫描根之外 ⇒ 必须另出版挑法，不然 13 与 1 会被读成同源。"""

    module = _load()
    root = _root(tmp_path, {"a.pt": RECIPE_OK, "b.pt": RECIPE_OK})
    report = module.read_face(
        root, 8, 1024, [(root / "checkpoints" / "a.pt").as_posix()], ["checkpoints"]
    )
    assert report["coverage"]["batch_mode"] == "only_named", report["coverage"]
    assert report["coverage"]["files_selected"] == 1 and report["coverage"]["files_in_face"] == 2
    assert report["coverage"]["with_recipe_total"] == 1, report["coverage"]
    assert report["verdict"] == "partial_consistent", report["verdict"]
    assert (
        module.read_face(root, 8, 1024, [], ["checkpoints"])["coverage"]["batch_mode"] == "budget"
    )


def test_partial_face_must_carry_the_partial_prefix(tmp_path: Path) -> None:
    """只读到一部分时判级词一律带 `partial_`；全读到时不许带。"""

    module = _load()
    root = _root(
        tmp_path,
        {"a.pt": RECIPE_OK, "b.pt": RECIPE_OK, "c.pt": RECIPE_OK},
    )
    full = module.read_face(root, 8, 1024, [], ["checkpoints"])
    assert full["coverage"]["face_is_partial"] is False, full["coverage"]
    assert full["verdict"] == "consistent", full["verdict"]
    subset = module.read_face(root, 1, 1024, [], ["checkpoints"])
    assert subset["coverage"]["face_is_partial"] is True, subset["coverage"]
    assert subset["verdict"] == "partial_consistent", subset["verdict"]
    assert subset["verdict"] in module.VERDICTS, subset["verdict"]
    assert subset["coverage"]["selection_order"] == "size_ascending", subset["coverage"]


def test_no_recipe_anywhere_is_a_third_state_not_green(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path, {"legacy_a.pt": None, "legacy_b.pt": {"tick": 1, "trainer": "x"}})
    report = module.read_face(root, 8, 1024, [], ["checkpoints"])
    assert report["verdict"] == "no_recorded_recipe_in_sample", report["verdict"]
    assert report["mismatches"] == [], report
    assert len(report["carriers_without_recipe"]) == 2, report["carriers_without_recipe"]
    assert report["coverage"]["no_recipe_reason_counts"] == {
        "metadata has no answer_source": 1,
        "no metadata dict": 1,
    }, report["coverage"]


def test_instrument_writes_nothing_into_the_envelope_face(tmp_path: Path) -> None:
    """只读承诺要能红：任何一枚 `.pt` 的 sha 在复算后必须不变。"""

    import hashlib

    module = _load()
    root = _root(tmp_path, {"a.pt": RECIPE_OK, "b.pt": None})
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "checkpoints").glob("*.pt")
    }
    module.read_face(root, 8, 1024, [], ["checkpoints"])
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "checkpoints").glob("*.pt")
    }
    assert before == after, (before, after)


def test_truncated_detail_rows_still_publish_their_total(tmp_path: Path) -> None:
    """明细被 `ROW_LIST_CAP` 切掉时总数必须另记一格——切掉的行不许伪装成「这一面没有」。

    这台仪器自己就是这条坑的当事人：装载面 13 枚，明细只印前 12 行。
    """

    module = _load()
    root = _root(tmp_path, {f"e{index}.pt": None for index in range(module.ROW_LIST_CAP + 1)})
    report = module.read_face(root, 32, 1024, [], ["checkpoints"])
    assert report["coverage"]["files_loaded"] == module.ROW_LIST_CAP + 1, report["coverage"]
    assert report["coverage"]["without_recipe_total"] == module.ROW_LIST_CAP + 1, report["coverage"]
    assert len(report["carriers_without_recipe"]) == module.ROW_LIST_CAP
    assert report["coverage"]["no_recipe_reason_counts"] == {
        "no metadata dict": module.ROW_LIST_CAP + 1
    }, report["coverage"]
    assert report["coverage"]["row_list_cap"] == module.ROW_LIST_CAP


def test_real_load_face_reading_is_reproduced_and_fully_covered() -> None:
    """真面锚点：默认装载面 13 枚**全读**、0 枚带来源记录、且两种缺失分支都在场。"""

    module = _load()
    sealed = json.loads(REPORT.read_text(encoding="utf-8"))
    #: 封印报告只出版 12 行明细（cap）；第 13 枚由 face_total_mb=154.7 反推为这枚 12.7MB 的件。
    sealed_files = [row["file"] for row in sealed["carriers_without_recipe"]]
    sealed_files.append("checkpoints/seed_n2b_rollback_20261008_n2c1.pt")
    #: checkpoints/ 现已增长到 26 枚；用只含封印 13 枚的临时硬链面复现旧读数，不把新件算进面。
    face_root = Path(tempfile.mkdtemp(prefix="n1_09_face_", dir=PROJECT_ROOT))
    try:
        (face_root / "checkpoints").mkdir(parents=True)
        for rel in sealed_files:
            target = face_root / rel
            try:
                os.link(PROJECT_ROOT / rel, target)
            except OSError:
                shutil.copy2(PROJECT_ROOT / rel, target)
        report = module.read_face(face_root, 64, 1024, [], ["checkpoints"])
    finally:
        shutil.rmtree(face_root, ignore_errors=True)
    assert report["verdict"] == sealed["verdict"] == "no_recorded_recipe_in_sample", (
        report["verdict"],
        sealed["verdict"],
    )
    assert report["coverage"]["files_in_face"] == 13, report["coverage"]
    assert report["coverage"]["files_loaded"] == 13, report["coverage"]
    assert report["coverage"]["without_recipe_total"] == 13, report["coverage"]
    assert len(report["carriers_without_recipe"]) == 12, "明细按 cap 截断，总数另记在 coverage"
    assert report["coverage"] == sealed["coverage"], (report["coverage"], sealed["coverage"])
    assert report["coverage"]["face_is_partial"] is False, report["coverage"]
    assert report["carriers_with_recipe"] == [], report["carriers_with_recipe"]
    counts = report["coverage"]["no_recipe_reason_counts"]
    assert sum(counts.values()) == 13, counts
    assert set(counts) == {"metadata has no answer_source", "no metadata dict"}, counts
    assert report["anything_written"] is False and report["opened_read_only"] is True


def test_cli_refuses_to_overwrite_and_writes_lf(tmp_path: Path) -> None:
    out = tmp_path / "face.json"
    out.write_text("{}", encoding="utf-8", newline="\n")
    refused = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-report", str(out)], capture_output=True, check=False
    )
    assert refused.returncode == 2, refused.stderr[:200]
    written = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-report", str(out), "--overwrite"],
        capture_output=True,
        check=False,
    )
    assert written.returncode == 0, written.stderr[:300]
    assert b"\r\n" not in out.read_bytes()


def test_git_head_disclosure_is_honest_on_both_branches(tmp_path: Path) -> None:
    """`git_head` 是披露格：真仓库给短 SHA，非仓库根给 `not_a_git_repo`，两支都不许崩。

    前一支防"编一个 SHA 当披露"，后一支防"为夹具根整台仪器炸掉"——只测其中一支这格就是恒真式。
    """

    module = _load()
    assert re.fullmatch(r"[0-9a-f]{7,40}", module.repo_head(PROJECT_ROOT)), module.repo_head(
        PROJECT_ROOT
    )
    foreign = _root(tmp_path, {"a.pt": RECIPE_OK})
    assert module.repo_head(foreign) == "not_a_git_repo", module.repo_head(foreign)
    report = module.read_face(foreign, 8, 1024, [], ["checkpoints"])
    assert report["git_head"] == "not_a_git_repo", report["git_head"]
    assert report["verdict"] == "consistent", report["verdict"]
