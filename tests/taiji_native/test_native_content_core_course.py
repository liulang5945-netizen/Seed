"""R1 world course contracts: split, coverage, protocol and materialised files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from taiji.internalization import content_digest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/training/build_native_core_r1_course.py"
SPLIT = ROOT / "plans/manifests/native_core_r1_split_v1.json"
TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
QUERY = ROOT / "plans/manifests/native_core_r1_query_v1.json"
HOLDOUT = ROOT / "plans/manifests/native_core_r1_holdout_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r1_course_audit_20261030.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_native_core_r1_course", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_course_split_and_coverage():
    module = _load_module()
    course = module.build_course()
    audit = module.audit_course(course)
    assert audit["status"] == "course_built", audit["failures"]
    assert audit["coverage"]["split_counts"] == {"train": 216, "query": 72, "holdout": 72}
    assert len(audit["coverage"]["slot_counts"]) == 36
    assert all(
        counts == {"train": 6, "query": 2, "holdout": 2}
        for counts in audit["coverage"]["slot_counts"].values()
    )
    assert audit["coverage"]["pairs_crossing_splits"] == []


def test_protocol_targets_and_same_query():
    module = _load_module()
    course = module.build_course()
    for episode in course["episodes"]:
        turns = episode["turns"]
        assert len(turns) == 8
        assert [turn["index"] for turn in turns if turn["scored"]] == [2, 4, 6, 7]
        assert turns[2]["text"] == turns[4]["text"]
        assert turns[0]["value"] != turns[3]["value"]
        assert turns[2]["target"]["target_text"].endswith(turns[0]["value"] + "。")
        assert turns[4]["target"]["target_text"].endswith(turns[3]["value"] + "。")
        assert turns[6]["target"]["target_text"].startswith("之前")
        assert turns[7]["target"]["target_text"] == "我不知道。"
        assert turns[5]["entity"] != turns[0]["entity"]
        assert turns[7]["attribute"] != turns[0]["attribute"]
        for turn in turns:
            if turn["scored"]:
                target = turn["target"]
                assert target["target_bytes_utf8"][0] == target["first_byte"]
                assert target["supervision"] == "target_t_at_step_t"


def test_course_is_deterministic():
    module = _load_module()
    left = module.build_course()
    right = module.build_course()
    assert content_digest(left) == content_digest(right)


def test_materialised_files_match_course():
    course = _load_module().build_course()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    query = json.loads(QUERY.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    assert split["split_policy"]["train_histories"] == 216
    assert split["split_policy"]["query_histories"] == 72
    assert split["split_policy"]["holdout_histories"] == 72
    assert len(split["histories"]) == 432
    assert train["training_read"] is True
    assert query["training_read"] is False
    assert holdout["training_read"] is False
    assert len(train["episodes"]) == 216 * 4
    assert len(query["episodes"]) == 72 * 4
    assert len(holdout["episodes"]) == 72 * 4
    ids = {
        episode["episode_id"]
        for episode in (*train["episodes"], *query["episodes"], *holdout["episodes"])
    }
    assert ids == {episode["episode_id"] for episode in course["episodes"]}
    for name, payload, expected_split in (
        ("train", train, "train"),
        ("query", query, "query"),
        ("holdout", holdout, "holdout"),
    ):
        assert all(episode["split"] == expected_split for episode in payload["episodes"]), name
        assert payload["model_calls"] == 0
        assert payload["sealed_text_read"] == 0


def test_history_pairs_never_cross_splits():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    pair_splits: dict[str, set[str]] = {}
    for history in split["histories"]:
        pair_splits.setdefault(history["pair_id"], set()).add(history["split"])
    assert all(len(splits) == 1 for splits in pair_splits.values())
    history_ids = [history["history_id"] for history in split["histories"]]
    assert len(set(history_ids)) == len(history_ids) == 432


def test_report_freezes_file_hashes():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert set(report["files"]) == {
        "plans/manifests/native_core_r1_split_v1.json",
        "plans/manifests/native_core_r1_train_v1.json",
        "plans/manifests/native_core_r1_query_v1.json",
        "plans/manifests/native_core_r1_holdout_v1.json",
    }
    for relative, expected in report["files"].items():
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
