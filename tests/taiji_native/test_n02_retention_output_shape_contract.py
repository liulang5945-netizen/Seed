"""`census_taiji_n02_retention_output_shape.py` 的合同测：判读值只有三个，且每个都能为假。

这台仪器只回答一件事——"cap0 这一面上，分数与形态各自动没动"。所以：

* 三支判读值各有真跑过的一支：分数翻转 ⇒ `scores_moved`；只动文本不动分 ⇒ `shape_moved_scores_stable`；
  两侧逐字相同 ⇒ `identical_face`（少一支就是恒真门牌）。
* 面不符必须 rc=2 且不落件：题集摘要不同、档路径相同、评测期训过练、件自述的档摘要与盘上字节不符——
  这四条各配一条负对照，因为"看着像同一张面"正是这条线付过代价的错法。
* 翻转项要连字节数一起出：本轮三条翻转项长度逐字节相同 ⇒ "掉分是因为变短"这句当场被否证，
  所以这一列是判据的一部分，不是装饰。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "census_taiji_n02_retention_output_shape.py"

CYCLIC_TEXT = "aaaaaaaab" * 2
PLAIN_TEXT = "这是一段正常的回答文字"


def _module():
    spec = importlib.util.spec_from_file_location("n02_shape_census_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_file() -> Path:
    """用一枚仓内真实文件当"检查点"，让 sha 自述守卫在测试里也走真路（不往仓库写东西）。"""

    return REPO / "plans" / "active" / "PLAN_INDEX.md"


def _artifact(text: str, score: int, *, eval_sha: str = "e" * 64, checkpoint: str = "") -> dict:
    target = Path(checkpoint) if checkpoint else _real_file()
    return {
        "format": "taiji-cap0-baseline-v1",
        "checkpoint": str(target),
        "declared_mode": "N",
        "chain": {"relax_legacy_guard": False, "constrained_decode": False},
        "trained_during_eval": False,
        "identity": {
            "eval_set_sha256": eval_sha,
            "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "dimensions": {
            name: {
                "tally": {"machine_scored_correct": score, "machine_scored_items": 20},
                "items": [
                    {
                        "id": f"{name}{index:02d}",
                        "score": score,
                        "verdict_text": text,
                        "expected_contains": ["a"],
                        "matched": [] if score == 0 else ["a"],
                    }
                    for index in range(1, 21)
                ],
            }
            for name in ("B", "C", "D", "E", "G")
        },
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")
    return path


def _argv(b: Path, a: Path, out: Path) -> list[str]:
    return ["--before", str(b), "--after", str(a), "--out-report", str(out)]


def _run(module, tmp_path, before, after, out_name="census.json"):
    b = _write(tmp_path, "before.json", before)
    a = _write(tmp_path, "after.json", after)
    out = tmp_path / out_name
    rc = module.main(_argv(b, a, out))
    return rc, out


def test_a_score_flip_is_named_with_its_byte_counts(tmp_path) -> None:
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = copy.deepcopy(before)
    after["checkpoint"] = str(tmp_path / "candidate.pt")
    Path(after["checkpoint"]).write_bytes(b"candidate-bytes")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"candidate-bytes").hexdigest()
    after["dimensions"]["D"]["items"][0]["score"] = 0
    after["dimensions"]["D"]["items"][0]["verdict_text"] = CYCLIC_TEXT
    after["dimensions"]["D"]["tally"]["machine_scored_correct"] = 0
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "scores_moved"
    row = next(r for r in payload["per_dimension"] if r["dimension"] == "D")
    assert row["score_lost"] == ["D01"]
    flip = row["flip_detail"][0]
    assert flip["id"] == "D01" and flip["score"] == [1, 0]
    assert flip["bytes"][0] != flip["bytes"][1], "翻转项必须连字节数一起出，否则'变短'这句没依据"
    assert payload["lineage"]["after_sha_matches_self_report"] is True


def test_identical_readouts_are_published_as_identical_face(tmp_path) -> None:
    module = _module()
    before, after = _artifact(PLAIN_TEXT, 1), _artifact(PLAIN_TEXT, 1)
    after["checkpoint"] = str(tmp_path / "candidate2.pt")
    Path(after["checkpoint"]).write_bytes(b"other-bytes")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"other-bytes").hexdigest()
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "identical_face"
    assert payload["totals"]["texts_changed"] == 0
    assert payload["totals"]["score_lost"] == 0


def test_moved_text_with_stable_scores_is_not_called_a_score_move(tmp_path) -> None:
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = copy.deepcopy(before)
    after["checkpoint"] = str(tmp_path / "candidate3.pt")
    Path(after["checkpoint"]).write_bytes(b"third-bytes")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"third-bytes").hexdigest()
    after["dimensions"]["E"]["items"][0]["verdict_text"] = CYCLIC_TEXT
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == "shape_moved_scores_stable"
    assert payload["totals"]["texts_changed"] == 1
    assert payload["totals"]["cyclic_like_before"] < payload["totals"]["cyclic_like_after"]


def test_a_different_eval_set_is_refused_and_writes_no_artifact(tmp_path) -> None:
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = _artifact(PLAIN_TEXT, 1, eval_sha="f" * 64)
    after["checkpoint"] = str(tmp_path / "candidate4.pt")
    Path(after["checkpoint"]).write_bytes(b"x")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"x").hexdigest()
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 2
    assert out.exists() is False


def test_a_stale_checkpoint_self_report_is_refused(tmp_path) -> None:
    """件自述的档摘要与盘上字节不符 ⇒ 这两张件不是它声称的那两个档跑出来的，拒。"""
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = _artifact(PLAIN_TEXT, 1)
    after["checkpoint"] = str(tmp_path / "candidate5.pt")
    Path(after["checkpoint"]).write_bytes(b"real-bytes")
    after["identity"]["checkpoint_sha256"] = "0" * 64
    assert _run(module, tmp_path, before, after)[0] == 2


def test_the_same_checkpoint_on_both_sides_is_refused(tmp_path) -> None:
    module = _module()
    before, after = _artifact(PLAIN_TEXT, 1), _artifact(PLAIN_TEXT, 1)
    assert _run(module, tmp_path, before, after)[0] == 2


def test_training_during_evaluation_is_refused(tmp_path) -> None:
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = _artifact(PLAIN_TEXT, 1)
    after["trained_during_eval"] = True
    assert _run(module, tmp_path, before, after)[0] == 2


def test_a_missing_dimension_refuses_naming_what_is_absent(tmp_path, capsys) -> None:
    module = _module()
    before = _artifact(PLAIN_TEXT, 1)
    after = _artifact(PLAIN_TEXT, 1)
    after["checkpoint"] = str(tmp_path / "candidate6.pt")
    Path(after["checkpoint"]).write_bytes(b"z")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"z").hexdigest()
    del after["dimensions"]["G"]
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 2
    assert out.exists() is False
    #: 拒绝消息必须点名缺的是哪一维，否则"少一维"与"少两维"给出同一句话，读不出来。
    assert "G" in capsys.readouterr().out


def test_a_published_census_artifact_is_not_clobbered(tmp_path) -> None:
    module = _module()
    before, after = _artifact(PLAIN_TEXT, 1), _artifact(PLAIN_TEXT, 1)
    after["checkpoint"] = str(tmp_path / "candidate7.pt")
    Path(after["checkpoint"]).write_bytes(b"seven")
    after["identity"]["checkpoint_sha256"] = hashlib.sha256(b"seven").hexdigest()
    rc, out = _run(module, tmp_path, before, after)
    assert rc == 0
    first = out.read_bytes()
    b = _write(tmp_path, "before.json", before)
    a = _write(tmp_path, "after.json", after)
    assert module.main(_argv(b, a, out)) == 2, "同名的第二趟必须被拒"
    assert out.read_bytes() == first
