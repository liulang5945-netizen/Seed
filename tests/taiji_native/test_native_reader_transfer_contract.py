"""Fixed reader transfer preserves all shared parameters and uses fresh Adam."""

from pathlib import Path

import pytest
import torch

from scripts.training import run_taiji_n1_reader_transfer as runner

# 2026-10-25 owner 裁定：data/ 下 checkpoints/语料不入 git——产物在位才跑，
# CI 上跳过（本机仍全量守护）。准入出处：roadmap/12 批 C。
_ARTIFACT = (
    Path(__file__).resolve().parents[2]
    / "data/checkpoints/n1_relative_continuation_20261015/s11_position_final.pt"
)
_ARTIFACT_PRESENT = _ARTIFACT.is_file()
_ARTIFACT_REASON = (
    "consumes data/ artifacts that are not git-stored "
    "(owner 2026-10-25: checkpoints/corpus stay out of git)"
)


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_exact_transfer_fresh_adam_and_zero_restore(tmp_path):
    torch.set_num_threads(1)
    model, optimizer = runner.fresh(runner.checked_train())
    record, payload, _ = runner.transfer_source()
    assert all(
        torch.equal(model.state_dict()[k], v) for k, v in payload["candidate"]["state"].items()
    )
    assert not optimizer.state
    assert torch.count_nonzero(model.question_fusion.weight) == 0
    assert torch.count_nonzero(model.question_fusion.bias) == 0
    assert runner.PROTOCOL["updates"] == 6144
    assert runner.PROTOCOL["qa_sequences"] == 9216
    assert runner.PROTOCOL["milestones"] == [0, 3072, 6144]
    saved = runner.save(model, optimizer, tmp_path / "zero.pt", 0)
    assert runner.restore(tmp_path / "zero.pt")["signature"] == saved["expected_signature"]
    rows = [dict(r) for r in record["train_rows"] if r["seed"] == 11 and r["arm"] == "position"]
    assert runner.transfer_equal(rows, record)
    rows[0]["raw_hex"] = "00"
    assert not runner.transfer_equal(rows, record)
    assert not runner.transfer_equal(rows[:-1], record)


@pytest.mark.skipif(not _ARTIFACT_PRESENT, reason=_ARTIFACT_REASON)
def test_refuse_nontrain_before_updates():
    model, optimizer = runner.fresh(runner.checked_train())
    with pytest.raises(ValueError, match="train"):
        runner.train_step(model, optimizer, {"split": "holdout"}, "forbidden")
    assert not optimizer.state
