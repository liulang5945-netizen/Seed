from __future__ import annotations

import hashlib
import json

import pytest
import torch

from scripts.training.run_taiji_n1_c2_calibration import (
    BASE_SHA,
    ITEM_IDS,
    MANIFEST_SHA,
    items,
    parameter_digest,
    paths,
    strict_value_hit,
    validate_resume,
    write_new,
)


def test_selected_surface_is_the_preregistered_balanced_content() -> None:
    selected = items()
    assert tuple(item["id"] for item in selected) == ITEM_IDS
    assert len({item["kind"] for item in selected}) == 8
    assert sum(item["answer_tell_position"] == 0 for item in selected) == 8
    assert sum(item["answer_tell_position"] == 1 for item in selected) == 8
    assert all(len(item["turns"]) == 3 for item in selected)


def test_strict_score_never_gives_substring_or_contradiction_credit() -> None:
    assert strict_value_hit("《海客谈瀛》。\n", ["海客谈瀛"])
    assert not strict_value_hit("不是海客谈瀛，而是另一部书。", ["海客谈瀛"])
    assert not strict_value_hit("海客谈瀛，还有水经注疏", ["海客谈瀛"])


def test_report_writer_preserves_prior_evidence_and_rejects_path_escape(tmp_path) -> None:
    report = tmp_path / "report.json"
    write_new(report, {"value": "保留"})
    original = report.read_bytes()
    assert b"\r\n" not in original
    with pytest.raises(FileExistsError):
        write_new(report, {"value": "替换"})
    assert report.read_bytes() == original
    with pytest.raises(ValueError):
        paths("../escape")


def test_parameter_digest_retains_raw_storage_identity_without_python_byte_iteration(
    monkeypatch,
) -> None:
    tensors = (
        torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0.5, 1.0], dtype=torch.bfloat16),
    )

    class Model:
        def parameter_tensors(self):
            return tensors

    reference = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().clone()
        reference.update(str(value.dtype).encode("ascii"))
        reference.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        reference.update(bytes(value.untyped_storage()))

    def forbidden_storage_iteration(*args, **kwargs):
        raise AssertionError("CPU budget cannot include Python iteration over every storage byte")

    monkeypatch.setattr(torch.Tensor, "untyped_storage", forbidden_storage_iteration)
    assert parameter_digest(Model()) == reference.hexdigest()


def test_resume_preserves_once_only_surface_and_unchanged_weights() -> None:
    selected = items()
    row = {
        "item_id": ITEM_IDS[0],
        "arm": "no-replay",
        "parameters_before": "frozen",
        "parameters_after": "frozen",
    }
    report = {
        "status": "budget-stop",
        "base_sha256": BASE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "item_ids": ITEM_IDS,
        "rows": [row],
    }
    validate_resume(report, selected, "frozen")
    with pytest.raises(ValueError, match="duplicate"):
        validate_resume(dict(report, rows=[row, row]), selected, "frozen")
    with pytest.raises(ValueError, match="different weights"):
        validate_resume(
            dict(report, rows=[dict(row, parameters_after="changed")]), selected, "frozen"
        )
    with pytest.raises(ValueError, match="one mechanical"):
        validate_resume(dict(report, mechanical_repair={"count": 1}), selected, "frozen")


def test_approved_budget_continuation_preserves_repair_count(monkeypatch) -> None:
    from scripts.training import run_taiji_n1_c2_calibration as runner

    previous = json.loads(runner.paths("20261015_r1")["final"].read_text(encoding="utf-8"))
    monkeypatch.setattr(runner, "BUDGET_CONTINUATION", True)
    runner.validate_resume(previous, items(), previous["support"]["model_parameter_digest"])
    with pytest.raises(ValueError, match="existing one repair"):
        runner.validate_resume(
            dict(previous, mechanical_repair={"count": 2}),
            items(),
            previous["support"]["model_parameter_digest"],
        )
