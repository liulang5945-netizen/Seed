"""检查点按名删除端点（``DELETE /api/train/checkpoint/{filename}``）守卫。

为什么需要这组守卫：面板让操作者挑着删旧检查点（它们会长期堆积到数十 MB 一枚），
但两枚检查点不得删——活跃件（当前承载应答）与已配置件（下次启动加载）：删掉它们
会直接破坏正在运行的应答与下一次启动。名字校验同样必须是"平坦文件名"，不能让
任何带目录/冒号/点前缀的输入摸到目录之外或被隐藏的临时件。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

import api.training.checkpoints as training_checkpoints


def _seed_dir(tmp_path: Path) -> Path:
    """造一个含两枚可见 .pt 的检查点目录（内容只要能被 stat）。"""

    root = tmp_path / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    (root / "old.pt").write_bytes(b"x")
    (root / "keep.pt").write_bytes(b"y")
    return root


def _isolate(tmp_path: Path, monkeypatch) -> Path:
    """把目录与活跃/配置判定都指向测试替身，绝不触碰真实文件。"""

    root = _seed_dir(tmp_path)
    monkeypatch.setattr(training_checkpoints, "_CHECKPOINT_DIR", root)
    monkeypatch.setattr(training_checkpoints, "_active_checkpoint_id", lambda: "")
    monkeypatch.setattr(training_checkpoints, "_configured_checkpoint_id", lambda: "")
    return root


def test_delete_removes_the_named_checkpoint(tmp_path: Path, monkeypatch) -> None:
    root = _isolate(tmp_path, monkeypatch)

    payload = training_checkpoints.delete_checkpoint("old.pt")

    assert payload["status"] == "success"
    assert payload["message"] == "检查点 old.pt 已删除"
    assert not (root / "old.pt").exists()
    assert (root / "keep.pt").exists()


def test_delete_refuses_the_active_checkpoint(tmp_path: Path, monkeypatch) -> None:
    root = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(training_checkpoints, "_active_checkpoint_id", lambda: "old.pt")

    with pytest.raises(HTTPException) as refused:
        training_checkpoints.delete_checkpoint("old.pt")

    assert refused.value.status_code == 409
    assert (root / "old.pt").exists(), "活跃检查点被删掉了，正在运行的应答会被破坏"


def test_delete_refuses_the_configured_checkpoint(tmp_path: Path, monkeypatch) -> None:
    root = _isolate(tmp_path, monkeypatch)
    monkeypatch.setattr(training_checkpoints, "_configured_checkpoint_id", lambda: "keep.pt")

    with pytest.raises(HTTPException) as refused:
        training_checkpoints.delete_checkpoint("keep.pt")

    assert refused.value.status_code == 409
    assert (root / "keep.pt").exists(), "已配置检查点被删掉了，下一次启动会失去基座"


def test_delete_refuses_names_that_leave_the_directory(tmp_path: Path, monkeypatch) -> None:
    root = _isolate(tmp_path, monkeypatch)

    for name in (
        "..",
        "../evil.pt",
        "sub/evil.pt",
        "sub\\evil.pt",
        "C:evil.pt",
        "notes.txt",
        ".hidden.pt",
    ):
        with pytest.raises(HTTPException) as refused:
            training_checkpoints.delete_checkpoint(name)
        assert refused.value.status_code == 400, f"{name} 应被名字校验拒绝"

    assert sorted(p.name for p in root.glob("*.pt")) == ["keep.pt", "old.pt"]


def test_delete_answers_404_for_a_missing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    _isolate(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as missing:
        training_checkpoints.delete_checkpoint("gone.pt")

    assert missing.value.status_code == 404
