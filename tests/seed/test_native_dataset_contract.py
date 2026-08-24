"""Native raw-byte dataset and training recommendation contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from api.training import recommend
from api.training import native
from seed.datasets import inspect_native_dataset, iter_native_documents

REPO = Path(__file__).resolve().parents[2]


def test_native_jsonl_report_matches_streaming_training_contract() -> None:
    path = REPO / "tests" / "fixtures" / "native_dataset_contract.jsonl"

    report = inspect_native_dataset(path)

    assert report.native_trainable is True
    assert report.format == "jsonl"
    assert report.documents == 2
    assert report.empty_documents == 1
    assert report.blank_lines == 1
    assert report.total_text_bytes == len("你好世界".encode("utf-8"))
    assert list(iter_native_documents([path])) == ["你好", "世界"]


def test_native_dataset_report_exposes_invalid_records() -> None:
    path = REPO / "tests" / "fixtures" / "native_dataset_broken.jsonl"

    report = inspect_native_dataset(path)

    assert report.native_trainable is False
    assert report.documents == 1
    assert report.invalid_records == 2
    assert len(report.errors) == 2

    with pytest.raises(ValueError, match="invalid JSON"):
        list(iter_native_documents([path]))


def test_native_recommendation_returns_capacity_not_transformer_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        recommend,
        "_detect_hardware",
        lambda: {
            "cpu_cores": 8,
            "ram_gb": 32.0,
            "cuda_available": False,
            "gpu_name": "",
            "vram_gb": 0,
        },
    )

    result = asyncio.run(
        recommend.get_training_recommendation(
            recommend.RecommendRequest(preset="small", dataset_size=12)
        )
    )

    assert result["runtime"] == "seed-taiji-native"
    selected = result["selected"]
    assert selected["capacity"]["planned_active_parameters"] <= 300_000
    assert selected["capacity"]["region_sizes"]
    assert "hidden_size" not in selected
    assert "num_layers" not in selected


def test_recommendation_rejects_unknown_native_preset() -> None:
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            recommend.get_training_recommendation(recommend.RecommendRequest(preset="legacy"))
        )

    assert exc_info.value.status_code == 400


def test_training_recommendation_has_no_legacy_runtime_reference() -> None:
    for relative in (
        "api/training/recommend.py",
        "api/training/resume.py",
        "api/training/native.py",
    ):
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "neuroplex" not in source
        assert "transformers" not in source
        assert "sleep_engine" not in source
    resume_source = (REPO / "api/training/resume.py").read_text(encoding="utf-8")
    assert "iter_native_documents" in resume_source
    assert "instruction" not in resume_source


def test_native_training_route_is_registered() -> None:
    paths = {route.path for route in native.router.routes}
    assert "/api/train/native" in paths
