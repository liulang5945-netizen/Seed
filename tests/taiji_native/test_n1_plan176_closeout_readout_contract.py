"""N176 收束判读器（`analyze_taiji_n1_plan176_sealed_closeout.py`）的契约册。

这台仪器只读那 1360 份固定输出，所以测它**不能**碰真件：这里用同型分母的合成行端到端跑
`main()`，证明门槛会响也会不响（F1 主色全对 ⇒ 该门转绿；只放一条给面不改判据必红）。

同时钉住 SPEC 那条归一化：结束标记、首尾空白、末尾句号都要被剥掉，非法 UTF-8 记失败。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTRUMENT = REPO / "scripts/training/analyze_taiji_n1_plan176_sealed_closeout.py"

GOOD = "白色。\n<|end|>".encode()


def _module():
    spec = importlib.util.spec_from_file_location("n176_closeout_readout", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def _row(
    arm: str,
    condition: str,
    *,
    correct: bool = True,
    level: str | None = None,
    query_kind: str | None = None,
    raw: bytes = GOOD,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "arm": arm,
        "condition": condition,
        "correct": correct,
        "generated_hex": raw.hex(),
        "valid_utf8": True,
        "terminated": raw.strip().endswith(b"<|end|>"),
    }
    if level is not None:
        row["level"] = level
    if query_kind is not None:
        row["query_kind"] = query_kind
    if condition == "N175_train_retention":
        row["free_prefix_match"] = 0.09375
        row.pop("valid_utf8")
        row.pop("terminated")
    return row


def _rows(arm: str, *, primary_correct: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows += [_row(arm, "matched", level="F0") for _ in range(128)]
    primary = [
        _row(arm, "matched", level="F1", query_kind="primary_color", correct=primary_correct)
        for _ in range(64)
    ]
    for index, row in enumerate(primary):
        row["episode_id"] = f"ep-{index:03d}"
    rows += primary
    rows += [_row(arm, "matched", level="F1", query_kind="distractor_color") for _ in range(64)]
    related = [_row(arm, "related_fact_flip", correct=primary_correct) for _ in range(64)]
    for index, row in enumerate(related):
        row["episode_id"] = f"ep-{index:03d}"
    rows += related
    rows += [_row(arm, "irrelevant_fact_flip") for _ in range(64)]
    rows += [_row(arm, "F0_train_at640") for _ in range(16)]
    rows += [_row(arm, "F0_train_final") for _ in range(16)]
    rows += [_row(arm, "N175_train_retention") for _ in range(16)]
    return rows


def _run(instrument, tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = tmp_path / "outputs.jsonl"
    outputs.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8", newline="\n"
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"status": "fixed_generation_complete", "attempt_sha256": "0" * 64}),
        encoding="utf-8",
        newline="\n",
    )
    report_path = tmp_path / "readout.json"
    originals = (instrument.OUTPUTS, instrument.RECEIPT, instrument.REPORT)
    instrument.OUTPUTS, instrument.RECEIPT, instrument.REPORT = outputs, receipt, report_path
    try:
        assert instrument.main() == 0
    finally:
        instrument.OUTPUTS, instrument.RECEIPT, instrument.REPORT = originals
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_a_failing_primary_gate_is_reported_as_not_met(instrument, tmp_path: Path) -> None:
    report = _run(instrument, tmp_path, _rows("arm_a", primary_correct=False))
    gate = report["per_arm"]["arm_a"]["gates"]
    assert gate["f0_valid_utf8_ge_0.99"] is True
    assert gate["f0_terminated_ge_0.95"] is True
    assert gate["f1_primary_ge_0.90"] is False
    assert gate["f1_related_flip_ge_0.90"] is False
    assert report["per_arm"]["arm_a"]["denominators"]["f0_matched"] == 128


def test_the_primary_gate_can_turn_green(instrument, tmp_path: Path) -> None:
    report = _run(instrument, tmp_path, _rows("arm_a", primary_correct=True))
    gate = report["per_arm"]["arm_a"]["gates"]
    assert gate["f1_primary_ge_0.90"] is True
    assert gate["f1_related_flip_ge_0.90"] is True
    assert report["flag_cross_check"]["mismatches"] == {"valid_utf8": 0, "terminated": 0}


def test_architecture_attribution_reports_every_spec_bullet(instrument, tmp_path: Path) -> None:
    """SPEC 要求"架构归因必须同时报告"五件：这里钉住五件都在件里，且两条读数由数算出。"""

    report = _run(instrument, tmp_path, _rows("arm_a", primary_correct=False))
    attribution = report["architecture_attribution"]
    for key in (
        "workspace_consumption",
        "real_content_use",
        "curriculum_order_effect",
        "competing_explanations",
        "retention",
        "conclusion",
    ):
        assert key in attribution, key
    assert attribution["real_content_use"]["related_flip_equals_primary_in_every_arm"] is True
    assert attribution["conclusion"] == "structural gap repaired, behavioural gain not demonstrated"


def test_denominator_drift_fails_closed(instrument, tmp_path: Path) -> None:
    rows = _rows("arm_a", primary_correct=True)
    rows.pop()  # 少一条保持行 ⇒ 分母与 SPEC 不符
    with pytest.raises(ValueError, match="fixed denominators differ"):
        _run(instrument, tmp_path, rows)


def test_normalization_strips_marker_whitespace_and_terminal_punctuation(instrument) -> None:
    assert instrument.normalize("  白色。\n".encode() + b"<|end|>") == ("白色", True)
    assert instrument.normalize(b"\xe7\x99\xbd\xe8\x89\xb2") == ("白色", False)
    assert instrument.normalize(b"\xff<|end|>") == (None, True)
