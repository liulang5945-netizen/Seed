"""Guards for `count_taiji_a30_eater_p_boundary_floor.py`（§102 判据 A 的第二条那一格）。

这些测试要钉的不是算术，而是**每一条捷径都必须响亮失败**：自己数出来的吃满预算代与仪器自述不一致、
生成行缺 `p_boundary_max` 列、探针期间件被改写——三种都必须是 rc=2，而不是一个看着合理的 0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _entry in (PROJECT_ROOT / "scripts" / "training", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import count_taiji_a30_eater_p_boundary_floor as counter  # noqa: E402

MAX_LENGTH = 256


def _generation(steps: int, p_max: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {"generation_steps": steps}
    if p_max is not None:
        row["p_boundary_max"] = p_max
    return row


def _report(generations: list[dict[str, Any]], *, declared: int | None = None,
            unchanged: bool = True) -> dict[str, Any]:
    counted = sum(1 for g in generations if g["generation_steps"] >= MAX_LENGTH)
    return {
        "format": "taiji-a30-stop-failure-v39",
        "max_length": MAX_LENGTH,
        "items_sha256": "deadbeef" * 4,
        "checkpoint_sha256": "a" * 64,
        "instrument_guard": {
            "generations_eating_full_budget": counted if declared is None else declared,
            "base_sha256_unchanged": unchanged,
        },
        "per_item": [{"id": "V001", "endstep_probe_v22": generations}],
    }


def _run(tmp_path: Path, report: dict[str, Any], capsys: pytest.CaptureFixture[str],
         extra: tuple[str, ...] = ()) -> tuple[int, dict[str, Any]]:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8", newline="\n")
    rc = counter.main(["--report", str(path), *extra])
    payload = json.loads(capsys.readouterr().out)
    return rc, payload["results"][0]


def test_floor_is_inclusive_and_only_counts_eaters() -> None:
    """`≥ 0.10` 含等号；没吃满预算的代再高也不进分子——分子与分母必须是同一群。"""

    report = _report([
        _generation(MAX_LENGTH, 0.10),
        _generation(MAX_LENGTH, 0.0999999),
        _generation(8, 0.90),
    ])
    result = counter.count_eaters(report, floor=counter.FROZEN_FLOOR)
    assert result["eaters_counted"] == 2
    assert result["eaters_above_floor"] == 1
    assert result["coherent_with_declaration"] is True
    assert result["share_of_eaters"] == 0.5


def test_wrong_declared_total_is_incoherent_not_a_number(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """两条独立读数不一致 ⇒ rc=2：把仪器自述当装饰，就等于这台仪器没有自检。"""

    report = _report([_generation(MAX_LENGTH, 0.5)], declared=99)
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "incoherent_eater_count"
    assert entry["eaters_declared_by_instrument"] == 99


def test_missing_column_never_reports_a_quiet_zero(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]) -> None:
    """缺列不许报 0（§95 那台仪器第一次造假零就缺在这一格）。"""

    report = _report([_generation(MAX_LENGTH, None), _generation(MAX_LENGTH, 0.5)])
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "column_missing_in_rows"
    assert entry["rows_without_p_boundary_max_column"] == 1


def test_moving_artifact_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """2026-10-03 那枚中途件就是这个形状：探针期间被训练器改写 ⇒ 不许拿去判 A-2。"""

    report = _report([_generation(MAX_LENGTH, 0.5)], unchanged=False)
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "artifact_changed_during_probe"


def test_items_sha_mismatch_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, 0.5)])
    rc, entry = _run(tmp_path, report, capsys, extra=("--expect-items-sha", "0" * 16))
    assert rc == 2
    assert entry["status"] == "items_sha_mismatch"


def test_ok_path_reports_below_line_as_a_conclusion(tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """够不到 ≥20 那条线是**读数**，不是仪器故障：status=ok 且 meets_frozen_line=False。"""

    report = _report([_generation(MAX_LENGTH, 0.30), _generation(MAX_LENGTH, 0.02)])
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 0
    assert entry["status"] == "ok"
    assert entry["eaters_above_floor"] == 1
    assert entry["meets_frozen_line"] is False


def test_out_report_is_lf_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, 0.30)])
    out = tmp_path / "counts.json"
    rc, _ = _run(tmp_path, report, capsys, extra=("--out-report", str(out)))
    assert rc == 0
    assert b"\r\n" not in out.read_bytes()
    assert json.loads(out.read_text(encoding="utf-8"))["results"][0]["eaters_above_floor"] == 1


def test_unreadable_input_is_refused_not_zero(tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    bogus = tmp_path / "bogus.json"
    bogus.write_text('{"format": "no-per-item"}', encoding="utf-8", newline="\n")
    rc = counter.main(["--report", str(bogus)])
    assert rc == 2
    assert "unreadable" in capsys.readouterr().out
