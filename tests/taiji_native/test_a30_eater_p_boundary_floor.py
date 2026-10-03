"""Guards for `count_taiji_a30_eater_p_boundary_floor.py`（§102 判据 A 的第二条那一格）。

这些测试要钉的不是算术，而是**每一条捷径都必须响亮失败**：自己数出来的吃满预算代与仪器自述不一致、
从不发 LF 那一群的数与 `peak_run_summary_v37.eaters_never_lf_n` 不一致、生成行缺 `p_boundary_max`
或缺 `lf_trace_v29.lf_step_count`、探针期间件被改写——都必须是 rc=2，而不是一个看着合理的数。

另有一条**人群口径**的钉子：§102 的 A-2 冻在"从不发 LF 那一群"上（那一群的实测上限 0.0478 < 0.10），
不是冻在全体拖写代上。今天装机底上全体拖写代里有 8 代够到 0.10，而这一群里是 0 ——把两人数搞混就是
拿一个更宽的口径去答一个更窄的问题，所以两个数分开报，`meets_frozen_line` 只认窄的那一个。
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

#: 已入库的装机底 ×挂 seedA ×门 OFF ×96 枚基线；§102 的 A 就是按这一装配冻的。
SEALED_BASELINE = PROJECT_ROOT / "reports/taiji_a30_stop_failure_self_v38_evidenceobserver_circuitseedA_96_20261003.json"


def _generation(steps: int, p_max: float | None, *, lf: int = 0) -> dict[str, Any]:
    row: dict[str, Any] = {"generation_steps": steps, "lf_trace_v29": {"lf_step_count": lf}}
    if p_max is not None:
        row["p_boundary_max"] = p_max
    return row


def _report(generations: list[dict[str, Any]], *, declared: int | None = None,
            never_lf_declared: int | None = None, unchanged: bool = True) -> dict[str, Any]:
    counted = sum(1 for g in generations if g["generation_steps"] >= MAX_LENGTH)
    never_lf = sum(1 for g in generations
                   if g["generation_steps"] >= MAX_LENGTH
                   and g["lf_trace_v29"]["lf_step_count"] == 0)
    return {
        "format": "taiji-a30-stop-failure-v39",
        "max_length": MAX_LENGTH,
        "items_sha256": "deadbeef" * 4,
        "checkpoint_sha256": "a" * 64,
        "instrument_guard": {
            "generations_eating_full_budget": counted if declared is None else declared,
            "base_sha256_unchanged": unchanged,
        },
        "peak_run_summary_v37": {"eaters_never_lf_n": never_lf if never_lf_declared is None
                                 else never_lf_declared},
        "per_item": [{"id": "V001", "endstep_probe_v22": generations}],
    }


def _run(tmp_path: Path, report: dict[str, Any], capsys: pytest.CaptureFixture[str],
         extra: tuple[str, ...] = ()) -> tuple[int, dict[str, Any]]:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8", newline="\n")
    rc = counter.main(["--report", str(path), *extra])
    payload = json.loads(capsys.readouterr().out)
    return rc, payload["results"][0]


def test_floor_is_inclusive_and_population_is_the_never_lf_one() -> None:
    """≥0.10 含等号；`meets_frozen_line` 只数从不发 LF 那一群——发过 LF 的那 8 代不算数。"""

    report = _report([
        _generation(MAX_LENGTH, 0.10, lf=0),        # never-LF eater, on the line ⇒ 窄口径 +1
        _generation(MAX_LENGTH, 0.50, lf=2),        # 发过 LF ⇒ 只进宽口径
        _generation(MAX_LENGTH, 0.0999999, lf=0),   # 差一点点 ⇒ 两个都不进
        _generation(8, 0.90, lf=0),                 # 没吃满预算 ⇒ 谁都不进
    ])
    result = counter.count_eaters(report, floor=counter.FROZEN_FLOOR)
    assert result["eaters_counted"] == 3
    assert result["never_lf_eaters_counted"] == 2
    assert result["eaters_above_floor_all_eaters"] == 2
    assert result["eaters_above_floor"] == 1
    assert result["share_of_never_lf_eaters"] == 0.5
    assert result["coherent_with_declaration"] is True
    assert result["coherent_with_never_lf_declaration"] is True


def test_wrong_declared_total_is_incoherent_not_a_number(tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, 0.5, lf=0)], declared=99)
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "incoherent_eater_count"


def test_wrong_declared_never_lf_count_is_also_incoherent(tmp_path: Path,
                                                          capsys: pytest.CaptureFixture[str]) -> None:
    """窄口径同样要绑住仪器自述：`peak_run_summary_v37.eaters_never_lf_n` 与我数的一致才许出数。"""

    report = _report([_generation(MAX_LENGTH, 0.5, lf=0), _generation(MAX_LENGTH, 0.2, lf=1)],
                     never_lf_declared=77)
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "incoherent_never_lf_count"
    assert entry["never_lf_eaters_declared"] == 77


def test_missing_boundary_column_never_reports_a_quiet_zero(tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, None, lf=0), _generation(MAX_LENGTH, 0.5, lf=0)])
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "column_missing_in_rows"
    assert entry["rows_without_p_boundary_max_column"] == 1


def test_missing_lf_trace_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """没有 `lf_step_count` 就答不了"从不发 LF"，只能拒——把缺列当 0 会造出一个假的"这一群是空的"。"""

    report = _report([_generation(MAX_LENGTH, 0.5, lf=0)])
    del report["per_item"][0]["endstep_probe_v22"][0]["lf_trace_v29"]
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8", newline="\n")
    rc = counter.main(["--report", str(path)])
    assert rc == 2
    assert "refused" in capsys.readouterr().out


def test_moving_artifact_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """2026-10-03 那枚中途件就是这个形状：探针期间被训练器改写 ⇒ 不许拿去判 A-2。"""

    report = _report([_generation(MAX_LENGTH, 0.5, lf=0)], unchanged=False)
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 2
    assert entry["status"] == "artifact_changed_during_probe"


def test_items_sha_mismatch_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, 0.5, lf=0)])
    rc, entry = _run(tmp_path, report, capsys, extra=("--expect-items-sha", "0" * 16))
    assert rc == 2
    assert entry["status"] == "items_sha_mismatch"


def test_ok_path_reports_below_line_as_a_conclusion(tmp_path: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    """够不到 ≥20 那条线是**读数**，不是仪器故障：status=ok 且 meets_frozen_line=False。"""

    report = _report([_generation(MAX_LENGTH, 0.30, lf=0), _generation(MAX_LENGTH, 0.02, lf=0)])
    rc, entry = _run(tmp_path, report, capsys)
    assert rc == 0
    assert entry["status"] == "ok"
    assert entry["eaters_above_floor"] == 1
    assert entry["meets_frozen_line"] is False


def test_out_report_is_lf_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = _report([_generation(MAX_LENGTH, 0.30, lf=0)])
    out = tmp_path / "counts.json"
    rc, _ = _run(tmp_path, report, capsys, extra=("--out-report", str(out)))
    assert rc == 0
    assert b"\r\n" not in out.read_bytes()
    assert json.loads(out.read_text(encoding="utf-8"))["results"][0]["eaters_above_floor"] == 1


def test_the_sealed_baseline_reproduces_the_frozen_zero() -> None:
    """硬锚点：§102 冻的 A-2 现值是 **0**（从不发 LF 那一群）。这台仪器在已入库的装机底基线件上必须读出 0，
    且那一群的规模必须等于件里自述的 236——否则要么人群口径错，要么 §102 的"现值"引用的是别的装配。
    """

    assert SEALED_BASELINE.is_file(), f"锚点件不在库里（{SEALED_BASELINE.name}），先查它去哪了"
    report = counter.read_report(SEALED_BASELINE)
    result = counter.count_eaters(report, floor=counter.FROZEN_FLOOR)
    assert result["never_lf_eaters_counted"] == 236
    assert result["eaters_above_floor"] == 0
    assert result["coherent_with_declaration"] is True
    assert result["coherent_with_never_lf_declaration"] is True
    # 宽口径只许比窄口径大，两数不许互换（今天装机底上它是 8，那是"发过 LF 的拖写代"）
    assert result["eaters_above_floor_all_eaters"] >= result["eaters_above_floor"]
