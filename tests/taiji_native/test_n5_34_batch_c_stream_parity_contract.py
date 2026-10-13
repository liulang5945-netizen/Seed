"""`PLAN-N5-21` 的 J-N5t-1 载体（`audit_taiji_batch_c_stream_parity.py`）的契约测。

这台仪器的判据是「两臂逐行同界」，而它**故意不判**的那一面（`window_symbols` 这类模型依赖量）
同样要钉住——否则下一批有人觉得"窗口数也该等"就把守卫改成了必然为假的那一支
（[[guard-must-be-able-to-fail]] 的反面：不判的那面也要有测证明它被出版、且不被当成判据）。

四支红/绿各自独立：同界为绿、边界挪一格必须红、少一行必须红、窗口量不等**不得**拖红。
另三支守件面：缺文件与空面都 rc=2 并点名路径，`--help` 必须走得动（argparse 那处历史坑），
覆盖已存在件默认拒写。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_batch_c_stream_parity.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_stream_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CMP = _load()


def _rows(count: int, session: str, window: int = 250_000) -> list[dict[str, Any]]:
    return [
        {
            "arm": "A",
            "session_id": session,
            "symbols_consumed": 250_000 * (index + 1),
            "symbols_budget": 250_000 * count,
            "absolute_tick": 1000 + 250_000 * (index + 1),
            "session_symbols": 250_000 * (index + 1),
            "window_symbols": window,
            "cumulative_online_accuracy": 0.25 + index / 100,
            "cumulative_mean_surprise": 7.0 - index / 100,
            "elapsed_seconds": 10.0 * (index + 1),
            "final": index == count - 1,
        }
        for index in range(count)
    ]


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    return path


def _run(tmp_path: Path, treated: Path, control: Path, name: str = "r.json") -> tuple[int, dict]:
    out = tmp_path / name
    rc = CMP.main(
        [
            "--treated-progress",
            str(treated),
            "--control-progress",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    return rc, json.loads(out.read_text(encoding="utf-8"))


def test_identical_boundaries_hold_and_publish_the_curve(tmp_path: Path) -> None:
    treated = _write(tmp_path / "t.jsonl", _rows(4, "t"))
    control = _write(tmp_path / "c.jsonl", _rows(4, "c"))
    rc, payload = _run(tmp_path, treated, control)
    assert rc == 0
    assert payload["format"] == "taiji-batch-c-stream-parity-v1"
    assert payload["criterion"] == "J-N5t-1"
    assert payload["verdict"] == "holds"
    assert payload["failing_verdicts"] == []
    assert payload["rows"] == {"treated": 4, "control": 4}
    assert payload["first_mismatch_row"] is None
    #: 两个 session 各自一枚值：`session_id` 里带着日期，两臂必然不同名，判的是"每臂内部只有一枚"。
    assert len(payload["sessions"]["treated"]) == 1
    assert len(payload["sessions"]["control"]) == 1
    #: 曲线是这台仪器的副产物：㊵-758 那两个点在这里变成逐行并排。
    assert len(payload["metric_curve"]) == 4
    assert payload["metric_curve"][0]["treated_cumulative_mean_surprise"] == pytest.approx(7.0)
    assert payload["metric_curve"][0]["delta_cumulative_mean_surprise"] == pytest.approx(0.0)
    assert "同流不等于同实现" in payload["reading_limit"]


def test_moving_one_boundary_row_by_one_symbol_must_go_red(tmp_path: Path) -> None:
    treated_rows = _rows(4, "t")
    control_rows = _rows(4, "c")
    control_rows[2]["symbols_consumed"] += 1
    treated = _write(tmp_path / "t.jsonl", treated_rows)
    control = _write(tmp_path / "c.jsonl", control_rows)
    rc, payload = _run(tmp_path, treated, control, "red.json")
    assert rc == 1
    assert payload["verdict"] == "rejected"
    assert payload["mismatched_boundary_keys"] == ["symbols_consumed"]
    assert payload["first_mismatch_row"] == 2
    assert payload["failing_verdicts"] == ["boundary_keys_identical_on_aligned_rows"]


def test_a_missing_row_makes_both_count_and_boundary_verdicts_red(tmp_path: Path) -> None:
    #: 少一行＝那臂早停或被截断：只判"行数相等"会漏掉"对齐后仍有差"这半句，两支必须同时红。
    treated = _write(tmp_path / "t.jsonl", _rows(4, "t"))
    control = _write(tmp_path / "c.jsonl", _rows(3, "c"))
    rc, payload = _run(tmp_path, treated, control, "short.json")
    assert rc == 1
    assert payload["rows"] == {"treated": 4, "control": 3}
    assert set(payload["failing_verdicts"]) == {
        "row_counts_equal",
        "boundary_keys_identical_on_aligned_rows",
    }


def test_window_symbols_may_differ_without_dragging_the_verdict_red(tmp_path: Path) -> None:
    #: `window_symbols = seen - mark[0]` 是模型依赖量：两臂不等是**结果**。
    #: 这一支钉的是"不判的那一面确实被出版，而且没有偷偷参与判决"。
    treated_rows = _rows(4, "t")
    control_rows = _rows(4, "c", window=249_998)
    treated = _write(tmp_path / "t.jsonl", treated_rows)
    control = _write(tmp_path / "c.jsonl", control_rows)
    rc, payload = _run(tmp_path, treated, control, "window.json")
    assert rc == 0
    assert payload["verdict"] == "holds"
    face = payload["window_face"]
    assert face["rows_compared"] == 4
    assert face["unequal_rows"] == 4
    assert face["treated_minus_control"] == {"min": 2, "max": 2}
    assert "window_symbols" not in json.dumps(payload["verdicts"])
    assert payload["metric_curve"][0]["delta_window_symbols"] == 2


def test_two_sessions_in_one_arm_go_red(tmp_path: Path) -> None:
    #: 跨进程追加过的进度流会让"按相邻行差算速率"算出负数（历史踩过），所以会话基数是判据。
    rows = _rows(4, "t")
    rows[3]["session_id"] = "t-resumed-9"
    treated = _write(tmp_path / "t.jsonl", rows)
    control = _write(tmp_path / "c.jsonl", _rows(4, "c"))
    rc, payload = _run(tmp_path, treated, control, "sessions.json")
    assert rc == 1
    assert payload["failing_verdicts"] == ["single_session_per_arm"]
    assert len(payload["sessions"]["treated"]) == 2


def test_missing_or_empty_stream_refuses_loudly(tmp_path: Path) -> None:
    treated = _write(tmp_path / "t.jsonl", _rows(2, "t"))
    gone = tmp_path / "never-was.jsonl"
    rc = CMP.main(
        [
            "--treated-progress",
            str(treated),
            "--control-progress",
            str(gone),
            "--out-report",
            str(tmp_path / "x.json"),
        ]
    )
    assert rc == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8", newline="\n")
    rc_empty = CMP.main(
        [
            "--treated-progress",
            str(treated),
            "--control-progress",
            str(empty),
            "--out-report",
            str(tmp_path / "y.json"),
        ]
    )
    assert rc_empty == 2


def test_an_unparseable_line_names_the_line_number(tmp_path: Path, capsys) -> None:
    treated = _write(tmp_path / "t.jsonl", _rows(2, "t"))
    broken = tmp_path / "c.jsonl"
    broken.write_text(
        json.dumps(_rows(2, "c")[0]) + "\n{not json}\n", encoding="utf-8", newline="\n"
    )
    rc = CMP.main(
        [
            "--treated-progress",
            str(treated),
            "--control-progress",
            str(broken),
            "--out-report",
            str(tmp_path / "z.json"),
        ]
    )
    assert rc == 2
    captured = capsys.readouterr()
    #: 坏行要点名到**哪一行**，否则"读不到"与"读到一半断了"会被混成同一条错误。
    assert "c.jsonl:2" in captured.err


def test_help_walks_and_refuses_to_overwrite(tmp_path: Path) -> None:
    #: argparse 的 help 里带裸 `%` 会让 `--help` 直接崩（全仓踩过四次），这里冒烟一次。
    with pytest.raises(SystemExit) as exit_info:
        CMP.main(["--help"])
    assert exit_info.value.code == 0
    #: 面变了才重记；默认不许覆盖别人（或上一批）的件。
    treated = _write(tmp_path / "t.jsonl", _rows(2, "t"))
    control = _write(tmp_path / "c.jsonl", _rows(2, "c"))
    out = tmp_path / "sealed.json"
    out.write_text("{}\n", encoding="utf-8", newline="\n")
    rc = CMP.main(
        [
            "--treated-progress",
            str(treated),
            "--control-progress",
            str(control),
            "--out-report",
            str(out),
        ]
    )
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "{}\n"
