"""DEBT-G91 的读数面契约测（`audit_taiji_stale_gate_reference_census.py`）。

这台仪器只回答一件事：一条"上一批门仍绿"的判据，它的绿是**当场跑的**还是**从封存件里读的**。
两档都必须能为空、也都能不为空——所以测里给合成样例，两形各造一次；
再加上"什么都没扫到就 rc=2"，防 glob 写窄把"零违例"读成假绿（[[absence-prove-by-enumeration-not-guessed-name]]）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_stale_gate_reference_census.py"
SEALED_REPORT = REPO / "reports" / "taiji_stale_gate_reference_census_20261010.json"

SEALED_SNIPPET = """
def evaluate():
    return {
        "previous_gates_remain_green": all(
            json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))["gate"]["passed"]
            for name in ("taiji_w7_p2_11_x_20260831.json", "taiji_w7_p2_12_y_20260831.json")
        ),
    }
"""

LIVE_SNIPPET = """
def evaluate():
    report = run_chain()
    return {"gate_ok": report["gate"]["passed"]}
"""


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("stale_gate_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CENSUS = _load()


def _write(root: Path, name: str, source: str) -> None:
    (root / name).write_text(source, encoding="utf-8", newline="\n")


def test_a_sealed_artifact_read_is_bucketed_as_sealed(tmp_path: Path) -> None:
    #: 关键形状：件名住在推导式的 `iter` 里，不在 `["gate"]` 下标自己的子树里——
    #: 只看下标本身会把这两处漏成 live（我第一版正是这样，现读改掉的）。
    _write(tmp_path, "sealed.py", SEALED_SNIPPET)
    rows = CENSUS.scan_file(tmp_path / "sealed.py")
    assert len(rows) == 1, rows
    assert rows[0]["bucket"] == "sealed_artifact"
    assert rows[0]["reads_file_chain"] is True
    assert "taiji_w7_p2_11_x_20260831.json" in rows[0]["artifact_names"]


def test_a_live_gate_read_is_not_bucketed_as_sealed(tmp_path: Path) -> None:
    _write(tmp_path, "live.py", LIVE_SNIPPET)
    rows = CENSUS.scan_file(tmp_path / "live.py")
    assert [row["bucket"] for row in rows] == ["live_object"]


def test_both_forms_in_one_directory_are_counted_separately(tmp_path: Path) -> None:
    _write(tmp_path, "sealed.py", SEALED_SNIPPET)
    _write(tmp_path, "live.py", LIVE_SNIPPET)
    rc = CENSUS.main(["--scan-root", str(tmp_path), "--out-report", str(tmp_path / "r.json")])
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["files_scanned"] == 2
    assert payload["sealed_artifact_sites"] == 1
    assert payload["live_object_sites"] == 1


def test_an_empty_scan_root_refuses_instead_of_reporting_zero_violations(tmp_path: Path) -> None:
    #: 零命中要连范围一起报：目录里没有 .py 时"零违例"是假绿，必须 rc=2。
    rc = CENSUS.main(["--scan-root", str(tmp_path)])
    assert rc == 2


def test_a_missing_scan_root_is_named(tmp_path: Path) -> None:
    rc = CENSUS.main(["--scan-root", str(tmp_path / "nope")])
    assert rc == 2


def test_unparsable_python_is_counted_not_silently_dropped(tmp_path: Path) -> None:
    _write(tmp_path, "broken.py", "def broken(:\n")
    rc = CENSUS.main(["--scan-root", str(tmp_path), "--out-report", str(tmp_path / "r.json")])
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["unparsed_file_count"] == 1, payload


def test_real_repo_face_publishes_four_sealed_sites_over_five_gates() -> None:
    #: DEBT-G91 的行为可复算而写：四处命中、被引用的是五枚 W7/P2 门（去掉 `reports` 目录常量那枚标记）。
    payload = json.loads(SEALED_REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["sealed_artifact_sites"] == 4
    names = {
        name
        for row in payload["sealed_reads"]
        for name in row["artifact_names"]
        if name.endswith(".json")
    }
    assert len(names) == 5, sorted(names)
    assert "taiji_w7_p2_11_ide_language_chain_20260831.json" in names
    files = {row["file"] for row in payload["sealed_reads"]}
    assert len(files) == 3, files


def test_help_lists_both_flags() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout is not None
    assert "--scan-root" in proc.stdout and "--out-report" in proc.stdout


#: ── v2（㊵-785）：合取只认 `.json` 时，读 `progress.jsonl` 的门会被**静默分进 live_object**，
#: 那是本器唯一往"绿是当场跑的"方向低报的形态。两条负对照各钉一个方向。
JSONL_SNIPPET = """
def evaluate():
    return {
        "previous_runs_remain_green": all(
            json.loads((ROOT / "output" / name).read_text(encoding="utf-8"))["gate"]["passed"]
            for name in ("arm_a_progress.jsonl", "arm_b_progress.jsonl")
        ),
    }
"""

UNNAMED_SNIPPET = """
def evaluate(progress_path):
    return {"gate_ok": json.loads(progress_path.read_text(encoding="utf-8"))["gate"]["passed"]}
"""


def test_a_jsonl_gate_read_is_bucketed_as_sealed(tmp_path: Path) -> None:
    #: 这里刻意用 `"output"` 而不是 `"reports"` 常量——v1 只靠 `.json` 字面量那一支，
    #: 于是这一处会读成 live；v2 必须把它读成 sealed。
    _write(tmp_path, "runs.py", JSONL_SNIPPET)
    rows = CENSUS.scan_file(tmp_path / "runs.py")
    assert len(rows) == 1, rows
    assert rows[0]["bucket"] == "sealed_artifact"
    assert rows[0]["reads_file_chain"] is True
    assert rows[0]["artifact_names"] == ["arm_a_progress.jsonl", "arm_b_progress.jsonl"]


def test_an_unnamed_file_read_stays_live_but_is_published(tmp_path: Path) -> None:
    #: 反方向：件名住在参数里（点不出名）时**不许**被扩扩展名顺手升成 sealed——
    #: 它留在 live_object，但必须由 `chain_true_unnamed_sites` 单独出版，免得抬高"当场跑"那格。
    _write(tmp_path, "unnamed.py", UNNAMED_SNIPPET)
    rc = CENSUS.main(["--scan-root", str(tmp_path), "--out-report", str(tmp_path / "r.json")])
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["format"] == "taiji-stale-gate-reference-census-v2"
    assert payload["artifact_suffixes"] == [".json", ".jsonl"]
    assert payload["sealed_artifact_sites"] == 0
    assert payload["live_object_sites"] == 1
    assert payload["chain_true_unnamed_sites"] == 1
    row = payload["chain_true_unnamed_reads"][0]
    assert row["reads_file_chain"] is True
    assert row["artifact_names"] == []


#: 第三类取路（v2 支②）：读文件链不在下标这一侧，而在**同一文件里那个名字的赋值**上。
BOUND_SNIPPET = """
def evaluate():
    payload = json.loads(
        (ROOT / "reports" / "taiji_w7_p2_11_ide_language_chain_20260831.json").read_text(
            encoding="utf-8"
        )
    )
    return {"previous_gate_still_green": payload["gate"]["passed"]}
"""


def test_a_name_bound_file_read_is_bucketed_as_sealed(tmp_path: Path) -> None:
    #: 两支成对：`payload = json.loads(…read_text())` 必须读成 sealed；
    #: 而 LIVE_SNIPPET（`report = run_chain()`，同名形状、不读文件）必须仍是 live——
    #: 少了这一支，"名字有读文件赋值"就成了一台只会亮的机器。
    _write(tmp_path, "bound.py", BOUND_SNIPPET)
    rows = CENSUS.scan_file(tmp_path / "bound.py")
    assert len(rows) == 1, rows
    assert rows[0]["bucket"] == "sealed_artifact"
    assert rows[0]["reads_file_chain"] is True
    assert rows[0]["artifact_names"] == [
        "<reports-dir-constant>",
        "taiji_w7_p2_11_ide_language_chain_20260831.json",
    ]
    _write(tmp_path, "live.py", LIVE_SNIPPET)
    live_rows = CENSUS.scan_file(tmp_path / "live.py")
    assert [row["bucket"] for row in live_rows] == ["live_object"]
    assert live_rows[0]["reads_file_chain"] is False
