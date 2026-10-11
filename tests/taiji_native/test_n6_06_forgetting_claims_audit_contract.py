"""`audit_taiji_episodic_forgetting_claims.py`（DEBT-G98/G99 配套，J-N2f-4 的改版测量式）的契约测。

改版原因不是洁癖：原判据"现扫『自动遗忘』字样，命中即违规"在两天里**自指命中两次**，
第二次命中的正是我自己那份禁令件 `PLAN-N2-06`（它抄写该词来禁止它）⇒ 排除单份文档的写法不成立。
本册因此同时钉住两支：肯定式必须被抓到（正例）、禁令本体不得被抓到（反例）。
只跑一支等于没测——这与"判读器两条分支都要走"同族。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_episodic_forgetting_claims.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO,
    )


def _scan(tmp_path: Path, lines: list[str]) -> dict[str, Any]:
    face = tmp_path / "plans"
    face.mkdir(parents=True, exist_ok=True)
    (face / "PLAN-N9-00_x_20261011.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    out = tmp_path / "report.json"
    proc = _run(["--root", str(tmp_path), "--face", "plans", "--out-report", str(out)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_affirmative_claim_is_flagged(tmp_path: Path) -> None:
    payload = _scan(tmp_path, ["# 判读", "", "本装配会自动遗忘过期经历。"])
    assert payload["violation_count"] == 1, payload
    assert payload["verdicts"]["no_affirmative_forgetting_claims"] is False


def test_the_prohibition_itself_is_not_a_violation(tmp_path: Path) -> None:
    payload = _scan(tmp_path, ["# 禁令", "", "E 删除与 θ 删除不是同一操作，不能承诺自动遗忘。"])
    assert payload["violation_count"] == 0, payload
    assert payload["verdicts"]["no_affirmative_forgetting_claims"] is True


def test_negated_capability_claim_is_not_a_violation(tmp_path: Path) -> None:
    payload = _scan(
        tmp_path, ["# 边界", "", "本仓未实现 automatic_forgetting，不得声称 supported。"]
    )
    assert payload["violation_count"] == 0, payload


def test_two_affirmative_claims_both_count(tmp_path: Path) -> None:
    payload = _scan(tmp_path, ["a", "系统会自动遗忘旧证据", "b", "该模块已经遗忘这条经历"])
    assert payload["violation_count"] == 2, payload


def test_empty_face_is_refused_not_read_as_green(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    proc = _run(["--root", str(tmp_path), "--face", "nothing_here", "--out-report", str(out)])
    assert proc.returncode == 2, proc.stdout
    assert "scan_face_empty" in proc.stdout
    assert not out.exists(), "refusal must not leave a half-written report"


def test_real_tree_has_no_affirmative_claim() -> None:
    out = REPO / "reports/taiji_episodic_forgetting_claims_20261011.json"
    proc = _run(["--out-report", str(out)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["files_scanned"] > 100, payload["files_scanned"]
    assert payload["violation_count"] == 0, payload["violations"]
