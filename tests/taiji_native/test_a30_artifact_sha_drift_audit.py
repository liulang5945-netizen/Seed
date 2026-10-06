"""`audit_taiji_artifact_sha_drift.py` 的守卫：分类必须**能为 false**，且两条已知锚点必须显形。

设计意图（`DEBT-G35` 的 ③）：`sha_drift` 这类损失按 `git ls-files` 与 markdown 链接审计都查不出来——
路径是对的、字节换了。唯一能抓它的方法是**把件里的 sha 与盘上重算的 sha 再比一遍**。
本文件钉的是分类逻辑本身，不是某一次的计数（计数会随收束与重训自然变化）。

两支已知锚点是**硬钉**的（件不在库就 raise，不 skip）：
`79b1a99cedf3…`（§2ai–§2an 那 10 份件记的重训臂）必须以非 `ok` 显形——这是我们已独立核过的真实损失；
`ca262807…`（装机基底）必须**不出现**在任何非 `ok` 条目里——它是当场 sha 相符的那一枚。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_REL = "scripts/training/audit_taiji_artifact_sha_drift.py"
TOOL = PROJECT_ROOT / TOOL_REL
LOST_ARM_SHA = "79b1a99cedf3e65bc76b0e40f4083f76ced62b4d070d5adb5a7ec2f725579a93"
GOOD_BASE_SHA = "ca2628077b21bc4c9a6f2fc06ff410d70311b1218a30b8794ecfd7e5a027d8bb"


def _tool():
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
    import importlib

    return importlib.import_module("audit_taiji_artifact_sha_drift")


def test_scan_face_is_the_tracked_file() -> None:
    tracked = (
        subprocess.run(
            ["git", "ls-files", TOOL_REL], cwd=PROJECT_ROOT, capture_output=True, check=True
        )
        .stdout.decode()
        .strip()
    )
    assert tracked == TOOL_REL, f"仪器不在版本控制面上（扫到 {tracked!r}）"


def _write_report(reports: Path, name: str, payload: dict) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    (reports / name).write_text(json.dumps(payload), encoding="utf-8")


def test_four_classifications_on_synthetic_reports(tmp_path: Path) -> None:
    tool = _tool()
    reports = tmp_path / "reports"
    artifact = tmp_path / "ckpt.pt"
    artifact.write_bytes(b"alpha-bytes")
    good = hashlib.sha256(artifact.read_bytes()).hexdigest()

    _write_report(
        reports, "ok.json", {"checkpoint": artifact.as_posix(), "checkpoint_sha256": good}
    )
    _write_report(
        reports, "drift.json", {"checkpoint": artifact.as_posix(), "checkpoint_sha256": "0" * 64}
    )
    _write_report(
        reports,
        "gone.json",
        {"checkpoint": (tmp_path / "nope.pt").as_posix(), "checkpoint_sha256": good},
    )
    _write_report(reports, "unpaired.json", {"model_reality": {"default_sha256": good}})
    #: 同 dict 里两枚路径＋一枚 sha：词根配不上时**不许**笛卡尔硬配（那正是第一版误报 31 条的形状）
    other = tmp_path / "other.pt"
    other.write_bytes(b"beta")
    _write_report(
        reports,
        "ambiguous.json",
        {
            "retrain_checkpoint": artifact.as_posix(),
            "base_checkpoint": other.as_posix(),
            "checkpoint_sha256_before": "0" * 64,
        },
    )

    payload = tool.audit(reports)
    verdicts = {
        c["report"].split(".")[0]: (c["verdict"], c["path_key"]) for c in payload["non_ok_claims"]
    }
    counts = payload["verdict_counts"]
    #: 保守配对的代价要写清：`ambiguous.json` 里两枚路径配不出一枚 sha ⇒ **宁可不配也不硬配**，
    #: 它落进 `unpaired` 计数（可见），而不是变成一条假 `sha_drift`。第一版就是把这里硬配成 31 条误报的。
    assert counts.get("sha_drift", 0) == 1, payload
    assert counts.get("ok", 0) == 1, payload
    assert verdicts["gone"][0] == "missing_file", verdicts
    assert payload["sha_claims_unpaired_with_path"] >= 2, payload
    assert all(c["resolved_path"] != other.as_posix() for c in payload["non_ok_claims"])


def test_cross_level_arm_pairing_is_not_silently_dropped(tmp_path: Path) -> None:
    """本仓约定：路径写在外层、sha 写在臂块里。只看同一个 dict 就会把 §2ai–§2an 整批静默丢掉。"""

    tool = _tool()
    reports = tmp_path / "reports"
    artifact = tmp_path / "boundary" / "checkpoint.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"current-bytes")
    _write_report(
        reports,
        "transfer.json",
        {
            "retrain_checkpoint": artifact.as_posix(),
            "runs": {"retrain": {"checkpoint_sha256_before": "f" * 64}},
        },
    )
    payload = tool.audit(reports)
    hit = [c for c in payload["non_ok_claims"] if c["path_key"] == "retrain_checkpoint"]
    assert len(hit) == 1 and hit[0]["verdict"] == "sha_drift", payload


def test_known_bad_anchor_is_visible_and_known_good_is_clean() -> None:
    """真面上的两支锚点：丢了的那枚必须显形，在场的那枚不许被误报。"""

    tool = _tool()
    if not (PROJECT_ROOT / "reports").is_dir():
        raise AssertionError("reports/ 不在库里，本支没有可核面")
    payload = tool.audit(PROJECT_ROOT / "reports")
    bad = [c for c in payload["non_ok_claims"] if c["recorded_sha16"] == LOST_ARM_SHA[:16]]
    assert bad, "已知-bad 锚点 79b1a99c… 没显形 ⇒ 本仪器的配对面有洞"
    assert all(c["verdict"] in ("sha_drift", "missing_file") for c in bad)
    wrongly_flagged = [
        c for c in payload["non_ok_claims"] if c["recorded_sha16"] == GOOD_BASE_SHA[:16]
    ]
    assert not wrongly_flagged, wrongly_flagged[:3]
    assert payload["verdict_counts"].get("ok", 0) > 100, payload["verdict_counts"]


def test_own_output_is_not_scanned_back_in(tmp_path: Path) -> None:
    """自喂防护：本件的输出里 `resolved_path` 与 `recorded_sha16` 成对，下一趟会把**自己的判定**
    当成新的"路径＋sha 声明"再扫一遍 ⇒ 计数随运行次数膨胀。这类件必须被排除且被计数可见。"""

    tool = _tool()
    reports = tmp_path / "reports"
    artifact = tmp_path / "ckpt.pt"
    artifact.write_bytes(b"alpha-bytes")
    good = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _write_report(
        reports, "real.json", {"checkpoint": artifact.as_posix(), "checkpoint_sha256": good}
    )
    _write_report(
        reports,
        "prev_run.json",
        {
            "format": "taiji-artifact-sha-drift-v1",
            "non_ok_claims": [{"resolved_path": artifact.as_posix(), "recorded_sha16": "0" * 16}],
        },
    )
    payload = tool.audit(reports)
    assert payload["own_outputs_excluded"] == 1, payload
    assert payload["claims_total"] == 1, payload
    assert all(c["report"] == "real.json" for c in payload["non_ok_claims"]), payload


def test_refuses_an_existing_target_and_requires_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _tool()
    target = tmp_path / "drift.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["audit.py", "--out-report", str(target)])
    assert tool.main(["--out-report", str(target)]) == 2
    assert "拒绝落盘" in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        tool.main([])
    assert caught.value.code == 2
