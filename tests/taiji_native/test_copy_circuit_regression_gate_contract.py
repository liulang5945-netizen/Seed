"""§4.3 回归门比对器的合同（零模型，纯合成报告）。

这条门判的是"装上复制回路有没有让既有能力变差"，所以它最容易出的两类错是
**把不可比的两个读数判成通过**，和**把没测的东西算成通过**——两类都有反向断言。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "score_taiji_r2_copy_circuit_regression_gate.py"

CHAIN_CONTROL = {"relax_legacy_guard": True, "constrained_decode": True}


def _report(
    *,
    dims: dict[str, Any],
    chain: dict[str, Any],
    git_head: str = "head-A",
    checkpoint_sha: str = "cksum-A",
    eval_set_sha: str = "eval-A",
) -> dict[str, Any]:
    return {
        "format": "taiji-cap0-baseline-v2",
        "identity": {
            "git_head": git_head,
            "checkpoint_sha256": checkpoint_sha,
            "eval_set_sha256": eval_set_sha,
        },
        "checkpoint": "checkpoints/seed_beta.pt",
        "trained_during_eval": False,
        "chain": chain,
        "dimensions": dims,
    }


def _driven(normalised: float, outputs: list[str]) -> dict[str, Any]:
    return {
        "name": "X",
        "tally": {"machine_normalised": normalised},
        "items": [
            {"id": f"I{n}", "turns": [{"raw_output": text}]} for n, text in enumerate(outputs)
        ],
    }


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, control: Path, treated: Path, *extra: str) -> dict[str, Any]:
    """在进程内调 `main()`——比对器是纯读数逻辑，没必要每条用例重启一次解释器。"""

    from scripts.training import score_taiji_r2_copy_circuit_regression_gate as gate

    out = tmp_path / f"gate-{len(list(tmp_path.glob('gate*.json')))}.json"
    argv = [
        "score_taiji_r2_copy_circuit_regression_gate.py",
        "--control",
        str(control),
        "--treated",
        str(treated),
        "--out-report",
        str(out),
        *extra,
    ]
    with patch.object(sys, "argv", argv):
        assert gate.main() == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _pair(tmp_path: Path, b_control: float, b_treated: float) -> tuple[Path, Path]:
    texts = ["阿岩住在杭州。", "结果是 42。"]
    control = _write(
        tmp_path / "control.json",
        _report(
            dims={
                "B": _driven(b_control, texts),
                "C": _driven(0.0, texts),
                "D": _driven(0.0, texts),
            },
            chain=dict(CHAIN_CONTROL),
        ),
    )
    treated = _write(
        tmp_path / "treated.json",
        _report(
            dims={
                "B": _driven(b_treated, texts),
                "C": _driven(0.0, texts),
                "D": _driven(0.25, texts),
            },
            chain={**CHAIN_CONTROL, "copy_circuit": "circuit-final.pt"},
        ),
    )
    return control, treated


def test_missing_judged_dimension_blocks_the_gate(tmp_path: Path) -> None:
    """判据里有维没跑 ⇒ 门不完整。绝不因为"跑的那几维都平"就叫通过。"""

    control, treated = _pair(tmp_path, 0.5, 0.5)
    stripped = json.loads(control.read_text(encoding="utf-8"))
    stripped["dimensions"].pop("C")
    control = _write(tmp_path / "control-noC.json", stripped)
    report = _run(tmp_path, control, treated, *_health_args(tmp_path))
    assert report["judged"]["C"]["status"] == "not_executed"
    assert "未通过" in report["verdict"]


def _healthy_pair(tmp_path: Path) -> tuple[Path, Path]:
    """一对 A 维健康报告：两臂所有真实性检查都为真。"""

    checks = {"A01_new_process_load": True, "A04_input_changes_output": True}
    control = _write(
        tmp_path / "health_control.json",
        _report(
            dims={"A": {"name": "模型真实性", "checks": dict(checks)}}, chain=dict(CHAIN_CONTROL)
        ),
    )
    treated = _write(
        tmp_path / "health_treated.json",
        _report(
            dims={"A": {"name": "模型真实性", "checks": dict(checks)}},
            chain={**CHAIN_CONTROL, "copy_circuit": "c.pt"},
        ),
    )
    return control, treated


def _health_args(tmp_path: Path) -> list[str]:
    control, treated = _healthy_pair(tmp_path)
    return ["--control-health", str(control), "--treated-health", str(treated)]


def test_comparable_and_equal_scores_close_the_gate(tmp_path: Path) -> None:
    control, treated = _pair(tmp_path, 0.5, 0.5)
    report = _run(tmp_path, control, treated, *_health_args(tmp_path))
    assert report["comparability"]["comparable"] is True
    assert report["judged"]["B"]["status"] == "ok"
    assert report["judged"]["A"]["status"] == "ok"
    # D 维只是观察项：它涨了不能替 B 作证，判据只认 A/B/C。
    assert report["observed_not_judged"]["D"]["treated"]["machine_normalised"] == 0.25
    assert "通过" in report["verdict"] and "未通过" not in report["verdict"]


def test_a_missing_health_pair_is_not_counted_as_pass(tmp_path: Path) -> None:
    """没跑 A 维 ⇒ 门不完整，绝不能因为 B/C 平了就叫"通过"。"""

    control, treated = _pair(tmp_path, 0.5, 0.6)
    report = _run(tmp_path, control, treated)
    assert report["judged"]["A"]["status"] == "not_executed"
    assert report["judged"]["A"]["counts_as_pass"] is False
    assert "未通过" in report["verdict"]


def test_regression_in_judged_dimension_is_reported_not_smoothed(tmp_path: Path) -> None:
    control, treated = _pair(tmp_path, 0.5, 0.4)
    report = _run(tmp_path, control, treated)
    assert report["judged"]["B"]["status"] == "regressed"
    assert report["judged"]["B"]["control"] == 0.5
    assert report["judged"]["B"]["treated"] == 0.4
    assert "未通过" in report["verdict"]


def test_control_arm_that_also_mounted_a_circuit_is_incomparable(tmp_path: Path) -> None:
    """两臂其实同链（对照也挂了电路）＝没有对照，必须拒绝判决而不是给个"通过"。"""

    control = _write(
        tmp_path / "control.json",
        _report(
            dims={"B": _driven(0.5, ["答。"])},
            chain={**CHAIN_CONTROL, "copy_circuit": "circuit-final.pt"},
        ),
    )
    treated = _write(
        tmp_path / "treated.json",
        _report(
            dims={"B": _driven(0.5, ["答。"])},
            chain={**CHAIN_CONTROL, "copy_circuit": "circuit-final.pt"},
        ),
    )
    report = _run(tmp_path, control, treated)
    assert report["comparability"]["comparable"] is False
    assert any("默认链路" in defect for defect in report["comparability"]["defects"])


def test_identity_drift_between_arms_blocks_the_verdict(tmp_path: Path) -> None:
    control, treated = _pair(tmp_path, 0.5, 0.9)
    drifted = json.loads(treated.read_text(encoding="utf-8"))
    drifted["identity"]["git_head"] = "head-B"
    _write(treated, drifted)
    report = _run(tmp_path, control, treated)
    assert report["comparability"]["comparable"] is False
    assert any("git_head" in defect for defect in report["comparability"]["defects"])
    assert "未通过" in report["verdict"]


def test_surface_degradation_fails_the_gate_even_with_flat_scores(tmp_path: Path) -> None:
    """分数没动但表层变差（可解码/成句）也是劣化——表层这条是独立的一判。"""

    texts_control = ["阿岩住在杭州。", "结果是 42。"]
    texts_treated = ["\ufffd\ufffd\ufffd\ufffd", "\ufffd\ufffd\ufffd"]
    control = _write(
        tmp_path / "control.json",
        _report(dims={"B": _driven(0.5, texts_control)}, chain=dict(CHAIN_CONTROL)),
    )
    treated = _write(
        tmp_path / "treated.json",
        _report(
            dims={"B": _driven(0.5, texts_treated)},
            chain={**CHAIN_CONTROL, "copy_circuit": "c.pt"},
        ),
    )
    report = _run(tmp_path, control, treated, *_health_args(tmp_path))
    assert report["surface"]["control"]["utf8_decodable_rate"] == 1.0
    assert report["surface"]["treated"]["utf8_decodable_rate"] == 0.0
    assert report["judged"]["B"]["status"] == "ok"
    assert report["surface_not_worse"] is False
    assert "未通过" in report["verdict"]


def test_a_dimension_loses_a_true_check_and_is_caught(tmp_path: Path) -> None:
    """A 维是布尔检查：对照里为真的项治疗里掉了，必须记 regressed（不能被 tally 缺失放掉）。"""

    control, treated = _pair(tmp_path, 0.5, 0.5)
    health_control = _write(
        tmp_path / "hc.json",
        _report(
            dims={
                "A": {
                    "name": "模型真实性",
                    "checks": {"A01_new_process_load": True, "A04_input_changes_output": True},
                }
            },
            chain=dict(CHAIN_CONTROL),
        ),
    )
    health_treated = _write(
        tmp_path / "ht.json",
        _report(
            dims={
                "A": {
                    "name": "模型真实性",
                    "checks": {"A01_new_process_load": True, "A04_input_changes_output": False},
                }
            },
            chain={**CHAIN_CONTROL, "copy_circuit": "c.pt"},
        ),
    )
    report = _run(
        tmp_path,
        control,
        treated,
        "--control-health",
        str(health_control),
        "--treated-health",
        str(health_treated),
    )
    assert report["judged"]["A"]["status"] == "regressed"
    assert report["judged"]["A"]["lost"] == ["A04_input_changes_output"]
    assert "未通过" in report["verdict"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
