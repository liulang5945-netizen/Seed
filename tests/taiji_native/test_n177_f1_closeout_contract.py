"""F1 结案收口器的判据册：12 run 的账、端点 sha、旧保持 3pp、attempt 唯一、判定不许手改。

全部用合成夹具，不碰真 run root；每条拒绝路径各一支。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT = PROJECT_ROOT / "scripts" / "training" / "closeout_taiji_n1_plan177_f1.py"
ARMS = ("main_credit_on", "main_credit_stop", "copy_only", "no_state")


def _module():
    spec = importlib.util.spec_from_file_location("n177_f1_closeout", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def instrument():
    return _module()


def _passing_scores() -> dict:
    return {
        "answer_exact": 0.9,
        "value_exact": 0.95,
        "state_exact": 0.95,
        "relevant_swap_delta": 0.6,
        "irrelevant_swap_keep": 0.99,
        "cut_drop": 0.5,
        "rescue_gain": 0.5,
        "main_minus_copy_only": 0.4,
        "main_minus_no_state": 0.5,
        "credit_gap": 0.3,
        "copy_only_median": 0.5,
        "no_state_median": 0.4,
    }


def _fixture(tmp_path, *, drop_pp: float = 1.0, verdict_label: str = "PASS_LEARNED_VALUE"):
    run_root = tmp_path / "runs"
    for seed in (0, 1, 2):
        for arm in ARMS:
            directory = run_root / f"{arm}__seed_{seed}"
            directory.mkdir(parents=True)
            endpoint = directory / "endpoint.pt"
            endpoint.write_bytes(f"{arm}-{seed}".encode())
            state = {
                "status": "completed",
                "updates": 1280,
                "process_cpu_seconds": 1000.0,
                "wall_seconds": 900.0,
                "endpoint": str(endpoint),
                "endpoint_sha256": instrument_sha(endpoint),
            }
            (directory / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    retention = tmp_path / "retention.json"
    retention.write_text(json.dumps({"worst_drop_pp": drop_pp}), encoding="utf-8")
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps(
            {"scores": _passing_scores(), "decision": {"verdict": verdict_label}, "gates": {}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "holdout.attempt.json").write_text("{}", encoding="utf-8")
    return run_root, retention, holdout, tmp_path / "closeout.json"


def instrument_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(instrument, report, run_root, retention, holdout) -> int:
    return instrument.main(
        [
            "--run-root",
            str(run_root),
            "--retention",
            str(retention),
            "--holdout",
            str(holdout),
            "--report",
            str(report),
        ]
    )


def test_closeout_writes_when_everything_agrees(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path)
    assert _run(instrument, report, run_root, retention, holdout) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS_LEARNED_VALUE"
    assert payload["cost"]["within_cap"] is True
    assert len(payload["runs"]) == 12


def test_a_missing_run_refuses(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path)
    (run_root / "copy_only__seed_2" / "run_state.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing F1 run state"):
        _run(instrument, report, run_root, retention, holdout)
    assert not report.exists()


def test_an_incomplete_run_refuses(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path)
    state_path = run_root / "main_credit_on__seed_0" / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="not completed"):
        _run(instrument, report, run_root, retention, holdout)


def test_an_endpoint_sha_mismatch_refuses(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path)
    (run_root / "no_state__seed_1" / "endpoint.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha differs"):
        _run(instrument, report, run_root, retention, holdout)


def test_too_large_a_retention_drop_refuses(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path, drop_pp=4.0)
    with pytest.raises(ValueError, match="3pp"):
        _run(instrument, report, run_root, retention, holdout)


def test_a_hand_edited_verdict_is_caught(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path, verdict_label="FAIL_COPY_ONLY")
    with pytest.raises(ValueError, match="recorded verdict"):
        _run(instrument, report, run_root, retention, holdout)


def test_zero_or_two_attempts_refuse(instrument, tmp_path) -> None:
    run_root, retention, holdout, report = _fixture(tmp_path)
    (tmp_path / "holdout.attempt.json").unlink()
    with pytest.raises(ValueError, match="exactly one attempt"):
        _run(instrument, report, run_root, retention, holdout)
