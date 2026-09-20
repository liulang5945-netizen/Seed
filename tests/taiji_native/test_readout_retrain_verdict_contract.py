"""Contract guard for the A/B/C verdict instrument (M1 + M2 + K2).

Why this exists: the verdict decides whether the readout side gets credit for the retrain, and the
one thing that must not happen is a verdict computed from arms that are not a valid matched triple
-- two arms, an arm that stopped short of the approved budget, or an arm trained from a different
substrate.  Those inputs would still produce numbers, and the numbers would still look like a
result.  So every one of them has to be refused in code, and each refusal is asserted here.

The other half is anti-post-hoc: K2's "no difference" threshold was not in the frozen §5, so it is
pre-registered in contract §5.1 *before any arm finished*.  That ordering is asserted too -- if
someone later moves the threshold, this file goes red.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
VERDICT = REPO / "scripts" / "training" / "eval_taiji_r2_readout_retrain.py"
RUNNER = REPO / "scripts" / "training" / "train_taiji_r2_readout_retrain.py"
CONTRACT = REPO / "plans" / "reference" / "M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md"

from api.seed_runtime import DEFAULT_CHECKPOINT as PRODUCT_DEFAULT  # noqa: E402

pytestmark = pytest.mark.skipif(
    not PRODUCT_DEFAULT.is_file(), reason="product default substrate absent"
)


def _load(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    for entry in (str(path.parent), str(REPO)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verdict() -> Any:
    return _load("_readout_retrain_verdict_under_test", VERDICT)


def _write_corpus(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _train_arms(tmp_path: Path, arms: str, symbols: int) -> Path:
    """Produce real arm checkpoints with the real runner, so the guards see real envelopes."""

    corpus = tmp_path / "arm.jsonl"
    _write_corpus(corpus, ["甲乙丙丁戊己庚辛壬癸"] * 40)
    manifest = tmp_path / "arm.json"
    manifest.write_text(
        json.dumps(
            {
                "output": str(corpus),
                "skip_symbols": 4,
                "first_emitted_row": 0,
                "replay_symbols_from_seen_region": 0,
                "skip_derivation": {"method": "test fixture"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "run"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--arms",
            arms,
            "--symbols",
            str(symbols),
            "--checkpoint-every",
            str(symbols),
            "--progress-every",
            str(symbols),
            "--out-dir",
            str(out_dir),
            "--corpus",
            str(corpus),
            "--lineage-manifest",
            str(manifest),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return out_dir


def _expect_refusal(verdict: Any, capsys: Any, phrase: str) -> None:
    """Two refusal styles live here: ``raise SystemExit("msg")`` carries the text in the exception,
    ``parser.error`` exits with code 2 and puts it on stderr.  Check both."""

    with pytest.raises(SystemExit) as info:
        verdict.main()
    message = f"{info.value}\n{capsys.readouterr().err}"
    assert phrase in message, message


# --------------------------------------------------------------------------- #
# The pairing guards: every way of getting a non-matched triple must be refused
# --------------------------------------------------------------------------- #


def test_a_missing_arm_is_refused(
    tmp_path: Path, verdict: Any, monkeypatch: Any, capsys: Any
) -> None:
    run_dir = _train_arms(tmp_path, "A,B", 40)
    monkeypatch.setattr(
        sys,
        "argv",
        ["verdict", "--run-dir", str(run_dir), "--symbols", "40", "--out-report", str(tmp_path / "v.json")],
    )
    _expect_refusal(verdict, capsys, "缺臂即作废配对")
    assert not (tmp_path / "v.json").exists()


def test_an_arm_that_did_not_finish_the_budget_is_refused(
    tmp_path: Path, verdict: Any, monkeypatch: Any, capsys: Any
) -> None:
    run_dir = _train_arms(tmp_path, "A,B,C", 40)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verdict",
            "--run-dir",
            str(run_dir),
            "--symbols",
            "80",
            "--out-report",
            str(tmp_path / "v.json"),
        ],
    )
    _expect_refusal(verdict, capsys, "did not finish the approved budget")
    assert not (tmp_path / "v.json").exists()


def test_arms_from_a_different_substrate_are_refused(
    tmp_path: Path, verdict: Any, monkeypatch: Any, capsys: Any
) -> None:
    """A verdict computed against one substrate but claiming another would be a stale reference."""

    import torch

    run_dir = _train_arms(tmp_path, "A,B,C", 40)
    tampered = run_dir / "C" / "checkpoint.pt"
    envelope = torch.load(tampered, map_location="cpu", weights_only=False)
    envelope["metadata"]["base_checkpoint_sha256"] = "0" * 64
    torch.save(envelope, tampered)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verdict",
            "--run-dir",
            str(run_dir),
            "--symbols",
            "40",
            "--out-report",
            str(tmp_path / "v.json"),
        ],
    )
    _expect_refusal(verdict, capsys, "拒判")
    assert not (tmp_path / "v.json").exists()


def test_an_existing_verdict_is_never_overwritten(
    tmp_path: Path, verdict: Any, monkeypatch: Any, capsys: Any
) -> None:
    existing = tmp_path / "v.json"
    existing.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["verdict", "--out-report", str(existing)])
    _expect_refusal(verdict, capsys, "never overwritten")
    assert existing.read_text(encoding="utf-8") == "{}"


# --------------------------------------------------------------------------- #
# A dry run must not be quotable as a verdict
# --------------------------------------------------------------------------- #


def test_dry_run_suppresses_every_judgment_block(verdict: Any) -> None:
    per_arm = {
        "A": {"usable": True, "flags": [1, 1, 0], "well_formed_rate": 0.6667},
        "B": {"usable": True, "flags": [0, 0, 0], "well_formed_rate": 0.0},
        "C": {"usable": True, "flags": [0, 1, 0], "well_formed_rate": 0.3333},
    }
    assert "suppressed" in verdict.judge_m1(per_arm, True)
    assert "suppressed" in verdict.judge_k2(per_arm, True)
    assert verdict.run_m2(Path("."), True)["status"] == "skipped_in_dry_run"


def test_the_dry_run_report_path_differs_from_the_verdict_path(verdict: Any) -> None:
    """A self-check must not occupy the path the real verdict refuses to overwrite."""

    assert verdict.DEFAULT_REPORT != verdict.DRY_RUN_REPORT
    assert verdict.DEFAULT_REPORT.name == "taiji_r2_readout_retrain_verdict_20260920.json"
    assert "pipeline_check" in verdict.DRY_RUN_REPORT.name


# --------------------------------------------------------------------------- #
# The gates themselves, and the pre-registration that has to exist before results
# --------------------------------------------------------------------------- #


def test_the_gate_thresholds_are_the_frozen_ones(verdict: Any) -> None:
    assert verdict.M1_MARGIN_PP == 15.0
    assert verdict.M2_MARGIN_ITEMS == 2
    assert verdict.M1_DIMENSIONS == ("B", "G")
    assert verdict.M2_DIMENSIONS == ("D", "E")


def test_m1_needs_both_the_margin_and_the_sign_test(verdict: Any) -> None:
    """Big margin with a non-significant pairing must not pass, and vice versa."""

    # A wins every pair against both rivals: margin and significance both hold.
    clean = {
        "A": {"usable": True, "flags": [1] * 20, "well_formed_rate": 1.0},
        "B": {"usable": True, "flags": [0] * 20, "well_formed_rate": 0.0},
        "C": {"usable": True, "flags": [0] * 20, "well_formed_rate": 0.0},
    }
    clean_verdict = verdict.judge_m1(clean, False)
    assert clean_verdict["delta_percentage_points"] == 100.0
    assert clean_verdict["sign_test_a_vs_best_rival"]["p_two_sided"] < 0.05
    assert clean_verdict["holds"] is True

    # 25 pp of margin, but A wins only 5 paired items and loses none: with n=5 the exact
    # two-sided sign test lands at p = 0.0625, just outside 0.05.  The gate must not pass on
    # the rate alone -- that is exactly the failure mode M1's two clauses exist to prevent.
    thin = {
        "A": {"usable": True, "flags": [1] * 12 + [0] * 8, "well_formed_rate": 0.6},
        "B": {"usable": True, "flags": [0] * 20, "well_formed_rate": 0.0},
        "C": {"usable": True, "flags": [1] * 7 + [0] * 13, "well_formed_rate": 0.35},
    }
    thin_verdict = verdict.judge_m1(thin, False)
    assert thin_verdict["best_rival"] == "C"
    assert thin_verdict["delta_percentage_points"] == 25.0
    assert thin_verdict["sign_test_a_vs_best_rival"]["p_two_sided"] == 0.0625
    assert thin_verdict["holds"] is False

    # Same margin, but the disagreements split evenly => not significant.
    noisy = {
        "A": {"usable": True, "flags": [1, 0, 1, 0, 1, 0, 1, 0], "well_formed_rate": 0.5},
        "B": {"usable": True, "flags": [0, 1, 0, 1, 1, 0, 1, 0], "well_formed_rate": 0.5},
        "C": {"usable": True, "flags": [0, 0, 0, 0, 0, 0, 0, 0], "well_formed_rate": 0.0},
    }
    assert verdict.judge_m1(noisy, False)["holds"] is False


def test_k2_operationalisation_is_pre_registered_in_the_contract() -> None:
    """K2's threshold was absent from the frozen §5; it must have been written down first."""

    text = CONTRACT.read_text(encoding="utf-8")
    assert "§5.1 K2 的操作化定义" in text
    assert "执行前预注册" in text
    assert "任何一臂都尚未跑完" in text
    assert "+15 pp" in text and "p ≥ 0.05" in text


def test_the_verdict_reuses_the_frozen_criterion_instead_of_rewriting_it(verdict: Any) -> None:
    """The sentence criterion must come from the FROZEN instrument, not from a local copy."""

    import diag_taiji_r2_surface_decode as frozen

    assert verdict.build_ngram_model is frozen.build_ngram_model
    assert verdict.well_formed is frozen.well_formed
    assert verdict.assert_criterion_discriminates is frozen.assert_criterion_discriminates
    assert verdict.sign_test is frozen.sign_test
    assert verdict.generate is frozen.generate
