"""Contract guard for the readout-retrain formal runner.

Why this exists: the owner approved (2026-09-20) the contract's §5 numeric lines and a per-arm
budget of 16,000,000 symbols.  Two things then have to be true in *code*, not in prose, before any
long run is launched:

* the approved ceiling and the "never write into the weights directory" rules are enforced by the
  runner itself -- a 33.6 h campaign that silently ran 10x the approved budget, or overwrote the
  product substrate, is not recoverable;
* a checkpoint written by one process can be read back and **continued by another process**, at the
  recorded corpus offset.  The precondition added to contract §4.1 says this must be demonstrated
  before training starts; this file is that demonstration (it uses real subprocesses, because
  "resumable in a new process" cannot be shown in-process).

It also pins the single-source-of-truth wiring: the calibration instrument and the runner must take
their arm definitions from ``readout_retrain_spec``, not each carry a copy that drifts.
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
RUNNER = REPO / "scripts" / "training" / "train_taiji_r2_readout_retrain.py"
CALIBRATOR = REPO / "scripts" / "training" / "calibrate_taiji_r2_readout_retrain.py"
SPEC = REPO / "scripts" / "training" / "readout_retrain_spec.py"

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
def runner() -> Any:
    return _load("_readout_retrain_runner_under_test", RUNNER)


def _spec_module() -> Any:
    """The spec module under its **canonical** name -- the one the two scripts import.

    Loading it through ``_load`` with a fixture-private name would execute it a second time and
    give a different ``ARMS`` object, so the identity assertion below would be testing the test.
    """

    if "readout_retrain_spec" not in sys.modules:
        _load("readout_retrain_spec", SPEC)
    return sys.modules["readout_retrain_spec"]


def _write_corpus(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": text}, ensure_ascii=False) + "\n" for text in texts),
        encoding="utf-8",
    )


def _write_manifest(path: Path, corpus: Path, skip_symbols: int) -> None:
    path.write_text(
        json.dumps(
            {
                "output": str(corpus),
                "skip_symbols": skip_symbols,
                "first_emitted_row": 0,
                "replay_symbols_from_seen_region": 0,
                "skip_derivation": {"method": "test fixture"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny synthetic corpus plus the lineage manifest that gives it a skip point."""

    path = tmp_path / "arm.jsonl"
    _write_corpus(path, ["甲乙丙丁戊己庚辛壬癸"] * 40)
    manifest = tmp_path / "arm.json"
    _write_manifest(manifest, path, 4)
    return path, manifest


def _expect_refusal(runner: Any, capsys: Any, phrase: str) -> None:
    """``parser.error`` exits with code 2 and puts the reason on stderr, not in the exception."""

    with pytest.raises(SystemExit) as info:
        runner.main()
    assert info.value.code not in (0, None)
    stderr = capsys.readouterr().err
    assert phrase in stderr, stderr


# --------------------------------------------------------------------------- #
# One source of truth for the arm definitions
# --------------------------------------------------------------------------- #


def test_both_scripts_take_the_arms_from_the_spec_module(runner: Any) -> None:
    calibrator = _load("_readout_retrain_calibrator_under_test", CALIBRATOR)
    spec_module = _spec_module()
    assert calibrator.ARMS is spec_module.ARMS
    assert runner.ARMS is spec_module.ARMS
    assert spec_module.APPROVED_SYMBOL_CEILING == 16_000_000


def test_the_specs_expected_surfaces_are_the_ones_the_contract_names() -> None:
    spec_module = _spec_module()
    assert {name: spec["write_surface"] for name, spec in spec_module.ARMS.items()} == {
        "A": ("predictive_readout",),
        "B": ("motor",),
        "C": (),
    }
    assert spec_module.ARMS["A"]["observe_kwargs"]["readout"] == "predictive"
    assert spec_module.ARMS["B"]["observe_kwargs"]["readout"] == "action"
    assert spec_module.ARMS["C"]["observe_kwargs"]["learn"] is False


# --------------------------------------------------------------------------- #
# The approved budget and the write-target rules are enforced in code
# --------------------------------------------------------------------------- #


def test_budget_ceiling_refuses_an_unapproved_symbol_count(
    runner: Any, monkeypatch: Any, capsys: Any
) -> None:
    over = runner.APPROVED_SYMBOL_CEILING + 1
    monkeypatch.setattr(sys, "argv", ["train", "--symbols", str(over)])
    _expect_refusal(runner, capsys, "exceeds the approved ceiling")


def test_write_target_inside_the_weights_directory_is_refused(
    runner: Any, monkeypatch: Any, capsys: Any
) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--out-dir", "checkpoints/readout-retrain"])
    _expect_refusal(runner, capsys, "must not live under")


def test_unknown_arm_fails_closed(runner: Any, monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["train", "--arms", "A,Z"])
    _expect_refusal(runner, capsys, "unknown arm")


def test_a_slice_inside_the_seen_prefix_is_refused(
    runner: Any, tmp_path: Path, corpus: tuple[Path, Path], monkeypatch: Any, capsys: Any
) -> None:
    source, manifest = corpus
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--arms",
            "C",
            "--symbols",
            "20",
            "--out-dir",
            str(tmp_path / "run"),
            "--corpus",
            str(source),
            "--lineage-manifest",
            str(manifest),
            "--skip-symbols",
            "0",
        ],
    )
    _expect_refusal(runner, capsys, "unseen data")
    assert not (tmp_path / "run" / "campaign_report.json").exists()


# --------------------------------------------------------------------------- #
# End to end, through real subprocesses: save here, continue somewhere else
# --------------------------------------------------------------------------- #


def _run_cli(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_both_processes_see_the_same_arm_surfaces_and_the_substrate_is_untouched(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    source, manifest = corpus
    out_dir = tmp_path / "run"
    _run_cli(
        [
            "--arms",
            "A,B,C",
            "--symbols",
            "45",
            "--checkpoint-every",
            "30",
            "--progress-every",
            "30",
            "--out-dir",
            str(out_dir),
            "--corpus",
            str(source),
            "--lineage-manifest",
            str(manifest),
        ]
    )
    for arm, expected in (("A", ["predictive_readout"]), ("B", ["motor"]), ("C", [])):
        report = json.loads((out_dir / arm / "run_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "completed", arm
        assert report["realised_write_surface"] == expected, arm
        assert report["write_surface_ok"] is True, arm
        assert report["symbols_consumed"] == 45, arm
    campaign = json.loads((out_dir / "campaign_report.json").read_text(encoding="utf-8"))
    assert campaign["status"] == "completed"
    assert campaign["base_checkpoint_unchanged"] is True


def test_a_second_process_continues_at_the_recorded_offset(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """The §4.1 precondition: a checkpoint must be loadable and resumable by a *new* process.

    The first process stops after 60 symbols; the second asks for 150 and must therefore do
    exactly 90 more, continuing from the recorded corpus offset rather than from the window start.
    """

    source, manifest = corpus
    out_dir = tmp_path / "run"
    common = [
        "--arms",
        "A",
        "--checkpoint-every",
        "60",
        "--progress-every",
        "60",
        "--out-dir",
        str(out_dir),
        "--corpus",
        str(source),
        "--lineage-manifest",
        str(manifest),
    ]
    _run_cli([*common, "--symbols", "60"])
    first = json.loads((out_dir / "A" / "run_report.json").read_text(encoding="utf-8"))
    assert first["resumed_from"] is None
    assert first["session"]["symbols"] == 60

    _run_cli([*common, "--symbols", "150"])
    second = json.loads((out_dir / "A" / "run_report.json").read_text(encoding="utf-8"))
    assert second["resumed_from"] is not None
    assert second["symbols_consumed"] == 150
    assert second["session"]["symbols"] == 90, "a resume must not replay the consumed prefix"
    assert second["base_sources"][-1].endswith("checkpoint.pt")
    # the two arms' streams are the same window: both started at the manifest's skip point
    assert second["skip_symbols"] == 4


def test_a_finished_arm_refuses_to_run_again(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    source, manifest = corpus
    out_dir = tmp_path / "run"
    args = [
        "--arms",
        "C",
        "--symbols",
        "30",
        "--checkpoint-every",
        "30",
        "--out-dir",
        str(out_dir),
        "--corpus",
        str(source),
        "--lineage-manifest",
        str(manifest),
    ]
    _run_cli(args)
    completed = subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode != 0
    assert "already consumed" in completed.stderr + completed.stdout
