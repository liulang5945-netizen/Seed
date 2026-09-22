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


def test_a_final_progress_entry_never_reports_a_fake_zero(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """A budget that is a multiple of ``--progress-every`` used to end on a fabricated 0.0.

    The loop's own write at ``consumed == symbols`` consumed the counters, and the post-loop
    write then emitted ``online_accuracy = 0.0`` for an empty window -- which reads exactly like
    "the arm collapsed".  The shipped A arm did this (2026-09-21).  Now an empty window reports
    ``null`` plus a cumulative line that is always meaningful.
    """

    source, manifest = corpus
    out_dir = tmp_path / "run"
    _run_cli(
        [
            "--arms",
            "A",
            "--symbols",
            "40",
            "--checkpoint-every",
            "40",
            "--progress-every",
            "40",
            "--out-dir",
            str(out_dir),
            "--corpus",
            str(source),
            "--lineage-manifest",
            str(manifest),
        ]
    )
    rows = [
        json.loads(line)
        for line in (out_dir / "A" / "progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    final = rows[-1]
    assert final["final"] is True
    assert final["window_symbols"] == 0
    assert final["online_accuracy"] is None, "an empty window must not report 0.0"
    assert final["mean_surprise"] is None
    assert final["cumulative_online_accuracy"] is not None

    report = json.loads((out_dir / "A" / "run_report.json").read_text(encoding="utf-8"))
    assert report["session_readout"]["online_accuracy"] is not None
    assert report["session_readout"]["comparable_across_arms"] is False


def test_a_finished_arm_is_skipped_so_a_campaign_can_be_re_run(
    tmp_path: Path, corpus: tuple[Path, Path]
) -> None:
    """Re-running the same command must be idempotent, including for a multi-arm campaign.

    The first version aborted the whole campaign at the first finished arm, so a campaign whose
    process died mid-way (which is exactly what happened to arm C) could not be resumed with the
    command it was started with.  Now a finished arm is skipped and recorded, and the arm that
    still has budget is the only one that does work.
    """

    source, manifest = corpus
    out_dir = tmp_path / "run"
    args = [
        "--arms",
        "B,C",
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
    first_session = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))[
        "session"
    ]["session_id"]

    summary = _run_cli(args)
    assert summary["status"] == "completed"
    assert summary["arms_status"] == {"B": "already_complete", "C": "already_complete"}
    assert summary["failed_arms"] == []
    # 已经跑满的臂不是 pending：把 already_complete 算成 pending 会吐出 `--arms B,C` 这种
    # 什么都不做的"续跑命令"（2026-09-22 实测踩到）。
    assert summary["arms_pending"] == []
    assert summary["resume_command"] is None
    second_session = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))[
        "session"
    ]["session_id"]
    assert second_session == first_session, "a skipped arm must not be re-run"


def test_a_stop_file_saves_and_exits_cleanly(tmp_path: Path, corpus: tuple[Path, Path]) -> None:
    """「现在把进度存下来然后停」必须是确定的能力，不能靠等下一个 checkpoint 边界。

    哨兵文件一出现就在下一次检查点落盘并退出：报告标 ``stopped_by_request``（**不是失败**），
    checkpoint 记下确切消费量，同一条命令再跑就从那里接着跑。机器要重启时这就是停止点。
    """

    source, manifest = corpus
    out_dir = tmp_path / "run"
    stop_file = tmp_path / "STOP_RUN"
    stop_file.write_text("stop", encoding="utf-8")
    summary = _run_cli(
        [
            "--arms",
            "C",
            "--symbols",
            "400",
            "--checkpoint-every",
            "100000",  # 比预算大 ⇒ 只有优雅停机那条路径会落盘
            "--progress-every",
            "100000",
            "--stop-check-every",
            "100",
            "--out-dir",
            str(out_dir),
            "--corpus",
            str(source),
            "--lineage-manifest",
            str(manifest),
            "--stop-file",
            str(stop_file),
        ]
    )
    assert summary["status"] == "completed", "优雅停机不是失败"
    assert summary["arms_status"] == {"C": "stopped_by_request"}
    assert summary["arms_pending"] == ["C"]
    assert summary["resume_command"].startswith("--arms C")

    import torch

    envelope = torch.load(out_dir / "C" / "checkpoint.pt", map_location="cpu", weights_only=False)
    consumed = int(envelope["metadata"]["symbols_consumed"])
    assert 0 < consumed < 400, "停机必须落盘，且落在预算之内"
    assert not stop_file.exists(), "哨兵用一次即失效，否则续跑会立刻又停"

    report = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "stopped_by_request"
    assert report["stop_reason"] == "stopped_by_request"
    assert report["symbols_consumed"] == consumed

    # 同一条命令再跑：从记录的消费量接着跑，不重放前缀
    resumed = _run_cli(
        [
            "--arms",
            "C",
            "--symbols",
            "400",
            "--checkpoint-every",
            "400",
            "--out-dir",
            str(out_dir),
            "--corpus",
            str(source),
            "--lineage-manifest",
            str(manifest),
        ]
    )
    assert resumed["arms_status"] == {"C": "completed"}
    final = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))
    assert final["symbols_consumed"] == 400
    assert final["session"]["symbols"] == 400 - consumed


def test_fresh_is_the_only_way_to_redo_a_finished_arm(
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
    before = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))
    assert before["symbols_consumed"] == 30
    assert before["fresh"] is False

    # ``--fresh`` 是唯一的重做通路：换新会话，且旧报告被**归档**而不是删掉。
    summary = _run_cli([*args, "--fresh"])
    assert summary["arms_status"] == {"C": "completed"}
    after = json.loads((out_dir / "C" / "run_report.json").read_text(encoding="utf-8"))
    assert after["session"]["session_id"] != before["session"]["session_id"]
    assert after["fresh"] is True
    assert after["archived_previous_report"] is not None
    assert Path(after["archived_previous_report"]).is_file()
