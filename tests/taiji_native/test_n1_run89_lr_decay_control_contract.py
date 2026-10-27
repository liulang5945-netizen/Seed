"""PLAN-N1-89 control-arm family contract.

The 89 family exists to make nine checkpoints differ by readout arm only, so every
risk here is a naming or boundary mistake that would silently mix two recipes. Both
sides of each guard are walked in-process, and nothing is written in the repo: no
report, no checkpoint, no query.

* position is refused - the family trains control arms only, run88c owns position;
* the DEBT-G144 fix is pinned, the score-row prefix set is derived from the same
  constant as the loss prefix and strictly contains the 18432 milestone that run88's
  controller skipped;
* the launcher is one parameterised file that must hand ``--arm`` to the controller;
* the verifier's gate recomputation and prefix comparison both answer True and False.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_lr_decay_run89 as run89,
)
from scripts.training import (  # noqa: E402
    verify_taiji_n1_transition_factorial_lr_decay_run89_batch as verify89,
)

FAMILIES = (
    ("terminal", "run89a", "Seed-N1-LRDecay-Run89A-20261017"),
    ("cut", "run89b", "Seed-N1-LRDecay-Run89B-20261017"),
)


def summary(**overrides) -> dict:
    data = {
        "free_exact": 192,
        "free_by_kind": dict.fromkeys(verify89.KINDS, 48),
        "pair_exact": 96,
        "rename_free_exact": 192,
        "rename_free_by_kind": dict.fromkeys(verify89.KINDS, 48),
        "rename_pair_exact": 96,
        "factorial": {view: {"joint_four_exact": 48} for view in ("original", "rename")},
    }
    data.update(overrides)
    return data


def record(rows: list[dict], losses: list[dict], measurements: list[dict]) -> dict:
    return {"rows": rows, "losses": losses, "measurements": measurements}


def test_position_arm_is_refused() -> None:
    with pytest.raises(ValueError, match="control arms only"):
        run89.configure("position")


@pytest.mark.parametrize("arm,stem,task", FAMILIES)
def test_family_paths_are_derived_from_the_arm(arm: str, stem: str, task: str) -> None:
    run89.configure(arm)
    assert arm == run89.ARM and stem == run89.RUN_STEM and task == run89.TASK_RUN
    assert Path(run89.REPORT).name == f"taiji_n1_transition_factorial_lr_decay_{stem}.json"
    assert (
        Path(run89.PREFLIGHT).name
        == f"taiji_n1_transition_factorial_lr_decay_{stem}_preflight.json"
    )
    assert run89.CHECKPOINT_ROOT.parent.name == stem
    assert run89.CHECKPOINT_ROOT.name == "n1_transition_factorial_lr_decay_20261017"
    assert run89.LAUNCHER.is_file() and run89.VERIFIER.is_file()


def test_the_two_families_do_not_share_receipt_paths() -> None:
    run89.configure("terminal")
    first = (run89.REPORT, run89.PREFLIGHT, run89.CHECKPOINT_ROOT)
    run89.configure("cut")
    second = (run89.REPORT, run89.PREFLIGHT, run89.CHECKPOINT_ROOT)
    assert all(a != b for a, b in zip(first, second, strict=True))


def test_prefix_boundary_is_one_constant_and_strictly_contains_the_milestone() -> None:
    run89.configure("terminal")
    assert run89.PREFIX_UPDATES == run89.DECAY_START
    assert (
        tuple(step for step in run89.MILESTONES if step <= run89.DECAY_START)
        == run89.PREFIX_ROW_STEPS
    )
    assert run89.DECAY_START in run89.PREFIX_ROW_STEPS
    strict_below = {step for step in run89.MILESTONES if step < run89.DECAY_START}
    assert strict_below < set(run89.PREFIX_ROW_STEPS)


def test_learning_rate_table_and_group_application() -> None:
    assert run89.learning_rate_for(0) == run89.learning_rate_for(18431) == 0.001
    assert run89.learning_rate_for(18432) == run89.learning_rate_for(24576) == 0.00025
    assert run89.verify_learning_rate_groups() is True
    with pytest.raises(ValueError, match="outside frozen run88 schedule"):
        run89.learning_rate_for(24579)


def test_protocol_names_the_arm_and_binds_the_position_family() -> None:
    run89.configure("terminal")
    protocol = run89.run_protocol({"schema": "n1-transition-factorial-fresh-paired-training-v1"})
    assert protocol["schema"] == ("n1-transition-factorial-terminal-final-cycle-lr-decay-run89a-v1")
    assert protocol["arms"] == ["terminal"]
    assert protocol["prefix_row_steps"] == list(run89.PREFIX_ROW_STEPS)
    assert protocol["cpu_limit_seconds"] == 4800
    assert protocol["prior_cpu_reserve_seconds"] == 0
    assert len(protocol["bound_position_report_sha256"]) == 64
    assert len(protocol["bound_position_verification_sha256"]) == 64


def test_code_identity_covers_position_binding_and_fails_loudly_when_incomplete() -> None:
    run89.configure("cut")
    protocol = run89.run_protocol({"schema": "x"})
    identity = run89.code_identity(protocol)
    assert identity["schema"] == "n1-transition-factorial-lr-decay-run89b-code-identity-v1"
    assert identity["controller_sha256"] == run89.sha256(Path(run89.__file__))
    assert identity["position_batch_report_sha256"] == run89.sha256(run89.POSITION_REPORT)
    assert "position_batch_verification_sha256" in identity
    kept = run89.PRIOR_FAILED
    try:
        run89.PRIOR_FAILED = (*kept, run89.ROOT / "no_such_run89_identity_input.json")
        with pytest.raises(FileNotFoundError):
            run89.code_identity(protocol)
    finally:
        run89.PRIOR_FAILED = kept


def test_launcher_is_one_parameterised_file_that_hands_over_the_arm() -> None:
    text = run89.LAUNCHER.read_text(encoding="utf-8")
    assert "[ValidateSet('terminal', 'cut')]" in text
    assert (
        "'--arm ' + $Arm + ' --execute-approved'"
        in text.replace("' --arm ' + $Arm", "'--arm ' + $Arm")
        or "--arm ' + $Arm" in text
    )
    for _, _, task in FAMILIES:
        assert task in text
    for stem in ("run89a", "run89b"):
        assert f"Seed-N1-LRDecay-{stem[-1].upper()}" in text or stem in text


def test_launcher_powershell_syntax_parses() -> None:
    #: PowerShell 语言工件只能由 PowerShell 解析器判——Ubuntu 腿默认没有 `powershell`，
    #: 那是环境缺席而不是回归。按候选名找解释器，找到就跑，两者都缺席就点名响亮 skip；
    #: 任何其它失败（rc≠0／解析出 error）仍照旧红。
    launcher = run89.LAUNCHER
    command = (
        "$errors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{launcher.as_posix()}', [ref]$null, [ref]$errors) | Out-Null; "
        "$errors.Count"
    )
    shell = next((name for name in ("powershell", "pwsh") if shutil.which(name)), None)
    if shell is None:
        pytest.skip(
            "宿主没有 PowerShell 解释器（powershell/pwsh 都不在 PATH）⇒ "
            f"{launcher.as_posix() or 'run89.LAUNCHER（未 configure）'} 的语法面在此不可判定"
        )
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("0"), result.stdout


def test_gate_recomputation_reads_minima_and_thresholds() -> None:
    assert verify89.recompute_gates(summary()) == {
        "original_min": 192,
        "original_kind_min": 48,
        "pair_min": 96,
        "rename_min": 192,
        "rename_kind_min": 48,
        "rename_pair_min": 96,
        "joint_min": 48,
    }
    failing = summary(free_by_kind={"lookup": 39, "revision": 48, "transition": 48, "unknown": 48})
    gates = verify89.recompute_gates(failing)
    assert gates["original_kind_min"] == 39 and gates["joint_min"] == 48


def test_require_is_fail_closed_and_walks_both_branches() -> None:
    verify89.require(True, "must not raise")
    with pytest.raises(verify89.Guard, match="boom"):
        verify89.require(False, "boom")


@pytest.mark.parametrize("arm,stem", [("terminal", "run89a"), ("cut", "run89b")])
def test_prefix_readings_answer_true_for_equal_and_false_for_moved_milestone(
    arm: str, stem: str
) -> None:
    run89.configure(arm)
    steps = list(run89.MILESTONES)
    rows = [{"completed_updates": step, "id": f"i{step}"} for step in steps]
    losses = [{"qa": 1.0} for _ in range(run89.UPDATES)]
    measures = [{"step": step} for step in range(run89.UPDATES)]
    current = record(rows, losses, measures)
    baseline = record([dict(row) for row in rows], list(losses), list(measures))
    equal = verify89.prefix_readings(current, baseline)
    assert equal["losses_prefix_equal"] and equal["measurements_prefix_equal"]
    assert equal["decay_milestone_rows_equal"] is True
    assert equal["terminal_rows_equal"] is True

    moved = [dict(row) for row in rows]
    for row in moved:
        if row["completed_updates"] == run89.DECAY_START:
            row["id"] = "tampered"
    drifted = verify89.prefix_readings(record(moved, losses, measures), baseline)
    assert drifted["decay_milestone_rows_equal"] is False
    assert drifted["strictly_below_decay_rows_equal"] is True

    shortened = [dict(row) for row in rows if row["completed_updates"] != run89.DECAY_START]
    dropped = verify89.prefix_readings(record(shortened, losses, measures), baseline)
    assert dropped["score_rows_by_milestone"][str(run89.DECAY_START)]["rows"] == 0
    assert dropped["decay_milestone_rows_equal"] is False


def test_verifier_output_name_follows_the_configured_family(tmp_path: Path) -> None:
    prefix = "taiji_n1_transition_factorial_lr_decay_"
    targets = {}
    for arm, stem, _task in FAMILIES:
        run89.configure(arm)
        name = f"{prefix}{run89.RUN_STEM}_batch_verification.json"
        assert name == f"{prefix}{stem}_batch_verification.json"
        targets[stem] = name
    assert targets["run89a"] != targets["run89b"], "two families would share one target"

    sealed = tmp_path / targets["run89b"]
    sealed.write_text("{}", encoding="utf-8")
    with pytest.raises(verify89.Guard, match="single attempt file"):
        verify89.execute("cut", sealed)
    assert sealed.read_text(encoding="utf-8") == "{}", "refusal must not rewrite the sealed file"
