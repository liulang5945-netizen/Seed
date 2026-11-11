"""D-20261111-01: 类型检查可以红，但不许把证据采集步骤短路掉。

改前 `Type check core modules with mypy` 是硬失败步：它一红，其后
materialize-bundle 与全量回归全部 skipped，Linux 腿一个读数都拿不到
（run 42849129318 实测）。这份契约钉住三件事：

1. 该步带 `continue-on-error: true`（否则后面的步骤会被跳过）；
2. 它的 outcome 进了末尾汇总，并且被 `check` 判红（否则红会静默消失）；
3. 汇总步自身不带 `continue-on-error`，且在出现红时 `exit 1`
   （否则「mypy 有错而汇总判绿」就会通过）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
TYPE_CHECK = "Type check core modules with mypy"
SUMMARY = "Summarize CI gate outcomes"
EVIDENCE_STEPS = ("Materialize the 09.8 subject bundle", "Run complete regression suite")


def _ubuntu_steps() -> list[dict]:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    return jobs["test"]["steps"]


def _index_of(steps: list[dict], prefix: str) -> int:
    for index, step in enumerate(steps):
        if str(step.get("name", "")).startswith(prefix):
            return index
    raise AssertionError(f"ci.yml 的 test job 里找不到以 {prefix!r} 开头的步骤")


def test_type_check_cannot_skip_the_evidence_steps():
    steps = _ubuntu_steps()
    index = _index_of(steps, TYPE_CHECK)
    assert steps[index].get("continue-on-error") is True, (
        "类型检查必须带 continue-on-error: true，否则它一红就会把后面的"
        " materialize-bundle 与全量回归一起跳过"
    )
    for name in EVIDENCE_STEPS:
        assert index < _index_of(steps, name), f"{name} 必须排在类型检查之后，才能被它短路"


def test_type_check_outcome_is_judged_by_the_summary():
    steps = _ubuntu_steps()
    type_check = steps[_index_of(steps, TYPE_CHECK)]
    step_id = type_check.get("id")
    assert step_id, "类型检查步骤必须有 id，否则汇总拿不到它的 outcome"

    summary = steps[_index_of(steps, SUMMARY)]
    env = {str(value) for value in (summary.get("env") or {}).values()}
    expected_outcome = f"${{{{ steps.{step_id}.outcome }}}}"
    assert expected_outcome in env, "汇总的 env 里必须接上类型检查的 outcome"
    expected_check = 'check "typecheck-core" "$OUT_TYPES"'
    assert expected_check in summary["run"], "汇总必须对该 outcome 调用 check"


def test_the_summary_itself_fails_when_any_gate_is_red():
    steps = _ubuntu_steps()
    summary = steps[_index_of(steps, SUMMARY)]
    assert summary.get("continue-on-error") is not True, "汇总步自己不许带 continue-on-error"
    assert 'if [ "$red" -ne 0 ]' in summary["run"], "汇总必须在有红时判红"
    assert "exit 1" in summary["run"], "汇总判红后必须 exit 1"
