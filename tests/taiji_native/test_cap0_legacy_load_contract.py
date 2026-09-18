"""Contract guard for the CAP-0 legacy-load counterfactual.

The counterfactual answers one question the inventory could not: **if the legacy
guard were relaxed, would the trained checkpoint load -- and would it talk?**  The
measured answer is "yes, and no": the state loads at its real ``tick``, but its
output is byte-for-byte the same template the untrained ``tick=2`` substrate emits.

That "no" is the load-bearing part.  It is what separates a correctness defect from
a capability gap, and a silent edit that made the recovered arm look conversational
would invert the conclusion.  So both halves are asserted here.

The probe runs three fresh-process arms (~9 s), so most of this file reads the committed report --
with one deliberate exception: ``test_a_fresh_probe_sample_reproduces_the_sealed_one`` re-runs the
probe in-process and compares leaf-by-leaf against the 09-18 sample, because a report-reading test
can never go red when the instrument changes (audit §3).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _report_leaves import leaves, tail

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "taiji_cap0_legacy_load_probe_20260915.json"
#: 09-18 重采样：写侧隔离（DEBT-I7）落地之后，默认基座已稳定，因此可以当复现参照。
RESAMPLE = REPO / "reports" / "taiji_cap0_legacy_load_probe_20260918.json"
RUNNER = REPO / "scripts" / "training" / "probe_taiji_cap0_legacy_load.py"


#: 允许"当场重跑"与参照样本不同的叶子：采样时刻、总耗时，以及每条臂的墙钟。
#: 刻意**不含** ``tick`` / ``saved_at_utc`` / 输出文本 —— 那正是"默认基座又被谁改了"要看住的东西。
VOLATILE_SAMPLE_FIELDS = frozenset({"seconds", "elapsed_seconds"})
VOLATILE_SAMPLE_PATHS = frozenset({"record.commit"})


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_probe_completed_and_edited_no_source(report):
    assert RUNNER.is_file()
    assert report["status"] == "completed"
    assert report.get("error") is None
    for arm in report["arms"].values():
        assert arm["source_edited"] is False
    joined = " ".join(report["does_not_do"])
    assert "edits no repository source" in joined
    assert "writes no checkpoint" in joined


def test_the_source_edited_flag_can_actually_say_true(tmp_path: Path) -> None:
    """Audit item B4: a flag that can only ever read False is not a measurement.

    ``source_edited`` used to be a hard-coded ``False``, which made the assertion above pass for
    the wrong reason forever.  It is now a before/after digest of ``taiji/``, so the helper must be
    shown to react -- otherwise nothing changed except the wording.
    """

    import importlib.util
    import sys

    probe_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "training"
        / "probe_taiji_cap0_legacy_load.py"
    )
    name = "_legacy_load_probe_for_b4"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, probe_path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    probe = sys.modules[name]

    repo = tmp_path / "repo"
    tree = repo / "taiji"
    tree.mkdir(parents=True)
    (tree / "model.py").write_bytes(b"x = 1\n")
    before = probe.source_fingerprint(repo)

    (tree / "model.py").write_bytes(b"x = 2\n")
    assert probe.source_fingerprint(repo) != before, "content change must be visible"
    (tree / "added.py").write_bytes(b"y = 1\n")
    assert probe.source_fingerprint(repo) != before, "an added file must be visible too"


def test_current_guard_loads_the_trained_checkpoint_after_the_migration(report):
    """M2-2i landed: the product entry point loads the trained v8 file without the guard patch.

    This test used to pin the refusal (``load_ok is False`` + the organ error).  Flipping it is the
    point: the old assertions read a committed report, so a code change alone would have left them
    green while describing a world that no longer exists.  What still has to hold is the *other*
    half of the lesson -- loading is not capability: the recovered output remains one template.
    """

    arm = report["arms"]["trained_current_guard"]
    assert arm["load_ok"] is True
    assert arm["tick"] == 16_000_000
    assert arm["chat_organ_backend"] == "native-readable"
    summary = arm["output_summary"]
    assert (
        summary["templated"] is True and summary["distinct_signatures"] == 1
    ), "a successful load must never be readable as a recovered language ability"


def test_relaxed_guard_recovers_the_trained_state(report):
    arm = report["arms"]["trained_relaxed_guard"]
    assert arm["load_ok"] is True
    assert arm["tick"] == 16_000_000
    # The wrapper must have taken the tolerant branch, or the recovery would be
    # attributable to something else.
    assert arm["legacy_guard_applied"] is True
    assert arm["guard_relaxed"] is True


def test_recovered_state_is_not_more_conversational(report):
    """The decisive negative half: same template at tick 16M as at tick 2."""

    trained = report["arms"]["trained_relaxed_guard"]
    control = report["arms"]["default_control_current_guard"]
    assert trained["output_summary"]["templated"] is True
    assert control["output_summary"]["templated"] is True
    assert (
        trained["output_summary"]["signature"] == control["output_summary"]["signature"]
    ), "the trained and untrained states must not diverge silently"

    verdict = report["verdict"]
    assert verdict["trained_state_recovered_by_relaxing_guard"] is True
    assert verdict["recovered_output_is_non_template"] is False
    assert "separate problems" in verdict["reading"]
    assert "will not produce conversation" in verdict["reading"]


def test_arms_are_verbatim_and_prompt_echoing(report):
    for name in ("trained_relaxed_guard", "default_control_current_guard"):
        turns = report["arms"][name]["turns"]
        assert len(turns) == 3
        for turn in turns:
            assert "raw_output" in turn, (name, turn["prompt"])
            assert turn["prompt"] in turn["raw_output"]
            assert turn["output_bytes"] > 0


def test_probe_states_its_own_limits(report):
    limits = report["limits"]
    assert "does not make the fix safe" in limits
    assert "policy decision" in limits
    assert "only a frozen evaluation set can" in limits
    # It must not claim to have decided anything.
    joined = " ".join(report["does_not_do"])
    assert "does not decide the refuse-vs-migrate policy question" in joined


def test_a_fresh_probe_sample_reproduces_the_sealed_one(tmp_path) -> None:
    """普查 §3 的"复现封存"半边（第三支）：**当场重跑三臂探针**，与 09-18 样本逐叶比较。

    本文件其余断言只读已提交 JSON，仪器改了也不会红。这一支会红的情形包括：
    某条臂不再能加载、恢复出的 tick 变了、原始输出不再是那条模板、或默认基座又被谁改写
    （``tick`` / ``saved_at_utc`` 刻意不在易变名单里）。参照必须用 09-18 那份：
    09-15 那份记的默认控制臂是 ``tick=36``，而那个基座已被套件改写掉（DEBT-I7/I9），
    拿它当参照只会测到基座漂移而不是仪器。
    实测成本 9.7 s（``elapsed_seconds`` 取自产物自身）。
    """

    from scripts.training.probe_taiji_cap0_legacy_load import run_probe

    fresh = run_probe(tmp_path / "probe.json")
    sealed = json.loads(RESAMPLE.read_text(encoding="utf-8"))
    old, new = leaves(sealed), leaves(fresh)

    assert old.keys() == new.keys(), "仪器少产/多产了字段"
    shared = old.keys() & new.keys()
    volatile = {
        path
        for path in shared
        if tail(path) in VOLATILE_SAMPLE_FIELDS or path in VOLATILE_SAMPLE_PATHS
    }
    drifted = {path for path in shared if old[path] != new[path]}
    assert drifted <= volatile, sorted(drifted - volatile)[:8]
    assert len(shared) > 60 and len(volatile) * 3 < len(shared), (len(shared), len(volatile))

    # 不依赖屏蔽集的正面表述：这条反事实的结论本身必须照样成立。
    for name in ("trained_current_guard", "trained_relaxed_guard", "default_control_current_guard"):
        fresh_arm, sealed_arm = fresh["arms"][name], sealed["arms"][name]
        assert fresh_arm["load_ok"] is True, name
        assert fresh_arm["output_summary"]["templated"] is True, name
        assert [turn["raw_output"] for turn in fresh_arm["turns"]] == [
            turn["raw_output"] for turn in sealed_arm["turns"]
        ], name
    assert fresh["verdict"]["recovered_tick"] == sealed["verdict"]["recovered_tick"] == 16_000_000
    assert fresh["verdict"]["recovered_output_is_non_template"] is False
