"""Contract guard for the CAP-0 legacy-load counterfactual.

The counterfactual answers one question the inventory could not: **if the legacy
guard were relaxed, would the trained checkpoint load -- and would it talk?**  The
measured answer is "yes, and no": the state loads at its real ``tick``, but its
output is byte-for-byte the same template the untrained ``tick=2`` substrate emits.

That "no" is the load-bearing part.  It is what separates a correctness defect from
a capability gap, and a silent edit that made the recovered arm look conversational
would invert the conclusion.  So both halves are asserted here.

The probe runs three fresh-process arms (~9 s), so this file reads the committed
report and never re-runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "reports" / "taiji_cap0_legacy_load_probe_20260915.json"
RUNNER = REPO / "scripts" / "training" / "probe_taiji_cap0_legacy_load.py"


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
