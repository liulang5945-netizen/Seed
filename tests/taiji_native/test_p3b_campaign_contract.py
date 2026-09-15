"""Contract guard for the P3b campaign runner (train_p3b_aligned.py / run_p3b_campaign.py).

P3b is the first mainline step that spends machine hours on a *training* run, so the parts that
keep it honest are pinned here as pure checks -- no training, no evaluation, no hashing of the
534 MB corpus:

* the budget tiers cannot silently drift from the frozen throughput calibration;
* the two checkpoints that must survive the run (the product default entry and the start state)
  cannot be selected as an output path, and no default path points at them;
* the trainer rebuilds its config from the checkpoint envelope **because no ``--scale`` profile
  equals the stored v8 profile** -- if that ever stops being true the CLI path should be used
  instead, and this test says so loudly;
* the arm paths never overlap, so treatment and control cannot overwrite each other;
* the stop-rule arithmetic (improved / regressed / stall streak / pending-grew) is the one the
  preregistration §4 describes, and the chain constant is shared with ``check_p3b_criteria``.

Measured payload: ``reports/taiji_p3b_throughput_calibration_20260915.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts" / "training" / "train_p3b_aligned.py"
CAMPAIGN = REPO / "scripts" / "training" / "run_p3b_campaign.py"
SUMMARIZER = REPO / "scripts" / "training" / "summarize_p3b.py"
WAITER = REPO / "scripts" / "training" / "wait_p3b_stage.py"
CRITERIA = REPO / "scripts" / "training" / "check_p3b_criteria.py"
CALIBRATION = REPO / "reports" / "taiji_p3b_throughput_calibration_20260915.json"
START_CHECKPOINT = REPO / "checkpoints" / "seed_beta.pt"


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
def trainer() -> Any:
    return _load("_p3b_trainer_under_test", TRAINER)


@pytest.fixture(scope="module")
def campaign() -> Any:
    return _load("_p3b_campaign_under_test", CAMPAIGN)


@pytest.fixture(scope="module")
def criteria() -> Any:
    return _load("_p3b_criteria_under_test", CRITERIA)


@pytest.fixture(scope="module")
def waiter() -> Any:
    return _load("_p3b_waiter_under_test", WAITER)


def _report(c: float | None, d: float | None, e: float | None, pending: int = 0) -> dict[str, Any]:
    def dim(score: float | None, key: str) -> dict[str, Any]:
        return {
            "tally": {
                "machine_normalised": score,
                "pending_human_review_items": pending if key in ("B", "G") else 0,
            },
            "items": [],
        }

    return {
        "chain": {"relax_legacy_guard": True, "constrained_decode": True},
        "trained_during_eval": False,
        "dimensions": {
            "B": dim(None, "B"),
            "C": dim(c, "C"),
            "D": dim(d, "D"),
            "E": dim(e, "E"),
            "G": dim(None, "G"),
        },
    }


# --------------------------------------------------------------------------- #
# Budget and provenance guards
# --------------------------------------------------------------------------- #


def test_budget_tiers_stay_tied_to_the_frozen_calibration(trainer: Any) -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    measured = float(calibration.get("steps_per_second", calibration.get("throughput", 0.0)))
    assert abs(measured - trainer.CALIBRATED_STEPS_PER_SECOND) <= 1.0

    for tier, symbols in trainer.BUDGET_TIERS.items():
        hours = symbols / measured / 3600.0
        assert abs(hours - float(tier.rstrip("h"))) < 1.5, tier
        # `epochs=1` means a corpus shorter than the budget ends the run early and the arm
        # silently trains on less data.  Both arm manifests carry their own emitted symbol
        # count, so the bound is measured, not a remembered constant.
        for manifest in (trainer.SUBSET_MANIFEST, trainer.CONTROL_MANIFEST):
            emitted = int(json.loads(manifest.read_text(encoding="utf-8"))["emitted_symbols"])
            assert symbols < emitted, (tier, manifest.name, emitted)


def test_protected_checkpoints_cannot_be_run_targets(trainer: Any, campaign: Any) -> None:
    """Membership first, behaviour second.

    Iterating ``PROTECTED_OUTPUTS`` to prove each entry is refused says nothing if an entry is
    simply deleted -- and the two entries that matter are the product's default checkpoint and
    the P3b start state, the only files this run must never touch.
    """

    names = {path.name for path in trainer.PROTECTED_OUTPUTS}
    assert {"seed_corpus.pt", "seed_beta.pt"} <= names, names
    assert trainer.START_CHECKPOINT in trainer.PROTECTED_OUTPUTS
    assert set(campaign.PROTECTED_CHECKPOINTS) <= set(trainer.PROTECTED_OUTPUTS)
    for path in (trainer.PROTECTED_OUTPUTS[0], trainer.START_CHECKPOINT):
        with pytest.raises(SystemExit, match="protected"):
            trainer._refuse_protected(path)
    checkpoint = trainer.arm_paths("treatment")[0]
    assert trainer._refuse_protected(checkpoint) == checkpoint.resolve()
    assert trainer._refuse_protected(trainer.arm_paths("control")[0]) != checkpoint.resolve()


def test_no_default_output_path_points_at_a_protected_checkpoint(
    trainer: Any, campaign: Any
) -> None:
    """A guard is only as good as its defaults: the shipped defaults must be safe too."""

    protected = {path.resolve() for path in trainer.PROTECTED_OUTPUTS}
    for arm in ("treatment", "control"):
        for path in trainer.arm_paths(arm) + campaign.campaign_paths(arm):
            assert Path(path).resolve() not in protected, (arm, path)


def test_config_must_be_rebuilt_from_the_envelope(trainer: Any) -> None:
    """Why ``--resume --scale`` cannot work: no scale profile equals seed_beta's."""

    from taiji import TaijiConfig

    config, meta = trainer.build_config_from_checkpoint(START_CHECKPOINT)
    assert meta["start_tick"] == 16_000_000
    assert meta["start_substrate_format"] == "taiji-native-v8"
    for scale in range(1, 6):
        profile = TaijiConfig.training_profile(scale=scale, seed=20260822)
        assert profile != config.taiji, f"scale {scale} now matches; the CLI path may be usable"


def test_both_arm_corpora_are_manifest_bound(
    trainer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "corpus.jsonl"
    other.write_text('{"text": "a"}\n', encoding="utf-8")
    record = trainer._guard_data_provenance(other)
    assert record["manifest_binding"].startswith("none (ad-hoc corpus")

    for arm_path, manifest_path in (
        (tmp_path / "p3b_dialogue_fresh.jsonl", tmp_path / "dialogue_manifest.json"),
        (tmp_path / "p3b_all_fresh.jsonl", tmp_path / "all_manifest.json"),
    ):
        arm_path.write_text('{"text": "老师：甲\n乙：乙\n"}\n', encoding="utf-8")
        manifest_path.write_text(json.dumps({"output_sha256": "0" * 64}), encoding="utf-8")
        # monkeypatch, not assignment: this trainer module is cached in sys.modules and shared
        # with test_p3b_arm_corpus_contract.py, which asserts on the real registry
        monkeypatch.setattr(trainer, "ARM_MANIFESTS", {arm_path: manifest_path})
        with pytest.raises(SystemExit, match="drifted"):
            trainer._guard_data_provenance(arm_path)
        manifest_path.write_text(
            json.dumps({"output_sha256": trainer._sha256(arm_path)}), encoding="utf-8"
        )
        bound = trainer._guard_data_provenance(arm_path)
        assert bound["manifest_binding"].endswith(manifest_path.name)


def test_missing_corpus_fails_closed(trainer: Any, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="missing"):
        trainer._guard_data_provenance(tmp_path / "absent.jsonl")


# --------------------------------------------------------------------------- #
# Campaign arithmetic (the frozen stop rules)
# --------------------------------------------------------------------------- #


def test_arms_own_disjoint_files(campaign: Any) -> None:
    treatment = campaign.campaign_paths("treatment")
    control = campaign.campaign_paths("control")
    assert set(map(str, treatment)).isdisjoint(set(map(str, control)))


def test_the_driver_and_the_trainer_agree_on_the_arm_path(trainer: Any, campaign: Any) -> None:
    """``campaign_paths()[0]`` is ``arm_paths()[0]``, not a lookalike recomputed in the driver.

    This catches path drift between the two files.  The distinct hazard of scoring a copy while
    the trainer writes the original is covered by the snapshot tests, which pin the file that
    gets scored back to this path.
    """

    for arm in ("treatment", "control"):
        assert campaign.campaign_paths(arm)[0] == trainer.arm_paths(arm)[0], arm


def test_the_campaign_cannot_point_an_arm_at_a_stray_corpus() -> None:
    """The trainer resolves its corpus from ``--arm``, and both arms are manifest-bound.

    A ``--corpus`` passthrough here would let a campaign train on a file that no manifest
    describes, which is precisely the situation where ``treatment - control`` stops meaning
    anything -- so the absence of that flag is the invariant, not an oversight.
    """

    source = CAMPAIGN.read_text(encoding="utf-8")
    assert '"--corpus"' not in source, "an arm corpus must come from the trainer's arm mapping"
    assert '"--arm"' in source
    assert '"--budget-tier"' in source, "the tier guard, not an ad-hoc symbol count, sizes the run"


def test_the_trainer_refuses_to_guess_which_arm_it_is(trainer: Any) -> None:
    """No default for ``--arm``: a defaulted arm resolves to treatment's live files.

    The arm decides the checkpoint, the progress log (appended, not rewritten) and the run
    report, so guessing wrong does not fail -- it contaminates the campaign in progress.
    """

    with pytest.raises(SystemExit):
        trainer.main(["--budget-tier", "48h"])
    source = TRAINER.read_text(encoding="utf-8")
    assert "--arm treatment --budget-tier" in source, "the documented usage must pass an arm"
    assert "--arm control --budget-tier" in source


def test_stage_integrity_accepts_whole_reports_and_flags_broken_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEBT-I6 is unfixed while a campaign is in flight, so the monitor has to be able to see it."""

    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    # monkeypatch, not assignment: this module is cached in sys.modules and shared
    monkeypatch.setattr(summarize, "PROJECT_ROOT", tmp_path)
    directory = tmp_path / "reports" / "p3b_stages" / "treatment"
    directory.mkdir(parents=True)
    good = {
        "eval_set": "plans/manifests/cap0_eval_set_v1.json",
        "eval_set_format": "cap0-eval-set-v1",
        "eval_set_frozen_on": "2026-09-15",
        "declared_mode": "N",
        "trained_during_eval": False,
        "dimensions": {
            key: {"items": [{"id": f"{key}{i}"} for i in range(20)]} for key in ("C", "D", "E")
        },
    }
    (directory / "ok.json").write_text(json.dumps(good), encoding="utf-8")
    assert summarize.stage_integrity("treatment", {"report": "ok.json"}) == []

    (directory / "torn.json").write_text(json.dumps(good)[:120], encoding="utf-8")
    assert summarize.stage_integrity("treatment", {"report": "torn.json"}) == ["torn_json"]

    short = json.loads(json.dumps(good))
    short["dimensions"]["D"]["items"] = []
    (directory / "short.json").write_text(json.dumps(short), encoding="utf-8")
    assert summarize.stage_integrity("treatment", {"report": "short.json"}) == ["D:items=0"]

    assert summarize.stage_integrity("treatment", {"report": "gone.json"}) == ["missing_report"]


def test_stage_row_marks_improvement_only_when_a_score_moves(campaign: Any) -> None:
    baseline = _report(0.0, 0.0625, 0.15, pending=20)  # P3a: B and G fully pending
    flat = campaign._stage_row(1, _report(0.0, 0.0625, 0.15), baseline, "flat.json")
    assert flat["improved"] is False and flat["regressed"] is False
    assert flat["all_three_strictly_higher"] is False

    better = campaign._stage_row(2, _report(0.1, 0.0625, 0.15), baseline, "better.json")
    assert better["improved"] is True and better["deltas_vs_p3a"]["C"] == 0.1

    mixed = campaign._stage_row(3, _report(0.1, 0.0, 0.15), baseline, "mixed.json")
    assert mixed["improved"] is True and mixed["regressed"] is True
    assert mixed["all_three_strictly_higher"] is False

    pending = campaign._stage_row(4, _report(0.1, 0.07, 0.2, pending=21), baseline, "pending.json")
    assert pending["pending_grew"] is True and pending["pending_delta_vs_p3a"]["B"] == 1


def test_stall_streak_counts_consecutive_non_improvement(campaign: Any) -> None:
    baseline = _report(0.0, 0.0625, 0.15)
    flat = campaign._stage_row(1, _report(0.0, 0.0625, 0.15), baseline, "a.json")
    good = campaign._stage_row(2, _report(0.05, 0.07, 0.2), baseline, "b.json")
    assert campaign._stall_streak([flat]) == 1
    assert campaign._stall_streak([flat, good]) == 0
    assert campaign._stall_streak([good, flat, flat, flat]) == 3
    assert campaign._stall_streak([flat, flat, good, flat]) == 1


def test_campaign_requires_the_p3a_chain(campaign: Any, criteria: Any) -> None:
    baseline = _report(0.0, 0.0625, 0.15)
    row = campaign._stage_row(1, _report(0.1, 0.1, 0.1), baseline, "ok.json")
    assert row["chain_matches_p3a"] is True
    assert campaign.REQUIRED_CHAIN == criteria.REQUIRED_CHAIN

    broken = _report(0.1, 0.1, 0.1)
    broken["chain"] = {"relax_legacy_guard": False, "constrained_decode": True}
    off = campaign._stage_row(2, broken, baseline, "off.json")
    assert off["chain_matches_p3a"] is False


def test_stop_definitions_are_recorded_in_every_report(campaign: Any) -> None:
    """A stop rule that exists only in prose gets re-interpreted under pressure.

    Asserted against the shipped payload object, not against source text: grepping words such as
    "improved" or "material" is satisfied by identifiers and docstrings alone, so deleting the
    whole ``stop_definitions`` block from the record would have kept the old check green.
    """

    definitions = campaign.STOP_DEFINITIONS
    assert set(definitions) >= {
        "improved",
        "stall",
        "regression",
        "stage_scoring",
        "comparability",
        "chain_required",
    }, sorted(definitions)
    assert definitions["improved"] == "max(C/D/E delta vs P3a) > 0"
    assert campaign.STALL_LIMIT == 3
    assert str(campaign.STALL_LIMIT) in definitions["stall"]
    # 20 items per dimension: one flipped item is 0.05, so the noise floor sits at two items
    assert campaign.REGRESSION_MARGIN == 0.10
    assert f"-{campaign.REGRESSION_MARGIN}" in definitions["regression"]
    assert "material" in definitions["regression"] and "persistent" in definitions["regression"]
    assert definitions["chain_required"] == campaign.REQUIRED_CHAIN
    for field in campaign.EVAL_SURFACE_FIELDS:
        assert field in definitions["comparability"], field
    assert "snapshots" in definitions["stage_scoring"]


def test_one_item_wobble_is_not_a_regression(campaign: Any) -> None:
    """A 48-hour campaign must not be killed by a single flipped eval item."""

    baseline = _report(0.0, 0.0625, 0.15, pending=20)
    one_item = campaign._stage_row(1, _report(0.0, 0.0625, 0.10), baseline, "a.json")
    assert one_item["regressed"] is True  # the row still reports the raw fact
    assert campaign._regression_kind([one_item]) is None

    two_items = campaign._stage_row(2, _report(0.0, 0.0625, 0.05), baseline, "b.json")
    assert campaign._regression_kind([two_items]) == "material"

    soft = campaign._stage_row(3, _report(0.0, 0.0, 0.10), baseline, "c.json")
    assert campaign._regression_kind([one_item, soft]) == "persistent"
    assert campaign._regression_kind([soft]) is None


def _envelope(path: Path, tick: int) -> Path:
    """A minimal stand-in for what `atomic_save` leaves behind: metadata carrying the tick."""

    import torch

    torch.save({"format": "taiji-native-v8", "metadata": {"tick": tick}}, path)
    return path


def test_scoring_freezes_the_checkpoint_before_reading_it(campaign: Any, tmp_path: Path) -> None:
    """The trainer overwrites its checkpoint while a ~205 s stage is running.

    Three separate hazards, all real: scoring the live file mixes two training states across
    dimensions; reusing a snapshot must not re-copy the moved-on live file; and a snapshot must
    never be labelled with a tick it does not contain.
    """

    live = _envelope(tmp_path / "seed_aligned.pt", 17_000_000)
    frozen, tick = campaign._snapshot(live, 17_000_000)
    assert tick == 17_000_000
    assert frozen.parent == tmp_path / "snapshots"
    assert frozen.read_bytes() == live.read_bytes()

    _envelope(live, 18_000_000)  # the trainer moves on mid-stage
    assert campaign._latest_tick(frozen) == 17_000_000, "the snapshot must be immutable"
    again, again_tick = campaign._snapshot(live, 17_000_000)
    assert again == frozen and again_tick == 17_000_000, "existing snapshot is reused, not recopied"


def test_a_snapshot_is_never_labelled_with_a_tick_it_does_not_hold(
    campaign: Any, tmp_path: Path
) -> None:
    """The caller's tick is a guess about a moving file; the envelope inside the copy decides."""

    live = _envelope(tmp_path / "seed_aligned.pt", 18_000_000)
    frozen, tick = campaign._snapshot(live, 17_000_000)
    assert tick == 18_000_000, "scoring a later state under an earlier label is undetectable later"
    assert frozen.name == "seed_aligned_tick_18000000.pt"
    assert not (tmp_path / "snapshots" / "seed_aligned_tick_17000000.pt").exists()


def test_a_snapshot_without_a_readable_tick_is_refused(campaign: Any, tmp_path: Path) -> None:
    live = tmp_path / "seed_aligned.pt"
    live.write_bytes(b"\x00\x01 truncated mid-write")
    with pytest.raises(SystemExit, match="no readable tick"):
        campaign._snapshot(live, 19_000_000)


# --------------------------------------------------------------------------- #
# Monitoring must not invent criteria
# --------------------------------------------------------------------------- #


def _stage(tick: int, scores: dict[str, float | None]) -> dict[str, Any]:
    return {
        "tick": tick,
        "scores": scores,
        "deltas_vs_p3a": {key: 0.0 for key in ("C", "D", "E")},
        "chain_matches_p3a": True,
    }


def test_arm_difference_is_matched_by_tick_not_by_order() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    treatment = {"stages": [_stage(17_000_000, {"C": 0.2, "D": 0.1, "E": 0.3})]}
    control = {
        "stages": [
            _stage(17_000_000, {"C": 0.1, "D": 0.0625, "E": 0.2}),
            _stage(18_000_000, {"C": 0.1, "D": 0.0625, "E": 0.15}),
        ]
    }
    rows = summarize.tick_table(treatment, control)
    assert [row["tick"] for row in rows] == [17_000_000, 18_000_000]
    assert rows[0]["treatment_minus_control"]["C"] == 0.1
    assert rows[0]["treatment_minus_control"]["E"] == 0.1
    assert rows[1]["treatment_minus_control"] is None  # treatment has no 18M stage yet
    assert rows[1]["control"] == {"C": 0.1, "D": 0.0625, "E": 0.15}


def test_missing_arm_reads_as_missing_not_zero() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    report = summarize.load_arm("control", root=Path(tempfile.mkdtemp()))
    assert report["status"] == "missing" and report["stages"] == []
    headline = summarize.arm_headline(report)
    assert headline["latest_scores"] is None and headline["criteria_verdict"] is None


def test_headline_only_carries_recorded_fields() -> None:
    """The summariser holds no **criterion** judgement -- J1-J5 stay in check_p3b_criteria.

    The one reading rule it does apply is the resolution threshold pre-declared in the
    novelty-matched amendment §4, and that rule has its own tests below.
    """

    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    source = SUMMARIZER.read_text(encoding="utf-8")
    for invented in ("machine_normalised", "min_lines", "passed =", "verdict ="):
        assert invented not in source, invented
    older = _stage(17_000_000, {"C": 0.2, "D": 0.1, "E": 0.3})
    newer = _stage(18_000_000, {"C": 0.4, "D": 0.2, "E": 0.5})
    headline = summarize.arm_headline(
        {"arm": "treatment", "stages": [older, newer], "status": "running"}
    )
    assert headline["latest_tick"] == 18_000_000, "latest must mean last, not first"
    assert headline["latest_scores"] == newer["scores"]


def _matched(tick: int, **deltas: Any) -> dict[str, Any]:
    return {"tick": tick, "treatment_minus_control": dict(deltas)}


def _all(delta: float) -> dict[str, Any]:
    return {"C": delta, "D": delta, "E": delta}


def test_one_item_of_twenty_is_not_an_effect() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    assert summarize.MAIN_EFFECT_RESOLUTION == 0.15  # three items, not one flipped answer
    assert summarize.CONSECUTIVE_CONFIRMATIONS == 2
    one_item = summarize.main_effect_verdict([_matched(17_000_000, **_all(0.05))])
    assert one_item["C"]["verdict"] == "not_detected_at_this_resolution"
    assert one_item["C"]["confirmed_direction"] is None
    one_point = summarize.main_effect_verdict([_matched(17_000_000, **_all(0.15))])
    assert one_point["C"]["verdict"] == "above_resolution_single_point"


def test_two_consecutive_same_direction_points_are_an_effect() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    twice = summarize.main_effect_verdict(
        [_matched(17_000_000, **_all(0.15)), _matched(18_000_000, **_all(0.20))]
    )
    assert twice["C"]["verdict"] == "effect_positive"
    assert twice["C"]["confirmed_direction"] == "positive"
    assert twice["C"]["longest_confirmed_run"] == 2
    negative = summarize.main_effect_verdict(
        [_matched(17_000_000, **_all(-0.15)), _matched(18_000_000, **_all(-0.25))]
    )
    assert negative["D"]["verdict"] == "effect_negative"


def test_a_direction_flip_resets_the_run_and_noise_cannot_undo_it() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    flipped = summarize.main_effect_verdict(
        [_matched(17_000_000, **_all(0.15)), _matched(18_000_000, **_all(-0.15))]
    )
    assert flipped["C"]["longest_confirmed_run"] == 1
    assert flipped["C"]["verdict"] == "above_resolution_single_point"
    faded = summarize.main_effect_verdict(
        [
            _matched(17_000_000, **_all(0.20)),
            _matched(18_000_000, **_all(0.20)),
            _matched(19_000_000, **_all(0.0)),
        ]
    )
    assert faded["C"]["verdict"] == "effect_positive"
    assert faded["C"]["latest_delta"] == 0.0
    assert faded["C"]["matched_ticks"] == 3


def test_ticks_without_both_arms_are_no_data_not_zero() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    rows = [
        _matched(17_000_000, C=0.15, D=None, E=None),
        {"tick": 18_000_000, "treatment_minus_control": None},
        _matched(19_000_000, C=0.15, D=0.15, E=None),
    ]
    out = summarize.main_effect_verdict(rows)
    assert out["C"]["matched_ticks"] == 2
    assert out["D"]["matched_ticks"] == 1
    assert out["E"]["matched_ticks"] == 0
    assert out["E"]["verdict"] == "no_matched_checkpoint"
    assert out["E"]["latest_delta"] is None


def test_the_summariser_says_what_it_cannot_judge() -> None:
    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    unjudged = summarize.main_effect_verdict([])["unjudged"]
    assert unjudged["dimensions_not_executed_by_this_runner"] == ["A", "F", "H"]
    assert "untested" in unjudged["consequence"]
    assert unjudged["resolution"] == 0.15


# --------------------------------------------------------------------------- #
# Comparability: two numbers off different surfaces are not an effect
# --------------------------------------------------------------------------- #


def _surface(items: list[str]) -> dict[str, Any]:
    report = {
        "eval_set": "plans/manifests/cap0_eval_set_v1.json",
        "eval_set_format": "cap0-eval-set-v1",
        "eval_set_frozen_on": "2026-09-15",
        "declared_mode": "N",
        "dimensions": {},
    }
    for key in ("C", "D", "E"):
        report["dimensions"][key] = {"items": [{"id": item} for item in items]}
    return report


def test_identical_eval_surface_reports_no_drift(campaign: Any) -> None:
    items = [f"{key}-{index}" for key in ("C", "D", "E") for index in range(20)]
    stage = _surface([])
    base = _surface([])
    for key in ("C", "D", "E"):
        stage["dimensions"][key]["items"] = [{"id": i} for i in items if i.startswith(key)]
        base["dimensions"][key]["items"] = [{"id": i} for i in items if i.startswith(key)]
    assert campaign._surface_drift(stage, base) == []


def test_swapped_eval_surface_or_item_order_is_drift(campaign: Any) -> None:
    items = [{"id": f"C{i}"} for i in range(3)]
    stage = _surface([])
    base = _surface([])
    stage["dimensions"]["C"]["items"] = items
    base["dimensions"]["C"]["items"] = items
    base["dimensions"]["D"]["items"] = base["dimensions"]["E"]["items"] = []
    assert campaign._surface_drift(stage, base) == []

    other = json.loads(json.dumps(stage))
    other["eval_set"] = "plans/manifests/cap0_eval_set_v2.json"
    assert campaign._surface_drift(other, base) == ["eval_set"]

    reordered = json.loads(json.dumps(stage))
    reordered["dimensions"]["C"]["items"] = list(reversed(reordered["dimensions"]["C"]["items"]))
    assert campaign._surface_drift(reordered, base) == ["C:item_ids"]

    echoed = json.loads(json.dumps(stage))
    echoed["declared_mode"] = "T"
    assert "declared_mode" in campaign._surface_drift(echoed, base)


# --------------------------------------------------------------------------- #
# Waiting is a reader.  Its only judgement is liveness, never capability.
# --------------------------------------------------------------------------- #


def test_waiter_reads_only_the_current_stop_key() -> None:
    waiter = _load("_p3b_waiter_under_test", WAITER)
    assert waiter.stop_of({"campaign_stop": "stalled"}) == "stalled"
    assert waiter.stop_of({}) is None
    assert waiter.stop_of({"campaign_stop": None}) is None
    # What this pins: the waiter reads the current key and nothing else. The reserved
    # episode-level token is deliberately not spelled out even to assert its absence -- the
    # fail-closed N2 disposition scanner audits every file that mentions it, so writing it here
    # would add this test to the very inventory it is meant to protect. That scanner, not this
    # line, is the enforcement. (It matches the token as a plain substring, so even citing the
    # scanner's own file name would pull a file into the surface -- hence "by role, not path".)
    source = WAITER.read_text(encoding="utf-8")
    assert "campaign_stop" in source
    assert waiter.STOP_KEY == "campaign_stop"


def test_waiter_needs_a_real_stage_count() -> None:
    waiter = _load("_p3b_waiter_under_test", WAITER)
    assert waiter.reached({}, 1) is False
    assert waiter.reached({"treatment": {}}, 1) is False
    assert waiter.reached({"treatment": {"stages": [{"tick": 1}]}}, 1) is True
    assert waiter.reached({"treatment": {"stages": [{"tick": 1}]}}, 2) is False
    assert waiter.reached({"treatment": {"stages": [{"tick": 1}]}, "control": {}}, 1) is True


def test_checkpoints_owed_arithmetic(waiter: Any) -> None:
    assert waiter.STALL_THRESHOLD == 2, "one pending checkpoint is normal scoring lag"
    assert waiter.unscored_checkpoints(18_000_000, 17_000_000, 1_000_000) == 1
    assert waiter.unscored_checkpoints(19_000_000, 17_000_000, 1_000_000) == 2
    assert waiter.unscored_checkpoints(16_050_000, 16_000_000, 1_000_000) == 0
    assert waiter.unscored_checkpoints(15_000_000, 16_000_000, 1_000_000) == 0, "never negative"
    for unknown in (
        (None, 17_000_000, 1_000_000),
        (18_000_000, None, 1_000_000),
        (18_000_000, 17_000_000, 0),
        (18_000_000, 17_000_000, None),
    ):
        assert waiter.unscored_checkpoints(*unknown) == 0, unknown


def test_an_orphaned_trainer_is_reported_as_a_stall(
    waiter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(waiter, "progress_tick", lambda arm: 18_000_000)
    report: dict[str, Any] = {
        "arm": "treatment",
        "status": "running",
        "checkpoint_every": 1_000_000,
        "stages": [{"tick": 17_000_000}],
    }
    state = waiter.arm_state(report)
    assert state["checkpoints_owed"] == 1
    assert state["driver_stalled"] is False, "the driver may simply be mid-scoring"
    report["stages"] = [{"tick": 16_000_000}]
    stalled = waiter.arm_state(report)
    assert stalled["checkpoints_owed"] == 2 and stalled["driver_stalled"] is True
    # a missing campaign record is no data, not a dead driver
    empty = waiter.arm_state({})
    assert empty["driver_stalled"] is False and empty["stages"] == 0
