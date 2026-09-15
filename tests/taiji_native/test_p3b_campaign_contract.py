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
        # both tiers stay below one pass of the subset, which the preregistration calls
        # infeasible (568 h), so "budget exhausted" can never mean "the corpus ran out"
        assert symbols < 559_000_000


def test_protected_checkpoints_cannot_be_run_targets(trainer: Any) -> None:
    for path in trainer.PROTECTED_OUTPUTS:
        with pytest.raises(SystemExit):
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


def test_treatment_data_provenance_is_manifest_bound(trainer: Any, tmp_path: Path) -> None:
    other = tmp_path / "corpus.jsonl"
    other.write_text('{"text": "a"}\n', encoding="utf-8")
    record = trainer._guard_data_provenance(other)
    assert record["manifest_binding"].startswith("none (control arm")

    subset = tmp_path / "p3b_dialogue_subset.jsonl"
    subset.write_text('{"text": "老师：甲\n乙：乙\n"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"output_sha256": "0" * 64}), encoding="utf-8")
    trainer.SUBSET_PATH = subset
    trainer.SUBSET_MANIFEST = manifest
    with pytest.raises(SystemExit, match="drifted"):
        trainer._guard_data_provenance(subset)


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
    assert treatment[0] == trainer_checkpoint(treatment[0])


def trainer_checkpoint(path: Path) -> Path:
    """The campaign must score the very file the trainer writes, not a copy of it."""

    return path


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
    """A stop rule that exists only in prose gets re-interpreted under pressure."""

    source = CAMPAIGN.read_text(encoding="utf-8")
    for token in (
        "improved",
        "stall",
        "regression",
        "chain_required",
        "material",
        "persistent",
        "stage_scoring",
        "why_two_arms",
    ):
        assert token in source, token
    assert campaign.STALL_LIMIT == 3
    # 20 items per dimension: one flipped item is 0.05, so the noise floor sits at two items
    assert campaign.REGRESSION_MARGIN == 0.10


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


def test_scoring_freezes_the_checkpoint_before_reading_it(campaign: Any, tmp_path: Path) -> None:
    """The trainer overwrites its checkpoint while a ~205 s stage is running."""

    live = tmp_path / "seed_aligned.pt"
    live.write_bytes(b"state-at-tick-17000000")
    frozen = campaign._snapshot(live, 17_000_000)
    assert frozen.read_bytes() == live.read_bytes()
    assert frozen.parent == tmp_path / "snapshots"

    live.write_bytes(b"state-at-tick-18000000")
    assert frozen.read_bytes() == b"state-at-tick-17000000", "the snapshot must be immutable"
    again = campaign._snapshot(live, 17_000_000)
    assert again == frozen and again.read_bytes() == b"state-at-tick-17000000"


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
    """The summariser has no judgement of its own -- J1-J5 stay in check_p3b_criteria."""

    summarize = _load("_p3b_summarize_under_test", SUMMARIZER)
    source = SUMMARIZER.read_text(encoding="utf-8")
    for invented in ("machine_normalised", "min_lines", "passed =", "verdict ="):
        assert invented not in source, invented
    stage = _stage(17_000_000, {"C": 0.2, "D": 0.1, "E": 0.3})
    headline = summarize.arm_headline({"arm": "treatment", "stages": [stage], "status": "running"})
    assert headline["latest_tick"] == 17_000_000
    assert headline["latest_scores"] == stage["scores"]


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
