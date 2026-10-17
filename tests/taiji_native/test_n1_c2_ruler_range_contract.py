"""C2 尺范围复核仪的合同册：四支判决支都要能走到，且"上界臂缺失／命中数越界"必须响亮拒绝。

这台仪器的存在理由是一条已经被用错的读数（DEBT-G157：七臂严格命中各 0/16 被当能力否证，
而连上界诊断臂 `oracle-correct` 也是 0），所以**拒绝支与判决支都要能为假**：夹具里每一支都造
一个真端点走不到的形状。最后一条是把新算法打在两枚**已封存真件**上——今天那枚件的读数必须
仍是"尺无范围、机制有位移"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import verify_taiji_n1_c2_ruler_range as rr  # noqa: E402


def write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    return path


def final_case(hits: dict) -> dict:
    return {
        "status": "complete",
        "summary": {arm: {"count": 16, "strict_hits": n} for arm, n in hits.items()},
    }


def placement_case(changes: dict) -> dict:
    return {
        "status": "complete",
        "summary": {arm: {"count": 16, "raw_changes_from_tail": n} for arm, n in changes.items()},
    }


ARMS = [
    "model-top1",
    "no-replay",
    rr.UPPER_BOUND_ARM,
    "pooled-correct",
    "shuffle-correct",
    "wrong-episode",
    rr.CONTROL_ARM,
]


def test_an_upper_bound_that_moves_means_the_ruler_has_range(tmp_path: Path) -> None:
    final = write(
        tmp_path / "f.json",
        final_case({arm: (3 if arm == rr.UPPER_BOUND_ARM else 0) for arm in ARMS}),
    )
    placement = write(
        tmp_path / "p.json",
        placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0, "wrong-episode": 3}),
    )
    report = rr.measure(rr.load(final, "complete"), rr.load(placement, "complete"))
    assert report["status"] == "ruler_has_range_and_upper_bound_moves", report["status"]
    assert report["ruler_has_dynamic_range"] is True
    assert report["can_be_cited_as_capability_falsification"] is True


def test_a_nonzero_arm_without_upper_bound_movement_is_its_own_branch(tmp_path: Path) -> None:
    final = write(
        tmp_path / "f2.json", final_case({arm: (2 if arm == "model-top1" else 0) for arm in ARMS})
    )
    placement = write(
        tmp_path / "p2.json",
        placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0, "wrong-episode": 3}),
    )
    report = rr.measure(rr.load(final, "complete"), rr.load(placement, "complete"))
    assert report["status"] == "ruler_has_range_upper_bound_still_zero", report["status"]
    assert report["upper_bound_strict_hits"] == 0 and report["ruler_has_dynamic_range"] is True


def test_all_zero_with_a_dose_move_is_the_branch_that_forbids_the_capability_sentence(
    tmp_path: Path,
) -> None:
    final = write(tmp_path / "f3.json", final_case({arm: 0 for arm in ARMS}))
    placement = write(
        tmp_path / "p3.json",
        placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0, "wrong-episode": 3}),
    )
    report = rr.measure(rr.load(final, "complete"), rr.load(placement, "complete"))
    assert report["status"] == "ruler_no_range_mechanism_moves", report["status"]
    assert report["can_be_cited_as_capability_falsification"] is False
    assert (
        rr.UPPER_BOUND_ARM in report["forbidden_sentence"] or "0/16" in report["forbidden_sentence"]
    )
    assert report["upper_bound_strict_hits"] == 0
    assert rr.UPPER_BOUND_ARM in report["falsification_blocked_because"], report[
        "falsification_blocked_because"
    ]


def test_silence_on_both_sides_is_not_a_verdict(tmp_path: Path) -> None:
    final = write(tmp_path / "f4.json", final_case({arm: 0 for arm in ARMS}))
    placement = write(
        tmp_path / "p4.json",
        placement_case({arm: 0 for arm in (rr.UPPER_BOUND_ARM, rr.CONTROL_ARM, "wrong-episode")}),
    )
    report = rr.measure(rr.load(final, "complete"), rr.load(placement, "complete"))
    assert report["status"] == "ruler_no_range_and_no_mechanism_move", report["status"]
    assert (
        report["mechanism_moves"] is False
        and report["can_be_cited_as_capability_falsification"] is False
    )


def test_guards_refuse_rather_than_guess(tmp_path: Path) -> None:
    #: 上界诊断臂不在场＝本判决的支点没了，必须拒绝而不是拿别的臂代答。
    missing = write(
        tmp_path / "g1.json", final_case({arm: 0 for arm in ARMS if arm != rr.UPPER_BOUND_ARM})
    )
    with pytest.raises(rr.Refuse, match="upper-bound arm"):
        rr.measure(
            rr.load(missing, "complete"),
            rr.load(
                write(
                    tmp_path / "g2.json", placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0})
                ),
                "complete",
            ),
        )
    #: 命中数大于分母＝件自己不自洽，不许 silently 当"有范围"。
    impossible = write(
        tmp_path / "g3.json", final_case({arm: (17 if arm == "model-top1" else 0) for arm in ARMS})
    )
    with pytest.raises(rr.Refuse, match="exceeds count"):
        rr.measure(
            rr.load(impossible, "complete"),
            rr.load(
                write(
                    tmp_path / "g4.json", placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0})
                ),
                "complete",
            ),
        )
    #: 非 complete 的真件（budget-stop）不许被当读数源。
    stopped = write(
        tmp_path / "g5.json", {**final_case({arm: 0 for arm in ARMS}), "status": "budget-stop"}
    )
    with pytest.raises(rr.Refuse, match="must be complete"):
        rr.load(stopped, "complete")


def test_the_write_target_is_one_shot(tmp_path: Path) -> None:
    final = write(tmp_path / "h1.json", final_case({arm: 0 for arm in ARMS}))
    placement = write(
        tmp_path / "h2.json",
        placement_case({rr.UPPER_BOUND_ARM: 6, rr.CONTROL_ARM: 0, "wrong-episode": 3}),
    )
    out = tmp_path / "out.json"
    argv = [
        "--final-report",
        str(final),
        "--placement-report",
        str(placement),
        "--out-report",
        str(out),
    ]
    assert rr.main(argv) == 0
    assert out.is_file()
    before = out.read_bytes()
    assert rr.main(argv) == 2, "second write must refuse"
    assert out.read_bytes() == before, "refusal must not touch the published report"


def test_the_sealed_pair_still_reads_no_range_with_a_live_mechanism(tmp_path: Path) -> None:
    """新算法第一次执行必须打在已知-good 的真件上：今天这两枚件的读数不许被修法改动。"""
    out = tmp_path / "real.json"
    assert rr.main(["--out-report", str(out)]) == 0
    entry = json.loads(out.read_text(encoding="utf-8"))
    assert entry["status"] == "ruler_no_range_mechanism_moves", entry["status"]
    assert entry["strict_hits_by_arm"] == {arm: 0 for arm in ARMS}, entry["strict_hits_by_arm"]
    assert entry["placement_raw_changes"] == {
        rr.UPPER_BOUND_ARM: 6,
        "wrong-episode": 3,
        rr.CONTROL_ARM: 0,
    }
    assert (
        entry["models_loaded"] == 0
        and entry["model_queries"] == 0
        and entry["parameter_updates"] == 0
    )
    assert entry["can_be_cited_as_capability_falsification"] is False
    #: 甲档要的两条合取项必须能在真件里逐臂回查，不能只写在 PLAN 的散文里。
    assert entry["format"] == "taiji-n1-c2-ruler-range-v2"
    assert entry["strict_hit_ruler_has_dynamic_range"] is False
    assert entry["placement_ruler_has_dynamic_range"] is True
    assert entry["option_a_conjunction_holds"] is True


def test_the_two_columns_are_judged_separately(tmp_path: Path) -> None:
    """v2 的支点：`ruler_has_dynamic_range` 说的是严格命中列，甲档用的是位移列。

    两列必须能为反向（今天真件就反向：严格全 0、位移 6/3/0），也能为同向零——位移列全零时
    甲档合取必须落假，而不是靠"机制看起来动过"。
    """
    diverging = rr.measure(
        final_case({arm: 0 for arm in ARMS}),
        placement_case({rr.UPPER_BOUND_ARM: 6, "wrong-episode": 3, rr.CONTROL_ARM: 0}),
    )
    assert diverging["format"] == "taiji-n1-c2-ruler-range-v2"
    assert diverging["strict_hit_ruler_has_dynamic_range"] is False
    assert diverging["placement_ruler_has_dynamic_range"] is True
    assert diverging["ruler_has_dynamic_range"] is diverging["strict_hit_ruler_has_dynamic_range"]
    assert diverging["option_a_column"] == "raw_changes_from_tail"
    assert diverging["option_a_dose_gt_control"] is True
    assert diverging["option_a_control_is_zero"] is True
    assert diverging["option_a_conjunction_holds"] is True

    flat = rr.measure(
        final_case({arm: 0 for arm in ARMS}),
        placement_case({rr.UPPER_BOUND_ARM: 0, "wrong-episode": 0, rr.CONTROL_ARM: 0}),
    )
    assert flat["placement_ruler_has_dynamic_range"] is False
    assert flat["option_a_dose_gt_control"] is False
    assert flat["option_a_control_is_zero"] is True
    assert flat["option_a_conjunction_holds"] is False
    assert flat["mechanism_moves"] is False
