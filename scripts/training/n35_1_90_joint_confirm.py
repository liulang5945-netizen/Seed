"""N35.1.90 (confirmation): same-subject six-module joint confirmation on 20261901-03.

Pre-registered: plans/reference/DESIGN-N35-1-90_20261126.md (D-20261126-671/672).
One run confirms (1) the 100-session same-subject durability subject (N34.5.90 config) and
(2) the open natural Chinese continuous task (NCH.3.04 config) on fresh confirmation seeds,
with R0 gates from plans/reference/SCORECARD-N35_20261126.md. Zero check/final reads.

Usage:
  python scripts/training/n35_1_90_joint_confirm.py --smoke
  python scripts/training/n35_1_90_joint_confirm.py
Output:
  reports/n35_1_90_joint_confirm.json (formal mode only)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.n31_guard import find_design, preflight  # noqa: E402
from scripts.training.n34_5_01_long_run_joint import _eval_gates  # noqa: E402
from scripts.training.nch_3_04_open_chinese_v3 import _eval_subject, train_arm  # noqa: E402
from taiji.long_run_trained_subject import RunConfig, run_long_run_subject  # noqa: E402
from taiji.open_chinese_subject import OpenChineseCognitiveSubject  # noqa: E402
from taiji.open_chinese_task_env import load_external_chinese_distractors  # noqa: E402
from taiji.session_store import SessionStore  # noqa: E402

CARD = "N35.1.90"
CONFIRM_SEEDS = (20261901, 20261902, 20261903)
SMOKE_SEEDS = (6921, 6922, 6923)
OPEN_STEPS = 720
SMOKE_OPEN_STEPS = 20
N_EVAL = 60
SMOKE_N_EVAL = 4
CAP_CPU = 650.0
OUT = ROOT / "reports" / "n35_1_90_joint_confirm.json"


def _eval_seeds(cs: int, offset: int, n: int) -> list[int]:
    return [cs * 1000 + offset + i for i in range(n)]


def _open_chinese_for_seed(
    cs: int, *, steps: int, n_eval: int, deadline: float, ext: tuple[str, ...]
) -> dict[str, Any]:
    arm_main, m0, m1 = train_arm(
        cs, use_history=True, use_copy=True, steps=steps, deadline=deadline
    )
    arm_stat, s0, s1 = train_arm(
        cs, use_history=False, use_copy=True, steps=steps, deadline=deadline
    )
    arm_nocopy, n0, n1 = train_arm(
        cs, use_history=True, use_copy=False, steps=steps, deadline=deadline
    )

    tr = _eval_seeds(cs, 0, n_eval)
    pa = _eval_seeds(cs, 100, n_eval)
    oov = _eval_seeds(cs, 200, n_eval)
    wild = _eval_seeds(cs, 300, n_eval)

    main_oov = _eval_subject(arm_main, oov, split="heldout_oov_entities")
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        arm_main.save_to_session_store(store, f"n35_1_90_{cs}", session_id=1)
        reloaded = OpenChineseCognitiveSubject.load_from_session_store(store, f"n35_1_90_{cs}")
        rel_oov = _eval_subject(reloaded, oov, split="heldout_oov_entities")
        store_exact = (
            all(
                torch.equal(a, b)
                for a, b in zip(arm_main.parameters(), reloaded.parameters(), strict=True)
            )
            and rel_oov == main_oov
        )
    return {
        "losses": {"main": [m0, m1], "stateless": [s0, s1], "no_copy": [n0, n1]},
        "open_chinese_memory_pg": {
            "train_seen": _eval_subject(arm_main, tr, split="train_seen"),
            "heldout_paraphrase": _eval_subject(arm_main, pa, split="heldout_paraphrase"),
            "heldout_oov_entities": main_oov,
            "real_corpus_distractor": (
                _eval_subject(
                    arm_main, wild, split="heldout_oov_entities", external_distractors=ext
                )
                if ext
                else None
            ),
        },
        "history_shuffle_ablation_oov": _eval_subject(
            arm_main, oov, split="heldout_oov_entities", ablate_history=True
        ),
        "stateless_single_turn_pg_oov": _eval_subject(arm_stat, oov, split="heldout_oov_entities"),
        "no_copy_vocab_only_oov": _eval_subject(arm_nocopy, oov, split="heldout_oov_entities"),
        "store_roundtrip_exact": store_exact,
    }


def _r0_checks(
    per_seed: dict[str, Any], spot_match: bool, *, n_sessions: int, n_events: int
) -> dict[str, bool]:
    lr = {k: v["long_run"] for k, v in per_seed.items()}
    base = _eval_gates(lr, spot_match, n_sessions=n_sessions, n_events=n_events)
    oc = {k: v["open_chinese"] for k, v in per_seed.items()}

    r1 = all(
        v["open_chinese_memory_pg"]["heldout_paraphrase"]["query_turn_success"] >= 0.75
        and v["open_chinese_memory_pg"]["heldout_paraphrase"]["coref_success"] >= 0.70
        and v["open_chinese_memory_pg"]["heldout_paraphrase"]["corr_neg_success"] >= 0.70
        for v in oc.values()
    )
    r2 = all(
        v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_turn_success"] >= 0.75
        and v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_answer_em"] >= 0.75
        and v["open_chinese_memory_pg"]["heldout_oov_entities"]["unknown_abstain_success"] >= 0.75
        and (
            v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_turn_success"]
            - v["stateless_single_turn_pg_oov"]["query_turn_success"]
            >= 0.50
        )
        and v["stateless_single_turn_pg_oov"]["known_answer_em"] <= 0.15
        and (
            v["open_chinese_memory_pg"]["heldout_oov_entities"]["known_answer_em"]
            - v["no_copy_vocab_only_oov"]["known_answer_em"]
            >= 0.40
        )
        and v["no_copy_vocab_only_oov"]["known_answer_em"] <= 0.15
        and (
            v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_turn_success"]
            - v["history_shuffle_ablation_oov"]["query_turn_success"]
            >= 0.35
        )
        for v in oc.values()
    )
    r3 = base["g3_physics_causal_endurance"] and all(
        v["physics"]["cross_family"]["collision_recall"] >= 0.95
        and v["physics"]["cross_family"]["collision_recall"]
        - v["physics"]["barrier_ablation"]["collision_recall_drop"]
        <= 0.02
        for v in lr.values()
    )
    r4 = base["g4_plastic_growth_retention"] and all(
        v["growth"]["p1_retention_drop"] <= 0.02 and v["growth"]["p2_acc"] >= 0.95
        for v in lr.values()
    )
    r5 = base["g5_credibility_drift_tracking"] and all(
        v["credibility"]["delta_brier"] < 0 and v["credibility"]["delta_post_abs"] < -0.03
        for v in lr.values()
    )
    r6 = base["g6_coupled_uncertainty_calibration"] and all(
        v["uncertainty"]["coupled_ece"] <= 0.06 for v in lr.values()
    )
    r7 = (
        base["g1_durability_and_recovery"]
        and base["g7_reproducibility"]
        and all(v["store_roundtrip_exact"] is True for v in oc.values())
    )
    return {
        "r0_1_open_paraphrase": r1,
        "r0_2_open_oov_and_controls": r2,
        "r0_3_physics_cross_family": r3,
        "r0_4_growth_retention": r4,
        "r0_5_credibility": r5,
        "r0_6_uncertainty": r6,
        "r0_7_longrun_recovery_repro": r7,
        "r0_8_discipline": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        design_file = find_design(CARD).name
        seeds = SMOKE_SEEDS
        cfg = RunConfig(
            n_sessions=12,
            events_per_session=20,
            crash_sessions=(3, 8),
            p1_end_session=4,
            p2_end_session=10,
            dormancy_session=11,
            ch_train_end_session=6,
            ch_updates_per_session=2,
            phy_steps_early=10,
            phy_steps_mid=10,
            grow_p1_steps=10,
            grow_p2_steps=10,
            grow_maint_steps=2,
        )
        open_steps, n_eval = SMOKE_OPEN_STEPS, SMOKE_N_EVAL
    else:
        design_file = preflight(CARD, list(CONFIRM_SEEDS), record=True).name
        seeds = CONFIRM_SEEDS
        cfg = RunConfig()
        open_steps, n_eval = OPEN_STEPS, N_EVAL

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    ext = load_external_chinese_distractors(
        ROOT / "data/simple_zh/dialogue_extended_clean.jsonl", max_items=48
    )
    spot = seeds[0]
    per_seed: dict[str, Any] = {}
    spot_match = False
    checks: dict[str, bool] = {}

    try:
        for seed in seeds:
            t0 = time.process_time()
            with tempfile.TemporaryDirectory() as tmp1:
                res = run_long_run_subject(tmp1, seed, deadline, cfg=cfg)
            if seed == spot and not args.smoke:
                with tempfile.TemporaryDirectory() as tmp2:
                    again = run_long_run_subject(tmp2, seed, deadline, cfg=cfg)
                spot_match = (
                    res["durability"]["chain_head"] == again["durability"]["chain_head"]
                    and res["durability"]["record_digest"] == again["durability"]["record_digest"]
                )
            elif seed == spot and args.smoke:
                spot_match = True
            long_cpu = round(time.process_time() - t0, 2)
            oc = _open_chinese_for_seed(
                seed, steps=open_steps, n_eval=n_eval, deadline=deadline, ext=ext
            )
            per_seed[str(seed)] = {
                "long_run": res,
                "open_chinese": oc,
                "cpu_long_run": long_cpu,
                "cpu": round(time.process_time() - t0, 2),
            }

        checks = _r0_checks(
            per_seed,
            spot_match,
            n_sessions=cfg.n_sessions,
            n_events=cfg.n_sessions * cfg.events_per_session,
        )
        if args.smoke:
            exit_ = "smoke"
        elif all(checks.values()):
            exit_ = "confirm_positive"
        else:
            exit_ = "confirm_partial"
    except (TimeoutError, FloatingPointError, RuntimeError, KeyError) as exc:
        exit_ = "engineering_failure"
        per_seed["error"] = f"{type(exc).__name__}: {exc}"

    cpu = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-n35-1-90-v1",
        "card": CARD,
        "design": design_file,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "seeds": list(seeds),
        "open_steps": open_steps,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0 if args.smoke else len(CONFIRM_SEEDS),
            "confirmation_seeds": [] if args.smoke else list(CONFIRM_SEEDS),
            "oracle_used_by_subject": False,
        },
    }
    if not args.smoke:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {"exit": exit_, "checks": checks, "cpu": cpu}
    if "error" in per_seed:
        summary["error"] = per_seed["error"]
    else:
        summary["seeds"] = {
            k: {
                "oov_q": v["open_chinese"]["open_chinese_memory_pg"]["heldout_oov_entities"][
                    "query_turn_success"
                ],
                "para_q": v["open_chinese"]["open_chinese_memory_pg"]["heldout_paraphrase"][
                    "query_turn_success"
                ],
                "cpu": v["cpu"],
            }
            for k, v in per_seed.items()
        }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ in ("confirm_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
