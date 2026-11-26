"""NCH.3.05 (exploration): Conv1d + Turn-GRU + SDPA v4 subject for coref margin & 2.5x speedup.

Design: plans/reference/DESIGN-NCH-3-05_20261126.md.
Trains `taiji-open-chinese-subject-v4` on exploration seeds 7101-7103 with zero changes to
`taiji/open_chinese_task_env.py`, zero confirmation seeds, and zero check/final reads.

Usage: python scripts/training/nch_3_05_open_chinese_v4.py [--smoke]
Output: reports/nch_3_05_open_chinese_v4.json
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
from torch import nn  # noqa: E402

from taiji.open_chinese_subject import (  # noqa: E402
    OpenChineseCognitiveSubject,
    build_turn_memory_tensors,
    collate_memory_batch,
    encode_target_blocks,
)
from taiji.open_chinese_task_env import (  # noqa: E402
    BROAD_TRAIN_CJK,
    HELDOUT_DISTRACTORS,
    HELDOUT_PARAPHRASE_PATTERNS,
    OOV_ENTITIES,
    OOV_PLACES,
    TRAIN_DISTRACTORS,
    TRAIN_ENTITIES,
    TRAIN_PATTERNS,
    TRAIN_PLACES,
    OpenChineseTaskEnv,
    eval_open_policy,
    load_external_chinese_distractors,
    oracle_policy,
    random_policy,
)
from taiji.session_store import SessionStore  # noqa: E402

CARD = "NCH.3.05"
DESIGN = "DESIGN-NCH-3-05_20261126.md"
TRAIN_SEEDS = (7101, 7102, 7103)
SPOT_CHECK_SEED = 7101
EVAL_TRAIN_SEEDS = list(range(8100, 8160))
EVAL_PARA_SEEDS = list(range(8200, 8260))
EVAL_OOV_SEEDS = list(range(8300, 8360))
EVAL_WILD_SEEDS = list(range(8400, 8460))

STEPS = 720
WARMUP_STEPS = 40
SWA_START = 580
SWA_EVERY = 20
BATCH_TURNS = 16
LR = 3e-3
LR_MIN = 1e-4
WEIGHT_DECAY = 1e-3
OPEN_LEXICON_PROB = 0.5
SMOKE_STEPS = 25
CAP_CPU = 520.0
OUT = ROOT / "reports" / "nch_3_05_open_chinese_v4.json"


def _sample_batch(
    ep_counter: int,
    seed: int,
    *,
    use_history: bool,
    gen: torch.Generator,
) -> tuple[list[dict[str, Any]], int]:
    """Collect 16 balanced turns, computing memory tensors only for kept turns."""
    samples: list[dict[str, Any]] = []
    ack_budget = 2
    while len(samples) < BATCH_TURNS:
        ep_seed = int(seed) * 1_000_000 + ep_counter
        ep_counter += 1
        env = OpenChineseTaskEnv(ep_seed, split="train_seen", open_lexicon_prob=OPEN_LEXICON_PROB)
        obs = env.reset()
        history: list[list[int]] = [list(obs["utf8_bytes"])]
        ep_raw: list[tuple[list[list[int]], int, str, bool]] = []
        while True:
            ep_raw.append(
                (
                    [list(h) for h in history],
                    env.intended_intent,
                    env.intended_answer,
                    env.current_meta["is_query"],
                )
            )
            obs_next, _, done = env.step(env.intended_intent, env.intended_answer)
            if done:
                break
            history.append(list(obs_next["utf8_bytes"]))

        for hist_snap, intent_id, ans, is_q in ep_raw:
            if is_q and len(samples) < BATCH_TURNS:
                mb, mf = build_turn_memory_tensors(hist_snap, use_history=use_history)
                samples.append(
                    {
                        "mem_blocks": mb,
                        "mem_feats": mf,
                        "target_intent": intent_id,
                        "target_blocks": encode_target_blocks(ans),
                    }
                )
        if ack_budget > 0 and len(samples) < BATCH_TURNS:
            non_queries = [r for r in ep_raw if not r[3]]
            pick = int(torch.randint(0, len(non_queries), (), generator=gen))
            hist_snap, intent_id, ans, _ = non_queries[pick]
            mb, mf = build_turn_memory_tensors(hist_snap, use_history=use_history)
            samples.append(
                {
                    "mem_blocks": mb,
                    "mem_feats": mf,
                    "target_intent": intent_id,
                    "target_blocks": encode_target_blocks(ans),
                }
            )
            ack_budget -= 1
    return samples[:BATCH_TURNS], ep_counter


def train_arm(
    seed: int,
    *,
    use_history: bool,
    use_copy: bool,
    steps: int,
    deadline: float,
) -> tuple[OpenChineseCognitiveSubject, float, float]:
    """Train one v4 subject arm deterministically with warmup + cosine LR + SWA."""
    subj = OpenChineseCognitiveSubject(seed, use_history=use_history, use_copy=use_copy)
    opt = torch.optim.AdamW(subj.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < WARMUP_STEPS:
            return float(step_idx + 1) / float(WARMUP_STEPS)
        progress = float(step_idx - WARMUP_STEPS) / float(max(1, steps - WARMUP_STEPS))
        cos_val = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
        min_ratio = LR_MIN / LR
        return min_ratio + (1.0 - min_ratio) * cos_val

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    gen = torch.Generator().manual_seed(
        int(seed) * 7919 + (11 if use_history else 31) + (0 if use_copy else 53)
    )
    ep_counter = 0
    init_loss = 0.0
    final_loss = 0.0
    swa_sum: dict[str, torch.Tensor] | None = None
    swa_count = 0

    subj.train()
    for step in range(steps):
        if time.process_time() > deadline:
            raise TimeoutError(
                f"nch_3_05 timeout hist={use_history} copy={use_copy} seed={seed} step={step}"
            )
        batch_samples, ep_counter = _sample_batch(
            ep_counter, seed, use_history=use_history, gen=gen
        )
        batch = collate_memory_batch(batch_samples)
        opt.zero_grad(set_to_none=True)
        loss, _, _ = subj.forward_loss(*batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss hist={use_history} copy={use_copy} seed={seed} step={step}"
            )
        loss.backward()
        nn.utils.clip_grad_norm_(subj.parameters(), 2.0)
        opt.step()
        sched.step()
        val = float(loss.item())
        if step == 0:
            init_loss = val
        final_loss = val

        step_num = step + 1
        if steps >= SWA_START and step_num >= SWA_START and step_num % SWA_EVERY == 0:
            if swa_sum is None:
                swa_sum = {k: v.detach().clone() for k, v in subj.state_dict().items()}
            else:
                for k, v in subj.state_dict().items():
                    swa_sum[k].add_(v.detach())
            swa_count += 1

    if swa_sum is not None and swa_count > 1:
        avg_state = {
            k: (v / float(swa_count) if v.is_floating_point() else v) for k, v in swa_sum.items()
        }
        subj.load_state_dict(avg_state, strict=True)

    subj.eval()
    return subj, round(init_loss, 4), round(final_loss, 4)


def _eval_subject(
    subj: OpenChineseCognitiveSubject,
    seeds: list[int],
    *,
    split: str,
    ablate_history: bool = False,
    external_distractors: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    ablate_gen = torch.Generator().manual_seed(20261126) if ablate_history else None

    def _policy(_env: OpenChineseTaskEnv, history: list[list[int]]) -> tuple[int, str, bool]:
        return subj.predict_turn(history, ablate_history_gen=ablate_gen)

    return eval_open_policy(
        _policy,
        seeds,
        split=split,
        external_distractors=external_distractors,
    )


def _calibrate(ext_distractors: tuple[str, ...]) -> dict[str, Any]:
    oov_chars = set("".join(OOV_ENTITIES)) | set("".join(OOV_PLACES))
    disjoint_ok = (
        set("".join(TRAIN_ENTITIES)).isdisjoint("".join(OOV_ENTITIES))
        and set("".join(TRAIN_PLACES)).isdisjoint("".join(OOV_PLACES))
        and set(BROAD_TRAIN_CJK).isdisjoint(oov_chars)
        and set(TRAIN_DISTRACTORS).isdisjoint(set(HELDOUT_DISTRACTORS))
        and all(
            set(TRAIN_PATTERNS[k]).isdisjoint(set(HELDOUT_PARAPHRASE_PATTERNS[k]))
            for k in TRAIN_PATTERNS
        )
    )
    ora_tr = eval_open_policy(oracle_policy, EVAL_TRAIN_SEEDS[:20], split="train_seen")
    ora_pa = eval_open_policy(oracle_policy, EVAL_PARA_SEEDS[:20], split="heldout_paraphrase")
    ora_oov = eval_open_policy(oracle_policy, EVAL_OOV_SEEDS[:20], split="heldout_oov_entities")
    rnd_oov = eval_open_policy(
        random_policy,
        EVAL_OOV_SEEDS[:20],
        split="heldout_oov_entities",
        gen=torch.Generator().manual_seed(0),
    )
    ora_wild = None
    if ext_distractors:
        ora_wild = eval_open_policy(
            oracle_policy,
            EVAL_WILD_SEEDS[:20],
            split="heldout_oov_entities",
            external_distractors=ext_distractors,
        )
    c0_pass = (
        disjoint_ok
        and ora_tr["episode_success"] == 1.0
        and ora_pa["episode_success"] == 1.0
        and ora_oov["episode_success"] == 1.0
        and (ora_wild is None or ora_wild["episode_success"] == 1.0)
        and rnd_oov["query_turn_success"] <= 0.05
    )
    return {
        "c0_pass": c0_pass,
        "disjoint_oov": disjoint_ok,
        "broad_cjk_pool": len(BROAD_TRAIN_CJK),
        "oracle_train": ora_tr["episode_success"],
        "oracle_paraphrase": ora_pa["episode_success"],
        "oracle_oov": ora_oov["episode_success"],
        "oracle_wild": ora_wild["episode_success"] if ora_wild else None,
        "random_oov_query_success": rnd_oov["query_turn_success"],
        "external_real_distractors": len(ext_distractors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + CAP_CPU
    steps = SMOKE_STEPS if args.smoke else STEPS

    ext_distractors = load_external_chinese_distractors(
        ROOT / "data/simple_zh/dialogue_extended_clean.jsonl", max_items=48
    )
    calib = _calibrate(ext_distractors)

    per_seed: dict[str, Any] = {}
    checks: dict[str, bool] = {}

    if not calib["c0_pass"]:
        exit_ = "engineering_failure"
    elif args.smoke:
        train_arm(SPOT_CHECK_SEED, use_history=True, use_copy=True, steps=5, deadline=deadline)
        t0 = time.process_time()
        subj, init_l, final_l = train_arm(
            SPOT_CHECK_SEED, use_history=True, use_copy=True, steps=steps, deadline=deadline
        )
        train_dt = time.process_time() - t0
        oov_eval = _eval_subject(subj, EVAL_OOV_SEEDS[:10], split="heldout_oov_entities")
        ms_per_step = round(train_dt / max(1, steps) * 1000.0, 2)
        projected_full_cpu = round((train_dt / max(1, steps)) * STEPS * 7.8 + 25.0, 1)
        checks = {
            "c0_calibration": True,
            "smoke_finite": True,
            "smoke_utf8_valid": oov_eval["utf8_valid_rate"] == 1.0,
            "smoke_loss_decreased": final_l < init_l,
        }
        per_seed[str(SPOT_CHECK_SEED)] = {
            "init_loss": init_l,
            "final_loss": final_l,
            "ms_per_step": ms_per_step,
            "projected_full_cpu": projected_full_cpu,
            "sample_oov_eval": oov_eval,
        }
        exit_ = "smoke" if all(checks.values()) else "engineering_failure"
    else:
        try:
            for seed in TRAIN_SEEDS:
                t0 = time.process_time()
                arm_main, m_l0, m_l1 = train_arm(
                    seed, use_history=True, use_copy=True, steps=steps, deadline=deadline
                )
                arm_stat, s_l0, s_l1 = train_arm(
                    seed, use_history=False, use_copy=True, steps=steps, deadline=deadline
                )
                arm_nocopy, n_l0, n_l1 = train_arm(
                    seed, use_history=True, use_copy=False, steps=steps, deadline=deadline
                )

                main_tr = _eval_subject(arm_main, EVAL_TRAIN_SEEDS, split="train_seen")
                main_pa = _eval_subject(arm_main, EVAL_PARA_SEEDS, split="heldout_paraphrase")
                main_oov = _eval_subject(arm_main, EVAL_OOV_SEEDS, split="heldout_oov_entities")
                main_wild = (
                    _eval_subject(
                        arm_main,
                        EVAL_WILD_SEEDS,
                        split="heldout_oov_entities",
                        external_distractors=ext_distractors,
                    )
                    if ext_distractors
                    else None
                )
                main_ablate = _eval_subject(
                    arm_main,
                    EVAL_OOV_SEEDS,
                    split="heldout_oov_entities",
                    ablate_history=True,
                )

                stat_oov = _eval_subject(arm_stat, EVAL_OOV_SEEDS, split="heldout_oov_entities")
                nocopy_oov = _eval_subject(arm_nocopy, EVAL_OOV_SEEDS, split="heldout_oov_entities")

                seed_cpu = round(time.process_time() - t0, 2)
                weights_exact = None
                store_exact = None
                if seed == SPOT_CHECK_SEED:
                    again, _, _ = train_arm(
                        seed, use_history=True, use_copy=True, steps=steps, deadline=deadline
                    )
                    weights_exact = all(
                        torch.equal(a, b)
                        for a, b in zip(arm_main.parameters(), again.parameters(), strict=True)
                    )
                    with tempfile.TemporaryDirectory() as tmp:
                        store = SessionStore(tmp)
                        arm_main.save_to_session_store(store, f"nch_3_05_{seed}", session_id=1)
                        reloaded = OpenChineseCognitiveSubject.load_from_session_store(
                            store, f"nch_3_05_{seed}"
                        )
                        rel_oov = _eval_subject(
                            reloaded, EVAL_OOV_SEEDS, split="heldout_oov_entities"
                        )
                        store_exact = (
                            all(
                                torch.equal(a, b)
                                for a, b in zip(
                                    arm_main.parameters(), reloaded.parameters(), strict=True
                                )
                            )
                            and rel_oov == main_oov
                        )

                per_seed[str(seed)] = {
                    "losses": {
                        "main": [m_l0, m_l1],
                        "stateless": [s_l0, s_l1],
                        "no_copy": [n_l0, n_l1],
                    },
                    "open_chinese_memory_pg": {
                        "train_seen": main_tr,
                        "heldout_paraphrase": main_pa,
                        "heldout_oov_entities": main_oov,
                        "real_corpus_distractor": main_wild,
                    },
                    "history_shuffle_ablation_oov": main_ablate,
                    "stateless_single_turn_pg_oov": stat_oov,
                    "no_copy_vocab_only_oov": nocopy_oov,
                    "weights_exact": weights_exact,
                    "store_roundtrip_exact": store_exact,
                    "cpu": seed_cpu,
                }

            checks = {
                "c0_calibration": bool(calib["c0_pass"]),
                "g1_train_seen": all(
                    v["open_chinese_memory_pg"]["train_seen"]["query_turn_success"] >= 0.85
                    and v["open_chinese_memory_pg"]["train_seen"]["query_answer_em"] >= 0.85
                    and v["open_chinese_memory_pg"]["train_seen"]["all_turn_success"] >= 0.85
                    and v["open_chinese_memory_pg"]["train_seen"]["utf8_valid_rate"] == 1.0
                    for v in per_seed.values()
                ),
                "g2_heldout_paraphrase": all(
                    v["open_chinese_memory_pg"]["heldout_paraphrase"]["query_turn_success"] >= 0.75
                    and v["open_chinese_memory_pg"]["heldout_paraphrase"]["coref_success"] >= 0.70
                    and v["open_chinese_memory_pg"]["heldout_paraphrase"]["corr_neg_success"]
                    >= 0.70
                    for v in per_seed.values()
                ),
                "g3_heldout_oov": all(
                    v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_turn_success"]
                    >= 0.75
                    and v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_answer_em"]
                    >= 0.75
                    and v["open_chinese_memory_pg"]["heldout_oov_entities"][
                        "unknown_abstain_success"
                    ]
                    >= 0.75
                    for v in per_seed.values()
                ),
                "g4_causal_and_open_controls": all(
                    (
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
                    for v in per_seed.values()
                ),
                "g5_repro_and_persistence": all(
                    v["weights_exact"] is True and v["store_roundtrip_exact"] is True
                    for v in per_seed.values()
                    if v["weights_exact"] is not None
                ),
                "g6_coref_margin_and_efficiency": all(
                    v["open_chinese_memory_pg"]["heldout_paraphrase"]["coref_success"] >= 0.80
                    and v["open_chinese_memory_pg"]["heldout_oov_entities"]["coref_success"] >= 0.80
                    and v["open_chinese_memory_pg"]["heldout_oov_entities"]["query_turn_success"]
                    >= 0.80
                    and v["open_chinese_memory_pg"]["heldout_paraphrase"]["corr_neg_success"]
                    >= 0.75
                    and v["cpu"] <= 140.0
                    for v in per_seed.values()
                ),
            }
            if all(checks.values()):
                exit_ = "signal_positive"
            elif checks["c0_calibration"] and checks["g4_causal_and_open_controls"]:
                exit_ = "signal_partial"
            else:
                exit_ = "signal_negative"
        except (TimeoutError, FloatingPointError) as exc:
            exit_ = "engineering_failure"
            per_seed["error"] = str(exc)

    cpu = round(time.process_time() - started, 2)
    payload = {
        "schema": "taiji-nch-3-05-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "steps": steps,
        "batch_turns": BATCH_TURNS,
        "open_lexicon_prob": OPEN_LEXICON_PROB,
        "train_seeds": list(TRAIN_SEEDS),
        "calibration": calib,
        "per_seed": per_seed,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "oracle_used_in_training": False,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit": exit_, "checks": checks, "cpu": cpu, "per_seed": per_seed},
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("signal_positive", "smoke") else 3


if __name__ == "__main__":
    raise SystemExit(main())
