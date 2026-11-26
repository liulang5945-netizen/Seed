"""NCH.3.01 (zero training): calibration of the open-vocabulary Chinese continuous-task harness.

Design: plans/reference/DESIGN-NCH-3-01_20261126.md.
Verifies character-disjoint OOV pools, zero answer leakage, oracle=1.0 / random<=0.05 across
train_seen / heldout_paraphrase / heldout_oov_entities (and external simple_zh distractors if
present), plus strict utf8_state subject decoding and SessionStore capsule roundtrip.
Zero training, zero confirmation seeds, zero check/final.

Usage: python scripts/training/nch_3_01_open_calibration.py
Output: reports/nch_3_01_open_calibration.json
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.open_chinese_subject import OpenChineseCognitiveSubject  # noqa: E402
from taiji.open_chinese_task_env import (  # noqa: E402
    HELDOUT_DISTRACTORS,
    HELDOUT_PARAPHRASE_PATTERNS,
    INTENTS,
    OOV_ENTITIES,
    OOV_PLACES,
    TASK_VERSION,
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

CARD = "NCH.3.01"
DESIGN = "DESIGN-NCH-3-01_20261126.md"
SPLITS = (
    ("train_seen", list(range(8100, 8140))),
    ("heldout_paraphrase", list(range(8200, 8240))),
    ("heldout_oov_entities", list(range(8300, 8340))),
)
CAP_CPU = 10.0
OUT = ROOT / "reports" / "nch_3_01_open_calibration.json"


def main() -> int:
    torch.set_num_threads(1)
    started = time.process_time()

    disjoint_ok = (
        set("".join(TRAIN_ENTITIES)).isdisjoint("".join(OOV_ENTITIES))
        and set("".join(TRAIN_PLACES)).isdisjoint("".join(OOV_PLACES))
        and set(TRAIN_DISTRACTORS).isdisjoint(set(HELDOUT_DISTRACTORS))
        and all(
            set(TRAIN_PATTERNS[k]).isdisjoint(set(HELDOUT_PARAPHRASE_PATTERNS[k]))
            for k in TRAIN_PATTERNS
        )
    )

    leak_free = True
    obs_clean = True
    oracle_results: dict[str, dict] = {}
    random_results: dict[str, dict] = {}

    for idx, (split_name, seeds) in enumerate(SPLITS):
        oracle_results[split_name] = eval_open_policy(oracle_policy, seeds, split=split_name)
        random_results[split_name] = eval_open_policy(
            random_policy,
            seeds,
            split=split_name,
            gen=torch.Generator().manual_seed(idx + 17),
        )
        for s in seeds:
            env = OpenChineseTaskEnv(s, split=split_name)
            obs = env.reset()
            if set(obs) != {"text", "utf8_bytes", "step", "turns"}:
                obs_clean = False
            while True:
                if env.current_meta["is_query"] and env.intended_answer in env.current_text():
                    leak_free = False
                _, _, done = env.step(env.intended_intent, env.intended_answer)
                if done:
                    break

    ext_distractors = load_external_chinese_distractors(
        ROOT / "data/simple_zh/dialogue_extended_clean.jsonl", max_items=48
    )
    ext_oracle = None
    if ext_distractors:
        ext_oracle = eval_open_policy(
            oracle_policy,
            list(range(8400, 8420)),
            split="heldout_oov_entities",
            external_distractors=ext_distractors,
        )

    # Verify untrained subject UTF-8 constraint & SessionStore capsule exactness
    subj = OpenChineseCognitiveSubject(7001, use_history=True, use_copy=True)
    probe_env = OpenChineseTaskEnv(8300, split="heldout_oov_entities")
    obs = probe_env.reset()
    history = [list(obs["utf8_bytes"])]
    while not probe_env.step(probe_env.intended_intent, probe_env.intended_answer)[2]:
        history.append(probe_env.current_bytes())

    utf8_subject_ok = True
    preds_before = []
    for t in range(len(history)):
        iid, txt, valid = subj.predict_turn(history[: t + 1])
        if not (0 <= iid < len(INTENTS)) or not valid:
            utf8_subject_ok = False
        preds_before.append((iid, txt, valid))

    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(tmp)
        subj.save_to_session_store(store, "nch_3_01_probe", session_id=1)
        reloaded = OpenChineseCognitiveSubject.load_from_session_store(store, "nch_3_01_probe")
        weights_exact = all(
            torch.equal(a, b) for a, b in zip(subj.parameters(), reloaded.parameters(), strict=True)
        )
        preds_after = [reloaded.predict_turn(history[: t + 1]) for t in range(len(history))]
        store_exact = weights_exact and (preds_before == preds_after)

    checks = {
        "disjoint_oov_and_patterns": disjoint_ok,
        "zero_query_answer_leak": leak_free,
        "obs_isolation_clean": obs_clean,
        "oracle_all_splits_exact": all(
            v["all_turn_success"] == 1.0 and v["episode_success"] == 1.0
            for v in oracle_results.values()
        )
        and (ext_oracle is None or ext_oracle["episode_success"] == 1.0),
        "random_low_across_splits": all(
            v["query_turn_success"] <= 0.05 for v in random_results.values()
        ),
        "utf8_subject_and_store_ready": utf8_subject_ok and store_exact,
    }
    cpu = round(time.process_time() - started, 2)
    exit_ = "open_env_ready" if (all(checks.values()) and cpu <= CAP_CPU) else "open_env_failure"
    payload = {
        "schema": "taiji-nch-3-01-v1",
        "card": CARD,
        "design": DESIGN,
        "created_utc": datetime.now(UTC).isoformat(),
        "task_version": TASK_VERSION,
        "pool_sizes": {
            "train_entities": len(TRAIN_ENTITIES),
            "oov_entities": len(OOV_ENTITIES),
            "train_places": len(TRAIN_PLACES),
            "oov_places": len(OOV_PLACES),
            "train_distractors": len(TRAIN_DISTRACTORS),
            "heldout_distractors": len(HELDOUT_DISTRACTORS),
            "external_real_distractors": len(ext_distractors),
        },
        "oracle_results": oracle_results,
        "random_results": random_results,
        "external_corpus_oracle": ext_oracle,
        "checks": checks,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": CAP_CPU},
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "trained_policies": 0,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exit": exit_, "checks": checks, "cpu": cpu}, ensure_ascii=False))
    return 0 if exit_ == "open_env_ready" else 3


if __name__ == "__main__":
    raise SystemExit(main())
