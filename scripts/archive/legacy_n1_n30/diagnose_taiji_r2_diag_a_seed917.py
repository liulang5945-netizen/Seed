"""Diagnosis (a): is seed-20260917's exact=0.25 real ability or coincidence?

Trains the lambda=0 control arm at the H3.8 budget (30 x 25, seed 20260917),
then answers the coincidence question directly:

* for every dev episode, was the **first response byte** the training-corpus mode
  byte at that position class (i.e. the high-frequency colour the model defaults to)?
* are the exact-hit episodes systematically the **short / high-frequency** shapes
  (``白``/``黄``) while the misses are the longer ones (``不是白``)?

Read-only w.r.t. existing artifacts: writes only its own checkpoint/report.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.sequence_workspace import (  # noqa: E402
    SequenceWorkspaceConfig,
    SequenceWorkspacePrototype,
    SequenceWorkspaceTrainer,
)

FIXTURE = Path("tests/fixtures/r2_h3_8_joint_sequence_v1.jsonl")
SEED = 20260917
EPOCHS = 30
MAX_EPISODES = 25
LEARNING_RATE = 0.01
GENERATION_LIMIT = 64
CHECKPOINT = Path("reports/r2_hgen_diag_a_seed917_control.pt")
REPORT = Path("reports/r2_hgen_diag_a_20260917.json")
BOUNDARY = int(SequenceWorkspaceConfig().boundary_symbol)


def _load_split(split: str) -> tuple[tuple[bytes, bytes], ...]:
    # Sorted exactly like the H-OBJ arm loader, so this training run reproduces
    # that arm's checkpoint byte-for-byte (episode order changes the result).
    episodes: list[tuple[bytes, bytes]] = []
    for line in (PROJECT_ROOT / FIXTURE).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["split"] == split:
            episodes.append((record["prefix"].encode("utf-8"), record["response"].encode("utf-8")))
    if not episodes:
        raise RuntimeError(f"{split} split is empty")
    return tuple(sorted(episodes))


def main() -> int:
    train_split = _load_split("train")
    dev_split = _load_split("dev")

    # The corpus mode response and its first byte (the "default answer" hypothesis).
    response_counter = collections.Counter(response for _, response in train_split)
    mode_response, mode_count = response_counter.most_common(1)[0]
    mode_first_byte = mode_response[0]

    torch.manual_seed(SEED)
    prototype = SequenceWorkspacePrototype(SequenceWorkspaceConfig(seed=SEED))
    trainer = SequenceWorkspaceTrainer(prototype, learning_rate=LEARNING_RATE)
    trainer.set_episodes(train_split)
    zero_digest = trainer.checkpoint()["checkpoint_digest"]
    for _ in range(EPOCHS):
        trainer.train_epoch(max_episodes=MAX_EPISODES)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for prefix, response in dev_split:
            result = prototype.generate(prefix, max_bytes=GENERATION_LIMIT)
            produced = bytes(result.bytes_out)
            rows.append(
                {
                    "reference": response.decode("utf-8", errors="replace"),
                    "generated": produced.decode("utf-8", errors="replace"),
                    "exact": produced == response,
                    "ref_len": len(response),
                    "gen_len": len(produced),
                    "first_byte_is_mode": bool(response) and response[0] == mode_first_byte,
                    "generated_is_mode": produced == mode_response,
                }
            )

    hits = [row for row in rows if row["exact"]]
    misses = [row for row in rows if not row["exact"]]

    def _share(items: list[dict[str, Any]], key: str) -> float:
        return sum(1 for item in items if item[key]) / max(1, len(items))

    payload: dict[str, Any] = {
        "format": "taiji-r2-hgen-diag-a-v1",
        "seed": SEED,
        "epochs": EPOCHS,
        "max_episodes": MAX_EPISODES,
        "zero_step_checkpoint_digest": zero_digest,
        "final_checkpoint_digest": trainer.checkpoint()["checkpoint_digest"],
        "corpus_mode_response": mode_response.decode("utf-8", errors="replace"),
        "corpus_mode_count": mode_count,
        "dev_episodes": len(rows),
        "exact_hits": len(hits),
        "hit_rows": hits,
        "miss_rows": misses,
        "coincidence_tests": {
            "hit_first_byte_is_mode_share": _share(hits, "first_byte_is_mode"),
            "miss_first_byte_is_mode_share": _share(misses, "first_byte_is_mode"),
            "hit_generated_is_mode_share": _share(hits, "generated_is_mode"),
            "hit_mean_ref_len": sum(row["ref_len"] for row in hits) / max(1, len(hits)),
            "miss_mean_ref_len": sum(row["ref_len"] for row in misses) / max(1, len(misses)),
        },
        "interpretation_rule": (
            "If hit episodes are dominated by short/high-frequency shapes whose first byte IS "
            "the corpus mode byte (and generation mostly echoes the mode response), the 0.25 "
            "is coincidence: the model repeats the marginal mode and sometimes the mode IS the "
            "answer. If hits include long/low-frequency answers with non-mode first bytes, "
            "positional conditioning exists to some degree."
        ),
    }
    target = PROJECT_ROOT / REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trainer.save(PROJECT_ROOT / CHECKPOINT)

    print(f"corpus mode response: {payload['corpus_mode_response']!r} x{mode_count}")
    print(f"dev exact hits: {len(hits)}/{len(rows)}")
    print("hit rows:")
    for row in hits:
        print(
            f"   ref={row['reference']!r} gen={row['generated']!r} "
            f"ref_len={row['ref_len']} first_byte_is_mode={row['first_byte_is_mode']} "
            f"generated_is_mode={row['generated_is_mode']}"
        )
    tests = payload["coincidence_tests"]
    print(
        f"first_byte_is_mode: hits {_share(hits, 'first_byte_is_mode'):.2f} "
        f"vs misses {_share(misses, 'first_byte_is_mode'):.2f}"
    )
    print(
        f"generated_is_mode (hits): {_share(hits, 'generated_is_mode'):.2f} | "
        f"mean ref len: hits {tests['hit_mean_ref_len']:.1f} vs misses {tests['miss_mean_ref_len']:.1f}"
    )
    print(f"report -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
