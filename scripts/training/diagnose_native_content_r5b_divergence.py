"""R5b divergence-point attribution probe (train-only, read-only).

Decision support for the pending R5c ruling.  For sampled scored turns whose
free generation diverges from the target, this probe asks two questions at the
exact divergence byte, under a self-prefix-consistent regime (the correct
target prefix is fed as the conditioning stream):

* guided: does the model's argmax at the divergence position match the target
  byte when given the correct prefix?
* recovery: after feeding the correct prefix, does the free continuation
  produce the correct remainder?

The answer matrix selects the concrete mechanism-D form of the R5c draft:
guided+recovery high -> inference-side anchoring; guided high/recovery low ->
decoding brittleness; guided low -> training-signal gap at template positions.

Reads the train manifest and the frozen R5b training report only; query,
holdout and sealed stay unread.  No optimizer update is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_eval import FACT_ROLES, TIME_FOR_CATEGORY, replay_episode  # noqa: E402
from taiji.native_content_training_v2 import R5BRunTrainer  # noqa: E402
from taiji.utf8_state import utf8_allowed  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5b_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5b_divergence_attribution_20261031.json"

SAMPLE_EPISODES = 216
SAMPLE_SEED = 20261030
RUN_NAME = "q1_new_core_v2_seed0"
MAX_NEW_BYTES = 48


def _guided_and_recovery(
    trainer: R5BRunTrainer,
    episode: Mapping[str, Any],
    turn: Mapping[str, Any],
    diverged: bytes,
    first: int,
) -> dict[str, Any]:
    """Feed the correct prefix, then read argmax at the divergence and free-run."""

    bundle = trainer.adapter.fresh_bundle(trainer.model)
    session = str(episode["episode_id"]).replace(":", "_")
    for other in episode["turns"]:
        if other["role"] in FACT_ROLES and int(other["index"]) < int(turn["index"]):
            raw = str(other["text"]).encode("utf-8")
            bundle.ingest(
                raw_span=raw,
                source_input_id=f"{episode['episode_id']}:{other['index']}",
                source_sha256=hashlib.sha256(raw).hexdigest(),
                owner="diag",
                scope="diag",
                session=session,
                turn_index=int(other["index"]),
                revision=0 if other["role"] in {"fact_v0", "distractor"} else 1,
                entity=other.get("entity"),
                attribute=other.get("attribute"),
                value=other.get("value"),
                status="resolved",
            )
    target = bytes(turn["target"]["target_bytes_utf8"])
    condition = bundle.condition(
        query=str(turn["text"]).encode("utf-8"),
        time_intent=TIME_FOR_CATEGORY[str(turn["category"])],
        owner="diag",
        scope="diag",
        session=session,
    )
    state = bundle.model.conditioned_state(
        condition, native_state_ref="diag:divergence", scope_id="diag", session_id=session
    )
    cache = bundle.model.content_read_cache(state, condition, store=bundle.store)
    prefix = target[:first]
    for symbol in prefix:
        logits, hidden = bundle.model.next_logits(state, cache=cache)
        state = bundle.model.observe_response(state, int(symbol), hidden)
    guided_logits, hidden = bundle.model.next_logits(state, cache=cache)
    guided_correct = int(guided_logits.argmax()) == target[first]
    generated: list[int] = list(diverged[:first])
    symbol_at_divergence = int(guided_logits.argmax())
    state = bundle.model.observe_response(state, symbol_at_divergence, hidden)
    generated.append(symbol_at_divergence)
    while len(generated) < MAX_NEW_BYTES:
        logits, hidden = bundle.model.next_logits(state, cache=cache)
        allowed = utf8_allowed(state.utf8_remaining, state.utf8_lead)
        choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        symbol = int(choices[logits[choices].argmax()])
        state = bundle.model.observe_response(state, symbol, hidden)
        generated.append(symbol)
        if bytes(generated).endswith(b"<|end|>"):
            break
    recovered = bytes(generated) == target
    return {
        "guided_argmax_correct": bool(guided_correct),
        "prefix_recovery_exact": bool(recovered),
    }


def _main(approve: bool) -> int:
    if not approve:
        raise SystemExit("diagnosis requires explicit approval (--approve-r5b)")
    if REPORT.exists():
        raise SystemExit(f"diagnosis receipt is write-once: {REPORT}")
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if payload.get("split") != "train" or not payload.get("training_read"):
        raise SystemExit("diagnosis requires the declared readable train split")
    episodes = list(payload["episodes"])
    chooser = random.Random(SAMPLE_SEED)
    sample = [
        episodes[index] for index in sorted(chooser.sample(range(len(episodes)), SAMPLE_EPISODES))
    ]
    training = json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))
    if training.get("status") != "r5b_training_complete":
        raise SystemExit("diagnosis requires the frozen R5b training report")
    started_at = datetime.now(UTC)
    started_cpu = time.process_time()
    trainer = R5BRunTrainer.load(
        ROOT / training["runs"][RUN_NAME]["endpoint_restore"]["checkpoint"]
    )
    tally: dict[str, dict[str, int]] = {}
    for episode in sample:
        result = replay_episode(trainer.adapter, trainer.model, episode)
        for turn in episode["turns"]:
            if not turn["scored"]:
                continue
            key = (str(episode["episode_id"]), int(turn["index"]))
            prediction = result.predictions[key]
            target = bytes(turn["target"]["target_bytes_utf8"])
            if prediction.produced == target:
                continue
            first = next(
                (
                    index
                    for index, (a, b) in enumerate(zip(prediction.produced, target, strict=False))
                    if a != b
                ),
                None,
            )
            if first is None:
                continue
            outcome = _guided_and_recovery(trainer, episode, turn, prediction.produced, first)
            label = (
                "guided_recovered" if outcome["prefix_recovery_exact"] else ("guided_not_recovered")
            )
            if outcome["guided_argmax_correct"]:
                label = "guided_correct_" + label
            else:
                label = "guided_wrong_" + label
            bucket = tally.setdefault(str(turn["category"]), {})
            bucket[label] = bucket.get(label, 0) + 1
    receipt = {
        "format": "taiji-native-core-r5b-divergence-attribution-v1",
        "version": 1,
        "date": "2026-10-31",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md",
        "run": RUN_NAME,
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            "episodes_sampled": len(sample),
            "sample_seed": SAMPLE_SEED,
        },
        "definition": "at the first divergence byte: guided argmax under the correct prefix, and exact recovery of the full answer when continuing freely from that prefix",
        "tally": tally,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": 0,
        },
        "cost": {"cpu_seconds": time.process_time() - started_cpu},
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    REPORT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "diagnosis_complete", "report": str(REPORT)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approve-r5b",
        action="store_true",
        help="required: covered by the user-approved R5b execution (read-only diagnosis)",
    )
    args = parser.parse_args()
    return _main(approve=args.approve_r5b)


if __name__ == "__main__":
    raise SystemExit(main())
