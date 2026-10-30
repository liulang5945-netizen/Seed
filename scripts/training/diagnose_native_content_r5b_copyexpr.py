"""R5b copy-expression failure decomposition (train-only, read-only).

The selection diagnosis showed zero selection errors; the whole gap is
``selection_correct_answer_wrong``.  This script decides which successor
target that implies, using the frozen q1 seed0 endpoint on the same 216
sampled train episodes:

* TF-guided argmax exact rate over the value bytes vs the template bytes -
  if the model already emits the right bytes under teacher forcing, the
  failure is inference-side drift; if it fails under forcing too, the learned
  copy mixture underweights the value bytes;
* first-divergence position of the wrong free generations - template bytes,
  value bytes, or the tail after the value.

Reads the train manifest and frozen R5b training report only; query, holdout
and sealed stay unread.  No optimizer update is performed.
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

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5b_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5b_copyexpr_diagnosis_20261031.json"

SAMPLE_EPISODES = 216
SAMPLE_SEED = 20261030
RUN_NAME = "q1_new_core_v2_seed0"


def _expected_value(episode: Mapping[str, Any], turn: Mapping[str, Any]) -> str | None:
    history = [
        candidate
        for candidate in episode["turns"]
        if int(candidate["index"]) < int(turn["index"])
        and candidate["role"] in FACT_ROLES
        and candidate.get("entity") == turn.get("entity")
        and candidate.get("attribute") == turn.get("attribute")
    ]
    history.sort(key=lambda item: int(item["index"]))
    category = str(turn["category"])
    if category == "current_query" and history:
        return str(history[-1].get("value"))
    if category == "previous_query" and len(history) >= 2:
        return str(history[-2].get("value"))
    return None


def _value_span(target: bytes, value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    raw = value.encode("utf-8")
    index = target.find(raw)
    if index < 0:
        return None
    return index, index + len(raw)


def _argmax_rate(rows: torch.Tensor, target: bytes, span: tuple[int, int] | None) -> dict[str, Any]:
    """Teacher-forced argmax match rate over value bytes and template bytes."""

    prediction = rows.argmax(dim=-1).tolist()
    inside = outside = inside_hit = outside_hit = 0
    for index, symbol in enumerate(target):
        hit = int(prediction[index]) == symbol
        if span is not None and span[0] <= index < span[1]:
            inside += 1
            inside_hit += int(hit)
        else:
            outside += 1
            outside_hit += int(hit)
    return {
        "value_bytes": inside,
        "value_argmax_exact": inside_hit / inside if inside else None,
        "template_bytes": outside,
        "template_argmax_exact": outside_hit / outside if outside else None,
    }


def _divergence(produced: bytes, target: bytes, span: tuple[int, int] | None) -> str:
    first = next(
        (index for index, (a, b) in enumerate(zip(produced, target, strict=False)) if a != b),
        None,
    )
    if first is None:
        return "length_divergence" if len(produced) != len(target) else "identical"
    if span is None:
        return "template_no_value_span"
    if first < span[0]:
        return "template_before_value"
    if first < span[1]:
        return "value_bytes"
    return "tail_after_value"


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
    adapter = trainer.adapter
    tf_stats: dict[str, dict[str, Any]] = {}
    divergence: dict[str, dict[str, int]] = {}
    for episode in sample:
        session = str(episode["episode_id"]).replace(":", "_")
        bundle = adapter.fresh_bundle(trainer.model)
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                raw = str(turn["text"]).encode("utf-8")
                bundle.ingest(
                    raw_span=raw,
                    source_input_id=f"{episode['episode_id']}:{turn['index']}",
                    source_sha256=hashlib.sha256(raw).hexdigest(),
                    owner="diag",
                    scope="diag",
                    session=session,
                    turn_index=int(turn["index"]),
                    revision=0 if turn["role"] in {"fact_v0", "distractor"} else 1,
                    entity=turn.get("entity"),
                    attribute=turn.get("attribute"),
                    value=turn.get("value"),
                    status="resolved",
                )
            if not turn["scored"]:
                continue
            target = bytes(turn["target"]["target_bytes_utf8"])
            category = str(turn["category"])
            query = str(turn["text"]).encode("utf-8")
            rows = bundle.score_target(
                query=query,
                target=target,
                time_intent=TIME_FOR_CATEGORY[category],
                owner="diag",
                scope="diag",
                session=session,
            )
            value = _expected_value(episode, turn)
            span = _value_span(target, value)
            stats = _argmax_rate(rows, target, span)
            bucket = tf_stats.setdefault(
                category,
                {
                    "value_hits": 0.0,
                    "value_bytes": 0,
                    "template_hits": 0.0,
                    "template_bytes": 0,
                    "turns": 0,
                },
            )
            bucket["value_hits"] += stats["value_argmax_exact"] or 0.0
            bucket["value_bytes"] += stats["value_bytes"]
            bucket["template_hits"] += stats["template_argmax_exact"] or 0.0
            bucket["template_bytes"] += stats["template_bytes"]
            bucket["turns"] += 1
        result = replay_episode(adapter, trainer.model, episode)
        for turn in episode["turns"]:
            if not turn["scored"]:
                continue
            key = (str(episode["episode_id"]), int(turn["index"]))
            prediction = result.predictions[key]
            target = bytes(turn["target"]["target_bytes_utf8"])
            if prediction.produced == target:
                continue
            value = _expected_value(episode, turn)
            span = _value_span(target, value)
            label = _divergence(prediction.produced, target, span)
            divergence.setdefault(str(turn["category"]), {})
            divergence[str(turn["category"])][label] = (
                divergence[str(turn["category"])].get(label, 0) + 1
            )

    tf_summary = {
        category: {
            "turns": bucket["turns"],
            "value_argmax_exact": (
                bucket["value_hits"] / bucket["value_bytes"] if bucket["value_bytes"] else None
            ),
            "template_argmax_exact": (
                bucket["template_hits"] / bucket["template_bytes"]
                if bucket["template_bytes"]
                else None
            ),
        }
        for category, bucket in sorted(tf_stats.items())
    }
    receipt = {
        "format": "taiji-native-core-r5b-copyexpr-diagnosis-v1",
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
        "definition": "TF-guided argmax exact over value/template bytes plus first-divergence classes of wrong free generations",
        "tf_guided_argmax": tf_summary,
        "free_generation_divergence": divergence,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": 0,
        },
        "cost": {
            "cpu_seconds": time.process_time() - started_cpu,
        },
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
