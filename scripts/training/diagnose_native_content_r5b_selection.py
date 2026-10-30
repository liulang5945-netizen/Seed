"""R5b train-only selection-error diagnosis (read-only, no training).

Step 08 closed as FAIL_BEHAVIOR with one remaining gap: the answer now truly
consumes the selected content (cut_drop 1.0), but the reader picks the correct
value only about half the time (current_query 0.352).  This script decomposes
that gap on the TRAIN split only, using the frozen q1 endpoints:

for every sampled scored current/previous/unknown turn, replay the episode,
take the first non-empty step selection, and classify it against the
course-defined ground truth (latest/previous fact of the same entity+attribute
before the query turn):

* selection_correct + answer_correct;
* selection_correct + answer wrong      (copy/expression failure);
* same entity+attribute, wrong value    (time/revision selection error);
* wrong entity/attribute or no selection (retrieval error).

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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_content_eval import (  # noqa: E402
    FACT_ROLES,
    replay_episode,
)
from taiji.native_content_training_v2 import R5BRunTrainer  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
TRAINING_REPORT = ROOT / "reports/taiji_native_core_r5b_training_20261030.json"
REPORT = ROOT / "reports/taiji_native_core_r5b_selection_diagnosis_20261031.json"

SAMPLE_EPISODES = 216
SAMPLE_SEED = 20261030


def _ground_truth(episode: Mapping[str, Any], turn: Mapping[str, Any]) -> dict[str, Any]:
    """The course-defined expected fact for one scored turn."""

    category = str(turn["category"])
    history = [
        candidate
        for candidate in episode["turns"]
        if int(candidate["index"]) < int(turn["index"])
        and candidate["role"] in FACT_ROLES
        and candidate.get("entity") == turn.get("entity")
        and candidate.get("attribute") == turn.get("attribute")
    ]
    history.sort(key=lambda item: int(item["index"]))
    expected_value = None
    if category == "current_query" and history:
        expected_value = str(history[-1].get("value"))
    elif category == "previous_query" and len(history) >= 2:
        expected_value = str(history[-2].get("value"))
    elif category == "unknown_query":
        expected_value = None
    return {
        "category": category,
        "entity": turn.get("entity"),
        "attribute": turn.get("attribute"),
        "expected_value": expected_value,
        "history_len": len(history),
    }


def _classify(
    expected: Mapping[str, Any],
    selected: tuple[str, str, int] | None,
    produced: bytes,
    target: bytes,
) -> str:
    if expected["category"] == "unknown_query":
        if produced == target:
            return "selection_correct_answer_correct"
        return "unknown_answer_wrong"
    if selected is None:
        return "retrieval_error_no_selection"
    selected_value = selected[1]
    if selected_value == expected["expected_value"]:
        if produced == target:
            return "selection_correct_answer_correct"
        return "selection_correct_answer_wrong"
    if expected["expected_value"] is not None and produced == target:
        return "selection_wrong_value_answer_right"
    return "selection_wrong_value_answer_wrong"


def _diagnose_run(name: str, episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoint = (
        ROOT
        / json.loads(TRAINING_REPORT.read_text(encoding="utf-8"))["runs"][name]["endpoint_restore"][
            "checkpoint"
        ]
    )
    trainer = R5BRunTrainer.load(checkpoint)
    adapter = trainer.adapter
    tally: dict[str, int] = {}
    per_category: dict[str, dict[str, int]] = {}
    turns_total = 0
    for episode in episodes:
        result = replay_episode(adapter, trainer.model, episode)
        for turn in episode["turns"]:
            if not turn["scored"]:
                continue
            key = (str(episode["episode_id"]), int(turn["index"]))
            expected = _ground_truth(episode, turn)
            prediction = result.predictions[key]
            selection = result.selected_content.get(key)
            label = _classify(
                expected, selection, prediction.produced, bytes(turn["target"]["target_bytes_utf8"])
            )
            tally[label] = tally.get(label, 0) + 1
            bucket = per_category.setdefault(expected["category"], {})
            bucket[label] = bucket.get(label, 0) + 1
            turns_total += 1
    return {"run": name, "turns": turns_total, "tally": tally, "per_category": per_category}


def _merge(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tally: dict[str, int] = {}
    per_category: dict[str, dict[str, int]] = {}
    turns = 0
    for row in runs:
        turns += int(row["turns"])
        for label, count in row["tally"].items():
            tally[label] = tally.get(label, 0) + count
        for category, bucket in row["per_category"].items():
            target = per_category.setdefault(category, {})
            for label, count in bucket.items():
                target[label] = target.get(label, 0) + count
    rates = {label: count / turns for label, count in sorted(tally.items())}
    category_rates = {
        category: {
            "turns": sum(bucket.values()),
            **{label: count / sum(bucket.values()) for label, count in sorted(bucket.items())},
        }
        for category, bucket in sorted(per_category.items())
    }
    return {"turns": turns, "tally": tally, "rates": rates, "per_category": category_rates}


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
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    runs = [
        _diagnose_run(name, sample)
        for name in sorted(training["runs"])
        if training["runs"][name]["block"] == "q1"
    ]
    merged = _merge(runs)
    receipt = {
        "format": "taiji-native-core-r5b-selection-diagnosis-v1",
        "version": 1,
        "date": "2026-10-31",
        "contract": "plans/reference/SPEC-NATIVE-CONTENT-CORE_R5B_COPYTRACK_CANDIDATE_20261030.md",
        "training_report": {
            "path": TRAINING_REPORT.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAINING_REPORT.read_bytes()).hexdigest(),
        },
        "manifest": {
            "path": TRAIN.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(TRAIN.read_bytes()).hexdigest(),
            "episodes_total": len(episodes),
            "episodes_sampled": len(sample),
            "sample_seed": SAMPLE_SEED,
        },
        "definition": "train-only selection-error taxonomy over frozen q1 endpoints; first non-empty step selection vs course ground truth",
        "runs": runs,
        "merged": merged,
        "read_boundaries": {
            "query_split_read": False,
            "holdout_split_read": False,
            "sealed_read": False,
            "optimizer_updates": 0,
        },
        "cost": {
            "cpu_seconds": time.process_time() - started_cpu,
            "wall_seconds": time.perf_counter() - started_wall,
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
        help="required: the user approved R5b execution covers this read-only diagnosis",
    )
    args = parser.parse_args()
    return _main(approve=args.approve_r5b)


if __name__ == "__main__":
    raise SystemExit(main())
