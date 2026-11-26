"""Measure the frozen N175 training rollout gate without opening the blind key."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import load_train_episodes
from scripts.training.evaluate_taiji_n1_plan175_v3_holdout import require_completed_runs

SELECTION = ROOT / "reports/taiji_n1_plan175_v3_train_rollout_selection_20261026.json"
OUTPUT = ROOT / "reports/taiji_n1_plan175_v3_train_rollout_20261026.json"


def prefix_match(generated: bytes, target: bytes) -> float:
    target = target[:128]
    count = 0
    for actual, expected in zip(generated, target, strict=False):
        if actual != expected:
            break
        count += 1
    return count / max(1, len(target))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("training rollout sample is already registered; do not resample")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    runs = require_completed_runs()
    pairs, _ = load_train_episodes()
    # Same 16 evenly spaced SHA-order positions as N174, on the N175 train pool.
    # Preserve the old oasst2: namespace when hashing the raw OASST identities.
    ordered = sorted(
        (episode for _, episode in pairs),
        key=lambda episode: hashlib.sha256(f"oasst2:{episode.episode_id}".encode()).hexdigest(),
    )
    sample = [ordered[round(index * (len(ordered) - 1) / 15)] for index in range(16)]
    assert len({episode.episode_id for episode in sample}) == 16
    selection = {
        "schema": "taiji-n1-plan175-train-rollout-selection-v1",
        "rule": "N174 SHA order and 16 evenly spaced positions, applied to N175 train pool",
        "pool_size": len(ordered),
        "episode_ids": [episode.episode_id for episode in sample],
        "registered_before_generation": True,
        "prefix_bytes": 128,
        "full_generation_byte_limit": 2048,
        "same_sample_as_n174": False,
        "heldout_text_read": 0,
        "blind_key_opened": False,
        "evaluator_sha256": digest(Path(__file__)),
    }
    if SELECTION.exists():
        registered = json.loads(SELECTION.read_text(encoding="utf-8"))
        for key, value in selection.items():
            if key != "evaluator_sha256" and registered.get(key) != value:
                raise ValueError("registered training sample differs; cannot resample")
    else:
        with SELECTION.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(selection, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    results = []
    for item in runs:
        trainer = item["trainer"]
        checkpoint = trainer.checkpoint(
            trainer.model.new_state(
                scope_id="n175-r", session_id="audit", native_state_ref="n175:rollout:audit"
            ),
            cursor={},
        )["checkpoint_digest"]
        rows = []
        with torch.no_grad():
            for episode in sample:
                state = trainer.model.new_state(
                    scope_id="n175-r",
                    session_id="train-rollout",
                    native_state_ref=f"n175:rollout:{episode.episode_id}",
                )
                state = trainer.model.step_task(
                    episode.prompt_bytes,
                    state,
                    scope_id=state.scope_id,
                    session_id=state.session_id,
                    source_id="prompt",
                    resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 4096},
                )
                generated = trainer.model.generate(state, max_new_bytes=2048).generated
                rows.append(
                    {
                        "episode_id": episode.episode_id,
                        "free_target_prefix_match": prefix_match(generated, episode.target_bytes),
                        "generated_bytes": len(generated),
                        "terminated_with_end_marker": generated.endswith(b"<|end|>"),
                        "generated_text": generated.decode("utf-8", errors="replace"),
                        "generated_hex": generated.hex(),
                    }
                )
        after = trainer.checkpoint(
            trainer.model.new_state(
                scope_id="n175-r", session_id="audit", native_state_ref="n175:rollout:audit"
            ),
            cursor={},
        )["checkpoint_digest"]
        if checkpoint != after:
            raise ValueError("training rollout unexpectedly changed the checkpoint")
        value = statistics.median(row["free_target_prefix_match"] for row in rows)
        results.append(
            {
                "seed": item["seed"],
                "train_free_match_median": value,
                "rollout_gate_passed": value >= 0.50,
                "checkpoint_unchanged": True,
                "rows": rows,
            }
        )
    report = {
        "schema": "taiji-n1-plan175-train-rollout-v1",
        "status": "complete_train_only_rollout",
        "selection_sha256": digest(SELECTION),
        "evaluator_sha256": digest(Path(__file__)),
        "runs": results,
        "R_passed": sum(row["rollout_gate_passed"] for row in results) >= 2,
        "training_updates": 0,
        "heldout_text_read": 0,
        "blind_key_opened": False,
        "process_cpu_seconds": time.process_time() - started_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "can_promote": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "R_passed", "process_cpu_seconds")}))


if __name__ == "__main__":
    main()
