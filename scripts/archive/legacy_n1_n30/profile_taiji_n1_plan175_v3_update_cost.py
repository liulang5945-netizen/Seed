"""Measure V3 optimizer-update cost on a second disjoint train-only batch."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import (
    CHECKPOINT as FIRST_CHECKPOINT,
)
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import (
    MAX_PROMPT_BYTES,
    MAX_TARGET_BYTES,
    choose_calibration_batch,
    load_train_episodes,
)
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import (
    OUTPUT as FIRST_REPORT,
)
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3

OUTPUT = ROOT / "reports/taiji_n1_plan175_v3_update_cost_20261026.json"
CHECKPOINT = ROOT / "data/checkpoints/n1_plan175_v3_calibration_20261026/tf_seed169/step0002.pt"


def verify_fresh_process(path: Path, checkpoint_digest: str, steps: int, exposures: int) -> float:
    code = (
        "import json,torch,sys,time; started=time.process_time(); "
        "from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,state,c=ConditionalLanguageTrainerV3.from_checkpoint(p); "
        "assert t.steps==int(sys.argv[3]) and t.target_byte_exposures==int(sys.argv[4]); "
        "assert p['checkpoint_digest']==sys.argv[2] and state.to_payload()['format']=='taiji-task-state-v3'; "
        "print(json.dumps({'digest':p['checkpoint_digest'],'cpu_seconds':time.process_time()-started}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), checkpoint_digest, str(steps), str(exposures)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if payload["digest"] != checkpoint_digest:
        raise ValueError("fresh-process calibration checkpoint digest differs")
    return float(payload["cpu_seconds"])


def main() -> None:
    if OUTPUT.exists() or CHECKPOINT.exists():
        raise FileExistsError("the update-cost profile is write-once")
    first = json.loads(FIRST_REPORT.read_text(encoding="utf-8"))
    first_checkpoint = torch.load(FIRST_CHECKPOINT, map_location="cpu", weights_only=False)
    if digest(FIRST_CHECKPOINT) != first["calibration_checkpoint_sha256"]:
        raise ValueError("first train-only calibration checkpoint SHA differs")
    if first_checkpoint["checkpoint_digest"] != first["calibration_checkpoint_digest"]:
        raise ValueError("first calibration checkpoint digest differs")
    excluded_ids = {row["target_id"] for row in first["calibration_batch"]}
    excluded_families = {row["family"] for row in first["calibration_batch"]}
    torch.set_num_threads(1)
    episodes, data = load_train_episodes()
    disjoint = [
        row
        for row in episodes
        if row[0]["target_id"] not in excluded_ids and row[0]["family"] not in excluded_families
    ]
    batch = choose_calibration_batch(disjoint)
    if any(
        row[0]["target_id"] in excluded_ids or row[0]["family"] in excluded_families
        for row in batch
    ):
        raise ValueError("second calibration batch overlaps the first batch")
    trainer, _, cursor = ConditionalLanguageTrainerV3.from_checkpoint(first_checkpoint)
    old_exposures = trainer.target_byte_exposures
    examples = [(episode.prompt_bytes, episode.target_bytes) for _, episode in batch]
    batch_target_bytes = sum(len(target) for _, target in examples)
    if any(
        len(prompt) > MAX_PROMPT_BYTES or len(target) > MAX_TARGET_BYTES
        for prompt, target in examples
    ):
        raise ValueError("selected train-only calibration sample exceeds its declared bounds")
    started_cpu, started_wall = time.process_time(), time.monotonic()
    metrics = trainer.update(examples)
    update_cpu, update_wall = time.process_time() - started_cpu, time.monotonic() - started_wall
    if trainer.target_byte_exposures - old_exposures != batch_target_bytes:
        raise ValueError("reported target-byte exposure delta differs from this batch")
    state = trainer.model.new_state(
        scope_id="n1-175", session_id="seed-169", native_state_ref="n1-175:calibration-step2"
    )
    next_cursor = {
        "epoch": 0,
        "batch": 2,
        "dataset_partition_sha256": data["partition_sha256"],
        "parent_cursor": cursor,
    }
    trainer.save(CHECKPOINT, state, cursor=next_cursor)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    restore_cpu = verify_fresh_process(
        CHECKPOINT, saved["checkpoint_digest"], trainer.steps, trainer.target_byte_exposures
    )
    result = {
        "schema": "taiji-n1-plan175-v3-update-cost-v1",
        "status": "second_disjoint_train_only_update_and_restore_complete",
        "v3_candidate_sha256": digest(ROOT / "taiji/conditional_language_v3.py"),
        "profile_script_sha256": digest(Path(__file__)),
        "first_calibration_report_sha256": digest(FIRST_REPORT),
        "first_checkpoint_sha256": digest(FIRST_CHECKPOINT),
        "current_checkpoint_sha256": digest(CHECKPOINT),
        "current_checkpoint_digest": saved["checkpoint_digest"],
        "partition_sha256": data["partition_sha256"],
        "eligible_train_examples": first["complete_train_corpus_candidates"],
        "eligible_train_families": first["complete_train_corpus_families"],
        "disjoint_batch": [
            {
                "target_id": candidate["target_id"],
                "family": candidate["family"],
                "prompt_bytes": len(episode.prompt_bytes),
                "target_bytes": len(episode.target_bytes),
            }
            for candidate, episode in batch
        ],
        "batch_target_bytes": batch_target_bytes,
        "update_metrics": metrics,
        "update_process_cpu_seconds": update_cpu,
        "update_wall_seconds": update_wall,
        "target_bytes_per_process_cpu_second": batch_target_bytes / update_cpu,
        "cumulative_target_exposures_before": old_exposures,
        "cumulative_target_exposures_after": trainer.target_byte_exposures,
        "fresh_process_restore": "passed",
        "fresh_process_restore_cpu_seconds": restore_cpu,
        "heldout_text_parsed": 0,
        "development_reads": 0,
        "training_updates_this_profile": 1,
        "can_promote": False,
        "correction": "The first report divided cumulative lifetime exposures by setup-plus-update CPU; this report uses this batch's exposure delta divided by optimizer-update CPU only.",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "eligible_train_examples",
                    "eligible_train_families",
                    "batch_target_bytes",
                    "update_process_cpu_seconds",
                    "target_bytes_per_process_cpu_second",
                    "fresh_process_restore",
                    "heldout_text_parsed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
