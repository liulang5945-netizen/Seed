"""Two matched train-only batches per candidate, after fresh-process checkpoint checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.training.audit_taiji_n1_plan176_learning_paths import CORPUS, PARENT, digest
from scripts.training.build_taiji_n1_plan176_foundation_corpus import ROOT, as_language_episode
from scripts.training.migrate_taiji_n1_plan176_v3_to_v4 import OUTPUT as V4_PARENT
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4
from taiji.persistence import atomic_save

OUTPUT = ROOT / "data/checkpoints/n1_plan176_train_only_calibration_20261026"
REPORT = ROOT / "reports/taiji_n1_plan176_train_only_calibration_20261026.json"


def fresh_restore(path: Path, version: int) -> dict:
    code = (
        "import json,sys,time,torch; torch.set_num_threads(1); started=time.process_time(); "
        f"from taiji.conditional_language_v{version} import ConditionalLanguageTrainerV{version} as T; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,s,c=T.from_checkpoint(p); assert t.checkpoint(s,cursor=c)['checkpoint_digest']==p['checkpoint_digest']; "
        "print(json.dumps({'status':'passed','steps':t.steps,'cpu_seconds':time.process_time()-started}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def loss_on_batch(model, batch) -> float:
    with torch.no_grad():
        numerator = sum(
            float(
                F.cross_entropy(
                    model.teacher_forced(prompt, target),
                    torch.tensor(list(target), dtype=torch.long),
                    reduction="sum",
                )
            )
            for prompt, target in batch
        )
    return numerator / sum(len(target) for _, target in batch)


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("preserve the original fixed calibration attempt")
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    child_cpu, results = 0.0, []
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    train = ROOT / corpus["output_directory"] / "train.jsonl"
    if digest(train) != corpus["split_sha256"]["train"]:
        raise ValueError("train-only calibration corpus differs")
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    groups = []
    for level in ("F0", "F1"):
        group = sorted(
            (r for r in rows if r["curriculum_level"] == level),
            key=lambda r: hashlib.sha256(("n176-cost-v1:" + r["episode_id"]).encode()).hexdigest(),
        )[:4]
        groups.append(group)
    batches = [
        [
            (as_language_episode(row).prompt_bytes, as_language_episode(row).target_bytes)
            for row in group
        ]
        for group in groups
    ]
    OUTPUT.mkdir(parents=True)
    for version, trainer_type, parent in (
        (3, ConditionalLanguageTrainerV3, PARENT),
        (4, ConditionalLanguageTrainerV4, V4_PARENT),
    ):
        trainer, state, _ = trainer_type.from_checkpoint(
            torch.load(parent, map_location="cpu", weights_only=False)
        )
        initial_steps, initial_exposures = trainer.steps, trainer.target_byte_exposures
        folder = OUTPUT / f"v{version}"
        zero, final = folder / "zero.pt", folder / "step2.pt"
        atomic_save(trainer.checkpoint(state, cursor={"batch": 0}), zero)
        zero_restore = fresh_restore(zero, version)
        child_cpu += zero_restore["cpu_seconds"]
        trainer, _, _ = trainer_type.from_checkpoint(
            torch.load(zero, map_location="cpu", weights_only=False)
        )
        before_loss = [loss_on_batch(trainer.model, batch) for batch in batches]
        updates = []
        for index, batch in enumerate(batches):
            if time.process_time() - started_cpu + child_cpu >= 300:
                raise TimeoutError("300 CPU-second calibration guard reached")
            began = time.process_time()
            update = trainer.update(batch)
            updates.append(
                {
                    "batch": index,
                    "episode_ids": [row["episode_id"] for row in groups[index]],
                    "current_batch_target_bytes": sum(len(target) for _, target in batch),
                    "process_cpu_seconds": time.process_time() - began,
                    "update_receipt": update,
                }
            )
        after_loss = [loss_on_batch(trainer.model, batch) for batch in batches]
        final_state = trainer.model.new_state(
            scope_id="n176-cost", session_id="done", native_state_ref="n176:calibration:step2"
        )
        atomic_save(trainer.checkpoint(final_state, cursor={"batch": 2}), final)
        final_restore = fresh_restore(final, version)
        child_cpu += final_restore["cpu_seconds"]
        expected_exposure = sum(sum(len(target) for _, target in batch) for batch in batches)
        if (
            trainer.steps - initial_steps != 2
            or trainer.target_byte_exposures - initial_exposures != expected_exposure
        ):
            raise ValueError(
                "calibration updates or target exposure differ from the fixed two batches"
            )
        results.append(
            {
                "version": version,
                "parent_checkpoint": str(parent.relative_to(ROOT)),
                "parent_checkpoint_sha256": digest(parent),
                "zero_checkpoint": str(zero.relative_to(ROOT)),
                "zero_checkpoint_sha256": digest(zero),
                "final_checkpoint": str(final.relative_to(ROOT)),
                "final_checkpoint_sha256": digest(final),
                "zero_fresh_restore": zero_restore,
                "final_fresh_restore": final_restore,
                "updates": updates,
                "extra_optimizer_updates": trainer.steps - initial_steps,
                "extra_target_byte_exposures": trainer.target_byte_exposures - initial_exposures,
                "learning_rate_inherited": trainer.learning_rate,
                "train_batch_loss_before": before_loss,
                "train_batch_loss_after": after_loss,
            }
        )
    report = {
        "schema": "taiji-n176-foundation-train-only-calibration-v1",
        "status": "two_updates_each_v3_v4_complete_restored",
        "train_corpus_sha256": digest(train),
        "calibration_source_sha256": digest(Path(__file__)),
        "candidate_sources_sha256": {
            str(v): digest(ROOT / f"taiji/conditional_language_v{v}.py") for v in (3, 4)
        },
        "runs": results,
        "training_updates": 4,
        "formal_training_updates": 0,
        "sealed_episodes_read": 0,
        "n175_key_opened": False,
        "formal_runs_start_from": "Uncalibrated original parent checkpoints, not these two-update artifacts",
        "process_cpu_seconds": time.process_time() - started_cpu,
        "child_process_cpu_seconds": child_cpu,
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_guard_seconds": 300,
        "can_promote": False,
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "training_updates",
                    "process_cpu_seconds",
                    "child_process_cpu_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
