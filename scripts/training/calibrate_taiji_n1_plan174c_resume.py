"""N1-174C: finish the scale comparison from the durable epoch-30 checkpoints.

N1-174B was truncated by its own runaway guard at epoch 37-38 after saving and
fresh-process-verifying epoch-20/30 checkpoints. This script reloads the
epoch-30 checkpoint, retrains epochs 31-40 with the identical batch schedule,
checks that the overlapping epoch losses reproduce the partial run, then reads
the preregistered epoch-40 gates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CPU_START = time.process_time()
WALL_START = time.monotonic()

import torch  # noqa: E402

from scripts.training import calibrate_taiji_n1_plan174b_scale as base  # noqa: E402
from scripts.training import diagnose_taiji_n1_plan167_train_decoder as old_diagnostic  # noqa: E402
from taiji.conditional_language_rollout import ConditionalLanguageRolloutTrainer  # noqa: E402
from taiji.conditional_language_v2 import ConditionalLanguageTrainerV2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan174c_scale_complete_20261025"
SOURCE_CHECKPOINTS = ROOT / "data/checkpoints/n1_plan174b_scale_20261025"
MODULE = "scripts.training.calibrate_taiji_n1_plan174c_resume"
START_EPOCH = 30
FINAL_EPOCH = 40
ARMS = ("tf", "ss")
SEEDS = (169, 170, 171)
REPRODUCTION_TOLERANCE = 1e-5


def partial_report(arm: str, seed: int) -> Path:
    return REPORTS / f"taiji_n1_plan174b_scale_20261025_{arm}_seed{seed}.json"


def complete_report(arm: str, seed: int) -> Path:
    return REPORTS / f"taiji_n1_plan174c_scale_complete_20261025_{arm}_seed{seed}.json"


def aggregate_report() -> Path:
    return REPORTS / "taiji_n1_plan174c_scale_complete_20261025.json"


def guard() -> None:
    if time.process_time() - CPU_START > 2600 or time.monotonic() - WALL_START > 4200:
        raise TimeoutError("N1-174C resume run reached its runaway-protection threshold")


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__),
        ROOT / "scripts/training/calibrate_taiji_n1_plan174b_scale.py",
        ROOT / "taiji/conditional_language_rollout.py",
        ROOT / "taiji/conditional_language_v2.py",
        ROOT / "taiji/task_core_v2.py",
        ROOT / "taiji/task_state_v2.py",
        ROOT / "scripts/training/calibrate_taiji_n1_plan171_holdout.py",
        ROOT / "tests/taiji_native/test_rollout_trainer.py",
        ROOT / "tests/taiji_native/test_task_language_candidate_v2.py",
        ROOT / "plans/reference/PLAN-N1-174C_scale_resume_20261025.md",
        ROOT / "plans/reference/SPEC-N1-174C_scale_resume_20261025.md",
    ]
    return {str(path.relative_to(ROOT)): old_diagnostic.frozen.sha256_file(path) for path in paths}


def load_trainer(arm: str, path: Path, train):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if arm == "ss":
        trainer, _, cursor = ConditionalLanguageRolloutTrainer.from_checkpoint(payload)
    else:
        trainer, _, cursor = ConditionalLanguageTrainerV2.from_checkpoint(payload)
    if trainer.model.parameter_count() != base.PARAMETERS:
        raise ValueError("resumed checkpoint parameter count differs from the frozen capacity")
    expected_exposures = START_EPOCH * sum(
        min(base.TARGET_BYTES, len(episode.target_bytes)) for episode in train
    )
    if (
        cursor != {"epoch": START_EPOCH, "batch": 125}
        or trainer.steps != START_EPOCH * 125
        or trainer.model.parameter_revision != START_EPOCH * 125
        or trainer.target_byte_exposures != expected_exposures
    ):
        raise ValueError("resumed checkpoint cursor/steps/revision/exposures differ")
    return payload, trainer, cursor


def run_resume(arm: str, seed: int) -> int:
    torch.set_num_threads(1)
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("resume identity differs from the preregistered arm/seed grid")
    report_path = complete_report(arm, seed)
    directory = CHECKPOINTS / f"{arm}_seed{seed}"
    if report_path.exists() or directory.exists():
        raise ValueError("N1-174C run already exists; do not overwrite or extend")
    directory.mkdir(parents=True)
    run = f"{arm}_seed{seed}"
    partial = json.loads(partial_report(arm, seed).read_text(encoding="utf-8"))
    checkpoint = SOURCE_CHECKPOINTS / run / "epoch30.pt"
    report = {
        "schema": "taiji-n1-plan174c-scale-resume-v1",
        "status": "running",
        "run": run,
        "arm": arm,
        "seed": seed,
        "start_epoch": START_EPOCH,
        "final_epoch": FINAL_EPOCH,
        "partial_report": str(partial_report(arm, seed).relative_to(ROOT)),
        "partial_report_sha256": old_diagnostic.frozen.sha256_file(partial_report(arm, seed)),
        "loaded_checkpoint": str(checkpoint.relative_to(ROOT)),
        "loaded_checkpoint_sha256": old_diagnostic.frozen.sha256_file(checkpoint),
        "source_sha256": source_hashes(),
        "dev_queries": 0,
        "can_promote": False,
        "epoch_log": list(partial.get("epoch_log", [])),
        "milestones": list(partial.get("milestones", [])),
        "resumed_epoch_log": [],
        "restores": {},
        "controls": [],
    }
    try:
        _, records, chosen, train = base.load_data()
        payload, trainer, cursor = load_trainer(arm, checkpoint, train)
        report["restores"]["loaded_epoch30"] = {
            "checkpoint_digest": payload["checkpoint_digest"],
            "steps": trainer.steps,
            "parameter_revision": trainer.model.parameter_revision,
            "target_byte_exposures": trainer.target_byte_exposures,
            "cursor": cursor,
        }
        rank_by_id = {record["episode_id"]: record["rank"] for record in records}
        items = [("train", episode, *base.example(episode)) for episode in base.train_sample(train)]
        items += [("heldout", episode, *base.example(episode)) for episode in chosen]
        prior_by_epoch = {entry["epoch"]: entry["loss_nats_mean"] for entry in report["epoch_log"]}
        mismatches = []
        reproduction = []
        for epoch in range(START_EPOCH + 1, FINAL_EPOCH + 1):
            batches, digest = base.epoch_batches(train, seed, epoch)
            losses = []
            for indices in batches:
                guard()
                record = trainer.update([base.example(train[position]) for position in indices])
                losses.append(record["loss_nats"])
            mean = sum(losses) / len(losses)
            entry = {
                "epoch": epoch,
                "order_digest": digest,
                "loss_nats_mean": mean,
                "loss_nats_last": losses[-1],
                "resumed": True,
            }
            report["resumed_epoch_log"].append(entry)
            report["epoch_log"].append(entry)
            if epoch in prior_by_epoch:
                difference = abs(mean - float(prior_by_epoch[epoch]))
                reproduction.append(
                    {
                        "epoch": epoch,
                        "partial_loss": float(prior_by_epoch[epoch]),
                        "resumed_loss": mean,
                        "absolute_difference": difference,
                    }
                )
                if difference > REPRODUCTION_TOLERANCE:
                    mismatches.append(epoch)
                    break
        if reproduction:
            report["reproduction"] = {
                "overlap_epochs": len(reproduction),
                "max_absolute_difference": max(
                    item["absolute_difference"] for item in reproduction
                ),
                "first": reproduction[0],
            }
        if mismatches:
            report.update(
                status="failed",
                error_type="ResumeMismatch",
                error=f"resumed loss differs from the partial run at epoch {mismatches[0]}",
            )
        else:
            report["milestones"].extend(base.evaluate(trainer.model, items, FINAL_EPOCH, run))
            final_rows = [row for row in report["milestones"] if row["epoch"] == FINAL_EPOCH]
            report["controls"] = base.controls(
                trainer.model,
                chosen,
                train,
                [row for row in final_rows if row["kind"] == "heldout"],
                rank_by_id,
            )
            report["summary"] = base.summarize(
                report["milestones"], report["controls"], FINAL_EPOCH
            )
            report["restores"]["final"] = base.save_verify(
                trainer,
                train[0].prompt_bytes,
                "final",
                directory,
                {"epoch": FINAL_EPOCH, "batch": len(train) // base.BATCH},
            )
            report.update(
                status="complete",
                parent_cpu_seconds=time.process_time() - CPU_START,
                cpu_seconds=time.process_time() - CPU_START,
                wall_seconds=time.monotonic() - WALL_START,
            )
    except Exception as exc:
        report.update(
            status="partial" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        report.update(
            parent_cpu_seconds=time.process_time() - CPU_START,
            cpu_seconds=time.process_time() - CPU_START,
            wall_seconds=time.monotonic() - WALL_START,
        )
    report["source_unchanged_after_run"] = all(
        old_diagnostic.frozen.sha256_file(ROOT / name) == value
        for name, value in report["source_sha256"].items()
    )
    if not report["source_unchanged_after_run"]:
        report.update(status="failed", source_error="input/source changed during the resume")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "run": run,
                "status": report["status"],
                "cpu_seconds": round(report["cpu_seconds"], 3),
                "wall_seconds": round(report["wall_seconds"], 3),
                "reproduction": report.get("reproduction"),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "complete" else 1


def verify(path: str) -> int:
    """Fresh-process verification of a v2 or rollout checkpoint."""

    torch.set_num_threads(1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") == ConditionalLanguageRolloutTrainer.FORMAT:
        trainer, state, cursor = ConditionalLanguageRolloutTrainer.from_checkpoint(payload)
    elif payload.get("format") == ConditionalLanguageTrainerV2.FORMAT:
        trainer, state, cursor = ConditionalLanguageTrainerV2.from_checkpoint(payload)
    else:
        raise ValueError("unsupported resume checkpoint format")
    signatures = base.checkpoint_signatures(trainer, state)
    print(
        json.dumps(
            {
                "checkpoint_digest": payload["checkpoint_digest"],
                "format": payload["format"],
                "steps": trainer.steps,
                "target_byte_exposures": trainer.target_byte_exposures,
                "parameter_count": trainer.model.parameter_count(),
                "cursor": cursor,
                **signatures,
            }
        )
    )
    return 0


def aggregate() -> int:
    hits = {arm: 0 for arm in ARMS}
    qualified = {arm: 0 for arm in ARMS}
    rollout = {arm: [] for arm in ARMS}
    complete = True
    runs = {}
    total_cpu = 0.0
    for arm in ARMS:
        for seed in SEEDS:
            path = complete_report(arm, seed)
            if not path.is_file():
                complete = False
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            complete = complete and report.get("status") == "complete"
            complete = complete and bool(report.get("source_unchanged_after_run"))
            total_cpu += report.get("cpu_seconds") or 0
            summary = report.get("summary", {})
            rollout[arm].append(summary.get("train_free_match_median"))
            arm_hits = 0
            for entry in report.get("controls", []):
                arm_hits += int(
                    entry["own_match"] >= base.OWN_GATE
                    and entry["own_match"] >= entry["control_a_match"] + base.MARGIN
                    and entry["own_match"] >= entry["control_b_match"] + base.MARGIN
                )
            hits[arm] += arm_hits
            qualified[arm] += sum(
                entry["own_match"] >= base.OWN_GATE for entry in report.get("controls", [])
            )
            runs[f"{arm}_seed{seed}"] = {
                "status": report.get("status"),
                "cpu_seconds": report.get("cpu_seconds"),
                "wall_seconds": report.get("wall_seconds"),
                "train_free_match_median": summary.get("train_free_match_median"),
                "heldout_free_match_mean": summary.get("heldout_free_match_mean"),
                "binding_hits": arm_hits,
                "reproduction_difference": report.get("reproduction", {}).get(
                    "absolute_difference"
                ),
                "error": report.get("error"),
            }
    r_arm = {
        arm: sum(value is not None and value >= base.ROLLOUT_GATE for value in rollout[arm]) >= 2
        for arm in ARMS
    }
    gate_arm = [
        arm
        for arm in ARMS
        if hits[arm] >= base.BINDING_HITS
        and max(hits.values()) - min(hits.values()) >= base.HIT_LEAD
    ]
    gate = bool(gate_arm) and complete
    status = (
        "PASS_BINDING"
        if gate
        else "FAIL_NO_GENERALIZATION" if any(r_arm.values()) else "INCONCLUSIVE_ROLLOUT"
    )
    if not complete:
        status = "PARTIAL"
    aggregate = {
        "schema": "taiji-n1-plan174c-scale-aggregate-v1",
        "status": status,
        "all_runs_complete": complete,
        "rollout_medians": rollout,
        "rollout_gate_by_arm": r_arm,
        "binding_hits": hits,
        "binding_qualified": qualified,
        "binding_gate_arms": gate_arm,
        "hit_lead": max(hits.values()) - min(hits.values()),
        "criteria": {
            "rollout_gate": base.ROLLOUT_GATE,
            "own_gate": base.OWN_GATE,
            "margin": base.MARGIN,
            "binding_hits": base.BINDING_HITS,
            "hit_lead": base.HIT_LEAD,
        },
        "runs": runs,
        "total_cpu_seconds": total_cpu,
        "dev_queries": 0,
        "can_promote": False,
    }
    aggregate_report().write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "hits": hits, "rollout": rollout}))
    return 0 if status != "PARTIAL" else 1


def spawn_all() -> int:
    processes = []
    for arm in ARMS:
        for seed in SEEDS:
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", MODULE, "--arm", arm, "--seed", str(seed)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
            )
    results = []
    for process in processes:
        stdout, stderr = process.communicate()
        results.append(
            {
                "returncode": process.returncode,
                "stdout": stdout.strip().splitlines()[-1:] if stdout else [],
                "stderr": stderr.strip().splitlines()[-3:] if stderr else [],
            }
        )
    print(json.dumps({"runs": results}, ensure_ascii=False))
    return aggregate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verify")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.verify:
        return verify(args.verify)
    if args.aggregate:
        return aggregate()
    if args.all:
        return spawn_all()
    if args.arm is None or args.seed is None:
        raise ValueError("a resume run needs both --arm and --seed")
    return run_resume(args.arm, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
