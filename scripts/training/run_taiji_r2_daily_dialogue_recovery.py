"""Restart interrupted R2 arms from frozen zero bundles with durable cursors.

Training only: this entry point never selects or generates dev prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROCESS_START_CPU = time.process_time()
import psutil
import torch

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from scripts.training import run_taiji_r2_daily_dialogue_warmstart_matched as matched
from scripts.training.taiji_r2_dialogue_recovery import (
    Cursor,
    RecoveryStore,
    _publish_json,
    reclaim_writer_lock,
    train_next,
)
from taiji import LanguageAlignmentTrainer, Taiji

ARMS = ("full_turn_1m", "warm_start", "fresh_same_capacity")
PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-RECOVERY_20261021.md"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_recovery_20261021"
PREFIX = ROOT / "reports/taiji_r2_daily_dialogue_recovery_20261021"
SAVE_INTERVAL = 32


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def host_cpu_seconds() -> float:
    """Cumulative machine CPU seconds; on the same boot it bounds dead-run CPU."""
    return sum(
        sum(value for field, value in zip(core._fields, core) if field not in {"idle", "iowait"})
        for core in psutil.cpu_times(percpu=True)
    )


def paths(arm: str) -> dict[str, Path]:
    if arm not in ARMS:
        raise ValueError("unknown recovery arm")
    return {
        "root": CHECKPOINT_ROOT / arm,
        "prepare_attempt": PREFIX.with_name(f"{PREFIX.name}_{arm}_prepare_attempt.json"),
        "prepare_events": CHECKPOINT_ROOT / arm / "prepare_events",
        **{
            key: PREFIX.with_name(f"{PREFIX.name}_{arm}_{key}.{extension}")
            for key, extension in (
                ("readiness", "json"),
                ("attempt", "json"),
                ("result", "json"),
                ("progress", "jsonl"),
            )
        },
    }


def zero_path(arm: str) -> Path:
    if arm == "full_turn_1m":
        return base.CHECKPOINT_ROOT / "full_turn_zero.pt"
    return matched._arm_paths(arm)["checkpoint_dir"] / "full_turn_zero.pt"


def source_bindings(arm: str) -> dict[str, Any]:
    old_attempt = base.ATTEMPT if arm == "full_turn_1m" else matched._arm_paths(arm)["attempt"]
    old = base._read_json(old_attempt)
    hashes = dict(old["source_hashes"])
    for filename, digest in hashes.items():
        if base.sha256(Path(filename)) != digest:
            raise base.Refuse(f"frozen source changed: {filename}")
    for filename in (
        PLAN,
        Path(__file__),
        ROOT / "scripts/training/taiji_r2_dialogue_recovery.py",
        ROOT / "tests/taiji_native/test_r2_dialogue_recovery_contract.py",
        ROOT / "tests/taiji_native/test_r2_daily_dialogue_recovery_contract.py",
    ):
        hashes[str(filename.resolve())] = base.sha256(filename)
    return {
        "run": "r2_daily_dialogue_recovery_20261021",
        "arm": arm,
        "zero_sha256": base.sha256(zero_path(arm)),
        "source_hashes": hashes,
    }


class Budget:
    def __init__(self, arm: str, charged: float = 0.0) -> None:
        self.limit = 240_000.0 if arm == "full_turn_1m" else 120_000.0
        self.started = time.process_time()
        self.charged = charged + max(0.0, self.started - PROCESS_START_CPU)
        self.child_seconds = 0.0

    def used(self) -> float:
        return self.charged + time.process_time() - self.started + self.child_seconds

    def guard(self, reserve: float = 120.0) -> None:
        if self.used() + reserve >= self.limit:
            raise base.Refuse("recovery CPU allocation exhausted; no implicit retry")
        if psutil.virtual_memory().available < 8 * 1024**3:
            raise base.Refuse("recovery available memory below 8 GiB")

    def ledger(self) -> dict[str, Any]:
        return {
            "charged_cpu_seconds": self.used(),
            "snapshot_utc": utc_now(),
            "host_cpu_seconds": host_cpu_seconds(),
            "logical_cpu_upper": os.cpu_count(),
            "host_boot_time": psutil.boot_time(),
            "cpu_limit": self.limit,
        }

    def resume_from(self, envelope: dict[str, Any]) -> None:
        self.charged = max(self.used(), tail_charge(envelope))
        self.started = time.process_time()
        self.child_seconds = 0.0


def tail_charge(envelope: dict[str, Any]) -> float:
    """Charge a killed session conservatively, including downtime, never as zero."""
    ledger = envelope["ledger"]
    host_cpu_delta = host_cpu_seconds() - float(ledger["host_cpu_seconds"])
    same_boot = psutil.boot_time() == float(ledger["host_boot_time"])
    if same_boot and host_cpu_delta >= 0:
        # Includes every process since the checkpoint, so unrelated host work only
        # makes the bound more conservative. A reboot resets counters; use wall time.
        lost_upper = host_cpu_delta
    else:
        elapsed = max(
            0.0,
            (datetime.now(UTC) - datetime.fromisoformat(ledger["snapshot_utc"])).total_seconds(),
        )
        cores = max(int(ledger["logical_cpu_upper"]), int(os.cpu_count() or 1))
        lost_upper = elapsed * cores
    return float(ledger["charged_cpu_seconds"]) + lost_upper


def commit(
    store: RecoveryStore,
    trainer: LanguageAlignmentTrainer,
    cursor: Cursor,
    epochs: int,
    budget: Budget,
) -> dict[str, Any]:
    budget.guard()
    host_before = host_cpu_seconds()
    try:
        receipt = store.commit(trainer, cursor, epochs, budget.ledger())
    except BaseException:
        # A failed or timed-out restore child still consumed CPU. Host delta is a
        # conservative bound when the child cannot return its own receipt.
        budget.child_seconds += max(0.0, host_cpu_seconds() - host_before)
        raise
    budget.child_seconds += float(receipt["verification"]["cpu_seconds"])
    budget.guard()
    return receipt


def restore_final(path: Path, stage: str, budget: Budget) -> dict[str, Any]:
    host_before = host_cpu_seconds()
    try:
        return base._independent_restore(path, stage, budget)
    except BaseException:
        budget.child_seconds += max(0.0, host_cpu_seconds() - host_before)
        raise


@contextmanager
def open_store(directory: Path, bindings: dict[str, Any], corpus, *, resume: bool = False):
    if (directory / "writer.lock").exists():
        if not resume:
            raise base.Refuse("writer lock exists; explicit resume is required")
        with reclaim_writer_lock(directory) as token:
            with RecoveryStore(directory, bindings, corpus, reclaim_token=token) as store:
                yield store
    else:
        with RecoveryStore(directory, bindings, corpus) as store:
            yield store


def latest_resume_stage(root: Path) -> str:
    for stage in ("assistant_sft", "pretraining"):
        if (root / stage / "latest.json").exists():
            return stage
    raise base.Refuse("no committed recovery stage to resume")


def preparation_journal(output, arm):
    """Validate and charge the immutable journal before any source/corpus work."""
    attempt = base._read_json(output["prepare_attempt"])
    runner_sha = base.sha256(Path(__file__))
    if attempt["runner_sha256"] != runner_sha or attempt.get("arm") != arm:
        raise base.Refuse("preparation journal runner/arm binding changed")
    candidates = [attempt]
    for event_path in output["prepare_events"].glob("*.json"):
        event = base._read_json(event_path)
        if event.get("runner_sha256") != runner_sha or event.get("arm") != arm:
            raise base.Refuse("preparation event runner/arm binding changed")
        candidates.append(event)
    newest = max(
        candidates, key=lambda item: datetime.fromisoformat(item["ledger"]["snapshot_utc"])
    )
    return max(
        tail_charge(newest), *(float(item["ledger"]["charged_cpu_seconds"]) for item in candidates)
    )


def preparation_resume_charge(output, bindings, pretrain, assistant):
    """Use verified durable envelopes; without one retain the initial attempt tail."""
    attempt = base._read_json(output["prepare_attempt"])
    if attempt["runner_sha256"] != base.sha256(Path(__file__)):
        raise base.Refuse("preparation runner source changed")
    if attempt["parent_zero_sha256"] != bindings["zero_sha256"]:
        raise base.Refuse("preparation zero parent changed")
    candidates = []
    for stage, corpus in (("pretraining", pretrain), ("assistant_canary", assistant)):
        directory = output["root"] / stage
        if (directory / "latest.json").exists():
            _, cursor, envelope = RecoveryStore(
                directory, {**bindings, "stage": stage}, corpus
            ).load_latest(restore_rng=False)
            if cursor != Cursor() or envelope["trainer"]["global_step"]:
                raise base.Refuse("preparation envelope contains formal updates")
            candidates.append(envelope)
    # Failed preparation sessions can have consumed CPU after the last envelope.
    for event_path in output["prepare_events"].glob("*.json"):
        event = base._read_json(event_path)
        if event.get("bindings") is None:
            if event.get("runner_sha256") != attempt["runner_sha256"]:
                raise base.Refuse("preparation event source binding changed")
        elif event.get("bindings") != bindings:
            raise base.Refuse("preparation event source binding changed")
        candidates.append(event)
    if not candidates:
        candidates = [attempt]
    newest = max(
        candidates, key=lambda item: datetime.fromisoformat(item["ledger"]["snapshot_utc"])
    )
    return tail_charge(newest)


def preparation_event(output, event, budget, bindings, **details):
    output["prepare_events"].mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "taiji-r2-preparation-event-v1",
        "event": event,
        "runner_sha256": base.sha256(Path(__file__)),
        "arm": bindings["arm"],
        "bindings": bindings,
        "ledger": budget.ledger(),
        "formal_updates": 0,
        "dev_queries": 0,
        **details,
    }
    _publish_json(output["prepare_events"] / (uuid.uuid4().hex + ".json"), payload)
    return payload


def prepared_zero(store, trainer, epochs, budget):
    if (store.directory / "latest.json").exists():
        restored, cursor, _ = store.load_latest(restore_rng=False)
        if (
            cursor != Cursor()
            or restored.checkpoint()["checkpoint_digest"]
            != trainer.checkpoint()["checkpoint_digest"]
        ):
            raise base.Refuse("existing prepared zero differs")
        manifest = base._read_json(store.directory / "latest.json")
        return base._read_json(store.directory / manifest["receipt"])
    return commit(store, trainer, Cursor(), epochs, budget)


def prepare(arm: str, *, resume: bool = False) -> dict[str, Any]:
    output = paths(arm)
    if output["readiness"].exists() or output["result"].exists():
        raise base.Refuse("recovery readiness/result already exists")
    if output["prepare_attempt"].exists() != resume:
        raise base.Refuse(
            "preparation attempt requires explicit --prepare-resume; new preparation cannot resume"
        )
    budget = Budget(arm)
    if not resume:
        # Persist before corpus parsing, probes or checkpoint work. A killed process
        # therefore leaves a budget anchor even when no envelope was ever saved.
        _publish_json(
            output["prepare_attempt"],
            {
                "schema": "taiji-r2-preparation-attempt-v1",
                "status": "preparing",
                "arm": arm,
                "runner_sha256": base.sha256(Path(__file__)),
                "parent_zero_sha256": base.sha256(zero_path(arm)),
                "ledger": budget.ledger(),
                "formal_updates": 0,
                "dev_queries": 0,
            },
        )
    bindings = None
    if resume:
        # No source validation, corpus load or checkpoint restore may precede this
        # charge: failures in those steps must inherit every earlier failure event.
        budget.charged = max(budget.used(), preparation_journal(output, arm))
        budget.started = time.process_time()
    try:
        bindings = source_bindings(arm)
        pretrain, assistant, _ = base._validate_sources()
        if resume:
            budget.resume_from(
                {
                    "ledger": {
                        **budget.ledger(),
                        "charged_cpu_seconds": preparation_resume_charge(
                            output, bindings, pretrain, assistant
                        ),
                    }
                }
            )
        budget.guard()
        with open_store(
            output["root"] / "prepare_owner",
            {**bindings, "stage": "prepare_owner"},
            pretrain,
            resume=resume,
        ):
            preparation_event(output, "prepare_session_started", budget, bindings, resume=resume)
            zero = torch.load(zero_path(arm), map_location="cpu", weights_only=True)
            trainer = LanguageAlignmentTrainer.from_checkpoint(zero, pretrain)
            if trainer.global_step != 0 or trainer.episode_count != 0 or trainer.history:
                raise base.Refuse("recovery origin is not a zero training bundle")
            if trainer.checkpoint()["checkpoint_digest"] != zero["checkpoint_digest"]:
                raise base.Refuse("recovery zero digest differs")
            probe = LanguageAlignmentTrainer.from_checkpoint(zero, pretrain)
            probe.train(epochs=1, max_episodes=1)
            if probe.global_step != len(pretrain.episodes[0].target_bytes):
                raise base.Refuse("pretraining credit probe failed")
            canary, _ = base._assistant_sft_trainer(
                Taiji.from_checkpoint(zero["model"], device="cpu"), assistant, trainer.code_revision
            )
            canary_zero = canary.checkpoint()
            canary_probe = LanguageAlignmentTrainer.from_checkpoint(canary_zero, assistant)
            canary_probe.train(epochs=1, max_episodes=1)
            if canary_probe.global_step != len(assistant.episodes[0].target_bytes):
                raise base.Refuse("assistant credit probe failed")
            with open_store(
                output["root"] / "pretraining",
                {**bindings, "stage": "pretraining"},
                pretrain,
                resume=resume,
            ) as store:
                zero_receipt = prepared_zero(store, trainer, base.PRETRAIN_EPOCHS, budget)
            with open_store(
                output["root"] / "assistant_canary",
                {**bindings, "stage": "assistant_canary"},
                assistant,
                resume=resume,
            ) as store:
                canary_receipt = prepared_zero(store, canary, 1, budget)
            if source_bindings(arm) != bindings:
                raise base.Refuse("preparation sources changed")
            result = {
                "schema": "taiji-r2-dialogue-recovery-readiness-v1",
                "status": "ready",
                "arm": arm,
                "bindings": bindings,
                "zero_restore": zero_receipt,
                "assistant_canary_restore": canary_receipt,
                "cpu_seconds": budget.used(),
                "formal_updates": 0,
                "dev_queries": 0,
                "can_promote": False,
            }
            base.write_new_json(output["readiness"], result)
            preparation_event(output, "prepare_ready", budget, bindings, status="ready")
            return result
    except BaseException as exc:
        if bindings is not None:
            preparation_event(
                output, "prepare_failed", budget, bindings, error=f"{type(exc).__name__}: {exc}"
            )
        else:
            # Validation itself can fail before full bindings become available.
            directory = output["prepare_events"]
            directory.mkdir(parents=True, exist_ok=True)
            _publish_json(
                directory / (uuid.uuid4().hex + ".json"),
                {
                    "event": "prepare_source_validation_failed",
                    "bindings": None,
                    "runner_sha256": base.sha256(Path(__file__)),
                    "arm": arm,
                    "ledger": budget.ledger(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "formal_updates": 0,
                },
            )
        raise


def execute(arm: str, *, resume: bool = False) -> dict[str, Any]:
    output = paths(arm)
    if output["result"].exists():
        raise base.Refuse("recovery result exists; do not retry")
    readiness = base._read_json(output["readiness"])
    bindings = source_bindings(arm)
    if readiness.get("status") != "ready" or readiness.get("bindings") != bindings:
        raise base.Refuse("recovery readiness/source binding differs")
    if resume and not output["attempt"].exists():
        raise base.Refuse("no persisted recovery attempt is available to resume")
    if not resume and (output["progress"].exists() or output["attempt"].exists()):
        raise base.Refuse("recovery attempt already exists; pass --resume explicitly")
    pretrain, assistant, _ = base._validate_sources()
    if not resume:
        base.write_new_json(
            output["attempt"],
            {
                "schema": "taiji-r2-dialogue-recovery-attempt-v1",
                "status": "training",
                "arm": arm,
                "started_utc": utc_now(),
                "bindings": bindings,
                "restart_from_zero": True,
                "dev_queries": 0,
                "can_promote": False,
            },
        )
    budget = Budget(arm, float(readiness["cpu_seconds"]))
    receipts: dict[str, Any] = {}
    final_payload: dict[str, Any] | None = None
    run_context = open_store(
        output["root"], {**bindings, "stage": "runner"}, pretrain, resume=resume
    )
    run_context.__enter__()
    progress = None
    try:
        progress = output["progress"].open("a" if resume else "x", encoding="utf-8", newline="\n")
        if resume:
            resume_stage = latest_resume_stage(output["root"])
            resume_corpus = assistant if resume_stage == "assistant_sft" else pretrain
            resume_store = RecoveryStore(
                output["root"] / resume_stage,
                {**bindings, "stage": resume_stage},
                resume_corpus,
            )
            _, _, envelope = resume_store.load_latest(restore_rng=False)
            budget.resume_from(envelope)
        for stage, corpus, epochs, expected_episodes, expected_updates in (
            ("pretraining", pretrain, 8, 30_096, 14_517_912),
            ("assistant_sft", assistant, 1, 525, 113_209),
        ):
            store_directory = output["root"] / stage
            store_bindings = {**bindings, "stage": stage}
            with open_store(store_directory, store_bindings, corpus, resume=resume) as store:
                latest = output["root"] / stage / "latest.json"
                if latest.exists():
                    trainer, cursor, envelope = store.load_latest()
                    if stage == "pretraining" and cursor.epoch == epochs:
                        final_payload = trainer.checkpoint()
                elif stage == "assistant_sft":
                    if final_payload is None:
                        raise base.Refuse("assistant calibration requires restored full-turn final")
                    trainer, _ = base._assistant_sft_trainer(
                        Taiji.from_checkpoint(final_payload["model"], device="cpu"),
                        assistant,
                        final_payload["code_revision"],
                    )
                    cursor = Cursor()
                    receipts[stage + "_zero"] = commit(store, trainer, cursor, epochs, budget)
                else:
                    raise base.Refuse("missing prepared zero recovery manifest")
                while cursor.epoch < epochs:
                    budget.guard()
                    record, cursor = train_next(trainer, cursor, epochs)
                    if trainer.episode_count % SAVE_INTERVAL == 0 or cursor.index == 0:
                        receipt = commit(store, trainer, cursor, epochs, budget)
                        base.write_jsonl(
                            progress,
                            {
                                "event": "durable_training_commit",
                                "arm": arm,
                                "stage": stage,
                                "episodes": trainer.episode_count,
                                "target_byte_updates": trainer.global_step,
                                "cursor": {"epoch": cursor.epoch, "index": cursor.index},
                                "cpu_seconds": budget.used(),
                                "receipt": receipt,
                            },
                        )
                        print(
                            f"{arm} {stage} durable={trainer.episode_count}/{expected_episodes} cpu={budget.used():.1f}",
                            flush=True,
                        )
                if (
                    trainer.episode_count != expected_episodes
                    or trainer.global_step != expected_updates
                ):
                    raise base.Refuse("recovery terminal episode/byte denominator differs")
                final_payload = trainer.checkpoint()
                final_path = output["root"] / f"{stage}_final.pt"
                if not final_path.exists():
                    base._atomic_save(final_payload, final_path)
                elif (
                    torch.load(final_path, map_location="cpu", weights_only=True)[
                        "checkpoint_digest"
                    ]
                    != final_payload["checkpoint_digest"]
                ):
                    raise base.Refuse("existing final checkpoint differs")
                restore = restore_final(final_path, stage, budget)
                receipts[stage] = {
                    "path": str(final_path.resolve()),
                    "sha256": base.sha256(final_path),
                    "restore": restore,
                    "episodes": trainer.episode_count,
                    "target_byte_updates": trainer.global_step,
                }
        if source_bindings(arm) != bindings:
            raise base.Refuse("recovery sources changed during execution")
        result = {
            "schema": "taiji-r2-dialogue-recovery-result-v1",
            "status": "complete",
            "complete": True,
            "arm": arm,
            "restart_from_zero": True,
            "bindings": bindings,
            "stages": receipts,
            "cpu_seconds": budget.used(),
            "dev_queries": 0,
            "generation_queries": 0,
            "can_promote": False,
        }
        base.write_new_json(output["result"], result)
        return result
    except BaseException as exc:
        if progress is not None:
            base.write_jsonl(
                progress,
                {
                    "event": "recovery_stopped",
                    "error": f"{type(exc).__name__}: {exc}",
                    "cpu_seconds": budget.used(),
                    "dev_queries": 0,
                },
            )
        raise
    finally:
        if progress is not None:
            progress.close()
        run_context.__exit__(*sys.exc_info())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--prepare-resume", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = (
        prepare(args.arm, resume=args.prepare_resume)
        if args.prepare_only or args.prepare_resume
        else execute(args.arm, resume=args.resume)
    )
    print(
        json.dumps(
            {"status": result["status"], "arm": args.arm, "cpu_seconds": result["cpu_seconds"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
