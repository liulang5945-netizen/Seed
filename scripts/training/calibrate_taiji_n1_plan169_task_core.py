"""One fixed train-only TaskCore pilot; engineering evidence, never a held-out claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

CPU_START = time.process_time()
WALL_START = time.monotonic()
CHILD_CPU = 0.0

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from scripts.training import diagnose_taiji_n1_plan167_train_decoder as old_diagnostic  # noqa: E402
from scripts.training import run_taiji_n1_plan168_isolated_train_fit as old_fit  # noqa: E402
from scripts.training import summarize_taiji_n1_plan167_train_decoder as statistics  # noqa: E402
from taiji.conditional_language import (
    ConditionalLanguageCandidate,
    ConditionalLanguageTrainer,
)  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.language_alignment import LanguageAlignmentTrainer  # noqa: E402
from taiji.task_core import TaskCoreConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_n1_plan169_task_core_calibration_20261025.json"
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.json")
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan169_task_core_20261025"
MODULE = "scripts.training.calibrate_taiji_n1_plan169_task_core"


def guard():
    if time.process_time() - CPU_START + CHILD_CPU > 1140 or time.monotonic() - WALL_START > 1740:
        raise TimeoutError("N1-169 reached its limit with 60 seconds reserved for publication")


def prompt_state(model, prompt):
    state = model.new_state(
        scope_id="n169-train", session_id="pilot", native_state_ref="research:n169-language-slice"
    )
    return model.step_task(
        prompt,
        state,
        scope_id=state.scope_id,
        session_id=state.session_id,
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 512},
    )


def checkpoint_signatures(trainer, state):
    with torch.no_grad():
        logits = trainer.model.next_logits(state)[0]
        continuation = trainer.model.generate(state, max_new_bytes=15).generated
    return {"next_logits_digest": content_digest(logits), "continuation_hex": continuation.hex()}


def save_verify(trainer, prompt, label):
    import subprocess

    global CHILD_CPU
    path = CHECKPOINTS / f"{label}.pt"
    with torch.no_grad():
        state = trainer.model.generate(prompt_state(trainer.model, prompt), max_new_bytes=1)
    cursor = {"update": trainer.steps, "next_batch": 0, "partition": "train-only"}
    expected = trainer.checkpoint(state, cursor=cursor)
    trainer.save(path, state, cursor=cursor)
    signatures = checkpoint_signatures(trainer, state)
    started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", MODULE, "--verify", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        stdout, stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        CHILD_CPU += old_fit.process_cpu(process)
        raise TimeoutError("candidate checkpoint fresh restore exceeded 120 seconds") from None
    cpu = old_fit.process_cpu(process)
    CHILD_CPU += cpu
    if process.returncode != 0:
        raise ValueError(f"candidate fresh restore failed: {stderr.strip()}")
    receipt = json.loads(stdout)
    if receipt["checkpoint_digest"] != expected["checkpoint_digest"] or any(
        receipt[key] != value for key, value in signatures.items()
    ):
        raise ValueError(
            "candidate payload/Adam/RNG/cursor or continued generation changed after restore"
        )
    return {
        **receipt,
        "path": str(path),
        "sha256": old_diagnostic.frozen.sha256_file(path),
        "process_cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - started,
    }


def evaluate(model, examples, step):
    rows = []
    with torch.no_grad():
        for episode, (prompt, target) in examples:
            guard()
            logits = model.teacher_forced(prompt, target)
            predicted = logits.argmax(-1).tolist()
            counts = [
                {"target": symbol, "raw_top1": guess}
                for symbol, guess in zip(target, predicted, strict=True)
            ]
            correct, total = statistics.cjk_accuracy(counts)
            state = model.generate(prompt_state(model, prompt), max_new_bytes=128)
            rows.append(
                {
                    "step": step,
                    "episode_id": episode.episode_id,
                    "target_bytes": len(target),
                    "tf_top1": sum(
                        symbol == guess for symbol, guess in zip(target, predicted, strict=True)
                    )
                    / len(target),
                    "nll_bits": float(F.cross_entropy(logits, torch.tensor(list(target))))
                    / math.log(2),
                    "cjk_correct": correct,
                    "cjk_total": total,
                    "generated_hex": state.generated.hex(),
                    "generated_text": state.generated.decode("utf-8", errors="replace"),
                    "utf8_pending": state.utf8_remaining,
                    "free_target_prefix_match": old_diagnostic.prefix_match(
                        state.generated, target
                    ),
                }
            )
    return rows


class LocalLanguageControl:
    """Callable native local-rule control; calibration instances are read-only."""

    def __init__(self, trainer, *, allow_updates=False):
        self.trainer, self.allow_updates = trainer, allow_updates

    def update_episode(self, episode, epoch):
        if not self.allow_updates:
            raise ValueError("calibration control is read-only")
        return old_fit.train_one(self.trainer, episode, epoch)

    def score(self, episodes):
        snapshot = self.trainer.model.checkpoint()
        before = content_digest(snapshot)
        rows = []
        started = time.process_time()
        for episode in episodes:
            guard()
            old_diagnostic.restore(self.trainer, snapshot)
            teacher, _ = old_diagnostic.teacher_forced(self.trainer, episode, "configured")
            old_diagnostic.restore(self.trainer, snapshot)
            raw = self.trainer._constrained_generate(episode, max_generation_bytes=128)
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "tf_top1": teacher["top1_raw"],
                    "free_target_prefix_match": old_diagnostic.prefix_match(
                        raw, episode.target_bytes
                    ),
                    "generated_text": self.trainer._generated_text(raw)[0],
                }
            )
        old_diagnostic.restore(self.trainer, snapshot)
        if content_digest(self.trainer.model.checkpoint()) != before:
            raise ValueError("old local control changed during read-only scoring")
        return {
            "parameter_count": self.trainer.model.parameter_count(),
            "read_only_verified": True,
            "cpu_seconds": time.process_time() - started,
            "records": rows,
        }


def run():
    torch.set_num_threads(1)
    if REPORT.exists() or CHECKPOINTS.exists():
        raise ValueError("N1-169 calibration already exists; do not overwrite or extend")
    CHECKPOINTS.mkdir(parents=True)
    report = {
        "schema": "taiji-n1-plan169-task-core-calibration-v1",
        "status": "running",
        "steps": [],
        "milestones": [],
        "restores": {},
        "controls": {},
        "dev_queries": 0,
        "can_promote": False,
        "world_goal_consumers_implemented": False,
        "source_sha256": {},
    }
    paths = [
        Path(__file__),
        old_diagnostic.OUTPUT,
        old_fit.REPORT,
        old_fit.CHECKPOINTS / "final.pt",
        ROOT / "taiji/task_state.py",
        ROOT / "taiji/task_core.py",
        ROOT / "taiji/conditional_language.py",
        ROOT / "taiji/adapter.py",
        ROOT / "plans/reference/PLAN-N1-169_shared_language_chain_design_20261025.md",
        ROOT / "plans/reference/SPEC-N1-169_shared_language_candidate_20261025.md",
    ]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): old_diagnostic.frozen.sha256_file(path) for path in paths
    }
    trainer = None
    try:
        corpus = old_diagnostic.base._load_assistant_train_corpus()
        episodes = old_fit.selected(corpus)
        examples = [
            (episode, (episode.prompt_bytes, episode.target_bytes[:128])) for episode in episodes
        ]
        batch = [example for _, example in examples]
        if sum(len(target) for _, target in batch) != 495:
            raise ValueError("calibration target exposure differs from preregistration")
        report["selection"] = [
            {
                "episode_id": ep.episode_id,
                "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
                "target_fragment_sha256": hashlib.sha256(target).hexdigest(),
                "target_bytes": len(target),
            }
            for ep, (prompt, target) in examples
        ]
        model = ConditionalLanguageCandidate(TaskCoreConfig(), bundle_id="n169-pilot-seed169")
        trainer = ConditionalLanguageTrainer(model)
        report["parameter_count"] = model.parameter_count()
        if model.parameter_count() > 1000000:
            raise ValueError("candidate exceeds the shared capacity ceiling")
        report["restores"]["zero"] = save_verify(trainer, batch[0][0], "zero")
        report["milestones"].extend(evaluate(model, examples, 0))
        for step in range(1, 65):
            guard()
            started = time.process_time()
            record = trainer.update(batch)
            record["update_cpu_seconds"] = time.process_time() - started
            report["steps"].append(record)
            if step in {1, 8, 32, 64}:
                report["milestones"].extend(evaluate(model, examples, step))
            if step == 1:
                report["restores"]["one_update"] = save_verify(trainer, batch[0][0], "one_update")
                # Resume the actual pilot from this verified bundle; never select by score.
                trainer, _, restored_cursor = ConditionalLanguageTrainer.from_checkpoint(
                    torch.load(CHECKPOINTS / "one_update.pt", map_location="cpu", weights_only=True)
                )
                if restored_cursor["update"] != 1:
                    raise ValueError("one-update continuation cursor differs")
                model = trainer.model
            PROGRESS.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        report["restores"]["final"] = save_verify(trainer, batch[0][0], "final")
        # No control update: initialization/history and target lengths differ from pilot.
        endpoint = next(
            row
            for row in old_diagnostic.frozen.read_json(old_diagnostic.READINESS)["endpoints"]
            if row["endpoint_id"] == "full_turn_1m"
        )
        native, _ = old_diagnostic.frozen._load_endpoint_trainer(endpoint, corpus)
        report["controls"]["full_turn_parent"] = LocalLanguageControl(native).score(episodes)
        payload = torch.load(
            old_fit.CHECKPOINTS / "final.pt", map_location="cpu", weights_only=True
        )
        local = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
        report["controls"]["n168_child"] = LocalLanguageControl(local).score(episodes)
        report.update(
            status="complete",
            target_byte_exposures=trainer.target_byte_exposures,
            optimizer_steps=trainer.steps,
        )
    except Exception as exc:
        report.update(
            status="partial" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        if trainer is not None:
            try:
                report["restores"]["partial"] = save_verify(trainer, batch[0][0], "partial")
            except Exception as restore_exc:
                report["restore_error"] = str(restore_exc)
    report["source_unchanged_after_run"] = all(
        old_diagnostic.frozen.sha256_file(ROOT / name) == digest
        for name, digest in report["source_sha256"].items()
    )
    if not report["source_unchanged_after_run"]:
        report.update(status="failed", source_error="input/source changed during calibration")
    report.update(
        parent_cpu_seconds=time.process_time() - CPU_START,
        child_cpu_seconds=CHILD_CPU,
        cpu_seconds=time.process_time() - CPU_START + CHILD_CPU,
        wall_seconds=time.monotonic() - WALL_START,
    )
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "steps": len(report["steps"]),
                "cpu_seconds": report["cpu_seconds"],
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.run:
        return run()
    torch.set_num_threads(1)
    payload = torch.load(args.verify, map_location="cpu", weights_only=True)
    trainer, state, cursor = ConditionalLanguageTrainer.from_checkpoint(payload)
    if (
        trainer.checkpoint(state, cursor=cursor)["checkpoint_digest"]
        != payload["checkpoint_digest"]
    ):
        raise ValueError("full checkpoint did not roundtrip")
    print(
        json.dumps(
            {
                "checkpoint_digest": payload["checkpoint_digest"],
                **checkpoint_signatures(trainer, state),
                "steps": trainer.steps,
                "target_byte_exposures": trainer.target_byte_exposures,
                "parameter_count": trainer.model.parameter_count(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
