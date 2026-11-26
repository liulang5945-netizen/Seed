"""N1-170: same 64-batch calibration for the per-byte evidence candidate v2.

Read-only reference numbers come from the frozen N1-169 report; this run never
touches dev/reference data, the default model or the N1-169 checkpoints.
"""

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
from taiji.conditional_language_v2 import (  # noqa: E402
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.task_core_v2 import TaskCoreConfigV2  # noqa: E402
from taiji.utf8_state import utf8_allowed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_n1_plan170_byte_evidence_calibration_20261025.json"
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.json")
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan170_byte_evidence_20261025"
N169_REPORT = ROOT / "reports/taiji_n1_plan169_task_core_calibration_20261025.json"
MODULE = "scripts.training.calibrate_taiji_n1_plan170_byte_evidence"


def guard() -> None:
    if time.process_time() - CPU_START + CHILD_CPU > 1140 or time.monotonic() - WALL_START > 1740:
        raise TimeoutError("N1-170 reached its limit with 60 seconds reserved for publication")


def prompt_state(model, prompt):
    state = model.new_state(
        scope_id="n170-train", session_id="pilot", native_state_ref="research:n170-language-slice"
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
        raise ValueError("candidate payload/Adam/RNG/cursor or continued generation changed")
    return {
        **receipt,
        "path": str(path),
        "sha256": old_diagnostic.frozen.sha256_file(path),
        "process_cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - started,
    }


def _runs(slots: list[int]) -> int:
    longest = current = 0
    previous = None
    for slot in slots:
        current = current + 1 if slot == previous else 1
        previous = slot
        longest = max(longest, current)
    return longest


def teacher_trace(model, prompt, target):
    state = prompt_state(model, prompt)
    cache = model.evidence_cache(state)
    logits, slots = [], []
    for symbol in target:
        weights = model.attention_weights(state, cache=cache)
        current, hidden = model.next_logits(state, cache=cache)
        logits.append(current)
        slots.append(int(weights.argmax()))
        state = model.observe_response(state, symbol, hidden)
    return torch.stack(logits), slots


def free_trace(model, prompt, limit=128):
    state = prompt_state(model, prompt)
    cache = model.evidence_cache(state)
    slots = []
    for _ in range(limit):
        if state.generated.endswith(b"<|end|>"):
            break
        weights = model.attention_weights(state, cache=cache)
        logits, hidden = model.next_logits(state, cache=cache)
        allowed = torch.tensor(utf8_allowed(state.utf8_remaining, state.utf8_lead))
        symbol = int(allowed[logits[allowed].argmax()])
        slots.append(int(weights.argmax()))
        state = model.observe_response(state, symbol, hidden)
    return state, slots


def evaluate(model, examples, step):
    rows = []
    with torch.no_grad():
        for episode, (prompt, target) in examples:
            guard()
            logits, tf_slots = teacher_trace(model, prompt, target)
            predicted = logits.argmax(-1).tolist()
            counts = [
                {"target": symbol, "raw_top1": guess}
                for symbol, guess in zip(target, predicted, strict=True)
            ]
            correct, total = statistics.cjk_accuracy(counts)
            state, free_slots = free_trace(model, prompt)
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
                    "tf_distinct_positions": len(set(tf_slots)),
                    "tf_longest_same_position_run": _runs(tf_slots),
                    "free_distinct_positions": len(set(free_slots)),
                    "free_longest_same_position_run": _runs(free_slots),
                }
            )
    return rows


def run() -> int:
    torch.set_num_threads(1)
    if REPORT.exists() or CHECKPOINTS.exists():
        raise ValueError("N1-170 calibration already exists; do not overwrite or extend")
    CHECKPOINTS.mkdir(parents=True)
    report = {
        "schema": "taiji-n1-plan170-byte-evidence-calibration-v1",
        "status": "running",
        "steps": [],
        "milestones": [],
        "restores": {},
        "reference": {},
        "dev_queries": 0,
        "can_promote": False,
        "world_goal_consumers_implemented": False,
        "source_sha256": {},
    }
    paths = [
        Path(__file__),
        N169_REPORT,
        ROOT / "taiji/task_state_v2.py",
        ROOT / "taiji/task_core_v2.py",
        ROOT / "taiji/conditional_language_v2.py",
        ROOT / "tests/taiji_native/test_task_language_candidate_v2.py",
        ROOT / "plans/reference/PLAN-N1-170_byte_evidence_architecture_20261025.md",
        ROOT / "plans/reference/SPEC-N1-170_byte_evidence_candidate_20261025.md",
    ]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): old_diagnostic.frozen.sha256_file(path) for path in paths
    }
    reference = json.loads(N169_REPORT.read_text(encoding="utf-8"))
    report["reference"] = {
        "report": str(N169_REPORT.relative_to(ROOT)),
        "sha256": old_diagnostic.frozen.sha256_file(N169_REPORT),
        "step64": [
            {
                "episode_id": row["episode_id"],
                "tf_top1": row["tf_top1"],
                "free_target_prefix_match": row["free_target_prefix_match"],
            }
            for row in reference["milestones"]
            if row["step"] == 64
        ],
        "parameter_count": reference["parameter_count"],
        "target_byte_exposures": reference["target_byte_exposures"],
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
        model = ConditionalLanguageCandidateV2(
            TaskCoreConfigV2(), bundle_id="n170-byte-evidence-seed169"
        )
        trainer = ConditionalLanguageTrainerV2(model)
        report["parameter_count"] = model.parameter_count()
        if model.parameter_count() != 970945 or model.parameter_count() > 1000000:
            raise ValueError("candidate parameter count differs from the frozen v2 capacity")
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
            PROGRESS.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        report["restores"]["final"] = save_verify(trainer, batch[0][0], "final")
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
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "steps": trainer.steps if trainer else 0,
                "cpu_seconds": round(report["cpu_seconds"], 6),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "complete" else 1


def verify(path: str) -> int:
    torch.set_num_threads(1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    trainer, state, cursor = ConditionalLanguageTrainerV2.from_checkpoint(payload)
    signatures = checkpoint_signatures(trainer, state)
    print(
        json.dumps(
            {
                "checkpoint_digest": payload["checkpoint_digest"],
                "steps": trainer.steps,
                "target_byte_exposures": trainer.target_byte_exposures,
                "parameter_count": trainer.model.parameter_count(),
                "cursor": cursor,
                **signatures,
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify")
    args = parser.parse_args()
    if args.verify:
        return verify(args.verify)
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
