"""N1-172: fit-first exposure probe on the frozen N1-171 selection.

Folds 0 and 1 keep their twelve training pairs and four held-out pairs; the only
change is an extended update budget (up to 176 full-batch updates) with an early
stop when the training median teacher-forced top-1 first reaches .90. The stop
test never touches held-out episodes. Write-once report, no dev/reference.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

CPU_START = time.process_time()
WALL_START = time.monotonic()
CHILD_CPU = 0.0

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from scripts.training import calibrate_taiji_n1_plan171_holdout as baseline  # noqa: E402
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
REPORT = ROOT / "reports/taiji_n1_plan172_fit_first_exposure_20261025.json"
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.json")
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan172_fit_first_20261025"
MODULE = "scripts.training.calibrate_taiji_n1_plan172_fit_first"
SELECTION_DIGEST = "5db33945cd22ce71d70d8d97f6af72462d4871aa239559876e79cd1e1edd6b40"
FOLDS = (0, 1)
TRAIN_EPISODES = 12
HELDOUT_EPISODES = 4
UPDATES = 176
CHECK_EVERY = 16
TARGET_BYTES = 128
PARAMETERS = 970945
OWN_GATE = 0.20
MARGIN = 0.15
FIT_GATE = 0.90
BINDING_HITS = 4
CHECK_STEPS = tuple(range(32, UPDATES + 1, CHECK_EVERY))


def guard() -> None:
    if time.process_time() - CPU_START + CHILD_CPU > 700 or time.monotonic() - WALL_START > 1240:
        raise TimeoutError("N1-172 reached its limit with 20 seconds reserved for publication")


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def example(episode) -> tuple[bytes, bytes]:
    return episode.prompt_bytes, episode.target_bytes[:TARGET_BYTES]


def prompt_state(model, prompt: bytes):
    state = model.new_state(
        scope_id="n172-train",
        session_id="fit-first",
        native_state_ref="research:n172-language-fit-first",
    )
    return model.step_task(
        prompt,
        state,
        scope_id=state.scope_id,
        session_id=state.session_id,
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 512},
    )


def teacher_trace(model, prompt: bytes, target: bytes):
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


def free_trace(model, prompt: bytes, limit: int = TARGET_BYTES):
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


def runs(slots: list[int]) -> int:
    longest = current = 0
    previous = None
    for slot in slots:
        current = current + 1 if slot == previous else 1
        previous = slot
        longest = max(longest, current)
    return longest


def train_tf_medians(model, train):
    values = []
    with torch.no_grad():
        for episode in train:
            target = episode.target_bytes[:TARGET_BYTES]
            logits, _ = teacher_trace(model, episode.prompt_bytes, target)
            predicted = logits.argmax(-1)
            reference = torch.tensor(list(target))
            values.append(float((predicted == reference).float().mean()))
    return values


def evaluate(model, items, step: int, fold: int):
    rows = []
    with torch.no_grad():
        for kind, episode, prompt, target in items:
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
                    "fold": fold,
                    "step": step,
                    "kind": kind,
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
                    "free_target_prefix_match": old_diagnostic.prefix_match(
                        state.generated, target
                    ),
                    "tf_distinct_positions": len(set(tf_slots)),
                    "tf_longest_same_position_run": runs(tf_slots),
                    "free_distinct_positions": len(set(free_slots)),
                }
            )
    return rows


def controls(model, fold: int, held, train, final_rows):
    by_id = {row["episode_id"]: row for row in final_rows}
    entries = []
    with torch.no_grad():
        for index, episode in enumerate(held):
            guard()
            row = by_id[episode.episode_id]
            generated = bytes.fromhex(row["generated_hex"])
            other = held[(index + 1) % len(held)]
            training = train[(3 * index + 1) % len(train)]
            logits, _ = teacher_trace(
                model, episode.prompt_bytes, other.target_bytes[:TARGET_BYTES]
            )
            mismatched = logits.argmax(-1).tolist()
            reference = list(other.target_bytes[:TARGET_BYTES])
            entries.append(
                {
                    "fold": fold,
                    "episode_id": episode.episode_id,
                    "own_match": row["free_target_prefix_match"],
                    "control_a_episode_id": other.episode_id,
                    "control_a_match": old_diagnostic.prefix_match(
                        generated, other.target_bytes[:TARGET_BYTES]
                    ),
                    "control_b_episode_id": training.episode_id,
                    "control_b_match": old_diagnostic.prefix_match(
                        generated, training.target_bytes[:TARGET_BYTES]
                    ),
                    "mismatched_tf_top1": sum(
                        symbol == guess for symbol, guess in zip(reference, mismatched, strict=True)
                    )
                    / len(reference),
                }
            )
    return entries


def summarize(fold_rows, control_rows, fit_checks, stopped, fit_reached):
    final = [row for row in fold_rows if row["step"] == stopped]
    train_final = [row for row in final if row["kind"] == "train"]
    held_final = [row for row in final if row["kind"] == "heldout"]
    if len(train_final) != TRAIN_EPISODES or len(held_final) != HELDOUT_EPISODES:
        raise ValueError("final evaluation rows differ from the fold split")
    return {
        "stopped_update": stopped,
        "fit_reached": fit_reached,
        "fit_checks": fit_checks,
        "train_tf_top1_median": median([row["tf_top1"] for row in train_final]),
        "train_free_match_median": median([row["free_target_prefix_match"] for row in train_final]),
        "train_free_match_values": [row["free_target_prefix_match"] for row in train_final],
        "heldout_tf_top1_median": median([row["tf_top1"] for row in held_final]),
        "heldout_free_match_values": [row["free_target_prefix_match"] for row in held_final],
        "heldout_free_match_mean": sum(row["free_target_prefix_match"] for row in held_final)
        / HELDOUT_EPISODES,
        "heldout_distinct_outputs": len({row["generated_hex"] for row in held_final}),
        "own_match_mean": sum(entry["own_match"] for entry in control_rows) / len(control_rows),
        "control_a_mean": sum(entry["control_a_match"] for entry in control_rows)
        / len(control_rows),
        "control_b_mean": sum(entry["control_b_match"] for entry in control_rows)
        / len(control_rows),
        "heldout_mismatched_tf_top1_mean": sum(
            entry["mismatched_tf_top1"] for entry in control_rows
        )
        / len(control_rows),
    }


def checkpoint_signatures(trainer, state):
    with torch.no_grad():
        logits = trainer.model.next_logits(state)[0]
        continuation = trainer.model.generate(state, max_new_bytes=15).generated
    return {"next_logits_digest": content_digest(logits), "continuation_hex": continuation.hex()}


def save_verify(trainer, prompt: bytes, label: str):
    global CHILD_CPU
    path = CHECKPOINTS / f"{label}.pt"
    with torch.no_grad():
        state = trainer.model.generate(prompt_state(trainer.model, prompt), max_new_bytes=1)
    cursor = {
        "fold": label,
        "update": trainer.steps,
        "next_batch": 0,
        "partition": "train-only",
    }
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


def split_fold(fold: int, chosen, records):
    held_ids = {record["episode_id"] for record in records if record["fold"] == fold}
    held = [episode for episode in chosen if episode.episode_id in held_ids]
    train = [episode for episode in chosen if episode.episode_id not in held_ids]
    if len(held) != HELDOUT_EPISODES or len(train) != TRAIN_EPISODES:
        raise ValueError("fold episode split differs from the preregistered counts")
    return held, train


def run_fold(fold: int, chosen, records):
    held, train = split_fold(fold, chosen, records)
    examples = [example(episode) for episode in train]
    if sum(len(target) for _, target in examples) != TRAIN_EPISODES * TARGET_BYTES:
        raise ValueError("training exposure differs from the preregistered byte budget")
    items = [("train", episode, *example(episode)) for episode in train]
    items += [("heldout", episode, *example(episode)) for episode in held]
    model = ConditionalLanguageCandidateV2(
        TaskCoreConfigV2(), bundle_id=f"n172-fit-first-fold{fold}"
    )
    trainer = ConditionalLanguageTrainerV2(model)
    if model.parameter_count() != PARAMETERS:
        raise ValueError("candidate parameter count differs from the frozen v2 capacity")
    fold_report = {
        "fold": fold,
        "train_episode_ids": [episode.episode_id for episode in train],
        "heldout_episode_ids": [episode.episode_id for episode in held],
        "parameter_count": model.parameter_count(),
        "steps": [],
        "fit_checks": [],
        "milestones": [],
        "controls": [],
    }
    zero = save_verify(trainer, examples[0][0], "zero") if fold == 0 else None
    stopped = UPDATES
    fit_reached = False
    for step in range(1, UPDATES + 1):
        guard()
        started = time.process_time()
        record = trainer.update(examples)
        record["fold"] = fold
        record["update_cpu_seconds"] = time.process_time() - started
        fold_report["steps"].append(record)
        if step in CHECK_STEPS:
            values = train_tf_medians(model, train)
            check = {"update": step, "train_tf_top1_median": median(values)}
            fold_report["fit_checks"].append(check)
            if check["train_tf_top1_median"] >= FIT_GATE:
                stopped = step
                fit_reached = True
                break
        PROGRESS.write_text(
            json.dumps(
                {
                    "status": "training",
                    "fold": fold,
                    "updates": len(fold_report["steps"]),
                    "last": record,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    fold_report["milestones"].extend(evaluate(model, items, stopped, fold))
    final_rows = [row for row in fold_report["milestones"] if row["step"] == stopped]
    fold_report["controls"] = controls(
        model, fold, held, train, [row for row in final_rows if row["kind"] == "heldout"]
    )
    final = save_verify(trainer, examples[0][0], f"fold{fold}_final")
    fold_report["summary"] = summarize(
        fold_report["milestones"],
        fold_report["controls"],
        fold_report["fit_checks"],
        stopped,
        fit_reached,
    )
    return fold_report, final, zero


def verdict(folds):
    fit_all = all(fold["summary"]["fit_reached"] for fold in folds)
    entries = [entry for fold in folds for entry in fold["controls"]]
    hits = [
        entry
        for entry in entries
        if entry["own_match"] >= OWN_GATE
        and entry["own_match"] >= entry["control_a_match"] + MARGIN
        and entry["own_match"] >= entry["control_b_match"] + MARGIN
    ]
    qualified = [entry for entry in entries if entry["own_match"] >= OWN_GATE]
    gate = fit_all and len(hits) >= BINDING_HITS
    if gate:
        status = "PASS_BINDING"
    elif fit_all:
        status = "FAIL_NO_GENERALIZATION"
    else:
        status = "INCONCLUSIVE_UNDERFIT"
    return {
        "status": status,
        "gate_passed": gate,
        "fit_all": fit_all,
        "fit_updates": [fold["summary"]["stopped_update"] for fold in folds],
        "binding_hit_count": len(hits),
        "qualified_count": len(qualified),
        "total_heldout": len(entries),
        "hit_episode_ids": [entry["episode_id"] for entry in hits],
        "qualified_episode_ids": [entry["episode_id"] for entry in qualified],
        "per_fold_qualified": [
            sum(entry["own_match"] >= OWN_GATE for entry in fold["controls"]) for fold in folds
        ],
    }


def run() -> int:
    torch.set_num_threads(1)
    if REPORT.exists() or CHECKPOINTS.exists():
        raise ValueError("N1-172 fit-first probe already exists; do not overwrite or extend")
    CHECKPOINTS.mkdir(parents=True)
    report = {
        "schema": "taiji-n1-plan172-fit-first-exposure-v1",
        "status": "running",
        "criteria": {
            "fit_gate": FIT_GATE,
            "own_gate": OWN_GATE,
            "margin": MARGIN,
            "binding_hits_required": BINDING_HITS,
            "max_updates": UPDATES,
            "check_steps": list(CHECK_STEPS),
            "target_bytes": TARGET_BYTES,
            "selection_digest_expected": SELECTION_DIGEST,
        },
        "folds": [],
        "restores": {},
        "dev_queries": 0,
        "can_promote": False,
        "source_sha256": {},
    }
    paths = [
        Path(__file__),
        ROOT / "scripts/training/calibrate_taiji_n1_plan171_holdout.py",
        ROOT / "taiji/task_state_v2.py",
        ROOT / "taiji/task_core_v2.py",
        ROOT / "taiji/conditional_language_v2.py",
        ROOT / "tests/taiji_native/test_task_language_candidate_v2.py",
        ROOT / "plans/reference/PLAN-N1-172_fit_first_exposure_20261025.md",
        ROOT / "plans/reference/SPEC-N1-172_fit_first_exposure_20261025.md",
    ]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): old_diagnostic.frozen.sha256_file(path) for path in paths
    }
    try:
        corpus = old_diagnostic.base._load_assistant_train_corpus()
        records, chosen, digest = baseline.select(corpus)
        if digest != SELECTION_DIGEST:
            raise ValueError("selection digest differs from the frozen N1-171 selection")
        report["selection"] = {
            "selection_digest": digest,
            "records": records,
            "folds": list(FOLDS),
        }
        zero_rows = None
        for fold in FOLDS:
            guard()
            held, train = split_fold(fold, chosen, records)
            if zero_rows is None:
                items = [("train", episode, *example(episode)) for episode in train]
                items += [("heldout", episode, *example(episode)) for episode in held]
                model = ConditionalLanguageCandidateV2(
                    TaskCoreConfigV2(), bundle_id=f"n172-fit-first-fold{fold}"
                )
                zero_rows = evaluate(model, items, 0, fold)
                report["zero_milestones"] = zero_rows
            fold_report, final, zero = run_fold(fold, chosen, records)
            if zero is not None:
                report["restores"]["zero"] = zero
            report["restores"][f"fold{fold}_final"] = final
            report["folds"].append(fold_report)
            if len(report["folds"]) == len(FOLDS):
                report["verdict"] = verdict(report["folds"])
            PROGRESS.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        report["verdict"] = verdict(report["folds"])
        report.update(
            status="complete",
            parent_cpu_seconds=time.process_time() - CPU_START,
            child_cpu_seconds=CHILD_CPU,
            cpu_seconds=time.process_time() - CPU_START + CHILD_CPU,
            wall_seconds=time.monotonic() - WALL_START,
        )
    except Exception as exc:
        report.update(
            status="partial" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
            parent_cpu_seconds=time.process_time() - CPU_START,
            child_cpu_seconds=CHILD_CPU,
            cpu_seconds=time.process_time() - CPU_START + CHILD_CPU,
            wall_seconds=time.monotonic() - WALL_START,
        )
    report["source_unchanged_after_run"] = all(
        old_diagnostic.frozen.sha256_file(ROOT / name) == value
        for name, value in report["source_sha256"].items()
    )
    if not report["source_unchanged_after_run"]:
        report.update(status="failed", source_error="input/source changed during the probe")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "verdict": report.get("verdict", {}).get("status"),
                "binding_hits": report.get("verdict", {}).get("binding_hit_count"),
                "fit_updates": report.get("verdict", {}).get("fit_updates"),
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
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    if args.verify:
        return verify(args.verify)
    if args.select_only:
        corpus = old_diagnostic.base._load_assistant_train_corpus()
        records, chosen, digest = baseline.select(corpus)
        held = {
            fold: [record["episode_id"][6:14] for record in records if record["fold"] == fold]
            for fold in FOLDS
        }
        print(
            json.dumps(
                {
                    "selection_digest": digest,
                    "matches_frozen": digest == SELECTION_DIGEST,
                    "heldout": held,
                    "train_episodes_per_fold": len(chosen) - len(held[FOLDS[0]]),
                },
                ensure_ascii=False,
            )
        )
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
