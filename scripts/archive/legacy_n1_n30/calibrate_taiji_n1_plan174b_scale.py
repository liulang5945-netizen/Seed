"""N1-174: scale comparison on 500 train episodes, two arms, three seeds.

Each run trains the v2 candidate on the frozen N1-171 held-out complement with
four-episode batches for 40 epochs. ``tf`` uses teacher forcing only; ``ss``
mixes the model's own greedy byte with probability p = min(0.5, step/256).
Held-out episodes never enter any update. Reports are per run and write-once;
``--all`` launches six single-threaded runs in parallel and ``--aggregate``
applies the preregistered R/G gates. N1-174B reruns the same design after the
N1-174 runs were truncated by a runaway guard that sat below the measured cost;
the guard is raised and epoch-20/30 checkpoints are added for durable recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
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
from taiji.conditional_language_rollout import ConditionalLanguageRolloutTrainer  # noqa: E402
from taiji.conditional_language_v2 import (  # noqa: E402
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.task_core_v2 import TaskCoreConfigV2  # noqa: E402
from taiji.utf8_state import utf8_allowed  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan174b_scale_20261025"
MODULE = "scripts.training.calibrate_taiji_n1_plan174b_scale"
SELECTION_DIGEST = "5db33945cd22ce71d70d8d97f6af72462d4871aa239559876e79cd1e1edd6b40"
ARMS = ("tf", "ss")
SEEDS = (169, 170, 171)
EPOCHS = 40
EVAL_EPOCHS = (10, 20, 40)
BATCH = 4
TARGET_BYTES = 128
PARAMETERS = 970945
TRAIN_EPISODES = 500
TRAIN_FAMILIES = 324
TRAIN_SAMPLE = 16
HELDOUT_EPISODES = 16
OWN_GATE = 0.20
MARGIN = 0.15
ROLLOUT_GATE = 0.50
BINDING_HITS = 24
HIT_LEAD = 8
MAX_PROBABILITY = 0.5
SCHEDULE_STEPS = 256


def run_report(arm: str, seed: int) -> Path:
    return REPORTS / f"taiji_n1_plan174b_scale_20261025_{arm}_seed{seed}.json"


def run_progress(arm: str, seed: int) -> Path:
    return REPORTS / f"taiji_n1_plan174b_scale_20261025_{arm}_seed{seed}_progress.json"


def aggregate_report() -> Path:
    return REPORTS / "taiji_n1_plan174b_scale_20261025.json"


def guard() -> None:
    if time.process_time() - CPU_START > 5000 or time.monotonic() - WALL_START > 7000:
        raise TimeoutError("N1-174B run reached its runaway-protection threshold")


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def example(episode) -> tuple[bytes, bytes]:
    return episode.prompt_bytes, episode.target_bytes[:TARGET_BYTES]


def load_data():
    corpus = old_diagnostic.base._load_assistant_train_corpus()
    records, chosen, digest = baseline.select(corpus)
    if digest != SELECTION_DIGEST:
        raise ValueError("selection digest differs from the frozen N1-171 selection")
    held_families = {record["family_id"] for record in records}
    train = [
        episode for episode in corpus.for_split("train") if episode.family_id not in held_families
    ]
    if len(train) != TRAIN_EPISODES or len({e.family_id for e in train}) != TRAIN_FAMILIES:
        raise ValueError("training corpus differs from the preregistered 500/324 split")
    held_ids = {record["episode_id"] for record in records}
    if held_ids & {episode.episode_id for episode in train}:
        raise ValueError("held-out episode appears on the training side")
    if any(not e.prompt_bytes or not e.target_bytes for e in train + chosen):
        raise ValueError("corpus contains an empty prompt or target")
    return corpus, records, chosen, train


def train_sample(train):
    ordered = sorted(train, key=lambda e: hashlib.sha256(e.episode_id.encode()).hexdigest())
    step = (len(ordered) - 1) / (TRAIN_SAMPLE - 1)
    sample = [ordered[round(index * step)] for index in range(TRAIN_SAMPLE)]
    if len({episode.episode_id for episode in sample}) != TRAIN_SAMPLE:
        raise ValueError("deterministic training sample repeats an episode")
    return sample


def epoch_batches(train, seed: int, epoch: int):
    indices = list(range(len(train)))
    random.Random((seed * 1000003) ^ epoch).shuffle(indices)
    order = [train[index].episode_id for index in indices]
    digest = hashlib.sha256("|".join(order).encode()).hexdigest()
    batches = [indices[start : start + BATCH] for start in range(0, len(indices), BATCH)]
    if len(batches) != len(train) // BATCH:
        raise ValueError("batch count differs from the preregistered 125 per epoch")
    return batches, digest


def prompt_state(model, prompt: bytes):
    state = model.new_state(
        scope_id="n174-train",
        session_id="scale",
        native_state_ref="research:n174-language-scale",
    )
    return model.step_task(
        prompt,
        state,
        scope_id=state.scope_id,
        session_id=state.session_id,
        source_id="prompt",
        resource_snapshot={
            "remaining_steps": 1,
            "max_observation_bytes": model.config.max_observation_bytes,
        },
    )


def teacher_trace(model, prompt: bytes, target: bytes):
    state = prompt_state(model, prompt)
    cache = model.evidence_cache(state)
    logits = []
    for symbol in target:
        current, hidden = model.next_logits(state, cache=cache)
        logits.append(current)
        state = model.observe_response(state, symbol, hidden)
    return torch.stack(logits)


def free_trace(model, prompt: bytes, limit: int = TARGET_BYTES):
    state = prompt_state(model, prompt)
    cache = model.evidence_cache(state)
    for _ in range(limit):
        if state.generated.endswith(b"<|end|>"):
            break
        logits, hidden = model.next_logits(state, cache=cache)
        allowed = torch.tensor(utf8_allowed(state.utf8_remaining, state.utf8_lead))
        symbol = int(allowed[logits[allowed].argmax()])
        state = model.observe_response(state, symbol, hidden)
    return state


def evaluate(model, items, epoch: int, run: str):
    rows = []
    with torch.no_grad():
        for kind, episode, prompt, target in items:
            guard()
            logits = teacher_trace(model, prompt, target)
            predicted = logits.argmax(-1).tolist()
            counts = [
                {"target": symbol, "raw_top1": guess}
                for symbol, guess in zip(target, predicted, strict=True)
            ]
            correct, total = statistics.cjk_accuracy(counts)
            state = free_trace(model, prompt)
            rows.append(
                {
                    "run": run,
                    "epoch": epoch,
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
                }
            )
    return rows


def controls(model, held, train, final_rows, rank_by_id):
    by_id = {row["episode_id"]: row for row in final_rows}
    fold_peers: dict[int, list] = {}
    for episode in held:
        fold_peers.setdefault(rank_by_id[episode.episode_id] % 4, []).append(episode)
    entries = []
    with torch.no_grad():
        for episode in held:
            guard()
            row = by_id[episode.episode_id]
            generated = bytes.fromhex(row["generated_hex"])
            rank = rank_by_id[episode.episode_id]
            peers = fold_peers[rank % 4]
            position = peers.index(episode)
            other = peers[(position + 1) % len(peers)]
            training = train[(rank * 37 + 11) % len(train)]
            logits = teacher_trace(model, episode.prompt_bytes, other.target_bytes[:TARGET_BYTES])
            mismatched = logits.argmax(-1).tolist()
            reference = list(other.target_bytes[:TARGET_BYTES])
            entries.append(
                {
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


def summarize(milestones, control_rows, epoch):
    final = [row for row in milestones if row["epoch"] == epoch]
    train_final = [row for row in final if row["kind"] == "train"]
    held_final = [row for row in final if row["kind"] == "heldout"]
    if len(train_final) != TRAIN_SAMPLE or len(held_final) != HELDOUT_EPISODES:
        raise ValueError("final evaluation rows differ from the preregistered samples")
    return {
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


def save_verify(trainer, prompt: bytes, label: str, directory: Path, cursor: dict):
    global CHILD_CPU
    path = directory / f"{label}.pt"
    with torch.no_grad():
        state = trainer.model.generate(prompt_state(trainer.model, prompt), max_new_bytes=1)
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
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        CHILD_CPU += old_fit.process_cpu(process)
        raise TimeoutError("scale checkpoint fresh restore exceeded 180 seconds") from None
    cpu = old_fit.process_cpu(process)
    CHILD_CPU += cpu
    if process.returncode != 0:
        raise ValueError(f"scale fresh restore failed: {stderr.strip()}")
    receipt = json.loads(stdout)
    if receipt["checkpoint_digest"] != expected["checkpoint_digest"] or any(
        receipt[key] != value for key, value in signatures.items()
    ):
        raise ValueError("scale checkpoint payload/Adam/RNG/cursor changed after restore")
    return {
        **receipt,
        "path": str(path),
        "sha256": old_diagnostic.frozen.sha256_file(path),
        "process_cpu_seconds": cpu,
        "wall_seconds": time.monotonic() - started,
    }


def source_hashes():
    paths = [
        Path(__file__),
        ROOT / "taiji/conditional_language_rollout.py",
        ROOT / "taiji/conditional_language_v2.py",
        ROOT / "taiji/task_core_v2.py",
        ROOT / "taiji/task_state_v2.py",
        ROOT / "scripts/training/calibrate_taiji_n1_plan171_holdout.py",
        ROOT / "tests/taiji_native/test_rollout_trainer.py",
        ROOT / "tests/taiji_native/test_task_language_candidate_v2.py",
        ROOT / "scripts/training/calibrate_taiji_n1_plan174_scale.py",
        ROOT / "plans/reference/PLAN-N1-174B_scale_comparison_rerun_20261025.md",
        ROOT / "plans/reference/SPEC-N1-174B_scale_comparison_rerun_20261025.md",
    ]
    return {str(path.relative_to(ROOT)): old_diagnostic.frozen.sha256_file(path) for path in paths}


def run_single(arm: str, seed: int, epochs: int) -> int:
    torch.set_num_threads(1)
    if arm not in ARMS or seed not in SEEDS or epochs != EPOCHS:
        raise ValueError("run identity differs from the preregistered arm/seed/epoch grid")
    report_path, progress_path = run_report(arm, seed), run_progress(arm, seed)
    directory = CHECKPOINTS / f"{arm}_seed{seed}"
    if report_path.exists() or directory.exists():
        raise ValueError("N1-174 run already exists; do not overwrite or extend")
    directory.mkdir(parents=True)
    run = f"{arm}_seed{seed}"
    report = {
        "schema": "taiji-n1-plan174-scale-run-v1",
        "status": "running",
        "run": run,
        "arm": arm,
        "seed": seed,
        "epochs": epochs,
        "batch": BATCH,
        "criteria": {
            "rollout_gate": ROLLOUT_GATE,
            "own_gate": OWN_GATE,
            "margin": MARGIN,
            "binding_hits": BINDING_HITS,
            "hit_lead": HIT_LEAD,
            "selection_digest_expected": SELECTION_DIGEST,
        },
        "source_sha256": source_hashes(),
        "dev_queries": 0,
        "can_promote": False,
        "restores": {},
        "epoch_log": [],
        "milestones": [],
        "controls": [],
    }
    trainer = None
    try:
        _, records, chosen, train = load_data()
        sample = train_sample(train)
        rank_by_id = {record["episode_id"]: record["rank"] for record in records}
        items = [("train", episode, *example(episode)) for episode in sample]
        items += [("heldout", episode, *example(episode)) for episode in chosen]
        model = ConditionalLanguageCandidateV2(
            TaskCoreConfigV2(), bundle_id=f"n174-scale-{arm}-{seed}"
        )
        if model.parameter_count() != PARAMETERS:
            raise ValueError("candidate parameter count differs from the frozen capacity")
        if arm == "ss":
            trainer = ConditionalLanguageRolloutTrainer(
                model, max_probability=MAX_PROBABILITY, schedule_steps=SCHEDULE_STEPS
            )
        else:
            trainer = ConditionalLanguageTrainerV2(model)
        report["restores"]["zero"] = save_verify(
            trainer, train[0].prompt_bytes, "zero", directory, {"epoch": 0, "batch": 0}
        )
        report["milestones"].extend(evaluate(model, items, 0, run))
        for epoch in range(1, epochs + 1):
            batches, digest = epoch_batches(train, seed, epoch)
            losses = []
            for index, indices in enumerate(batches, 1):
                guard()
                batch = [example(train[position]) for position in indices]
                record = trainer.update(batch)
                losses.append(record["loss_nats"])
                if index % 16 == 0:
                    progress_path.write_text(
                        json.dumps(
                            {
                                "run": run,
                                "status": "training",
                                "epoch": epoch,
                                "batch": index,
                                "loss_nats": record["loss_nats"],
                                "probability": record.get("probability"),
                            }
                        ),
                        encoding="utf-8",
                    )
            report["epoch_log"].append(
                {
                    "epoch": epoch,
                    "order_digest": digest,
                    "loss_nats_mean": sum(losses) / len(losses),
                    "loss_nats_last": losses[-1],
                }
            )
            if epoch in EVAL_EPOCHS:
                report["milestones"].extend(evaluate(model, items, epoch, run))
                progress_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
            if epoch in (20, 30):
                report["restores"][f"epoch{epoch}"] = save_verify(
                    trainer,
                    train[0].prompt_bytes,
                    f"epoch{epoch}",
                    directory,
                    {"epoch": epoch, "batch": len(train) // BATCH},
                )
                progress_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        final_epoch = epochs
        final_rows = [row for row in report["milestones"] if row["epoch"] == final_epoch]
        report["controls"] = controls(
            model,
            chosen,
            train,
            [row for row in final_rows if row["kind"] == "heldout"],
            rank_by_id,
        )
        report["summary"] = summarize(report["milestones"], report["controls"], final_epoch)
        report["restores"]["final"] = save_verify(
            trainer,
            train[0].prompt_bytes,
            "final",
            directory,
            {"epoch": final_epoch, "batch": len(train) // BATCH},
        )
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
        report.update(status="failed", source_error="input/source changed during the run")
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
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "complete" else 1


def pilot(arm: str, seed: int, epochs: int = 2) -> int:
    torch.set_num_threads(1)
    _, _, _, train = load_data()
    model = ConditionalLanguageCandidateV2(TaskCoreConfigV2(), bundle_id=f"n174-pilot-{arm}-{seed}")
    trainer = (
        ConditionalLanguageRolloutTrainer(
            model, max_probability=MAX_PROBABILITY, schedule_steps=SCHEDULE_STEPS
        )
        if arm == "ss"
        else ConditionalLanguageTrainerV2(model)
    )
    started = time.process_time()
    losses = []
    for epoch in range(1, epochs + 1):
        batches, _ = epoch_batches(train, seed, epoch)
        for indices in batches:
            losses.append(trainer.update([example(train[position]) for position in indices]))
    cpu = time.process_time() - started
    print(
        json.dumps(
            {
                "arm": arm,
                "seed": seed,
                "epochs": epochs,
                "updates": len(losses),
                "cpu_seconds": round(cpu, 3),
                "cpu_per_update": round(cpu / len(losses), 3),
                "first_loss": round(losses[0]["loss_nats"], 3),
                "last_loss": round(losses[-1]["loss_nats"], 3),
                "projected_40_epoch_cpu": round(cpu / epochs * EPOCHS, 1),
            }
        )
    )
    return 0


def smoke(arm: str, seed: int) -> int:
    torch.set_num_threads(1)
    _, records, chosen, train = load_data()
    rank_by_id = {record["episode_id"]: record["rank"] for record in records}
    items = [("train", episode, *example(episode)) for episode in train_sample(train)]
    items += [("heldout", episode, *example(episode)) for episode in chosen]
    model = ConditionalLanguageCandidateV2(TaskCoreConfigV2(), bundle_id=f"n174-smoke-{arm}-{seed}")
    trainer = (
        ConditionalLanguageRolloutTrainer(
            model, max_probability=MAX_PROBABILITY, schedule_steps=SCHEDULE_STEPS
        )
        if arm == "ss"
        else ConditionalLanguageTrainerV2(model)
    )
    directory = Path(tempfile.mkdtemp(prefix="n174-smoke-"))
    report = {"milestones": []}
    report["milestones"].extend(evaluate(model, items, 0, f"smoke_{arm}"))
    report["restores"] = {
        "zero": save_verify(trainer, train[0].prompt_bytes, "zero", directory, {"epoch": 0})
    }
    batches, digest = epoch_batches(train, seed, 1)
    losses = []
    for indices in batches:
        losses.append(trainer.update([example(train[position]) for position in indices]))
    report["milestones"].extend(evaluate(model, items, 1, f"smoke_{arm}"))
    report["controls"] = controls(
        model,
        chosen,
        train,
        [row for row in report["milestones"] if row["epoch"] == 1 and row["kind"] == "heldout"],
        rank_by_id,
    )
    report["summary"] = summarize(report["milestones"], report["controls"], 1)
    report["restores"]["final"] = save_verify(
        trainer, train[0].prompt_bytes, "final", directory, {"epoch": 1, "batch": len(batches)}
    )
    print(
        json.dumps(
            {
                "arm": arm,
                "epoch_order_digest": digest,
                "loss_nats_mean": sum(item["loss_nats"] for item in losses) / len(losses),
                "controls": len(report["controls"]),
                "summary": report["summary"],
                "restores": {
                    name: {"steps": value["steps"], "sha256": value["sha256"][:16]}
                    for name, value in report["restores"].items()
                },
                "checkpoint_dir": str(directory),
            },
            ensure_ascii=False,
        )
    )
    return 0


def verify(path: str) -> int:
    torch.set_num_threads(1)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") == ConditionalLanguageRolloutTrainer.FORMAT:
        trainer, state, cursor = ConditionalLanguageRolloutTrainer.from_checkpoint(payload)
    elif payload.get("format") == ConditionalLanguageTrainerV2.FORMAT:
        trainer, state, cursor = ConditionalLanguageTrainerV2.from_checkpoint(payload)
    else:
        raise ValueError("unsupported scale checkpoint format")
    signatures = checkpoint_signatures(trainer, state)
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
    reports = {}
    complete = True
    for arm in ARMS:
        for seed in SEEDS:
            path = run_report(arm, seed)
            if not path.is_file():
                complete = False
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            reports[f"{arm}_seed{seed}"] = report
            complete = complete and report.get("status") == "complete"
            complete = complete and bool(report.get("source_unchanged_after_run"))
    rollout = {arm: [] for arm in ARMS}
    hits = {arm: 0 for arm in ARMS}
    qualified = {arm: 0 for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            report = reports.get(f"{arm}_seed{seed}")
            if not report or report.get("status") != "complete":
                continue
            summary = report["summary"]
            rollout[arm].append(summary["train_free_match_median"])
            hits[arm] += sum(
                entry["own_match"] >= OWN_GATE
                and entry["own_match"] >= entry["control_a_match"] + MARGIN
                and entry["own_match"] >= entry["control_b_match"] + MARGIN
                for entry in report["controls"]
            )
            qualified[arm] += sum(entry["own_match"] >= OWN_GATE for entry in report["controls"])
    r_arm = {arm: sum(value >= ROLLOUT_GATE for value in rollout[arm]) >= 2 for arm in ARMS}
    r_any = any(r_arm.values())
    gate_arm = [
        arm
        for arm in ARMS
        if hits[arm] >= BINDING_HITS and max(hits.values()) - min(hits.values()) >= HIT_LEAD
    ]
    gate = bool(gate_arm) and complete
    if gate:
        status = "PASS_BINDING"
    elif r_any:
        status = "FAIL_NO_GENERALIZATION"
    else:
        status = "INCONCLUSIVE_ROLLOUT"
    if not complete:
        status = "PARTIAL"
    aggregate = {
        "schema": "taiji-n1-plan174-scale-aggregate-v1",
        "status": status,
        "all_runs_complete": complete,
        "rollout_medians": rollout,
        "rollout_gate_by_arm": r_arm,
        "binding_hits": hits,
        "binding_qualified": qualified,
        "binding_gate_arms": gate_arm,
        "hit_lead": max(hits.values()) - min(hits.values()),
        "criteria": {
            "rollout_gate": ROLLOUT_GATE,
            "own_gate": OWN_GATE,
            "margin": MARGIN,
            "binding_hits": BINDING_HITS,
            "hit_lead": HIT_LEAD,
        },
        "runs": {
            name: {
                "status": report.get("status"),
                "cpu_seconds": report.get("cpu_seconds"),
                "wall_seconds": report.get("wall_seconds"),
                "train_free_match_median": (
                    report.get("summary", {}).get("train_free_match_median")
                ),
                "heldout_free_match_mean": (
                    report.get("summary", {}).get("heldout_free_match_mean")
                ),
                "binding_hits": sum(
                    entry["own_match"] >= OWN_GATE
                    and entry["own_match"] >= entry["control_a_match"] + MARGIN
                    and entry["own_match"] >= entry["control_b_match"] + MARGIN
                    for entry in report.get("controls", [])
                ),
                "error": report.get("error"),
            }
            for name, report in reports.items()
        },
        "dev_queries": 0,
        "can_promote": False,
        "total_cpu_seconds": sum(report.get("cpu_seconds") or 0 for report in reports.values()),
        "total_wall_seconds_max": max(
            [report.get("wall_seconds") or 0 for report in reports.values()] or [0]
        ),
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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--verify")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        if args.arm is None or args.seed is None:
            raise ValueError("smoke needs both --arm and --seed")
        return smoke(args.arm, args.seed)
    if args.pilot:
        if args.arm is None or args.seed is None:
            raise ValueError("pilot needs both --arm and --seed")
        return pilot(args.arm, args.seed)
    if args.verify:
        return verify(args.verify)
    if args.aggregate:
        return aggregate()
    if args.all:
        return spawn_all()
    if args.arm is None or args.seed is None:
        raise ValueError("a single run needs both --arm and --seed")
    return run_single(args.arm, args.seed, args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
