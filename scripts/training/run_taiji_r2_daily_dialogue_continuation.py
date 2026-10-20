"""Continue the matched OASST2 dialogue arms from their retry2 final bundles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.language_alignment import (
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    _atomic_save,
)

BASE_PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md"
BASE_PLAN_SHA256 = "d921465ddb99e325d6623eee787e8e55cf17555575e8c3dcf605c847cf2de655"
PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-CONTINUATION_20261021.md"
PLAN_SHA256 = "13e5ec9fd87fe08397a54fdc40ee174599276bbff6e4d443334544facf822143"
BUILDER = ROOT / "scripts/data_prep/build_taiji_r2_daily_dialogue_corpus.py"
RUNNER = Path(__file__).resolve()
RAW_SOURCE = ROOT / "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json"
PARENT_CHECKPOINT = ROOT / "reports/r2_h3_7b_validation/dev_byte_aligned/checkpoint.pt"
PARENT_REPORT = ROOT / "reports/r2_h3_7b_validation/dev_byte_aligned/report.json"
RETRY2_RESULT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_result.json"
RETRY2_GENERATIONS = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_generations.json"
RETRY2_ATTRIBUTION = ROOT / "reports/taiji_r2_daily_dialogue_20261021_attribution.json"
REPORT_ROOT = ROOT / "reports"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation"
READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_readiness.json"
ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_attempt.json"
PROGRESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_progress.jsonl"
GENERATIONS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_generations.json"
REVIEW = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_generation_review.md"
RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_continuation_result.json"

ARMS = ("dialogue_context", "single_turn")
ADDITIONAL_EPOCHS = 6
CPU_LIMIT_SECONDS = 30_000.0
GENERATION_RESERVE_SECONDS = 6_000.0
USED_FAMILY_IDS = None
EXPECTED_CONTINUATION_FAMILIES = 70
EXPECTED_SPLITS = {"train": 525, "dev": 62, "final": 123, "retention": 0}
EMAIL_OR_PHONE = re.compile(r"(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?\d[\d ()-]{7,}\d))")
GENERATION_ENDPOINT_ROW_FIELDS = {
    "parent_dialogue_context": "parent_with_history",
    "parent_single_turn": "parent_single_turn",
    "old_dialogue_context": "old_dialogue_context",
    "old_single_turn": "old_single_turn",
    "continued_dialogue_context": "continued_dialogue_context",
    "continued_single_turn": "continued_single_turn",
}
GENERATION_PROGRESS_FIELDS = (
    "generated_text",
    "generated_bytes_length",
    "raw_generated_sha256",
    "response_boundary_present",
    "pii_pattern_redacted",
)

RETRY2_CHECKPOINTS = {
    "parent_dialogue_context": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2/dialogue_context_zero.pt",
        "sha256": "b1227ceff57d80cf04105a8231bd6dda55006c8fd92dcb5418bcb322750be760",
        "arm": "dialogue_context",
    },
    "parent_single_turn": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2/single_turn_zero.pt",
        "sha256": "438fd37088038477b1983d27d827debe42af539ebf5eeb780683070da860fbdb",
        "arm": "single_turn",
    },
    "old_dialogue_context": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2/dialogue_context_final.pt",
        "sha256": "c264eca87f1f9199b488d09afa1288a9f9ce65f63135ed7443b7278bb180428b",
        "arm": "dialogue_context",
    },
    "old_single_turn": {
        "path": ROOT
        / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2/single_turn_final.pt",
        "sha256": "2b54d88917265a92bae6a5b8578ac15b699ff1f39fe19591b0390860d3a270b6",
        "arm": "single_turn",
    },
}


class Refuse(RuntimeError):
    """The previous dialogue endpoint or continuation contract is invalid."""


class CpuBudget:
    def __init__(self, limit: float) -> None:
        self.limit = float(limit)
        self.started = time.process_time()
        self.child_seconds = 0.0

    def used(self) -> float:
        return time.process_time() - self.started + self.child_seconds

    def guard(self, reserve: float = 0.0) -> None:
        used = self.used()
        if used + reserve > self.limit:
            raise TimeoutError(
                f"R2 dialogue continuation CPU exceeded: {used:.3f}+{reserve:.3f}/{self.limit:.3f}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2 continuation report is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def write_jsonl(stream, payload: dict[str, Any]) -> None:
    json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Refuse(f"expected a JSON object: {path}")
    return value


def _arm_corpus(base: LanguageEpisodeCorpus, arm: str) -> LanguageEpisodeCorpus:
    if arm == "dialogue_context":
        return base
    if arm != "single_turn":
        raise ValueError(f"unknown R2 dialogue arm: {arm}")
    return LanguageEpisodeCorpus(
        tuple(replace(episode, history=()) for episode in base.episodes), base.source_files
    )


def _load_inputs() -> tuple[
    LanguageEpisodeCorpus, dict[str, Any], dict[str, Any], dict[str, str], set[str]
]:
    for path, expected, label in (
        (BASE_PLAN, BASE_PLAN_SHA256, "base plan"),
        (PLAN, PLAN_SHA256, "continuation plan"),
        (
            RAW_SOURCE,
            "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751",
            "OASST2 raw archive",
        ),
        (
            EPISODES,
            "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d",
            "OASST2 episodes",
        ),
        (
            MANIFEST,
            "3f5b453aaab88576afd458f7b81bdcff1f1bfbf1ffd94890cb72602270211472",
            "OASST2 manifest",
        ),
        (
            RETRY2_RESULT,
            "e939e4c91219b709e0a3799db13610fc9cb1407916d3a9240b4af451f00411f0",
            "retry2 result",
        ),
        (
            RETRY2_GENERATIONS,
            "32b97f4a66d562a607967eb17ac0fbb452307cd429806f551ff3c42acbc0d02f",
            "retry2 generations",
        ),
        (
            RETRY2_ATTRIBUTION,
            "8ae2ee97776d65aaa83cc79b71429f1b119dd671064a645d9d90fceed7cf87d2",
            "retry2 attribution",
        ),
    ):
        if sha256(path) != expected:
            raise Refuse(f"R2 continuation {label} SHA-256 changed")

    base_corpus = LanguageEpisodeCorpus.from_jsonl([EPISODES.resolve()])
    manifest = _read_json(MANIFEST)
    if base_corpus.digest != manifest.get("corpus_digest") or base_corpus.sample_counts != {
        "train": 525,
        "dev": 62,
        "final": 123,
        "retention": 0,
    }:
        raise Refuse("frozen OASST2 corpus digest or split counts differ")
    prior_result = _read_json(RETRY2_RESULT)
    prior_generations = _read_json(RETRY2_GENERATIONS)
    if (
        prior_result.get("status") != "complete"
        or prior_result.get("can_promote") is not False
        or prior_result.get("training_arms") != list(ARMS)
        or prior_result.get("plan_local_response_observations")
        != {"dialogue_context": 226_418, "single_turn": 226_418}
        or prior_generations.get("status") != "complete"
        or len(prior_generations.get("records", [])) != 12
    ):
        raise Refuse("retry2 source result is not the expected complete dialogue batch")
    used_families = {str(row["family_id"]) for row in prior_generations["records"]}
    final_families = {
        episode.family_id
        for episode in base_corpus.for_split("final")
        if episode.family_id not in used_families
    }
    if len(used_families) != 12 or len(final_families) != EXPECTED_CONTINUATION_FAMILIES:
        raise Refuse("remaining final family count differs from the frozen continuation plan")

    source_hashes = {
        str(path.resolve()): sha256(path)
        for path in (
            BASE_PLAN,
            PLAN,
            RAW_SOURCE,
            EPISODES,
            MANIFEST,
            RETRY2_RESULT,
            RETRY2_GENERATIONS,
            RETRY2_ATTRIBUTION,
            PARENT_CHECKPOINT,
            PARENT_REPORT,
        )
    }
    for key, value in RETRY2_CHECKPOINTS.items():
        if sha256(value["path"]) != value["sha256"]:
            raise Refuse(f"retry2 checkpoint changed: {key}")
        source_hashes[str(value["path"].resolve())] = value["sha256"]
    return base_corpus, manifest, prior_result, source_hashes, used_families


def _independent_restore(checkpoint: Path, arm: str, budget: CpuBudget) -> dict[str, Any]:
    budget.guard(reserve=15)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restore-worker",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--arm",
            arm,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=max(30.0, min(14_400.0, budget.limit - budget.used() - 1.0)),
        check=False,
    )
    if process.returncode != 0:
        raise Refuse(f"fresh-process continuation restore failed: {process.stderr[-2_000:]}")
    report = json.loads(process.stdout)
    if report.get("complete") is not True or report.get("equal") is not True:
        raise Refuse(f"fresh-process continuation restore mismatch: {checkpoint}")
    budget.child_seconds += float(report.get("cpu_seconds", 0.0))
    budget.guard()
    return {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint), **report}


def _select_remaining_final(
    base_corpus: LanguageEpisodeCorpus, used_families: set[str]
) -> tuple[LanguageEpisode, ...]:
    grouped: dict[str, list[LanguageEpisode]] = defaultdict(list)
    for episode in base_corpus.for_split("final"):
        if episode.family_id not in used_families:
            grouped[episode.family_id].append(episode)

    def stable_key(family_id: str) -> str:
        return hashlib.sha256(
            f"r2-dialogue-continuation-final-v1\0{family_id}".encode()
        ).hexdigest()

    selected = tuple(
        min(grouped[family_id], key=lambda episode: episode.episode_id)
        for family_id in sorted(grouped, key=stable_key)
    )
    if len(selected) != EXPECTED_CONTINUATION_FAMILIES:
        raise Refuse("continuation final sample is not exactly 70 unused conversation trees")
    if len({episode.family_id for episode in selected}) != len(selected):
        raise Refuse("continuation final sample repeats a conversation tree")
    return selected


def _generate_readonly(
    trainer: LanguageAlignmentTrainer, episode: LanguageEpisode
) -> dict[str, Any]:
    snapshot = trainer.model.checkpoint()
    try:
        raw = trainer._generate(episode)
        text, utf8_valid, no_replacement, boundary, stop = trainer._generated_text(raw)
    finally:
        trainer.model.restore(snapshot)
        trainer._configure_developmental_mode()
        trainer._configure_response_start_readout()
        trainer._configure_response_phase_readout()
    redacted = bool(EMAIL_OR_PHONE.search(text))
    safe_text = EMAIL_OR_PHONE.sub("[已遮蔽]", text)
    compact = "".join(safe_text.split())
    grams = [compact[i : i + 4] for i in range(max(0, len(compact) - 3))]
    repeat_rate = 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)
    return {
        "generated_text": safe_text,
        "raw_generated_sha256": hashlib.sha256(raw).hexdigest(),
        "pii_pattern_redacted": redacted,
        "generated_bytes_hex": None if redacted else raw.hex(),
        "generated_bytes_length": len(raw),
        "utf8_valid": utf8_valid,
        "no_replacement": no_replacement,
        "response_boundary_present": boundary,
        "stop_reason": stop,
        "repeated_4gram_rate": round(repeat_rate, 6),
    }


def _generation_output_for_endpoint(row: dict[str, Any], endpoint: str) -> dict[str, Any]:
    return row[GENERATION_ENDPOINT_ROW_FIELDS[endpoint]]


def _generation_progress_outputs(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        endpoint: {
            field: _generation_output_for_endpoint(row, endpoint)[field]
            for field in GENERATION_PROGRESS_FIELDS
        }
        for endpoint in GENERATION_ENDPOINT_ROW_FIELDS
    }


def _summary(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(outputs)
    texts = [str(row["generated_text"]) for row in outputs]
    return {
        "generation_count": count,
        "utf8_valid_rate": sum(bool(row["utf8_valid"]) for row in outputs) / count,
        "no_replacement_rate": sum(bool(row["no_replacement"]) for row in outputs) / count,
        "boundary_rate": sum(bool(row["response_boundary_present"]) for row in outputs) / count,
        "mean_output_bytes": sum(int(row["generated_bytes_length"]) for row in outputs) / count,
        "unique_output_count": len(set(texts)),
        "duplicate_output_rate": 1.0 - len(set(texts)) / count,
        "mean_repeated_4gram_rate": sum(float(row["repeated_4gram_rate"]) for row in outputs)
        / count,
    }


def _review_markdown(records: list[dict[str, Any]], *, sample_count: int = 12) -> str:
    review = records[:sample_count]
    lines = [
        "# OASST2剩余final树续训生成样例",
        "",
        "不含参考回复，不做逐字匹配。以下为续训前父档、2-epoch旧候选/对照及8-epoch新候选/对照的并列自由生成样例。",
        "",
    ]
    for row in review:
        lines.extend((f"## 样本 {row['index']}（history={row['history_turn_count']}）", ""))
        for index, (user, assistant) in enumerate(row["history"], 1):
            lines.extend((f"历史用户{index}：{user}", f"历史助手{index}：{assistant}", ""))
        lines.extend((f"当前用户：{row['current_user_turn']}", ""))
        for label, key in (
            ("parent（含历史）", "parent_with_history"),
            ("parent（单轮）", "parent_single_turn"),
            ("2-epoch多轮候选", "old_dialogue_context"),
            ("2-epoch单轮对照", "old_single_turn"),
            ("8-epoch多轮候选", "continued_dialogue_context"),
            ("8-epoch单轮对照", "continued_single_turn"),
        ):
            output = str(row[key]["generated_text"]).replace("```", "` ` `")
            lines.extend(
                (
                    f"**{label}**（{row[key]['stop_reason']}，{row[key]['generated_bytes_length']}字节）",
                    "",
                    "```text",
                    output,
                    "```",
                    "",
                )
            )
    return "\n".join(lines) + "\n"


def _continuation_review_markdown(records: list[dict[str, Any]], *, sample_count: int = 12) -> str:
    lines = [
        "# OASST2未消费final对话树续训生成样例",
        "",
        "不含参考回复，不做逐字匹配。样例来自前批未查询的70棵树；比较父档、2-epoch终点和8-epoch续训终点。",
        "",
    ]
    for row in records[:sample_count]:
        lines.extend((f"## 样本 {row['index']}（history={row['history_turn_count']}）", ""))
        for index, (user, assistant) in enumerate(row["history"], 1):
            lines.extend((f"历史用户{index}：{user}", f"历史助手{index}：{assistant}", ""))
        lines.extend((f"当前用户：{row['current_user_turn']}", ""))
        for label, key in (
            ("父档含历史", "parent_with_history"),
            ("父档单轮", "parent_single_turn"),
            ("2 epoch 多轮", "old_dialogue_context"),
            ("2 epoch 单轮", "old_single_turn"),
            ("8 epoch 多轮", "continued_dialogue_context"),
            ("8 epoch 单轮", "continued_single_turn"),
        ):
            output = str(row[key]["generated_text"]).replace("```", "` ` `")
            lines.extend(
                (
                    f"**{label}**（{row[key]['stop_reason']}，{row[key]['generated_bytes_length']}字节）",
                    "",
                    "```text",
                    output,
                    "```",
                    "",
                )
            )
    return "\n".join(lines) + "\n"


def write_once_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"R2 continuation text output is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()


def execute() -> dict[str, Any]:
    started = time.process_time()
    outputs = (READINESS, ATTEMPT, PROGRESS, GENERATIONS, REVIEW, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("R2 continuation reports are write-once")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"R2 continuation checkpoint root already exists: {CHECKPOINT_ROOT}")

    base_corpus, manifest, prior_result, source_hashes, used_families = _load_inputs()
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    setup_cpu = time.process_time() - started
    budget = CpuBudget(CPU_LIMIT_SECONDS - setup_cpu)
    if budget.limit <= 0:
        raise TimeoutError("R2 continuation CPU budget exhausted during input validation")

    readiness = {
        "schema": "taiji-r2-daily-dialogue-continuation-readiness-v1",
        "status": "ready",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "corpus_digest": base_corpus.digest,
        "split_episode_counts": base_corpus.sample_counts,
        "previous_used_final_trees": len(used_families),
        "remaining_final_trees": EXPECTED_CONTINUATION_FAMILIES,
        "parent_final_checkpoint_sha256": {
            arm: RETRY2_CHECKPOINTS[f"old_{arm}"]["sha256"] for arm in ARMS
        },
        "additional_epochs_per_arm": ADDITIONAL_EPOCHS,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "parameter_updates": 0,
        "generation_queries": 0,
        "can_promote": False,
    }
    write_new_json(READINESS, readiness)
    attempt = {
        "schema": "taiji-r2-daily-dialogue-continuation-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_hashes": dict(sorted(source_hashes.items())),
        "previous_retry2_result_sha256": sha256(RETRY2_RESULT),
        "previous_final_checkpoints": {
            arm: RETRY2_CHECKPOINTS[f"old_{arm}"]["sha256"] for arm in ARMS
        },
        "additional_epochs_per_arm": ADDITIONAL_EPOCHS,
        "train_episodes_per_epoch": EXPECTED_SPLITS["train"],
        "remaining_final_tree_count": EXPECTED_CONTINUATION_FAMILIES,
        "generation_outputs_planned": EXPECTED_CONTINUATION_FAMILIES * 6,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "formal_parameter_updates": 0,
        "formal_generation_queries": 0,
        "can_promote": False,
    }
    write_new_json(ATTEMPT, attempt)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    zero_trainers: dict[str, LanguageAlignmentTrainer] = {}
    old_trainers: dict[str, LanguageAlignmentTrainer] = {}
    trainers: dict[str, LanguageAlignmentTrainer] = {}
    zero_records: dict[str, dict[str, Any]] = {}
    final_records: dict[str, dict[str, Any]] = {}
    episode_counts = Counter()
    observation_counts = Counter()
    generation_count = 0
    active_arm: str | None = None
    active_trainer: LanguageAlignmentTrainer | None = None
    try:
        arm_corpora = {arm: _arm_corpus(base_corpus, arm) for arm in ARMS}
        retry2_parent_final_restores = {
            arm: _independent_restore(RETRY2_CHECKPOINTS[f"old_{arm}"]["path"], arm, budget)
            for arm in ARMS
        }
        for arm, receipt in retry2_parent_final_restores.items():
            write_jsonl(progress, {"event": "retry2_parent_final_restore", "arm": arm, **receipt})
        progress.flush()
        # Load the already trained 2-epoch endpoints and parent zero endpoints.
        for key, spec in RETRY2_CHECKPOINTS.items():
            if sha256(spec["path"]) != spec["sha256"]:
                raise Refuse(f"retry2 checkpoint changed before continuation: {key}")
            payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
            restored = LanguageAlignmentTrainer.from_checkpoint(payload, arm_corpora[spec["arm"]])
            if key.startswith("parent_"):
                zero_trainers[spec["arm"]] = restored
            else:
                old_trainers[spec["arm"]] = restored

        for arm in ARMS:
            budget.guard(reserve=1_000)
            old_path = RETRY2_CHECKPOINTS[f"old_{arm}"]["path"]
            old_sha = RETRY2_CHECKPOINTS[f"old_{arm}"]["sha256"]
            if sha256(old_path) != old_sha:
                raise Refuse(f"retry2 final parent changed before continuation: {arm}")
            start = old_trainers[arm]
            zero_path = CHECKPOINT_ROOT / f"{arm}_zero.pt"
            if zero_path.exists():
                raise FileExistsError(zero_path)
            payload = start.checkpoint()
            _atomic_save(payload, zero_path)
            zero_sha = sha256(zero_path)
            restore = _independent_restore(zero_path, arm, budget)
            if restore.get("checkpoint_digest") != payload["checkpoint_digest"]:
                raise Refuse(f"continuation {arm} zero bundle mismatch")
            trainers[arm] = copy.deepcopy(start)
            if trainers[arm].checkpoint()["checkpoint_digest"] != payload["checkpoint_digest"]:
                raise Refuse(f"continuation {arm} in-memory zero copy differs from saved bundle")
            zero_records[arm] = {
                "path": str(zero_path.resolve()),
                "sha256": zero_sha,
                "checkpoint_digest": payload["checkpoint_digest"],
                "corpus_digest": arm_corpora[arm].digest,
                "start_from_retry2_final_sha256": old_sha,
                "fresh_process_restore": restore,
            }
        if len(zero_records) != 2:
            raise Refuse("continuation zero bundle preflight incomplete")

        for arm in ARMS:
            active_arm = arm
            active_trainer = trainers[arm]
            budget.guard(reserve=GENERATION_RESERVE_SECONDS)
            start_global_step = active_trainer.global_step
            start_episode_count = active_trainer.episode_count
            local_episode_index = 0

            def on_progress(record: dict[str, Any]) -> None:
                nonlocal local_episode_index
                local_episode_index += 1
                episode_counts[arm] += 1
                observation_counts[arm] += int(record["observations"])
                write_jsonl(
                    progress,
                    {
                        "event": "continuation_training_episode",
                        "arm": arm,
                        "epoch": int(record["epoch"]),
                        "episode_index": local_episode_index,
                        "episode_id": str(record["episode_id"]),
                        "family_id": str(record["family_id"]),
                        "target_bytes": int(record["target_bytes"]),
                        "prefix_bytes": int(record["prefix_bytes"]),
                        "observations": int(record["observations"]),
                        "response_mean_surprise": float(record["response_mean_surprise"]),
                        "cumulative_continuation_observations": int(observation_counts[arm]),
                    },
                )
                if local_episode_index % 16 == 0:
                    progress.flush()
                if local_episode_index % 64 == 0:
                    print(
                        f"r2-dialogue-continuation arm={arm} "
                        f"episode={local_episode_index}/{ADDITIONAL_EPOCHS * EXPECTED_SPLITS['train']} "
                        f"observations={observation_counts[arm]} cpu={budget.used():.1f}",
                        flush=True,
                    )
                budget.guard(reserve=GENERATION_RESERVE_SECONDS)

            trained = active_trainer.train(epochs=ADDITIONAL_EPOCHS, progress=on_progress)
            if (
                trained.get("status") != "completed"
                or trained.get("episodes") != ADDITIONAL_EPOCHS * EXPECTED_SPLITS["train"]
            ):
                raise Refuse(f"continuation arm did not complete all frozen episodes: {arm}")
            if active_trainer.global_step - start_global_step != observation_counts[arm]:
                raise Refuse(f"continuation response-observation delta differs: {arm}")
            if (
                active_trainer.episode_count - start_episode_count
                != ADDITIONAL_EPOCHS * EXPECTED_SPLITS["train"]
            ):
                raise Refuse(f"continuation episode count differs: {arm}")

            final_path = CHECKPOINT_ROOT / f"{arm}_final.pt"
            if final_path.exists():
                raise FileExistsError(final_path)
            final_payload = active_trainer.checkpoint()
            _atomic_save(final_payload, final_path)
            final_sha = sha256(final_path)
            final_restore = _independent_restore(final_path, arm, budget)
            if final_restore.get("checkpoint_digest") != final_payload["checkpoint_digest"]:
                raise Refuse(f"continuation final full-bundle restore mismatch: {arm}")
            final_records[arm] = {
                "path": str(final_path.resolve()),
                "sha256": final_sha,
                "checkpoint_digest": final_payload["checkpoint_digest"],
                "corpus_digest": arm_corpora[arm].digest,
                "start_from_retry2_final_sha256": RETRY2_CHECKPOINTS[f"old_{arm}"]["sha256"],
                "parent_total_global_step_before_continuation": start_global_step,
                "final_global_step": active_trainer.global_step,
                "additional_response_observations": int(observation_counts[arm]),
                "additional_training_episodes": int(episode_counts[arm]),
                "additional_epochs": ADDITIONAL_EPOCHS,
                "fresh_process_restore": final_restore,
            }
            active_trainer = None
            active_arm = None
            progress.flush()

        remaining_final = _select_remaining_final(base_corpus, used_families)
        if len(remaining_final) != EXPECTED_CONTINUATION_FAMILIES:
            raise Refuse("unconsumed final family count differs after continuation")

        generation_trainers: dict[str, LanguageAlignmentTrainer] = {}
        for arm in ARMS:
            generation_trainers[f"parent_{arm}"] = zero_trainers[arm]
            generation_trainers[f"old_{arm}"] = old_trainers[arm]
            generation_trainers[f"continued_{arm}"] = trainers[arm]
        before_digests = {
            name: trainer.checkpoint()["checkpoint_digest"]
            for name, trainer in generation_trainers.items()
        }

        generation_rows = []
        for index, episode in enumerate(remaining_final, 1):
            budget.guard()
            flat = replace(episode, history=())
            row = {
                "index": index,
                "episode_id": episode.episode_id,
                "family_id": episode.family_id,
                "history_turn_count": len(episode.history),
                "history": [[user, assistant] for user, assistant in episode.history],
                "current_user_turn": episode.user_input,
                "parent_with_history": _generate_readonly(
                    generation_trainers["parent_dialogue_context"], episode
                ),
                "parent_single_turn": _generate_readonly(
                    generation_trainers["parent_single_turn"], flat
                ),
                "old_dialogue_context": _generate_readonly(
                    generation_trainers["old_dialogue_context"], episode
                ),
                "old_single_turn": _generate_readonly(generation_trainers["old_single_turn"], flat),
                "continued_dialogue_context": _generate_readonly(
                    generation_trainers["continued_dialogue_context"], episode
                ),
                "continued_single_turn": _generate_readonly(
                    generation_trainers["continued_single_turn"], flat
                ),
                "reference_answer_included": False,
                "exact_response_scored": False,
            }
            generation_rows.append(row)
            generation_count += len(generation_trainers)
            write_jsonl(
                progress,
                {
                    "event": "continuation_generation_review",
                    "index": index,
                    "episode_id": episode.episode_id,
                    "family_id": episode.family_id,
                    "history_turn_count": len(episode.history),
                    "outputs": _generation_progress_outputs(row),
                    "reference_answer_included": False,
                },
            )
            progress.flush()
            if index % 10 == 0:
                print(
                    f"r2-dialogue-continuation generation={index}/{len(remaining_final)} cpu={budget.used():.1f}",
                    flush=True,
                )

        after_digests = {
            name: trainer.checkpoint()["checkpoint_digest"]
            for name, trainer in generation_trainers.items()
        }
        if before_digests != after_digests:
            raise Refuse("R2 continuation generation mutated a parent/old/new checkpoint")
        for path, expected in source_hashes.items():
            if not Path(path).is_file() or sha256(Path(path)) != expected:
                raise Refuse(f"R2 continuation source changed during generation: {path}")

        all_outputs = {
            name: [
                {
                    "episode_id": row["episode_id"],
                    "output": _generation_output_for_endpoint(row, name),
                }
                for row in generation_rows
            ]
            for name in generation_trainers
        }
        generation_metrics = {name: _summary(rows) for name, rows in all_outputs.items()}
        generation_report = {
            "schema": "taiji-r2-daily-dialogue-continuation-generations-v1",
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "corpus_digest": base_corpus.digest,
            "previously_consumed_tree_families": len(used_families),
            "new_final_tree_families": len({row["family_id"] for row in generation_rows}),
            "new_final_episodes": len(generation_rows),
            "input_history_strata": dict(
                Counter(
                    "with_history" if row["history_turn_count"] else "without_history"
                    for row in generation_rows
                )
            ),
            "generation_configuration": {
                "max_generation_bytes": 512,
                "constrained_decode": True,
                "sampling": False,
                "reference_answer_included": False,
                "exact_response_scored": False,
            },
            "generation_outputs": generation_count,
            "metrics": generation_metrics,
            "records": generation_rows,
            "can_promote": False,
        }
        write_new_json(GENERATIONS, generation_report)
        write_once_text(REVIEW, _continuation_review_markdown(generation_rows))

        cumulative_cpu = budget.used() + setup_cpu
        if cumulative_cpu > CPU_LIMIT_SECONDS:
            raise TimeoutError("R2 continuation exceeded its frozen CPU limit")
        result = {
            "schema": "taiji-r2-daily-dialogue-continuation-result-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "attempt_sha256": sha256(ATTEMPT),
            "progress_sha256": sha256(PROGRESS),
            "generations_sha256": sha256(GENERATIONS),
            "review_sha256": sha256(REVIEW),
            "source_hashes": dict(sorted(source_hashes.items())),
            "source_hashes_unchanged": True,
            "previous_retry2_result_sha256": sha256(RETRY2_RESULT),
            "retry2_parent_final_fresh_process_restores": retry2_parent_final_restores,
            "new_final_tree_families": len({row["family_id"] for row in generation_rows}),
            "new_final_episodes": len(generation_rows),
            "continuation_arms": list(ARMS),
            "additional_epochs_per_arm": ADDITIONAL_EPOCHS,
            "train_episodes_per_epoch": EXPECTED_SPLITS["train"],
            "additional_training_episodes": {arm: int(episode_counts[arm]) for arm in ARMS},
            "additional_response_observations": {arm: int(observation_counts[arm]) for arm in ARMS},
            "continuation_zero_checkpoints": zero_records,
            "continuation_final_checkpoints": final_records,
            "generation_output_count": generation_count,
            "generation_outputs_per_endpoint": len(generation_rows),
            "reference_answer_scoring_queries": 0,
            "dev_queries": 0,
            "generation_metrics": generation_metrics,
            "generation_review_path": str(REVIEW.resolve()),
            "cpu_seconds": round(cumulative_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "can_promote": False,
        }
        write_new_json(RESULT, result)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        partial = None
        if active_arm is not None and active_trainer is not None:
            try:
                partial_path = CHECKPOINT_ROOT / f"{active_arm}_partial.pt"
                if not partial_path.exists():
                    payload = active_trainer.checkpoint()
                    _atomic_save(payload, partial_path)
                    partial = {
                        "path": str(partial_path.resolve()),
                        "sha256": sha256(partial_path),
                        "checkpoint_digest": payload["checkpoint_digest"],
                        "global_step": active_trainer.global_step,
                    }
            except Exception as save_error:
                partial = {"save_error": f"{type(save_error).__name__}: {save_error}"}
        failure = {
            "schema": "taiji-r2-daily-dialogue-continuation-result-v1",
            "status": "partial" if sum(observation_counts.values()) else "failed_before_training",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "source_hashes": dict(sorted(source_hashes.items()))
            if "source_hashes" in locals()
            else {},
            "completed_training_episodes": dict(episode_counts),
            "completed_response_observations": dict(observation_counts),
            "completed_generation_outputs": generation_count,
            "retry2_parent_final_fresh_process_restores": locals().get(
                "retry2_parent_final_restores", {}
            ),
            "active_arm": active_arm,
            "partial_checkpoint": partial,
            "cpu_seconds": round(budget.used() + setup_cpu if "budget" in locals() else 0.0, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "errors": [error],
            "can_promote": False,
        }
        try:
            if not RESULT.exists():
                write_new_json(RESULT, failure)
        except Exception:
            pass
        raise
    finally:
        try:
            progress.flush()
            progress.close()
        except Exception:
            pass


def _restore_worker(checkpoint: Path, arm: str) -> dict[str, Any]:
    started = time.process_time()
    base = LanguageEpisodeCorpus.from_jsonl([EPISODES.resolve()])
    corpus = _arm_corpus(base, arm)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    return {
        "complete": True,
        "equal": True,
        "arm": arm,
        "checkpoint_digest": payload["checkpoint_digest"],
        "corpus_digest": corpus.digest,
        "global_step": trainer.global_step,
        "episode_count": trainer.episode_count,
        "trainer_history_count": len(trainer.history),
        "cpu_seconds": round(time.process_time() - started, 6),
    }


def main() -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.restore_worker:
        if args.checkpoint is None or args.arm is None:
            raise SystemExit("--restore-worker requires --checkpoint and --arm")
        print(
            json.dumps(
                _restore_worker(args.checkpoint, args.arm), ensure_ascii=False, sort_keys=True
            )
        )
        return 0
    try:
        report = execute()
    except Exception as exc:
        print(f"R2 continuation refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "additional_response_observations": report["additional_response_observations"],
                "generation_output_count": report["generation_output_count"],
                "cpu_seconds": report["cpu_seconds"],
                "can_promote": report["can_promote"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
