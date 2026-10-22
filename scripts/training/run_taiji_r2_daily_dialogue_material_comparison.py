"""Continue the native Taiji language model on OASST2 Chinese dialogue trees.

The paired conditions use the same source targets and training examples. The
candidate receives up to three clean history turns; its control receives only
the current user turn. Final evaluation emits text only and never reads or
scores the held-out reference response.
"""

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
from collections import Counter
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
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    _atomic_save,
)
from taiji.model import Taiji
from taiji.response_plan_target import ByteAlignedResponsePlanTargetEncoder

PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-CALIBRATION_20261021.md"
PLAN_SHA256 = "d921465ddb99e325d6623eee787e8e55cf17555575e8c3dcf605c847cf2de655"
RETRY1_ADDENDUM = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-RETRY1_20261021.md"
RETRY1_ADDENDUM_SHA256 = "94321e6db8b3c8cdbadc34d239896ee4d205f7b43bdfca22c136cca046c515b4"
RETRY_ADDENDUM = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-RETRY2_20261021.md"
RETRY_ADDENDUM_SHA256 = "7b3b462d7e99f9c5e8e60e68aee0af7c8bc03bccd0a55d5f56d43a59ae1c4b6b"
BUILDER = ROOT / "scripts/data_prep/build_taiji_r2_daily_dialogue_corpus.py"
RAW_SOURCE = ROOT / "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
DATA_MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json"
PARENT_CHECKPOINT = ROOT / "reports/r2_h3_7b_validation/dev_byte_aligned/checkpoint.pt"
PARENT_REPORT = ROOT / "reports/r2_h3_7b_validation/dev_byte_aligned/report.json"
REPORT_ROOT = ROOT / "reports"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry2"

READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_readiness.json"
ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_attempt.json"
PROGRESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_progress.jsonl"
GENERATIONS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_generations.json"
REVIEW = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_generation_review.md"
RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry2_result.json"
FAILED_ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_attempt.json"
FAILED_RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_result.json"
FAILED_READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_data_readiness.json"
FAILED_ZERO = ROOT / "data/checkpoints/r2_daily_dialogue_oasst2_20261021/dialogue_context_zero.pt"
RETRY1_ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry1_attempt.json"
RETRY1_RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry1_result.json"
RETRY1_READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_20261021_retry1_readiness.json"
RETRY1_ZERO = (
    ROOT / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_retry1/dialogue_context_zero.pt"
)
RETRY1_RESTORE_REAUDIT = (
    REPORT_ROOT / "taiji_r2_daily_dialogue_retry1_restore_reaudit_20261021.json"
)

SOURCE_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
RAW_SOURCE_SHA256 = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
EPISODE_SHA256 = "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d"
MANIFEST_CORPUS_DIGEST = "bec58e46045aaccddb7811fb9b8f755291dd70845ad888bfe438f8678e589af6"
PARENT_CHECKPOINT_SHA256 = "88de777fe05cdafb0a164e7fe7c21dc4417edc7fd3ce325f91885fd547708287"
PARENT_REPORT_SHA256 = "366f06d74ded0560bccf571ebf4177d07901e75b3ee8bc64cb50c478202ada4f"

ARMS = ("dialogue_context", "single_turn")
EPOCHS = 2
CPU_LIMIT_SECONDS = 120_000.0
GENERATION_RESERVE_SECONDS = 20_000.0
GENERATION_SAMPLE_PER_HISTORY = 6
GENERATION_SAMPLE_SALT = "oasst2-generation-review-v1"
EXPECTED_SPLITS = {"train": 525, "dev": 62, "final": 123, "retention": 0}
EMAIL_OR_PHONE = re.compile(r"(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?\d[\d ()-]{7,}\d))")


class Refuse(RuntimeError):
    """A frozen source, corpus, checkpoint, or execution contract failed."""


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
                f"R2 daily-dialogue CPU budget exceeded: {used:.3f}+{reserve:.3f}/{self.limit:.3f}"
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"R2 dialogue output is write-once: {path}")
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
        raise Refuse(f"expected JSON object: {path}")
    return value


def _arm_corpus(base: LanguageEpisodeCorpus, arm: str) -> LanguageEpisodeCorpus:
    if arm == "dialogue_context":
        return base
    if arm != "single_turn":
        raise ValueError(f"unknown R2 dialogue arm: {arm}")
    episodes = tuple(replace(episode, history=()) for episode in base.episodes)
    return LanguageEpisodeCorpus(episodes, base.source_files)


def _load_source_bundle() -> (
    tuple[LanguageEpisodeCorpus, dict[str, Any], dict[str, Any], dict[str, str]]
):
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("R2 dialogue plan SHA-256 differs from the runner binding")
    if sha256(RETRY_ADDENDUM) != RETRY_ADDENDUM_SHA256:
        raise Refuse("R2 dialogue retry addendum SHA-256 differs from the runner binding")
    if sha256(RETRY1_ADDENDUM) != RETRY1_ADDENDUM_SHA256:
        raise Refuse("R2 dialogue retry1 addendum SHA-256 differs from retry2 addendum")
    if sha256(FAILED_ATTEMPT) != "8e60098cdf43da29dc3a9d585f65cb43c9f62b002fe53ed02ace9f3d2b05021e":
        raise Refuse("first R2 dialogue attempt SHA-256 differs from retry addendum")
    if sha256(FAILED_RESULT) != "e1ad7ee49ca9068193e305d4aa48948a36e85a7e64b3c0b75351450702e6b791":
        raise Refuse("first R2 dialogue failure result SHA-256 differs from retry addendum")
    if (
        sha256(FAILED_READINESS)
        != "e514951eef02530fad8dc6d1f28a9ff16544e52dd1786fceffcf375a22dadd0f"
    ):
        raise Refuse("first R2 dialogue readiness SHA-256 differs from retry addendum")
    if sha256(FAILED_ZERO) != "31804e42757737414f5139f94d99b97330dca7f2ccad5cc74c3b63f6261ddac5":
        raise Refuse("preserved first-attempt zero bundle SHA-256 differs from retry addendum")
    if sha256(RETRY1_ATTEMPT) != "a372760b91edf1da19de42072d65ba6a29ed16ebcacf0cde1a60a631ec4eeba8":
        raise Refuse("retry1 attempt SHA-256 differs from retry2 addendum")
    if sha256(RETRY1_RESULT) != "70618266bbc279b40c3a6f153c22d3d2759493b30d54fa264e610f9df0d3893a":
        raise Refuse("retry1 failed result SHA-256 differs from retry2 addendum")
    if (
        sha256(RETRY1_READINESS)
        != "3cd10431c1fa2df1e5641d687870dd9b83e294944d3f570ced162afc0802e33c"
    ):
        raise Refuse("retry1 readiness SHA-256 differs from retry2 addendum")
    if sha256(RETRY1_ZERO) != "a3df6516571072bd4d95bba27dc4708d8f6537a027b022fa77f4cbd0a118b2e9":
        raise Refuse("retry1 zero bundle SHA-256 differs from retry2 addendum")
    if (
        sha256(RETRY1_RESTORE_REAUDIT)
        != "cdce71e1c721b4aa6865647e3b5713d79e7ee09d7a209360479beec718d464c2"
    ):
        raise Refuse("retry1 read-only restore re-audit SHA-256 differs from retry2 addendum")
    failed_attempt = _read_json(FAILED_ATTEMPT)
    failed_result = _read_json(FAILED_RESULT)
    retry1_attempt = _read_json(RETRY1_ATTEMPT)
    retry1_result = _read_json(RETRY1_RESULT)
    retry1_reaudit = _read_json(RETRY1_RESTORE_REAUDIT)
    if (
        failed_attempt.get("plan_sha256") != PLAN_SHA256
        or failed_attempt.get("formal_parameter_updates") != 0
        or failed_attempt.get("formal_generation_queries") != 0
        or failed_result.get("status") != "failed_before_training"
        or failed_result.get("complete") is not False
        or failed_result.get("completed_response_observations") != {}
        or not any("_observations" in error for error in failed_result.get("errors", []))
        or retry1_attempt.get("retry_addendum_sha256") != RETRY1_ADDENDUM_SHA256
        or retry1_attempt.get("formal_parameter_updates") != 0
        or retry1_attempt.get("formal_generation_queries") != 0
        or retry1_result.get("status") != "failed_before_training"
        or retry1_result.get("completed_response_observations") != {}
        or retry1_result.get("completed_generation_outputs") != 0
        or not any("_observations" in error for error in retry1_result.get("errors", []))
        or retry1_reaudit.get("status") != "passed"
        or retry1_reaudit.get("fresh_process_restore") is not True
        or retry1_reaudit.get("formal_parameter_updates") != 0
    ):
        raise Refuse("prior R2 dialogue attempts do not match their frozen pre-training failures")
    if sha256(RAW_SOURCE) != RAW_SOURCE_SHA256:
        raise Refuse("pinned OASST2 source archive SHA-256 changed")
    if sha256(EPISODES) != EPISODE_SHA256:
        raise Refuse("prepared OASST2 episode JSONL SHA-256 changed")
    if sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
        raise Refuse("H3.7B parent checkpoint SHA-256 changed")
    if sha256(PARENT_REPORT) != PARENT_REPORT_SHA256:
        raise Refuse("H3.7B parent report SHA-256 changed")
    manifest = _read_json(DATA_MANIFEST)
    if (
        manifest.get("status") != "ready"
        or manifest.get("source_repository") != "OpenAssistant/oasst2"
        or manifest.get("source_revision") != SOURCE_REVISION
        or manifest.get("source_sha256") != RAW_SOURCE_SHA256
        or manifest.get("episode_jsonl_sha256") != EPISODE_SHA256
        or manifest.get("corpus_digest") != MANIFEST_CORPUS_DIGEST
        or manifest.get("selected_dialogue_episode_count") != 710
        or manifest.get("split_episode_counts") != EXPECTED_SPLITS
    ):
        raise Refuse("OASST2 derived corpus manifest differs from the frozen data contract")

    base_corpus = LanguageEpisodeCorpus.from_jsonl([EPISODES.resolve()])
    if base_corpus.digest != MANIFEST_CORPUS_DIGEST or base_corpus.sample_counts != EXPECTED_SPLITS:
        raise Refuse("reloaded OASST2 corpus digest or split counts differ")
    families: dict[str, set[str]] = {}
    for episode in base_corpus.episodes:
        families.setdefault(episode.family_id, set()).add(episode.split)
    if any(len(splits) != 1 for splits in families.values()):
        raise Refuse("conversation tree family crossed a corpus split")
    family_counts = Counter(next(iter(splits)) for splits in families.values())
    if family_counts != Counter(manifest.get("split_family_counts", {})):
        raise Refuse("conversation family split counts differ from the frozen manifest")

    parent_payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=True)
    parent_report = _read_json(PARENT_REPORT)
    if (
        parent_payload.get("format") != LanguageAlignmentTrainer.CHECKPOINT_FORMAT
        or parent_payload.get("version") != LanguageAlignmentTrainer.CHECKPOINT_VERSION
        or parent_payload.get("corpus_digest") != parent_report.get("corpus_digest")
        or parent_report.get("status") != "completed_dev_evaluation"
        or parent_report.get("dev_trained", {}).get("response_boundary_rate") != 0.0
        or parent_report.get("dev_trained", {}).get("utf8_valid_rate") != 1.0
        or parent_payload.get("config", {}).get("response_plan_target_geometry")
        != "h3_7b_byte_aligned_chunks"
        or parent_payload.get("config", {}).get("max_generation_bytes") != 512
    ):
        raise Refuse("selected H3.7B parent state or its recorded generation baseline differs")

    source_hashes = {
        str(path.resolve()): sha256(path)
        for path in (
            PLAN,
            RETRY1_ADDENDUM,
            RETRY_ADDENDUM,
            BUILDER,
            RAW_SOURCE,
            EPISODES,
            DATA_MANIFEST,
            PARENT_CHECKPOINT,
            PARENT_REPORT,
            FAILED_ATTEMPT,
            FAILED_RESULT,
            FAILED_READINESS,
            FAILED_ZERO,
            RETRY1_ATTEMPT,
            RETRY1_RESULT,
            RETRY1_READINESS,
            RETRY1_ZERO,
            RETRY1_RESTORE_REAUDIT,
        )
    }
    return base_corpus, manifest, parent_payload, source_hashes


def _new_trainer(
    parent_payload: dict[str, Any],
    corpus: LanguageEpisodeCorpus,
) -> LanguageAlignmentTrainer:
    model_payload = parent_payload.get("model")
    if not isinstance(model_payload, dict):
        raise Refuse("H3.7B parent checkpoint is missing its Taiji model bundle")
    model = Taiji.from_checkpoint(model_payload)
    config = LanguageAlignmentConfig(**parent_payload["config"])
    if config.max_generation_bytes != 512 or not config.response_plan_readout:
        raise Refuse("R2 dialogue config differs from the frozen H3.7B byte-aligned parent")
    encoder = ByteAlignedResponsePlanTargetEncoder.fit(
        model,
        corpus,
        slots=config.response_plan_slots,
        slot_width=config.response_plan_width // config.response_plan_slots,
        phase_stride=config.response_plan_phase_stride,
    )
    targets = {
        episode.episode_id: encoder.encode_response(episode.response)
        for episode in corpus.for_split("train")
    }
    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        config=config,
        code_revision=str(parent_payload.get("code_revision") or "working-tree"),
        response_plan_target_encoder=encoder,
        response_plan_targets=targets,
    )
    # Keep the prior course history attached while counting this plan's byte
    # updates separately in the outer execution ledger.
    trainer.global_step = int(parent_payload.get("global_step", 0))
    trainer.episode_count = int(parent_payload.get("episode_count", 0))
    trainer.history = [dict(item) for item in parent_payload.get("history", [])]
    trainer.developmental_history = [
        dict(item) for item in parent_payload.get("developmental_history", [])
    ]
    return trainer


def _independent_restore(
    path: Path,
    arm: str,
    budget: CpuBudget,
) -> dict[str, Any]:
    budget.guard(reserve=15)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    timeout = max(30.0, min(14_400.0, budget.limit - budget.used() - 1.0))
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restore-worker",
            "--checkpoint",
            str(path.resolve()),
            "--arm",
            arm,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        raise Refuse(f"fresh-process checkpoint restore failed: {process.stderr[-2000:]}")
    restored = json.loads(process.stdout)
    if restored.get("complete") is not True or restored.get("equal") is not True:
        raise Refuse(f"fresh-process checkpoint digest mismatch: {path}")
    budget.child_seconds += float(restored.get("cpu_seconds", 0.0))
    budget.guard()
    return {"path": str(path.resolve()), "sha256": sha256(path), **restored}


def _select_final_review_episodes(corpus: LanguageEpisodeCorpus) -> tuple[LanguageEpisode, ...]:
    final = corpus.for_split("final")
    by_family: dict[str, list[LanguageEpisode]] = {}
    for episode in final:
        by_family.setdefault(episode.family_id, []).append(episode)
    representatives = [
        min(values, key=lambda item: item.episode_id) for values in by_family.values()
    ]

    def order_key(episode: LanguageEpisode) -> str:
        raw = f"{GENERATION_SAMPLE_SALT}\0{episode.family_id}\0{episode.episode_id}".encode()
        return hashlib.sha256(raw).hexdigest()

    groups = {
        "without_history": sorted(
            (episode for episode in representatives if not episode.history), key=order_key
        ),
        "with_history": sorted(
            (episode for episode in representatives if episode.history), key=order_key
        ),
    }
    if any(len(values) < GENERATION_SAMPLE_PER_HISTORY for values in groups.values()):
        raise Refuse(
            "final split lacks enough distinct conversation trees for the frozen review sample"
        )
    return tuple(
        groups[stratum][index]
        for index in range(GENERATION_SAMPLE_PER_HISTORY)
        for stratum in ("without_history", "with_history")
    )


def _generate_read_only(
    trainer: LanguageAlignmentTrainer, episode: LanguageEpisode
) -> dict[str, Any]:
    model_snapshot = trainer.model.checkpoint()
    try:
        raw = trainer._generate(episode)
        text, valid_utf8, no_replacement, boundary, stop_reason = trainer._generated_text(raw)
    finally:
        trainer.model.restore(model_snapshot)
        trainer._configure_developmental_mode()
        trainer._configure_response_start_readout()
        trainer._configure_response_phase_readout()
    raw_digest = hashlib.sha256(raw).hexdigest()
    redacted = bool(EMAIL_OR_PHONE.search(text))
    safe_text = EMAIL_OR_PHONE.sub("[已遮蔽]", text)
    compact = "".join(safe_text.split())
    fourgrams = [compact[index : index + 4] for index in range(max(0, len(compact) - 3))]
    repeat_rate = 0.0 if not fourgrams else 1.0 - len(set(fourgrams)) / len(fourgrams)
    return {
        "generated_text": safe_text,
        "generated_bytes_hex": None if redacted else raw.hex(),
        "raw_generated_sha256": raw_digest,
        "pii_pattern_redacted": redacted,
        "generated_bytes_length": len(raw),
        "utf8_valid": valid_utf8,
        "no_replacement": no_replacement,
        "response_boundary_present": boundary,
        "stop_reason": stop_reason,
        "repeated_4gram_rate": round(repeat_rate, 6),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    texts = [str(row["output"]["generated_text"]) for row in records]
    return {
        "generation_count": count,
        "utf8_valid_rate": sum(bool(row["output"]["utf8_valid"]) for row in records) / count,
        "no_replacement_rate": sum(bool(row["output"]["no_replacement"]) for row in records)
        / count,
        "boundary_rate": sum(bool(row["output"]["response_boundary_present"]) for row in records)
        / count,
        "mean_output_bytes": sum(int(row["output"]["generated_bytes_length"]) for row in records)
        / count,
        "unique_output_count": len(set(texts)),
        "duplicate_output_rate": 1.0 - len(set(texts)) / count,
        "mean_repeated_4gram_rate": sum(
            float(row["output"]["repeated_4gram_rate"]) for row in records
        )
        / count,
    }


def _markdown_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# OASST2自然中文对话自由生成样本",
        "",
        "本页不含参考回复，也不做逐字匹配评分。输入全部来自未训练的对话树；输出用于直接检查自然度、上下文衔接、结束和重复。",
        "",
    ]
    for index, row in enumerate(records, 1):
        lines.extend((f"## 样本 {index}", ""))
        for pair_index, (user, assistant) in enumerate(row["history"], 1):
            lines.extend(
                (f"历史用户{pair_index}：{user}", f"历史助手{pair_index}：{assistant}", "")
            )
        lines.append(f"当前用户：{row['current_user_turn']}")
        lines.append("")
        for label, key in (
            ("父档（含历史）", "parent_with_history"),
            ("父档（单轮）", "parent_single_turn"),
            ("多轮候选", "dialogue_context"),
            ("单轮对照", "single_turn"),
        ):
            generated = str(row[key]["generated_text"]).replace("```", "` ` `")
            lines.extend(
                (
                    f"**{label}**（{row[key]['stop_reason']}，{row[key]['generated_bytes_length']}字节）",
                    "",
                    "```text",
                    generated,
                    "```",
                    "",
                )
            )
    return "\n".join(lines) + "\n"


def execute() -> dict[str, Any]:
    started = time.process_time()
    outputs = (READINESS, ATTEMPT, PROGRESS, GENERATIONS, REVIEW, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("R2 daily-dialogue reports are write-once")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(
            f"R2 daily-dialogue checkpoint root already exists: {CHECKPOINT_ROOT}"
        )

    base_corpus, manifest, parent_payload, source_hashes = _load_source_bundle()
    setup_cpu = time.process_time() - started
    budget = CpuBudget(CPU_LIMIT_SECONDS - setup_cpu)
    if budget.limit <= 0:
        raise TimeoutError("R2 daily-dialogue CPU budget exhausted during source validation")

    source_hashes = dict(source_hashes)
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("R2 daily-dialogue plan SHA changed during readiness")

    readiness = {
        "schema": "taiji-r2-daily-dialogue-readiness-v1",
        "status": "ready",
        "plan_sha256": PLAN_SHA256,
        "retry_addendum_sha256": RETRY_ADDENDUM_SHA256,
        "retry1_addendum_sha256": RETRY1_ADDENDUM_SHA256,
        "superseded_failed_attempt_sha256": sha256(FAILED_ATTEMPT),
        "superseded_failed_result_sha256": sha256(FAILED_RESULT),
        "reused_source_zero_checkpoint_sha256": sha256(FAILED_ZERO),
        "retry1_failed_attempt_sha256": sha256(RETRY1_ATTEMPT),
        "retry1_failed_result_sha256": sha256(RETRY1_RESULT),
        "retry1_restore_reaudit_sha256": sha256(RETRY1_RESTORE_REAUDIT),
        "source_revision": SOURCE_REVISION,
        "source_sha256": RAW_SOURCE_SHA256,
        "prepared_episode_sha256": EPISODE_SHA256,
        "prepared_corpus_digest": base_corpus.digest,
        "prepared_episode_count": len(base_corpus.episodes),
        "split_episode_counts": base_corpus.sample_counts,
        "split_tree_counts": manifest["split_family_counts"],
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_checkpoint_digest": parent_payload["checkpoint_digest"],
        "parent_training_checkpoint_updates": parent_payload.get("global_step"),
        "parent_training_episodes": parent_payload.get("episode_count"),
        "parent_dev_utf8_valid_rate": 1.0,
        "parent_dev_response_boundary_rate": 0.0,
        "parent_dev_exact_response_rate": 0.0,
        "parameter_updates": 0,
        "generation_queries": 0,
        "source_hashes": dict(sorted(source_hashes.items())),
        "can_promote": False,
    }
    write_new_json(READINESS, readiness)

    attempt = {
        "schema": "taiji-r2-daily-dialogue-attempt-v1",
        "status": "started",
        "plan_sha256": PLAN_SHA256,
        "retry_addendum_sha256": RETRY_ADDENDUM_SHA256,
        "retry_of": {
            "attempt_sha256": sha256(FAILED_ATTEMPT),
            "failed_result_sha256": sha256(FAILED_RESULT),
            "formal_parameter_updates": 0,
            "formal_generation_queries": 0,
            "preserved_zero_checkpoint_sha256": sha256(FAILED_ZERO),
        },
        "retry1_of": {
            "addendum_sha256": RETRY1_ADDENDUM_SHA256,
            "attempt_sha256": sha256(RETRY1_ATTEMPT),
            "failed_result_sha256": sha256(RETRY1_RESULT),
            "preserved_zero_checkpoint_sha256": sha256(RETRY1_ZERO),
            "read_only_restore_reaudit_sha256": sha256(RETRY1_RESTORE_REAUDIT),
        },
        "readiness_sha256": sha256(READINESS),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "corpus_digest": base_corpus.digest,
        "training_arms": list(ARMS),
        "epochs_per_arm": EPOCHS,
        "train_episodes_per_arm_per_epoch": EXPECTED_SPLITS["train"],
        "formal_parameter_updates": 0,
        "formal_generation_queries": 0,
        "max_generation_bytes": 512,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "final_review_sample_count": GENERATION_SAMPLE_PER_HISTORY * 2,
        "temporary_preflight_history": [
            {
                "method": "checkpoint_roundtrip_preflight with max_generation_bytes=512",
                "status": "interrupted_without_returned_result",
                "persistent_checkpoint": False,
                "temporary_child_update_may_have_run": True,
                "cpu_seconds": None,
            },
            {
                "method": "manual zero-step roundtrip over two arms",
                "status": "interrupted_without_returned_result",
                "persistent_checkpoint": False,
                "parameter_updates": 0,
                "generation_queries": 0,
                "cpu_seconds": None,
            },
        ],
        "source_hashes": dict(sorted(source_hashes.items())),
        "can_promote": False,
    }
    write_new_json(ATTEMPT, attempt)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    zero_trainers: dict[str, LanguageAlignmentTrainer] = {}
    trainers: dict[str, LanguageAlignmentTrainer] = {}
    zero_records: dict[str, dict[str, Any]] = {}
    final_records: dict[str, dict[str, Any]] = {}
    completed_episode_count = Counter()
    new_update_count = Counter()
    generation_count = 0
    active_arm: str | None = None
    active_trainer: LanguageAlignmentTrainer | None = None
    try:
        arm_corpora = {arm: _arm_corpus(base_corpus, arm) for arm in ARMS}
        for arm in ARMS:
            budget.guard(reserve=500)
            if sha256(PARENT_CHECKPOINT) != PARENT_CHECKPOINT_SHA256:
                raise Refuse("H3.7B parent checkpoint changed before zero bundle creation")
            trainer = _new_trainer(parent_payload, arm_corpora[arm])
            checkpoint_path = CHECKPOINT_ROOT / f"{arm}_zero.pt"
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            zero_payload = trainer.checkpoint()
            _atomic_save(zero_payload, checkpoint_path)
            zero_sha = sha256(checkpoint_path)
            restore = _independent_restore(checkpoint_path, arm, budget)
            if restore.get("checkpoint_digest") != zero_payload["checkpoint_digest"]:
                raise Refuse(f"R2 {arm} zero bundle digest changed after independent restore")
            zero_trainers[arm] = trainer
            trainers[arm] = copy.deepcopy(trainer)
            if trainers[arm].checkpoint()["checkpoint_digest"] != zero_payload["checkpoint_digest"]:
                raise Refuse(f"R2 {arm} candidate copy differs from the saved zero bundle")
            zero_records[arm] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": zero_sha,
                "checkpoint_digest": zero_payload["checkpoint_digest"],
                "corpus_digest": arm_corpora[arm].digest,
                "fresh_process_restore": restore,
            }
            del zero_payload
        if len(zero_records) != len(ARMS) or any(
            not row["fresh_process_restore"].get("complete") for row in zero_records.values()
        ):
            raise Refuse("R2 daily-dialogue zero checkpoint preflight did not pass for both arms")

        for arm in ARMS:
            active_arm = arm
            active_trainer = trainers[arm]
            budget.guard(reserve=GENERATION_RESERVE_SECONDS)
            start_global_step = active_trainer.global_step
            start_episode_count = active_trainer.episode_count
            local_episode_index = 0

            # B023: 默认参数把当前迭代的 arm 钉进闭包（train 在同一次迭代内同步调用，
            # 行为不变；防的是未来把回调挪出循环体后的晚绑定串臂）。
            def on_progress(record: dict[str, Any], arm: str = arm) -> None:
                nonlocal local_episode_index
                local_episode_index += 1
                completed_episode_count[arm] += 1
                new_update_count[arm] += int(record["observations"])
                progress_row = {
                    "event": "training_episode",
                    "arm": arm,
                    "epoch": int(record["epoch"]),
                    "episode_index": local_episode_index,
                    "episode_id": str(record["episode_id"]),
                    "family_id": str(record["family_id"]),
                    "target_bytes": int(record["target_bytes"]),
                    "prefix_bytes": int(record["prefix_bytes"]),
                    "observations": int(record["observations"]),
                    "response_mean_surprise": float(record["response_mean_surprise"]),
                    "cumulative_plan_observations": int(new_update_count[arm]),
                }
                write_jsonl(progress, progress_row)
                if local_episode_index % 16 == 0:
                    progress.flush()
                if local_episode_index % 64 == 0:
                    print(
                        f"r2-daily-dialogue arm={arm} epoch={record['epoch']} "
                        f"episode={local_episode_index} updates={new_update_count[arm]} "
                        f"cpu={budget.used():.1f}",
                        flush=True,
                    )
                budget.guard(reserve=GENERATION_RESERVE_SECONDS)

            training_result = active_trainer.train(epochs=EPOCHS, progress=on_progress)
            if (
                training_result.get("status") != "completed"
                or training_result.get("episodes") != EPOCHS * EXPECTED_SPLITS["train"]
            ):
                raise Refuse(f"R2 {arm} did not finish the frozen full training course")
            if active_trainer.global_step - start_global_step != new_update_count[arm]:
                raise Refuse(f"R2 {arm} global-step delta differs from progress ledger")
            if (
                active_trainer.episode_count - start_episode_count
                != EPOCHS * EXPECTED_SPLITS["train"]
            ):
                raise Refuse(f"R2 {arm} episode count differs from frozen schedule")

            final_path = CHECKPOINT_ROOT / f"{arm}_final.pt"
            if final_path.exists():
                raise FileExistsError(final_path)
            final_payload = active_trainer.checkpoint()
            _atomic_save(final_payload, final_path)
            final_sha = sha256(final_path)
            final_restore = _independent_restore(final_path, arm, budget)
            if final_restore.get("checkpoint_digest") != final_payload["checkpoint_digest"]:
                raise Refuse(f"R2 {arm} final checkpoint digest changed after independent restore")
            final_records[arm] = {
                "path": str(final_path.resolve()),
                "sha256": final_sha,
                "checkpoint_digest": final_payload["checkpoint_digest"],
                "corpus_digest": arm_corpora[arm].digest,
                "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
                "plan_local_response_observations": int(new_update_count[arm]),
                "plan_local_training_episodes": int(completed_episode_count[arm]),
                "episodes_per_epoch": EXPECTED_SPLITS["train"],
                "epochs": EPOCHS,
                "fresh_process_restore": final_restore,
                "response_plan_target_digest": active_trainer.response_plan_target_digest,
                "can_promote": False,
            }
            del final_payload
            active_trainer = None
            active_arm = None
            progress.flush()

        if set(final_records) != set(ARMS) or any(
            not row["fresh_process_restore"].get("complete") for row in final_records.values()
        ):
            raise Refuse("R2 daily-dialogue final bundle restores are incomplete")
        for path, expected in source_hashes.items():
            if not Path(path).is_file() or sha256(Path(path)) != expected:
                raise Refuse(f"R2 daily-dialogue source changed during training: {path}")
        budget.guard(reserve=0)

        review_episodes = _select_final_review_episodes(base_corpus)
        if len(review_episodes) != GENERATION_SAMPLE_PER_HISTORY * 2:
            raise Refuse("R2 final generation review sample count differs")
        for arm in ARMS:
            if sha256(Path(zero_records[arm]["path"])) != zero_records[arm]["sha256"]:
                raise Refuse(f"R2 {arm} zero checkpoint changed before generation")
            if sha256(Path(final_records[arm]["path"])) != final_records[arm]["sha256"]:
                raise Refuse(f"R2 {arm} final checkpoint changed before generation")

        generation_trainers = {
            "parent_with_history": zero_trainers["dialogue_context"],
            "parent_single_turn": zero_trainers["single_turn"],
            "dialogue_context": trainers["dialogue_context"],
            "single_turn": trainers["single_turn"],
        }
        before_digests = {
            name: trainer.checkpoint()["checkpoint_digest"]
            for name, trainer in generation_trainers.items()
        }
        generation_rows = []
        for index, episode in enumerate(review_episodes, 1):
            budget.guard(reserve=0)
            flat_episode = replace(episode, history=())
            row = {
                "index": index,
                "episode_id": episode.episode_id,
                "family_id": episode.family_id,
                "history_turn_count": len(episode.history),
                "history": [[user, assistant] for user, assistant in episode.history],
                "current_user_turn": episode.user_input,
                "parent_with_history": _generate_read_only(
                    generation_trainers["parent_with_history"], episode
                ),
                "parent_single_turn": _generate_read_only(
                    generation_trainers["parent_single_turn"], flat_episode
                ),
                "dialogue_context": _generate_read_only(
                    generation_trainers["dialogue_context"], episode
                ),
                "single_turn": _generate_read_only(
                    generation_trainers["single_turn"], flat_episode
                ),
                "reference_answer_included": False,
                "exact_response_scored": False,
            }
            generation_rows.append(row)
            generation_count += len(generation_trainers)
            write_jsonl(
                progress,
                {
                    "event": "generation_review_row",
                    "index": index,
                    "episode_id": episode.episode_id,
                    "family_id": episode.family_id,
                    "history_turn_count": len(episode.history),
                    "current_user_turn": episode.user_input,
                    "outputs": {
                        name: {
                            "generated_text": row[name]["generated_text"],
                            "generated_bytes_length": row[name]["generated_bytes_length"],
                            "raw_generated_sha256": row[name]["raw_generated_sha256"],
                            "pii_pattern_redacted": row[name]["pii_pattern_redacted"],
                            "utf8_valid": row[name]["utf8_valid"],
                            "response_boundary_present": row[name]["response_boundary_present"],
                            "stop_reason": row[name]["stop_reason"],
                        }
                        for name in generation_trainers
                    },
                    "reference_answer_included": False,
                },
            )
            progress.flush()
            if index % 4 == 0:
                print(
                    f"r2-daily-dialogue generation={index}/{len(review_episodes)} cpu={budget.used():.1f}",
                    flush=True,
                )
        after_digests = {
            name: trainer.checkpoint()["checkpoint_digest"]
            for name, trainer in generation_trainers.items()
        }
        if before_digests != after_digests:
            raise Refuse("R2 final generation mutated a parent or candidate checkpoint")
        if any(
            row["reference_answer_included"] or row["exact_response_scored"]
            for row in generation_rows
        ):
            raise Refuse("R2 final generation path exposed or scored a reference response")
        for path, expected in source_hashes.items():
            if not Path(path).is_file() or sha256(Path(path)) != expected:
                raise Refuse(f"R2 daily-dialogue source changed during final generation: {path}")

        all_generation_records = {
            name: [
                {"episode_id": row["episode_id"], "output": row[name]} for row in generation_rows
            ]
            for name in generation_trainers
        }
        generation_metrics = {name: _summary(rows) for name, rows in all_generation_records.items()}
        generation_report = {
            "schema": "taiji-r2-daily-dialogue-generations-v1",
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "corpus_digest": base_corpus.digest,
            "final_split_episodes_available": EXPECTED_SPLITS["final"],
            "review_sample_count": len(review_episodes),
            "review_sample_unique_trees": len({row["family_id"] for row in generation_rows}),
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
            "metrics": generation_metrics,
            "records": generation_rows,
            "can_promote": False,
        }
        write_new_json(GENERATIONS, generation_report)
        write_once_text(REVIEW, _markdown_report(generation_rows))

        cumulative_cpu = budget.used() + setup_cpu
        if cumulative_cpu > CPU_LIMIT_SECONDS:
            raise TimeoutError("R2 daily-dialogue completion exceeded the frozen CPU limit")
        result = {
            "schema": "taiji-r2-daily-dialogue-material-comparison-result-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "retry_addendum_sha256": RETRY_ADDENDUM_SHA256,
            "retry_of": {
                "attempt_sha256": sha256(FAILED_ATTEMPT),
                "failed_result_sha256": sha256(FAILED_RESULT),
                "formal_parameter_updates": 0,
                "formal_generation_queries": 0,
                "preserved_zero_checkpoint_sha256": sha256(FAILED_ZERO),
            },
            "retry1_of": {
                "addendum_sha256": RETRY1_ADDENDUM_SHA256,
                "attempt_sha256": sha256(RETRY1_ATTEMPT),
                "failed_result_sha256": sha256(RETRY1_RESULT),
                "preserved_zero_checkpoint_sha256": sha256(RETRY1_ZERO),
                "read_only_restore_reaudit_sha256": sha256(RETRY1_RESTORE_REAUDIT),
            },
            "readiness_sha256": sha256(READINESS),
            "attempt_sha256": sha256(ATTEMPT),
            "progress_sha256": sha256(PROGRESS),
            "generations_sha256": sha256(GENERATIONS),
            "review_markdown_sha256": sha256(REVIEW),
            "source_hashes": dict(sorted(source_hashes.items())),
            "source_hashes_unchanged": True,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "corpus_digest": base_corpus.digest,
            "split_episode_counts": base_corpus.sample_counts,
            "zero_checkpoints": zero_records,
            "final_checkpoints": final_records,
            "training_arms": list(ARMS),
            "epochs_per_arm": EPOCHS,
            "train_episodes_per_arm_per_epoch": EXPECTED_SPLITS["train"],
            "plan_local_training_episodes": {
                arm: int(completed_episode_count[arm]) for arm in ARMS
            },
            "plan_local_response_observations": {arm: int(new_update_count[arm]) for arm in ARMS},
            "training_summary": {
                arm: {
                    "plan_local_training_episodes": int(completed_episode_count[arm]),
                    "plan_local_response_observations": int(new_update_count[arm]),
                    "epochs": EPOCHS,
                    "training_complete": True,
                }
                for arm in ARMS
            },
            "generation_query_count": len(generation_rows) * len(generation_trainers),
            "generation_queries_per_endpoint": len(generation_rows),
            "dev_queries": 0,
            "reference_answer_scoring_queries": 0,
            "generation_metrics": generation_metrics,
            "generation_review_path": str(REVIEW.resolve()),
            "cpu_seconds": round(cumulative_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "temporary_preflight_attempts_unmetered": True,
            "errors": [],
            "can_promote": False,
        }
        write_new_json(RESULT, result)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        partial_checkpoint = None
        if active_trainer is not None and active_arm is not None:
            try:
                partial_path = CHECKPOINT_ROOT / f"{active_arm}_partial.pt"
                if not partial_path.exists():
                    payload = active_trainer.checkpoint()
                    _atomic_save(payload, partial_path)
                    partial_checkpoint = {
                        "path": str(partial_path.resolve()),
                        "sha256": sha256(partial_path),
                        "checkpoint_digest": payload["checkpoint_digest"],
                        "global_step": active_trainer.global_step,
                    }
            except Exception as save_error:
                partial_checkpoint = {"save_error": f"{type(save_error).__name__}: {save_error}"}
        failure = {
            "schema": "taiji-r2-daily-dialogue-material-comparison-result-v1",
            "status": "partial" if sum(new_update_count.values()) else "failed_before_training",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "corpus_digest": base_corpus.digest if "base_corpus" in locals() else None,
            "completed_training_episodes": dict(completed_episode_count),
            "completed_response_observations": dict(new_update_count),
            "completed_generation_outputs": generation_count,
            "active_arm": active_arm,
            "partial_checkpoint": partial_checkpoint,
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


def write_once_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"R2 dialogue text output is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()


def _restore_worker(checkpoint: Path, arm: str) -> dict[str, Any]:
    started = time.process_time()
    base_corpus = LanguageEpisodeCorpus.from_jsonl([EPISODES.resolve()])
    corpus = _arm_corpus(base_corpus, arm)
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
        "response_plan_target_digest": trainer.response_plan_target_digest,
        "cpu_seconds": round(time.process_time() - started, 6),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    torch.set_num_threads(1)
    args = _parse_args(argv)
    if args.restore_worker:
        if args.checkpoint is None or args.arm is None:
            raise SystemExit("--restore-worker requires --checkpoint and --arm")
        report = _restore_worker(args.checkpoint, args.arm)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    try:
        report = execute()
    except Exception as exc:
        print(f"R2 daily-dialogue run refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "plan_local_response_observations": report.get("plan_local_response_observations"),
                "generation_query_count": report.get("generation_query_count"),
                "cpu_seconds": report.get("cpu_seconds"),
                "can_promote": report.get("can_promote"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
