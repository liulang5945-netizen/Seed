"""Pretrain a fresh Taiji model on role-aware OASST2 turns, then tune responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep.audit_taiji_r2_daily_dialogue_material_inventory import (
    _json_selected_fields,
)
from taiji import (
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    Taiji,
    TaijiConfig,
)
from taiji.internalization import content_digest
from taiji.language_alignment import LANGUAGE_RESPONSE_PLAN_TARGET_H37B, _atomic_save
from taiji.response_plan_target import ByteAlignedResponsePlanTargetEncoder

PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md"
PLAN_SHA256 = "d043f1e18e17057c3546e28c6dc6b816034676f7e0f4ba43d511626d8cec9521"
BUILDER = ROOT / "scripts/data_prep/build_taiji_r2_daily_dialogue_full_turn_pretraining_corpus.py"
BUILDER_SHA256 = "d88b126ffd3caeccddfc6094d182403e052af8dbde80358871a6378034e0dfd8"
PRETRAIN_EPISODES = (
    ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_episodes.jsonl"
)
PRETRAIN_EPISODES_SHA256 = "ec4d6f292dc8b9f05dfb8fa83ca4881ea51f3e4743de864bc7b7e78a76ec8b52"
PRETRAIN_MANIFEST = (
    ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_manifest.json"
)
PRETRAIN_MANIFEST_SHA256 = "15e5cfa7c628ebf6d599f49265bc41fa57f423a7605e6eb21b70aa2621f12474"
PRETRAIN_READINESS = (
    ROOT / "reports/taiji_r2_daily_dialogue_full_turn_pretraining_data_readiness_20261021.json"
)
PRETRAIN_READINESS_SHA256 = "725dbfc0756dde0837be03ed0aab46e763d44b35784e3803eb866520a6f3068c"
RAW_SOURCE = ROOT / "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
RAW_SOURCE_SHA256 = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
ASSISTANT_EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
ASSISTANT_EPISODES_SHA256 = "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d"
ASSISTANT_MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json"
ASSISTANT_MANIFEST_SHA256 = "3f5b453aaab88576afd458f7b81bdcff1f1bfbf1ffd94890cb72602270211472"
MATERIAL_INVENTORY = (
    ROOT / "reports/taiji_r2_daily_dialogue_material_inventory_recovery1_20261021.json"
)
MATERIAL_INVENTORY_SHA256 = "6bf34320c2963236f916fa920eb19911bf369ea8dc5ccdb120a81b1a2d3a5dc8"
OLD_GENERATIONS = (
    ROOT / "reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_generations.json"
)
OLD_GENERATIONS_SHA256 = "07b439f2bf536132a88e302ffc4fb0b0e13397d4cc19e4d0f5b603319cff8c19"
OLD_RESULT = ROOT / "reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_result.json"
OLD_RESULT_SHA256 = "eb55fa66914745187a7507a6e4d0f0eb68c8f21f148638e037d1195736737d76"
BASELINE_CHECKPOINT = (
    ROOT
    / "data/checkpoints/r2_daily_dialogue_oasst2_20261021_continuation/dialogue_context_final.pt"
)
BASELINE_CHECKPOINT_SHA256 = "41655eb8b5fcd7e446acb924e94e458ce9b1e966a96a95517198a5ac38df2086"
REPORT_ROOT = ROOT / "reports"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_full_turn_20261021"
READINESS = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_readiness.json"
ATTEMPT = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_attempt.json"
PROGRESS = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_progress.jsonl"
GENERATIONS = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_generations.json"
REVIEW = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_generation_review.md"
RESULT = REPORT_ROOT / "taiji_r2_daily_dialogue_full_turn_20261021_result.json"

CAPACITY_BUDGET = 1_000_000
SEED = 20261022
PRETRAIN_EPOCHS = 8
ASSISTANT_CALIBRATION_EPOCHS = 1
CPU_LIMIT_SECONDS = 240_000.0
PREVIOUS_DATA_BUILD_CPU_SECONDS = 12.65625
GENERATION_RESERVE_SECONDS = 9_000.0
MAX_GENERATION_BYTES = 512
MAX_HISTORY_PAIRS = 3
MAX_PREFIX_BYTES = 4_096
PRETRAIN_EPISODE_COUNT = 3762
PRETRAIN_FAMILY_COUNT = 508
PRETRAIN_TARGET_BYTES_PER_EPOCH = 1_814_739
ASSISTANT_TRAIN_EPISODE_COUNT = 525
ASSISTANT_TRAIN_TARGET_BYTES = 113_209
DEV_FAMILY_COUNT = 39
GENERATION_ENDPOINTS = ("previous_8epoch_candidate", "full_turn_pretrained_candidate")
EMAIL_OR_PHONE = re.compile(r"(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?\d[\d ()-]{7,}\d))")


class Refuse(RuntimeError):
    """A frozen dialogue-pretraining source or checkpoint contract failed."""


class CpuBudget:
    def __init__(self, limit: float) -> None:
        self.limit = float(limit)
        self.started = time.process_time()
        self.child_seconds = 0.0

    def used(self) -> float:
        return time.process_time() - self.started + self.child_seconds

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve > self.limit:
            raise TimeoutError(
                f"full-turn dialogue CPU budget exceeded: "
                f"{self.used():.3f}+{reserve:.3f}/{self.limit:.3f}"
            )


@dataclass(frozen=True)
class PromptOnlyEpisode:
    episode_id: str
    family_id: str
    system: str
    context: str
    history: tuple[tuple[str, str], ...]
    user_input: str
    unknown_policy: str

    @property
    def prompt_bytes(self) -> bytes:
        parts = [
            "<|system|>",
            self.system,
            "<|policy|>",
            self.unknown_policy,
            "<|context|>",
            self.context if self.context else "<|none|>",
        ]
        for user, assistant in self.history:
            parts.extend(("<|history_user|>", user, "<|history_assistant|>", assistant))
        parts.extend(("<|user|>", self.user_input, "<|assistant|>"))
        return ("\n".join(parts) + "\n").encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"full-turn training report is write-once: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def write_jsonl(stream, payload: dict[str, Any]) -> None:
    json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Refuse(f"expected JSON object: {path}")
    return value


def _load_pretrain_corpus() -> LanguageEpisodeCorpus:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [PRETRAIN_EPISODES.resolve()], require_eval_splits=False
    )
    if len(corpus.episodes) != PRETRAIN_EPISODE_COUNT:
        raise Refuse("full-turn pretraining episode count differs")
    if corpus.sample_counts != {
        "train": PRETRAIN_EPISODE_COUNT,
        "dev": 0,
        "final": 0,
        "retention": 0,
    }:
        raise Refuse("full-turn pretraining corpus contains a non-train split")
    if len({episode.family_id for episode in corpus.episodes}) != PRETRAIN_FAMILY_COUNT:
        raise Refuse("full-turn pretraining family count differs")
    if Counter(episode.target_role for episode in corpus.episodes) != {
        "assistant": 2244,
        "user": 1518,
    }:
        raise Refuse("full-turn pretraining target-role counts differ")
    return corpus


def _load_assistant_train_corpus() -> LanguageEpisodeCorpus:
    episodes = []
    with ASSISTANT_EPISODES.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            metadata = _json_selected_fields(line, {"split"})
            if metadata.get("split") != "train":
                continue
            episodes.append(LanguageEpisode.from_payload(json.loads(line)))
    corpus = LanguageEpisodeCorpus(
        tuple(episodes),
        ((str(ASSISTANT_EPISODES.resolve()), ASSISTANT_EPISODES_SHA256),),
        require_eval_splits=False,
    )
    if len(corpus.episodes) != ASSISTANT_TRAIN_EPISODE_COUNT:
        raise Refuse("assistant calibration corpus count differs")
    if len({episode.family_id for episode in corpus.episodes}) != 340:
        raise Refuse("assistant calibration train-family count differs")
    if any(episode.target_role != "assistant" for episode in corpus.episodes):
        raise Refuse("assistant calibration corpus contains a non-assistant target")
    if (
        sum(len(episode.target_bytes) for episode in corpus.episodes)
        != ASSISTANT_TRAIN_TARGET_BYTES
    ):
        raise Refuse("assistant calibration target byte count differs")
    return corpus


def _fresh_code_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _source_paths() -> tuple[Path, ...]:
    return (
        PLAN,
        BUILDER,
        PRETRAIN_EPISODES,
        PRETRAIN_MANIFEST,
        PRETRAIN_READINESS,
        RAW_SOURCE,
        ASSISTANT_EPISODES,
        ASSISTANT_MANIFEST,
        MATERIAL_INVENTORY,
        OLD_GENERATIONS,
        OLD_RESULT,
        BASELINE_CHECKPOINT,
        ROOT / "taiji/language_alignment.py",
        ROOT / "taiji/model.py",
        ROOT / "taiji/response_plan_target.py",
        ROOT / "scripts/data_prep/audit_taiji_r2_daily_dialogue_material_inventory.py",
        ROOT / "tests/taiji_native/test_r2_daily_dialogue_pretraining_contract.py",
        ROOT / "tests/taiji_native/test_r2_daily_dialogue_full_turn_corpus_contract.py",
        Path(__file__).resolve(),
    )


def _validate_sources() -> tuple[LanguageEpisodeCorpus, LanguageEpisodeCorpus, dict[str, Any]]:
    expected = {
        PLAN: PLAN_SHA256,
        BUILDER: BUILDER_SHA256,
        PRETRAIN_EPISODES: PRETRAIN_EPISODES_SHA256,
        PRETRAIN_MANIFEST: PRETRAIN_MANIFEST_SHA256,
        PRETRAIN_READINESS: PRETRAIN_READINESS_SHA256,
        RAW_SOURCE: RAW_SOURCE_SHA256,
        ASSISTANT_EPISODES: ASSISTANT_EPISODES_SHA256,
        ASSISTANT_MANIFEST: ASSISTANT_MANIFEST_SHA256,
        MATERIAL_INVENTORY: MATERIAL_INVENTORY_SHA256,
        OLD_GENERATIONS: OLD_GENERATIONS_SHA256,
        OLD_RESULT: OLD_RESULT_SHA256,
        BASELINE_CHECKPOINT: BASELINE_CHECKPOINT_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise Refuse(f"full-turn dialogue source changed: {path}")
    manifest = _read_json(PRETRAIN_MANIFEST)
    readiness = _read_json(PRETRAIN_READINESS)
    if (
        readiness.get("status") != "ready"
        or readiness.get("episode_count") != PRETRAIN_EPISODE_COUNT
        or readiness.get("family_count") != PRETRAIN_FAMILY_COUNT
        or readiness.get("target_bytes_with_end_markers_by_role")
        != {"assistant": 1_643_502, "user": 171_237}
        or manifest.get("episode_jsonl_sha256") != PRETRAIN_EPISODES_SHA256
        or manifest.get("train_family_count") != PRETRAIN_FAMILY_COUNT
        or manifest.get("episode_count") != PRETRAIN_EPISODE_COUNT
        or manifest.get("target_bytes_with_end_markers_by_role")
        != {"assistant": 1_643_502, "user": 171_237}
    ):
        raise Refuse("full-turn pretraining readiness or manifest differs")
    pretrain_corpus = _load_pretrain_corpus()
    assistant_corpus = _load_assistant_train_corpus()
    pretrain_target_bytes = sum(len(episode.target_bytes) for episode in pretrain_corpus.episodes)
    if pretrain_target_bytes != PRETRAIN_TARGET_BYTES_PER_EPOCH:
        raise Refuse("loaded full-turn training digest has a different target-byte total")
    return pretrain_corpus, assistant_corpus, manifest


def _independent_restore(checkpoint: Path, stage: str, budget: CpuBudget) -> dict[str, Any]:
    budget.guard(reserve=20)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--restore-worker",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--stage",
            stage,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise Refuse(f"fresh-process {stage} restore failed: {process.stderr[-2000:]}")
    receipt = json.loads(process.stdout)
    if receipt.get("complete") is not True or receipt.get("equal") is not True:
        raise Refuse(f"fresh-process {stage} restore receipt is incomplete")
    budget.child_seconds += float(receipt.get("cpu_seconds", 0.0))
    budget.guard()
    return {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint), **receipt}


def _restore_worker(checkpoint: Path, stage: str) -> dict[str, Any]:
    if stage == "pretraining":
        corpus = _load_pretrain_corpus()
    elif stage == "assistant_sft":
        corpus = _load_assistant_train_corpus()
    else:
        raise Refuse(f"unknown restore stage: {stage}")
    started = time.process_time()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    if trainer.checkpoint()["checkpoint_digest"] != payload["checkpoint_digest"]:
        raise Refuse(f"{stage} checkpoint restore digest differs")
    return {
        "complete": True,
        "equal": True,
        "stage": stage,
        "checkpoint_digest": payload["checkpoint_digest"],
        "corpus_digest": corpus.digest,
        "episode_count": trainer.episode_count,
        "global_step": trainer.global_step,
        "trainer_history_count": len(trainer.history),
        "active_parameters": trainer.model.parameter_count(),
        "cpu_seconds": round(time.process_time() - started, 6),
    }


def _model_config() -> TaijiConfig:
    return TaijiConfig.capacity_profile(
        CAPACITY_BUDGET,
        seed=SEED,
        additional_predictive_readouts=1,
        response_plan_width=48,
    )


def _pretraining_config() -> LanguageAlignmentConfig:
    return LanguageAlignmentConfig(
        response_plan_readout=False,
        response_plan_width=48,
        response_plan_variant="factorized_v1",
        response_plan_target_geometry="signed_hash_span",
        max_generation_bytes=MAX_GENERATION_BYTES,
        constrained_decode=True,
        use_memory=False,
        developmental_mode="static",
    )


def _assistant_sft_config() -> LanguageAlignmentConfig:
    return LanguageAlignmentConfig(
        response_plan_readout=True,
        response_plan_width=48,
        response_plan_variant="factorized_v1",
        response_plan_slots=4,
        response_plan_phase_stride=16,
        response_plan_target_geometry=LANGUAGE_RESPONSE_PLAN_TARGET_H37B,
        max_generation_bytes=MAX_GENERATION_BYTES,
        constrained_decode=True,
        use_memory=False,
        developmental_mode="static",
    )


def _assistant_sft_trainer(
    model: Taiji, corpus: LanguageEpisodeCorpus, code_revision: str
) -> tuple[LanguageAlignmentTrainer, ByteAlignedResponsePlanTargetEncoder]:
    encoder = ByteAlignedResponsePlanTargetEncoder.fit(model, corpus)
    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        config=_assistant_sft_config(),
        code_revision=code_revision,
        response_plan_target_encoder=encoder,
    )
    return trainer, encoder


def _generate_readonly(trainer: LanguageAlignmentTrainer, episode: Any) -> dict[str, Any]:
    snapshot = trainer.model.checkpoint()
    try:
        raw = trainer._generate(episode)
        text, utf8_valid, no_replacement, boundary, stop_reason = trainer._generated_text(raw)
    finally:
        trainer.model.restore(snapshot)
        trainer._configure_developmental_mode()
        trainer._configure_response_start_readout()
        trainer._configure_response_phase_readout()
        trainer._configure_response_plan_readout()
    redacted = bool(EMAIL_OR_PHONE.search(text))
    safe_text = EMAIL_OR_PHONE.sub("[已遮蔽]", text)
    compact = "".join(safe_text.split())
    fourgrams = [compact[index : index + 4] for index in range(max(0, len(compact) - 3))]
    repeat_rate = 0.0 if not fourgrams else 1.0 - len(set(fourgrams)) / len(fourgrams)
    return {
        "generated_text": safe_text,
        "raw_generated_sha256": hashlib.sha256(raw).hexdigest(),
        "generated_bytes_hex": None if redacted else raw.hex(),
        "generated_bytes_length": len(raw),
        "utf8_valid": utf8_valid,
        "no_replacement": no_replacement,
        "response_boundary_present": boundary,
        "stop_reason": stop_reason,
        "pii_pattern_redacted": redacted,
        "repeated_4gram_rate": round(repeat_rate, 6),
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


def _select_dev_prompts() -> tuple[PromptOnlyEpisode, ...]:
    representatives: dict[str, tuple[str, int]] = {}
    with ASSISTANT_EPISODES.open("r", encoding="utf-8") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            metadata = _json_selected_fields(line, {"episode_id", "family_id", "split"})
            if metadata.get("split") != "dev":
                continue
            family_id = str(metadata["family_id"])
            episode_id = str(metadata["episode_id"])
            if family_id not in representatives or episode_id < representatives[family_id][0]:
                representatives[family_id] = (episode_id, offset)
    if len(representatives) != DEV_FAMILY_COUNT:
        raise Refuse("unqueried dev-family count differs from the frozen face")

    prompts: list[PromptOnlyEpisode] = []
    with ASSISTANT_EPISODES.open("r", encoding="utf-8") as stream:
        for family_id in sorted(
            representatives, key=lambda item: hashlib.sha256(item.encode()).hexdigest()
        ):
            episode_id, offset = representatives[family_id]
            stream.seek(offset)
            line = stream.readline()
            fields = _json_selected_fields(
                line,
                {
                    "episode_id",
                    "family_id",
                    "split",
                    "system",
                    "context",
                    "history",
                    "user_input",
                    "unknown_policy",
                },
            )
            if fields.get("episode_id") != episode_id or fields.get("family_id") != family_id:
                raise Refuse("dev prompt seek did not return the frozen episode")
            history = tuple(
                (str(pair["user"]), str(pair["assistant"])) for pair in fields["history"]
            )
            prompts.append(
                PromptOnlyEpisode(
                    episode_id=episode_id,
                    family_id=family_id,
                    system=str(fields["system"]),
                    context=str(fields["context"]),
                    history=history,
                    user_input=str(fields["user_input"]),
                    unknown_policy=str(fields["unknown_policy"]),
                )
            )
    return tuple(prompts)


def _review_markdown(rows: list[dict[str, Any]], *, sample_count: int = 12) -> str:
    lines = [
        "# 全轮预训练后的OASST2未查询dev对话生成样例",
        "",
        "不读取或比较参考回复。每个样本并列旧8-epoch回复式候选与全轮预训练＋assistant校准候选。",
        "",
    ]
    for row in rows[:sample_count]:
        lines.extend((f"## 样本 {row['index']}（history={row['history_turn_count']}）", ""))
        for turn, (user, assistant) in enumerate(row["history"], 1):
            lines.extend((f"历史用户{turn}：{user}", f"历史助手{turn}：{assistant}", ""))
        lines.extend((f"当前用户：{row['current_user_turn']}", ""))
        for label, key in (
            ("既有8-epoch候选", "previous_8epoch_candidate"),
            ("全轮预训练候选", "full_turn_pretrained_candidate"),
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
    return "\n".join(lines).rstrip() + "\n"


def execute() -> dict[str, Any]:
    started = time.process_time()
    budget = CpuBudget(CPU_LIMIT_SECONDS - PREVIOUS_DATA_BUILD_CPU_SECONDS)
    output_paths = (READINESS, ATTEMPT, PROGRESS, GENERATIONS, REVIEW, RESULT)
    if any(path.exists() for path in output_paths):
        raise FileExistsError("R2-DIALOGUE-06 reports are write-once")
    if CHECKPOINT_ROOT.exists():
        raise FileExistsError(f"R2-DIALOGUE-06 checkpoint root exists: {CHECKPOINT_ROOT}")

    pretrain_corpus, assistant_corpus, pretrain_manifest = _validate_sources()
    pretrain_bytes_per_epoch = sum(
        len(episode.target_bytes) for episode in pretrain_corpus.episodes
    )
    sft_target_bytes = sum(len(episode.target_bytes) for episode in assistant_corpus.episodes)
    if pretrain_bytes_per_epoch != PRETRAIN_TARGET_BYTES_PER_EPOCH:
        raise Refuse("role-aware pretraining byte total differs")
    if sft_target_bytes != ASSISTANT_TRAIN_TARGET_BYTES:
        raise Refuse("assistant calibration byte total differs")
    expected_pretrain_episodes = PRETRAIN_EPISODE_COUNT * PRETRAIN_EPOCHS
    expected_pretrain_updates = pretrain_bytes_per_epoch * PRETRAIN_EPOCHS

    source_paths = _source_paths()
    source_hashes = {str(path.resolve()): sha256(path) for path in source_paths}
    code_revision = _fresh_code_revision()
    readiness = {
        "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-readiness-v1",
        "status": "ready",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": dict(sorted(source_hashes.items())),
        "builder_provenance_corpus_digest": pretrain_manifest["corpus_digest"],
        "loaded_pretraining_corpus_digest": pretrain_corpus.digest,
        "assistant_sft_train_corpus_digest": assistant_corpus.digest,
        "pretraining_episode_count": PRETRAIN_EPISODE_COUNT,
        "pretraining_train_family_count": PRETRAIN_FAMILY_COUNT,
        "pretraining_byte_updates_per_epoch": pretrain_bytes_per_epoch,
        "pretraining_epochs": PRETRAIN_EPOCHS,
        "pretraining_episodes_planned": expected_pretrain_episodes,
        "pretraining_byte_updates_planned": expected_pretrain_updates,
        "assistant_sft_episode_count": ASSISTANT_TRAIN_EPISODE_COUNT,
        "assistant_sft_epochs": ASSISTANT_CALIBRATION_EPOCHS,
        "assistant_sft_byte_updates_planned": sft_target_bytes,
        "expected_total_target_byte_updates": expected_pretrain_updates + sft_target_bytes,
        "capacity_budget": CAPACITY_BUDGET,
        "seed": SEED,
        "dev_families_reserved_for_generation": DEV_FAMILY_COUNT,
        "consumed_final_families_excluded": 82,
        "cpu_seconds_preparation": PREVIOUS_DATA_BUILD_CPU_SECONDS,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "parameter_updates": 0,
        "model_loads": 0,
        "generation_queries": 0,
        "dev_queries": 0,
        "reference_answer_scoring_queries": 0,
        "can_promote": False,
    }
    write_new_json(READINESS, readiness)
    attempt = {
        "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-attempt-v1",
        "status": "started",
        "readiness_sha256": sha256(READINESS),
        "source_hashes": dict(sorted(source_hashes.items())),
        "code_revision": code_revision,
        "capacity_budget": CAPACITY_BUDGET,
        "seed": SEED,
        "pretraining_epochs": PRETRAIN_EPOCHS,
        "assistant_sft_epochs": ASSISTANT_CALIBRATION_EPOCHS,
        "expected_pretraining_episodes": expected_pretrain_episodes,
        "expected_pretraining_byte_updates": expected_pretrain_updates,
        "expected_assistant_sft_byte_updates": sft_target_bytes,
        "dev_free_generation_outputs_planned": DEV_FAMILY_COUNT * len(GENERATION_ENDPOINTS),
        "parameter_updates": 0,
        "generation_queries": 0,
        "can_promote": False,
    }
    write_new_json(ATTEMPT, attempt)

    if shutil.disk_usage(CHECKPOINT_ROOT.parent).free < 512 * 1024**2:
        raise Refuse("less than 512 MiB remains for complete dialogue checkpoints")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    active_stage: str | None = None
    active_trainer: LanguageAlignmentTrainer | None = None
    pretrain_episode_count = 0
    pretrain_byte_updates = 0
    sft_episode_count = 0
    sft_byte_updates = 0
    generation_count = 0
    persisted_generation_count = 0
    restore_receipts: dict[str, dict[str, Any]] = {}
    try:
        budget.guard(reserve=GENERATION_RESERVE_SECONDS)
        model = Taiji(_model_config(), device="cpu")
        model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
        active_parameters = model.parameter_count()
        if active_parameters > CAPACITY_BUDGET:
            raise Refuse(
                f"full-turn candidate exceeds the frozen capacity budget: {active_parameters}"
            )
        pretrain_config = _pretraining_config()
        pretrainer = LanguageAlignmentTrainer(
            model,
            pretrain_corpus,
            config=pretrain_config,
            code_revision=code_revision,
        )
        active_stage = "pretraining"
        active_trainer = pretrainer

        pretrain_zero_path = CHECKPOINT_ROOT / "full_turn_zero.pt"
        zero_payload = pretrainer.checkpoint()
        _atomic_save(zero_payload, pretrain_zero_path)
        zero_restore = _independent_restore(pretrain_zero_path, "pretraining", budget)
        if zero_restore.get("checkpoint_digest") != zero_payload["checkpoint_digest"]:
            raise Refuse("full-turn zero checkpoint failed independent restore")
        restore_receipts["full_turn_zero"] = zero_restore
        sft_canary_model = Taiji.from_checkpoint(zero_payload["model"], device="cpu")
        sft_canary, sft_canary_encoder = _assistant_sft_trainer(
            sft_canary_model, assistant_corpus, code_revision
        )
        if sft_canary.model.parameter_count() > CAPACITY_BUDGET:
            raise Refuse("assistant calibration canary exceeds the frozen capacity budget")
        sft_canary_zero_path = CHECKPOINT_ROOT / "assistant_sft_canary_zero.pt"
        sft_canary_payload = sft_canary.checkpoint()
        _atomic_save(sft_canary_payload, sft_canary_zero_path)
        sft_canary_restore = _independent_restore(sft_canary_zero_path, "assistant_sft", budget)
        if sft_canary_restore.get("checkpoint_digest") != sft_canary_payload["checkpoint_digest"]:
            raise Refuse("assistant calibration canary failed independent restore")
        restore_receipts["assistant_sft_canary_zero"] = sft_canary_restore
        sft_canary_probe = LanguageAlignmentTrainer.from_checkpoint(
            sft_canary_payload, assistant_corpus
        )
        sft_canary_probe_result = sft_canary_probe.train(epochs=1, max_episodes=1)
        if (
            sft_canary_probe_result.get("status") != "completed"
            or sft_canary_probe.global_step != len(assistant_corpus.episodes[0].target_bytes)
            or len(sft_canary_probe.history) != 1
        ):
            raise Refuse("assistant calibration canary credit preflight failed")
        if sft_canary.checkpoint()["checkpoint_digest"] != sft_canary_payload["checkpoint_digest"]:
            raise Refuse("assistant calibration canary probe mutated its saved parent")
        del sft_canary, sft_canary_probe, sft_canary_model
        write_jsonl(
            progress,
            {
                "event": "assistant_sft_canary_checkpoint_and_credit_pass",
                "checkpoint_sha256": sha256(sft_canary_zero_path),
                "checkpoint_digest": sft_canary_payload["checkpoint_digest"],
                "fresh_process_restore": sft_canary_restore,
                "target_encoder_format": sft_canary_encoder.FORMAT,
                "formal_updates": 0,
            },
        )
        probe = LanguageAlignmentTrainer.from_checkpoint(zero_payload, pretrain_corpus)
        probe_result = probe.train(epochs=1, max_episodes=1)
        first_target_bytes = len(pretrain_corpus.episodes[0].target_bytes)
        if (
            probe_result.get("status") != "completed"
            or probe.global_step != first_target_bytes
            or probe.episode_count != 1
        ):
            raise Refuse("one-message full-turn learning credit preflight failed")
        if pretrainer.checkpoint()["checkpoint_digest"] != zero_payload["checkpoint_digest"]:
            raise Refuse("full-turn learning preflight mutated its saved parent")
        write_jsonl(
            progress,
            {
                "event": "full_turn_zero_restore_and_credit_pass",
                "checkpoint_sha256": sha256(pretrain_zero_path),
                "checkpoint_digest": zero_payload["checkpoint_digest"],
                "fresh_process_restore": zero_restore,
                "one_episode_probe_target_bytes": first_target_bytes,
                "formal_updates": 0,
            },
        )

        pretrain_episode_index = 0

        def on_pretrain_progress(record: dict[str, Any]) -> None:
            nonlocal pretrain_episode_index, pretrain_episode_count, pretrain_byte_updates
            pretrain_episode_index += 1
            pretrain_episode_count += 1
            pretrain_byte_updates += int(record["observations"])
            target_role = pretrain_corpus.episodes[
                (pretrain_episode_index - 1) % PRETRAIN_EPISODE_COUNT
            ].target_role
            write_jsonl(
                progress,
                {
                    "event": "full_turn_pretraining_episode",
                    "episode_index": pretrain_episode_index,
                    "epoch": int(record["epoch"]),
                    "episode_id": str(record["episode_id"]),
                    "family_id": str(record["family_id"]),
                    "target_role": target_role,
                    "target_bytes": int(record["target_bytes"]),
                    "prefix_bytes": int(record["prefix_bytes"]),
                    "observations": int(record["observations"]),
                    "response_mean_surprise": float(record["response_mean_surprise"]),
                },
            )
            if pretrain_episode_index % 128 == 0:
                print(
                    f"r2-full-turn-pretraining episode={pretrain_episode_index}/"
                    f"{expected_pretrain_episodes} updates={pretrain_byte_updates} "
                    f"cpu={budget.used():.1f}",
                    flush=True,
                )
            budget.guard(reserve=GENERATION_RESERVE_SECONDS)

        pretrain_result = pretrainer.train(
            epochs=PRETRAIN_EPOCHS,
            progress=on_pretrain_progress,
        )
        if (
            pretrain_result.get("status") != "completed"
            or pretrain_result.get("episodes") != expected_pretrain_episodes
            or pretrain_episode_count != expected_pretrain_episodes
            or pretrain_byte_updates != expected_pretrain_updates
            or pretrainer.global_step != expected_pretrain_updates
        ):
            raise Refuse("full-turn pretraining did not complete the frozen episode/update budget")

        pretrain_final_path = CHECKPOINT_ROOT / "full_turn_final.pt"
        pretrain_final_payload = pretrainer.checkpoint()
        _atomic_save(pretrain_final_payload, pretrain_final_path)
        pretrain_final_restore = _independent_restore(pretrain_final_path, "pretraining", budget)
        if (
            pretrain_final_restore.get("checkpoint_digest")
            != pretrain_final_payload["checkpoint_digest"]
        ):
            raise Refuse("full-turn final checkpoint failed independent restore")
        restore_receipts["full_turn_final"] = pretrain_final_restore
        write_jsonl(
            progress,
            {
                "event": "full_turn_final_checkpoint_restored",
                "checkpoint_sha256": sha256(pretrain_final_path),
                "checkpoint_digest": pretrain_final_payload["checkpoint_digest"],
                "global_step": pretrainer.global_step,
                "episode_count": pretrainer.episode_count,
                "fresh_process_restore": pretrain_final_restore,
            },
        )

        pretrain_model = Taiji.from_checkpoint(pretrain_final_payload["model"], device="cpu")
        if pretrain_model.parameter_count() > CAPACITY_BUDGET:
            raise Refuse("pretrained model capacity changed beyond the frozen ceiling")
        assistant_trainer, target_encoder = _assistant_sft_trainer(
            pretrain_model, assistant_corpus, code_revision
        )
        if assistant_trainer.model.parameter_count() > CAPACITY_BUDGET:
            raise Refuse("assistant-calibrated model exceeds the frozen capacity budget")
        active_stage = "assistant_sft"
        active_trainer = assistant_trainer

        sft_zero_path = CHECKPOINT_ROOT / "assistant_sft_zero.pt"
        sft_zero_payload = assistant_trainer.checkpoint()
        _atomic_save(sft_zero_payload, sft_zero_path)
        sft_zero_restore = _independent_restore(sft_zero_path, "assistant_sft", budget)
        if sft_zero_restore.get("checkpoint_digest") != sft_zero_payload["checkpoint_digest"]:
            raise Refuse("assistant calibration zero checkpoint failed independent restore")
        restore_receipts["assistant_sft_zero"] = sft_zero_restore
        sft_probe = LanguageAlignmentTrainer.from_checkpoint(sft_zero_payload, assistant_corpus)
        sft_probe_result = sft_probe.train(epochs=1, max_episodes=1)
        first_sft_target_bytes = len(assistant_corpus.episodes[0].target_bytes)
        if (
            sft_probe_result.get("status") != "completed"
            or sft_probe.global_step != first_sft_target_bytes
            or sft_probe.episode_count != 1
        ):
            raise Refuse("assistant calibration credit preflight failed")
        if (
            assistant_trainer.checkpoint()["checkpoint_digest"]
            != sft_zero_payload["checkpoint_digest"]
        ):
            raise Refuse("assistant calibration probe mutated its saved parent")
        write_jsonl(
            progress,
            {
                "event": "assistant_sft_zero_restore_and_credit_pass",
                "checkpoint_sha256": sha256(sft_zero_path),
                "checkpoint_digest": sft_zero_payload["checkpoint_digest"],
                "fresh_process_restore": sft_zero_restore,
                "one_episode_probe_target_bytes": first_sft_target_bytes,
                "formal_updates": 0,
            },
        )

        def on_sft_progress(record: dict[str, Any]) -> None:
            nonlocal sft_episode_count, sft_byte_updates
            sft_episode_count += 1
            sft_byte_updates += int(record["observations"])
            write_jsonl(
                progress,
                {
                    "event": "assistant_sft_episode",
                    "episode_index": sft_episode_count,
                    "epoch": int(record["epoch"]),
                    "episode_id": str(record["episode_id"]),
                    "family_id": str(record["family_id"]),
                    "target_bytes": int(record["target_bytes"]),
                    "prefix_bytes": int(record["prefix_bytes"]),
                    "observations": int(record["observations"]),
                    "response_mean_surprise": float(record["response_mean_surprise"]),
                },
            )
            if sft_episode_count % 64 == 0:
                print(
                    f"r2-full-turn-assistant-sft episode={sft_episode_count}/"
                    f"{ASSISTANT_TRAIN_EPISODE_COUNT} updates={sft_byte_updates} "
                    f"cpu={budget.used():.1f}",
                    flush=True,
                )
            budget.guard(reserve=GENERATION_RESERVE_SECONDS)

        sft_result = assistant_trainer.train(
            epochs=ASSISTANT_CALIBRATION_EPOCHS,
            progress=on_sft_progress,
        )
        if (
            sft_result.get("status") != "completed"
            or sft_result.get("episodes") != ASSISTANT_TRAIN_EPISODE_COUNT
            or sft_episode_count != ASSISTANT_TRAIN_EPISODE_COUNT
            or sft_byte_updates != sft_target_bytes
            or assistant_trainer.global_step != sft_target_bytes
        ):
            raise Refuse("assistant calibration did not complete the frozen budget")

        sft_final_path = CHECKPOINT_ROOT / "assistant_sft_final.pt"
        sft_final_payload = assistant_trainer.checkpoint()
        _atomic_save(sft_final_payload, sft_final_path)
        sft_final_restore = _independent_restore(sft_final_path, "assistant_sft", budget)
        if sft_final_restore.get("checkpoint_digest") != sft_final_payload["checkpoint_digest"]:
            raise Refuse("assistant calibration final checkpoint failed independent restore")
        restore_receipts["assistant_sft_final"] = sft_final_restore
        write_jsonl(
            progress,
            {
                "event": "assistant_sft_final_checkpoint_restored",
                "checkpoint_sha256": sha256(sft_final_path),
                "checkpoint_digest": sft_final_payload["checkpoint_digest"],
                "global_step": assistant_trainer.global_step,
                "episode_count": assistant_trainer.episode_count,
                "fresh_process_restore": sft_final_restore,
            },
        )

        baseline_payload = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=True)
        baseline_digest = content_digest(
            {key: value for key, value in baseline_payload.items() if key != "checkpoint_digest"}
        )
        if baseline_payload.get("checkpoint_digest") != baseline_digest or baseline_payload.get(
            "corpus_digest"
        ) != _read_json(ASSISTANT_MANIFEST).get("corpus_digest"):
            raise Refuse("existing 8-epoch comparison checkpoint provenance differs")
        baseline_model = Taiji.from_checkpoint(baseline_payload["model"], device="cpu")
        baseline_config = replace(
            LanguageAlignmentConfig(**dict(baseline_payload["config"])),
            response_plan_target_geometry="signed_hash_span",
        )
        baseline_trainer = LanguageAlignmentTrainer(
            baseline_model,
            assistant_corpus,
            config=baseline_config,
            code_revision=code_revision,
        )
        candidate_trainer = LanguageAlignmentTrainer.from_checkpoint(
            sft_final_payload, assistant_corpus
        )
        prompts = _select_dev_prompts()
        if len(prompts) != DEV_FAMILY_COUNT:
            raise Refuse("dev generation family count differs")
        endpoint_trainers = {
            "previous_8epoch_candidate": baseline_trainer,
            "full_turn_pretrained_candidate": candidate_trainer,
        }
        before_digests = {
            name: content_digest(trainer.model.checkpoint())
            for name, trainer in endpoint_trainers.items()
        }
        generation_rows: list[dict[str, Any]] = []
        progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
        try:
            for index, prompt in enumerate(prompts, 1):
                row: dict[str, Any] = {
                    "index": index,
                    "episode_id": prompt.episode_id,
                    "family_id": prompt.family_id,
                    "history_turn_count": len(prompt.history),
                    "history": [[user, assistant] for user, assistant in prompt.history],
                    "current_user_turn": prompt.user_input,
                }
                for endpoint, trainer in endpoint_trainers.items():
                    budget.guard()
                    output = _generate_readonly(trainer, prompt)
                    row[endpoint] = output
                    generation_count += 1
                    write_jsonl(
                        progress,
                        {
                            "event": "dev_free_generation_output_persisted",
                            "index": index,
                            "episode_id": prompt.episode_id,
                            "family_id": prompt.family_id,
                            "endpoint": endpoint,
                            "output": output,
                            "reference_answer_included": False,
                            "exact_response_scored": False,
                        },
                    )
                    persisted_generation_count += 1
                row["reference_answer_included"] = False
                row["exact_response_scored"] = False
                generation_rows.append(row)
                write_jsonl(
                    progress,
                    {
                        "event": "dev_family_complete",
                        "index": index,
                        "episode_id": prompt.episode_id,
                        "family_id": prompt.family_id,
                        "persisted_endpoint_count": len(endpoint_trainers),
                        "reference_answer_included": False,
                    },
                )
                if index % 5 == 0:
                    print(
                        f"r2-full-turn-generation family={index}/{len(prompts)} "
                        f"outputs={generation_count} cpu={budget.used():.1f}",
                        flush=True,
                    )
                budget.guard()
        finally:
            progress.flush()
            progress.close()

        after_digests = {
            name: content_digest(trainer.model.checkpoint())
            for name, trainer in endpoint_trainers.items()
        }
        if before_digests != after_digests:
            raise Refuse("free generation mutated a model checkpoint")
        if (
            generation_count != DEV_FAMILY_COUNT * len(endpoint_trainers)
            or persisted_generation_count != generation_count
        ):
            raise Refuse("free-generation output count differs")
        for path, expected in source_hashes.items():
            if not Path(path).is_file() or sha256(Path(path)) != expected:
                raise Refuse(f"full-turn training source changed during execution: {path}")

        generation_metrics = {
            endpoint: _summary([row[endpoint] for row in generation_rows])
            for endpoint in endpoint_trainers
        }
        generation_report = {
            "schema": "taiji-r2-daily-dialogue-full-turn-generation-v1",
            "status": "complete",
            "plan_sha256": PLAN_SHA256,
            "source_hashes": dict(sorted(source_hashes.items())),
            "input_families": len({row["family_id"] for row in generation_rows}),
            "input_episode_count": len(generation_rows),
            "generation_outputs": persisted_generation_count,
            "input_history_strata": dict(
                Counter(
                    "with_history" if row["history_turn_count"] else "without_history"
                    for row in generation_rows
                )
            ),
            "max_generation_bytes": MAX_GENERATION_BYTES,
            "constrained_decode": True,
            "sampling": False,
            "reference_answer_included": False,
            "exact_response_scored": False,
            "metrics": generation_metrics,
            "records": generation_rows,
            "can_promote": False,
        }
        write_new_json(GENERATIONS, generation_report)
        with REVIEW.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_review_markdown(generation_rows))
            stream.flush()

        cumulative_cpu = PREVIOUS_DATA_BUILD_CPU_SECONDS + budget.used()
        if cumulative_cpu > CPU_LIMIT_SECONDS:
            raise TimeoutError("R2-DIALOGUE-06 exceeded its frozen cumulative CPU limit")
        result = {
            "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-result-v1",
            "status": "complete",
            "complete": True,
            "plan_sha256": PLAN_SHA256,
            "source_hashes": dict(sorted(source_hashes.items())),
            "pretraining_loaded_corpus_digest": pretrain_corpus.digest,
            "pretraining_builder_provenance_digest": pretrain_manifest["corpus_digest"],
            "assistant_sft_train_corpus_digest": assistant_corpus.digest,
            "capacity_budget": CAPACITY_BUDGET,
            "initial_seed": SEED,
            "model_parameters_after_response_plan_readout": active_parameters,
            "full_turn_pretraining": {
                "episode_count": pretrain_episode_count,
                "episodes_per_epoch": PRETRAIN_EPISODE_COUNT,
                "epochs": PRETRAIN_EPOCHS,
                "byte_updates": pretrain_byte_updates,
                "byte_updates_per_epoch": pretrain_bytes_per_epoch,
                "final_global_step": pretrainer.global_step,
                "final_checkpoint": {
                    "path": str(pretrain_final_path.resolve()),
                    "sha256": sha256(pretrain_final_path),
                    "checkpoint_digest": pretrain_final_payload["checkpoint_digest"],
                },
            },
            "assistant_calibration": {
                "episode_count": sft_episode_count,
                "epochs": ASSISTANT_CALIBRATION_EPOCHS,
                "byte_updates": sft_byte_updates,
                "byte_updates_per_epoch": sft_target_bytes,
                "final_global_step": assistant_trainer.global_step,
                "final_checkpoint": {
                    "path": str(sft_final_path.resolve()),
                    "sha256": sha256(sft_final_path),
                    "checkpoint_digest": sft_final_payload["checkpoint_digest"],
                },
            },
            "fresh_process_restores": restore_receipts,
            "total_target_byte_updates": pretrain_byte_updates + sft_byte_updates,
            "free_generation_outputs": generation_count,
            "persisted_generation_outputs": persisted_generation_count,
            "unique_dev_families": len({row["family_id"] for row in generation_rows}),
            "generation_report_sha256": sha256(GENERATIONS),
            "generation_review_sha256": sha256(REVIEW),
            "progress_sha256": sha256(PROGRESS),
            "generation_metrics": generation_metrics,
            "dev_queries": generation_count,
            "reference_answer_scoring_queries": 0,
            "exact_match_scoring_queries": 0,
            "cpu_seconds_including_data_build": round(cumulative_cpu, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "can_promote": False,
        }
        write_new_json(RESULT, result)
        return result
    except Exception as exc:
        partial: dict[str, Any] | None = None
        if active_trainer is not None:
            partial_path = CHECKPOINT_ROOT / f"{active_stage or 'unknown'}_partial.pt"
            if not partial_path.exists():
                try:
                    payload = active_trainer.checkpoint()
                    _atomic_save(payload, partial_path)
                    partial = {
                        "path": str(partial_path.resolve()),
                        "sha256": sha256(partial_path),
                        "checkpoint_digest": payload["checkpoint_digest"],
                        "global_step": active_trainer.global_step,
                        "episode_count": active_trainer.episode_count,
                    }
                except Exception as save_error:
                    partial = {"save_error": f"{type(save_error).__name__}: {save_error}"}
        failure = {
            "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-result-v1",
            "status": "partial"
            if pretrain_byte_updates or sft_byte_updates
            else "failed_before_updates",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "source_hashes": dict(sorted(source_hashes.items()))
            if "source_hashes" in locals()
            else {},
            "active_stage": active_stage,
            "pretraining_episodes": pretrain_episode_count,
            "pretraining_byte_updates": pretrain_byte_updates,
            "assistant_sft_episodes": sft_episode_count,
            "assistant_sft_byte_updates": sft_byte_updates,
            "completed_generation_outputs": generation_count,
            "persisted_generation_outputs": locals().get("persisted_generation_count", 0),
            "partial_checkpoint": partial,
            "fresh_process_restores": restore_receipts,
            "cpu_seconds_including_data_build": round(
                PREVIOUS_DATA_BUILD_CPU_SECONDS + budget.used(), 6
            ),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "errors": [f"{type(exc).__name__}: {exc}"],
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


def main() -> int:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-worker", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stage", choices=("pretraining", "assistant_sft"))
    args = parser.parse_args()
    if args.restore_worker:
        if args.checkpoint is None or args.stage is None:
            parser.error("--restore-worker requires --checkpoint and --stage")
        report = _restore_worker(args.checkpoint, args.stage)
        print(json.dumps(report, ensure_ascii=False))
        return 0
    try:
        result = execute()
    except Exception as exc:
        print(f"R2 full-turn pretraining stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"status": result["status"], "result": str(RESULT.resolve())}, ensure_ascii=False
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
