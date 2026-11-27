"""One train-only V3 update to measure complete-answer CPU cost and recovery."""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import time
import unicodedata

import torch

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.run_taiji_r2_daily_dialogue_full_turn_pretraining import (
    _json_selected_fields,
)
from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3
from taiji.language_alignment import LanguageEpisode

PARTITION = ROOT / "reports/taiji_n1_plan175_metadata_partition_20261026.json"
MIGRATION = ROOT / "reports/taiji_n1_plan175_v3_parent_migration_20261026.json"
RAW_SOURCE = ROOT / "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
OLD_EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
PARENT = ROOT / "data/checkpoints/n1_plan175_v3_parent_20261026/tf_seed169/zero.pt"
OUTPUT = ROOT / "reports/taiji_n1_plan175_v3_train_only_calibration_20261026.json"
CHECKPOINT = ROOT / "data/checkpoints/n1_plan175_v3_calibration_20261026/tf_seed169/step0001.pt"
MAX_TARGET_BYTES = 2048
MAX_PROMPT_BYTES = 4096


def text_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def build_episode(candidate: dict, chain: list[dict]) -> LanguageEpisode:
    if len(chain) < 2 or len(chain) % 2:
        raise ValueError("dialogue chain must contain complete user/assistant pairs")
    if chain[-1]["role"] != "assistant" or chain[-2]["role"] not in {"user", "prompter"}:
        raise ValueError("candidate target is not an assistant reply to a user turn")
    history = tuple(
        (chain[index]["text"], chain[index + 1]["text"]) for index in range(0, len(chain) - 2, 2)
    )
    return LanguageEpisode(
        episode_id=candidate["target_id"],
        family_id=candidate["family"],
        task_family="dialogue_pretraining",
        split="train",
        user_input=chain[-2]["text"],
        response=chain[-1]["text"],
        history=history,
    )


def load_train_episodes() -> tuple[list[tuple[dict, LanguageEpisode]], dict]:
    if digest(RAW_SOURCE) != "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751":
        raise ValueError("frozen OASST2 bytes changed")
    if digest(OLD_EPISODES) != "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d":
        raise ValueError("frozen old episode bytes changed")
    partition = json.loads(PARTITION.read_text(encoding="utf-8"))
    if partition["status"] != "candidate_partition_not_quality_approved":
        raise ValueError("unexpected N175 partition status")
    protected = set(partition["holdout_families"]) | set(partition["protected_families"])
    rows = {}
    fields = {"message_id", "parent_id", "role", "lang", "deleted"}
    with gzip.open(RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = _json_selected_fields(line, fields)
            identity = row.get("message_id")
            if not identity or identity in rows:
                raise ValueError("source contains a missing or duplicate message ID")
            rows[identity] = row
    chains, needed = {}, set()
    for candidate in partition["train_candidates"]:
        identity, chain, seen = candidate["target_id"], [], set()
        while identity is not None:
            if identity not in rows or identity in seen:
                raise ValueError("training candidate has a broken or cyclic ancestry chain")
            seen.add(identity)
            chain.append(identity)
            identity = rows[identity].get("parent_id")
        chain.reverse()
        if chain[0] in protected or chain[0] != candidate["family"]:
            raise ValueError("training candidate crosses a protected root family")
        chains[candidate["target_id"]] = chain
        needed.update(chain)
    with gzip.open(RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            identity = _json_selected_fields(line, {"message_id"}).get("message_id")
            if identity in needed:
                rows[identity]["text"] = _json_selected_fields(line, {"text"}).get("text")

    counts = collections.Counter()
    episodes, seen_questions = [], set()
    for candidate in partition["train_candidates"]:
        chain = [rows[identity] for identity in chains[candidate["target_id"]]]
        valid = True
        for index, row in enumerate(chain):
            role_ok = row["role"] in ({"user", "prompter"} if index % 2 == 0 else {"assistant"})
            if (
                row.get("deleted")
                or row.get("lang") != "zh"
                or not role_ok
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
            ):
                valid = False
                break
        if not valid:
            counts["invalid_chinese_context"] += 1
            continue
        try:
            episode = build_episode(candidate, chain)
        except (TypeError, ValueError):
            counts["reserved_or_invalid_text"] += 1
            continue
        if len(episode.target_bytes) > MAX_TARGET_BYTES:
            counts["complete_target_over_2048_bytes"] += 1
            continue
        if len(episode.prompt_bytes) > MAX_PROMPT_BYTES:
            counts["serialized_prompt_over_4096_bytes"] += 1
            continue
        key = tuple(text_key(row["text"]) for row in chain[:-1])
        if key in seen_questions:
            counts["normalized_duplicate_context"] += 1
            continue
        seen_questions.add(key)
        episodes.append((candidate, episode))
        counts["eligible_complete_answer"] += 1
    if len({episode.family_id for _, episode in episodes}) != len(episodes):
        counts["eligible_family_count"] = len({episode.family_id for _, episode in episodes})
    else:
        counts["eligible_family_count"] = len(episodes)
    return episodes, {
        "counts": dict(counts),
        "train_text_nodes_parsed": len(needed),
        "heldout_text_nodes_parsed": 0,
        "partition_sha256": digest(PARTITION),
        "raw_source_sha256": digest(RAW_SOURCE),
        "old_episode_source_sha256": digest(OLD_EPISODES),
    }


def choose_calibration_batch(
    episodes: list[tuple[dict, LanguageEpisode]],
) -> list[tuple[dict, LanguageEpisode]]:
    selected, families = [], set()
    for threshold in (128, 512, 1024, 2048):
        options = [
            row
            for row in episodes
            if row[0]["family"] not in families and len(row[1].target_bytes) <= MAX_TARGET_BYTES
        ]
        if not options:
            break
        choice = min(
            options,
            key=lambda row: (
                abs(len(row[1].target_bytes) - threshold),
                hashlib.sha256(row[0]["target_id"].encode()).hexdigest(),
            ),
        )
        selected.append(choice)
        families.add(choice[0]["family"])
    if len(selected) < 3:
        raise ValueError(
            "eligible train corpus is too small for the fixed length-stratified calibration"
        )
    return selected


def main() -> None:
    if OUTPUT.exists() or CHECKPOINT.exists():
        raise FileExistsError("the one-shot calibration output already exists")
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    if migration["status"] != "three_seed_parent_migrated_no_training":
        raise ValueError("three complete V3 parents are required before calibration")
    seed_parent = next(row for row in migration["parents"] if row["seed"] == 169)
    if digest(PARENT) != seed_parent["v3_checkpoint_sha256"]:
        raise ValueError("V3 parent checkpoint SHA differs from migration receipt")
    torch.set_num_threads(1)
    wall_started, cpu_started = time.monotonic(), time.process_time()
    episodes, data = load_train_episodes()
    batch = choose_calibration_batch(episodes)
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    trainer, _, cursor = ConditionalLanguageTrainerV3.from_checkpoint(payload)
    if trainer.steps != 5000 or trainer.target_byte_exposures != 2159160:
        raise ValueError("migrated parent is not the complete N1-174 seed169 endpoint")
    examples = [(episode.prompt_bytes, episode.target_bytes) for _, episode in batch]
    batch_target_bytes = sum(len(target) for _, target in examples)
    update_cpu_started, update_wall_started = time.process_time(), time.monotonic()
    metrics = trainer.update(examples)
    update_cpu = time.process_time() - update_cpu_started
    update_wall = time.monotonic() - update_wall_started
    if trainer.target_byte_exposures - 2159160 != batch_target_bytes:
        raise ValueError("target exposure delta differs from this calibration batch")
    state = trainer.model.new_state(
        scope_id="n1-175", session_id="seed-169", native_state_ref="n1-175:calibration"
    )
    trainer.save(
        CHECKPOINT,
        state,
        cursor={"epoch": 0, "batch": 1, "dataset_partition_sha256": data["partition_sha256"]},
    )
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    checkpoint_sha = digest(CHECKPOINT)
    if (
        saved["checkpoint_digest"]
        != trainer.checkpoint(
            state,
            cursor={"epoch": 0, "batch": 1, "dataset_partition_sha256": data["partition_sha256"]},
        )["checkpoint_digest"]
    ):
        raise ValueError("calibration checkpoint did not reproduce after durable save")
    elapsed_cpu, elapsed_wall = time.process_time() - cpu_started, time.monotonic() - wall_started
    result = {
        "schema": "taiji-n1-plan175-v3-train-only-calibration-v2",
        "status": "one_train_only_update_complete",
        "v3_candidate_sha256": digest(ROOT / "taiji/conditional_language_v3.py"),
        "migration_report_sha256": digest(MIGRATION),
        "parent_checkpoint_sha256": seed_parent["v3_checkpoint_sha256"],
        "parent_checkpoint_digest": payload["checkpoint_digest"],
        "data": data,
        "complete_train_corpus_candidates": len(episodes),
        "complete_train_corpus_families": len({episode.family_id for _, episode in episodes}),
        "calibration_batch": [
            {
                "target_id": candidate["target_id"],
                "family": candidate["family"],
                "prompt_bytes": len(episode.prompt_bytes),
                "target_bytes": len(episode.target_bytes),
            }
            for candidate, episode in batch
        ],
        "update_metrics": metrics,
        "batch_target_bytes": batch_target_bytes,
        "update_process_cpu_seconds": update_cpu,
        "update_wall_seconds": update_wall,
        "elapsed_process_cpu_seconds": elapsed_cpu,
        "elapsed_wall_seconds": elapsed_wall,
        "target_bytes_per_process_cpu_second": batch_target_bytes / update_cpu,
        "cumulative_target_exposures_before": 2159160,
        "cumulative_target_exposures_after": metrics["target_byte_exposures"],
        "calibration_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "calibration_checkpoint_sha256": checkpoint_sha,
        "calibration_checkpoint_digest": saved["checkpoint_digest"],
        "heldout_text_parsed": 0,
        "development_reads": 0,
        "can_promote": False,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "complete_train_corpus_candidates",
                    "complete_train_corpus_families",
                    "elapsed_process_cpu_seconds",
                    "elapsed_wall_seconds",
                    "target_bytes_per_process_cpu_second",
                    "heldout_text_parsed",
                    "can_promote",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
