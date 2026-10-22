"""Build role-aware full-turn pretraining episodes from safe OASST2 train trees."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep import build_taiji_r2_daily_dialogue_corpus as base_builder
from scripts.data_prep.audit_taiji_r2_daily_dialogue_material_inventory import (
    _json_selected_fields,
)
from taiji.language_alignment import LanguageEpisode, LanguageEpisodeCorpus

PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-FULL-TURN-PRETRAINING_20261021.md"
PLAN_SHA256 = "d043f1e18e17057c3546e28c6dc6b816034676f7e0f4ba43d511626d8cec9521"
RAW = ROOT / base_builder.SOURCE_RELATIVE_PATH
RAW_SHA256 = base_builder.SOURCE_SHA256
SPLIT_MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json"
SPLIT_MANIFEST_SHA256 = "3f5b453aaab88576afd458f7b81bdcff1f1bfbf1ffd94890cb72602270211472"
TRAIN_EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
TRAIN_EPISODES_SHA256 = "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d"
INVENTORY = ROOT / "reports/taiji_r2_daily_dialogue_material_inventory_recovery1_20261021.json"
INVENTORY_SHA256 = "6bf34320c2963236f916fa920eb19911bf369ea8dc5ccdb120a81b1a2d3a5dc8"
OUTPUT = ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_episodes.jsonl"
MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_dialogue_full_turn_pretraining_manifest.json"
REPORT = ROOT / "reports/taiji_r2_daily_dialogue_full_turn_pretraining_data_readiness_20261021.json"
CPU_LIMIT_SECONDS = 300.0
MAX_HISTORY_PAIRS = 3
MAX_PREFIX_BYTES = 4_096
SYSTEM_PROMPT = base_builder.SYSTEM_PROMPT


class Refuse(RuntimeError):
    """The frozen source, split, or full-turn corpus contract is invalid."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Refuse(f"expected JSON object: {path}")
    return value


def _hash_id(value: str, prefix: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _episode_for_message(
    row: dict[str, Any], accepted: dict[str, dict[str, Any]]
) -> LanguageEpisode:
    role = str(row.get("role"))
    message_id = str(row["message_id"])
    tree_id = str(row["message_tree_id"])
    text = str(row["text"]).strip()
    target_role = "assistant" if role == "assistant" else "user"
    history = base_builder._history_for(row, accepted)
    user_input = ""

    if target_role == "assistant":
        prompt = accepted.get(str(row.get("parent_id") or ""))
        if (
            prompt is not None
            and prompt.get("role") == "prompter"
            and prompt.get("message_tree_id") == row.get("message_tree_id")
            and len(str(prompt["text"]).strip().encode("utf-8"))
            <= base_builder.MAX_CURRENT_USER_BYTES
        ):
            user_input = str(prompt["text"]).strip()
            history = base_builder._history_for(prompt, accepted)
        else:
            user_input = ""
            history = ()

    while True:
        episode = LanguageEpisode(
            episode_id=_hash_id(message_id, "oasst2-turn"),
            family_id=_hash_id(tree_id, "oasst2-tree"),
            task_family="full_turn_pretraining",
            split="train",
            user_input=user_input,
            response=text,
            context="",
            system=SYSTEM_PROMPT,
            history=history,
            target_role=target_role,
        )
        if len(episode.prompt_bytes) <= MAX_PREFIX_BYTES:
            return episode
        if not history:
            raise Refuse(f"single-message prefix exceeds the limit for {episode.episode_id}")
        history = history[1:]


def build() -> tuple[LanguageEpisodeCorpus, dict[str, Any]]:
    for path, expected, label in (
        (PLAN, PLAN_SHA256, "full-turn pretraining plan"),
        (RAW, RAW_SHA256, "pinned OASST2 archive"),
        (SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256, "prepared split manifest"),
        (TRAIN_EPISODES, TRAIN_EPISODES_SHA256, "prepared split episode JSONL"),
        (INVENTORY, INVENTORY_SHA256, "verified material inventory"),
    ):
        if sha256(path) != expected:
            raise Refuse(f"{label} SHA-256 changed")

    inventory = _read_json(INVENTORY)
    if (
        inventory.get("status") != "complete"
        or inventory.get("model_loads") != 0
        or inventory.get("generation_queries") != 0
        or inventory.get("unqueried_evaluation_inventory", {}).get("dev_message_text_inspected")
        is not False
        or inventory.get("unqueried_evaluation_inventory", {}).get("final_message_text_inspected")
        is not False
    ):
        raise Refuse("the material inventory does not preserve the frozen blind boundary")

    manifest = _read_json(SPLIT_MANIFEST)
    if manifest.get(
        "corpus_digest"
    ) != "bec58e46045aaccddb7811fb9b8f755291dd70845ad888bfe438f8678e589af6" or manifest.get(
        "split_episode_counts"
    ) != {
        "train": 525,
        "dev": 62,
        "final": 123,
        "retention": 0,
    }:
        raise Refuse("current OASST2 split manifest differs from the frozen inventory")
    family_split: dict[str, str] = {}
    with TRAIN_EPISODES.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            metadata = _json_selected_fields(line, {"family_id", "split"})
            family = str(metadata["family_id"])
            split = str(metadata["split"])
            prior = family_split.setdefault(family, split)
            if prior != split:
                raise Refuse(f"conversation family crosses splits at line {line_number}")

    dev_family_count = sum(split == "dev" for split in family_split.values())
    final_family_count = sum(split == "final" for split in family_split.values())
    if dev_family_count != 39 or final_family_count != 82:
        raise Refuse("existing dev/final family counts differ from the inventory contract")

    accepted: dict[str, dict[str, Any]] = {}
    rejection_counts = Counter()
    source_chinese_messages = 0
    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for _, line in enumerate(stream, 1):
            metadata = _json_selected_fields(line, {"lang", "message_tree_id"})
            if metadata.get("lang") != "zh":
                continue
            source_chinese_messages += 1
            tree_id = metadata.get("message_tree_id")
            if not isinstance(tree_id, str) or not tree_id:
                rejection_counts["missing_tree_id"] += 1
                continue
            if base_builder._split_for_tree(tree_id) != "train":
                continue
            row = json.loads(line)
            if not base_builder._base_accepted(row):
                rejection_counts["base_review_or_source_filter"] += 1
                continue
            if not base_builder._safe_accepted(row):
                rejection_counts["safety_or_privacy_filter"] += 1
                continue
            if row.get("role") not in {"prompter", "assistant"}:
                rejection_counts["unsupported_role"] += 1
                continue
            message_id = row.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                rejection_counts["missing_message_id"] += 1
                continue
            accepted[message_id] = row

    episodes = tuple(
        _episode_for_message(row, accepted)
        for _, row in sorted(
            accepted.items(),
            key=lambda item: (
                _hash_id(str(item[1]["message_tree_id"]), "family"),
                _hash_id(str(item[0]), "episode"),
            ),
        )
    )
    role_counts = Counter(episode.target_role for episode in episodes)
    text_bytes = Counter()
    target_bytes = Counter()
    for episode in episodes:
        text_bytes[episode.target_role] += len(episode.response.encode("utf-8"))
        target_bytes[episode.target_role] += len(episode.target_bytes)
    family_count = len({episode.family_id for episode in episodes})
    if (
        len(episodes) != 3762
        or family_count != 508
        or text_bytes["assistant"] + text_bytes["user"] != 1_780_881
        or role_counts != {"assistant": 2244, "user": 1518}
    ):
        raise Refuse("full-turn source count differs from the frozen material inventory")

    corpus = LanguageEpisodeCorpus(
        episodes,
        ((str(RAW.resolve()), RAW_SHA256),),
        require_eval_splits=False,
    )
    source_paths = (PLAN, RAW, SPLIT_MANIFEST, TRAIN_EPISODES, INVENTORY, Path(__file__).resolve())
    corpus_manifest = {
        "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-corpus-v1",
        "status": "ready",
        "plan_sha256": PLAN_SHA256,
        "source_repository": base_builder.SOURCE_REPOSITORY,
        "source_revision": base_builder.SOURCE_REVISION,
        "source_license": "Apache-2.0 (pinned OASST2 dataset metadata)",
        "source_sha256": RAW_SHA256,
        "source_hashes": {str(path.resolve()): sha256(path) for path in source_paths},
        "source_chinese_message_count": source_chinese_messages,
        "split_rule": "existing OASST2 message_tree_id hash split; only train families included",
        "excluded_dev_family_count": dev_family_count,
        "excluded_consumed_final_family_count": final_family_count,
        "train_family_count": family_count,
        "episode_count": len(episodes),
        "target_role_counts": dict(sorted(role_counts.items())),
        "target_text_bytes_by_role": dict(sorted(text_bytes.items())),
        "target_bytes_with_end_markers_by_role": dict(sorted(target_bytes.items())),
        "prefix_byte_limit": MAX_PREFIX_BYTES,
        "history_pair_limit": MAX_HISTORY_PAIRS,
        "filters": {
            "base": "lang=zh, ready_for_export, nondeleted, positive review, nonsynthetic",
            "safety": list(base_builder.FILTERED_FLAGS),
            "email_or_phone_pattern": base_builder.EMAIL_OR_PHONE.pattern,
            "reserved_marker_filter": True,
            "dev_final_content_loaded": False,
            "raw_text_emitted_in_manifest": False,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "corpus_digest": corpus.digest,
        "episode_jsonl_sha256": None,
    }
    return corpus, corpus_manifest


def write() -> dict[str, Any]:
    started = time.process_time()
    if OUTPUT.exists() or MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("full-turn pretraining corpus artifacts are write-once")
    corpus, manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        for episode in corpus.episodes:
            stream.write(json.dumps(episode.to_payload(), ensure_ascii=False, sort_keys=True))
            stream.write("\n")
        stream.flush()
    manifest["episode_jsonl_sha256"] = sha256(OUTPUT)
    manifest["prepared_cpu_seconds"] = round(time.process_time() - started, 6)
    manifest["prepared_corpus_digest"] = corpus.digest
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
    report = {
        "schema": "taiji-r2-daily-dialogue-full-turn-pretraining-data-readiness-v1",
        "status": "ready",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": manifest["source_hashes"],
        "manifest_sha256": sha256(MANIFEST),
        "episode_jsonl_sha256": manifest["episode_jsonl_sha256"],
        "corpus_digest": corpus.digest,
        "episode_count": len(corpus.episodes),
        "family_count": manifest["train_family_count"],
        "target_role_counts": manifest["target_role_counts"],
        "target_bytes_with_end_markers_by_role": manifest["target_bytes_with_end_markers_by_role"],
        "prepared_cpu_seconds": manifest["prepared_cpu_seconds"],
        "model_loads": 0,
        "parameter_updates": 0,
        "generation_queries": 0,
        "dev_final_content_loaded": False,
        "can_promote": False,
    }
    with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
    return report


if __name__ == "__main__":
    result = write()
    print(
        json.dumps(
            {
                "status": result["status"],
                "episode_count": result["episode_count"],
                "report": str(REPORT.resolve()),
            },
            ensure_ascii=False,
        )
    )
