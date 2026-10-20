"""Count safe OASST2 dialogue expansion candidates without exposing message text."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data_prep import build_taiji_r2_daily_dialogue_corpus as builder
from taiji.language_alignment import LanguageEpisode

PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY_20261021.md"
PLAN_SHA256 = "508a068880fc70ba76137d53c938b36490f91aec9425a83393a5bc5a547fa407"
RECOVERY_PLAN = (
    ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-MATERIAL-INVENTORY-RECOVERY1_20261021.md"
)
RECOVERY_PLAN_SHA256 = "89b01976a1d177df003bd36bcc4b07397e9652c2c8bddc64884424813d6835fa"
PRELIMINARY_REPORT = ROOT / "reports/taiji_r2_daily_dialogue_material_inventory_20261021.json"
PRELIMINARY_REPORT_SHA256 = "19ec1cad3d98b1802066f275c8a96224ce0fd2acb4091e284ae50094362d0280"
PRELIMINARY_CPU_SECONDS = 12.53125
RAW = ROOT / builder.SOURCE_RELATIVE_PATH
RAW_SHA256 = builder.SOURCE_SHA256
MANIFEST = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json"
MANIFEST_SHA256 = "3f5b453aaab88576afd458f7b81bdcff1f1bfbf1ffd94890cb72602270211472"
EPISODES = ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl"
EPISODES_SHA256 = "4759163516064e5d2cc2e2fb5d505c0a915736218abdecb3d61771c16569b06d"
FIRST_GENERATIONS = ROOT / "reports/taiji_r2_daily_dialogue_20261021_retry2_generations.json"
FIRST_GENERATIONS_SHA256 = "32b97f4a66d562a607967eb17ac0fbb452307cd429806f551ff3c42acbc0d02f"
RECOVERY_GENERATIONS = (
    ROOT / "reports/taiji_r2_daily_dialogue_20261021_continuation_recovery1_generations.json"
)
RECOVERY_GENERATIONS_SHA256 = "07b439f2bf536132a88e302ffc4fb0b0e13397d4cc19e4d0f5b603319cff8c19"
OUTPUT = ROOT / "reports/taiji_r2_daily_dialogue_material_inventory_recovery1_20261021.json"
CPU_LIMIT_SECONDS = 300.0
BYTE_BUCKETS = ((0, 128), (129, 256), (257, 480), (481, 1024), (1025, 1_000_000))


class Refuse(RuntimeError):
    """A source or split boundary differs from the frozen inventory plan."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise Refuse(f"expected JSON object: {path}")
    return result


def _skip_json_string(value: str, position: int) -> int:
    if position >= len(value) or value[position] != '"':
        raise Refuse("expected a JSON string while skipping a value")
    position += 1
    while position < len(value):
        char = value[position]
        if char == "\\":
            position += 2
        elif char == '"':
            return position + 1
        else:
            position += 1
    raise Refuse("unterminated JSON string while skipping a value")


def _skip_json_value(value: str, position: int) -> int:
    while position < len(value) and value[position] in " \t\r\n":
        position += 1
    if position >= len(value):
        raise Refuse("missing JSON value")
    char = value[position]
    if char == '"':
        return _skip_json_string(value, position)
    if char == "{":
        position += 1
        while True:
            while position < len(value) and value[position] in " \t\r\n,":
                position += 1
            if position >= len(value):
                raise Refuse("unterminated JSON object")
            if value[position] == "}":
                return position + 1
            position = _skip_json_string(value, position)
            while position < len(value) and value[position] in " \t\r\n":
                position += 1
            if position >= len(value) or value[position] != ":":
                raise Refuse("invalid JSON object while skipping a value")
            position = _skip_json_value(value, position + 1)
    if char == "[":
        position += 1
        while True:
            while position < len(value) and value[position] in " \t\r\n,":
                position += 1
            if position >= len(value):
                raise Refuse("unterminated JSON array")
            if value[position] == "]":
                return position + 1
            position = _skip_json_value(value, position)
    while position < len(value) and value[position] not in ",]} \t\r\n":
        position += 1
    return position


def _json_selected_fields(line: str, wanted: set[str]) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    position = 0
    while position < len(line) and line[position] in " \t\r\n":
        position += 1
    if position >= len(line) or line[position] != "{":
        raise Refuse("expected a JSON object")
    position += 1
    selected: dict[str, Any] = {}
    while True:
        while position < len(line) and line[position] in " \t\r\n,":
            position += 1
        if position >= len(line):
            raise Refuse("unterminated JSON object")
        if line[position] == "}":
            return selected
        key, position = decoder.raw_decode(line, position)
        while position < len(line) and line[position] in " \t\r\n":
            position += 1
        if position >= len(line) or line[position] != ":":
            raise Refuse("invalid JSON object key/value separator")
        position += 1
        while position < len(line) and line[position] in " \t\r\n":
            position += 1
        if key in wanted:
            field, position = decoder.raw_decode(line, position)
            selected[str(key)] = field
        else:
            position = _skip_json_value(line, position)
        if wanted.issubset(selected):
            return selected


def _episode_metadata(line: str) -> tuple[str, str]:
    metadata = _json_selected_fields(line, {"family_id", "split"})
    if not isinstance(metadata.get("family_id"), str) or metadata.get("split") not in {
        "train",
        "dev",
        "final",
    }:
        raise Refuse("episode metadata prefix is incomplete")
    return metadata["family_id"], metadata["split"]


def _family_split_map() -> tuple[dict[str, set[str]], dict[str, int]]:
    families: dict[str, set[str]] = defaultdict(set)
    episode_counts = Counter()
    with EPISODES.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                family, split = _episode_metadata(line)
            except Refuse as exc:
                raise Refuse(
                    f"missing episode metadata prefix at JSONL line {line_number}"
                ) from exc
            families[split].add(family)
            episode_counts[split] += 1
    return dict(families), dict(episode_counts)


def _length_bucket(size: int) -> str:
    for lower, upper in BYTE_BUCKETS:
        if lower <= size <= upper:
            return f"{lower}-{upper}"
    return "other"


def _episode_for(
    prompt: dict[str, Any], response: dict[str, Any], accepted: dict[str, dict[str, Any]]
) -> bool:
    history = builder._history_for(prompt, accepted)
    while True:
        episode = LanguageEpisode(
            episode_id=f"oasst2:{response['message_id']}",
            family_id=f"oasst2-tree:{response['message_tree_id']}",
            task_family="daily_dialogue",
            split="train",
            user_input=str(prompt["text"]).strip(),
            response=str(response["text"]).strip(),
            context="",
            system=builder.SYSTEM_PROMPT,
            history=history,
        )
        if len(episode.prompt_bytes) <= builder.MAX_PREFIX_BYTES:
            return True
        if not history:
            return False
        history = history[1:]


def execute() -> dict[str, Any]:
    started = time.process_time()
    if OUTPUT.exists():
        raise FileExistsError(f"material inventory report is write-once: {OUTPUT}")
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("inventory plan SHA-256 changed")
    if sha256(RECOVERY_PLAN) != RECOVERY_PLAN_SHA256:
        raise Refuse("inventory recovery plan SHA-256 changed")
    if sha256(PRELIMINARY_REPORT) != PRELIMINARY_REPORT_SHA256:
        raise Refuse("preliminary inventory report SHA-256 changed")
    for path, expected, label in (
        (RAW, RAW_SHA256, "pinned OASST2 archive"),
        (MANIFEST, MANIFEST_SHA256, "prepared corpus manifest"),
        (EPISODES, EPISODES_SHA256, "prepared episode JSONL"),
        (FIRST_GENERATIONS, FIRST_GENERATIONS_SHA256, "R2-DIALOGUE-01 generations"),
        (RECOVERY_GENERATIONS, RECOVERY_GENERATIONS_SHA256, "R2-DIALOGUE-03R1 generations"),
    ):
        if sha256(path) != expected:
            raise Refuse(f"{label} SHA-256 changed")

    manifest = _read_json(MANIFEST)
    split_families, split_episode_counts = _family_split_map()
    if split_episode_counts != {"train": 525, "dev": 62, "final": 123}:
        raise Refuse("prepared episode split counts differ")
    consumed_final = {
        str(row["family_id"])
        for report in (_read_json(FIRST_GENERATIONS), _read_json(RECOVERY_GENERATIONS))
        for row in report["records"]
    }
    expected_final = split_families.get("final", set())
    if len(consumed_final) != 82 or consumed_final != expected_final:
        raise Refuse("the existing final family set is not fully and uniquely consumed")

    current_train_families = split_families.get("train", set())
    dev_families = split_families.get("dev", set())
    final_families = split_families.get("final", set())
    accepted: dict[str, dict[str, Any]] = {}
    source_counts = Counter()
    train_raw_tree_ids: set[str] = set()
    dev_metadata_families: set[str] = set()
    final_metadata_families: set[str] = set()

    with gzip.open(RAW, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            source_counts["all_messages"] += 1
            try:
                metadata = _json_selected_fields(line, {"lang", "message_tree_id"})
            except Refuse as exc:
                raise Refuse(f"invalid source metadata at line {line_number}") from exc
            if metadata.get("lang") != "zh":
                continue
            source_counts["chinese_messages"] += 1
            tree_id = metadata.get("message_tree_id")
            if not isinstance(tree_id, str) or not tree_id:
                source_counts["missing_tree_id"] += 1
                continue
            split = builder._split_for_tree(tree_id)
            if split == "dev":
                source_counts["dev_messages_metadata_only"] += 1
                dev_metadata_families.add(f"oasst2-tree:{tree_id}")
                continue
            if split == "final":
                source_counts["final_messages_metadata_only"] += 1
                final_metadata_families.add(f"oasst2-tree:{tree_id}")
                continue

            row = json.loads(line)
            train_raw_tree_ids.add(tree_id)
            source_counts["train_messages_for_safety_filter"] += 1
            if not builder._base_accepted(row):
                source_counts["train_base_review_or_source_rejected"] += 1
                continue
            if not builder._safe_accepted(row):
                source_counts["train_safety_or_privacy_rejected"] += 1
                continue
            message_id = row.get("message_id")
            if not isinstance(message_id, str) or not message_id:
                source_counts["train_missing_message_id"] += 1
                continue
            accepted[message_id] = row

    prompters: dict[str, dict[str, Any]] = {}
    train_role_counts = Counter()
    accepted_text_bytes_by_role = Counter()
    accepted_message_lengths = Counter()
    for message_id, row in accepted.items():
        role = str(row.get("role") or "unknown")
        text = str(row["text"]).strip()
        size = len(text.encode("utf-8"))
        train_role_counts[role] += 1
        accepted_text_bytes_by_role[role] += size
        accepted_message_lengths[f"{role}/{_length_bucket(size)}"] += 1
        if (
            role == "prompter"
            and size <= builder.MAX_CURRENT_USER_BYTES
            and not builder._has_reserved_marker(text)
        ):
            prompters[message_id] = row

    assistant_rank_counts = Counter()
    assistant_reply_length_counts = Counter()
    candidates_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted.values():
        if row.get("role") != "assistant":
            continue
        rank = row.get("rank")
        rank_key = "unranked" if rank is None else (str(rank) if rank in (0, 1, 2) else "other")
        assistant_rank_counts[rank_key] += 1
        text = str(row["text"]).strip()
        size = len(text.encode("utf-8"))
        assistant_reply_length_counts[_length_bucket(size)] += 1
        if rank not in (None, 0, 1) or not text or size > builder.MAX_TARGET_BYTES:
            continue
        if builder._has_reserved_marker(text):
            continue
        prompt_id = str(row.get("parent_id") or "")
        prompt = prompters.get(prompt_id)
        if prompt is None:
            continue
        if prompt.get("message_tree_id") != row.get("message_tree_id"):
            continue
        if _episode_for(prompt, row, accepted):
            candidates_by_prompt[prompt_id].append(row)

    all_rank01_candidates = sum(len(rows) for rows in candidates_by_prompt.values())
    one_best_per_prompt = len(candidates_by_prompt)
    current_selected_train_episodes = split_episode_counts["train"]
    if one_best_per_prompt != current_selected_train_episodes:
        source_counts["builder_count_reconciliation_mismatch"] = abs(
            one_best_per_prompt - current_selected_train_episodes
        )

    current_accepted_train_families = {
        f"oasst2-tree:{row['message_tree_id']}" for row in accepted.values()
    }
    unqueried_dev_in_manifest = len(dev_families & dev_metadata_families)
    consumed_final_in_manifest = len(final_families & final_metadata_families)
    if unqueried_dev_in_manifest != len(dev_families) or consumed_final_in_manifest != len(
        final_families
    ):
        raise Refuse("family metadata does not cover the current split families")

    current_cpu = time.process_time() - started
    cumulative_cpu = PRELIMINARY_CPU_SECONDS + current_cpu
    if cumulative_cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError(
            f"material inventory CPU exceeded: {cumulative_cpu:.3f}/{CPU_LIMIT_SECONDS:.3f}"
        )
    source_paths = (
        PLAN,
        RECOVERY_PLAN,
        PRELIMINARY_REPORT,
        RAW,
        MANIFEST,
        EPISODES,
        FIRST_GENERATIONS,
        RECOVERY_GENERATIONS,
        Path(__file__).resolve(),
    )
    report = {
        "schema": "taiji-r2-daily-dialogue-material-inventory-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "recovery_plan_sha256": RECOVERY_PLAN_SHA256,
        "supersedes_report_sha256": PRELIMINARY_REPORT_SHA256,
        "superseded_report_source_binding_valid": False,
        "superseded_attempt_may_have_decoded_dev_final_text": True,
        "source_hashes": {str(path.resolve()): sha256(path) for path in source_paths},
        "source_counts": dict(source_counts),
        "current_manifest": {
            "source_chinese_message_count": manifest["source_chinese_message_count"],
            "accepted_chinese_message_count": manifest["accepted_chinese_message_count"],
            "selected_episodes": split_episode_counts,
            "selected_families": manifest["split_family_counts"],
            "corpus_digest": manifest["corpus_digest"],
        },
        "train_family_inventory": {
            "current_episode_train_families": len(current_train_families),
            "safe_accepted_train_families_with_messages": len(current_accepted_train_families),
            "raw_train_families_with_chinese_messages": len(train_raw_tree_ids),
            "safe_accepted_train_messages_by_role": dict(sorted(train_role_counts.items())),
            "safe_accepted_train_text_bytes_by_role": dict(
                sorted(accepted_text_bytes_by_role.items())
            ),
            "safe_accepted_train_message_length_buckets": dict(
                sorted(accepted_message_lengths.items())
            ),
            "assistant_rank_counts": dict(sorted(assistant_rank_counts.items())),
            "assistant_reply_length_buckets": dict(sorted(assistant_reply_length_counts.items())),
            "one_best_rank01_response_per_prompt": one_best_per_prompt,
            "all_rank01_response_candidates": all_rank01_candidates,
            "additional_rank01_alternatives_over_one_best": all_rank01_candidates
            - one_best_per_prompt,
            "current_selected_train_episodes": current_selected_train_episodes,
            "builder_reconciliation_matches": one_best_per_prompt
            == current_selected_train_episodes,
        },
        "unqueried_evaluation_inventory": {
            "dev_family_count_unqueried": len(dev_families),
            "dev_family_metadata_counted": unqueried_dev_in_manifest,
            "dev_message_text_inspected": False,
            "consumed_final_family_count_excluded": len(consumed_final),
            "final_family_metadata_counted": consumed_final_in_manifest,
            "final_message_text_inspected": False,
            "fresh_generation_candidate_face": "the 39 existing dev families remain untouched and may be reserved as a one-time free-generation surface in a later frozen plan",
        },
        "safety_and_privacy": {
            "train_accepted_filters_reused": True,
            "raw_message_text_emitted": False,
            "message_or_user_ids_emitted": False,
            "dev_final_text_used_for_scoring": False,
        },
        "parameter_updates": 0,
        "model_loads": 0,
        "generation_queries": 0,
        "dev_queries": 0,
        "final_queries": 0,
        "checkpoint_writes": 0,
        "preliminary_attempt_cpu_seconds": PRELIMINARY_CPU_SECONDS,
        "recovery_attempt_cpu_seconds": round(current_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative_cpu, 6),
        "cpu_seconds": round(cumulative_cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
    }
    if sha256(RAW) != RAW_SHA256:
        raise Refuse("pinned OASST2 archive changed during inventory")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
    return report


if __name__ == "__main__":
    result = execute()
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": str(OUTPUT.resolve()),
                "cpu_seconds": result["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
