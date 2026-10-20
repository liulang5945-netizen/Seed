"""Build a family-disjoint Chinese dialogue corpus from the pinned OASST2 export.

The source archive stays local under data/conversation/OASST2. This builder
emits only accepted Chinese assistant turns, their clean prompt/history, stable
tree-level splits, and a content-addressed manifest; it never emits source user
IDs or raw rejected rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.language_alignment import LanguageEpisode, LanguageEpisodeCorpus  # noqa: E402

SOURCE_REPOSITORY = "OpenAssistant/oasst2"
SOURCE_REVISION = "179dd21fc55192153d94adb0e0ce8f69e222bf75"
SOURCE_SHA256 = "a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
SOURCE_RELATIVE_PATH = "data/conversation/OASST2/2023-11-05_oasst2_ready.messages.jsonl.gz"
FILTERED_FLAGS = (
    "pii",
    "spam",
    "lang_mismatch",
    "not_appropriate",
    "hate_speech",
    "sexual_content",
)
SYSTEM_PROMPT = "你在进行自然的中文对话。结合上文简洁回应；不确定时坦诚说明。"
MAX_CURRENT_USER_BYTES = 1_024
MAX_TARGET_BYTES = 480
MAX_HISTORY_PAIRS = 3
MAX_PREFIX_BYTES = 4_096
SPLIT_SALT = "oasst2-daily-tree-split-v1"
EMAIL_OR_PHONE = re.compile(r"(?i)([\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:\+?\d[\d ()-]{7,}\d))")
RESERVED_MARKERS = (
    "<|system|>",
    "<|policy|>",
    "<|context|>",
    "<|history_user|>",
    "<|history_assistant|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|none|>",
)


class CorpusRefusal(RuntimeError):
    """The pinned source or corpus-construction contract is not satisfied."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _flag_value(row: dict[str, Any], name: str) -> float:
    labels = row.get("labels") or {}
    value = labels.get(name, 0.0) if isinstance(labels, dict) else 0.0
    if isinstance(value, dict):
        value = value.get("value", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _base_accepted(row: dict[str, Any]) -> bool:
    return bool(
        row.get("lang") == "zh"
        and row.get("tree_state") == "ready_for_export"
        and row.get("deleted") is False
        and row.get("review_result") is True
        and row.get("synthetic") is False
        and isinstance(row.get("text"), str)
        and row["text"].strip()
    )


def _safe_accepted(row: dict[str, Any]) -> bool:
    text = str(row.get("text", ""))
    return (
        not any(_flag_value(row, name) > 0 for name in FILTERED_FLAGS)
        and not bool(EMAIL_OR_PHONE.search(text))
        and not _has_reserved_marker(text)
    )


def _split_for_tree(tree_id: str) -> str:
    slot = int(hashlib.sha256(f"{SPLIT_SALT}\0{tree_id}".encode()).hexdigest()[:8], 16) % 10_000
    if slot < 7_000:
        return "train"
    if slot < 8_000:
        return "dev"
    return "final"


def _history_for(
    prompt: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    current_prompt = prompt
    visited = {str(prompt.get("message_id"))}
    while len(pairs) < MAX_HISTORY_PAIRS:
        previous_assistant_id = current_prompt.get("parent_id")
        previous_assistant = by_id.get(str(previous_assistant_id))
        if (
            previous_assistant is None
            or previous_assistant.get("role") != "assistant"
            or previous_assistant.get("message_tree_id") != prompt.get("message_tree_id")
            or str(previous_assistant.get("message_id")) in visited
        ):
            break
        previous_prompt = by_id.get(str(previous_assistant.get("parent_id")))
        if (
            previous_prompt is None
            or previous_prompt.get("role") != "prompter"
            or previous_prompt.get("message_tree_id") != prompt.get("message_tree_id")
            or str(previous_prompt.get("message_id")) in visited
        ):
            break
        visited.add(str(previous_assistant["message_id"]))
        visited.add(str(previous_prompt["message_id"]))
        pairs.append(
            (str(previous_prompt["text"]).strip(), str(previous_assistant["text"]).strip())
        )
        current_prompt = previous_prompt
    return tuple(reversed(pairs))


def _has_reserved_marker(*values: str) -> bool:
    return any(marker in value for value in values for marker in RESERVED_MARKERS)


def build(source: Path) -> tuple[LanguageEpisodeCorpus, dict[str, Any]]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.relative_to(ROOT).as_posix() != SOURCE_RELATIVE_PATH:
        raise CorpusRefusal("OASST2 archive path differs from the frozen project data path")
    archive_sha = sha256(source)
    if archive_sha != SOURCE_SHA256:
        raise CorpusRefusal("OASST2 archive SHA-256 differs from the pinned source")

    all_rows = 0
    chinese_rows: list[dict[str, Any]] = []
    rejected = Counter()
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusRefusal(f"invalid source JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise CorpusRefusal(f"source row {line_number} is not an object")
            all_rows += 1
            if row.get("lang") == "zh":
                chinese_rows.append(row)

    accepted: dict[str, dict[str, Any]] = {}
    for row in chinese_rows:
        message_id = row.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            rejected["missing_message_id"] += 1
            continue
        if not _base_accepted(row):
            rejected["base_review_or_source_filter"] += 1
            continue
        if not _safe_accepted(row):
            rejected["privacy_or_safety_filter"] += 1
            for name in FILTERED_FLAGS:
                if _flag_value(row, name) > 0:
                    rejected[f"positive_{name}"] += 1
            if EMAIL_OR_PHONE.search(str(row.get("text", ""))):
                rejected["email_or_phone_pattern"] += 1
            if _has_reserved_marker(str(row.get("text", ""))):
                rejected["reserved_marker_filter"] += 1
            continue
        accepted[message_id] = row

    prompter_rows = [
        row
        for row in accepted.values()
        if row.get("role") == "prompter"
        and len(str(row["text"]).strip().encode("utf-8")) <= MAX_CURRENT_USER_BYTES
        and not _has_reserved_marker(str(row["text"]))
    ]
    prompter_by_id = {str(row["message_id"]): row for row in prompter_rows}
    candidates_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted.values():
        if row.get("role") != "assistant" or row.get("rank") not in (None, 0, 1):
            rejected["assistant_rank_filter"] += 1
            continue
        target = str(row["text"]).strip()
        if not target or len(target.encode("utf-8")) > MAX_TARGET_BYTES:
            rejected["target_length_filter"] += 1
            continue
        if _has_reserved_marker(target):
            rejected["reserved_marker_filter"] += 1
            continue
        prompt_id = str(row.get("parent_id") or "")
        if prompt_id not in prompter_by_id:
            rejected["missing_or_rejected_prompt_parent"] += 1
            continue
        candidates_by_prompt[prompt_id].append(row)

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for prompt_id, responses in candidates_by_prompt.items():
        prompt = prompter_by_id[prompt_id]
        response = min(
            responses,
            key=lambda row: (
                int(row["rank"]) if row.get("rank") is not None else 99,
                str(row["message_id"]),
            ),
        )
        selected.append((prompt, response))

    episodes: list[LanguageEpisode] = []
    for prompt, response in selected:
        tree_id = str(response["message_tree_id"])
        prompt_text = str(prompt["text"]).strip()
        response_text = str(response["text"]).strip()
        history = _history_for(prompt, accepted)
        while history:
            episode = LanguageEpisode(
                episode_id=f"oasst2:{response['message_id']}",
                family_id=f"oasst2-tree:{tree_id}",
                task_family="daily_dialogue",
                split=_split_for_tree(tree_id),
                user_input=prompt_text,
                response=response_text,
                context="",
                system=SYSTEM_PROMPT,
                history=history,
            )
            if len(episode.prompt_bytes) <= MAX_PREFIX_BYTES:
                break
            history = history[1:]
        else:
            episode = LanguageEpisode(
                episode_id=f"oasst2:{response['message_id']}",
                family_id=f"oasst2-tree:{tree_id}",
                task_family="daily_dialogue",
                split=_split_for_tree(tree_id),
                user_input=prompt_text,
                response=response_text,
                context="",
                system=SYSTEM_PROMPT,
                history=(),
            )
        if len(episode.prompt_bytes) > MAX_PREFIX_BYTES:
            rejected["prefix_length_filter"] += 1
            continue
        episodes.append(episode)

    source_files = ((str(source), archive_sha),)
    corpus = LanguageEpisodeCorpus(tuple(episodes), source_files)
    tree_splits: dict[str, set[str]] = defaultdict(set)
    for episode in corpus.episodes:
        tree_splits[episode.family_id].add(episode.split)
    if any(len(splits) != 1 for splits in tree_splits.values()):
        raise CorpusRefusal("an OASST2 conversation tree crossed corpus splits")

    episode_counts = corpus.sample_counts
    family_counts = Counter()
    history_counts = Counter()
    target_lengths = []
    for episode in corpus.episodes:
        family_counts[(episode.split, episode.family_id)] += 1
        history_counts[(episode.split, len(episode.history))] += 1
        target_lengths.append(len(episode.target_bytes))
    manifest = {
        "schema": "taiji-r2-oasst2-daily-dialogue-corpus-v1",
        "status": "ready"
        if all(episode_counts[split] > 0 for split in ("train", "dev", "final"))
        else "failed",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_license": "Apache-2.0 (Hugging Face dataset metadata)",
        "source_release": "2023-11-05 OASST2 ready messages",
        "source_path": str(source),
        "source_sha256": archive_sha,
        "source_archive_bytes": source.stat().st_size,
        "source_message_count": all_rows,
        "source_chinese_message_count": len(chinese_rows),
        "accepted_chinese_message_count": len(accepted),
        "selected_dialogue_episode_count": len(corpus.episodes),
        "selected_family_count": len(tree_splits),
        "split_episode_counts": episode_counts,
        "split_family_counts": dict(sorted(Counter(split for split, _ in family_counts).items())),
        "history_turn_counts": {
            f"{split}/{history_count}": count
            for (split, history_count), count in sorted(history_counts.items())
        },
        "target_byte_length": {
            "min": min(target_lengths),
            "median": sorted(target_lengths)[len(target_lengths) // 2],
            "max": max(target_lengths),
            "total_with_end_markers": sum(target_lengths),
        },
        "split_salt": SPLIT_SALT,
        "split_rule": "hash(message_tree_id) mod 10000: [0,7000) train, [7000,8000) dev, [8000,10000) final",
        "filter_policy": {
            "language": "zh",
            "tree_state": "ready_for_export",
            "deleted": False,
            "review_result": True,
            "synthetic": False,
            "excluded_positive_flags": list(FILTERED_FLAGS),
            "email_or_phone_pattern": EMAIL_OR_PHONE.pattern,
            "assistant_rank_candidates": [None, 0, 1],
            "one_best_assistant_response_per_prompter": True,
            "max_user_input_bytes": MAX_CURRENT_USER_BYTES,
            "max_target_bytes": MAX_TARGET_BYTES,
            "max_history_pairs": MAX_HISTORY_PAIRS,
            "max_prefix_bytes": MAX_PREFIX_BYTES,
        },
        "rejection_counts": dict(sorted(rejected.items())),
        "corpus_digest": corpus.digest,
        "episode_jsonl_sha256": None,
        "episode_ids": [episode.episode_id for episode in corpus.episodes],
    }
    return corpus, manifest


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / SOURCE_RELATIVE_PATH,
    )
    parser.add_argument(
        "--episodes",
        type=Path,
        default=ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_episodes.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/conversation/OASST2/taiji_r2_daily_dialogue_manifest.json",
    )
    parser.add_argument("--write", action="store_true", help="write derived data and manifest once")
    args = parser.parse_args(argv)
    try:
        corpus, manifest = build(args.source)
        serialized_rows = [
            json.dumps(
                episode.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            for episode in corpus.episodes
        ]
        episode_bytes = ("\n".join(serialized_rows) + "\n").encode("utf-8")
        episode_sha = hashlib.sha256(episode_bytes).hexdigest()
        manifest["episode_jsonl_sha256"] = episode_sha
        trainer_corpus = LanguageEpisodeCorpus(
            corpus.episodes,
            ((str(args.episodes.resolve()), episode_sha),),
        )
        manifest["prepared_episode_digest"] = manifest["corpus_digest"]
        manifest["corpus_digest"] = trainer_corpus.digest
        manifest["trainer_corpus_digest"] = trainer_corpus.digest
        if args.write:
            if args.episodes.exists() or args.manifest.exists():
                raise FileExistsError("dialogue corpus outputs are write-once")
            write_once(args.episodes, episode_bytes)
            write_once(
                args.manifest,
                (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "source_sha256": manifest["source_sha256"],
                    "corpus_digest": manifest["corpus_digest"],
                    "selected_dialogue_episode_count": manifest["selected_dialogue_episode_count"],
                    "split_episode_counts": manifest["split_episode_counts"],
                    "split_family_counts": manifest["split_family_counts"],
                    "episode_jsonl_sha256": manifest["episode_jsonl_sha256"],
                    "written": args.write,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    except (CorpusRefusal, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"R2 dialogue corpus refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
