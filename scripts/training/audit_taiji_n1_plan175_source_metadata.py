"""Metadata-only census: reply edges are not independent user prompts."""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def analyze(rows, old_episodes):
    cache = {}

    def root_of(identity):
        path, seen = [], set()
        while identity in rows and identity not in cache:
            if identity in seen:
                raise ValueError("source parent graph has a cycle")
            seen.add(identity)
            path.append(identity)
            parent = rows[identity].get("parent_id")
            if parent not in rows:
                break
            identity = parent
        root = cache.get(identity, identity)
        for node in path:
            cache[node] = root
        return root

    old_roots, protected_roots, old_train_prompts = set(), set(), set()
    unmapped = 0
    for episode in old_episodes:
        identity = episode["episode_id"].split(":", 1)[-1]
        if identity not in rows:
            unmapped += 1
            continue
        family = root_of(identity)
        old_roots.add(family)
        if episode["split"] != "train":
            protected_roots.add(family)
        else:
            old_train_prompts.add(rows[identity].get("parent_id"))
    pairs = []
    for identity, row in rows.items():
        parent = rows.get(row.get("parent_id"), {})
        if (
            row.get("role") == "assistant"
            and row.get("lang") == "zh"
            and not row.get("deleted")
            and parent.get("role") in {"prompter", "user"}
            and parent.get("lang") == "zh"
            and not parent.get("deleted")
        ):
            pairs.append((identity, row, root_of(identity)))
    pool = [item for item in pairs if item[2] not in protected_roots]
    return {
        "message_records": len(rows),
        "metadata_zh_reply_candidates": len(pairs),
        "distinct_prompt_parent_nodes": len({item[1]["parent_id"] for item in pairs}),
        "root_families": len({item[2] for item in pairs}),
        "rank_counts": dict(collections.Counter(str(item[1].get("rank")) for item in pairs)),
        "old_all_root_families": len(old_roots),
        "old_nontrain_root_families": len(protected_roots),
        "old_train_distinct_prompt_parent_nodes": len(old_train_prompts),
        "old_episode_ids_unmapped": unmapped,
        "pool_reply_candidates": len(pool),
        "pool_root_families": len({item[2] for item in pool}),
        "pool_distinct_prompt_parent_nodes": len({item[1]["parent_id"] for item in pool}),
        "pool_rank_zero_candidates": sum(item[1].get("rank") == 0 for item in pool),
        "pool_families_not_in_old_episode_pool": len({item[2] for item in pool} - old_roots),
    }


def main():
    started = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "reports" or output.exists():
        parser.error("output must be a new report under reports/")
    from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as source

    if (
        digest(source.RAW_SOURCE) != source.RAW_SOURCE_SHA256
        or digest(source.ASSISTANT_EPISODES) != source.ASSISTANT_EPISODES_SHA256
    ):
        raise ValueError("frozen source/episode bytes differ")
    rows, old = {}, []
    fields = {"message_id", "parent_id", "role", "lang", "deleted", "rank"}
    with gzip.open(source.RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            if time.process_time() - started > 90:
                raise TimeoutError("bounded metadata census exceeded 90 CPU seconds")
            row = source._json_selected_fields(line, fields)
            if row.get("message_id") in rows:
                raise ValueError("duplicate message ID in frozen source")
            if row.get("message_id"):
                rows[row["message_id"]] = row
    with source.ASSISTANT_EPISODES.open(encoding="utf-8") as stream:
        for line in stream:
            old.append(source._json_selected_fields(line, {"episode_id", "split"}))
    counts = analyze(rows, old)
    payload = {
        "schema": "taiji-n1-plan175-metadata-feasibility-v1",
        "status": (
            "metadata_complete"
            if counts["old_episode_ids_unmapped"] == 0
            else "source_mapping_incomplete"
        ),
        "raw_source_sha256": source.RAW_SOURCE_SHA256,
        "old_episode_source_sha256": source.ASSISTANT_EPISODES_SHA256,
        "audit_source_sha256": digest(Path(__file__)),
        "counts": counts,
        "scope": "metadata-tagged zh replies with valid zh prompt parent; not body language/quality or semantic dedup",
        "text_bodies_parsed": 0,
        "quality_filtering_done": False,
        "new_holdout_selected": False,
        "model_loads": 0,
        "training_updates": 0,
        "cpu_seconds": time.process_time() - started,
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
