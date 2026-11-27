"""Freeze rank-zero, one-target metadata partitions without reading text bodies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest

SALT = "n175-root-holdout-v1-20261026"


def partition(rows, old, holdout_count=64):
    roots = {}

    def root_of(identity):
        trail, seen = [], set()
        while identity not in roots:
            if identity not in rows:
                raise ValueError("missing ancestor")
            if identity in seen:
                raise ValueError("parent cycle")
            seen.add(identity)
            trail.append(identity)
            parent = rows[identity].get("parent_id")
            if parent is None:
                roots[identity] = identity
                break
            identity = parent
        root = roots[identity]
        for node in trail:
            roots[node] = root
        return root

    old_roots, protected = set(), set()
    for episode in old:
        identity = episode["episode_id"].split(":", 1)[-1]
        family = root_of(identity)
        old_roots.add(family)
        if episode["split"] != "train":
            protected.add(family)
    candidates = {}
    for identity, row in sorted(rows.items()):
        parent = rows.get(row.get("parent_id"), {})
        if not (
            row.get("role") == "assistant"
            and row.get("lang") == "zh"
            and row.get("rank") == 0
            and not row.get("deleted")
            and parent.get("role") in {"user", "prompter"}
            and parent.get("lang") == "zh"
            and not parent.get("deleted")
        ):
            continue
        family = root_of(identity)
        if family in protected:
            continue
        prompt = row["parent_id"]
        if prompt in candidates:
            raise ValueError("multiple rank-zero targets for one prompt")
        candidates[prompt] = {"prompt_id": prompt, "target_id": identity, "family": family}
    fresh = {item["family"] for item in candidates.values()} - old_roots
    if holdout_count < 1 or len(fresh) < holdout_count:
        raise ValueError("insufficient fresh families for fixed holdout")
    held = set(
        sorted(fresh, key=lambda x: hashlib.sha256(f"{SALT}:{x}".encode()).hexdigest())[
            :holdout_count
        ]
    )
    train, holdout = [], []
    for prompt in sorted(candidates):
        item = candidates[prompt]
        (holdout if item["family"] in held else train).append(item)
    return {
        "train_candidates": train,
        "holdout_candidates": holdout,
        "holdout_families": sorted(held),
        "protected_families": sorted(protected),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != ROOT / "reports" or output.exists():
        parser.error("output must be new under reports/")
    from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as source

    if (
        digest(source.RAW_SOURCE) != source.RAW_SOURCE_SHA256
        or digest(source.ASSISTANT_EPISODES) != source.ASSISTANT_EPISODES_SHA256
    ):
        raise ValueError("frozen source changed")
    rows = {}
    fields = {"message_id", "parent_id", "role", "lang", "deleted", "rank"}
    with gzip.open(source.RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = source._json_selected_fields(line, fields)
            identity = row.get("message_id")
            if not identity or identity in rows:
                raise ValueError("missing or duplicate message ID")
            rows[identity] = row
    with source.ASSISTANT_EPISODES.open(encoding="utf-8") as stream:
        old = [source._json_selected_fields(line, {"episode_id", "split"}) for line in stream]
    result = partition(rows, old)
    payload = {
        "schema": "taiji-n175-metadata-partition-v1",
        "status": "candidate_partition_not_quality_approved",
        "salt": SALT,
        "raw_source_sha256": source.RAW_SOURCE_SHA256,
        "old_episode_source_sha256": source.ASSISTANT_EPISODES_SHA256,
        "script_sha256": digest(Path(__file__)),
        "text_bodies_parsed": 0,
        "training_updates": 0,
        **result,
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {"status": payload["status"], **{key: len(value) for key, value in result.items()}}
        )
    )


if __name__ == "__main__":
    main()
