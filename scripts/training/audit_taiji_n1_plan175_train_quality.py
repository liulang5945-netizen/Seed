"""Train-only structural/text checks; never parse held-out family text."""

from __future__ import annotations

import collections
import gzip
import json
import unicodedata
from pathlib import Path

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest

PARTITION = ROOT / "reports/taiji_n1_plan175_metadata_partition_20261026.json"
OUTPUT = ROOT / "reports/taiji_n1_plan175_train_quality_20261026.json"


def check_chain(chain):
    """Require complete alternating Chinese context and a complete bounded target."""
    for index, row in enumerate(chain):
        expected = {"prompter", "user"} if index % 2 == 0 else {"assistant"}
        if row.get("deleted") or row.get("lang") != "zh" or row.get("role") not in expected:
            return "invalid_context"
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            return "empty_text"
        if any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in text):
            return "control_character"
    if len(chain) < 2 or len(chain) % 2:
        return "invalid_context"
    if len(chain[-1]["text"].encode("utf-8")) > 512:
        return "target_over_512_bytes"
    if sum(len(row["text"].encode("utf-8")) for row in chain[:-1]) > 4096:
        return "context_over_4096_bytes"
    return "structurally_eligible_not_semantically_verified"


def main():
    from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as source

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    spec = json.loads(PARTITION.read_text(encoding="utf-8"))
    if (
        digest(source.RAW_SOURCE) != spec["raw_source_sha256"]
        or digest(source.ASSISTANT_EPISODES) != spec["old_episode_source_sha256"]
    ):
        raise ValueError("frozen source changed")
    rows = {}
    with gzip.open(source.RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = source._json_selected_fields(
                line, {"message_id", "parent_id", "role", "lang", "deleted"}
            )
            rows[row["message_id"]] = row
    needed, chains = set(), {}
    forbidden = set(spec["holdout_families"]) | set(spec["protected_families"])
    for item in spec["train_candidates"]:
        identity, chain, seen = item["target_id"], [], set()
        while identity is not None:
            if identity not in rows or identity in seen:
                raise ValueError("broken ancestor chain")
            seen.add(identity)
            chain.append(identity)
            identity = rows[identity].get("parent_id")
        if chain[-1] in forbidden or chain[-1] != item["family"]:
            raise ValueError("partition family mismatch or protected body access")
        chains[item["target_id"]] = list(reversed(chain))
        needed.update(chain)
    with gzip.open(source.RAW_SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            identity = source._json_selected_fields(line, {"message_id"})["message_id"]
            if identity in needed:
                rows[identity]["text"] = source._json_selected_fields(line, {"text"}).get("text")
    reasons, accepted, seen_questions = collections.Counter(), [], set()
    target_lengths = []
    for item in spec["train_candidates"]:
        chain = [rows[identity] for identity in chains[item["target_id"]]]
        target_text = chain[-1].get("text")
        if isinstance(target_text, str):
            target_lengths.append(len(target_text.encode("utf-8")))
        reason = check_chain(chain)
        if reason == "structurally_eligible_not_semantically_verified":
            key = tuple(
                " ".join(unicodedata.normalize("NFKC", row["text"]).split()) for row in chain[:-1]
            )
            if key in seen_questions:
                reason = "normalized_duplicate_context"
            else:
                seen_questions.add(key)
                accepted.append(item)
        reasons[reason] += 1
    result = {
        "schema": "taiji-n175-train-structural-quality-v1",
        "partition_sha256": digest(PARTITION),
        "script_sha256": digest(Path(__file__)),
        "candidate_count": len(spec["train_candidates"]),
        "target_byte_threshold_counts": {
            str(limit): sum(length <= limit for length in target_lengths)
            for limit in (128, 512, 1024, 2048, 4096)
        },
        "full_target_bytes_all_candidates": sum(target_lengths),
        "max_target_bytes": max(target_lengths, default=0),
        "length_limit_scope": "512-byte scenario only; long targets are not semantically rejected",
        "reason_counts": dict(reasons),
        "accepted_metadata": accepted,
        "accepted_families": len({item["family"] for item in accepted}),
        "train_text_nodes_parsed": len(needed),
        "heldout_text_nodes_parsed": 0,
        "semantic_correctness_verified": False,
        "training_updates": 0,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "accepted_metadata"}))


if __name__ == "__main__":
    main()
