"""Reproducible train-only receipt for the complete-answer V3 candidate corpus."""

from __future__ import annotations

import collections
import json
from pathlib import Path

from scripts.training.audit_taiji_n1_plan175_source_metadata import ROOT, digest
from scripts.training.calibrate_taiji_n1_plan175_v3_train_only import load_train_episodes

OUTPUT = ROOT / "reports/taiji_n1_plan175_v3_train_corpus_v2_20261026.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    episodes, data = load_train_episodes()
    targets = [len(episode.target_bytes) for _, episode in episodes]
    prompts = [len(episode.prompt_bytes) for _, episode in episodes]
    family_counts = collections.Counter(episode.family_id for _, episode in episodes)
    manifest = [
        {
            "target_id": candidate["target_id"],
            "family": candidate["family"],
            "prompt_bytes": len(episode.prompt_bytes),
            "target_bytes": len(episode.target_bytes),
        }
        for candidate, episode in episodes
    ]
    payload = {
        "schema": "taiji-n1-plan175-v3-train-corpus-v1",
        "status": "train_only_structural_corpus_audited",
        "audit_script_sha256": digest(Path(__file__)),
        "builder_script_sha256": digest(
            ROOT / "scripts/training/calibrate_taiji_n1_plan175_v3_train_only.py"
        ),
        **data,
        "episode_count": len(episodes),
        "family_count": len(family_counts),
        "single_episode_families": sum(count == 1 for count in family_counts.values()),
        "families_with_multiple_episodes": sum(count > 1 for count in family_counts.values()),
        "complete_target_bytes_per_epoch": sum(targets),
        "serialized_prompt_bytes_per_epoch": sum(prompts),
        "target_byte_threshold_counts": {
            str(limit): sum(value <= limit for value in targets) for limit in (128, 512, 1024, 2048)
        },
        "max_complete_target_bytes": max(targets),
        "max_serialized_prompt_bytes": max(prompts),
        "candidate_manifest_sha256": __import__("hashlib")
        .sha256(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )
        .hexdigest(),
        "candidate_manifest": manifest,
        "heldout_text_parsed": 0,
        "training_updates": 0,
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "episode_count",
                    "family_count",
                    "complete_target_bytes_per_epoch",
                    "serialized_prompt_bytes_per_epoch",
                    "target_byte_threshold_counts",
                    "heldout_text_parsed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
