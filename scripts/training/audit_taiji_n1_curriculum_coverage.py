"""Static post-fit audit: target support, never model evaluation or training."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402


def coverage(manifest):
    train = [item for item in manifest["episodes"] if item["split"] == "train"]
    characters = Counter(byte for item in train for byte in item["answer"].encode())
    words = {item["answer"] for item in train}
    result = {
        "train_answer_classes": sorted(words),
        "train_positive_target_byte_counts": {
            str(key): value for key, value in sorted(characters.items())
        },
        "by_split": {},
    }
    for split in ("dev", "holdout"):
        subset = [item for item in manifest["episodes"] if item["split"] == split]
        missing = sorted(
            {byte for item in subset for byte in item["answer"].encode() if byte not in characters}
        )
        affected = [
            item["id"]
            for item in subset
            if any(byte not in characters for byte in item["answer"].encode())
        ]
        result["by_split"][split] = {
            "items": len(subset),
            "unseen_target_bytes": missing,
            "unseen_target_characters": bytes(missing).decode("ascii"),
            "affected_ids": affected,
            "new_answer_classes": sorted({item["answer"] for item in subset} - words),
        }
    return result


def audit():
    report = ROOT / "reports/taiji_n1_curriculum_target_coverage_20261015.json"
    if report.exists():
        raise FileExistsError("coverage audit refuses overwrite")
    manifest = paired.checked_manifest()
    output = {
        "schema": "n1-curriculum-target-support-v1",
        "manifest_sha256": paired.MANIFEST_SHA,
        "model_queries": 0,
        "optimizer_steps": 0,
        "coverage": coverage(manifest),
        "code_sha256": paired.sha(Path(__file__)),
        "cpu_seconds": time.process_time(),
        "reading_limit": "Missing positive target support is a curriculum confound, not proof of impossible generation or the unique cause of failure. Expand training support rather than remove hard evaluation bytes; use a fresh independently frozen surface.",
    }
    paired.write_new(report, output)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    audit()
