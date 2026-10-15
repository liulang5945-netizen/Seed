"""PLAN-N1-18 diverse observed values with positive target alphabet support."""

from __future__ import annotations

import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training.build_taiji_n1_evidence_curriculum import (  # noqa: E402
    DIRECTIONS,
    KINDS,
    preflight,
)


def build():
    rng = random.Random(18)
    names, words = set(), set()

    def novel(length, used):
        while True:
            value = "".join(rng.choices(string.ascii_lowercase, k=length))
            if value not in used:
                used.add(value)
                return value

    episodes = []
    for split, count in (("train", 192), ("dev", 32), ("holdout", 32)):
        for index in range(count):
            kind = KINDS[index % 4]
            name, other, third = (novel(10, names) for _ in range(3))
            first, second = novel(6, words), novel(6, words)
            if kind == "lookup":
                observations = [f"{name} uses {first}.", f"{other} uses {second}."]
                rng.shuffle(observations)
                question, answer = f"material of {name}?", first
            elif kind == "revision":
                observations = [f"{name} uses {first}.", f"{name} now uses {second}."]
                question, answer = f"material of {name}?", second
            elif kind == "transition":
                start, turn = rng.randrange(4), rng.choice((-1, 1))
                observations = [
                    f"{name} faces {DIRECTIONS[start]}.",
                    f"{name} turns {'right' if turn == 1 else 'left'}.",
                ]
                question, answer = f"direction of {name}?", DIRECTIONS[(start + turn) % 4]
            else:
                observations = [f"{other} uses {first}.", f"{third} uses {second}."]
                question, answer = f"material of {name}?", "unknown"
            episodes.append(
                {
                    "id": f"{split}-{index:03d}",
                    "split": split,
                    "kind": kind,
                    "entities": [name, other, third],
                    "observations": observations,
                    "question": question,
                    "answer": answer,
                }
            )
    return {"schema": "n1-native-evidence-curriculum-v3", "seed": 18, "episodes": episodes}


def validate(manifest):
    episodes = manifest["episodes"]
    if manifest["schema"] != "n1-native-evidence-curriculum-v3":
        raise ValueError("v3 schema required")
    if len(episodes) != 256 or len({e["id"] for e in episodes}) != 256:
        raise ValueError("v3 needs all 256 unique episode identities")
    counts, identities, values = {}, {}, {}
    for split, size in (("train", 192), ("dev", 32), ("holdout", 32)):
        selected = [item for item in episodes if item["split"] == split]
        counts[split] = dict(Counter(item["kind"] for item in selected))
        if len(selected) != size or counts[split] != {kind: size // 4 for kind in KINDS}:
            raise ValueError("v3 split/kind denominator mismatch")
        identities[split] = {name for item in selected for name in item["entities"]}
        values[split] = {
            text.removesuffix(".").split()[-1]
            for item in selected
            for text in item["observations"]
            if " uses " in text
        }
    for left, right in (("train", "dev"), ("train", "holdout"), ("dev", "holdout")):
        if identities[left] & identities[right] or values[left] & values[right]:
            raise ValueError("v3 split entity/value leakage")
    for item in episodes:
        if len(item["observations"]) != 2 or any(
            len(name) != 10 or not name.isalpha() or not name.islower() for name in item["entities"]
        ):
            raise ValueError("v3 inference identity/count shortcut")
        if item["answer"] in item["question"]:
            raise ValueError("answer leaked into query")
        if item["kind"] == "transition" and item["answer"] in " ".join(item["observations"]):
            raise ValueError("transition target directly present in observations")
    train = [item for item in episodes if item["split"] == "train"]
    target_counts = Counter(char for item in train for char in item["answer"])
    if any(target_counts[char] < 8 for char in string.ascii_lowercase):
        raise ValueError("v3 positive target alphabet coverage below 8 per letter")
    material_gold = {item["answer"] for item in train if item["kind"] in ("lookup", "revision")}
    if len(material_gold) < 80:
        raise ValueError("v3 material targets collapsed to a small answer vocabulary")
    return {
        "counts": counts,
        "positive_target_letter_counts": dict(sorted(target_counts.items())),
        "distinct_train_material_targets": len(material_gold),
    }


def main():
    path = ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json"
    report = ROOT / "reports/taiji_n1_native_evidence_curriculum_v3_preflight_20261015.json"
    if path.exists() or report.exists():
        raise FileExistsError("v3 curriculum artifacts already exist; no overwrite")
    manifest = build()
    admission = validate(manifest)
    paired.write_new(path, manifest)
    result = {
        "schema": "n1-content-curriculum-v3-preflight-v1",
        "manifest_sha256": paired.sha(path),
        "admission": admission,
        "generator_sha256": paired.sha(Path(__file__)),
        "model_identity": paired.identity(),
        "holdout_model_queries": 0,
        "dev_model_queries": 0,
    }
    try:
        result.update(preflight(manifest))
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        paired.write_new(report, result)
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
