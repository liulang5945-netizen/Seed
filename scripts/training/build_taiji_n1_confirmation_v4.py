"""PLAN62: generate one sealed confirmation face; never imports a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/n1_relation_confirmation_v4_20261015.json"
REPORT = ROOT / "reports/taiji_n1_v4_face_freeze_20261015.json"
SOURCES = (
    ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json",
    ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json",
    ROOT / "plans/manifests/n1_renaming_curriculum_20261015.json",
)
EXPECTED = (
    "ac56fb0aba2ef602779f58749e6f8c35324c40c433fd16ce55b89689235b0604",
    "572a270bcb5e292baee7577b5c0ec2d3f313bd93ae338fa6d5984a646a2fe9c6",
    "c44ea78f40d7255688013eb4054ae803bd04ee97dc5ba2e0c02b82e301ce6fdd",
)
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusions():
    if tuple(sha(p) for p in SOURCES) != EXPECTED:
        raise ValueError("frozen source manifests changed")
    v3, old, curriculum = [json.loads(p.read_text(encoding="utf-8")) for p in SOURCES]
    names = {n for e in v3["episodes"] for n in e["entities"]}
    names.update(
        n
        for row in old["rows"]
        for variant in row["variants"].values()
        for n in variant["entities"]
    )
    names.update(
        n for entry in [*curriculum["stream"], *curriculum["probes"]] for n in entry["names"]
    )
    words = {
        text.removesuffix(".").split()[-1]
        for e in v3["episodes"]
        for text in e["observations"]
        if " uses " in text
    }
    return names, words


def build():
    names, words = exclusions()
    rng = random.Random(62)

    def novel(length, used):
        while True:
            value = "".join(rng.choices(string.ascii_lowercase, k=length))
            if value not in used:
                used.add(value)
                return value

    episodes = []
    for index in range(32):
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
                f"{name} turns {'right' if turn==1 else 'left'}.",
            ]
            question, answer = f"direction of {name}?", DIRECTIONS[(start + turn) % 4]
        else:
            observations = [f"{other} uses {first}.", f"{third} uses {second}."]
            question, answer = f"material of {name}?", "unknown"
        episodes.append(
            dict(
                id=f"v4-confirm-{index:03d}",
                split="holdout",
                kind=kind,
                entities=[name, other, third],
                observations=observations,
                question=question,
                answer=answer,
            )
        )
    return dict(
        schema="n1-relation-confirmation-v4",
        seed=62,
        excluded_source_sha256=list(EXPECTED),
        episodes=episodes,
    )


def validate(manifest):
    if manifest != build():
        raise ValueError("exact frozen seed62 confirmation face required")
    names, words = exclusions()
    episodes = manifest["episodes"]
    new_names = [n for e in episodes for n in e["entities"]]
    new_words = [
        text.removesuffix(".").split()[-1]
        for e in episodes
        for text in e["observations"]
        if " uses " in text
    ]
    counts = dict(Counter(e["kind"] for e in episodes))
    if (
        len(set(new_names)) != 96
        or set(new_names) & names
        or len(set(new_words)) != len(new_words)
        or set(new_words) & words
        or counts != {kind: 8 for kind in KINDS}
        or any(e["split"] != "holdout" for e in episodes)
    ):
        raise ValueError("disjoint complete confirmation face required")
    return dict(
        items=32,
        by_kind=counts,
        entities=96,
        materials=len(new_words),
        model_queries=0,
        training_items=0,
        dev_items=0,
    )


def execute():
    if MANIFEST.exists() or REPORT.exists():
        raise FileExistsError("one exclusive face; no regeneration")
    face = build()
    summary = validate(face)
    with MANIFEST.open("x", encoding="utf-8") as f:
        json.dump(face, f, ensure_ascii=False, indent=2)
        f.write("\n")
    result = dict(
        schema="n1-v4-face-freeze-v1",
        manifest_sha256=sha(MANIFEST),
        code_sha256=sha(Path(__file__)),
        source_sha256=list(EXPECTED),
        summary=summary,
        cpu_seconds=time.process_time(),
        complete=True,
    )
    with REPORT.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    if parser.parse_args().execute_approved:
        print(json.dumps(execute()))
    else:
        print(json.dumps(dict(read_only=True, items=32, model_queries=0)))
