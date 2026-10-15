"""Frozen train-derived consistent renaming for PLAN43; no runtime parser."""

from __future__ import annotations

import random
import string
from copy import deepcopy


def rename_item(item, names):
    if item["split"] != "train" or len(names) != 3 or len(set(names)) != 3:
        raise ValueError("train item and three unique replacement names required")
    if any(len(n) != 10 or any(c not in string.ascii_lowercase for c in n) for n in names):
        raise ValueError("ten lowercase bytes per entity required")
    mapping = dict(zip(item["entities"], names, strict=True))
    if set(mapping) & set(names):
        raise ValueError("new names must be disjoint from original names")
    result = deepcopy(item)
    result["entities"] = list(names)

    def rewritten(text):
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text

    result["question"] = rewritten(item["question"])
    result["observations"] = [rewritten(text) for text in item["observations"]]
    return result


def second_question(item):
    if item["split"] != "train":
        raise ValueError("only train pairs allowed")
    if item["kind"] not in ("lookup", "unknown"):
        return None
    entity = item["question"].removeprefix("material of ").removesuffix("?")
    names = [o.split()[0] for o in item["observations"]]
    index = next(i for i, n in enumerate(names) if n != entity) if item["kind"] == "lookup" else 0
    result = deepcopy(item)
    result.update(
        id="pair-" + item["id"],
        parent_id=item["id"],
        original_kind=item["kind"],
        kind="lookup",
        question=f"material of {names[index]}?",
        answer=item["observations"][index].removesuffix(".").split()[-1],
    )
    return result


def build_curriculum(train, forbidden):
    if len(train) != 192 or any(e["split"] != "train" for e in train):
        raise ValueError("all 192 original train items required")
    rng = random.Random(43)
    used = set(forbidden)

    def names():
        output = []
        for _ in range(3):
            while True:
                name = "".join(rng.choices(string.ascii_lowercase, k=10))
                if name not in used:
                    break
            used.add(name)
            output.append(name)
        return output

    stream = [dict(parent_id=train[i % 192]["id"], names=names()) for i in range(3072)]
    probes = [dict(parent_id=e["id"], names=names()) for e in train]
    return dict(
        schema="n1-consistent-renaming-curriculum-v1",
        seed=43,
        updates=6144,
        schedule="even-original-odd-renamed-parent-floor-step-over2-mod192",
        stream=stream,
        probes=probes,
    )


def validate_curriculum(record, train, forbidden):
    expected = build_curriculum(train, forbidden)
    if {k: record.get(k) for k in expected} != expected:
        raise ValueError("curriculum must exactly match frozen train-only seed43 recipe")
    return record


def train_item(step, train, curriculum):
    if not 0 <= step < 6144:
        raise ValueError("fixed 6144-step schedule only")
    item = train[(step // 2) % 192]
    if step % 2:
        entry = curriculum["stream"][step // 2]
        if entry["parent_id"] != item["id"]:
            raise ValueError("paired schedule parent mismatch")
        item = rename_item(item, entry["names"])
    return item, second_question(item)
