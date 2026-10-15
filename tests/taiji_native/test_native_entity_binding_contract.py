import json
import random
from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_entity_binding as diagnostic


def items():
    return json.loads(diagnostic.calibration.MANIFEST.read_text(encoding="utf-8"))


def test_complete_paired_train_manifest():
    rows = diagnostic.build(items())
    assert len(rows) == 96
    assert {r["kind"] for r in rows} == {"lookup", "unknown"}
    assert diagnostic.build(items()) == rows
    old_names = {n for e in items()["episodes"] for n in e["entities"]}
    for row in rows:
        original, renamed, switched = [row["variants"][k] for k in diagnostic.CONDITIONS]
        assert original["answer"] == renamed["answer"]
        assert not set(renamed["entities"]) & old_names
        assert list(map(len, original["observations"])) == list(map(len, renamed["observations"]))
        for old, new in zip(original["entities"], renamed["entities"], strict=True):
            assert (old in original["question"]) == (new in renamed["question"])
            for text, changed in zip(
                original["observations"], renamed["observations"], strict=True
            ):
                assert (old in text) == (new in changed)
        assert switched["observations"] == original["observations"]
        entity = switched["question"].split()[-1].removesuffix("?")
        assert (
            next(
                o.split()[-1].removesuffix(".")
                for o in original["observations"]
                if o.startswith(entity + " ")
            )
            == switched["answer"]
        )
        assert switched["question"] != original["question"]
        assert diagnostic.paired.project(renamed).question == renamed["question"].encode()


def test_forbid_nontrain_before_variants():
    for split in ("dev", "holdout"):
        item = deepcopy(items()["episodes"][0])
        item["split"] = split
        with pytest.raises(ValueError, match="only train"):
            diagnostic.variants(item, set(), random.Random(34))
