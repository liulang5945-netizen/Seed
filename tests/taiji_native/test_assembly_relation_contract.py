from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.training.build_taiji_a1_relation_manifest import (
    RELATION_MANIFEST_FORMAT,
    build_relation_manifest,
    relation_corpus_from_manifest,
)
from taiji import AssemblyRelationCorpus, AssemblyRelationExample


def test_relation_contract_rejects_leaked_pairs() -> None:
    clean = AssemblyRelationExample(0, 1, (1, 2, 3, 4), 2)
    with pytest.raises(ValueError, match="disjoint"):
        AssemblyRelationCorpus(
            atom_count=2,
            train=(clean,),
            unseen_composition=(clean,),
            boundary_perturbed=(AssemblyRelationExample(0, 1, (1, 2, 7, 3, 4), 3, "boundary"),),
            random_chunk=(AssemblyRelationExample(0, 1, (1, 7, 2, 3, 4), 2, "random_chunk"),),
        )


def test_relation_manifest_has_shared_atoms_and_unseen_ordered_pairs(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for index in range(500):
            handle.write(json.dumps({"text": f"组合原子样本 {index} 的结构内容。"}) + "\n")

    first = build_relation_manifest([corpus_path], atom_count=8, max_atom_symbols=24)
    second = build_relation_manifest([corpus_path], atom_count=8, max_atom_symbols=24)

    assert first == second
    assert first["format"] == RELATION_MANIFEST_FORMAT
    assert len(first["atoms"]) == 8
    train_pairs = {(item["left_atom"], item["right_atom"]) for item in first["roles"]["train"]}
    unseen_pairs = {
        (item["left_atom"], item["right_atom"]) for item in first["roles"]["unseen_composition"]
    }
    assert train_pairs.isdisjoint(unseen_pairs)
    assert train_pairs | unseen_pairs
    assert {
        (item["left_atom"], item["right_atom"]) for item in first["roles"]["boundary_perturbed"]
    } == unseen_pairs
    assert all(256 in item["sequence"] for item in first["roles"]["boundary_perturbed"])
    assert all(256 in item["sequence"] for item in first["roles"]["random_chunk"])

    restored = relation_corpus_from_manifest(first)
    assert restored.train_pairs == frozenset(train_pairs)
    assert restored.unseen_pairs == frozenset(unseen_pairs)
