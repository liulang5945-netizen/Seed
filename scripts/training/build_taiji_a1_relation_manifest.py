"""Build the structural A1 assembly-relation manifest.

This benchmark deliberately keeps composition provenance outside the model
input.  Atoms are reusable byte sequences selected from the corpus; training
and holdout contain disjoint ordered atom pairs while sharing the atoms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji import AssemblyRelationCorpus, AssemblyRelationExample  # noqa: E402

RELATION_MANIFEST_FORMAT = "taiji-a1-assembly-relation-v1"
BOUNDARY_SYMBOL = 256


def _text_from_record(record: Any) -> str | None:
    if isinstance(record, str):
        return record if record.strip() else None
    if not isinstance(record, dict):
        return None
    for key in ("text", "content", "prompt", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _bucket(text: str, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{text}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10_000


def _atom_symbols(text: str, *, max_symbols: int) -> tuple[int, ...]:
    symbols = tuple(text.encode()[: int(max_symbols)])
    if len(symbols) < 4:
        raise ValueError("A1 atom must contain at least four UTF-8 bytes")
    return symbols


def _insert_at(sequence: tuple[int, ...], position: int, symbol: int) -> tuple[int, ...]:
    return (*sequence[:position], int(symbol), *sequence[position:])


def _example(
    atoms: list[tuple[int, ...]],
    left_atom: int,
    right_atom: int,
    *,
    perturbation: str,
    seed: int,
) -> AssemblyRelationExample:
    left = atoms[left_atom]
    right = atoms[right_atom]
    clean = (*left, *right)
    split_index = len(left)
    if perturbation == "clean":
        sequence = clean
        observed_split = split_index
    elif perturbation == "boundary":
        sequence = _insert_at(clean, split_index, BOUNDARY_SYMBOL)
        observed_split = split_index + 1
    elif perturbation == "random_chunk":
        generator = random.Random(int(seed))
        position = generator.randint(1, len(clean) - 1)
        sequence = _insert_at(clean, position, BOUNDARY_SYMBOL)
        observed_split = split_index + int(position <= split_index)
    else:
        raise ValueError(f"unsupported assembly relation perturbation: {perturbation}")
    return AssemblyRelationExample(
        left_atom=left_atom,
        right_atom=right_atom,
        sequence=sequence,
        split_index=observed_split,
        perturbation=perturbation,
    )


def build_relation_manifest(
    corpus_paths: list[Path | str],
    *,
    atom_count: int = 8,
    max_atom_symbols: int = 32,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Build disjoint train/holdout ordered pairs over shared atoms."""

    if atom_count < 2:
        raise ValueError("A1 relation atom_count must be at least two")
    if max_atom_symbols < 4:
        raise ValueError("A1 relation max_atom_symbols must be at least four")

    atoms: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    scanned = 0
    for raw_path in corpus_paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                scanned += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = _text_from_record(record)
                if text is None or _bucket(text, seed=seed) >= 8_000:
                    continue
                atom = _atom_symbols(text, max_symbols=max_atom_symbols)
                if atom in seen:
                    continue
                seen.add(atom)
                atoms.append(atom)
                if len(atoms) >= atom_count:
                    break
        if len(atoms) >= atom_count:
            break
    if len(atoms) < atom_count:
        raise ValueError(f"corpus did not provide enough distinct atoms: {len(atoms)}/{atom_count}")

    all_pairs = [
        (left, right) for left in range(atom_count) for right in range(atom_count) if left != right
    ]
    train_pairs = [pair for pair in all_pairs if (pair[0] * 31 + pair[1] + int(seed)) % 4 != 0]
    holdout_pairs = [pair for pair in all_pairs if pair not in train_pairs]
    if not train_pairs or not holdout_pairs:
        raise ValueError("A1 relation split produced an empty pair role")

    train = tuple(
        _example(atoms, left, right, perturbation="clean", seed=seed + index)
        for index, (left, right) in enumerate(train_pairs)
    )
    unseen = tuple(
        _example(atoms, left, right, perturbation="clean", seed=seed + index)
        for index, (left, right) in enumerate(holdout_pairs)
    )
    boundary = tuple(
        _example(atoms, left, right, perturbation="boundary", seed=seed + index)
        for index, (left, right) in enumerate(holdout_pairs)
    )
    random_chunk = tuple(
        _example(atoms, left, right, perturbation="random_chunk", seed=seed + 97 * index)
        for index, (left, right) in enumerate(holdout_pairs)
    )
    corpus = AssemblyRelationCorpus(
        atom_count=atom_count,
        train=train,
        unseen_composition=unseen,
        boundary_perturbed=boundary,
        random_chunk=random_chunk,
    )

    def serialize(example: AssemblyRelationExample) -> dict[str, Any]:
        return {
            "left_atom": example.left_atom,
            "right_atom": example.right_atom,
            "sequence": list(example.sequence),
            "split_index": example.split_index,
            "perturbation": example.perturbation,
        }

    return {
        "format": RELATION_MANIFEST_FORMAT,
        "source": [str(Path(path)) for path in corpus_paths],
        "selection": {
            "method": "sha256-train-atom-pool-plus-deterministic-pair-split",
            "seed": int(seed),
            "scanned_rows": scanned,
            "atom_count": atom_count,
            "max_atom_symbols": max_atom_symbols,
            "boundary_symbol": BOUNDARY_SYMBOL,
            "pair_split": "(left * 31 + right + seed) % 4 == 0 is holdout",
        },
        "atoms": [list(atom) for atom in atoms],
        "roles": {
            "train": [serialize(example) for example in corpus.train],
            "unseen_composition": [serialize(example) for example in corpus.unseen_composition],
            "boundary_perturbed": [serialize(example) for example in corpus.boundary_perturbed],
            "random_chunk": [serialize(example) for example in corpus.random_chunk],
        },
    }


def relation_corpus_from_manifest(payload: dict[str, Any]) -> AssemblyRelationCorpus:
    if payload.get("format") != RELATION_MANIFEST_FORMAT:
        raise ValueError("unsupported Taiji A1 relation manifest format")

    def examples(name: str) -> tuple[AssemblyRelationExample, ...]:
        return tuple(
            AssemblyRelationExample(
                left_atom=int(item["left_atom"]),
                right_atom=int(item["right_atom"]),
                sequence=tuple(int(symbol) for symbol in item["sequence"]),
                split_index=int(item["split_index"]),
                perturbation=str(item["perturbation"]),
            )
            for item in payload["roles"][name]
        )

    return AssemblyRelationCorpus(
        atom_count=len(payload["atoms"]),
        train=examples("train"),
        unseen_composition=examples("unseen_composition"),
        boundary_perturbed=examples("boundary_perturbed"),
        random_chunk=examples("random_chunk"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--atom-count", type=int, default=8)
    parser.add_argument("--max-atom-symbols", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    manifest = build_relation_manifest(
        args.corpus,
        atom_count=args.atom_count,
        max_atom_symbols=args.max_atom_symbols,
        seed=args.seed,
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    corpus = relation_corpus_from_manifest(manifest)
    print(
        json.dumps(
            {
                "format": RELATION_MANIFEST_FORMAT,
                "atom_count": corpus.atom_count,
                "train_pairs": len(corpus.train_pairs),
                "unseen_pairs": len(corpus.unseen_pairs),
                "manifest": str(args.manifest_out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
