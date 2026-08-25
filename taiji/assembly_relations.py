"""Contracts for the structural A1 assembly-composition benchmark."""

from __future__ import annotations

from dataclasses import dataclass

SymbolSequence = tuple[int, ...]


@dataclass(frozen=True)
class AssemblyRelationExample:
    """One ordered composition of two reusable atoms.

    ``left_atom`` and ``right_atom`` are evaluation metadata only.  They are
    never passed to the model; the model receives ``sequence``.  Keeping the
    provenance outside the input makes the benchmark test compositional
    binding rather than a hand-written vocabulary or answer table.
    """

    left_atom: int
    right_atom: int
    sequence: SymbolSequence
    split_index: int
    perturbation: str = "clean"

    def __post_init__(self) -> None:
        if int(self.left_atom) < 0 or int(self.right_atom) < 0:
            raise ValueError("assembly atom ids must be non-negative")
        if not self.sequence:
            raise ValueError("assembly relation sequence cannot be empty")
        if any(int(symbol) < 0 for symbol in self.sequence):
            raise ValueError("assembly relation sequence contains a negative symbol")
        if not 0 < int(self.split_index) < len(self.sequence):
            raise ValueError("assembly relation split_index must be inside the sequence")
        if not self.perturbation:
            raise ValueError("assembly relation perturbation cannot be empty")

    @property
    def pair(self) -> tuple[int, int]:
        return int(self.left_atom), int(self.right_atom)


@dataclass(frozen=True)
class AssemblyRelationCorpus:
    """A1 v2 contract for unseen ordered atom combinations."""

    atom_count: int
    train: tuple[AssemblyRelationExample, ...]
    unseen_composition: tuple[AssemblyRelationExample, ...]
    boundary_perturbed: tuple[AssemblyRelationExample, ...]
    random_chunk: tuple[AssemblyRelationExample, ...]

    def __post_init__(self) -> None:
        if int(self.atom_count) < 2:
            raise ValueError("assembly relation corpus requires at least two atoms")
        roles = (
            self.train,
            self.unseen_composition,
            self.boundary_perturbed,
            self.random_chunk,
        )
        if any(not role for role in roles):
            raise ValueError("assembly relation corpus roles cannot be empty")

        for role in roles:
            for example in role:
                if example.left_atom >= self.atom_count or example.right_atom >= self.atom_count:
                    raise ValueError("assembly relation atom id exceeds atom_count")

        train_pairs = {example.pair for example in self.train}
        unseen_pairs = {example.pair for example in self.unseen_composition}
        if train_pairs & unseen_pairs:
            raise ValueError("train and unseen assembly pairs must be disjoint")
        for name, role in (
            ("boundary_perturbed", self.boundary_perturbed),
            ("random_chunk", self.random_chunk),
        ):
            role_pairs = {example.pair for example in role}
            if role_pairs != unseen_pairs:
                raise ValueError(f"{name} must preserve the unseen pair set")

    @property
    def train_pairs(self) -> frozenset[tuple[int, int]]:
        return frozenset(example.pair for example in self.train)

    @property
    def unseen_pairs(self) -> frozenset[tuple[int, int]]:
        return frozenset(example.pair for example in self.unseen_composition)
