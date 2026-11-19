"""09.6.4 oracles for the consensus birth-direction rule.

The rule must split supervised rounds (not raw steps), stay deterministic, return
unit directions of the P-column width, and never name the evaluation set in code.
"""

from __future__ import annotations

import ast

import torch

from taiji.birth_direction import (
    RULE_NAME,
    consensus_directions,
    window_edges,
    window_nll,
)
from taiji.living_core import LivingCoreConfig, LivingCoreReference
from taiji.state_migration import CareerJournal, career_step_specs

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
JOURNAL = CareerJournal(
    subject_id="direction-rule",
    prompt="观察材料开头十六个字节用于对齐。".encode(),
    target="答复内容。".encode(),
    goal="根据已见材料回答。".encode(),
    records=(),
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)


def fresh_parent() -> LivingCoreReference:
    torch.manual_seed(20261101)
    return LivingCoreReference(CONFIG).double()


def test_window_edges_split_supervised_rounds():
    specs = career_step_specs(JOURNAL)
    rounds = [index for index, spec in enumerate(specs) if spec["kind"] == "round"]
    first_round = rounds[0]
    assert first_round > 0, "this journal must start with prompt patches"
    for parts in (2, 4):
        edges = window_edges(JOURNAL, parts)
        assert len(edges) == parts
        assert edges[0][0] == first_round
        assert edges[-1][1] == len(specs)
        for (_, end), (start, _) in zip(edges, edges[1:], strict=False):
            assert end == start, "windows must be contiguous"
        covered = [index for start, end in edges for index in range(start, end)]
        assert covered == rounds[: len(covered)]


def test_window_edges_reject_degenerate_declarations():
    for parts in (1, 0, -3):
        try:
            window_edges(JOURNAL, parts)
        except ValueError:
            continue
        raise AssertionError(f"parts={parts} should be rejected")


def test_window_nll_rejects_a_prompt_only_window():
    parent = fresh_parent()
    try:
        window_nll(parent, JOURNAL, 0, 1)
    except ValueError:
        return
    raise AssertionError("a prompt-only window has no supervised rounds and must be rejected")


def test_consensus_directions_are_unit_and_group_shaped():
    parent = fresh_parent()
    directions = consensus_directions(parent, JOURNAL, parts=2, count=2)
    assert set(directions) == {(RULE_NAME, group) for group in range(CONFIG.groups)}
    for group in range(CONFIG.groups):
        vectors = directions[(RULE_NAME, group)]
        assert len(vectors) == 2
        for vector in vectors:
            assert vector.shape == (parent.P.shape[1],)
            assert abs(float(vector.norm()) - 1.0) <= 1e-9


def test_consensus_directions_are_deterministic():
    first = consensus_directions(fresh_parent(), JOURNAL, parts=2, count=2)
    second = consensus_directions(fresh_parent(), JOURNAL, parts=2, count=2)
    for key in first:
        for left, right in zip(first[key], second[key], strict=True):
            assert torch.equal(left, right)


def test_rule_source_never_names_the_evaluation_set():
    import inspect

    import taiji.birth_direction as module

    tree = ast.parse(inspect.getsource(module))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    seen = names | attributes | imports
    for forbidden in ("HOLDOUT", "RETENTION", "pricing"):
        assert forbidden not in seen, forbidden
