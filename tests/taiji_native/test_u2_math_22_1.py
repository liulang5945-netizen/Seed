"""Regression guards for the U2 22.1 operator fixtures, not model capability."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/training/verify_taiji_u2_math_22_1.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("u2_math_22_1_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allowed_zero_chi_retains_right_derivative():
    verifier = load_verifier()

    def value(data):
        return torch.tensor(data, dtype=torch.float64)

    chi = value(0.0).requires_grad_()
    memory = value([[0.2]])
    result = verifier.fp1(
        memory, value([[1.0]]), value([[1.0]]), value([0.8]), value([1.0]), chi, value(10.0)
    )
    assert torch.equal(result, memory)
    assert torch.allclose(torch.autograd.grad(result.sum(), chi)[0], value(0.76))


def test_local_operator_and_counterexample_groups_pass():
    verifier = load_verifier()
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        groups = verifier.run_checks()
    finally:
        torch.set_num_threads(previous)
    assert len(groups) == 13
    assert all(row["passed"] for row in groups)
    assert all(
        row["evidence"]["subtraction_mutant_rejected"]
        for row in groups
        if row["name"] == "sequence"
    )


def test_unknown_outcome_keeps_an_unsettled_receipt_pending():
    verifier = load_verifier()
    receipt = {"id": "pending", "goal": "old", "discount": 0.9, "old_value": 0.0}
    outcome = {
        "id": "unknown0",
        "receipt_id": "pending",
        "known": False,
        "utility": 0.0,
        "terminal": False,
    }
    settled = set()
    assert verifier.settle(receipt, outcome, {"old": 1.0}, settled) is None
    assert not settled
    assert verifier.settle(receipt, {**outcome, "known": True}, {"old": 1.0}, settled) == 0.9
    assert settled == {"pending"}
