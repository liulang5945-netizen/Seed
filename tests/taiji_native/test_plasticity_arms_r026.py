"""R026 plasticity-arms oracles: the preregistered six-arm comparison on one parent."""

from __future__ import annotations

import copy

import torch

from taiji import growth_credit as gc
from taiji import native_development as nd
from taiji.living_core import LivingCoreConfig, LivingCoreReference, RawSource
from taiji.state_migration import CareerJournal

CONFIG = LivingCoreConfig(groups=2, width=8, units=12, adapter_rank=2, key_width=4, value_width=4)
RECORDS = (
    RawSource("s1", b"abc target ab", version=1, observed_at=0),
    RawSource("s2", b"secondary ab", version=2, observed_at=4),
)
FIT = CareerJournal(
    subject_id="r026-fit",
    prompt=b"observe-context-" * 4,
    target=(b"target-bytes-ab" * 7)[:100],
    goal=b"growth-goal",
    records=RECORDS,
    raw_nu=(0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3),
    known_nu=(True, True, False, True, True, True, False, True),
    now=10,
)
HOLDOUT = CareerJournal(
    subject_id="r026-holdout",
    prompt=b"probe-holdout-" * 6,
    target=b"probe target ab" * 4,
    goal=b"growth-goal",
    records=RECORDS,
    raw_nu=FIT.raw_nu,
    known_nu=FIT.known_nu,
    now=10,
)
RETENTION = CareerJournal(
    subject_id="r026-retention",
    prompt=b"retention-context-" * 6,
    target=b"keep target ab" * 4,
    goal=b"growth-goal",
    records=RECORDS,
    raw_nu=FIT.raw_nu,
    known_nu=FIT.known_nu,
    now=10,
)

_CACHE = None


def comparison():
    """Run all six arms once from the same parent; cached across checks."""
    global _CACHE
    if _CACHE is None:
        torch.set_num_threads(1)
        parent = LivingCoreReference(CONFIG).double()
        book, _ = gc.collect_credit(parent, FIT)
        steps = len(book.deltas)
        fit, check = range(0, steps // 2), range(steps // 2, steps)
        full_credits = {n: gc.branch_credit(book, b) for b, n in enumerate(book.branch_ids)}
        fit_credits = {n: gc.branch_credit(book, b, fit) for b, n in enumerate(book.branch_ids)}
        verdicts = {}
        for b, n in enumerate(book.branch_ids):
            tau_noise = gc.calibrate_tau_noise(book, b, fit)
            verdicts[n] = gc.branch_verdict(book, b, tau_noise, fit, check)

        def factory(model):
            return torch.optim.Adam(
                model.parameters(), lr=nd.UPDATE_LR, betas=(0.9, 0.999), eps=1e-8
            )

        coverage_child, coverage_provider, coverage_plan = nd.coverage_birth_arm(
            parent, book, fit_credits, verdicts
        )
        del coverage_child, coverage_provider
        unit_total = coverage_plan["units_total"]
        arms = {
            "coverage_birth": lambda: nd.run_arm(
                "coverage_birth",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                lambda p: nd.coverage_birth_arm(p, book, fit_credits, verdicts),
                factory,
            ),
            "gradmax_schedule": lambda: nd.run_arm(
                "gradmax_schedule",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                lambda p: nd.gradmax_arm(p, full_credits),
                factory,
            ),
            "conditional_svd": lambda: nd.run_arm(
                "conditional_svd",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                lambda p: nd.conditional_svd_arm(p, book, fit_credits, verdicts),
                factory,
            ),
            "random_gate": lambda: nd.run_arm(
                "random_gate",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                lambda p: nd.random_gate_arm(p, book, unit_total),
                factory,
            ),
            "replay_only": lambda: nd.run_arm(
                "replay_only",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                None,
                factory,
            ),
            "capacity_random": lambda: nd.run_arm(
                "capacity_random",
                copy.deepcopy(parent),
                FIT,
                HOLDOUT,
                RETENTION,
                lambda p: nd.capacity_random_arm(p, unit_total),
                factory,
            ),
        }
        results = [
            arms[name]()
            for name in (
                "coverage_birth",
                "gradmax_schedule",
                "conditional_svd",
                "random_gate",
                "replay_only",
                "capacity_random",
            )
        ]
        _CACHE = {
            "parent": parent,
            "book": book,
            "verdicts": verdicts,
            "coverage_plan": coverage_plan,
            "unit_total": unit_total,
            "arms": {row["arm"]: row for row in results},
            "ordered": results,
            "adjudication": nd.adjudicate(results),
        }
    return _CACHE


def test_r026_six_arms_run_from_same_parent():
    data = comparison()
    assert set(data["arms"]) == {
        "coverage_birth",
        "gradmax_schedule",
        "conditional_svd",
        "random_gate",
        "replay_only",
        "capacity_random",
    }
    for row in data["ordered"]:
        assert len(row["update_losses"]) == nd.UPDATE_STEPS
        assert all(loss > 0.0 for loss in row["update_losses"])
        for value in (row["nll_fit_post"], row["delta_nll_holdout"], row["drift_retention"]):
            assert value == value and abs(value) != float("inf")


def test_r026_unit_counts_match_contract():
    data = comparison()
    arms = data["arms"]
    assert arms["coverage_birth"]["units_total"] == data["unit_total"]
    assert arms["gradmax_schedule"]["units_total"] == 8
    assert arms["conditional_svd"]["units_total"] == arms["coverage_birth"]["units_total"]
    assert arms["random_gate"]["units_total"] == data["unit_total"]
    assert arms["capacity_random"]["units_total"] == data["unit_total"]
    assert arms["replay_only"]["units_total"] == 0


def test_r026_preregistered_adjudication_recorded():
    data = comparison()
    verdict = data["adjudication"]
    assert verdict["verdict"] in {"positive", "indeterminable", "negative"}
    ranked = sorted(data["ordered"], key=lambda row: row["delta_nll_holdout"])
    assert ranked[0]["arm"] in {row["arm"] for row in data["ordered"]}
    breaches = verdict["retention_breaches"]
    expected = [
        row["arm"] for row in data["ordered"] if row["drift_retention"] > nd.RETENTION_TOLERANCE
    ]
    assert breaches == expected


def test_r026_credit_diagnostics_present():
    data = comparison()
    diagnostics = nd.credit_diagnostics(
        data["parent"],
        {n: gc.branch_credit(data["book"], b) for b, n in enumerate(data["book"].branch_ids)},
        {
            n: gc.branch_credit(data["book"], b, range(len(data["book"].deltas) // 2))
            for b, n in enumerate(data["book"].branch_ids)
        },
    )
    assert 0.0 < diagnostics["kappa"] <= 1.0
    assert diagnostics["branch_group_ranks"]
