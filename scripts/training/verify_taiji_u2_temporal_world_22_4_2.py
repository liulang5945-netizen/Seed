"""Public RGB oracle and no-history controls on declared contract fixtures."""

# ruff: noqa: E402 -- retain startup timing for the bounded fixture run
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_task_contract import score_all, validate_distinguishability
from taiji.u2_task_episode import initial_current, public_oracle, temporal_fixture, with_intent

CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_temporal_fixture_20261112.json"
REPORT = ROOT / "reports/taiji_u2_temporal_world_22_4_2_20261112.json"


def verify():
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for path, expected in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / path).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == expected
        ), path
    groups = {"A_initial_current": [], "B_previous": [], "U_unknown": []}
    for seed in contract["seeds"]:
        pair = [temporal_fixture(seed, x) for x in (-0.55, 0.55)]
        assert all(
            torch.equal(a, b) for a, b in zip(pair[0].commands, pair[1].commands, strict=True)
        )
        assert pair[0].blank_input == pair[1].blank_input
        for fixture in pair:
            groups["A_initial_current"].append(initial_current(fixture))
            groups["B_previous"].append(fixture)
            groups["U_unknown"].append(with_intent(fixture, "unknown"))
    metrics = {}
    for name, fixtures in groups.items():
        cases = tuple(f.case for f in fixtures)
        validate_distinguishability(cases)
        responses = {
            f.case.request.request_id: (public_oracle(f.case.request.raw, f.images), True)
            for f in fixtures
        }
        scores = score_all(cases, responses)
        assert scores["correct"] / scores["denominator"] >= 0.95, name
        labels = Counter(f.case.target for f in fixtures)
        metrics[name] = {
            **scores,
            "label_counts": {k.decode(): v for k, v in labels.items()},
            "constant_answer_ceiling": max(labels.values()) / len(fixtures),
        }
    previous = groups["B_previous"]
    by_input = defaultdict(Counter)
    for fixture in previous:
        by_input[fixture.blank_input][fixture.case.target] += 1
    blank_bound = sum(max(counts.values()) for counts in by_input.values()) / len(previous)
    assert blank_bound <= 0.6
    assert metrics["A_initial_current"]["constant_answer_ceiling"] <= 0.6
    wrong_time = sum(
        public_oracle(with_intent(f, "current").case.request.raw, f.images) == f.case.target
        for f in previous
    ) / len(previous)
    assert wrong_time <= 0.6
    sample = previous[0].case
    assert score_all((sample,), {sample.request.request_id: (b"\xff", True)})["correct"] == 0
    assert score_all((sample,), {sample.request.request_id: (sample.target, False)})["correct"] == 0
    assert score_all((sample,), {})["denominator"] == 1
    extra = score_all(
        (sample,), {sample.request.request_id: (sample.target, True), "extra": (b"x", True)}
    )
    assert extra["denominator"] == 2
    report = {
        "schema": "taiji-u2-temporal-fixture-verification-v1",
        "verdict": "passed_public_oracle_temporal_microfixtures",
        "contract_sha256": hashlib.sha256(
            CONTRACT.read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
        "groups": metrics,
        "blank_previous_information_ceiling": blank_bound,
        "wrong_time_accuracy": wrong_time,
        "worlds": len(previous),
        "scored_requests": sum(len(g) for g in groups.values()),
        "model_forwards": 0,
        "optimizer_steps": 0,
        "project_data_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "torch": torch.__version__, "threads": 1},
        "capability_qualifications": [],
        "pending": [
            "external curriculum action intake in the subject",
            "C acquisition utility",
            "D capacity pressure",
            "J same-subject joint sessions",
            "full world course and split admission",
        ],
    }
    assert report["cpu_seconds"] <= contract["cpu_cap"]
    if REPORT.exists():
        raise FileExistsError(REPORT)
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "groups": metrics,
                "blank_ceiling": blank_bound,
                "cpu_seconds": report["cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    verify()
