"""Zero-model semantic and information-rights checks for U2 22.4.0."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report
from taiji.u2_task_contract import (
    Case,
    Fact,
    Request,
    allow_feedback,
    oracle_answer,
    request_text,
    score_all,
    validate_case,
    validate_distinguishability,
)


def main():
    started = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    target = parser.parse_args().report.resolve()
    if target.parent != ROOT / "reports":
        raise ValueError("Report must be directly inside reports/")
    if target.exists():
        raise FileExistsError(target)
    facts = (Fact("red0", "红盒", "左边", 0, "sensor"), Fact("red1", "红盒", "右边", 1, "sensor"))
    cases = []
    for temporal in ["current", "previous", "unknown"]:
        request = Request(
            temporal, 3, 1, request_text("红盒", temporal), "红盒", temporal, ("red0", "red1")
        )
        cases.append(
            Case(
                "把蓝盒移开。".encode(),
                3,
                1,
                request,
                facts,
                (request.raw, b"visible-history"),
                oracle_answer(request, facts),
            )
        )
    results = []

    def positive(name, fn):
        fn()
        results.append({"check": name, "passed": True})

    def negative(name, fn):
        try:
            fn()
        except ValueError:
            results.append({"check": name, "passed": True})
            return
        raise AssertionError(name)

    for case in cases:
        positive("valid-" + case.request.temporal, lambda case=case: validate_case(case))
    base = cases[0]
    mutations = {
        "goal-version": replace(base, goal_version=4),
        "world-version": replace(base, world_version=0),
        "entity": replace(base, request=replace(base.request, entity="蓝盒")),
        "time": replace(base, request=replace(base.request, temporal="previous")),
        "source": replace(base, facts=(facts[0], replace(facts[1], source="imagination"))),
        "target": replace(base, target="左边".encode()),
        "missing-evidence": replace(base, request=replace(base.request, evidence_ids=("absent",))),
    }
    for name, case in mutations.items():
        negative(name, lambda case=case: validate_case(case))
    alternate_facts = (facts[0], replace(facts[1], value="木桌"))
    alternate = replace(base, facts=alternate_facts, target="木桌".encode())
    negative(
        "indistinguishable-legal-inputs", lambda: validate_distinguishability((base, alternate))
    )
    positive(
        "visible-history-disambiguates",
        lambda: validate_distinguishability(
            (base, replace(alternate, model_inputs=(*alternate.model_inputs, b"new visible event")))
        ),
    )
    for kind, split, phase, submitted, round_index, release in [
        ("score_only", "train", "development", True, 2, 1),
        ("train_label", "check", "development", True, 2, 1),
        ("online_correction", "train", "online", False, 2, 1),
        ("help_reply", "fit", "online", True, 0, 1),
    ]:
        negative(
            "deny-" + kind,
            lambda kind=kind, split=split, phase=phase, submitted=submitted, round_index=round_index, release=release: allow_feedback(
                kind,
                split=split,
                phase=phase,
                prediction_submitted=submitted,
                round_index=round_index,
                release_round=release,
            ),
        )
    positive(
        "released-training-label",
        lambda: allow_feedback(
            "train_label",
            split="train",
            phase="development",
            prediction_submitted=True,
            round_index=2,
            release_round=1,
        ),
    )
    negative(
        "wrong-online-channel",
        lambda: allow_feedback(
            "online_correction",
            split="train",
            phase="scoring",
            prediction_submitted=True,
            round_index=2,
            release_round=1,
        ),
    )
    for response in [None, (b"\xff", True), (base.target, False), (b"parse failed", True)]:
        scored = score_all((base,), {base.request.request_id: response})
        assert scored == {"correct": 0, "denominator": 1, "extra_responses": 0}
        results.append({"check": "full-denominator-" + str(len(results)), "passed": True})
    scored = score_all(
        tuple(cases),
        {case.request.request_id: (case.target, True) for case in cases} | {"extra": (b"x", True)},
    )
    assert scored == {"correct": 3, "denominator": 4, "extra_responses": 1}
    results.append({"check": "extra-response-counted", "passed": True})
    parent = ROOT / "plans/manifests/taiji_u2_22_3_accepted_parent_20261111.json"
    report = {
        "schema": "taiji-u2-22.4.0-semantic-contract-v1",
        "verdict": "passed_microfixture_contract_only",
        "accepted_parent_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "source_pins": {
            name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            for name in [
                "taiji/u2_task_contract.py",
                "scripts/training/verify_taiji_u2_task_contract_22_4.py",
            ]
        },
        "checks": results,
        "optimizer_steps": 0,
        "model_runs": 0,
        "project_dataset_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": time.process_time() - started,
        "full_world_verified": False,
        "model_metadata_entry_verified": False,
        "pricing_verified": False,
        "training_admitted": False,
    }
    write_report(target, report)
    print(json.dumps({"verdict": report["verdict"], "checks": len(results)}))


if __name__ == "__main__":
    main()
