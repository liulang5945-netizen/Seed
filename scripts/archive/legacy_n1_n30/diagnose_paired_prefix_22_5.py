"""Diagnose privileged teacher history versus the learner's actual prior answer."""

# ruff: noqa: E402 -- include startup costs
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.first_a_support_22_5 import LABELS, accept, output
from scripts.training.paired_episode_22_5 import next_request, start_episode
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest


def replay(model, state, content, *, verify_greedy=False):
    for symbol in [*content, 256]:
        logp, _, details = output(model, state)
        if verify_greedy:
            assert int(logp.argmax()) == symbol
        state = accept(model, state, symbol, details)
    return state


def rank(model, state, guard):
    scores = {}
    for label in LABELS:
        local, terms = state, []
        for symbol in [*label.encode(), 256]:
            guard()
            logp, _, details = output(model, local)
            terms.append(-logp[symbol])
            local = accept(model, local, symbol, details)
        scores[label] = float(torch.stack(terms).mean())
    return {"word": min(scores, key=lambda word: scores[word]), "scores": scores}


def diagnose():
    torch.set_num_threads(1)
    path = ROOT / "reports/taiji_u2_paired_prefix_diagnosis_20261113.json"
    if path.exists():
        raise FileExistsError(path)
    contract = json.loads(
        (
            ROOT / "plans/manifests/taiji_u2_paired_prefix_diagnosis_contract_20261113.json"
        ).read_text(encoding="utf-8")
    )
    admission = json.loads(
        (ROOT / "plans/manifests/taiji_u2_paired_learning_admission_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        (ROOT / "reports/taiji_u2_paired_learning_20261113.json").read_text(encoding="utf-8")
    )
    data = atomic_load(admission["data"]["data_path"])
    assert value_digest(data) == admission["data"]["data_digest"]

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("paired_prefix_diagnosis_budget_stop")

    profiles = {}
    with StepGuard() as blocked, torch.no_grad():
        for name in admission["learners"]:
            if name not in result["results"]:
                continue
            endpoint = result["results"][name]
            model, _, _ = load_learner(endpoint["checkpoint"])
            digest = value_digest(model.state_dict())
            record = {}
            for split in ("train", "dev"):
                actual_first = {
                    row["episode_id"]: row
                    for row in endpoint["final"][split]["rows"]
                    if row["turn"] == 0
                }
                rows = []
                for episode in data[split]["episodes"]:
                    if episode["audit"]["position"] != 1:
                        continue
                    guard()
                    first = start_episode(model, data[split], episode)
                    first_ranking = rank(model, first, guard)
                    teacher = replay(model, first, episode["targets"][0])
                    teacher = next_request(model, teacher, episode, 1)
                    teacher_ranking = rank(model, teacher, guard)
                    raw = actual_first[episode["id"]]
                    actual_ranking = None
                    if raw["eos"]:
                        actual = replay(
                            model, first, bytes.fromhex(raw["output_hex"]), verify_greedy=True
                        )
                        actual = next_request(model, actual, episode, 1)
                        actual_ranking = rank(model, actual, guard)
                    rows.append(
                        {
                            "episode_id": episode["id"],
                            "first_target": episode["targets"][0].decode(),
                            "second_target": episode["targets"][1].decode(),
                            "first_rank": first_ranking,
                            "second_teacher_rank": teacher_ranking,
                            "second_actual_rank": actual_ranking,
                        }
                    )
                assert len(rows) == 60
                record[split] = {
                    "episodes": 60,
                    "first_rank_accuracy": sum(
                        row["first_rank"]["word"] == row["first_target"] for row in rows
                    )
                    / 60,
                    "second_teacher_rank_accuracy": sum(
                        row["second_teacher_rank"]["word"] == row["second_target"] for row in rows
                    )
                    / 60,
                    "second_actual_rank_accuracy": sum(
                        row["second_actual_rank"] is not None
                        and row["second_actual_rank"]["word"] == row["second_target"]
                        for row in rows
                    )
                    / 60,
                    "rows": rows,
                }
            assert value_digest(model.state_dict()) == digest
            profiles[name] = record
    guard()
    receipt = {
        "schema": "taiji-u2-paired-prefix-diagnosis-v1",
        "scope": contract["scope"],
        "scientific_boundary": contract["scientific_boundary"],
        "profiles": profiles,
        "optimizer_steps": blocked.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(path, receipt)
    print(
        json.dumps(
            {
                name: {
                    split: {key: value for key, value in summary.items() if key != "rows"}
                    for split, summary in record.items()
                }
                for name, record in profiles.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    diagnose()
