"""Post-hoc train/fit and complete-answer likelihood diagnosis of terminal learners."""

# ruff: noqa: E402 -- include startup in measured cost
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.first_a_support_22_5 import LABELS, evaluate, task_loss
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest


def diagnose():
    torch.set_num_threads(1)
    path = ROOT / "reports/taiji_u2_evidence_A_readout_diagnosis_20261112.json"
    if path.exists():
        raise FileExistsError(path)
    contract = json.loads(
        (
            ROOT / "plans/manifests/taiji_u2_evidence_A_posthoc_readout_contract_20261112.json"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "plans/manifests/taiji_u2_evidence_A_execution_20261112.json").read_text(
            encoding="utf-8"
        )
    )
    result_path = ROOT / "reports/taiji_u2_evidence_A_learning_20261112.json"
    recovered = result_path.with_name(result_path.stem + "_v2.json")
    if recovered.exists():
        result_path = recovered
    result = json.loads(result_path.read_text(encoding="utf-8"))
    data = atomic_load(Path(manifest["data"]["run_root"]) / "data.bin")
    assert value_digest(data) == manifest["data"]["digest"]
    records = {}
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for example in data["train"]:
        counts[example["public"]["request"].decode()][example["target"].decode()] += 1
    conditional_entropy = sum(
        -count * math.log(count / sum(row.values()))
        for row in counts.values()
        for count in row.values()
    ) / len(data["train"])
    prior = {
        question: max(LABELS, key=lambda label: row[label]) for question, row in counts.items()
    }
    question_only = {
        "scope": "Post-hoc empirical train question-class prior; excluded from model runtime and original baseline gates",
        "train_counts": dict(counts),
        "empirical_train_conditional_entropy_nats": conditional_entropy,
        "empirical_train_byte_nll_optimum_for_question_only": conditional_entropy / 7,
        "train_majority_accuracy": sum(
            example["target"].decode() == prior[example["public"]["request"].decode()]
            for example in data["train"]
        )
        / len(data["train"]),
        "fit_majority_accuracy": sum(
            example["target"].decode() == prior[example["public"]["request"].decode()]
            for example in data["fit"]
        )
        / len(data["fit"]),
        "normalization": "All four supervised words are six UTF-8 bytes plus EOS; divide sequence entropy by7",
    }

    def guard():
        if (
            time.process_time() > contract["cpu_cap"]
            or time.monotonic() - STARTED > contract["wall_cap"]
        ):
            raise RuntimeError("readout_diagnosis_budget_stop")

    with StepGuard() as blocked, torch.no_grad():
        for arm in manifest["arm_order"]:
            if arm not in result["arms"]:
                continue
            model, _, _ = load_learner(result["arms"][arm]["checkpoint"])
            parameter_digest = value_digest(model.state_dict())
            entry = {"train8_free": evaluate(model, data["train"][:8], budget_guard=guard)}
            for split in ("train", "fit"):
                rows = []
                for example in data[split][:8]:
                    scores = {}
                    for word in LABELS:
                        guard()
                        scores[word] = float(
                            task_loss(model, [{**example, "target": word.encode()}])
                        )
                    prediction = min(scores, key=lambda word: scores[word])
                    truth = example["target"].decode()
                    rows.append(
                        {
                            "id": example["id"],
                            "target": truth,
                            "ranked_answer": prediction,
                            "correct": prediction == truth,
                            "candidate_nll": scores,
                        }
                    )
                entry[f"{split}8_rescoring"] = {
                    "accuracy": sum(row["correct"] for row in rows) / len(rows),
                    "teacher_mean_nll": sum(row["candidate_nll"][row["target"]] for row in rows)
                    / len(rows),
                    "rows": rows,
                }
            assert value_digest(model.state_dict()) == parameter_digest
            records[arm] = entry
    guard()
    report = {
        "schema": "taiji-u2-evidence-A-readout-diagnosis-v1",
        "scope": contract["scope"],
        "scientific_boundary": contract["scientific_boundary"],
        "arms": records,
        "question_only_prior": question_only,
        "optimizer_steps": blocked.calls,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(path, report)
    print(
        json.dumps(
            {
                arm: {
                    "train_free": row["train8_free"]["accuracy"],
                    "train_rank": row["train8_rescoring"]["accuracy"],
                    "fit_rank": row["fit8_rescoring"]["accuracy"],
                }
                for arm, row in records.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    diagnose()
