"""Post-hoc, read-only loss/ranking/rollout attribution on frozen pilot weights."""

# ruff: noqa: E402 -- record startup cost
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import LABELS, evaluate, load_learner, task_loss
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import atomic_load, value_digest


def diagnose():
    torch.set_num_threads(1)
    result = json.loads(
        (ROOT / "reports/taiji_u2_first_A_pilot_20261112.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "plans/manifests/taiji_u2_first_A_execution_20261112_v3.json").read_text(
            encoding="utf-8"
        )
    )
    data = atomic_load(Path(manifest["run_root"]) / "data.bin")
    rows = {}
    with StepGuard() as guard, torch.no_grad():
        for architecture, trained in result["models"].items():
            model, _, _ = load_learner(trained["checkpoint"])
            parameter_id = value_digest(model.state_dict())
            record = {}
            for split, examples in [("train16", data["train"][:16]), ("fit32", data["fit"])]:
                ranks = []
                nll = []
                image_cut = []
                for example in examples:
                    scores = {}
                    for label in LABELS:
                        alternative = {**example, "target": label.encode()}
                        scores[label] = float(task_loss(model, [alternative]))
                    truth = example["target"].decode()
                    best = min(scores, key=scores.get)
                    cut = copy.deepcopy(example)
                    cut["public"]["observations"]["rgb"].zero_()
                    nll.append(scores[truth])
                    image_cut.append(float(task_loss(model, [cut])) - scores[truth])
                    ranks.append(
                        {
                            "id": example["id"],
                            "truth": truth,
                            "ranked_answer": best,
                            "scores": scores,
                            "correct": best == truth,
                        }
                    )
                record[split] = {
                    "teacher_mean_nll": sum(nll) / len(nll),
                    "candidate_rescoring_accuracy": sum(r["correct"] for r in ranks) / len(ranks),
                    "image_cut_nll_increase": sum(image_cut) / len(image_cut),
                    "rows": ranks,
                }
            record["train16_free_generation"] = evaluate(model, data["train"][:16])
            assert parameter_id == value_digest(model.state_dict())
            rows[architecture] = record
    report = {
        "schema": "taiji-first-A-readonly-attribution-v1",
        "scope": "post-hoc diagnostic; four-answer likelihood rescoring is not native free-generation ability",
        "models": rows,
        "optimizer_steps": guard.calls,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    path = ROOT / "reports/taiji_u2_first_A_attribution_20261112.json"
    if path.exists():
        raise FileExistsError(path)
    write_report(path, report)
    print(
        json.dumps(
            {
                name: {
                    "train_free": r["train16_free_generation"]["accuracy"],
                    "train_rank": r["train16"]["candidate_rescoring_accuracy"],
                    "fit_rank": r["fit32"]["candidate_rescoring_accuracy"],
                    "image_effect": r["fit32"]["image_cut_nll_increase"],
                }
                for name, r in rows.items()
            }
        )
    )


if __name__ == "__main__":
    diagnose()
