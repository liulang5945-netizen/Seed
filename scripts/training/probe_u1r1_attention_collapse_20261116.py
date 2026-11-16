"""Locate why U.1r1 question-conditioned column attention stayed at the uniform ceiling.

Zero optimizer steps. Reads only the fit partition and this card's own final capsules.
All attention numbers come from the module's own forward; the query interventions edit
W_q in place instead of recomputing logits outside the module.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_u1_unit_read_20261116 as u1
from scripts.training import run_u1r1_position_20261116 as r1
from taiji.u2_history import value_digest
from taiji.u2_neural_units import apply_capsule, load_capsule

REPORT = ROOT / "reports/taiji_u2_u1r1_attention_collapse_20261116.json"
T = r1.T
UNIFORM = math.log(T)
CAP_CPU, CAP_WALL, RESERVE = 90.0, 180.0, 10.0
UNIFORM_RATIO = 0.999
SHARP_RATIO = 0.98
FORCED_SCALE = 32.0


def attention_stats(graph: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    entropies: list[float] = []
    maxima: list[float] = []
    ranges: list[float] = []
    vectors: list[torch.Tensor] = []
    keys: list[tuple[float, ...]] = []
    hits: list[int] = []
    preds: list[int] = []
    with torch.no_grad():
        for item in episodes:
            data = u1.cast_item(item, graph.spec.torch_dtype)
            out = graph(data["channels"], data["gates"], graph.initial_state())
            attn = out["attention"]
            entropies.append(float(-(attn * torch.log(attn + 1e-12)).sum()))
            maxima.append(float(attn.max()))
            ranges.append(float(torch.log(attn.max()) - torch.log(attn.min())))
            vectors.append(attn.clone())
            keys.append(tuple(float(v) for v in attn))
            pred = int(torch.argmax(out["readout"]).item())
            preds.append(pred)
            hits.append(int(pred == item["label"]))
    count = len(episodes)
    distances = [
        float((vectors[i] - vectors[j]).abs().sum())
        for i in range(count)
        for j in range(i + 1, count)
    ]
    mean_entropy = sum(entropies) / count
    return {
        "entropy_mean": round(mean_entropy, 6),
        "entropy_min": round(min(entropies), 6),
        "entropy_max": round(max(entropies), 6),
        "entropy_fraction_of_uniform_ceiling": round(mean_entropy / UNIFORM, 6),
        "effective_columns_mean": round(sum(math.exp(e) for e in entropies) / count, 6),
        "max_attention_mean": round(sum(maxima) / count, 6),
        "max_attention_max": round(max(maxima), 6),
        "logit_range_mean": round(sum(ranges) / count, 6),
        "attention_distinct": len(set(keys)),
        "attention_pairwise_l1_mean": round(sum(distances) / len(distances), 9),
        "fit_accuracy": sum(hits) / count,
        "fit_distinct": len(set(preds)),
        "denominator": count,
    }


def trained_clone(seed: int) -> Any:
    graph = r1.build_graph(seed)
    apply_capsule(graph, load_capsule(r1.final_capsule_path(seed)))
    return graph


def w_q_norm(graph: Any) -> float:
    return float(graph.W_q.detach().norm())


def task_credit(graph: Any, item: dict[str, Any]) -> dict[str, Any]:
    data = u1.cast_item(item, graph.spec.torch_dtype)
    graph.zero_grad(set_to_none=True)
    logits = graph(data["channels"], data["gates"], graph.initial_state())["readout"]
    target = torch.tensor(data["label"], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
    loss.backward()
    credit = {
        name: float(param.grad.abs().sum())
        for name, param in graph.named_parameters()
        if param.grad is not None
    }
    return {
        "loss": round(float(loss.detach()), 6),
        "credit": credit,
        "parameters_with_gradient": len(credit),
        "parameters_total": len(list(graph.named_parameters())),
    }


def classify(rows: list[dict[str, Any]], complete: bool) -> str:
    if not complete or not rows:
        return "not_established"
    for row in rows:
        controls = row["controls"]
        if not (controls["capsule_digest_matches_run"] and controls["trained_entropy_matches_run"]):
            return "anchor_mismatch"
        if not (controls["zero_query_is_uniform"] and controls["forced_query_took_effect"]):
            return "instrument_invalid"
    trained = [row["arms"]["trained"] for row in rows]
    forced = [row["arms"]["trained_forced_query"] for row in rows]
    if not all(a["entropy_mean"] >= UNIFORM_RATIO * UNIFORM for a in trained):
        return "attention_sharpened_value_unreadable"
    if not all(a["entropy_mean"] <= SHARP_RATIO * UNIFORM for a in forced):
        return "column_keys_indistinguishable"
    if all(row["question_distinct"] == 1 for row in rows):
        return "question_channel_constant"
    if all(a["attention_distinct"] == 1 for a in trained):
        return "query_projection_insensitive_to_question"
    return "question_modulates_but_logits_flat"


def main() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    contract = json.loads(r1.CONTRACT.read_text(encoding="utf-8"))
    run_report = json.loads(r1.RUN_REPORT.read_text(encoding="utf-8"))
    if run_report["criteria_sha256"] != contract["criteria_sha256"]:
        raise ValueError("criteria_sha256_mismatch_between_contract_and_run")
    run_arms = {int(arm["seed"]): arm for arm in run_report["arms"]}
    data = u1.load_data(contract)
    episodes = u1.prepare_episodes(data, r1.UnitGraphSpec().n_units)
    fit = episodes["fit"]
    rows: list[dict[str, Any]] = []
    budget_stopped = False
    for seed in contract["criteria"]["seeds"]:
        if time.process_time() - started_cpu >= CAP_CPU - RESERVE:
            budget_stopped = True
            break
        if time.monotonic() - started_wall >= CAP_WALL - RESERVE:
            budget_stopped = True
            break
        init = r1.build_graph(seed)
        trained = trained_clone(seed)
        zero_query = trained_clone(seed)
        forced = trained_clone(seed)
        with torch.no_grad():
            zero_query.W_q.zero_()
            generator = torch.Generator().manual_seed(seed + 97)
            forced.W_q.copy_(
                FORCED_SCALE
                * torch.randn(forced.W_q.shape, generator=generator, dtype=forced.W_q.dtype)
            )
        arms = {
            "init": attention_stats(init, fit),
            "trained": attention_stats(trained, fit),
            "trained_zero_query": attention_stats(zero_query, fit),
            "trained_forced_query": attention_stats(forced, fit),
        }
        questions = [
            tuple(float(v) for v in u1.cast_item(item, torch.float64)["channels"]["question"][0])
            for item in fit
        ]
        credit = task_credit(trained, fit[0])
        run_arm = run_arms[int(seed)]
        rows.append(
            {
                "seed": int(seed),
                "arms": arms,
                "question_distinct": len(set(questions)),
                "question_nonzero_entries_mean": round(
                    sum(sum(1 for v in q if v != 0.0) for q in questions) / len(questions), 6
                ),
                "question_l2_mean": round(
                    sum(math.sqrt(sum(v * v for v in q)) for q in questions) / len(questions), 6
                ),
                "w_q_frobenius_init": round(w_q_norm(init), 9),
                "w_q_frobenius_trained": round(w_q_norm(trained), 9),
                "w_q_frobenius_forced": round(w_q_norm(forced), 9),
                "task_credit": credit,
                "controls": {
                    "capsule_digest_matches_run": value_digest(
                        load_capsule(r1.final_capsule_path(seed))
                    )
                    == run_arm["final_capsule_digest"],
                    "trained_entropy_matches_run": arms["trained"]["entropy_mean"]
                    == run_arm["attention_entropy_mean"],
                    "zero_query_is_uniform": abs(
                        arms["trained_zero_query"]["entropy_mean"] - UNIFORM
                    )
                    < 1e-6
                    and abs(arms["trained_zero_query"]["max_attention_mean"] - 1.0 / T) < 1e-9,
                    "forced_query_took_effect": w_q_norm(forced)
                    > 10.0 * max(w_q_norm(trained), 1e-12)
                    and arms["trained_forced_query"]["entropy_mean"]
                    != arms["trained"]["entropy_mean"],
                },
            }
        )
    complete = (not budget_stopped) and len(rows) == len(contract["criteria"]["seeds"])
    verdict = classify(rows, complete)
    payload = {
        "schema": "taiji-u2-u1r1-attention-collapse-v1",
        "card": "22.5.U.1r1",
        "phase": "mechanism_probe",
        "anchored_run_report": "reports/taiji_u2_u1r1_position_run_20261116.json",
        "run_verdict": run_report["verdict"],
        "criteria_sha256": contract["criteria_sha256"],
        "columns_T": T,
        "uniform_entropy_ceiling": round(UNIFORM, 6),
        "thresholds": {
            "uniform_ratio": UNIFORM_RATIO,
            "forced_sharp_ratio": SHARP_RATIO,
            "forced_scale": FORCED_SCALE,
        },
        "seeds": rows,
        "classification": verdict,
        "capability_scope": "mechanism localization for the U.1r1 negative; no capability claim",
        "discipline": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "data_file_reads": 1,
            "partitions_evaluated": ["fit"],
            "backward_passes": len(rows),
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "sealed_partitions_read": [],
            "capability_claims": 0,
        },
        "budget": {
            "cap_cpu": CAP_CPU,
            "cap_wall": CAP_WALL,
            "reserve": RESERVE,
            "budget_stopped": budget_stopped,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    r1.write_report(REPORT, payload)
    print(f"ATTENTION_PROBE classification={verdict} seeds={len(rows)} rc=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
