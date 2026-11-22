"""UQ-1: uncertainty expression and handling (temperature calibration + abstain/help).

Final implementation (after disclosed pilots, D-20261122-305): each task owns a
48-dim readout on its own 24-unit block, so later task training cannot shift the
earlier task's readout input; p24's incoming connections from the p27 block are
pruned and the shared input mixing is frozen after p24 (structural isolation).
This keeps the pre-registered protocol/gates but removes a known cross-task coupling
artifact documented in the design card.

Protocol: Subject72 seed 20261120; p24 300 steps, p27 500 steps; separate calibration
and test splits; per-task temperature by NLL grid search on calibration; metrics on
test = ECE, Brier, AUROC, selective accuracy at >=70% coverage, utility with
correct=+1 / wrong=-3 / help=-0.1; controls = T=1, T on permuted labels, random help,
always answer. Single seed, <=300 CPU, zero check/final.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_24s_f1c_20261122 import capsule_state_diff
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72

ROOT = Path(r"E:\Seed")
CRITERIA = ROOT / "plans/manifests/taiji_u2_uq1_criteria_20261122.json"
OUT = ROOT / "reports/taiji_u2_uq1_20261122.json"
SEED = 20261120
FIT = 256
FIT_OFFSET = 10_000_000
BLOCK = {"p24": (24, 48), "p27": (48, 72)}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> float:
    logp = torch.log_softmax(logits / temperature, dim=-1)
    return float(-logp[torch.arange(logits.shape[0]), labels].mean())


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, grid: list[float]) -> dict:
    rows = [{"T": t, "nll": nll(logits, labels, t)} for t in grid]
    best = min(rows, key=lambda r: r["nll"])
    return {"T": best["T"], "nll": best["nll"]}


def probability(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).float()
    total = probs.shape[0]
    edges = torch.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        value += count / total * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return value


def brier(probs: torch.Tensor, labels: torch.Tensor) -> float:
    onehot = F.one_hot(labels, num_classes=probs.shape[-1]).float()
    return float(((probs - onehot) ** 2).sum(dim=-1).mean())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    sorted_vals = values[order]
    i = 0
    n = values.numel()
    while i < n:
        j = i
        while j + 1 < n and float(sorted_vals[j + 1]) == float(sorted_vals[i]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def auroc(conf: torch.Tensor, correct: torch.Tensor) -> float:
    pos = correct.bool()
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(_average_ranks(conf)[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def selective_metrics(probs: torch.Tensor, labels: torch.Tensor, tau: float) -> dict:
    conf, pred = probs.max(dim=-1)
    correct = pred == labels
    answered = conf >= tau
    coverage = float(answered.float().mean())
    always_acc = float(correct.float().mean())
    sel_acc = float(correct[answered].float().mean()) if int(answered.sum()) else 0.0
    errors = ~correct
    error_catch = float((errors & (~answered)).sum()) / max(1, int(errors.sum()))
    return {
        "tau": float(tau),
        "coverage": coverage,
        "always_acc": always_acc,
        "selective_acc": sel_acc,
        "selective_gain": sel_acc - always_acc,
        "help_rate": 1.0 - coverage,
        "error_catch": error_catch,
        "n_errors": int(errors.sum()),
    }


def utility_metrics(probs: torch.Tensor, labels: torch.Tensor, tau: float, weights: dict) -> dict:
    conf, pred = probs.max(dim=-1)
    correct = pred == labels
    reward = torch.where(
        correct,
        torch.full_like(conf, float(weights["correct"])),
        torch.full_like(conf, float(weights["wrong"])),
    )
    always = float(reward.mean())
    answered = conf >= tau
    conf_help = float(
        torch.where(answered, reward, torch.full_like(conf, float(weights["help"]))).mean()
    )
    help_rate = float((~answered).float().mean())
    random_help = float(((1.0 - help_rate) * reward + help_rate * float(weights["help"])).mean())
    return {
        "always_answer": always,
        "confidence_help": conf_help,
        "random_help_matched": random_help,
        "gain_vs_always": conf_help - always,
        "gain_vs_random": conf_help - random_help,
        "help_rate": help_rate,
    }


def task_metrics(logits, labels, calib, crit) -> dict:
    grid_spec = crit["protocol"]["temperature_grid"]
    grid = [
        round(grid_spec[0] + i * grid_spec[2], 2)
        for i in range(int((grid_spec[1] - grid_spec[0]) / grid_spec[2]) + 1)
    ]
    cal_logits, cal_labels = calib
    fit = fit_temperature(cal_logits, cal_labels, grid)
    perm = fit_temperature(cal_logits, (cal_labels + 1) % 4, grid)
    probs_fit = probability(logits, fit["T"])
    probs_t1 = probability(logits, 1.0)
    probs_perm = probability(logits, perm["T"])
    conf_calib = probability(cal_logits, fit["T"]).max(dim=-1).values
    tau = float(torch.quantile(conf_calib, 1.0 - crit["protocol"]["coverage_target"]))
    correct = probs_fit.max(dim=-1).indices == labels
    return {
        "temperature_fit": fit["T"],
        "temperature_permuted": perm["T"],
        "ece": ece(probs_fit, labels, crit["protocol"]["ece_bins"]),
        "ece_T1": ece(probs_t1, labels, crit["protocol"]["ece_bins"]),
        "ece_permuted": ece(probs_perm, labels, crit["protocol"]["ece_bins"]),
        "brier": brier(probs_fit, labels),
        "brier_T1": brier(probs_t1, labels),
        "nll": nll(logits, labels, fit["T"]),
        "nll_T1": nll(logits, labels, 1.0),
        "auroc": auroc(probs_fit.max(dim=-1).values, correct),
        "mean_confidence": float(probs_fit.max(dim=-1).values.mean()),
        "accuracy": float(correct.float().mean()),
        "selective": selective_metrics(probs_fit, labels, tau),
        "utility": utility_metrics(probs_fit, labels, tau, crit["protocol"]["utility"]),
    }


# ---------------------------------------------------------------------------
# task-specific readout on the task's own 24-unit block
# ---------------------------------------------------------------------------
def own_logits(subj, task: str, ticks: torch.Tensor) -> torch.Tensor:
    mask = subj.masks[task].view(1, 1, -1).expand(ticks.shape[0], ticks.shape[1], subj.n_units)
    out = subj.graph(
        _channels_seq(ticks), mask, subj.graph.initial_state(batch_size=ticks.shape[0])
    )
    lo, hi = BLOCK[task]
    z, a = out["next_state"]["z"][:, lo:hi], out["next_state"]["a"][:, lo:hi]
    return subj.phase_heads[f"{task}u"](torch.cat([z, a], dim=-1))


def train(subj, opt, task: str, steps: int) -> None:
    subj.train_mode()
    for k in range(steps):
        if task == "p24":
            b = _make_p24_stream(48, SEED * 25000 + k)
            labels = b["val_labels"]
            ticks = b["ticks"]
        else:
            b = _make_p27_physics_batch(48, SEED * 41000 + k)
            labels = b["labels"]
            ticks = b["ticks"]
        loss = F.cross_entropy(own_logits(subj, task, ticks), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_grads(subj, task)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()


def capture(subj, task: str, ticks: torch.Tensor) -> torch.Tensor:
    subj.eval_mode()
    with torch.no_grad():
        return own_logits(subj, task, ticks)


def main() -> int:
    started = time.process_time()
    crit = json.loads(CRITERIA.read_text(encoding="utf-8"))
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    digest = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == crit["criteria_sha256"], "criteria_sha256_mismatch"
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    subj = Subject72(SEED)
    subj.set_masks(True)
    for name, (lo, hi) in {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}.items():
        subj.graph.set_task_route(name, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for head in subj.phase_heads.values():
        for p in head.parameters():
            p.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 401)
        subj.phase_heads["p24u"] = PhaseHead(48, 4, 64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + 402)
        subj.phase_heads["p27u"] = PhaseHead(48, 4, 64)
    for head in (subj.phase_heads["p24u"],):
        for p in head.parameters():
            p.requires_grad_(True)
    for p in subj.graph.selective_units.parameters():
        p.requires_grad_(True)
    # structural isolation of the p24 block: it reads only its own units plus inputs
    with torch.no_grad():
        units = subj.graph.selective_units
        for cols in (slice(0, 24), slice(48, 72)):
            units.w[24:48, cols] = 0.0
            units.write_state.weight[24:48, cols] = 0.0

    lr, wd = crit["protocol"]["lr"], crit["protocol"]["wd"]
    opt24 = torch.optim.Adam(subj.params(), lr=lr, weight_decay=wd)
    train(subj, opt24, "p24", crit["protocol"]["p24_steps"])
    p24_calib = _make_p24_stream(FIT, SEED + crit["protocol"]["p24_calib_seed_offset"] + FIT_OFFSET)
    p24_test = _make_p24_stream(FIT, SEED + crit["protocol"]["p24_test_seed_offset"] + FIT_OFFSET)
    p24_after_p24 = float(
        (capture(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["val_labels"])
        .float()
        .mean()
    )

    # ---------- protection: isolate p24 from all other blocks and later training ----------
    for p in subj.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for p in subj.phase_heads["p24u"].parameters():
        p.requires_grad_(False)
    for p in subj.phase_heads["p27u"].parameters():
        p.requires_grad_(True)
    opt27 = torch.optim.Adam(subj.params(), lr=lr, weight_decay=wd)
    train(subj, opt27, "p27", crit["protocol"]["p27_steps"])

    p27_calib = _make_p27_physics_batch(
        FIT, SEED + crit["protocol"]["p27_calib_seed_offset"] + FIT_OFFSET
    )
    p27_test = _make_p27_physics_batch(
        FIT, SEED + crit["protocol"]["p27_test_seed_offset"] + FIT_OFFSET
    )
    p24_retention = float(
        (capture(subj, "p24", p24_test["ticks"]).argmax(-1) == p24_test["val_labels"])
        .float()
        .mean()
    )
    p27_acc = float(
        (capture(subj, "p27", p27_test["ticks"]).argmax(-1) == p27_test["labels"]).float().mean()
    )
    tasks = {
        "p24": task_metrics(
            capture(subj, "p24", p24_test["ticks"]),
            p24_test["val_labels"],
            (capture(subj, "p24", p24_calib["ticks"]), p24_calib["val_labels"]),
            crit,
        ),
        "p27": task_metrics(
            capture(subj, "p27", p27_test["ticks"]),
            p27_test["labels"],
            (capture(subj, "p27", p27_calib["ticks"]), p27_calib["labels"]),
            crit,
        ),
    }
    capsule_diff = capsule_state_diff(subj, p24_test["ticks"][:32], torch.ones(subj.n_units))
    retention_drop = max(0.0, p24_after_p24 - p24_retention)

    gates = {
        "G0": tasks["p24"]["accuracy"] >= crit["gates"]["G0_p24_acc_min"]
        and tasks["p27"]["accuracy"] >= crit["gates"]["G0_p27_acc_min"],
        "G1": all(
            t["ece"] <= crit["gates"]["G1_ece_max"]
            and t["brier"] <= t["brier_T1"] - crit["gates"]["G1_brier_gain_min"]
            for t in tasks.values()
        ),
        "G2": all(t["auroc"] >= crit["gates"]["G2_auroc_min"] for t in tasks.values()),
        "G3": all(
            t["selective"]["coverage"] >= crit["gates"]["G3_coverage_min"]
            and t["selective"]["selective_gain"] >= crit["gates"]["G3_selective_gain_min"]
            for t in tasks.values()
        ),
        "G4": all(
            t["utility"]["gain_vs_always"] >= crit["gates"]["G4_utility_gain_vs_always_min"]
            and t["utility"]["gain_vs_random"] >= crit["gates"]["G4_utility_gain_vs_random_min"]
            for t in tasks.values()
        ),
        "G5": all(
            t["ece"] <= t["ece_permuted"] - crit["gates"]["G5_ece_drop_vs_permuted_min"]
            for t in tasks.values()
        ),
        "G6": capsule_diff <= crit["gates"]["G6_capsule_diff_max"]
        and retention_drop <= crit["gates"]["G6_p24_retention_drop_max"],
    }
    if all(gates.values()):
        verdict = "uq1_verified"
    elif gates["G0"] and gates["G1"]:
        verdict = "uq1_calibrated_only"
    elif gates["G0"]:
        verdict = "uq1_partial"
    else:
        verdict = "uq1_baseline_insufficient"
    cost = round(time.process_time() - started, 2)
    if cost > crit["budget_cpu"]:
        verdict = "budget_stop"
    payload = {
        "schema": "taiji-u2-uq1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "criteria_sha256": crit["criteria_sha256"],
        "authority": crit["authority"],
        "seed": SEED,
        "implementation": {
            "readout": "per-task 48-dim head on own 24-unit block",
            "protection": "freeze cross_channel_mix + prune p24 incoming from p27 after p24",
        },
        "training": {
            "p24_steps": crit["protocol"]["p24_steps"],
            "p27_steps": crit["protocol"]["p27_steps"],
            "p24_accuracy_after_p24": p24_after_p24,
            "p24_accuracy_after_p27": p24_retention,
            "p27_accuracy": p27_acc,
        },
        "tasks": tasks,
        "capsule_state_diff": capsule_diff,
        "retention_drop": retention_drop,
        "gates": gates,
        "verdict": verdict,
        "run_conditions": {"threads": torch.get_num_threads()},
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": cost},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "training": payload["training"],
                "summary": {
                    name: {
                        "acc": round(t["accuracy"], 4),
                        "T": t["temperature_fit"],
                        "T_perm": t["temperature_permuted"],
                        "ece": round(t["ece"], 4),
                        "ece_T1": round(t["ece_T1"], 4),
                        "ece_perm": round(t["ece_permuted"], 4),
                        "brier": round(t["brier"], 4),
                        "brier_T1": round(t["brier_T1"], 4),
                        "auroc": round(t["auroc"], 4),
                        "coverage": round(t["selective"]["coverage"], 4),
                        "sel_acc": round(t["selective"]["selective_acc"], 4),
                        "gain": round(t["selective"]["selective_gain"], 4),
                        "catch": round(t["selective"]["error_catch"], 4),
                        "util_always": round(t["utility"]["always_answer"], 4),
                        "util_conf": round(t["utility"]["confidence_help"], 4),
                        "util_random": round(t["utility"]["random_help_matched"], 4),
                    }
                    for name, t in tasks.items()
                },
                "capsule_diff": capsule_diff,
                "retention_drop": retention_drop,
                "cpu": cost,
                "criteria_sha256": crit["criteria_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
