"""N31.1.03: task-agnostic learned routing on the GRU+read_gate backbone (exploration).

Pre-registered in plans/reference/DESIGN-N31-1-02_20261124.md (D-20261124-490).
All arms are task-agnostic at evaluation: the forward pass receives ticks only.

  R2A  GRU+read_gate, 24 shared units + 6 routed experts x 8 units; learned input router
       (noisy top-2, straight-through); union head; fixed replay buffer; load balance
  C2   same backbone, all 72 units always active, no router, same union head and replay
       (after D-483 the backbone is GRU+read_gate, so C2 is also the C5 strong baseline)
  C1   C2 without replay (sequential fine-tuning; forgetting upper bound for cell03)
  C3   C1 plus EWC anchoring; lambda selected on the tune split from LAMBDAS

Order CN -> P24 (P-B) -> P27 (P-D), STEPS per task, Adam, gradient clip 1.0.
Exploration seeds only; 0 check / 0 final.

Usage:
  python scripts/training/n31_1_03_routed_gru.py <seed> --smoke
  python scripts/training/n31_1_03_routed_gru.py <seed> --arms R2A,C2,C1,C3
  python scripts/training/n31_1_03_routed_gru.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.training.n31_guard import preflight  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)

CARD = "N31.1.03"
SCHEMA = "taiji-n31-1-03-v1"
REPORTS = ROOT / "reports"
TASKS = ("cn", "p24", "p27")
METRICS = ("cn", "p24", "conflict", "p27")
GATES = {"cn": 0.85, "p24": 0.75, "conflict": 0.85, "p27": 0.70}
OFFSET = {"cn": 0, "p24": 8, "p27": 12}
N_CLASS = {"cn": 8, "p24": 4, "p27": 4}
N_UNION = 16
INPUT = 48
HIDDEN = 72
N_SHARED = 24
N_EXPERTS = 6
EXPERT_UNITS = 8
TOP_K = 1  # N31.1.03: single-variable change (was 2 in N31.1.02)
READOUT_HIDDEN = 64
STEPS = 1200
SMOKE_STEPS = 40
LR = {"cn": 6e-3, "p24": 7e-3, "p27": 7e-3}
BATCH = 48
REPLAY_N = 256
REPLAY_BATCH = 16
LB_COEF = 0.01
ROUTER_NOISE = 1.0
LAMBDAS = (1.0, 30.0, 1000.0)
EVAL_N = 256
FISHER_BATCHES = 16
FISHER_BATCH = 16
ARMS = ("R2A", "C2", "C1", "C3")
ARM_CFG = {"R2A": (True, True), "C2": (False, True), "C1": (False, False), "C3": (False, False)}
CARD_CPU_CAP = 1500.0
SEED_CPU_CAP = 575.0
EXPLORATION_SEEDS = (20261120, 20261121, 20261122)
EVAL_OFFSET = {
    "fit": {"cn": 10_000_700, "p24": 10_000_212, "p27": 10_000_081},
    "tune": {"cn": 20_000_333, "p24": 20_000_334, "p27": 20_000_335},
}
PARTNER = {"cn": "p24", "p24": "p27", "p27": "cn"}
SHARED_ROWS = torch.tensor([g * HIDDEN + j for g in range(3) for j in range(N_SHARED)])

assert N_SHARED + N_EXPERTS * EXPERT_UNITS == HIDDEN


class CapExceeded(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


# ----------------------------------------------------------------------------- data


def make_batch(task: str, n: int, seed: int, *, eval_templates: bool = False) -> dict[str, Any]:
    if task == "cn":
        if eval_templates:
            b = _build_bidirectional_cn_32dim(n, seed, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND)
        else:
            b = _build_bidirectional_cn_32dim(n, seed, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
        return {"ticks": b["ticks"], "labels": b["labels"], "conflict": torch.zeros(n)}
    if task == "p24":
        b = _make_p24_stream(n, seed)
        return {"ticks": b["ticks"], "labels": b["val_labels"], "conflict": b["conflict_labels"]}
    b = _make_p27_physics_batch(n, seed)
    return {"ticks": b["ticks"], "labels": b["labels"], "conflict": torch.zeros(n)}


def train_seed(task: str, seed: int, k: int) -> int:
    return {"cn": seed * 6000, "p24": seed * 25000, "p27": seed * 41000}[task] + k


def replay_seed(task: str, seed: int) -> int:
    return seed * 977 + 5_000_000 + 1000 * TASKS.index(task)


def fisher_seed(task: str, seed: int, b: int) -> int:
    return seed * 977 + 7_000_000 + 1000 * TASKS.index(task) + 50 * b


def eval_sets(seed: int, split: str) -> dict[str, dict[str, Any]]:
    off = EVAL_OFFSET[split]
    return {t: make_batch(t, EVAL_N, seed + off[t], eval_templates=True) for t in TASKS}


def valid_mask(ticks: torch.Tensor) -> torch.Tensor:
    return (ticks.abs().sum(-1) > 0).float()


# ---------------------------------------------------------------------------- model


class RoutedGRU(nn.Module):
    """GRU cell over 72 units with the final-tick read_gate; optional learned unit routing."""

    def __init__(self, routed: bool) -> None:
        super().__init__()
        self.routed = routed
        self.cell = nn.GRUCell(INPUT, HIDDEN)
        self.read_gate = nn.Linear(INPUT, HIDDEN)
        nn.init.constant_(self.read_gate.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, READOUT_HIDDEN), nn.Tanh(), nn.Linear(READOUT_HIDDEN, N_UNION + 1)
        )
        # Created last so every other parameter has the same init across arms.
        self.router = nn.Linear(INPUT, N_EXPERTS) if routed else None

    def route(
        self, ticks: torch.Tensor, valid: torch.Tensor, *, noisy: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.router is not None
        pooled = (ticks * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1.0)
        logits = self.router(pooled)
        if noisy:
            logits = logits + torch.randn_like(logits) * ROUTER_NOISE
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(TOP_K, dim=-1).indices
        hard = torch.zeros_like(probs).scatter(1, top, 1.0)
        gate = hard * (1.0 + probs - probs.detach())
        return gate, probs, hard

    def forward(
        self,
        ticks: torch.Tensor,
        *,
        noisy: bool = False,
        expert_gate: torch.Tensor | None = None,
        shared_on: bool = True,
    ) -> dict[str, Any]:
        batch, horizon, _ = ticks.shape
        valid = valid_mask(ticks)
        probs: torch.Tensor | None = None
        hard: torch.Tensor | None = None
        mask = torch.ones(batch, HIDDEN)
        if self.routed:
            if expert_gate is None:
                expert_gate, probs, hard = self.route(ticks, valid, noisy=noisy)
            experts = expert_gate.repeat_interleave(EXPERT_UNITS, dim=1)
            shared = torch.full((batch, N_SHARED), 1.0 if shared_on else 0.0)
            mask = torch.cat([shared, experts], dim=1)
        elif not shared_on:
            mask[:, :N_SHARED] = 0.0
        h = torch.zeros(batch, HIDDEN)
        for t in range(horizon):
            new = self.cell(ticks[:, t], h) * mask
            v = valid[:, t : t + 1]
            h = v * new + (1.0 - v) * h
        last = (valid * torch.arange(horizon, dtype=valid.dtype)).argmax(-1)
        x_last = ticks[torch.arange(batch), last]
        out = h * torch.sigmoid(self.read_gate(x_last))
        return {"logits": self.head(out), "probs": probs, "hard": hard}


def task_loss(
    model: RoutedGRU, task: str, data: dict[str, Any], *, noisy: bool
) -> tuple[torch.Tensor, dict[str, Any]]:
    out = model(data["ticks"], noisy=noisy)
    lg = out["logits"]
    loss = F.cross_entropy(lg[:, :N_UNION], data["labels"] + OFFSET[task])
    if task == "p24":
        loss = loss + F.binary_cross_entropy_with_logits(lg[:, N_UNION], data["conflict"])
    return loss, out


def load_balance(probs: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    frac = hard.mean(0) / TOP_K
    return N_EXPERTS * (frac * probs.mean(0)).sum()


def shared_grad_norm(model: RoutedGRU) -> float:
    total = 0.0
    for p in (model.cell.weight_ih, model.cell.weight_hh, model.cell.bias_ih, model.cell.bias_hh):
        if p.grad is not None:
            total += float(p.grad[SHARED_ROWS].pow(2).sum())
    return math.sqrt(total)


def fisher_anchor(
    model: RoutedGRU, task: str, seed: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    params = dict(model.named_parameters())
    fisher = {n: torch.zeros_like(p) for n, p in params.items()}
    for b in range(FISHER_BATCHES):
        data = make_batch(task, FISHER_BATCH, fisher_seed(task, seed, b))
        model.zero_grad(set_to_none=True)
        loss, _ = task_loss(model, task, data, noisy=False)
        loss.backward()
        for n, p in params.items():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2) / FISHER_BATCHES
    mean = torch.cat([f.flatten() for f in fisher.values()]).mean().clamp(min=1e-12)
    model.zero_grad(set_to_none=True)
    return {n: f / mean for n, f in fisher.items()}, {
        n: p.detach().clone() for n, p in params.items()
    }


def ewc_penalty(
    model: RoutedGRU,
    anchors: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]],
    lam: float,
) -> torch.Tensor:
    total = torch.zeros(())
    for fisher, star in anchors:
        for n, p in model.named_parameters():
            total = total + (fisher[n] * (p - star[n]).pow(2)).sum()
    return 0.5 * lam * total


# ----------------------------------------------------------------------------- eval


def evaluate(
    model: RoutedGRU,
    sets: dict[str, dict[str, Any]],
    tasks: tuple[str, ...],
    *,
    gate_fn: Any = None,
    shared_on: bool = True,
) -> dict[str, float]:
    res: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for task in tasks:
            d = sets[task]
            gate = gate_fn(task) if gate_fn is not None else None
            lg = model(d["ticks"], expert_gate=gate, shared_on=shared_on)["logits"]
            pred = lg[:, :N_UNION].argmax(-1)
            lo = OFFSET[task]
            res[task] = float((pred == d["labels"] + lo).float().mean())
            in_range = (pred >= lo) & (pred < lo + N_CLASS[task])
            res[f"{task}_task_id"] = float(in_range.float().mean())
            if task == "p24":
                hit = (lg[:, N_UNION] > 0).float() == d["conflict"]
                res["conflict"] = float(hit.float().mean())
    return res


def min_task(res: dict[str, float]) -> float:
    return min(res[m] for m in METRICS)


def mean3(res: dict[str, float]) -> float:
    return sum(res[t] for t in TASKS) / len(TASKS)


def route_controls(model: RoutedGRU, fit: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed + 5151)

    def random_gate(task: str) -> torch.Tensor:
        n = fit[task]["ticks"].shape[0]
        top = torch.rand(n, N_EXPERTS, generator=gen).topk(TOP_K, dim=-1).indices
        return torch.zeros(n, N_EXPERTS).scatter(1, top, 1.0)

    def learned_hard(task: str) -> torch.Tensor:
        ticks = fit[task]["ticks"]
        with torch.no_grad():
            _, _, hard = model.route(ticks, valid_mask(ticks), noisy=False)
        return hard

    def permuted_gate(task: str) -> torch.Tensor:
        return learned_hard(PARTNER[task])

    usage = {t: [round(float(x), 4) for x in learned_hard(t).mean(0)] for t in TASKS}
    return {
        "random_route": evaluate(model, fit, TASKS, gate_fn=random_gate),
        "permuted_route": evaluate(model, fit, TASKS, gate_fn=permuted_gate),
        "shared_off": evaluate(model, fit, TASKS, shared_on=False),
        "route_usage": usage,
    }


# ---------------------------------------------------------------------------- train


def run_arm(
    arm: str,
    seed: int,
    steps: int,
    sets: dict[str, dict[str, dict[str, Any]]],
    deadline: float,
    ewc_lambda: float | None = None,
) -> dict[str, Any]:
    routed, replay = ARM_CFG[arm]
    torch.manual_seed(seed + 9001)
    model = RoutedGRU(routed)
    gen = torch.Generator().manual_seed(seed + 4242)
    buffers: dict[str, dict[str, Any]] = {}
    anchors: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
    after_phase: dict[str, dict[str, float]] = {}
    shared_grad: dict[str, float] = {}
    t0 = time.process_time()
    for phase, task in enumerate(TASKS):
        opt = torch.optim.Adam(model.parameters(), lr=LR[task])
        gsum = 0.0
        for k in range(steps):
            if time.process_time() > deadline:
                raise CapExceeded(f"{arm} phase {task} step {k}")
            model.train()
            loss, out = task_loss(
                model, task, make_batch(task, BATCH, train_seed(task, seed, k)), noisy=True
            )
            probs, hards = [out["probs"]], [out["hard"]]
            if replay:
                for old in TASKS[:phase]:
                    idx = torch.randint(0, REPLAY_N, (REPLAY_BATCH,), generator=gen)
                    rb = {key: val[idx] for key, val in buffers[old].items()}
                    r_loss, r_out = task_loss(model, old, rb, noisy=True)
                    loss = loss + r_loss
                    probs.append(r_out["probs"])
                    hards.append(r_out["hard"])
            if routed and phase > 0:
                loss = loss + LB_COEF * load_balance(torch.cat(probs), torch.cat(hards))
            if ewc_lambda is not None and anchors:
                loss = loss + ewc_penalty(model, anchors, ewc_lambda)
            if not torch.isfinite(loss):
                raise NonFiniteLoss(f"{arm} phase {task} step {k}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gsum += shared_grad_norm(model)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        shared_grad[task] = gsum / max(steps, 1)
        after_phase[task] = evaluate(model, sets["fit"], TASKS[: phase + 1])
        if replay:
            buffers[task] = make_batch(task, REPLAY_N, replay_seed(task, seed))
        if ewc_lambda is not None:
            anchors.append(fisher_anchor(model, task, seed))
    final = after_phase[TASKS[-1]]
    bwt = {
        "cn": final["cn"] - after_phase["cn"]["cn"],
        "p24": final["p24"] - after_phase["p24"]["p24"],
        "conflict": final["conflict"] - after_phase["p24"]["conflict"],
    }
    row: dict[str, Any] = {
        "status": "complete",
        "final": final,
        "after_phase": after_phase,
        "bwt": bwt,
        "forgetting": sum(max(0.0, -v) for v in bwt.values()) / len(bwt),
        "shared_grad_mean": shared_grad,
        "min_task": min_task(final),
        "mean3": mean3(final),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    if routed:
        row.update(route_controls(model, sets["fit"], seed))
    if ewc_lambda is not None:
        row["lambda"] = ewc_lambda
        row["tune_final"] = evaluate(model, sets["tune"], TASKS)
    row["cpu"] = round(time.process_time() - t0, 2)
    return row


def run_c3(
    seed: int, steps: int, sets: dict[str, Any], deadline: float, lambdas: tuple[float, ...]
) -> dict[str, Any]:
    runs = [run_arm("C3", seed, steps, sets, deadline, ewc_lambda=lam) for lam in lambdas]
    best = max(runs, key=lambda r: sum(r["tune_final"][m] for m in METRICS))
    chosen = dict(best)
    chosen["all_lambdas"] = [
        {"lambda": r["lambda"], "tune": r["tune_final"], "fit": r["final"], "cpu": r["cpu"]}
        for r in runs
    ]
    chosen["cpu"] = round(sum(r["cpu"] for r in runs), 2)
    return chosen


# ------------------------------------------------------------------------ indicators


def seed_indicators(arms: dict[str, Any]) -> dict[str, Any]:
    r = arms.get("R2A", {})
    if r.get("status") != "complete":
        return {"available": False}
    fin = r["final"]
    ind: dict[str, Any] = {"available": True, "min_task": min_task(fin)}
    ind["g01_margin"] = min(fin[m] - GATES[m] for m in METRICS)
    ind["g01"] = ind["g01_margin"] >= 0.0
    drop_rand = min_task(fin) - min_task(r["random_route"])
    drop_perm = min_task(fin) - min_task(r["permuted_route"])
    ind["g02_drop"] = {"random": drop_rand, "permuted": drop_perm}
    ind["g02"] = min(drop_rand, drop_perm) >= 0.20
    c1 = arms.get("C1", {})
    bwt_ok = all(v >= -0.05 for v in r["bwt"].values())
    grad_ok = sum(v > 1e-8 for v in r["shared_grad_mean"].values()) >= 2
    if c1.get("status") == "complete":
        forget_ok: bool | None = r["forgetting"] <= 0.5 * c1["forgetting"]
        ind["c1_forgetting"] = c1["forgetting"]
    else:
        forget_ok = None
    ind["g03_parts"] = {"bwt": bwt_ok, "shared_grad": grad_ok, "half_c1": forget_ok}
    ind["g03"] = bool(bwt_ok and grad_ok and forget_ok) if forget_ok is not None else None
    bases = [arms[a]["mean3"] for a in ("C2", "C3") if arms.get(a, {}).get("status") == "complete"]
    ind["g04_base"] = max(bases) if bases else None
    ind["g04"] = (r["mean3"] >= max(bases) - 0.02) if bases else None
    if arms.get("C2", {}).get("status") == "complete":
        ind["r2a_minus_c5"] = r["mean3"] - arms["C2"]["mean3"]
    drops = {t: fin[t] - r["shared_off"][t] for t in TASKS}
    ind["g05_drop"] = drops
    ind["g05"] = sum(d >= 0.10 for d in drops.values()) >= 2
    return ind


def aggregate() -> int:
    rows = []
    for seed in EXPLORATION_SEEDS:
        path = REPORTS / f"n31_1_03_seed{seed}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
    complete = all(
        r is not None
        and r.get("status") == "complete"
        and all(r["arms"].get(a, {}).get("status") == "complete" for a in ("R2A", "C2", "C1"))
        for r in rows
    )
    summary: dict[str, Any] = {"schema": SCHEMA + "-summary", "card": CARD}
    if not complete:
        summary["exit"] = "engineering_failure"
    else:
        inds = [r["indicators"] for r in rows if r is not None]
        margins = [i["g01_margin"] for i in inds]

        def count(key: str) -> int:
            return sum(bool(i.get(key)) for i in inds)

        r2a = [r["arms"]["R2A"]["mean3"] for r in rows if r is not None]
        base_means = []
        for arm in ("C2", "C3"):
            vals = [r["arms"][arm]["mean3"] for r in rows if r is not None and arm in r["arms"]]
            if len(vals) == len(rows):
                base_means.append(sum(vals) / len(vals))
        g04_mean = sum(r2a) / len(r2a) >= max(base_means) - 0.02
        positive = (
            min(margins) >= 0.02
            and count("g02") >= 2
            and count("g03") >= 2
            and count("g05") >= 2
            and g04_mean
        )
        if positive:
            summary["exit"] = "signal_positive"
        elif min(margins) >= -0.05 or count("g01") >= 1:
            summary["exit"] = "inconclusive"
        else:
            summary["exit"] = "signal_negative"
        summary.update(
            {
                "proxy_weakest_seed_min_task": min(i["min_task"] for i in inds),
                "weakest_g01_margin": min(margins),
                "g01_seeds": count("g01"),
                "g02_seeds": count("g02"),
                "g03_seeds": count("g03"),
                "g05_seeds": count("g05"),
                "g04_on_mean": g04_mean,
                "r2a_mean3": sum(r2a) / len(r2a),
                "baseline_mean3": base_means,
            }
        )
    summary["cpu_total"] = round(sum(r["cost"]["cpu"] for r in rows if r is not None), 2)
    out = REPORTS / "n31_1_03_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------------------ main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--cap-cpu", type=float, default=SEED_CPU_CAP)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    if args.top_k is not None:
        global TOP_K
        TOP_K = int(args.top_k)
    if args.aggregate:
        return aggregate()
    if args.seed not in EXPLORATION_SEEDS:
        raise SystemExit(f"{CARD} is an exploration card; seed must be in {EXPLORATION_SEEDS}")
    design = preflight(CARD, [args.seed], record=False)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not set(arm_names) <= set(ARMS) or not {"R2A", "C2", "C1"} <= set(arm_names):
        raise SystemExit(f"--arms must include R2A,C2,C1 and only {ARMS}; got {arm_names}")
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + args.cap_cpu
    steps = SMOKE_STEPS if args.smoke else STEPS
    lambdas = (LAMBDAS[1],) if args.smoke else LAMBDAS
    sets = {"fit": eval_sets(args.seed, "fit"), "tune": eval_sets(args.seed, "tune")}
    arms_out: dict[str, Any] = {}
    status = "complete"
    for arm in arm_names:
        try:
            if arm == "C3":
                arms_out[arm] = run_c3(args.seed, steps, sets, deadline, lambdas)
            else:
                arms_out[arm] = run_arm(arm, args.seed, steps, sets, deadline)
        except CapExceeded as exc:
            arms_out[arm] = {"status": "incomplete", "detail": str(exc)}
            status = "incomplete"
            break
        except NonFiniteLoss as exc:
            arms_out[arm] = {"status": "nan", "detail": str(exc)}
            status = "engineering_failure"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "card": CARD,
        "design": design.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "smoke": args.smoke,
        "status": status,
        "protocol": {
            "steps_per_task": steps,
            "lr": LR,
            "batch": BATCH,
            "replay": {"buffer": REPLAY_N, "batch_per_old_task": REPLAY_BATCH},
            "router": {
                "experts": N_EXPERTS,
                "units": EXPERT_UNITS,
                "top_k": TOP_K,
                "noise": ROUTER_NOISE,
                "load_balance": LB_COEF,
                "shared": N_SHARED,
            },
            "ewc_lambdas": list(lambdas),
            "eval_n": EVAL_N,
            "gates": GATES,
        },
        "arms": arms_out,
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": args.cap_cpu},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    if args.smoke:
        per_arm = {}
        for arm, row in arms_out.items():
            if row.get("status") == "complete":
                per_arm[arm] = (
                    row["cpu"] * (STEPS / SMOKE_STEPS) * (len(LAMBDAS) if arm == "C3" else 1)
                )
        all_arms = sum(per_arm.values()) * len(EXPLORATION_SEEDS)
        no_c3 = sum(v for a, v in per_arm.items() if a != "C3") * len(EXPLORATION_SEEDS)
        if status == "complete" and all_arms <= CARD_CPU_CAP:
            planned = ",".join(ARMS)
        elif status == "complete" and no_c3 <= CARD_CPU_CAP:
            planned = "R2A,C2,C1"
        else:
            planned = ""
        report["projection"] = {
            "per_arm_per_seed": per_arm,
            "all_arms": round(all_arms, 1),
            "without_c3": round(no_c3, 1),
        }
        report["planned_arms"] = planned
        out = REPORTS / f"n31_1_03_smoke_seed{args.seed}.json"
    else:
        report["indicators"] = seed_indicators(arms_out)
        out = REPORTS / f"n31_1_03_seed{args.seed}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{CARD} seed {args.seed} status={status} cpu={report['cost']['cpu']} -> {out.name}")
    if args.smoke:
        print(f"projection={report['projection']} planned_arms={report['planned_arms']!r}")
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
