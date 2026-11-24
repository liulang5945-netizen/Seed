"""N32.0.01 / N32.0.90: RT columns on GRU+read_gate -- the release-02 substitute (D-495).

N32.0.01 (exploration, seeds 20261120-22): units per column in {24, 48}; each column is
  trained 2,400 steps and evaluated at 1,200 and 2,400.  Columns are disjoint and the
  optimiser is per column, so the 1,200-step snapshot is exactly the 1,200-step
  configuration.  Selection rule and exits: plans/reference/DESIGN-N32-0-01_20261124.md.
N32.0.90 (confirmation, seeds 20261201-03): only the configuration chosen by the N32.0.01
  summary, plus the full-standard baseline C2 (N31.1.02 protocol, same steps) for
  release-05.  Protocol: plans/reference/DESIGN-N32-0-90_20261124.md.

Usage:
  python scripts/training/n32_0_rt_admission.py N32.0.01 <seed> [--smoke]
  python scripts/training/n32_0_rt_admission.py N32.0.90 <seed>
  python scripts/training/n32_0_rt_admission.py <card> --aggregate
"""

from __future__ import annotations

import argparse
import json
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

from scripts.training.n31_1_02_routed_gru import (  # noqa: E402
    BATCH,
    GATES,
    LR,
    METRICS,
    TASKS,
    CapExceeded,
    NonFiniteLoss,
    eval_sets,
    make_batch,
    run_arm,
    train_seed,
)
from scripts.training.n31_guard import preflight  # noqa: E402
from taiji.gru_rt import ARCH_VERSION, SCOPE, RTSubject  # noqa: E402

SCHEMA = "taiji-n32-0-rt-v1"
REPORTS = ROOT / "reports"
OUT_DIM = {"cn": 8, "p24": 5, "p27": 4}
EXPLORE_UNITS = (24, 48)
CHECKPOINTS = (1200, 2400)
ORDER = ((24, 1200), (24, 2400), (48, 1200), (48, 2400))
SMOKE_STEPS = 40
EXPLORATION_SEEDS = (20261120, 20261121, 20261122)
CONFIRMATION_SEEDS = (20261201, 20261202, 20261203)
CARD_CPU_CAP = {"N32.0.01": 600.0, "N32.0.90": 600.0}
SEED_CPU_CAP = 200.0
CAPSULE_TOL = 1e-6


def rt_loss(model: RTSubject, task: str, data: dict[str, Any]) -> torch.Tensor:
    lg = model(data["ticks"], task)
    if task == "p24":
        return F.cross_entropy(lg[:, :4], data["labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], data["conflict"]
        )
    return F.cross_entropy(lg, data["labels"])


def eval_task(model: RTSubject, sets: dict[str, Any], task: str) -> dict[str, float]:
    model.eval()
    d = sets[task]
    with torch.no_grad():
        lg = model(d["ticks"], task)
    cls_logits = lg[:, :4] if task == "p24" else lg
    res = {task: float((cls_logits.argmax(-1) == d["labels"]).float().mean())}
    if task == "p24":
        res["conflict"] = float(((lg[:, 4] > 0).float() == d["conflict"]).float().mean())
    return res


def eval_all(model: RTSubject, sets: dict[str, Any]) -> dict[str, float]:
    res: dict[str, float] = {}
    for task in TASKS:
        res.update(eval_task(model, sets, task))
    return res


def capsule_diff(model: RTSubject, sets: dict[str, Any]) -> float:
    restored = RTSubject.from_capsule(model.capsule())
    model.eval()
    restored.eval()
    worst = 0.0
    with torch.no_grad():
        for task in TASKS:
            a = model(sets[task]["ticks"], task)
            b = restored(sets[task]["ticks"], task)
            worst = max(worst, float((a - b).abs().max()))
    return worst


def summarize(final: dict[str, float]) -> dict[str, Any]:
    return {
        "final": final,
        "g01_margin": min(final[m] - GATES[m] for m in METRICS),
        "mean3": sum(final[t] for t in TASKS) / len(TASKS),
    }


def train_rt(
    seed: int, units: int, checkpoints: tuple[int, ...], sets: dict[str, Any], deadline: float
) -> dict[str, Any]:
    init_seed = seed + 9001
    model = RTSubject(OUT_DIM, units, init_seed)
    snaps: dict[int, dict[str, dict[str, torch.Tensor]]] = {s: {} for s in checkpoints}
    immediate: dict[int, dict[str, float]] = {s: {} for s in checkpoints}
    t0 = time.process_time()
    for task in TASKS:
        params = model.task_parameters(task)
        opt = torch.optim.Adam(params, lr=LR[task])
        model.train()
        for k in range(max(checkpoints)):
            if time.process_time() > deadline:
                raise CapExceeded(f"units {units} task {task} step {k}")
            loss = rt_loss(model, task, make_batch(task, BATCH, train_seed(task, seed, k)))
            if not torch.isfinite(loss):
                raise NonFiniteLoss(f"units {units} task {task} step {k}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if k + 1 in checkpoints:
                col = model.column(task)
                snaps[k + 1][task] = {n: v.detach().clone() for n, v in col.state_dict().items()}
                immediate[k + 1].update(eval_task(model, sets["fit"], task))
                model.train()
    top = max(checkpoints)
    live = eval_all(model, sets["fit"])
    live_bwt = {m: live[m] - immediate[top][m] for m in METRICS}
    configs: dict[str, Any] = {}
    for steps in checkpoints:
        m = RTSubject(OUT_DIM, units, init_seed)
        for task in TASKS:
            m.column(task).load_state_dict(snaps[steps][task])
        row = summarize(eval_all(m, sets["fit"]))
        row.update({"units": units, "steps": steps, "capsule_diff": capsule_diff(m, sets["fit"])})
        row["matches_immediate"] = row["final"] == immediate[steps]
        configs[f"u{units}_s{steps}"] = row
    return {
        "configs": configs,
        "live_bwt": live_bwt,
        "live_bwt_exact_zero": all(v == 0.0 for v in live_bwt.values()),
        "n_params": {t: model.n_params(t) for t in TASKS} | {"total": model.n_params()},
        "cpu": round(time.process_time() - t0, 2),
    }


def load_chosen() -> dict[str, int]:
    path = REPORTS / "n32_0_01_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("exit") != "signal_positive" or not summary.get("chosen"):
        raise SystemExit(f"N32.0.90 needs a signal_positive N32.0.01 summary; got {summary}")
    return {"units": int(summary["chosen"]["units"]), "steps": int(summary["chosen"]["steps"])}


def seed_ok(report: dict[str, Any], key: str, threshold: float) -> bool:
    row = report["rt"]["configs"].get(key)
    return bool(
        row
        and row["g01_margin"] >= threshold
        and row["capsule_diff"] <= CAPSULE_TOL
        and row["matches_immediate"]
        and report["rt"]["live_bwt_exact_zero"]
    )


def aggregate(card: str) -> int:
    seeds = EXPLORATION_SEEDS if card == "N32.0.01" else CONFIRMATION_SEEDS
    stem = card.lower().replace(".", "_")
    rows = []
    for seed in seeds:
        path = REPORTS / f"{stem}_seed{seed}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)
    complete = [r for r in rows if r is not None and r.get("status") == "complete"]
    summary: dict[str, Any] = {"schema": SCHEMA + "-summary", "card": card, "scope": SCOPE}
    summary["cpu_total"] = round(sum(r["cost"]["cpu"] for r in rows if r is not None), 2)
    if len(complete) != len(seeds):
        summary["exit"] = "engineering_failure"
    elif card == "N32.0.01":
        table = {}
        chosen = None
        for units, steps in ORDER:
            key = f"u{units}_s{steps}"
            margins = [r["rt"]["configs"][key]["g01_margin"] for r in complete]
            passed = all(seed_ok(r, key, 0.02) for r in complete)
            table[key] = {"weakest_margin": min(margins), "all_seeds_plus_0_02": passed}
            if passed and chosen is None:
                chosen = {"units": units, "steps": steps}
        best = max(v["weakest_margin"] for v in table.values())
        summary["configs"] = table
        summary["chosen"] = chosen
        if chosen is not None:
            summary["exit"] = "signal_positive"
        elif best >= -0.05:
            summary["exit"] = "inconclusive"
        else:
            summary["exit"] = "signal_negative"
    else:
        key = complete[0]["config_key"]
        oks = [seed_ok(r, key, 0.0) for r in complete]
        rt3 = [r["rt"]["configs"][key]["mean3"] for r in complete]
        c2 = [r["c2"]["mean3"] for r in complete if r.get("c2", {}).get("status") == "complete"]
        summary.update(
            {
                "config": key,
                "seeds_pass": sum(oks),
                "weakest_margin": min(r["rt"]["configs"][key]["g01_margin"] for r in complete),
                "rt_mean3": sum(rt3) / len(rt3),
                "c2_mean3": sum(c2) / len(c2) if c2 else None,
                "exit": "n32_0_90_pass" if all(oks) else "n32_0_90_fail",
            }
        )
    out = REPORTS / f"{stem}_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("card", choices=sorted(CARD_CPU_CAP))
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate(args.card)
    confirm = args.card == "N32.0.90"
    allowed = CONFIRMATION_SEEDS if confirm else EXPLORATION_SEEDS
    if args.seed not in allowed:
        raise SystemExit(f"{args.card}: seed must be in {allowed}")
    if confirm and args.smoke:
        raise SystemExit("no smoke on confirmation seeds")
    chosen = load_chosen() if confirm else None
    design = preflight(args.card, [args.seed], record=confirm)
    torch.set_num_threads(1)
    started = time.process_time()
    deadline = started + SEED_CPU_CAP
    sets = {"fit": eval_sets(args.seed, "fit"), "tune": eval_sets(args.seed, "tune")}
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "card": args.card,
        "design": design.name,
        "arch_version": ARCH_VERSION,
        "scope": SCOPE,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "smoke": args.smoke,
        "gates": GATES,
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": not confirm},
    }
    status = "complete"
    try:
        if chosen is not None:
            report["config_key"] = f"u{chosen['units']}_s{chosen['steps']}"
            report["rt"] = train_rt(args.seed, chosen["units"], (chosen["steps"],), sets, deadline)
            report["c2"] = run_arm("C2", args.seed, chosen["steps"], sets, deadline)
        else:
            cps = (SMOKE_STEPS // 2, SMOKE_STEPS) if args.smoke else CHECKPOINTS
            runs = [train_rt(args.seed, u, cps, sets, deadline) for u in EXPLORE_UNITS]
            pairs = [(f"u{u}", r) for u, r in zip(EXPLORE_UNITS, runs, strict=True)]
            report["rt"] = {
                "configs": {k: v for r in runs for k, v in r["configs"].items()},
                "live_bwt": {key: r["live_bwt"] for key, r in pairs},
                "live_bwt_exact_zero": all(r["live_bwt_exact_zero"] for r in runs),
                "n_params": {key: r["n_params"] for key, r in pairs},
                "cpu": {key: r["cpu"] for key, r in pairs},
            }
    except CapExceeded as exc:
        status, report["detail"] = "incomplete", str(exc)
    except NonFiniteLoss as exc:
        status, report["detail"] = "engineering_failure", str(exc)
    report["status"] = status
    cpu = round(time.process_time() - started, 2)
    report["cost"] = {"cpu": cpu, "cap_cpu": SEED_CPU_CAP}
    stem = args.card.lower().replace(".", "_")
    if args.smoke:
        projected = cpu * (max(CHECKPOINTS) / SMOKE_STEPS) * len(EXPLORATION_SEEDS)
        report["projection"] = {"card_cpu": round(projected, 1), "cap": CARD_CPU_CAP[args.card]}
        report["projection_ok"] = status == "complete" and projected <= CARD_CPU_CAP[args.card]
        out = REPORTS / f"{stem}_smoke_seed{args.seed}.json"
    else:
        out = REPORTS / f"{stem}_seed{args.seed}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{args.card} seed {args.seed} status={status} cpu={cpu} -> {out.name}")
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
