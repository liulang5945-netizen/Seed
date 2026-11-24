"""Harder-M3 diagnostic: does SelectiveNeuralUnits beat a read_gate-matched GRU off the ceiling?

Exploration only, 0 check / 0 final. Follows D-20261123-329: GRU + the same final-tick
read_gate reached 1.000 on the original M3 (T=12, 3 slots, 4 values), tying
SelectivePersistentGraph, so that task cannot separate the two. This script raises
difficulty along three levels declared before any run and compares ``selective`` with
``gru_gate`` at equal analytic FLOPs; plain ``gru`` is kept as a floor reference.

Levels (pre-declared, not tuned after reading):
  H1  T=24, slots=3, values=4, overwrites=0   longer delay, more distractor ticks
  H2  T=24, slots=6, values=4, overwrites=0   more concurrent items
  H3  T=40, slots=6, values=8, overwrites=2   long horizon + latest-write-wins

Decision rules (per level; written before running and computed by this script):
  chance = 1 / values
  ceiling      selective and gru_gate both have mean fit >= 0.98
  floor        both have mean fit <= chance + 0.05
  otherwise the level is informative; with paired per-seed d = selective - gru_gate:
    selective_better   d >= +0.05 on every seed
    gru_gate_better    d <= -0.05 on every seed
    no_difference      |mean d| <= 0.02 and |d| <= 0.05 on every seed
    inconclusive       anything else
Overall verdict:
  presentation_failure                no informative level
  unit_dynamics_contribution_bounded  >= 1 selective_better and no gru_gate_better
  gru_gate_better_somewhere           >= 1 gru_gate_better and no selective_better
  no_independent_contribution         every informative level is no_difference
  mixed                               anything else
Secondary (reported, never judged): mean fit over checkpoints (learning-speed AUC).

Same batches for every arm within a seed and level (paired design); fit/tune splits are
disjoint from training seeds and from each other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import SEEDS, build_graph  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
    _clip_and_step,
    _eval_selective_preds,
    _trainable_params,
)
from scripts.training.verify_taiji_selective_bridge_20261121 import (  # noqa: E402
    BATCH,
    FIT_OFFSET,
    _GRUSequenceBaseline,
    gru_flops_per_seq,
    selective_flops_per_seq,
)
from taiji.u2_selective_units import SelectivePersistentGraph, SelectiveUnitSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_m3_hard_20261123.json"
INPUT = 48
READOUT_HIDDEN = 64
DEFAULT_STEPS = 1500
DEFAULT_CAP_CPU = 1800.0
LR_DEFAULT = 7e-3
LR_GRID = (3e-3, 7e-3, 1.5e-2)
FIT = 512
TUNE = 256
ARM_NAMES = ("selective", "gru_gate", "gru")
ARM_SEED_OFFSET = {"selective": 11, "gru_gate": 14, "gru": 12}

VALUE_BASE = 0  # value one-hot 0..7
NOISE_LO, NOISE_HI = 8, 16  # Gaussian distractor noise
KEY_BASE = 24  # key one-hot 24..31
WRITE_CH = 32
QUERY_CH = 33

CEILING = 0.98
FLOOR_MARGIN = 0.05
BETTER = 0.05
EQUIV_MEAN = 0.02
EQUIV_SEED = 0.05


@dataclass(frozen=True)
class Level:
    name: str
    horizon: int
    slots: int
    values: int
    overwrites: int

    @property
    def chance(self) -> float:
        return 1.0 / self.values


LEVELS = (
    Level("H1", 24, 3, 4, 0),
    Level("H2", 24, 6, 4, 0),
    Level("H3", 40, 6, 8, 2),
)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def make_m3h(count: int, seed: int, level: Level) -> dict[str, torch.Tensor]:
    """Generalised M3: ``slots`` keyed ``values``-class items written at random ticks.

    Layout matches ``_make_m3_data``: every tick carries a distractor value and key
    without the WRITE flag; write ticks carry value + key + WRITE; the final tick carries
    the query key + QUERY. With ``overwrites > 0`` some slots are written again later and
    the label is the value of the latest write to the queried slot.
    """
    steps, n_slots, n_values = level.horizon, level.slots, level.values
    n_writes = n_slots + level.overwrites
    if not (1 <= n_slots <= 8 and 2 <= n_values <= 8 and n_writes <= steps - 1):
        raise ValueError(f"invalid level {level}")
    gen = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, steps, INPUT, dtype=torch.float32)
    d_val = torch.randint(0, n_values, (count, steps), generator=gen)
    d_key = torch.randint(0, n_slots, (count, steps), generator=gen)
    ticks.scatter_(2, (VALUE_BASE + d_val).unsqueeze(-1), 1.0)
    ticks.scatter_(2, (KEY_BASE + d_key).unsqueeze(-1), 1.0)
    noise = torch.randn(count, steps, NOISE_HI - NOISE_LO, generator=gen) * 0.2
    ticks[:, :, NOISE_LO:NOISE_HI] = noise

    query = torch.randint(0, n_slots, (count,), generator=gen)
    labels = torch.zeros(count, dtype=torch.int64)
    for b in range(count):
        times = sorted(torch.randperm(steps - 1, generator=gen)[:n_writes].tolist())
        slots = torch.randperm(n_slots, generator=gen).tolist()
        slots += torch.randint(0, n_slots, (level.overwrites,), generator=gen).tolist()
        order = torch.randperm(n_writes, generator=gen).tolist()
        slots = [slots[i] for i in order]
        latest: dict[int, int] = {}
        for t, slot in zip(times, slots, strict=True):
            value = int(torch.randint(0, n_values, (1,), generator=gen))
            ticks[b, t, VALUE_BASE : VALUE_BASE + 8] = 0.0
            ticks[b, t, VALUE_BASE + value] = 1.0
            ticks[b, t, KEY_BASE : KEY_BASE + 8] = 0.0
            ticks[b, t, KEY_BASE + slot] = 1.0
            ticks[b, t, WRITE_CH] = 1.0
            latest[slot] = value
        labels[b] = latest[int(query[b])]

    last = steps - 1
    ticks[:, last, VALUE_BASE : VALUE_BASE + 8] = 0.0
    ticks[:, last, KEY_BASE : KEY_BASE + 8] = 0.0
    ticks[torch.arange(count), last, KEY_BASE + query] = 1.0
    ticks[:, last, QUERY_CH] = 1.0
    return {"ticks": ticks, "labels": labels}


# ---------------------------------------------------------------------------
# FLOPs matching (same convention as Bridge 1: 2 * MACs + elementwise)
# ---------------------------------------------------------------------------


def gru_gate_flops(steps: int, hidden: int, out: int) -> int:
    base = gru_flops_per_seq(steps, hidden, readout_hidden=READOUT_HIDDEN, output_dim=out)
    return int(base + 2 * INPUT * hidden + 5 * hidden)


def level_dims(level: Level) -> dict[str, Any]:
    steps, out = level.horizon, level.values
    target = selective_flops_per_seq(
        steps, groups=8, n_units=32, readout_hidden=READOUT_HIDDEN, output_dim=out
    )

    def gru_f(h: int) -> int:
        return gru_flops_per_seq(steps, h, readout_hidden=READOUT_HIDDEN, output_dim=out)

    def gate_f(h: int) -> int:
        return gru_gate_flops(steps, h, out)

    gru_h = min(range(4, 257), key=lambda h: abs(gru_f(h) - target))
    gate_h = min(range(4, 257), key=lambda h: abs(gate_f(h) - target))
    return {
        "target_flops": target,
        "hidden": {"gru": gru_h, "gru_gate": gate_h},
        "flops": {"selective": target, "gru": gru_f(gru_h), "gru_gate": gate_f(gate_h)},
    }


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


class _GatedGRU(nn.Module):
    """GRU whose final hidden state passes the same multiplicative read_gate as Selective."""

    def __init__(self, hidden: int, out: int) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUT, hidden, batch_first=True)
        self.read_gate = nn.Linear(INPUT, hidden)
        nn.init.constant_(self.read_gate.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, READOUT_HIDDEN), nn.Tanh(), nn.Linear(READOUT_HIDDEN, out)
        )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(ticks)
        gate = torch.sigmoid(self.read_gate(ticks[:, -1, :]))
        return cast(torch.Tensor, self.head(seq[:, -1, :] * gate))


@dataclass
class ArmRunner:
    forward: Callable[[torch.Tensor], torch.Tensor]
    predict: Callable[[torch.Tensor], torch.Tensor]
    update: Callable[[torch.Tensor], None]
    params: list[nn.Parameter]


def selective_runner(seed: int, out: int, lr: float) -> ArmRunner:
    base = build_graph(seed)  # build_graph reseeds torch internally; build it first
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(seed + ARM_SEED_OFFSET["selective"])
    spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=4,
        output_dim=out,
        readout_hidden=READOUT_HIDDEN,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + ARM_SEED_OFFSET["selective"],
    )
    graph = SelectivePersistentGraph(base, spec)
    graph.freeze_base_()
    params = _trainable_params(graph)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    n_units = graph.selective_spec.n_units

    def forward(ticks: torch.Tensor) -> torch.Tensor:
        gates = torch.ones(ticks.shape[0], ticks.shape[1], n_units)
        res = graph(_channels_seq(ticks), gates, graph.initial_state(batch_size=ticks.shape[0]))
        return cast(torch.Tensor, res["readout"])

    def predict(ticks: torch.Tensor) -> torch.Tensor:
        return _eval_selective_preds(graph, ticks)

    def update(loss: torch.Tensor) -> None:
        _clip_and_step(opt, graph, loss)

    return ArmRunner(forward, predict, update, params)


def baseline_runner(name: str, seed: int, hidden: int, out: int, lr: float) -> ArmRunner:
    torch.manual_seed(seed + ARM_SEED_OFFSET[name])
    model: nn.Module
    if name == "gru":
        model = _GRUSequenceBaseline(hidden, output_dim=out, readout_hidden=READOUT_HIDDEN)
    elif name == "gru_gate":
        model = _GatedGRU(hidden, out)
    else:
        raise ValueError(f"unknown arm {name!r}")
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    def forward(ticks: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, model(ticks))

    def predict(ticks: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, model(ticks).argmax(dim=-1))

    def update(loss: torch.Tensor) -> None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    return ArmRunner(forward, predict, update, params)


def _accuracy(runner: ArmRunner, data: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        return float((runner.predict(data["ticks"]) == data["labels"]).float().mean())


def run_arm(
    name: str,
    seed: int,
    level: Level,
    level_idx: int,
    dims: dict[str, Any],
    *,
    steps: int,
    lr: float,
    checkpoints: set[int],
    fit: dict[str, torch.Tensor],
    tune: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    t0 = time.process_time()
    if name == "selective":
        runner = selective_runner(seed, level.values, lr)
    else:
        runner = baseline_runner(name, seed, dims["hidden"][name], level.values, lr)
    loss_fn = nn.CrossEntropyLoss()
    curve: dict[str, float] = {}
    for step in range(steps):
        batch = make_m3h(BATCH, seed * 10000 + 1_000_000 * (level_idx + 1) + step, level)
        loss = loss_fn(runner.forward(batch["ticks"]), batch["labels"])
        runner.update(loss)
        if step + 1 in checkpoints:
            curve[str(step + 1)] = _accuracy(runner, fit)
    return {
        "lr": lr,
        "fit_acc": _accuracy(runner, fit),
        "tune_acc": None if tune is None else _accuracy(runner, tune),
        "fit_curve": curve,
        "auc": sum(curve.values()) / len(curve) if curve else float("nan"),
        "trainable_params": int(sum(p.numel() for p in runner.params if p.requires_grad)),
        "cpu": round(time.process_time() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Verdicts (pre-declared rules, see module docstring)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def level_verdict(level: Level, sel: dict[int, float], gate: dict[int, float]) -> dict[str, Any]:
    seeds = sorted(set(sel) & set(gate))
    if not seeds:
        return {"verdict": "incomplete"}
    sel_mean = _mean([sel[s] for s in seeds])
    gate_mean = _mean([gate[s] for s in seeds])
    diffs = [sel[s] - gate[s] for s in seeds]
    mean_d = _mean(diffs)
    if sel_mean >= CEILING and gate_mean >= CEILING:
        verdict = "ceiling"
    elif sel_mean <= level.chance + FLOOR_MARGIN and gate_mean <= level.chance + FLOOR_MARGIN:
        verdict = "floor"
    elif all(d >= BETTER for d in diffs):
        verdict = "selective_better"
    elif all(d <= -BETTER for d in diffs):
        verdict = "gru_gate_better"
    elif abs(mean_d) <= EQUIV_MEAN and all(abs(d) <= EQUIV_SEED for d in diffs):
        verdict = "no_difference"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "seeds": seeds,
        "selective_mean": sel_mean,
        "gru_gate_mean": gate_mean,
        "paired_diff_per_seed": diffs,
        "paired_diff_mean": mean_d,
        "chance": level.chance,
    }


def overall_verdict(per_level: dict[str, dict[str, Any]]) -> str:
    informative = [
        v["verdict"]
        for v in per_level.values()
        if v["verdict"] not in {"ceiling", "floor", "incomplete"}
    ]
    if not informative:
        return "presentation_failure"
    if "selective_better" in informative and "gru_gate_better" not in informative:
        return "unit_dynamics_contribution_bounded"
    if "gru_gate_better" in informative and "selective_better" not in informative:
        return "gru_gate_better_somewhere"
    if all(v == "no_difference" for v in informative):
        return "no_independent_contribution"
    return "mixed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--levels", default=",".join(lv.name for lv in LEVELS))
    parser.add_argument("--lr-sweep", action="store_true")
    parser.add_argument("--lrs", default="", help="comma-separated lr grid (overrides LR_GRID)")
    parser.add_argument("--fit", type=int, default=FIT, help="fit split size")
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--cap-cpu", type=float, default=DEFAULT_CAP_CPU)
    parser.add_argument("--smoke", action="store_true", help="30 steps, first seed, no report")
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()
    rules_sha256 = hashlib.sha256((__doc__ or "").encode("utf-8")).hexdigest()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wanted = {s.strip() for s in args.levels.split(",") if s.strip()}
    levels = [lv for lv in LEVELS if lv.name in wanted]
    if not levels:
        raise SystemExit(f"no known levels in {sorted(wanted)}")
    steps = args.steps
    if args.smoke:
        steps, seeds = 30, seeds[:1]
    checkpoints = {max(1, steps // 4), max(1, steps // 2), steps}
    grid = tuple(float(x) for x in args.lrs.split(",") if x.strip()) if args.lrs else LR_GRID
    lrs = grid if args.lr_sweep else (LR_DEFAULT,)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not set(arm_names) <= set(ARM_NAMES) or not {"selective", "gru_gate"} <= set(arm_names):
        raise SystemExit(f"--arms must include selective and gru_gate, got {arm_names}")
    fit_size = args.fit
    dims = {lv.name: level_dims(lv) for lv in LEVELS}

    status = "complete"
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row: dict[str, Any] = {"seed": seed, "levels": {}}
        for level in levels:
            idx = LEVELS.index(level)
            fit = make_m3h(fit_size, seed + 9000 + FIT_OFFSET + 100 * (idx + 1), level)
            tune = (
                make_m3h(TUNE, seed + 9500 + FIT_OFFSET + 100 * (idx + 1), level)
                if args.lr_sweep
                else None
            )
            arms: dict[str, Any] = {}
            for name in arm_names:
                runs = []
                for lr in lrs:
                    if time.process_time() - started > args.cap_cpu:
                        status = "budget_stop"
                        break
                    runs.append(
                        run_arm(
                            name,
                            seed,
                            level,
                            idx,
                            dims[level.name],
                            steps=steps,
                            lr=lr,
                            checkpoints=checkpoints,
                            fit=fit,
                            tune=tune,
                        )
                    )
                if not runs:
                    break
                chosen = max(runs, key=lambda r: r["tune_acc"]) if args.lr_sweep else runs[0]
                arms[name] = {"chosen": chosen, "all_lrs": runs if args.lr_sweep else None}
                print(
                    f"seed={seed} {level.name} arm={name:<9} lr={chosen['lr']:.4g} "
                    f"fit={chosen['fit_acc']:.4f} auc={chosen['auc']:.4f} "
                    f"cpu={chosen['cpu']:.1f}"
                )
            row["levels"][level.name] = arms
            if status == "budget_stop":
                break
        rows.append(row)
        if status == "budget_stop":
            break

    summary: dict[str, Any] = {}
    per_level_verdict: dict[str, dict[str, Any]] = {}
    for level in levels:
        lv_sum: dict[str, Any] = {}
        per_seed: dict[str, dict[int, float]] = {}
        for name in arm_names:
            accs = {
                r["seed"]: r["levels"][level.name][name]["chosen"]["fit_acc"]
                for r in rows
                if name in r["levels"].get(level.name, {})
            }
            aucs = [
                r["levels"][level.name][name]["chosen"]["auc"]
                for r in rows
                if name in r["levels"].get(level.name, {})
            ]
            per_seed[name] = accs
            if accs:
                lv_sum[name] = {
                    "fit_acc_mean": _mean(list(accs.values())),
                    "fit_acc_min": min(accs.values()),
                    "fit_acc_per_seed": accs,
                    "auc_mean": _mean(aucs),
                    "hidden": dims[level.name]["hidden"].get(name),
                    "flops_ratio": dims[level.name]["flops"][name]
                    / dims[level.name]["target_flops"],
                }
        summary[level.name] = lv_sum
        per_level_verdict[level.name] = level_verdict(
            level, per_seed.get("selective", {}), per_seed.get("gru_gate", {})
        )
    verdict = overall_verdict(per_level_verdict) if status == "complete" else "budget_stop"

    payload: dict[str, Any] = {
        "schema": "taiji-u2-m3-hard-v1",
        "card": "Exploration diagnostic: Selective vs read_gate-matched GRU on harder M3",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "levels": [asdict(lv) for lv in levels],
            "seeds": seeds,
            "steps": steps,
            "batch": BATCH,
            "fit_size": fit_size,
            "arms": arm_names,
            "lrs": list(lrs),
            "lr_selection": "max tune_acc" if args.lr_sweep else "fixed (Bridge-1 value)",
            "optimizer": "Adam, weight_decay=1e-4, clip_grad_norm=1.0",
            "checkpoints": sorted(checkpoints),
            "dims": {lv.name: dims[lv.name] for lv in levels},
            "accuracy_resolution": 1.0 / fit_size,
            "binomial_se_at_p0.5": math.sqrt(0.25 / fit_size),
            "decision_rules_sha256": rules_sha256,
        },
        "rows": rows,
        "summary": summary,
        "level_verdicts": per_level_verdict,
        "verdict": verdict,
        "status": status,
        "claim_boundary": (
            "exploration diagnostic on a synthetic task; not a capability claim; fit split only"
        ),
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": args.cap_cpu},
    }

    if args.smoke:
        print(json.dumps({"smoke": True, "levels": per_level_verdict}, indent=2))
        return 0
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": status,
                "verdict": verdict,
                "level_verdicts": {k: v["verdict"] for k, v in per_level_verdict.items()},
                "means": {
                    k: {a: round(s["fit_acc_mean"], 4) for a, s in v.items()}
                    for k, v in summary.items()
                },
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
