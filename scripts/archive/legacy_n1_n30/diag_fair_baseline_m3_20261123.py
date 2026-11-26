"""Fair-baseline diagnostic for D-227 Bridge 1 on M3 (exploration only, 0 check / 0 final).

Question
--------
Bridge 1 reported SelectivePersistentGraph 1.000 vs equal-FLOPs GRU 0.530 on M3 (T=12).
Only the Selective arm had the final-tick multiplicative ``read_gate``; D-225 had already
shown that the Selective arm itself stalls near 0.58 without it. This script asks whether
the gap comes from the selective unit dynamics or from the read-out gate.

Arms (same M3 data stream, batch, steps, optimizer, weight decay, clip, fit split):
  selective          Bridge-1 Selective arm, reproduced exactly
  selective_no_gate  identical init, read_gate pinned to 1.0 (weight 0, bias +30, frozen)
  gru                Bridge-1 GRU arm, reproduced exactly
  gru_gate           GRU + the same final-tick read_gate (Linear(48 -> state), bias +1.0)
  lstm               LSTM, read-out on [h_T, c_T] (mirrors Selective's [z, a])
  lstm_gate          LSTM + the same read_gate on [h_T, c_T]
Every baseline hidden size is re-matched to the Selective analytic FLOPs.

Notes
-----
* Bridge 1 uses ``groups=8`` while the base graph has 6 groups, so
  ``SelectivePersistentGraph.encode_sequence_inputs`` discards the base encoding. A fresh
  frozen base graph is therefore numerically equivalent and is the default;
  ``--trained-base`` restores the exact Bridge-1 joint parent (slower).
* ``--lr-sweep`` tunes every arm (including Selective) over ``LR_GRID`` on a disjoint tune
  split and reports fit accuracy at the tuned lr.
* This is an exploration diagnostic, not a capability claim and not a pass/fail gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections.abc import Callable
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
    _make_m3_data,
    _trainable_params,
)
from scripts.training.verify_taiji_selective_bridge_20261121 import (  # noqa: E402
    BATCH,
    FIT,
    FIT_OFFSET,
    _GRUSequenceBaseline,
    gru_flops_per_seq,
    match_gru_hidden,
    selective_flops_per_seq,
)
from taiji.u2_selective_units import SelectivePersistentGraph, SelectiveUnitSpec  # noqa: E402

OUT = ROOT / "reports/taiji_u2_fair_baseline_m3_20261123.json"
BRIDGE_REPORT = ROOT / "reports/taiji_u2_selective_bridge_20261121.json"
HORIZON = 12
INPUT = 48
OUTPUT = 4
READOUT_HIDDEN = 64
DEFAULT_STEPS = 1000
DEFAULT_CAP_CPU = 900.0
LR_DEFAULT = 7e-3
LR_GRID = (3e-3, 7e-3, 1.5e-2)
TUNE = 256
FIT_SEED_OFFSET = 9000  # identical to Bridge 1 fit split
TUNE_SEED_OFFSET = 9500  # disjoint split, only used by --lr-sweep
ARM_NAMES = ("selective", "selective_no_gate", "gru", "gru_gate", "lstm", "lstm_gate")
ARM_SEED_OFFSET = {
    "selective": 11,
    "selective_no_gate": 11,  # paired with selective: identical init except the gate
    "gru": 12,
    "gru_gate": 14,
    "lstm": 15,
    "lstm_gate": 16,
}


# ---------------------------------------------------------------------------
# FLOPs accounting (same convention as Bridge 1: 2 * MACs + elementwise)
# ---------------------------------------------------------------------------


def gru_gate_flops(steps: int, hidden: int) -> int:
    return int(gru_flops_per_seq(steps, hidden) + 2 * INPUT * hidden + 5 * hidden)


def lstm_flops(steps: int, hidden: int, *, gate: bool) -> int:
    per_tick = 8 * (INPUT * hidden + hidden * hidden) + 15 * hidden
    state = 2 * hidden
    readout = 2 * state * READOUT_HIDDEN + 2 * READOUT_HIDDEN * OUTPUT
    extra = 2 * INPUT * state + 5 * state if gate else 0
    return int(steps * per_tick + readout + extra)


def match_hidden(flops_fn: Callable[[int], int], target: int) -> tuple[int, int]:
    best = min(range(4, 161), key=lambda h: abs(flops_fn(h) - target))
    return best, flops_fn(best)


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------


class _GatedGRU(nn.Module):
    """GRU whose final hidden state passes the same multiplicative read_gate as Selective."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gru = nn.GRU(INPUT, hidden, batch_first=True)
        self.read_gate = nn.Linear(INPUT, hidden)
        nn.init.constant_(self.read_gate.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, READOUT_HIDDEN), nn.Tanh(), nn.Linear(READOUT_HIDDEN, OUTPUT)
        )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(ticks)
        gate = torch.sigmoid(self.read_gate(ticks[:, -1, :]))
        return cast(torch.Tensor, self.head(out[:, -1, :] * gate))


class _LSTMBaseline(nn.Module):
    """LSTM read out on [h_T, c_T]; optional read_gate identical to Selective's."""

    def __init__(self, hidden: int, *, gate: bool) -> None:
        super().__init__()
        self.lstm = nn.LSTM(INPUT, hidden, batch_first=True)
        self.read_gate: nn.Linear | None = None
        if gate:
            self.read_gate = nn.Linear(INPUT, 2 * hidden)
            nn.init.constant_(self.read_gate.bias, 1.0)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, READOUT_HIDDEN), nn.Tanh(), nn.Linear(READOUT_HIDDEN, OUTPUT)
        )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        out, (_, c_n) = self.lstm(ticks)
        state = torch.cat([out[:, -1, :], c_n[-1]], dim=-1)
        if self.read_gate is not None:
            state = state * torch.sigmoid(self.read_gate(ticks[:, -1, :]))
        return cast(torch.Tensor, self.head(state))


def _build_selective(seed: int, base_graph: Any, *, read_gate: bool) -> SelectivePersistentGraph:
    torch.manual_seed(seed + ARM_SEED_OFFSET["selective"])
    spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=4,
        output_dim=OUTPUT,
        readout_hidden=READOUT_HIDDEN,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + ARM_SEED_OFFSET["selective"],
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    graph.freeze_base_()
    if not read_gate:
        with torch.no_grad():
            graph.read_gate.weight.zero_()
            graph.read_gate.bias.fill_(30.0)  # sigmoid(30) == 1.0 in float32
        for parameter in graph.read_gate.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    return graph


# ---------------------------------------------------------------------------
# One arm, one seed, one lr
# ---------------------------------------------------------------------------


def _accuracy(
    predict: Callable[[torch.Tensor], torch.Tensor], data: dict[str, torch.Tensor]
) -> float:
    with torch.no_grad():
        return float((predict(data["ticks"]) == data["labels"]).float().mean())


def run_arm(
    name: str,
    seed: int,
    base_graph: Any,
    hidden: dict[str, int],
    *,
    steps: int,
    lr: float,
    checkpoints: set[int],
    fit: dict[str, torch.Tensor],
    tune: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    t0 = time.process_time()
    loss_fn = nn.CrossEntropyLoss()

    if name in {"selective", "selective_no_gate"}:
        graph = _build_selective(seed, base_graph, read_gate=(name == "selective"))
        params = _trainable_params(graph)
        opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
        n_units = graph.selective_spec.n_units

        def _sel_forward(ticks: torch.Tensor) -> torch.Tensor:
            gates = torch.ones(ticks.shape[0], ticks.shape[1], n_units)
            out = graph(_channels_seq(ticks), gates, graph.initial_state(batch_size=ticks.shape[0]))
            return cast(torch.Tensor, out["readout"])

        def _sel_predict(ticks: torch.Tensor) -> torch.Tensor:
            return _eval_selective_preds(graph, ticks)

        def _sel_update(loss: torch.Tensor) -> None:
            _clip_and_step(opt, graph, loss)

        forward, predict, update = _sel_forward, _sel_predict, _sel_update

    else:
        torch.manual_seed(seed + ARM_SEED_OFFSET[name])
        model: nn.Module
        if name == "gru":
            model = _GRUSequenceBaseline(
                hidden[name], output_dim=OUTPUT, readout_hidden=READOUT_HIDDEN
            )
        elif name == "gru_gate":
            model = _GatedGRU(hidden[name])
        elif name == "lstm":
            model = _LSTMBaseline(hidden[name], gate=False)
        elif name == "lstm_gate":
            model = _LSTMBaseline(hidden[name], gate=True)
        else:
            raise ValueError(f"unknown arm {name!r}")
        params = list(model.parameters())
        base_opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

        def _base_forward(ticks: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, model(ticks))

        def _base_predict(ticks: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, model(ticks).argmax(dim=-1))

        def _base_update(loss: torch.Tensor) -> None:
            base_opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            base_opt.step()

        forward, predict, update = _base_forward, _base_predict, _base_update

    curve: dict[str, float] = {}
    for step in range(steps):
        batch = _make_m3_data(BATCH, seed * 10000 + step, horizon=HORIZON)
        loss = loss_fn(forward(batch["ticks"]), batch["labels"])
        update(loss)
        if step + 1 in checkpoints:
            curve[str(step + 1)] = _accuracy(predict, fit)

    fit_acc = _accuracy(predict, fit)
    return {
        "lr": lr,
        "fit_acc": fit_acc,
        "tune_acc": None if tune is None else _accuracy(predict, tune),
        "fit_curve": curve,
        "trainable_params": int(sum(p.numel() for p in params if p.requires_grad)),
        "cpu": round(time.process_time() - t0, 3),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _bridge_reference() -> dict[int, dict[str, float]]:
    """Per-seed Bridge-1 accuracies, used only to check exact reproduction."""
    if not BRIDGE_REPORT.exists():
        return {}
    data = json.loads(BRIDGE_REPORT.read_text(encoding="utf-8"))
    out: dict[int, dict[str, float]] = {}
    for row in data.get("rows", []):
        b1 = row.get("bridge1_equal_flops", {})
        out[int(row["seed"])] = {
            "selective": float(b1.get("selective", {}).get("acc", float("nan"))),
            "gru": float(b1.get("gru", {}).get("acc", float("nan"))),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--lr-sweep", action="store_true")
    parser.add_argument("--trained-base", action="store_true")
    parser.add_argument("--cap-cpu", type=float, default=DEFAULT_CAP_CPU)
    parser.add_argument(
        "--smoke", action="store_true", help="30 steps, first seed, no report written"
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    started = time.process_time()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - set(ARM_NAMES))
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    steps = args.steps
    if args.smoke:
        steps, seeds = 30, seeds[:1]
    checkpoints = {max(1, steps // 4), max(1, steps // 2), steps}
    lrs = LR_GRID if args.lr_sweep else (LR_DEFAULT,)

    target = selective_flops_per_seq(
        HORIZON, groups=8, n_units=32, readout_hidden=READOUT_HIDDEN, output_dim=OUTPUT
    )
    gru_h, gru_f = match_gru_hidden(target, steps=HORIZON)
    gg_h, gg_f = match_hidden(lambda h: gru_gate_flops(HORIZON, h), target)
    ls_h, ls_f = match_hidden(lambda h: lstm_flops(HORIZON, h, gate=False), target)
    lg_h, lg_f = match_hidden(lambda h: lstm_flops(HORIZON, h, gate=True), target)
    hidden = {"gru": gru_h, "gru_gate": gg_h, "lstm": ls_h, "lstm_gate": lg_h}
    flops = {
        "selective": target,
        "selective_no_gate": target,
        "gru": gru_f,
        "gru_gate": gg_f,
        "lstm": ls_f,
        "lstm_gate": lg_f,
    }

    status = "complete"
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        if args.trained_base:
            from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent

            base_graph, _ = train_joint_parent(seed)
        else:
            base_graph = build_graph(seed)
            for parameter in base_graph.parameters():
                parameter.requires_grad_(False)
        fit = _make_m3_data(FIT, seed + FIT_SEED_OFFSET + FIT_OFFSET, horizon=HORIZON)
        tune = (
            _make_m3_data(TUNE, seed + TUNE_SEED_OFFSET + FIT_OFFSET, horizon=HORIZON)
            if args.lr_sweep
            else None
        )
        row: dict[str, Any] = {"seed": seed, "arms": {}}
        for name in arms:
            runs = []
            for lr in lrs:
                if time.process_time() - started > args.cap_cpu:
                    status = "budget_stop"
                    break
                runs.append(
                    run_arm(
                        name,
                        seed,
                        base_graph,
                        hidden,
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
            row["arms"][name] = {"chosen": chosen, "all_lrs": runs if args.lr_sweep else None}
            print(f"seed={seed} arm={name:<17} lr={chosen['lr']:.4g} fit={chosen['fit_acc']:.4f}")
        rows.append(row)
        if status == "budget_stop":
            break

    # ---- summary -----------------------------------------------------------
    summary: dict[str, Any] = {}
    for name in arms:
        per_seed = {
            r["seed"]: r["arms"][name]["chosen"]["fit_acc"] for r in rows if name in r["arms"]
        }
        if not per_seed:
            continue
        curves: dict[str, list[float]] = {}
        for r in rows:
            if name in r["arms"]:
                for k, v in r["arms"][name]["chosen"]["fit_curve"].items():
                    curves.setdefault(k, []).append(v)
        any_run = next(r["arms"][name]["chosen"] for r in rows if name in r["arms"])
        summary[name] = {
            "fit_acc_mean": _mean(list(per_seed.values())),
            "fit_acc_min": min(per_seed.values()),
            "fit_acc_per_seed": per_seed,
            "fit_curve_mean": {
                k: _mean(v) for k, v in sorted(curves.items(), key=lambda kv: int(kv[0]))
            },
            "chosen_lrs": [r["arms"][name]["chosen"]["lr"] for r in rows if name in r["arms"]],
            "hidden": hidden.get(name),
            "flops": flops[name],
            "flops_ratio_to_selective": flops[name] / target,
            "trainable_params": any_run["trainable_params"],
        }

    def paired(a: str, b: str) -> dict[str, Any] | None:
        if a not in summary or b not in summary:
            return None
        diffs = [
            summary[a]["fit_acc_per_seed"][s] - summary[b]["fit_acc_per_seed"][s]
            for s in summary[a]["fit_acc_per_seed"]
            if s in summary[b]["fit_acc_per_seed"]
        ]
        return {"mean_diff": _mean(diffs), "min_diff": min(diffs), "per_seed": diffs}

    best_gated = max(
        (n for n in ("gru_gate", "lstm_gate") if n in summary),
        key=lambda n: summary[n]["fit_acc_mean"],
        default=None,
    )
    contrasts = {
        "read_gate_effect_selective": paired("selective", "selective_no_gate"),
        "read_gate_effect_gru": paired("gru_gate", "gru"),
        "read_gate_effect_lstm": paired("lstm_gate", "lstm"),
        "selective_minus_best_gated_baseline": (
            None
            if best_gated is None
            else {"baseline": best_gated, **(paired("selective", best_gated) or {})}
        ),
        "selective_no_gate_minus_gru": paired("selective_no_gate", "gru"),
    }

    # ---- exact reproduction check against the D-227 report ------------------
    reference = _bridge_reference()
    reproduction: dict[str, Any] = {"applicable": steps == DEFAULT_STEPS and not args.lr_sweep}
    if reproduction["applicable"] and reference:
        checks = {}
        for name in ("selective", "gru"):
            if name not in summary:
                continue
            for seed, acc in summary[name]["fit_acc_per_seed"].items():
                ref = reference.get(int(seed), {}).get(name)
                if ref is not None:
                    checks[f"{name}@{seed}"] = {
                        "new": acc,
                        "bridge": ref,
                        "abs_diff": abs(acc - ref),
                    }
        reproduction["checks"] = checks
        reproduction["exact"] = bool(checks) and all(c["abs_diff"] == 0.0 for c in checks.values())

    resolution = 1.0 / FIT
    payload: dict[str, Any] = {
        "schema": "taiji-u2-fair-baseline-m3-v1",
        "card": (
            "Exploration diagnostic: read_gate-matched GRU/LSTM controls"
            " for D-20261121-227 Bridge 1"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "task": "M3 multi-item delayed query (T=12), _make_m3_data",
            "seeds": seeds,
            "steps": steps,
            "batch": BATCH,
            "fit_size": FIT,
            "fit_seed": f"seed + {FIT_SEED_OFFSET} + {FIT_OFFSET}",
            "tune_seed": f"seed + {TUNE_SEED_OFFSET} + {FIT_OFFSET}" if args.lr_sweep else None,
            "lrs": list(lrs),
            "lr_selection": "max tune_acc" if args.lr_sweep else "fixed (Bridge-1 value)",
            "optimizer": "Adam, weight_decay=1e-4, clip_grad_norm=1.0",
            "trained_base": bool(args.trained_base),
            "base_encoding_used_by_selective": False,
            "checkpoints": sorted(checkpoints),
            "flops_target": target,
            "accuracy_resolution": resolution,
            "binomial_se_at_p0.5": math.sqrt(0.25 / FIT),
        },
        "rows": rows,
        "summary": summary,
        "contrasts": contrasts,
        "reproduction": reproduction,
        "status": status,
        "claim_boundary": (
            "exploration diagnostic on a synthetic task; not a capability claim; fit split only"
        ),
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2), "cap_cpu": args.cap_cpu},
    }

    if args.smoke:
        print(json.dumps({"smoke": True, "summary": summary}, ensure_ascii=False, indent=2))
        return 0
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "status": status,
                "summary": {k: v["fit_acc_mean"] for k, v in summary.items()},
                "contrasts": contrasts,
                "reproduction_exact": reproduction.get("exact"),
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
