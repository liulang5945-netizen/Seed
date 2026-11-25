"""N32.2.03 exploration: task-agnostic shared S vs external-column RT on the v2 composition data.

Pre-registered: plans/reference/DESIGN-N32-2-03_20261125.md (D-20261125-511).
Zero check/final. Exploration seeds 20261120-24 only; confirmation seeds are never touched.

Usage:
  python scripts/training/n32_2_03_s_vs_rt.py <seed>
  python scripts/training/n32_2_03_s_vs_rt.py <seed> --smoke
  python scripts/training/n32_2_03_s_vs_rt.py --aggregate
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from taiji import composition_encoding as enc
from taiji import native_composition_v2 as c2
from taiji.gru_rt import RTSubject

ROOT = Path(r"E:\Seed")
CARD = "N32.2.03"
KINDS = ("q1", "q2", "q3")
STEPS = 1200
BATCH = 48
LR = 3e-3
CLIP = 1.0
REPLAY = 16
EVAL_PER_KIND = 1024
CKPT = (150, 300, 600, 1200)
SMOKE_STEPS = 40
GATE = 0.85
G2_MIN = 0.05
G3_MIN = -0.05
TRANSFER_STEPS = 300
ANSWER_INDEX = {a: i for i, a in enumerate(enc.ANSWERS)}


# ---------------------------------------------------------------------------- data
def load_rows(split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {k: [] for k in KINDS}
    for row in c2.rows(split):
        grouped[str(row["kind"])].append(row)
    return grouped


def encode_rows(rows: list[dict[str, Any]], idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    ticks = torch.stack([enc.encode(rows[i]["scene"], str(rows[i]["text"])) for i in idxs])
    labels = torch.tensor([ANSWER_INDEX[str(rows[i]["answer"])] for i in idxs], dtype=torch.long)
    return ticks, labels


def build_eval(
    rows: dict[str, list[dict[str, Any]]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for kind in KINDS:
        idxs = list(range(min(EVAL_PER_KIND, len(rows[kind]))))
        out[kind] = encode_rows(rows[kind], idxs)
    return out


# ---------------------------------------------------------------------------- models
class SharedGRU(torch.nn.Module):
    """Task-agnostic: forward takes ticks only."""

    def __init__(self, units: int = 96, seed: int = 0) -> None:
        super().__init__()
        self.units = units
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.cell = torch.nn.GRUCell(enc.INPUT_DIM, units)
            self.read_gate = torch.nn.Linear(enc.INPUT_DIM, units)
            torch.nn.init.constant_(self.read_gate.bias, 1.0)
            self.head = torch.nn.Sequential(
                torch.nn.Linear(units, 64), torch.nn.Tanh(), torch.nn.Linear(64, len(enc.ANSWERS))
            )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = ticks.shape
        valid = ticks.abs().sum(-1) > 0
        state = ticks.new_zeros(batch, self.units)
        for t in range(horizon):
            step = self.cell(ticks[:, t, :], state)
            state = torch.where(valid[:, t].unsqueeze(-1), step, state)
        last_idx = (valid.sum(1) - 1).clamp(min=0)
        last_input = ticks[torch.arange(batch), last_idx, :]
        graded = state * torch.sigmoid(self.read_gate(last_input))
        return self.head(graded)


def build_rt(seed: int) -> RTSubject:
    return RTSubject({k: len(enc.ANSWERS) for k in KINDS}, units=48, seed=seed)


# ---------------------------------------------------------------------------- training
def lr_at(step: int, total: int) -> float:
    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, total))))


def train_shared(
    model: SharedGRU,
    rows: dict[str, list[dict[str, Any]]],
    order: tuple[str, ...],
    steps: int,
    replay: bool,
    eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ckpt: dict[str, dict[int, float]] | None = None,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    done: list[str] = []
    for kind in order:
        done.append(kind)
        data = rows[kind]
        gen = torch.Generator().manual_seed(hash((model.__class__.__name__, kind)) % (2**31))
        for i in range(steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(i, steps)
            idxs = torch.randint(0, len(data), (BATCH,), generator=gen).tolist()
            ticks, labels = encode_rows(data, idxs)
            loss = F.cross_entropy(model(ticks), labels)
            if replay and len(done) > 1:
                for old in done[:-1]:
                    old_rows = rows[old]
                    sel = torch.randint(0, len(old_rows), (REPLAY,), generator=gen).tolist()
                    t_old, y_old = encode_rows(old_rows, sel)
                    loss = loss + F.cross_entropy(model(t_old), y_old)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            if ckpt is not None and kind == "q3" and (i + 1) in CKPT:
                ckpt["s"][i + 1] = eval_shared(model, eval_data, "q3")
                model.train()


def train_rt_column(
    model: RTSubject,
    rows: dict[str, list[dict[str, Any]]],
    kind: str,
    steps: int,
    eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ckpt: dict[str, dict[int, float]] | None = None,
) -> None:
    params = [p for n, p in model.named_parameters() if n.startswith(f"columns.{kind}")]
    opt = torch.optim.Adam(params, lr=LR)
    model.train()
    data = rows[kind]
    gen = torch.Generator().manual_seed(hash(("rt", kind)) % (2**31))
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(i, steps)
        idxs = torch.randint(0, len(data), (BATCH,), generator=gen).tolist()
        ticks, labels = encode_rows(data, idxs)
        loss = F.cross_entropy(model(ticks, kind), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        if ckpt is not None and kind == "q3" and (i + 1) in CKPT:
            ckpt.setdefault("rt", {})[i + 1] = eval_rt(model, eval_data, "q3")


@torch.no_grad()
def eval_shared(
    model: SharedGRU, eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]], kind: str
) -> float:
    ticks, labels = eval_data[kind]
    model.eval()
    correct = 0
    for start in range(0, ticks.shape[0], 256):
        logits = model(ticks[start : start + 256])
        correct += int((logits.argmax(-1) == labels[start : start + 256]).sum())
    return correct / ticks.shape[0]


@torch.no_grad()
def eval_rt(
    model: RTSubject, eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]], kind: str
) -> float:
    ticks, labels = eval_data[kind]
    model.eval()
    correct = 0
    for start in range(0, ticks.shape[0], 256):
        logits = model(ticks[start : start + 256], kind)
        correct += int((logits.argmax(-1) == labels[start : start + 256]).sum())
    return correct / ticks.shape[0]


def transfer_probe(model: SharedGRU, rows: dict[str, list[dict[str, Any]]], eval_data) -> float:
    probe = copy.deepcopy(model)
    for p in probe.cell.parameters():
        p.requires_grad_(False)
    for p in probe.read_gate.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(probe.head.parameters(), lr=LR)
    data = rows["q3"]
    gen = torch.Generator().manual_seed(424242)
    probe.train()
    for _ in range(TRANSFER_STEPS):
        idxs = torch.randint(0, len(data), (BATCH,), generator=gen).tolist()
        ticks, labels = encode_rows(data, idxs)
        loss = F.cross_entropy(probe(ticks), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return eval_shared(probe, eval_data, "q3")


# ---------------------------------------------------------------------------- main
def run_one(seed: int, smoke: bool) -> dict[str, Any]:
    started = time.process_time()
    steps = SMOKE_STEPS if smoke else STEPS
    rows = load_rows("train")
    eval_data = build_eval(load_rows("val"))
    s = SharedGRU(units=96, seed=seed + 9001)
    rt = build_rt(seed + 9001)
    n_s = sum(p.numel() for p in s.parameters())
    n_rt = sum(p.numel() for p in rt.parameters())

    # RT: train each column sequentially, recording q3 checkpoints
    q3_ckpt_rt: dict[int, float] = {}
    train_rt_column(rt, rows, "q1", steps, eval_data)
    train_rt_column(rt, rows, "q2", steps, eval_data)
    train_rt_column(rt, rows, "q3", steps, eval_data, ckpt={"rt": q3_ckpt_rt})

    # S: sequential with replay; q3 checkpoints
    q3_ckpt_s: dict[int, float] = {}
    train_shared(s, rows, ("q1",), steps, replay=False, eval_data=eval_data)
    s_after_q1 = {k: eval_shared(s, eval_data, k) for k in KINDS}
    train_shared(s, rows, ("q2",), steps, replay=True, eval_data=eval_data)
    s_after_q2 = {k: eval_shared(s, eval_data, k) for k in KINDS}
    transfer = transfer_probe(s, rows, eval_data) if not smoke else 0.0
    train_shared(s, rows, ("q3",), steps, replay=True, eval_data=eval_data, ckpt={"s": q3_ckpt_s})
    s_final = {k: eval_shared(s, eval_data, k) for k in KINDS}
    rt_final = {k: eval_rt(rt, eval_data, k) for k in KINDS}

    g1_margin = min(s_final[k] - GATE for k in KINDS)
    common = [c for c in CKPT if c in q3_ckpt_s and c in q3_ckpt_rt]
    g2 = sum(q3_ckpt_s[c] - q3_ckpt_rt[c] for c in common) / len(common) if common else 0.0
    g3_bwt = min(s_final[k] - s_after_q2[k] for k in ("q1", "q2"))
    ceiling = bool(q3_ckpt_rt.get(150, 0.0) >= 0.95)
    payload = {
        "schema": "taiji-n32-2-03-v1",
        "card": CARD,
        "design": "DESIGN-N32-2-03_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "smoke": smoke,
        "steps_per_stage": steps,
        "params": {"shared_s": n_s, "rt": n_rt, "ratio_s_over_rt": n_s / max(1, n_rt)},
        "s_acc": {"after_q1": s_after_q1, "after_q2": s_after_q2, "final": s_final},
        "rt_acc_final": rt_final,
        "q3_checkpoints": {
            "s": {str(k): v for k, v in sorted(q3_ckpt_s.items())},
            "rt": {str(k): v for k, v in sorted(q3_ckpt_rt.items())},
        },
        "transfer_probe_q3": transfer,
        "metrics": {
            "g1_margin": g1_margin,
            "g2_shared_gain": g2,
            "g3_bwt_min": g3_bwt,
            "ceiling": ceiling,
        },
        "gates": {"q1": GATE, "q2": GATE, "q3": GATE},
        "cost": {"cpu": round(time.process_time() - started, 2)},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    return payload


def aggregate() -> int:
    reports = []
    missing = []
    for seed in (20261120, 20261121, 20261122, 20261123, 20261124):
        path = ROOT / f"reports/n32_2_03_seed{seed}.json"
        if not path.exists():
            missing.append(seed)
        else:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
    if missing or len(reports) != 5:
        exit_ = "engineering_failure"
        summary = {
            "schema": "taiji-n32-2-03-v1-summary",
            "card": CARD,
            "exit": exit_,
            "missing_seeds": missing,
        }
    else:
        positives = [
            r["metrics"]["g1_margin"] >= 0.05
            and r["metrics"]["g2_shared_gain"] >= 0.05
            and r["metrics"]["g3_bwt_min"] >= G3_MIN
            and not r["metrics"]["ceiling"]
            for r in reports
        ]
        proxy = min(min(r["metrics"]["g1_margin"], r["metrics"]["g2_shared_gain"]) for r in reports)
        if all(positives):
            exit_ = "signal_positive"
        elif proxy >= -0.05 or any(r["metrics"]["ceiling"] for r in reports):
            exit_ = "inconclusive"
        else:
            exit_ = "signal_negative"
        summary = {
            "schema": "taiji-n32-2-03-v1-summary",
            "card": CARD,
            "exit": exit_,
            "proxy_weakest_seed": proxy,
            "positives": positives,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "g1_margin": r["metrics"]["g1_margin"],
                    "g2": r["metrics"]["g2_shared_gain"],
                    "g3_bwt_min": r["metrics"]["g3_bwt_min"],
                    "ceiling": r["metrics"]["ceiling"],
                    "cpu": r["cost"]["cpu"],
                }
                for r in reports
            ],
            "cpu_total": round(sum(r["cost"]["cpu"] for r in reports), 2),
        }
    out = ROOT / "reports/n32_2_03_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if exit_ in ("signal_positive", "inconclusive", "signal_negative") else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return aggregate()
    if args.seed is None:
        raise SystemExit("seed required")
    torch.set_num_threads(1)
    payload = run_one(args.seed, args.smoke)
    suffix = "_smoke" if args.smoke else ""
    out = ROOT / f"reports/n32_2_03_seed{args.seed}{suffix}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "cost": payload["cost"],
                "metrics": payload["metrics"],
                "params": payload["params"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
