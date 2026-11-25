"""N32.3.01: R1 learnability pre-check (single task, no arm comparison).

Pre-registered: plans/reference/DESIGN-N32-3-01_20261125.md (D-20261125-515).
Trains a single-column GRU+read_gate learner on Q2 (and a few Q3 diagnostics),
to find whether a single task is learnable at all (gate 0.85) before any S-vs-RT
comparison. Zero check/final. Exploration seed only; confirmation seeds untouched.

Usage: python scripts/training/n32_3_01_learnability_precheck.py [--smoke]
Output: reports/n32_3_01_precheck.json
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

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from taiji.gru_rt import RTSubject

sys.path.insert(0, r"E:\Seed\scripts\training")
import n32_2_03_s_vs_rt as m  # reuse data/encoding helpers

ROOT = Path(r"E:\Seed")
CARD = "N32.3.01"
SEED = 20261120
GATE = 0.85
SMOKE_STEPS = 40
CONFIGS: tuple[dict[str, Any], ...] = (
    {"kind": "q2", "units": 48, "steps": 1200, "balanced": False},
    {"kind": "q2", "units": 48, "steps": 3000, "balanced": False},
    {"kind": "q2", "units": 48, "steps": 6000, "balanced": False},
    {"kind": "q3", "units": 48, "steps": 3000, "balanced": False},
    {"kind": "q3", "units": 48, "steps": 3000, "balanced": True},
)


def lr_at(step: int, total: int) -> float:
    return 3e-3 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, total))))


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(m.enc.ANSWERS)).clamp(min=1)
    return (labels.numel() / (len(counts) * counts)).float()


def eval_column(
    model: RTSubject, eval_data: dict[str, tuple[torch.Tensor, torch.Tensor]], kind: str
) -> dict[str, float]:
    ticks, labels = eval_data[kind]
    model.eval()
    with torch.no_grad():
        correct = 0
        unknown = 0
        for start in range(0, ticks.shape[0], 256):
            logits = model(ticks[start : start + 256], kind)
            pred = logits.argmax(-1)
            correct += int((pred == labels[start : start + 256]).sum())
            unknown += int((pred == m.enc.ANSWERS.index("未知")).sum())
    n = ticks.shape[0]
    return {"acc": correct / n, "pred_unknown_rate": unknown / n, "n": n}


def run_config(cfg: dict[str, Any], rows, eval_data, smoke: bool) -> dict[str, Any]:
    started = time.process_time()
    steps = SMOKE_STEPS if smoke else int(cfg["steps"])
    kind, units = str(cfg["kind"]), int(cfg["units"])
    model = RTSubject({kind: len(m.enc.ANSWERS)}, units=units, seed=SEED + 9001)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=3e-3)
    data = rows[kind]
    gen = m.rng_for(SEED + 500, kind)
    weights = None
    if cfg["balanced"] and not smoke:
        labels_all = torch.tensor(
            [m.ANSWER_INDEX[str(r["answer"])] for r in data], dtype=torch.long
        )
        weights = class_weights(labels_all)
    model.train()
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(i, steps)
        idxs = torch.randint(0, len(data), (m.BATCH,), generator=gen).tolist()
        ticks, labels = m.encode_rows(data, idxs)
        loss = F.cross_entropy(model(ticks, kind), labels, weight=weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    out_eval = eval_column(model, eval_data, kind)
    return {
        "config": cfg,
        "steps_run": steps,
        "eval": out_eval,
        "pass": out_eval["acc"] >= GATE,
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()
    rows = m.load_rows("train")
    eval_data = m.build_eval(m.load_rows("val"))
    results = [run_config(cfg, rows, eval_data, args.smoke) for cfg in CONFIGS]
    cpu = round(time.process_time() - started, 2)
    q2 = [r for r in results if r["config"]["kind"] == "q2"]
    q3 = [r for r in results if r["config"]["kind"] == "q3"]
    q2_pass = any(r["pass"] for r in q2)
    q3_pass = any(r["pass"] for r in q3)
    if args.smoke:
        exit_ = "smoke"
    elif q2_pass and q3_pass:
        exit_ = "precheck_pass"
    elif q2_pass:
        exit_ = "precheck_partial"
    else:
        exit_ = "precheck_fail"
    payload = {
        "schema": "taiji-n32-3-01-v1",
        "card": CARD,
        "design": "DESIGN-N32-3-01_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "smoke": args.smoke,
        "gate": GATE,
        "results": results,
        "q2_pass": q2_pass,
        "q3_pass": q3_pass,
        "exit": exit_,
        "cost": {"cpu": cpu, "cap_cpu": 600.0},
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    out = ROOT / "reports/n32_3_01_precheck.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "q2_pass": q2_pass,
                "q3_pass": q3_pass,
                "cpu": cpu,
                "best": max((r["eval"]["acc"] for r in results), default=0.0),
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("precheck_pass", "precheck_partial", "precheck_fail", "smoke") else 1


if __name__ == "__main__":
    raise SystemExit(main())
