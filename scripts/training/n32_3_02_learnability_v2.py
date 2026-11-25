"""N32.3.02: R1 learnability diagnostic v2 (pre-encoded, large batch, MLP reference, G1-G3).

Pre-registered: plans/reference/DESIGN-N32-3-02_20261125.md (D-20261125-518).
Zero check/final; exploration seed only; confirmation seeds untouched.

Arms (Q2 = one-hop relation, Q3 = two-hop):
  A  GRU u48, Q2 only,  2,500 steps, batch 256
  B  GRU u96, Q2 only,  2,500 steps, batch 256
  C  MLP reference (1152->256->10), Q2 only, 2,500 steps, batch 512
  D  course GRU u96: Q2 1,500 steps -> Q3 1,500 steps (balanced CE)

Data are pre-encoded once to float16 tensors; each step casts its batch to float32.
Usage: python scripts/training/n32_3_02_learnability_v2.py [--smoke]
Output: reports/n32_3_02_precheck.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, r"E:\Seed")
sys.path.insert(0, r"E:\Seed\scripts\training")
import n32_2_03_s_vs_rt as m
import torch
import torch.nn.functional as F

ROOT = Path(r"E:\Seed")
CARD = "N32.3.02"
SEED = 20261120
GATE = 0.85
G3_MIN = -0.05
SMOKE_STEPS = 40
BATCH = 256
BATCH_MLP = 512
LR = 3e-3
Q2_TRAIN_ROWS = 24000
Q3_TRAIN_ROWS = 24000
STEPS = {"A": 2500, "B": 2500, "C": 2500, "D1": 1500, "D2": 1500}


class MLP(torch.nn.Module):
    def __init__(
        self, in_dim: int = 24 * 48, hidden: int = 256, out: int = 10, seed: int = 0
    ) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_dim, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, out)
            )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        return self.net(ticks.reshape(ticks.shape[0], -1))


def lr_at(step: int, total: int) -> float:
    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / max(1, total))))


def subset_q3(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic 24k subset stratified by answer class."""
    gen = m.rng_for(SEED + 13, "q3subset")
    by_answer: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_answer.setdefault(str(row["answer"]), []).append(i)
    out: list[int] = []
    per = Q3_TRAIN_ROWS // max(1, len(by_answer))
    for idxs in by_answer.values():
        perm = torch.randperm(len(idxs), generator=gen).tolist()
        out.extend(idxs[j] for j in perm[:per])
    out = out[:Q3_TRAIN_ROWS]
    out.sort()
    return [rows[i] for i in out]


def preencode(rows: list[dict[str, Any]], n: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    rows = rows[:n]
    idxs = list(range(len(rows)))
    ticks = torch.stack([m.enc.encode(rows[i]["scene"], str(rows[i]["text"])) for i in idxs])
    labels = torch.tensor([m.ANSWER_INDEX[str(rows[i]["answer"])] for i in idxs], dtype=torch.long)
    half = ticks.to(torch.float16)
    digest = hashlib.sha256(half.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    return half, labels, digest


def eval_model(model: torch.nn.Module, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, ticks.shape[0], 512):
            logits = model(ticks[start : start + 512].float())
            correct += int((logits.argmax(-1) == labels[start : start + 512]).sum())
    return correct / ticks.shape[0]


def train_arm(
    model: torch.nn.Module,
    ticks: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    batch: int,
    seed_tag: str,
    weight: torch.Tensor | None = None,
) -> float:
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=LR)
    gen = m.rng_for(SEED + 101, seed_tag)
    model.train()
    for i in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(i, steps)
        idx = torch.randint(0, ticks.shape[0], (min(batch, ticks.shape[0]),), generator=gen)
        loss = F.cross_entropy(model(ticks[idx].float()), labels[idx], weight=weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.process_time()
    enc_started = time.process_time()
    rows = m.load_rows("train")
    val = m.load_rows("val")
    q2_ticks, q2_labels, q2_digest = preencode(rows["q2"], Q2_TRAIN_ROWS)
    q3_rows = subset_q3(rows["q3"])
    q3_ticks, q3_labels, q3_digest = preencode(q3_rows, Q3_TRAIN_ROWS)
    v2_ticks, v2_labels, _ = preencode(val["q2"], 1024)
    v3_ticks, v3_labels, _ = preencode(val["q3"], 1024)
    enc_cpu = round(time.process_time() - enc_started, 2)

    steps = {k: (SMOKE_STEPS if args.smoke else v) for k, v in STEPS.items()}
    results: dict[str, Any] = {}

    a = m.SharedGRU(units=48, seed=SEED + 9001)
    train_arm(a, q2_ticks, q2_labels, steps["A"], BATCH, "A")
    results["A_gru48_q2"] = {"q2": eval_model(a, v2_ticks, v2_labels), "steps": steps["A"]}

    b = m.SharedGRU(units=96, seed=SEED + 9001)
    train_arm(b, q2_ticks, q2_labels, steps["B"], BATCH, "B")
    results["B_gru96_q2"] = {"q2": eval_model(b, v2_ticks, v2_labels), "steps": steps["B"]}

    c = MLP(seed=SEED + 9001)
    train_arm(c, q2_ticks, q2_labels, steps["C"], BATCH_MLP, "C")
    results["C_mlp_q2"] = {"q2": eval_model(c, v2_ticks, v2_labels), "steps": steps["C"]}

    d = m.SharedGRU(units=96, seed=SEED + 9001)
    train_arm(d, q2_ticks, q2_labels, steps["D1"], BATCH, "D1")
    q2_peak = eval_model(d, v2_ticks, v2_labels)
    q3_counts = torch.bincount(q3_labels, minlength=len(m.enc.ANSWERS)).clamp(min=1)
    q3_weight = (q3_labels.numel() / (len(q3_counts) * q3_counts)).float()
    train_arm(d, q3_ticks, q3_labels, steps["D2"], BATCH, "D2", weight=q3_weight)
    q2_after = eval_model(d, v2_ticks, v2_labels)
    q3_final = eval_model(d, v3_ticks, v3_labels)
    results["D_course_gru96"] = {
        "q2_peak": q2_peak,
        "q2_after_q3": q2_after,
        "q3_after_q3": q3_final,
        "steps": [steps["D1"], steps["D2"]],
    }

    g1 = results["A_gru48_q2"]["q2"] >= GATE or results["B_gru96_q2"]["q2"] >= GATE
    best_gru_q2 = max(results["A_gru48_q2"]["q2"], results["B_gru96_q2"]["q2"])
    g2 = best_gru_q2 - results["C_mlp_q2"]["q2"]
    g3 = q2_after - q2_peak
    if args.smoke:
        exit_ = "smoke"
    elif g1 and g3 >= G3_MIN:
        exit_ = "learnable"
    elif g1:
        exit_ = "learnable_no_retention"
    else:
        exit_ = "not_learnable"
    payload = {
        "schema": "taiji-n32-3-02-v1",
        "card": CARD,
        "design": "DESIGN-N32-3-02_20261125.md",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "smoke": args.smoke,
        "batch": {"gru": BATCH, "mlp": BATCH_MLP},
        "steps": steps,
        "subset": {
            "q2_rows": int(q2_labels.numel()),
            "q3_rows": int(q3_labels.numel()),
            "q2_digest": q2_digest,
            "q3_digest": q3_digest,
        },
        "results": results,
        "gates": {
            "G1_learnable": g1,
            "G2_recurrent_gain_over_mlp": g2,
            "G3_bwt": g3,
            "G3_min": G3_MIN,
        },
        "exit": exit_,
        "cost": {
            "cpu": round(time.process_time() - started, 2),
            "encode_cpu": enc_cpu,
            "cap_cpu": 600.0,
        },
        "discipline": {"check_reads": 0, "final_reads": 0, "exploratory": True},
    }
    out = ROOT / "reports/n32_3_02_precheck.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "exit": exit_,
                "g1": g1,
                "g2": round(g2, 4),
                "g3": round(g3, 4),
                "cpu": payload["cost"]["cpu"],
                "enc_cpu": enc_cpu,
                "accs": {
                    k: {kk: round(vv, 4) for kk, vv in v.items() if kk != "steps"}
                    for k, v in results.items()
                },
            },
            ensure_ascii=False,
        )
    )
    return 0 if exit_ in ("learnable", "learnable_no_retention", "not_learnable", "smoke") else 1


if __name__ == "__main__":
    raise SystemExit(main())
