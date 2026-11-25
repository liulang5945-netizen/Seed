"""N32.5.04: ModalityEdgeHop Q3 two-hop learnability benchmark (Route P1, <= 320 CPU).

Pre-registered: plans/reference/DESIGN-N32-5-04_20261125.md (D-20261125-540).
Reuses the exact bit-verified Q3 dataset from N32.5.01 (SHA256 digests asserted,
same moments_dim standardization) and tests task-agnostic ModalityEdgeHop on
10,000 steps x 256 (screening seed 20261120).

Usage:
  python scripts/training/n32_5_04_modality_edge_hop_q3.py --smoke
  python scripts/training/n32_5_04_modality_edge_hop_q3.py
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
from scripts.training.n32_5_01_q3_gru_learnability import (  # noqa: E402
    AUG_COPIES,
    BATCH,
    EVAL_EVERY,
    EXPECTED_DIGESTS,
    GATE_GRU,
    Q3_LEN,
    SCREEN_SEED,
    TRAIN_PROBE,
    CapExceeded,
    NonFiniteLoss,
    encode_q3_fast,
    int_seed,
    moments_dim,
    tensor_digest,
)
from taiji import composition_encoding as ce  # noqa: E402
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.modality_edge_hop import (  # noqa: E402
    ARCH_VERSION,
    ModalityEdgeHop,
    extract_modality_masks,
)
from taiji.native_composition import NAMES  # noqa: E402

CARD = "N32.5.04"
ARM = "ModalityEdgeHop_q3"
D_MODEL = 64
D_EDGE = 64
N_EDGE_LAYERS = 2
N_HOPS = 2
STEPS = 10000
LR = 3e-3
WD = 1e-4
WARMUP_STEPS = 200
SMOKE_STEPS = 50
CAP_CPU = 320.0
PROJECTION_MARGIN = 1.10

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_5_04_smoke.json"
FULL_REPORT = REPORTS / "n32_5_04_learnability.json"


def build_data() -> dict[str, Any]:
    train_x_raw, train_y = encode_q3_fast(nc2.load_scenes("train"), AUG_COPIES)
    val_x_raw, val_y = encode_q3_fast(nc2.load_scenes("val"), 0)
    digests = {
        "train": tensor_digest(train_x_raw, train_y),
        "val": tensor_digest(val_x_raw, val_y),
    }
    for split, expected in EXPECTED_DIGESTS.items():
        if digests[split] != expected:
            raise AssertionError(
                f"{CARD} fast encoder {split} digest mismatch: {digests[split]} != {expected}"
            )
    train_sm, train_qm = extract_modality_masks(train_x_raw)
    val_sm, val_qm = extract_modality_masks(val_x_raw)
    mu, sd = moments_dim(train_x_raw)
    probe = torch.randperm(train_x_raw.shape[0], generator=torch.Generator().manual_seed(7))
    return {
        "train_x": train_x_raw,
        "train_sm": train_sm,
        "train_qm": train_qm,
        "train_y": train_y,
        "val_x": val_x_raw,
        "val_sm": val_sm,
        "val_qm": val_qm,
        "val_y": val_y,
        "probe_idx": probe[:TRAIN_PROBE],
        "mu": mu,
        "sd": sd,
        "digests": digests,
        "majority": float(torch.bincount(val_y).max() / val_y.numel()),
    }


def prep(x16: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    mu: torch.Tensor = data["mu"]
    sd: torch.Tensor = data["sd"]
    return (x16.float() - mu) / sd


def make_model(seed: int) -> ModalityEdgeHop:
    return ModalityEdgeHop(
        len(ce.ANSWERS),
        d_model=D_MODEL,
        d_edge=D_EDGE,
        n_edge_layers=N_EDGE_LAYERS,
        n_hops=N_HOPS,
        seed=int_seed(CARD, seed, ARM, "init"),
    )


def _lr_at(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR * float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(max(total_steps - WARMUP_STEPS, 1))
    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _entity_mean(val_detail: dict[str, Any]) -> float:
    per_class = val_detail["per_class"]
    vals = [float(per_class[name]) for name in NAMES if name in per_class]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_edge(
    model: ModalityEdgeHop,
    data: dict[str, Any],
    x16: torch.Tensor,
    sm: torch.Tensor,
    qm: torch.Tensor,
    y: torch.Tensor,
    *,
    detail: bool = False,
) -> dict[str, Any]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x16.shape[0], 1024):
            logits = model(
                prep(x16[i : i + 1024], data),
                sm[i : i + 1024],
                qm[i : i + 1024],
            )
            preds.append(logits.argmax(-1))
    pred = torch.cat(preds)
    out: dict[str, Any] = {"acc": float((pred == y).float().mean()), "n": int(y.numel())}
    if detail:
        unknown = ce.ANSWER_INDEX["未知"]
        out["pred_unknown_rate"] = float((pred == unknown).float().mean())
        out["per_class"] = {
            a: float((pred[y == i] == i).float().mean())
            for i, a in enumerate(ce.ANSWERS)
            if int((y == i).sum()) > 0
        }
    return out


def train_seed(
    seed: int, data: dict[str, Any], deadline: float, steps: int = STEPS
) -> dict[str, Any]:
    model = make_model(seed)
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, ARM, "batches"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    n = data["train_x"].shape[0]
    curve: list[list[float]] = []

    t0 = time.process_time()
    eval_cpu_total = 0.0
    model.train()
    for i in range(steps):
        if time.process_time() > deadline:
            raise CapExceeded(f"{ARM} seed {seed} step {i}")
        lr = _lr_at(i, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        logits = model(
            prep(data["train_x"][idx], data),
            data["train_sm"][idx],
            data["train_qm"][idx],
        )
        loss = F.cross_entropy(logits, data["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{ARM} seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == steps:
            e0 = time.process_time()
            acc = evaluate_edge(
                model,
                data,
                data["val_x"],
                data["val_sm"],
                data["val_qm"],
                data["val_y"],
            )["acc"]
            eval_cpu_total += time.process_time() - e0
            curve.append([i + 1, acc])
            model.train()

    e1 = time.process_time()
    val_detail = evaluate_edge(
        model,
        data,
        data["val_x"],
        data["val_sm"],
        data["val_qm"],
        data["val_y"],
        detail=True,
    )
    probe = data["probe_idx"]
    probe_res = evaluate_edge(
        model,
        data,
        data["train_x"][probe],
        data["train_sm"][probe],
        data["train_qm"][probe],
        data["train_y"][probe],
    )
    eval_cpu_total += time.process_time() - e1

    total_cpu = time.process_time() - t0
    train_loop_cpu = max(total_cpu - eval_cpu_total, 0.0)
    return {
        "arm": ARM,
        "arch_version": ARCH_VERSION,
        "kind": "q3",
        "seed": seed,
        "steps": steps,
        "presentations": steps * BATCH,
        "val": val_detail,
        "entity_mean_acc": round(_entity_mean(val_detail), 6),
        "train_probe_acc": probe_res["acc"],
        "curve": curve,
        "n_params": model.n_params(),
        "cpu": round(total_cpu, 2),
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu_total": round(eval_cpu_total, 3),
    }


def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    row = train_seed(SCREEN_SEED, data, deadline, steps=SMOKE_STEPS)
    model = make_model(SCREEN_SEED)
    e0 = time.process_time()
    evaluate_edge(
        model,
        data,
        data["val_x"],
        data["val_sm"],
        data["val_qm"],
        data["val_y"],
        detail=True,
    )
    eval_cpu = time.process_time() - e0
    per_step_cpu = row["train_loop_cpu"] / SMOKE_STEPS

    smoke_cpu = time.process_time() - started
    n_evals = STEPS // EVAL_EVERY + 2
    per_seed = (per_step_cpu * STEPS + n_evals * eval_cpu) * PROJECTION_MARGIN
    proj_1 = round(encode_cpu + per_seed, 2)
    planned = "screen_only" if smoke_cpu + proj_1 <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "per_step_cpu": per_step_cpu,
        "eval_cpu": eval_cpu,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_1seed": proj_1,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    proj = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["projection"]
    planned = str(proj["planned"])
    if planned != "screen_only":
        return {
            "planned": planned,
            "runs": [],
            "exit": "budget_insufficient",
            "reason": f"smoke projection does not fit CAP_CPU={CAP_CPU}",
        }

    deadline = started + budget
    screen = train_seed(SCREEN_SEED, data, deadline, steps=STEPS)
    acc = float(screen["val"]["acc"])
    exit_label = "screen_passed_needs_rep_budget" if acc >= GATE_GRU else "q3_not_learnable"
    return {
        "planned": planned,
        "runs": [screen],
        "screen_acc": acc,
        "screen_entity_mean_acc": screen["entity_mean_acc"],
        "accs": [acc],
        "exit": exit_label,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    design = preflight(CARD, [SCREEN_SEED], record=False).name
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)

    report: dict[str, Any] = {
        "schema": "taiji-n32-5-04-v1",
        "card": CARD,
        "arm": ARM,
        "arch_version": ARCH_VERSION,
        "model_cfg": {
            "d_model": D_MODEL,
            "d_edge": D_EDGE,
            "n_edge_layers": N_EDGE_LAYERS,
            "n_hops": N_HOPS,
        },
        "design": [design],
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            "q3": {
                "train_rows": int(data["train_y"].numel()),
                "val_rows": int(data["val_y"].numel()),
                "aug_copies": AUG_COPIES,
                "digests": data["digests"],
                "val_majority": data["majority"],
            }
        },
        "length": Q3_LEN,
        "gates": {"q3": GATE_GRU},
        "encode_cpu": encode_cpu,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "exploratory": True},
    }

    status = 0
    if args.smoke:
        report["projection"] = smoke(data, started, encode_cpu)
        out = SMOKE_REPORT
        if report["projection"]["planned"] == "budget_insufficient":
            status = 3
    else:
        smoke_doc = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
        budget = CAP_CPU - float(smoke_doc["projection"]["smoke_cpu"])
        try:
            report.update(full(data, started, budget))
        except CapExceeded as exc:
            report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
        except NonFiniteLoss as exc:
            report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
        out = FULL_REPORT
        if report.get("exit") in ("engineering_failure", "budget_insufficient"):
            status = 3

    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        k: report.get(k) for k in ("exit", "planned", "accs", "screen_entity_mean_acc", "cost")
    }
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
