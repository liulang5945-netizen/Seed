"""N32.4.01: Q3 learnability check (+ Q2 retrain control) with MLP engineering gate.

Pre-registered: plans/reference/DESIGN-N32-4-01_20261125.md (D-20261125-523).
Pipeline follows N32.3.03 (D-522): standardised inputs, name-relabel augmentation x3 on
train worlds only, question-first (QF) tick order, GRU u96 + final-tick read_gate.

  E0  MLP on Q3 (960 -> 256 -> 256 -> 10), 8,000 steps x 256.  Gate: val >= 0.90.
  G3  GRU u96 on Q3 (QF), 6,000 steps x 256.  Screen seed 20261120 >= 0.85, then 20261121/22.
  R2  GRU u96 on Q2 (QF), 6,000 steps x 256.  Report-only regression control.

Exploration seeds only; confirm split and v2 confirm worlds are never read.

Usage:
  python scripts/training/n32_4_01_q3_learnability.py --smoke
  python scripts/training/n32_4_01_q3_learnability.py
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.training.n31_guard import preflight  # noqa: E402
from taiji import composition_encoding as ce  # noqa: E402
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.native_composition import NAMES  # noqa: E402

CARD = "N32.4.01"
SEEDS = (20261120, 20261121, 20261122)
SCREEN_SEED = SEEDS[0]
AUG_COPIES = 3
AUG_SEED = 20261125
N_SCENE = 3
LENGTHS = {"q2": 13, "q3": 20}
GATE_MLP = 0.90
GATE_GRU = 0.85
CAP_CPU = 800.0
BATCH = 256
HIDDEN = 96
EVAL_EVERY = 2000
SMOKE_STEPS = 40
TRAIN_PROBE = 4000
ARMS: dict[str, dict[str, Any]] = {
    "E0_mlp_q3": {
        "kind": "q3",
        "model": "mlp",
        "steps": 16000,
        "lr": 1e-3,
        "wd": 1e-4,
        "cosine": False,
    },
    "G3_gru96_q3": {
        "kind": "q3",
        "model": "gru",
        "steps": 6000,
        "lr": 3e-3,
        "wd": 0.0,
        "cosine": True,
    },
    "R2_gru96_q2": {
        "kind": "q2",
        "model": "gru",
        "steps": 6000,
        "lr": 3e-3,
        "wd": 0.0,
        "cosine": True,
    },
}
REPORTS = ROOT / "reports"


class CapExceeded(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


def int_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**62)


def qf_order(length: int) -> list[int]:
    return [*range(N_SCENE, length), *range(N_SCENE)]


# ----------------------------------------------------------------------------- data
def relabel(scene: dict[str, Any], perm: list[str]) -> dict[str, Any]:
    mapping = dict(zip(NAMES, perm, strict=True))
    bodies = [
        {"name": mapping[b["name"]], "position": list(b["position"])} for b in scene["bodies"]
    ]
    return {**scene, "bodies": bodies}


def encode_kind(
    scenes: list[dict[str, Any]], kind: str, copies: int
) -> tuple[torch.Tensor, torch.Tensor]:
    length = LENGTHS[kind]
    gen = torch.Generator().manual_seed(AUG_SEED)
    ticks: list[torch.Tensor] = []
    labels: list[int] = []
    for scene in scenes:
        variants = [scene]
        for _ in range(copies):
            perm = [NAMES[i] for i in torch.randperm(len(NAMES), generator=gen).tolist()]
            variants.append(relabel(scene, perm))
        for v in variants:
            for q in nc2.questions(v):
                if q["kind"] != kind:
                    continue
                if len(v["bodies"]) + len(q["text"]) != length:
                    raise ValueError(
                        f"unexpected {kind} length: {len(v['bodies']) + len(q['text'])}"
                    )
                ticks.append(ce.encode(v, q["text"])[:length].to(torch.float16))
                labels.append(ce.ANSWER_INDEX[str(q["answer"])])
    return torch.stack(ticks), torch.tensor(labels, dtype=torch.long)


def moments(x16: torch.Tensor, flat: bool) -> tuple[torch.Tensor, torch.Tensor]:
    length = x16.shape[1]
    width = length * 48 if flat else 48
    s = torch.zeros(width, dtype=torch.float64)
    s2 = torch.zeros(width, dtype=torch.float64)
    n = 0
    for i in range(0, x16.shape[0], 4096):
        c = x16[i : i + 4096].double()
        c = c.reshape(c.shape[0], -1) if flat else c.reshape(-1, 48)
        s += c.sum(0)
        s2 += (c * c).sum(0)
        n += c.shape[0]
    mu = s / n
    sd = (s2 / n - mu * mu).clamp(min=0.0).sqrt()
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return mu.float(), sd.float()


def digest(x: torch.Tensor, y: torch.Tensor) -> str:
    return hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


def build_data() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("q2", "q3"):
        train_x, train_y = encode_kind(nc2.load_scenes("train"), kind, AUG_COPIES)
        val_x, val_y = encode_kind(nc2.load_scenes("val"), kind, 0)
        probe = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(7))
        out[kind] = {
            "train_x": train_x,
            "train_y": train_y,
            "val_x": val_x,
            "val_y": val_y,
            "probe_idx": probe[:TRAIN_PROBE],
            "norm_dim": moments(train_x, flat=False),
            "norm_flat": moments(train_x, flat=True),
            "digests": {"train": digest(train_x, train_y), "val": digest(val_x, val_y)},
            "majority": float(torch.bincount(val_y).max() / val_y.numel()),
        }
    return out


def prep(x16: torch.Tensor, spec: dict[str, Any], data: dict[str, Any]) -> torch.Tensor:
    d = data[spec["kind"]]
    x = x16.float()
    if spec["model"] == "mlp":
        mu, sd = d["norm_flat"]
        return (x.reshape(x.shape[0], -1) - mu) / sd
    mu, sd = d["norm_dim"]
    x = (x - mu) / sd
    return x[:, qf_order(x.shape[1])]


# ---------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, in_dim: int, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, len(ce.ANSWERS)),
            )
            for mod in self.net:
                if isinstance(mod, nn.Linear):
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class GatedGRU(nn.Module):
    def __init__(self, seed: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gru = nn.GRU(48, hidden, batch_first=True)
            self.read_gate = nn.Linear(48, hidden)
            nn.init.constant_(self.read_gate.bias, 1.0)
            self.head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, len(ce.ANSWERS))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(x)
        gate = torch.sigmoid(self.read_gate(x[:, -1]))
        out: torch.Tensor = self.head(seq[:, -1] * gate)
        return out


def make_model(spec: dict[str, Any], seed: int, arm: str) -> nn.Module:
    init = int_seed(CARD, seed, arm, "init")
    if spec["model"] == "mlp":
        return MLP(LENGTHS[spec["kind"]] * 48, init)
    return GatedGRU(init)


# ----------------------------------------------------------------------- train / eval
def evaluate(
    model: nn.Module,
    spec: dict[str, Any],
    data: dict[str, Any],
    x16: torch.Tensor,
    y: torch.Tensor,
    detail: bool = False,
) -> dict[str, Any]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x16.shape[0], 1024):
            preds.append(model(prep(x16[i : i + 1024], spec, data)).argmax(-1))
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


def train_arm(
    arm: str, seed: int, data: dict[str, Any], deadline: float, steps: int | None = None
) -> dict[str, Any]:
    spec = ARMS[arm]
    d = data[spec["kind"]]
    total = spec["steps"] if steps is None else steps
    model = make_model(spec, seed, arm)
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, arm, "batches"))
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    n = d["train_x"].shape[0]
    curve: list[list[float]] = []
    t0 = time.process_time()
    model.train()
    for i in range(total):
        if time.process_time() > deadline:
            raise CapExceeded(f"{arm} seed {seed} step {i}")
        if spec["cosine"]:
            lr = spec["lr"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * i / total)))
            for g in opt.param_groups:
                g["lr"] = lr
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        loss = F.cross_entropy(model(prep(d["train_x"][idx], spec, data)), d["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{arm} seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == total:
            acc = evaluate(model, spec, data, d["val_x"], d["val_y"])["acc"]
            curve.append([i + 1, acc])
            model.train()
    probe = d["probe_idx"]
    return {
        "arm": arm,
        "kind": spec["kind"],
        "seed": seed,
        "steps": total,
        "presentations": total * BATCH,
        "val": evaluate(model, spec, data, d["val_x"], d["val_y"], detail=True),
        "train_probe_acc": evaluate(model, spec, data, d["train_x"][probe], d["train_y"][probe])[
            "acc"
        ],
        "curve": curve,
        "n_params": sum(p.numel() for p in model.parameters()),
        "cpu": round(time.process_time() - t0, 2),
    }


# ---------------------------------------------------------------------------- driver
def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    per_step: dict[str, float] = {}
    eval_cost: dict[str, float] = {}
    for arm, spec in ARMS.items():
        row = train_arm(arm, SCREEN_SEED, data, deadline, steps=SMOKE_STEPS)
        model = make_model(spec, SCREEN_SEED, arm)
        d = data[spec["kind"]]
        t0 = time.process_time()
        evaluate(model, spec, data, d["val_x"], d["val_y"], detail=True)
        eval_cost[arm] = time.process_time() - t0
        per_step[arm] = max(row["cpu"] - 3 * eval_cost[arm], 0.0) / SMOKE_STEPS

    def arm_cost(arm: str) -> float:
        steps = ARMS[arm]["steps"]
        return per_step[arm] * steps + (steps / EVAL_EVERY + 2) * eval_cost[arm]

    smoke_cpu = time.process_time() - started
    g3 = arm_cost("G3_gru96_q3")
    full_all = encode_cpu + arm_cost("E0_mlp_q3") + g3 + arm_cost("R2_gru96_q2") + 2 * g3
    full_no_r2 = encode_cpu + arm_cost("E0_mlp_q3") + 3 * g3
    if smoke_cpu + full_all <= CAP_CPU:
        planned = "all"
    elif smoke_cpu + full_no_r2 <= CAP_CPU:
        planned = "no_r2"
    else:
        planned = ""
    return {
        "per_step_cpu": per_step,
        "eval_cpu": eval_cost,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_full_all": round(full_all, 1),
        "projected_full_no_r2": round(full_no_r2, 1),
        "cap": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    deadline = started + budget
    planned = json.loads((REPORTS / "n32_4_01_smoke.json").read_text(encoding="utf-8"))[
        "projection"
    ]["planned"]
    if planned not in ("all", "no_r2"):
        raise SystemExit("smoke projection does not fit the budget; card not run")
    out: dict[str, Any] = {"planned": planned, "runs": []}
    mlp = train_arm("E0_mlp_q3", SCREEN_SEED, data, deadline)
    out["runs"].append(mlp)
    if mlp["val"]["acc"] < GATE_MLP:
        out["exit"] = "engineering_failure"
        out["reason"] = f"MLP Q3 gate {mlp['val']['acc']:.3f} < {GATE_MLP}"
        return out
    screen = train_arm("G3_gru96_q3", SCREEN_SEED, data, deadline)
    out["runs"].append(screen)
    out["best_arm"] = "G3_gru96_q3"
    if screen["val"]["acc"] < GATE_GRU:
        out["exit"] = "q3_not_learnable"
        return out
    reps = [train_arm("G3_gru96_q3", s, data, deadline) for s in SEEDS[1:]]
    out["runs"].extend(reps)
    accs = [screen["val"]["acc"], *[r["val"]["acc"] for r in reps]]
    out["best_arm_accs"] = accs
    out["exit"] = "learnable" if min(accs) >= GATE_GRU else "learnable_unstable"
    if planned == "all":
        r2 = train_arm("R2_gru96_q2", SCREEN_SEED, data, deadline)
        out["runs"].append(r2)
        out["q2_retrain"] = {"acc": r2["val"]["acc"], "ok": r2["val"]["acc"] >= GATE_GRU}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    designs = {preflight(CARD, [s], record=False).name for s in SEEDS}
    torch.set_num_threads(1)
    started = time.process_time()
    data = build_data()
    encode_cpu = round(time.process_time() - started, 2)
    report: dict[str, Any] = {
        "schema": "taiji-n32-4-01-v1",
        "card": CARD,
        "design": sorted(designs),
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            kind: {
                "train_rows": int(data[kind]["train_y"].numel()),
                "val_rows": int(data[kind]["val_y"].numel()),
                "aug_copies": AUG_COPIES,
                "digests": data[kind]["digests"],
                "val_majority": data[kind]["majority"],
            }
            for kind in ("q2", "q3")
        },
        "lengths": LENGTHS,
        "gates": {"mlp_engineering": GATE_MLP, "gru": GATE_GRU},
        "encode_cpu": encode_cpu,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "exploratory": True},
    }
    status = 0
    if args.smoke:
        report["projection"] = smoke(data, started, encode_cpu)
        out = REPORTS / "n32_4_01_smoke.json"
        status = 0 if report["projection"]["planned"] else 3
    else:
        smoke_doc = json.loads((REPORTS / "n32_4_01_smoke.json").read_text(encoding="utf-8"))
        budget = CAP_CPU - float(smoke_doc["projection"]["smoke_cpu"])
        try:
            report.update(full(data, started, budget))
        except CapExceeded as exc:
            report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
        except NonFiniteLoss as exc:
            report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
        out = REPORTS / "n32_4_01_learnability.json"
        status = 0 if report.get("exit") != "engineering_failure" else 3
    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report.get(k) for k in ("exit", "best_arm", "best_arm_accs", "q2_retrain", "cost")}
            | ({"projection": report["projection"]} if args.smoke else {}),
            ensure_ascii=False,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
