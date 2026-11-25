"""N32.3.03: Q2 learnability v3 -- standardised inputs, name-relabel augmentation, MLP gate.

Pre-registered: plans/reference/DESIGN-N32-3-03_20261125.md (D-20261125-521).
Q2 only (one-hop relation, contract v2).  Exploration seeds only; confirm split never read.

  E0  MLP reference (624 -> 256 -> 256 -> 10, ReLU, He init), Adam 1e-3, wd 1e-4, batch 256,
      8,000 steps (~2.05M presentations), seed 20261120.  Engineering gate: val >= 0.90.
  SF  GRU u96 + final-tick read_gate, scene ticks first (encoding as stored), 6,000 steps,
      batch 256 (~1.54M presentations), Adam 3e-3 cosine -> 10 %, clip 1.0, seed 20261120.
  QF  as SF, question ticks first (single variable).
  If max(SF, QF) >= 0.85 the better arm (tie -> SF) is rerun on seeds 20261121 / 20261122.

Inputs are standardised with statistics of the training tensor (GRU: per feature dim;
MLP: per flattened feature).  Training worlds (k < 6000) are augmented with 3 random
relabellings of the six names each (fixed generator); validation worlds (6000-6999) are
never relabelled or trained on.  Batch order uses a sha256-derived generator per (seed, arm).

Usage:
  python scripts/training/n32_3_03_learnability_v3.py --smoke   # projection only
  python scripts/training/n32_3_03_learnability_v3.py           # full card
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

CARD = "N32.3.03"
SEEDS = (20261120, 20261121, 20261122)
SCREEN_SEED = SEEDS[0]
AUG_COPIES = 3
AUG_SEED = 20261125
T = 13
N_SCENE = 3
QF_ORDER = [*range(N_SCENE, T), *range(N_SCENE)]
GATE_MLP = 0.90
GATE_GRU = 0.85
CAP_CPU = 442.0
BATCH = 256
HIDDEN = 96
EVAL_EVERY = 2000
SMOKE_STEPS = 40
TRAIN_PROBE = 4000
ARMS: dict[str, dict[str, Any]] = {
    "E0_mlp": {
        "model": "mlp",
        "order": "scene_first",
        "steps": 8000,
        "lr": 1e-3,
        "wd": 1e-4,
        "cosine": False,
    },
    "SF_gru96": {
        "model": "gru",
        "order": "scene_first",
        "steps": 6000,
        "lr": 3e-3,
        "wd": 0.0,
        "cosine": True,
    },
    "QF_gru96": {
        "model": "gru",
        "order": "question_first",
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


# ----------------------------------------------------------------------------- data


def relabel(scene: dict[str, Any], perm: list[str]) -> dict[str, Any]:
    mapping = dict(zip(NAMES, perm, strict=True))
    bodies = [
        {"name": mapping[b["name"]], "position": list(b["position"])} for b in scene["bodies"]
    ]
    return {**scene, "bodies": bodies}


def encode_q2(scenes: list[dict[str, Any]], copies: int) -> tuple[torch.Tensor, torch.Tensor]:
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
                if q["kind"] != "q2":
                    continue
                if len(v["bodies"]) + len(q["text"]) != T:
                    raise ValueError(f"unexpected Q2 length: {q['text']!r}")
                x = ce.encode(v, q["text"])
                ticks.append(x[:T].to(torch.float16))
                labels.append(ce.ANSWER_INDEX[str(q["answer"])])
    return torch.stack(ticks), torch.tensor(labels, dtype=torch.long)


def digest(x: torch.Tensor, y: torch.Tensor) -> str:
    return hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


def moments(x16: torch.Tensor, flat: bool) -> tuple[torch.Tensor, torch.Tensor]:
    width = T * 48 if flat else 48
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


def build_data() -> dict[str, Any]:
    train_x, train_y = encode_q2(nc2.load_scenes("train"), AUG_COPIES)
    val_x, val_y = encode_q2(nc2.load_scenes("val"), 0)
    probe = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(7))
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "probe_idx": probe[:TRAIN_PROBE],
        "norm_dim": moments(train_x, flat=False),
        "norm_flat": moments(train_x, flat=True),
        "digests": {"train": digest(train_x, train_y), "val": digest(val_x, val_y)},
    }


def prep(x16: torch.Tensor, spec: dict[str, Any], data: dict[str, Any]) -> torch.Tensor:
    x = x16.float()
    if spec["model"] == "mlp":
        mu, sd = data["norm_flat"]
        return (x.reshape(x.shape[0], -1) - mu) / sd
    mu, sd = data["norm_dim"]
    x = (x - mu) / sd
    return x[:, QF_ORDER] if spec["order"] == "question_first" else x


# ---------------------------------------------------------------------------- models


class MLP(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.net = nn.Sequential(
                nn.Linear(T * 48, 256),
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
    return MLP(init) if spec["model"] == "mlp" else GatedGRU(init)


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
    total = spec["steps"] if steps is None else steps
    model = make_model(spec, seed, arm)
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, arm, "batches"))
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["wd"])
    n = data["train_x"].shape[0]
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
        loss = F.cross_entropy(model(prep(data["train_x"][idx], spec, data)), data["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{arm} seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == total:
            acc = evaluate(model, spec, data, data["val_x"], data["val_y"])["acc"]
            curve.append([i + 1, acc])
            model.train()
    probe = data["probe_idx"]
    return {
        "arm": arm,
        "seed": seed,
        "steps": total,
        "presentations": total * BATCH,
        "val": evaluate(model, spec, data, data["val_x"], data["val_y"], detail=True),
        "train_probe_acc": evaluate(
            model, spec, data, data["train_x"][probe], data["train_y"][probe]
        )["acc"],
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
        t0 = time.process_time()
        evaluate(model, spec, data, data["val_x"], data["val_y"], detail=True)
        eval_cost[arm] = time.process_time() - t0
        per_step[arm] = max(row["cpu"] - 3 * eval_cost[arm], 0.0) / SMOKE_STEPS

    def arm_cost(arm: str) -> float:
        steps = ARMS[arm]["steps"]
        return per_step[arm] * steps + (steps / EVAL_EVERY + 2) * eval_cost[arm]

    gru_worst = max(arm_cost("SF_gru96"), arm_cost("QF_gru96"))
    smoke_cpu = time.process_time() - started
    full_all = encode_cpu + arm_cost("E0_mlp") + arm_cost("SF_gru96") + arm_cost("QF_gru96")
    full_all += 2 * gru_worst
    full_no_qf = encode_cpu + arm_cost("E0_mlp") + 3 * arm_cost("SF_gru96")
    if smoke_cpu + full_all <= CAP_CPU:
        planned = "all"
    elif smoke_cpu + full_no_qf <= CAP_CPU:
        planned = "no_qf"
    else:
        planned = ""
    return {
        "per_step_cpu": per_step,
        "eval_cpu": eval_cost,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_full_all": round(full_all, 1),
        "projected_full_no_qf": round(full_no_qf, 1),
        "cap": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    deadline = started + budget
    smoke_path = REPORTS / "n32_3_03_smoke.json"
    planned = json.loads(smoke_path.read_text(encoding="utf-8"))["projection"]["planned"]
    if planned not in ("all", "no_qf"):
        raise SystemExit("smoke projection does not fit the budget; card not run")
    out: dict[str, Any] = {"planned": planned, "runs": []}
    mlp = train_arm("E0_mlp", SCREEN_SEED, data, deadline)
    out["runs"].append(mlp)
    if mlp["val"]["acc"] < GATE_MLP:
        out["exit"] = "engineering_failure"
        out["reason"] = f"MLP reference {mlp['val']['acc']:.3f} < {GATE_MLP}"
        return out
    gru_arms = ["SF_gru96", "QF_gru96"] if planned == "all" else ["SF_gru96"]
    screen = {arm: train_arm(arm, SCREEN_SEED, data, deadline) for arm in gru_arms}
    out["runs"].extend(screen.values())
    best = max(gru_arms, key=lambda a: (screen[a]["val"]["acc"], a == "SF_gru96"))
    out["best_arm"] = best
    if screen[best]["val"]["acc"] < GATE_GRU:
        out["exit"] = "gru_not_learnable"
        return out
    reps = [train_arm(best, s, data, deadline) for s in SEEDS[1:]]
    out["runs"].extend(reps)
    accs = [screen[best]["val"]["acc"], *[r["val"]["acc"] for r in reps]]
    out["best_arm_accs"] = accs
    out["exit"] = "learnable" if min(accs) >= GATE_GRU else "learnable_unstable"
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
        "schema": "taiji-n32-3-03-v1",
        "card": CARD,
        "design": sorted(designs),
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            "train_rows": int(data["train_y"].numel()),
            "val_rows": int(data["val_y"].numel()),
            "aug_copies": AUG_COPIES,
            "digests": data["digests"],
            "val_majority": float(torch.bincount(data["val_y"]).max() / data["val_y"].numel()),
        },
        "gates": {"mlp_engineering": GATE_MLP, "gru": GATE_GRU},
        "encode_cpu": encode_cpu,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0, "exploratory": True},
    }
    status = 0
    if args.smoke:
        report["projection"] = smoke(data, started, encode_cpu)
        out = REPORTS / "n32_3_03_smoke.json"
        status = 0 if report["projection"]["planned"] else 3
    else:
        smoke_cpu = json.loads((REPORTS / "n32_3_03_smoke.json").read_text(encoding="utf-8"))
        budget = CAP_CPU - float(smoke_cpu["projection"]["smoke_cpu"])
        try:
            report.update(full(data, started, budget))
        except CapExceeded as exc:
            report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
        except NonFiniteLoss as exc:
            report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
        out = REPORTS / "n32_3_03_learnability.json"
        status = 0 if report.get("exit") != "engineering_failure" else 3
    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report.get(k) for k in ("exit", "best_arm", "best_arm_accs", "cost")}
            | ({"projection": report["projection"]} if args.smoke else {}),
            ensure_ascii=False,
        )
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
