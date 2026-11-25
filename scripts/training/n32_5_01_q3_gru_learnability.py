"""N32.5.01: Lightweight Q3 GRU learnability check (Route X, <= 250 CPU).

Pre-registered: plans/reference/DESIGN-N32-5-01_20261125.md (D-20261125-534).
Retires the falsified flat-MLP >= 0.90 engineering gate (N32.4.04) and replaces it with
bit-exact SHA256 digest verification of a fast cached Q3 encoder, then tests G3 GRU u96
(QF tick order, standardised inputs, name-relabel augmentation x3) directly on Q3.

Usage:
  python scripts/training/n32_5_01_q3_gru_learnability.py --smoke
  python scripts/training/n32_5_01_q3_gru_learnability.py
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

CARD = "N32.5.01"
SEEDS = (20261120, 20261121, 20261122)
SCREEN_SEED = SEEDS[0]
AUG_COPIES = 3
AUG_SEED = 20261125
N_SCENE = 3
Q3_LEN = 20
STEPS = 6000
BATCH = 256
HIDDEN = 96
LR = 3e-3
EVAL_EVERY = 2000
SMOKE_STEPS = 100
TRAIN_PROBE = 4000
GATE_GRU = 0.85
CAP_CPU = 250.0
PROJECTION_MARGIN = 1.10

EXPECTED_DIGESTS = {
    "train": "df7b1ad98a6c7c908ebdb9ddba7e4e407c839edaba01371fccf2bb1a27a034e9",
    "val": "7f3f3c580ae1f8577b1046c304513d99fa4ab150e7fb0965f1ac315c665d7928",
}

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_5_01_smoke.json"
FULL_REPORT = REPORTS / "n32_5_01_learnability.json"


class CapExceeded(RuntimeError):
    pass


class NonFiniteLoss(RuntimeError):
    pass


def int_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**62)


def qf_order(length: int = Q3_LEN) -> list[int]:
    return [*range(N_SCENE, length), *range(N_SCENE)]


# ---------------------------------------------------------------------- fast bit-exact data
def relabel(scene: dict[str, Any], perm: list[str]) -> dict[str, Any]:
    mapping = dict(zip(NAMES, perm, strict=True))
    bodies = [
        {"name": mapping[b["name"]], "position": list(b["position"])} for b in scene["bodies"]
    ]
    return {**scene, "bodies": bodies}


def _question_ticks_f16(text: str, char_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    q_len = len(text)
    qt = torch.zeros(q_len, ce.INPUT_DIM, dtype=torch.float16)
    for k, ch in enumerate(text):
        emb = char_cache.get(ch)
        if emb is None:
            emb = ce.char_embedding(ch).to(torch.float16)
            char_cache[ch] = emb
        qt[k, ce.EMBED_LO : ce.EMBED_HI] = emb
        qt[k, ce.QUESTION_FLAG] = 1.0
    return qt


def encode_q3_fast(scenes: list[dict[str, Any]], copies: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-identical fast replacement for N32.4's encode_kind(scenes, 'q3', copies)."""
    n_pairs = len(nc2.PAIRS)
    variants_per_scene = 1 + copies
    total = len(scenes) * variants_per_scene * n_pairs
    ticks = torch.empty((total, Q3_LEN, ce.INPUT_DIM), dtype=torch.float16)
    labels: list[int] = []

    char_cache: dict[str, torch.Tensor] = {}
    name_embed: dict[str, torch.Tensor] = {}
    name_idx: dict[str, int] = {}
    for idx, name in enumerate(NAMES):
        ch = name[0]
        emb = char_cache.get(ch)
        if emb is None:
            emb = ce.char_embedding(ch).to(torch.float16)
            char_cache[ch] = emb
        name_embed[name] = emb
        name_idx[name] = idx

    q_cache: dict[str, torch.Tensor] = {}
    gen = torch.Generator().manual_seed(AUG_SEED)
    row = 0

    for scene in scenes:
        bodies = scene["bodies"]
        n_bodies = len(bodies)
        order = ce.scene_order(int(scene["seed"]), n_bodies)
        pos_f16 = [
            torch.tensor([float(v) for v in list(bodies[i]["position"])[:3]], dtype=torch.float16)
            for i in order
        ]
        # Verify finite coordinates matching ce.encode
        for t_vec in pos_f16:
            if not torch.isfinite(t_vec).all():
                raise ValueError("non-finite position")

        variants = [scene]
        for _ in range(copies):
            perm = [NAMES[i] for i in torch.randperm(len(NAMES), generator=gen).tolist()]
            variants.append(relabel(scene, perm))

        for v in variants:
            vbodies = v["bodies"]
            scene_block = torch.zeros((n_bodies, ce.INPUT_DIM), dtype=torch.float16)
            for t, i in enumerate(order):
                bname = str(vbodies[i]["name"])
                scene_block[t, name_idx[bname]] = 1.0
                pvec = pos_f16[t]
                scene_block[t, 6 : 6 + pvec.numel()] = pvec
                scene_block[t, ce.SCENE_FLAG] = 1.0
                scene_block[t, ce.EMBED_LO : ce.EMBED_HI] = name_embed[bname]

            for q in nc2.questions(v):
                if q["kind"] != "q3":
                    continue
                qtext = str(q["text"])
                if n_bodies + len(qtext) != Q3_LEN:
                    raise ValueError(f"unexpected q3 length: {n_bodies + len(qtext)}")
                qt = q_cache.get(qtext)
                if qt is None:
                    qt = _question_ticks_f16(qtext, char_cache)
                    q_cache[qtext] = qt
                ticks[row, :n_bodies].copy_(scene_block)
                ticks[row, n_bodies:].copy_(qt)
                labels.append(ce.ANSWER_INDEX[str(q["answer"])])
                row += 1

    if row != total:
        raise ValueError(f"unexpected row count {row} != {total}")
    return ticks, torch.tensor(labels, dtype=torch.long)


def moments_dim(x16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.zeros(ce.INPUT_DIM, dtype=torch.float64)
    s2 = torch.zeros(ce.INPUT_DIM, dtype=torch.float64)
    n = 0
    for i in range(0, x16.shape[0], 4096):
        c = x16[i : i + 4096].double().reshape(-1, ce.INPUT_DIM)
        s += c.sum(0)
        s2 += (c * c).sum(0)
        n += c.shape[0]
    mu = s / n
    sd = (s2 / n - mu * mu).clamp(min=0.0).sqrt()
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return mu.float(), sd.float()


def tensor_digest(x: torch.Tensor, y: torch.Tensor) -> str:
    return hashlib.sha256(x.numpy().tobytes() + y.numpy().tobytes()).hexdigest()


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
                f"N32.5.01 fast encoder {split} digest mismatch: " f"{digests[split]} != {expected}"
            )
    mu, sd = moments_dim(train_x_raw)
    order = qf_order(Q3_LEN)
    train_x = train_x_raw[:, order].contiguous()
    val_x = val_x_raw[:, order].contiguous()
    probe = torch.randperm(train_x.shape[0], generator=torch.Generator().manual_seed(7))
    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
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


# --------------------------------------------------------------------------- model
class GatedGRU(nn.Module):
    def __init__(self, seed: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gru = nn.GRU(ce.INPUT_DIM, hidden, batch_first=True)
            self.read_gate = nn.Linear(ce.INPUT_DIM, hidden)
            nn.init.constant_(self.read_gate.bias, 1.0)
            self.head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, len(ce.ANSWERS))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.gru(x)
        gate = torch.sigmoid(self.read_gate(x[:, -1]))
        out: torch.Tensor = self.head(seq[:, -1] * gate)
        return out


# ----------------------------------------------------------------------- train / eval
def evaluate(
    model: nn.Module,
    data: dict[str, Any],
    x16: torch.Tensor,
    y: torch.Tensor,
    detail: bool = False,
) -> dict[str, Any]:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x16.shape[0], 1024):
            preds.append(model(prep(x16[i : i + 1024], data)).argmax(-1))
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
    model = GatedGRU(int_seed(CARD, seed, "G3_gru96_q3", "init"))
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, "G3_gru96_q3", "batches"))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    n = data["train_x"].shape[0]
    curve: list[list[float]] = []

    t0 = time.process_time()
    eval_cpu_total = 0.0
    model.train()
    for i in range(steps):
        if time.process_time() > deadline:
            raise CapExceeded(f"G3_gru96_q3 seed {seed} step {i}")
        lr = LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * i / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        loss = F.cross_entropy(model(prep(data["train_x"][idx], data)), data["train_y"][idx])
        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"G3_gru96_q3 seed {seed} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (i + 1) % EVAL_EVERY == 0 or i + 1 == steps:
            e0 = time.process_time()
            acc = evaluate(model, data, data["val_x"], data["val_y"])["acc"]
            eval_cpu_total += time.process_time() - e0
            curve.append([i + 1, acc])
            model.train()

    e1 = time.process_time()
    val_detail = evaluate(model, data, data["val_x"], data["val_y"], detail=True)
    probe = data["probe_idx"]
    probe_res = evaluate(model, data, data["train_x"][probe], data["train_y"][probe])
    eval_cpu_total += time.process_time() - e1

    total_cpu = time.process_time() - t0
    train_loop_cpu = max(total_cpu - eval_cpu_total, 0.0)
    return {
        "arm": "G3_gru96_q3",
        "kind": "q3",
        "seed": seed,
        "steps": steps,
        "presentations": steps * BATCH,
        "val": val_detail,
        "train_probe_acc": probe_res["acc"],
        "curve": curve,
        "n_params": sum(p.numel() for p in model.parameters()),
        "cpu": round(total_cpu, 2),
        "train_loop_cpu": round(train_loop_cpu, 3),
        "eval_cpu_total": round(eval_cpu_total, 3),
    }


# --------------------------------------------------------------------------- driver
def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    row = train_seed(SCREEN_SEED, data, deadline, steps=SMOKE_STEPS)
    model = GatedGRU(int_seed(CARD, SCREEN_SEED, "G3_gru96_q3", "init"))
    e0 = time.process_time()
    evaluate(model, data, data["val_x"], data["val_y"], detail=True)
    eval_cpu = time.process_time() - e0
    per_step_cpu = row["train_loop_cpu"] / SMOKE_STEPS

    smoke_cpu = time.process_time() - started
    n_evals = STEPS // EVAL_EVERY + 2
    per_seed = (per_step_cpu * STEPS + n_evals * eval_cpu) * PROJECTION_MARGIN
    proj_1 = round(encode_cpu + per_seed, 2)
    proj_3 = round(encode_cpu + 3.0 * per_seed, 2)

    if smoke_cpu + proj_3 <= CAP_CPU:
        planned = "all_3seeds"
    elif smoke_cpu + proj_1 <= CAP_CPU:
        planned = "screen_only"
    else:
        planned = "budget_insufficient"

    return {
        "encode_cpu": encode_cpu,
        "per_step_cpu": per_step_cpu,
        "eval_cpu": eval_cpu,
        "smoke_cpu": round(smoke_cpu, 2),
        "projected_1seed": proj_1,
        "projected_3seeds": proj_3,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
    }


def full(data: dict[str, Any], started: float, budget: float) -> dict[str, Any]:
    proj = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))["projection"]
    planned = str(proj["planned"])
    if planned == "budget_insufficient" or planned not in ("all_3seeds", "screen_only"):
        return {
            "planned": planned,
            "runs": [],
            "exit": "budget_insufficient",
            "reason": f"smoke projection does not fit CAP_CPU={CAP_CPU}",
        }

    deadline = started + budget
    out: dict[str, Any] = {"planned": planned, "runs": []}
    screen = train_seed(SCREEN_SEED, data, deadline, steps=STEPS)
    out["runs"].append(screen)
    out["screen_acc"] = screen["val"]["acc"]
    if screen["val"]["acc"] < GATE_GRU:
        out["exit"] = "q3_not_learnable"
        out["accs"] = [screen["val"]["acc"]]
        return out

    if planned == "screen_only":
        out["exit"] = "screen_passed_needs_rep_budget"
        out["accs"] = [screen["val"]["acc"]]
        return out

    reps = [train_seed(s, data, deadline, steps=STEPS) for s in SEEDS[1:]]
    out["runs"].extend(reps)
    accs = [screen["val"]["acc"], *[r["val"]["acc"] for r in reps]]
    out["accs"] = accs
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
        "schema": "taiji-n32-5-01-v1",
        "card": CARD,
        "design": sorted(designs),
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
        "gates": {"gru": GATE_GRU},
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
    summary_keys = ("exit", "planned", "accs", "cost")
    summary = {k: report.get(k) for k in summary_keys}
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
