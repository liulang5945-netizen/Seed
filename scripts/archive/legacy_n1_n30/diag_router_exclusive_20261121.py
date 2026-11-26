"""Stage 2b: sequential exclusive top-k routing (3 seeds, <=500 CPU, 0 check).

Each task learns a route over units, but later tasks may only select units not claimed by
earlier tasks (P-A 24, P-B from remaining 48, P-D from remaining 24).  Protection + zero
weight decay as in J-RT.  Goal: CN>=0.95, P-B>=0.85, P-D>=0.80, routes disjoint.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.training.diag_learned_router_20261121 as base  # noqa: E402
from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)

OUT = ROOT / "reports/taiji_u2_router_exclusive_20261121.json"
CAP_CPU = 500.0
SEEDS = (20261120, 20261121, 20261122)
BATCH = 48
K = 24
TASKS = ("cn", "p24", "p27")


class ExclusiveRouter(base.RouterSubject):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.routes: dict[str, list[int]] = {}
        self.prior_claimed: set[int] = set()

    def pick(self, name: str) -> list[int]:
        logits = self.router[self.task_index[name]].clone()
        for idx in self.prior_claimed:
            logits[idx] = -1e9
        return logits.topk(K).indices.tolist()

    def logits_route(
        self, name: str, ticks: torch.Tensor, step_mask=None, picks=None
    ) -> torch.Tensor:
        idx = self.pick(name) if picks is None else picks
        mask = torch.zeros(1, 1, self.n_units)
        mask[0, 0, idx] = 1.0
        out = self.graph(
            _channels_seq(ticks),
            mask.expand(ticks.shape[0], ticks.shape[1], self.n_units),
            self.graph.initial_state(batch_size=ticks.shape[0]),
            step_mask=step_mask,
        )
        z, a = out["next_state"]["z"], out["next_state"]["a"]
        head = self.phase_heads[name]
        return head(torch.cat([z, a], dim=-1))

    def lock(self, name: str) -> None:
        idx = self.pick(name)
        self.routes[name] = sorted(idx)
        self.prior_claimed |= set(idx)


def protect(subject: ExclusiveRouter, name: str, picks: list[int]) -> None:
    selected = set(picks)
    n = subject.n_units
    for _n, p in subject.graph.selective_units.named_parameters():
        if p.grad is None or p.ndim == 0:
            continue
        for idx in range(n):
            if idx not in selected:
                if p.shape[0] == n:
                    p.grad[idx] = 0.0
                if p.ndim == 2 and p.shape[1] == n:
                    p.grad[:, idx] = 0.0
    rg = subject.graph.read_gate
    if rg.weight.grad is not None:
        keep = set()
        for i in selected:
            keep.add(i)
            keep.add(n + i)
        for idx in range(rg.weight.grad.shape[0]):
            if idx not in keep:
                rg.weight.grad[idx] = 0.0
                rg.bias.grad[idx] = 0.0


def step(subject, opt, loss, name, picks):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    protect(subject, name, picks)
    if subject.router.grad is not None:
        for i in range(len(TASKS)):
            if i != subject.task_index[name]:
                subject.router.grad[i] = 0.0
    torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
    opt.step()


def eval_task(subject: ExclusiveRouter, name: str, ticks, labels) -> float:
    subject.eval_mode()
    with torch.no_grad():
        sm = (ticks.abs().sum(-1) > 0).float() if name == "cn" else None
        logits = subject.logits_route(name, ticks, sm, picks=subject.routes[name])
    return float((logits.argmax(-1) == labels).float().mean())


def run(seed: int) -> dict:
    torch.set_num_threads(1)
    subject = ExclusiveRouter(seed)
    for p in subject.graph.selective_readout.parameters():
        p.requires_grad_(False)
    subject.train_mode()
    opt = torch.optim.Adam(subject.params() + [subject.router], lr=6e-3, weight_decay=0.0)
    for k in range(1000):
        b = _build_bidirectional_cn_32dim(
            BATCH, seed * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        picks = subject.pick("cn")
        step(
            subject,
            opt,
            F.cross_entropy(subject.logits_route("cn", b["ticks"], sm, picks), b["labels"]),
            "cn",
            picks,
        )
    subject.lock("cn")
    for k in range(900):
        b = _make_p24_stream(BATCH, seed * 25000 + k)
        picks = subject.pick("p24")
        logits = subject.logits_route("p24", b["ticks"], None, picks)
        loss = F.cross_entropy(logits[:, :4], b["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], b["conflict_labels"])
        step(subject, opt, loss, "p24", picks)
    subject.lock("p24")
    for k in range(800):
        b = _make_p27_physics_batch(BATCH, seed * 41000 + k)
        picks = subject.pick("p27")
        step(
            subject,
            opt,
            F.cross_entropy(subject.logits_route("p27", b["ticks"], None, picks), b["labels"]),
            "p27",
            picks,
        )
    subject.lock("p27")
    fit_cn = _build_bidirectional_cn_32dim(
        256, seed + 700 + 10_000_000, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    fit_p24 = _make_p24_stream(128, seed + 212 + 10_000_000)
    fit_p27 = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    cn = eval_task(subject, "cn", fit_cn["ticks"], fit_cn["labels"])
    subject.eval_mode()
    with torch.no_grad():
        p24 = float(
            (
                subject.logits_route("p24", fit_p24["ticks"], None, subject.routes["p24"])[
                    :, :4
                ].argmax(-1)
                == fit_p24["val_labels"]
            )
            .float()
            .mean()
        )
        p27 = float(
            (
                subject.logits_route("p27", fit_p27["ticks"], None, subject.routes["p27"]).argmax(
                    -1
                )
                == fit_p27["labels"]
            )
            .float()
            .mean()
        )
    overlap = (
        (set(subject.routes["cn"]) & set(subject.routes["p24"]))
        | (set(subject.routes["cn"]) & set(subject.routes["p27"]))
        | (set(subject.routes["p24"]) & set(subject.routes["p27"]))
    )
    return {
        "seed": seed,
        "cn": cn,
        "p24": p24,
        "p27": p27,
        "routes": subject.routes,
        "overlap": sorted(overlap),
    }


def main():
    started = time.process_time()
    rows = [run(seed) for seed in SEEDS]
    gates = {
        "coexist": all(r["cn"] >= 0.95 and r["p24"] >= 0.85 and r["p27"] >= 0.80 for r in rows),
        "disjoint": all(not r["overlap"] for r in rows),
    }
    payload = {
        "schema": "taiji-u2-router-exclusive-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "rows": rows,
        "gates": gates,
        "verdict": (
            "exclusive_router_verified" if all(gates.values()) else "exclusive_router_partial"
        ),
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(OUT), payload)
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "gates": gates,
                "rows": [{k: r[k] for k in ("seed", "cn", "p24", "p27", "overlap")} for r in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["verdict"] == "exclusive_router_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
