"""Stage 2 smoke: learned top-k task routing (1 seed, <=200 CPU, 0 check).

Each task owns a learned logit row over 72 units; forward uses a straight-through hard top-k
(k=24) gate; protection masks gradients/weight update to units outside the selected set, and
only the current task's router row trains.  Goal: CN>=0.95, P-B>=0.85, P-D>=0.80, capsule
restore diff 0. Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72  # noqa: E402

OUT = ROOT / "reports/taiji_u2_learned_router_smoke_20261121.json"
CAP_CPU = 200.0
SEED = 20261120
BATCH = 48
K = 24
TASKS = ("cn", "p24", "p27")


class RouterSubject(Subject72):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.router = nn.Parameter(torch.zeros(len(TASKS), self.n_units))
        self.task_index = {name: i for i, name in enumerate(TASKS)}
        self.selected: dict[str, list[int]] = {}

    def route(self, name: str, soft: bool = True) -> torch.Tensor:
        logits = self.router[self.task_index[name]]
        hard = torch.zeros(self.n_units)
        hard[logits.topk(K).indices] = 1.0
        if not soft:
            return hard
        return hard + torch.softmax(logits, dim=0) - torch.softmax(logits, dim=0).detach()

    def logits_route(self, name: str, ticks: torch.Tensor, step_mask=None) -> torch.Tensor:
        mask = self.route(name).view(1, 1, self.n_units)
        out = self.graph(
            __import__(
                "scripts.training.verify_taiji_algorithm_holistic_20261121",
                fromlist=["_channels_seq"],
            )._channels_seq(ticks),
            mask.expand(ticks.shape[0], ticks.shape[1], self.n_units),
            self.graph.initial_state(batch_size=ticks.shape[0]),
            step_mask=step_mask,
        )
        z, a = out["next_state"]["z"], out["next_state"]["a"]
        head = self.phase_heads[name]
        return head(torch.cat([z, a], dim=-1))


def protect(subject: RouterSubject, name: str) -> None:
    selected = set(subject.route(name, soft=False).nonzero().flatten().tolist())
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
    if subject.router.grad is not None:
        for i in range(len(TASKS)):
            if i != subject.task_index[name]:
                subject.router.grad[i] = 0.0


def step(subject, opt, loss, name):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    protect(subject, name)
    torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
    opt.step()


def eval_all(subject: RouterSubject) -> dict:
    fit_cn = _build_bidirectional_cn_32dim(
        256, SEED + 700 + 10_000_000, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    fit_p24 = _make_p24_stream(128, SEED + 212 + 10_000_000)
    fit_p27 = _make_p27_physics_batch(128, SEED + 81 + 10_000_000)
    subject.eval_mode()
    with torch.no_grad():
        sm = (fit_cn["ticks"].abs().sum(-1) > 0).float()
        cn = float(
            (subject.logits_route("cn", fit_cn["ticks"], sm).argmax(-1) == fit_cn["labels"])
            .float()
            .mean()
        )
        p24 = float(
            (
                subject.logits_route("p24", fit_p24["ticks"])[:, :4].argmax(-1)
                == fit_p24["val_labels"]
            )
            .float()
            .mean()
        )
        p27 = float(
            (subject.logits_route("p27", fit_p27["ticks"]).argmax(-1) == fit_p27["labels"])
            .float()
            .mean()
        )
    return {"cn": cn, "p24": p24, "p27": p27}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    subject = RouterSubject(SEED)
    for p in subject.graph.selective_readout.parameters():
        p.requires_grad_(False)
    subject.train_mode()
    for name in TASKS:
        for p in subject.phase_heads[name].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subject.params() + [subject.router], lr=6e-3, weight_decay=0.0)
    for k in range(1000):
        b = _build_bidirectional_cn_32dim(
            BATCH, SEED * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(
            subject,
            opt,
            F.cross_entropy(subject.logits_route("cn", b["ticks"], sm), b["labels"]),
            "cn",
        )
    for k in range(900):
        b = _make_p24_stream(BATCH, SEED * 25000 + k)
        logits = subject.logits_route("p24", b["ticks"])
        loss = F.cross_entropy(logits[:, :4], b["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], b["conflict_labels"])
        step(subject, opt, loss, "p24")
    for k in range(800):
        b = _make_p27_physics_batch(BATCH, SEED * 41000 + k)
        step(
            subject,
            opt,
            F.cross_entropy(subject.logits_route("p27", b["ticks"]), b["labels"]),
            "p27",
        )
    scores = eval_all(subject)
    routes = {
        name: sorted(subject.route(name, soft=False).nonzero().flatten().tolist()) for name in TASKS
    }
    passed = scores["cn"] >= 0.95 and scores["p24"] >= 0.85 and scores["p27"] >= 0.80
    payload = {
        "schema": "taiji-u2-learned-router-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "scores": scores,
        "routes": routes,
        "route_sizes": {k: len(v) for k, v in routes.items()},
        "verdict": "learned_router_smoke_passed" if passed else "learned_router_smoke_failed",
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
