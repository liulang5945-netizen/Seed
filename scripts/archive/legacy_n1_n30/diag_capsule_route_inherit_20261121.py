"""Capsule inheritance of task routes + protected subject (1 seed, <=200 CPU, 0 check).

Trains one protected Subject72 (three tasks, 24 units each), stores routes in capsule_v2,
then restores into a fresh graph and verifies all three tasks reproduce exactly.
Run under standing authorization; zero check/final.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.diag_protect_smoke_20261121 import mask_grads  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead  # noqa: E402
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72  # noqa: E402

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_capsule_route_inherit_20261121.json"
CAP_CPU = 200.0
SEED = 20261120
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 128


def step(opt, loss, subject, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subject, keep)
    torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
    opt.step()


def train(seed: int) -> Subject72:
    torch.set_num_threads(1)
    subject = Subject72(seed)
    subject.set_masks(True)
    for name, (lo, hi) in BLOCKS.items():
        subject.graph.set_task_route(name, list(range(lo, hi)))
    for p in subject.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for name, head in subject.phase_heads.items():
        for p in head.parameters():
            p.requires_grad_(name == "cn")
    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=0.0)
    for k in range(1000):
        b = _build_bidirectional_cn_32dim(
            BATCH, seed * 6000 + k, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(opt, F.cross_entropy(subject.logits("cn", b["ticks"], sm), b["labels"]), subject, "cn")
    for p in subject.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for name in ("p24", "p27"):
        for p in subject.phase_heads[name].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=0.0)
    for k in range(900):
        b = _make_p24_stream(BATCH, seed * 25000 + k)
        logits = subject.logits("p24", b["ticks"])
        loss = F.cross_entropy(logits[:, :4], b["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], b["conflict_labels"])
        step(opt, loss, subject, "p24")
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=0.0)
    for k in range(800):
        b = _make_p27_physics_batch(BATCH, seed * 41000 + k)
        step(opt, F.cross_entropy(subject.logits("p27", b["ticks"]), b["labels"]), subject, "p27")
    return subject


def scores(subject: Subject72, seed: int) -> dict[str, float]:
    subject.eval_mode()
    fit_cn = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    fit_p24 = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    fit_p27 = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)
    with torch.no_grad():
        sm = (fit_cn["ticks"].abs().sum(-1) > 0).float()
        cn = float(
            (subject.logits("cn", fit_cn["ticks"], sm).argmax(-1) == fit_cn["labels"])
            .float()
            .mean()
        )
        p24 = float(
            (subject.logits("p24", fit_p24["ticks"])[:, :4].argmax(-1) == fit_p24["val_labels"])
            .float()
            .mean()
        )
        p27 = float(
            (subject.logits("p27", fit_p27["ticks"]).argmax(-1) == fit_p27["labels"]).float().mean()
        )
    return {"cn": cn, "p24": p24, "p27": p27}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    subject = train(SEED)
    before = scores(subject, SEED)
    cap = build_capsule_v2(subject.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
        head_path = Path(tmp) / "heads.pt"
        torch.save({k: v.state_dict() for k, v in subject.phase_heads.items()}, head_path)
        heads = torch.load(head_path, weights_only=True)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subject.graph.base_graph), subject.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    restored = Subject72(SEED)
    restored.graph = fresh
    restored.n_units = fresh.selective_units.n_units
    restored.masks = {name: fresh.task_route_mask(name) for name in BLOCKS}
    for name, state in heads.items():
        dim = state["net.0.weight"].shape[1]
        width = state["net.0.weight"].shape[0]
        last = (
            state["net.3.weight"].shape[0]
            if "net.3.weight" in state
            else state["net.2.weight"].shape[0]
        )
        head = PhaseHead(dim, last, width)
        head.load_state_dict(state)
        restored.phase_heads[name] = head
    after = scores(restored, SEED)
    diff = max(abs(before[k] - after[k]) for k in before)
    payload = {
        "schema": "taiji-u2-capsule-route-inherit-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "before": before,
        "after": after,
        "max_abs_diff": diff,
        "routes": fresh.export_task_routes(),
        "verdict": (
            "capsule_route_inheritance_verified"
            if diff <= 1e-6
            else "capsule_route_inheritance_partial"
        ),
        "disclosure": "1 seed; heads serialized separately; zero check/final",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if diff <= 1e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
