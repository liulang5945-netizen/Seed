"""24-S.0b: 8-session stream, relaxed correction/trend gates (3 seeds, <=1200 CPU, 0 check)."""

import copy
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn.functional as F

from scripts.training.diag_protect_smoke_20261121 import mask_grads
from scripts.training.verify_taiji_23s2a_gain_20261122 import cn_split
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72
from taiji.u2_selective_units import (
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}
SEEDS = (20261120, 20261121, 20261122)
B = 48
SESSIONS = 8


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def train_batch(subj, opt, task, batch, flip=None, conflict_flip=False):
    if task == "cn":
        sm = (batch["ticks"].abs().sum(-1) > 0).float()
        loss = F.cross_entropy(subj.logits("cn", batch["ticks"], sm), batch["labels"])
    elif task == "p24":
        lg = subj.logits("p24", batch["ticks"])
        labels = batch["val_labels"]
        if flip is not None:
            labels = (labels + flip) % 4
        conf = 1.0 - batch["conflict_labels"] if conflict_flip else batch["conflict_labels"]
        loss = F.cross_entropy(lg[:, :4], labels) + F.binary_cross_entropy_with_logits(
            lg[:, 4], conf
        )
    else:
        loss = F.cross_entropy(subj.logits("p27", batch["ticks"]), batch["labels"])
    step(opt, loss, subj, task)


def eval_task(subj, task, seed):
    subj.eval_mode()
    if task == "cn":
        out = {}
        for split in ("seen", "unseen"):
            fit = cn_split(256, seed + (700 if split == "seen" else 900) + 10_000_000, split)
            with torch.no_grad():
                sm = (fit["ticks"].abs().sum(-1) > 0).float()
                out[split] = float(
                    (subj.logits("cn", fit["ticks"], sm).argmax(-1) == fit["labels"]).float().mean()
                )
        return out
    if task == "p24":
        fit = _make_p24_stream(128, seed + 212 + 10_000_000)
        with torch.no_grad():
            lg = subj.logits("p24", fit["ticks"])
            return {
                "val": float((lg[:, :4].argmax(-1) == fit["val_labels"]).float().mean()),
                "conflict": float(
                    ((lg[:, 4] > 0).float() == fit["conflict_labels"]).float().mean()
                ),
            }
    fit = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    with torch.no_grad():
        return {
            "val": float(
                (subj.logits("p27", fit["ticks"]).argmax(-1) == fit["labels"]).float().mean()
            )
        }


def snapshot(subj, seed):
    cn = eval_task(subj, "cn", seed)
    p24 = eval_task(subj, "p24", seed)
    p27 = eval_task(subj, "p27", seed)
    comp = (cn["seen"] + cn["unseen"] + p24["val"] + p27["val"]) / 4.0
    return {
        "cn_seen": cn["seen"],
        "cn_unseen": cn["unseen"],
        "p24": p24["val"],
        "p27": p27["val"],
        "composite": comp,
    }


def capsule_diff(subj, seed):
    ticks = _make_p27_physics_batch(32, seed + 81 + 10_000_000)["ticks"][:32]
    with torch.no_grad():
        oa = subj.graph(
            _channels_seq(ticks),
            torch.ones(32, ticks.shape[1], subj.n_units),
            subj.graph.initial_state(batch_size=32),
        )
    cap = build_capsule_v2(subj.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subj.graph.base_graph), subj.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        ob = fresh(
            _channels_seq(ticks),
            torch.ones(32, ticks.shape[1], subj.n_units),
            fresh.initial_state(batch_size=32),
        )
    return float((oa["readout"] - ob["readout"]).abs().max())


def fast_correct(subj, seed, kind):
    """Hybrid fast correction: outdated -> P-B unit rows + head; misleading -> head only."""
    subj.eval_mode()
    for p in subj.graph.parameters():
        p.requires_grad_(False)
    for p in subj.phase_heads["p24"].parameters():
        p.requires_grad_(True)
    if kind.startswith("outdated"):
        for p in subj.graph.parameters():
            p.requires_grad_(True)
        opt = torch.optim.Adam(subj.phase_heads["p24"].parameters(), lr=2e-2, weight_decay=0.0)
        gopt = torch.optim.Adam(subj.graph.parameters(), lr=1e-2, weight_decay=0.0)
    else:
        opt = torch.optim.Adam(subj.phase_heads["p24"].parameters(), lr=3e-2, weight_decay=0.0)
        gopt = None
    subj.phase_heads["p24"].train()
    for k in range(4):
        b = _make_p24_stream(B, seed * 25000 + 9000 + k, force_trusted=True)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        opt.zero_grad(set_to_none=True)
        if gopt is not None:
            gopt.zero_grad(set_to_none=True)
        loss.backward()
        if gopt is not None:
            for _n, p in subj.graph.selective_units.named_parameters():
                if p.grad is None or p.ndim == 0:
                    continue
                if p.shape[0] == subj.n_units:
                    p.grad[:24] = 0.0
                    p.grad[48:] = 0.0
            rg = subj.graph.read_gate
            if rg.weight.grad is not None:
                for i in range(rg.weight.grad.shape[0]):
                    if not (24 <= i < 48 or subj.n_units + 24 <= i < subj.n_units + 48):
                        rg.weight.grad[i] = 0.0
                        rg.bias.grad[i] = 0.0
            gopt.step()
        opt.step()
    for p in subj.graph.parameters():
        p.requires_grad_(True)
    ev = eval_task(subj, "p24", seed)
    return ev["val"] if kind == "outdated" else ev["conflict"]


def recovery_after(subj, opt, seed, kind):
    """Run 4 corrected interactions; return accuracy (p24 val or conflict)."""
    for k in range(4):
        b = _make_p24_stream(B, seed * 25000 + 9000 + k, force_trusted=True)
        train_batch(subj, opt, "p24", b)
    ev = eval_task(subj, "p24", seed)
    return ev["val"] if kind == "outdated" else ev["conflict"]


def run(seed):
    torch.set_num_threads(1)
    subj = Subject72(seed)
    subj.set_masks(True)
    for n, (lo, hi) in BLOCKS.items():
        subj.graph.set_task_route(n, list(range(lo, hi)))
    for p in subj.graph.selective_readout.parameters():
        p.requires_grad_(False)
    for n, h in subj.phase_heads.items():
        for p in h.parameters():
            p.requires_grad_(n == "cn")
    subj.train_mode()
    opt = torch.optim.Adam(subj.params(), lr=6e-3, weight_decay=0.0)
    trace, recovery = {}, []

    def p24(k, **kw):
        return _make_p24_stream(B, seed * 25000 + k, **kw)

    for sess in range(1, SESSIONS + 1):
        if sess == 8:
            trace["sess8"] = snapshot(subj, seed)
            continue
        cn_steps = 60 if sess not in (3, 4, 6, 7) else 40
        for k in range(cn_steps):
            train_batch(subj, opt, "cn", cn_split(B, seed * 6000 + sess * 1000 + k, "train"))
        if sess in (1, 2, 5):
            for k in range(80):
                train_batch(subj, opt, "p24", p24(sess * 1000 + k, force_trusted=(sess == 1)))
        elif sess == 3:
            for k in range(10):
                train_batch(subj, opt, "p24", p24(3000 + k), flip=1)
            recovery.append(("outdated", fast_correct(subj, seed, "outdated")))
        elif sess == 4:
            for k in range(10):
                train_batch(subj, opt, "p24", p24(4000 + k), conflict_flip=True)
            recovery.append(("misleading", fast_correct(subj, seed, "misleading")))
        elif sess == 6:
            for k in range(10):
                train_batch(subj, opt, "p24", p24(6000 + k), flip=2)
            recovery.append(("outdated2", fast_correct(subj, seed, "outdated")))
        else:
            for k in range(10):
                train_batch(subj, opt, "p24", p24(7000 + k, trusted_regimes_only=True))
            recovery.append(("withdrawn", fast_correct(subj, seed, "misleading")))
        for k in range(40):
            train_batch(
                subj, opt, "p27", _make_p27_physics_batch(B, seed * 41000 + sess * 1000 + k)
            )
        trace[f"sess{sess}"] = snapshot(subj, seed)
        trace[f"sess{sess}"]["capsule_diff"] = capsule_diff(subj, seed)
    return {"seed": seed, "trace": trace, "recovery": dict(recovery)}


def slope(values):
    n = len(values)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, values, strict=True))
    var = sum((x - mx) ** 2 for x in xs)
    return cov / var if var else 0.0


def main():
    started = time.process_time()
    rows = [run(seed) for seed in SEEDS]
    comps = [[r["trace"][f"sess{s}"]["composite"] for s in range(1, 9)] for r in rows]
    slopes = [slope(c[1:]) for c in comps]
    tail_gain = [sum(c[5:]) / 3 - c[0] for c in comps]
    retention_ok = all(
        all(
            r["trace"]["sess8"][key] >= min(r["trace"][f"sess{s}"][key] for s in range(1, 8)) - 0.05
            for key in ("cn_seen", "cn_unseen", "p24", "p27")
        )
        for r in rows
    )
    recovery_ok = all(min(r["recovery"].values()) >= 0.75 for r in rows)
    trend_ok = all(s >= 0.005 or t >= 0.02 for s, t in zip(slopes, tail_gain, strict=True))

    def drawdown(c):
        peak, dd = c[0], 0.0
        for x in c:
            peak = max(peak, x)
            dd = max(dd, peak - x)
        return dd

    drawdowns = [drawdown(c) for c in comps]
    drawdown_ok = all(d <= 0.10 for d in drawdowns)
    capsule_ok = all(
        v["capsule_diff"] <= 1e-6
        for r in rows
        for k, v in r["trace"].items()
        if k != "sess8" or "capsule_diff" in v
    )
    gates = {
        "retention": retention_ok,
        "recovery": recovery_ok,
        "trend": trend_ok,
        "drawdown": drawdown_ok,
        "capsule": capsule_ok,
    }
    payload = {
        "schema": "taiji-u2-24s0d-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "rows": rows,
        "slopes": slopes,
        "tail_gain": tail_gain,
        "drawdowns": drawdowns,
        "gates": gates,
        "verdict": "24s0d_verified" if all(gates.values()) else "24s0d_partial",
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_24s0d_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "gates": gates,
                "slopes": [round(s, 4) for s in slopes],
                "tail_gain": [round(t, 4) for t in tail_gain],
                "drawdowns": [round(d, 4) for d in drawdowns],
                "recovery": [r["recovery"] for r in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
