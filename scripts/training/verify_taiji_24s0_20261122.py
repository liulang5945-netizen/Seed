"""24-S.0: five-session experience stream with four correction events (3 seeds, <=1200 CPU, 0 check)."""

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
SESSIONS = 5


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


def q_of(subj, seed):
    cn = eval_task(subj, "cn", seed)
    p24 = eval_task(subj, "p24", seed)
    p27 = eval_task(subj, "p27", seed)
    return min(cn["seen"], cn["unseen"], p24["val"], p27["val"]), {
        "cn_seen": cn["seen"],
        "cn_unseen": cn["unseen"],
        "p24": p24["val"],
        "p27": p27["val"],
    }


def capsule_diff(subj, cap, seed):
    ticks = _make_p27_physics_batch(32, seed + 81 + 10_000_000)["ticks"][:32]
    with torch.no_grad():
        oa = subj.graph(
            _channels_seq(ticks),
            torch.ones(32, ticks.shape[1], subj.n_units),
            subj.graph.initial_state(batch_size=32),
        )
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
    trace, checks = {}, []

    def cn_batch(k):
        return cn_split(B, seed * 6000 + k, "train")

    def p24_batch(k, **kw):
        return _make_p24_stream(B, seed * 25000 + k, **kw)

    def p27_batch(k):
        return _make_p27_physics_batch(B, seed * 41000 + k)

    for sess in range(1, SESSIONS + 1):
        if sess <= 2:
            for k in range(80):
                train_batch(subj, opt, "cn", cn_batch(sess * 1000 + k))
            for k in range(80):
                train_batch(subj, opt, "p24", p24_batch(sess * 1000 + k, force_trusted=(sess == 1)))
            for k in range(40):
                train_batch(subj, opt, "p27", p27_batch(sess * 1000 + k))
        elif sess == 3:
            for k in range(60):
                train_batch(subj, opt, "cn", cn_batch(3000 + k))
            for k in range(20):
                train_batch(subj, opt, "p24", p24_batch(3000 + k), flip=1)
            for k in range(2):
                train_batch(subj, opt, "p24", p24_batch(3100 + k, force_trusted=True))
            checks.append(("outdated", eval_task(subj, "p24", seed)["val"]))
            for k in range(40):
                train_batch(subj, opt, "p27", p27_batch(3000 + k))
        elif sess == 4:
            for k in range(60):
                train_batch(subj, opt, "cn", cn_batch(4000 + k))
            for k in range(20):
                train_batch(subj, opt, "p24", p24_batch(4000 + k), conflict_flip=True)
            for k in range(2):
                train_batch(subj, opt, "p24", p24_batch(4100 + k, force_trusted=True))
            checks.append(("misleading", eval_task(subj, "p24", seed)["conflict"]))
            for k in range(40):
                train_batch(subj, opt, "p27", p27_batch(4000 + k))
        q, parts = q_of(subj, seed)
        cap = build_capsule_v2(subj.graph)
        trace[f"sess{sess}"] = {"q": q, **parts, "capsule_diff": capsule_diff(subj, cap, seed)}

    s1q = trace["sess1"]["q"]
    gains = {k: trace[k]["q"] - s1q for k in ("sess2", "sess3", "sess4")}
    return {
        "seed": seed,
        "trace": trace,
        "gains": gains,
        "retention": trace["sess5"]["q"] - s1q,
        "recovery": dict(checks),
    }


def main():
    started = time.process_time()
    rows = [run(seed) for seed in SEEDS]
    retention_ok = all(r["trace"]["sess5"]["q"] >= r["trace"]["sess1"]["q"] - 0.02 for r in rows)
    recovery_ok = all(min(r["recovery"].values()) >= 0.85 for r in rows)
    sustained = [sum(1 for k in ("sess2", "sess3", "sess4") if r["gains"][k] >= 0.05) for r in rows]
    gain_ok = all(s >= 2 for s in sustained)
    capsule_ok = all(v["capsule_diff"] <= 1e-6 for r in rows for v in r["trace"].values())
    gates = {
        "retention": retention_ok,
        "recovery": recovery_ok,
        "gain_sustained": gain_ok,
        "capsule": capsule_ok,
    }
    payload = {
        "schema": "taiji-u2-24s0-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "rows": rows,
        "gates": gates,
        "verdict": "24s0_verified" if all(gates.values()) else "24s0_partial",
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_24s0_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "gates": gates,
                "retention": [round(r["retention"], 4) for r in rows],
                "gains": [{k: round(v, 4) for k, v in r["gains"].items()} for r in rows],
                "recovery": [r["recovery"] for r in rows],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
