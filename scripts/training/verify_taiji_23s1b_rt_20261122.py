import copy
import hashlib
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
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq
from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    BI_CN_CHAR_INDEX,
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_PROJ32,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    CN_COLOURS,
    CN_MAX_LEN,
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


CN_CLASSES = len(CN_COLOURS)
HOLDOUT = {(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4), (6, 7), (7, 6)}


def cn_split(count: int, seed: int, split: str) -> dict[str, torch.Tensor]:
    """Unseen-pair holdout: train/seen exclude HOLDOUT ordered pairs, unseen uses only them."""
    first = BI_CN_TRAIN_FIRST if split == "train" else BI_CN_FIT_FIRST
    second = BI_CN_TRAIN_SECOND if split == "train" else BI_CN_FIT_SECOND
    all_tpls = [*[(t, True) for t in first], *[(t, False) for t in second]]
    rows, labels = [], []
    idx = 0
    attempts = 0
    while len(rows) < count and attempts < count * 200:
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        pair = (c1, c2)
        ok = pair in HOLDOUT if split == "unseen" else pair not in HOLDOUT
        idx += 1
        attempts += 1
        if not ok:
            continue
        tpl, pick_first = all_tpls[len(rows) % len(all_tpls)]
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            ci = BI_CN_CHAR_INDEX.get(ch)
            if ci is not None:
                seq[ti, :32] = BI_CN_PROJ32[ci]
        rows.append(seq)
        labels.append(c1 if pick_first else c2)
    return {"ticks": torch.stack(rows), "labels": torch.tensor(labels, dtype=torch.long)}


def step(opt, loss, subj, keep):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subj, keep)
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    opt.step()


def eval_cn(subj, seed):
    subj.eval_mode()
    fit = cn_split(256, seed + 700 + 10_000_000, "seen")
    with torch.no_grad():
        sm = (fit["ticks"].abs().sum(-1) > 0).float()
        return float(
            (subj.logits("cn", fit["ticks"], sm).argmax(-1) == fit["labels"]).float().mean()
        )


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
    for k in range(1000):
        b = cn_split(B, seed * 6000 + k, "train")
        sm = (b["ticks"].abs().sum(-1) > 0).float()
        step(opt, F.cross_entropy(subj.logits("cn", b["ticks"], sm), b["labels"]), subj, "cn")
    cn_peak = eval_cn(subj, seed)
    for p in subj.graph.cross_channel_mix.parameters():
        p.requires_grad_(False)
    for n in ("p24", "p27"):
        for p in subj.phase_heads[n].parameters():
            p.requires_grad_(True)
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(900):
        b = _make_p24_stream(B, seed * 25000 + k)
        lg = subj.logits("p24", b["ticks"])
        loss = F.cross_entropy(lg[:, :4], b["val_labels"]) + F.binary_cross_entropy_with_logits(
            lg[:, 4], b["conflict_labels"]
        )
        step(opt, loss, subj, "p24")
    fit24 = _make_p24_stream(128, seed + 212 + 10_000_000)
    subj.eval_mode()
    with torch.no_grad():
        lg = subj.logits("p24", fit24["ticks"])
        p24_peak = float((lg[:, :4].argmax(-1) == fit24["val_labels"]).float().mean())
    opt = torch.optim.Adam(subj.params(), lr=7e-3, weight_decay=0.0)
    for k in range(1300):
        b = _make_p27_physics_batch(B, seed * 41000 + k)
        step(opt, F.cross_entropy(subj.logits("p27", b["ticks"]), b["labels"]), subj, "p27")
    subj.eval_mode()
    fit27 = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    with torch.no_grad():
        lg = subj.logits("p27", fit27["ticks"])
        p27 = float((lg.argmax(-1) == fit27["labels"]).float().mean())
        no = fit27["ticks"].clone()
        no[:, :6, 8:11] = 0.0
        ab = float((subj.logits("p27", no).argmax(-1) == fit27["labels"]).float().mean())
    cn = eval_cn(subj, seed)
    unseen = cn_split(256, seed + 900 + 10_000_000, "unseen")
    subj.eval_mode()
    with torch.no_grad():
        smu = (unseen["ticks"].abs().sum(-1) > 0).float()
        cn_unseen = float(
            (subj.logits("cn", unseen["ticks"], smu).argmax(-1) == unseen["labels"]).float().mean()
        )
    subj.eval_mode()
    with torch.no_grad():
        lg = subj.logits("p24", fit24["ticks"])
        p24 = float((lg[:, :4].argmax(-1) == fit24["val_labels"]).float().mean())
        conf = float(((lg[:, 4] > 0).float() == fit24["conflict_labels"]).float().mean())
    damage = max(0.0, cn_peak - cn, p24_peak - p24)
    ticks = fit27["ticks"][:32]
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
        heads = {k: v.state_dict() for k, v in subj.phase_heads.items()}
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subj.graph.base_graph), subj.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        ob = fresh(
            _channels_seq(ticks),
            torch.ones(32, ticks.shape[1], fresh.selective_units.n_units),
            fresh.initial_state(batch_size=32),
        )
        capdiff = float((oa["readout"] - ob["readout"]).abs().max())
    restored = Subject72(seed)
    restored.graph = fresh
    restored.n_units = fresh.selective_units.n_units
    restored.masks = {n: fresh.task_route_mask(n) for n in BLOCKS}
    from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead

    for n, s in heads.items():
        w = s["net.0.weight"]
        last = s["net.3.weight"].shape[0] if "net.3.weight" in s else s["net.2.weight"].shape[0]
        h = PhaseHead(w.shape[1], last, w.shape[0])
        h.load_state_dict(s)
        restored.phase_heads[n] = h
    after = {}
    restored.eval_mode()
    with torch.no_grad():
        sm = (fit24["ticks"].abs().sum(-1) > 0).float()
        fit_cn = cn_split(256, seed + 700 + 10_000_000, "seen")
        after["cn"] = float(
            (
                restored.logits(
                    "cn", fit_cn["ticks"], (fit_cn["ticks"].abs().sum(-1) > 0).float()
                ).argmax(-1)
                == fit_cn["labels"]
            )
            .float()
            .mean()
        )
        after["p24"] = float(
            (restored.logits("p24", fit24["ticks"])[:, :4].argmax(-1) == fit24["val_labels"])
            .float()
            .mean()
        )
        after["p27"] = float(
            (restored.logits("p27", fit27["ticks"]).argmax(-1) == fit27["labels"]).float().mean()
        )
    restore_diff = max(abs(after["cn"] - cn), abs(after["p24"] - p24), abs(after["p27"] - p27))
    return {
        "seed": seed,
        "cn": cn,
        "unseen": cn_unseen,
        "p24": p24,
        "p27": p27,
        "conflict": conf,
        "ablation": p27 - ab,
        "damage": damage,
        "capdiff": capdiff,
        "restore_diff": restore_diff,
        "routes": fresh.export_task_routes(),
    }


started = time.process_time()
rows = [run(s) for s in SEEDS]
g = {
    "cn_seen": min(r["cn"] for r in rows) >= 0.95,
    "cn_unseen": min(r["unseen"] for r in rows) >= 0.85,
    "p24": min(r["p24"] for r in rows) >= 0.85,
    "conflict": min(r["conflict"] for r in rows) >= 0.95,
    "p27": min(r["p27"] for r in rows) >= 0.80,
    "old_damage": max(r["damage"] for r in rows) <= 0.02,
    "capsule": max(r["capdiff"] for r in rows) <= 1e-6
    and max(r["restore_diff"] for r in rows) <= 1e-6,
}
verdict = "23s1b_rt_verified" if all(g.values()) else "23s1b_rt_partial"
crit_path = Path(r"E:\Seed\plans/manifests/taiji_u2_23s1b_rt_criteria_20261122.json")
crit = json.loads(crit_path.read_text(encoding="utf-8"))
body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
assert hashlib.sha256(canon).hexdigest() == crit["criteria_sha256"], "criteria_sha256 mismatch"
assert crit["budget_cpu"] == 900.0 and crit["seeds"] == list(SEEDS)
p = {
    "schema": "taiji-u2-s1rt-formal-v1",
    "criteria_sha256": crit["criteria_sha256"],
    "criteria": crit,
    "created_utc": datetime.now(UTC).isoformat(),
    "seeds": list(SEEDS),
    "rows": rows,
    "gates": g,
    "verdict": "23s1b_rt_verified" if all(g.values()) else "23s1b_rt_partial",
    "run_conditions": {
        "threads": torch.get_num_threads(),
        "OMP_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
    },
    "discipline": {"check_reads": 0, "final_reads": 0},
    "cost": {"cpu": round(time.process_time() - started, 2)},
}
Path(r"E:\Seed\reports\taiji_u2_23s1b_rt_20261122.json").write_text(
    json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(
    json.dumps(
        {
            "verdict": p["verdict"],
            "gates": g,
            "rows": [
                {
                    k: round(r[k], 4)
                    for k in (
                        "seed",
                        "cn",
                        "p24",
                        "p27",
                        "conflict",
                        "ablation",
                        "damage",
                        "capdiff",
                        "restore_diff",
                    )
                }
                for r in rows
            ],
            "cpu": p["cost"]["cpu"],
        },
        ensure_ascii=False,
    )
)
