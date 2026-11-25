"""N32.6.01: R1 Pilot — Task-agnostic shared ModalityEdgeHop (S) vs external RT columns (<= 520 CPU).

Pre-registered: plans/reference/DESIGN-N32-6-01_20261125.md (D-20261125-544).
Compares one task-agnostic shared ModalityEdgeHop (S, 121,612 params, replay of old kinds)
against three disjoint external-routed ModalityEdgeHop columns (RT, 3 x 121,612 params)
on the identical-format taiji-23-composition-v2 Q1 -> Q2 -> Q3 curriculum.

Usage:
  python scripts/training/n32_6_01_r1_edge_hop.py --smoke
  python scripts/training/n32_6_01_r1_edge_hop.py
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
    AUG_SEED,
    BATCH,
    EXPECTED_DIGESTS,
    Q3_LEN,
    SCREEN_SEED,
    CapExceeded,
    NonFiniteLoss,
    int_seed,
    relabel,
    tensor_digest,
)
from taiji import composition_encoding as ce  # noqa: E402
from taiji import native_composition_v2 as nc2  # noqa: E402
from taiji.modality_edge_hop import (  # noqa: E402
    ARCH_VERSION,
    ModalityEdgeHop,
    extract_modality_masks,
)
from taiji.native_composition import DIRECTIONS, NAMES  # noqa: E402

CARD = "N32.6.01"
KINDS = ("q1", "q2", "q3")
STAGE_STEPS = {"q1": 1500, "q2": 3000, "q3": 6000}
Q3_CKPTS = (1000, 2000, 4000, 6000)
REPLAY_BATCH = 64
REPLAY_WEIGHT = 0.5
D_MODEL = 64
D_EDGE = 64
N_EDGE_LAYERS = 2
N_HOPS = 2
LR = 3e-3
WD = 1e-4
WARMUP_STEPS = 200
GATE_MIN = 0.85
GATE_POS = 0.90
G2_MIN = 0.05
G3_MIN = -0.05
CEILING_ACC = 0.95
SMOKE_STEPS = 40
CAP_CPU = 580.0
PROJECTION_MARGIN = 1.10

REPORTS = ROOT / "reports"
SMOKE_REPORT = REPORTS / "n32_6_01_smoke.json"
FULL_REPORT = REPORTS / "n32_6_01_r1_pilot.json"


# ---------------------------------------------------------------------- fast multi-kind data
def _question_ticks_f16(text: str, char_cache: dict[str, torch.Tensor]) -> torch.Tensor:
    qt = torch.zeros(len(text), ce.INPUT_DIM, dtype=torch.float16)
    for k, ch in enumerate(text):
        emb = char_cache.get(ch)
        if emb is None:
            emb = ce.char_embedding(ch).to(torch.float16)
            char_cache[ch] = emb
        qt[k, ce.EMBED_LO : ce.EMBED_HI] = emb
        qt[k, ce.QUESTION_FLAG] = 1.0
    return qt


def encode_all_kinds_fast(
    scenes: list[dict[str, Any]], copies: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Bit-identical padded-to-20 encoder for Q1, Q2, and Q3."""
    counts = {"q1": 1, "q2": len(DIRECTIONS), "q3": len(nc2.PAIRS)}
    variants_per_scene = 1 + copies
    ticks: dict[str, torch.Tensor] = {
        k: torch.zeros(
            (len(scenes) * variants_per_scene * cnt, Q3_LEN, ce.INPUT_DIM), dtype=torch.float16
        )
        for k, cnt in counts.items()
    }
    labels: dict[str, list[int]] = {k: [] for k in KINDS}
    rows_idx = {k: 0 for k in KINDS}

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

    for scene in scenes:
        bodies = scene["bodies"]
        n_bodies = len(bodies)
        order = ce.scene_order(int(scene["seed"]), n_bodies)
        pos_f16 = [
            torch.tensor([float(v) for v in list(bodies[i]["position"])[:3]], dtype=torch.float16)
            for i in order
        ]
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
                kind = str(q["kind"])
                qtext = str(q["text"])
                end_t = n_bodies + len(qtext)
                if end_t > Q3_LEN:
                    raise ValueError(f"unexpected {kind} length: {end_t}")
                qt = q_cache.get(qtext)
                if qt is None:
                    qt = _question_ticks_f16(qtext, char_cache)
                    q_cache[qtext] = qt
                r = rows_idx[kind]
                ticks[kind][r, :n_bodies].copy_(scene_block)
                ticks[kind][r, n_bodies:end_t].copy_(qt)
                labels[kind].append(ce.ANSWER_INDEX[str(q["answer"])])
                rows_idx[kind] = r + 1

    return {k: (ticks[k], torch.tensor(labels[k], dtype=torch.long)) for k in KINDS}


def union_moments_dim(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    s = torch.zeros(ce.INPUT_DIM, dtype=torch.float64)
    s2 = torch.zeros(ce.INPUT_DIM, dtype=torch.float64)
    n = 0
    for x16 in tensors:
        for i in range(0, x16.shape[0], 4096):
            c = x16[i : i + 4096].double().reshape(-1, ce.INPUT_DIM)
            s += c.sum(0)
            s2 += (c * c).sum(0)
            n += c.shape[0]
    mu = s / n
    sd = (s2 / n - mu * mu).clamp(min=0.0).sqrt()
    sd = torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    return mu.float(), sd.float()


def build_data() -> dict[str, Any]:
    train_raw = encode_all_kinds_fast(nc2.load_scenes("train"), AUG_COPIES)
    val_raw = encode_all_kinds_fast(nc2.load_scenes("val"), 0)

    q3_train_dig = tensor_digest(*train_raw["q3"])
    q3_val_dig = tensor_digest(*val_raw["q3"])
    if q3_train_dig != EXPECTED_DIGESTS["train"]:
        raise AssertionError(f"Q3 train digest mismatch: {q3_train_dig}")
    if q3_val_dig != EXPECTED_DIGESTS["val"]:
        raise AssertionError(f"Q3 val digest mismatch: {q3_val_dig}")

    mu, sd = union_moments_dim([train_raw[k][0] for k in KINDS])

    out: dict[str, Any] = {
        "mu": mu,
        "sd": sd,
        "q3_digests": {"train": q3_train_dig, "val": q3_val_dig},
    }
    for split, raw in (("train", train_raw), ("val", val_raw)):
        out[split] = {}
        for k in KINDS:
            x, y = raw[k]
            sm, qm = extract_modality_masks(x)
            out[split][k] = {"x": x, "sm": sm, "qm": qm, "y": y}
    return out


def prep(x16: torch.Tensor, data: dict[str, Any]) -> torch.Tensor:
    mu: torch.Tensor = data["mu"]
    sd: torch.Tensor = data["sd"]
    return (x16.float() - mu) / sd


# --------------------------------------------------------------------------- training helpers
def _lr_at(step: int, total_steps: int) -> float:
    if step < WARMUP_STEPS:
        return LR * float(step + 1) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(max(total_steps - WARMUP_STEPS, 1))
    return LR * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _entity_mean(per_class: dict[str, float]) -> float:
    vals = [float(per_class[name]) for name in NAMES if name in per_class]
    return sum(vals) / len(vals) if vals else 0.0


def evaluate_kind(
    model: ModalityEdgeHop,
    data: dict[str, Any],
    split: str,
    kind: str,
    *,
    detail: bool = False,
) -> dict[str, Any]:
    d = data[split][kind]
    x16, sm, qm, y = d["x"], d["sm"], d["qm"], d["y"]
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
        out["entity_mean_acc"] = round(_entity_mean(out["per_class"]), 6)
    return out


def evaluate_mixed_equivalence(
    model: ModalityEdgeHop, data: dict[str, Any], *, n_per_kind: int = 300
) -> dict[str, Any]:
    """Verify task-agnostic S gives identical argmax predictions on shuffled mixed batches."""
    xs, sms, qms, sep_preds = [], [], [], []
    model.eval()
    with torch.no_grad():
        for kind in KINDS:
            d = data["val"][kind]
            xk, smk, qmk = d["x"][:n_per_kind], d["sm"][:n_per_kind], d["qm"][:n_per_kind]
            xs.append(xk)
            sms.append(smk)
            qms.append(qmk)
            sep_preds.append(model(prep(xk, data), smk, qmk).argmax(-1))
        x_cat = torch.cat(xs, dim=0)
        sm_cat = torch.cat(sms, dim=0)
        qm_cat = torch.cat(qms, dim=0)
        sep_cat = torch.cat(sep_preds, dim=0)
        perm = torch.randperm(x_cat.shape[0], generator=torch.Generator().manual_seed(20261125))
        mix_pred = model(prep(x_cat[perm], data), sm_cat[perm], qm_cat[perm]).argmax(-1)
    mismatches = int((mix_pred != sep_cat[perm]).sum().item())
    return {"checked": int(x_cat.shape[0]), "mismatches": mismatches, "exact": mismatches == 0}


def make_column(seed: int, tag: str) -> ModalityEdgeHop:
    return ModalityEdgeHop(
        len(ce.ANSWERS),
        d_model=D_MODEL,
        d_edge=D_EDGE,
        n_edge_layers=N_EDGE_LAYERS,
        n_hops=N_HOPS,
        seed=int_seed(CARD, seed, tag, "init"),
    )


def _train_stage(
    model: ModalityEdgeHop,
    data: dict[str, Any],
    kind: str,
    steps: int,
    seed: int,
    tag: str,
    deadline: float,
    *,
    replay_kinds: tuple[str, ...] = (),
    ckpts: tuple[int, ...] = (),
) -> dict[str, Any]:
    d = data["train"][kind]
    n = d["x"].shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    gen = torch.Generator().manual_seed(int_seed(CARD, seed, tag, kind, "batches"))
    rep_gens = {
        old: torch.Generator().manual_seed(int_seed(CARD, seed, tag, kind, "replay", old))
        for old in replay_kinds
    }
    ckpt_accs: dict[str, float] = {}
    replay_batches = 0
    t0 = time.process_time()
    eval_cpu = 0.0

    model.train()
    for i in range(steps):
        if time.process_time() > deadline:
            raise CapExceeded(f"{tag} {kind} step {i}")
        lr = _lr_at(i, steps)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, n, (BATCH,), generator=gen)
        logits = model(prep(d["x"][idx], data), d["sm"][idx], d["qm"][idx])
        loss = F.cross_entropy(logits, d["y"][idx])

        for old in replay_kinds:
            do = data["train"][old]
            r_idx = torch.randint(0, do["x"].shape[0], (REPLAY_BATCH,), generator=rep_gens[old])
            r_logits = model(prep(do["x"][r_idx], data), do["sm"][r_idx], do["qm"][r_idx])
            loss = loss + REPLAY_WEIGHT * F.cross_entropy(r_logits, do["y"][r_idx])
            replay_batches += 1

        if not torch.isfinite(loss):
            raise NonFiniteLoss(f"{tag} {kind} step {i}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        step_num = i + 1
        if step_num in ckpts:
            e0 = time.process_time()
            acc = evaluate_kind(model, data, "val", kind)["acc"]
            eval_cpu += time.process_time() - e0
            ckpt_accs[str(step_num)] = acc
            model.train()

    total_cpu = time.process_time() - t0
    return {
        "ckpts": ckpt_accs,
        "replay_batches": replay_batches,
        "train_loop_cpu": max(total_cpu - eval_cpu, 0.0),
        "eval_cpu": eval_cpu,
        "cpu": total_cpu,
    }


def run_shared(
    seed: int,
    data: dict[str, Any],
    deadline: float,
    stage_steps: dict[str, int] | None = None,
    *,
    skip_eval: bool = False,
) -> dict[str, Any]:
    steps_map = STAGE_STEPS if stage_steps is None else stage_steps
    model = make_column(seed, "S")
    done: list[str] = []
    peaks: dict[str, float] = {}
    stage_stats: dict[str, Any] = {}
    q3_ckpts: dict[str, float] = {}
    total_replay_batches = 0
    total_cpu = 0.0

    for kind in KINDS:
        ckpts = Q3_CKPTS if (not skip_eval and kind == "q3") else ()
        res = _train_stage(
            model,
            data,
            kind,
            steps_map[kind],
            seed,
            "S",
            deadline,
            replay_kinds=tuple(done),
            ckpts=ckpts,
        )
        stage_stats[kind] = res
        total_replay_batches += res["replay_batches"]
        total_cpu += res["cpu"]
        if kind == "q3":
            q3_ckpts = res["ckpts"]
        done.append(kind)
        if not skip_eval and kind != "q3":
            for old in done:
                acc = evaluate_kind(model, data, "val", old)["acc"]
                peaks[old] = max(peaks.get(old, 0.0), acc)

    if skip_eval:
        return {"stage_stats": stage_stats, "cpu": round(total_cpu, 2)}

    final_eval = {k: evaluate_kind(model, data, "val", k, detail=True) for k in KINDS}
    mixed = evaluate_mixed_equivalence(model, data)
    return {
        "arm": "S",
        "arch_version": ARCH_VERSION,
        "n_params": model.n_params(),
        "replay_batches": total_replay_batches,
        "peaks_before_q3": peaks,
        "q3_ckpts": q3_ckpts,
        "final": final_eval,
        "final_accs": {k: final_eval[k]["acc"] for k in KINDS},
        "mixed_equivalence": mixed,
        "stage_stats": stage_stats,
        "cpu": round(total_cpu, 2),
    }


def run_rt(
    seed: int,
    data: dict[str, Any],
    deadline: float,
    stage_steps: dict[str, int] | None = None,
    *,
    skip_eval: bool = False,
) -> dict[str, Any]:
    steps_map = STAGE_STEPS if stage_steps is None else stage_steps
    cols = {k: make_column(seed, f"RT_{k}") for k in KINDS}
    stage_stats: dict[str, Any] = {}
    q3_ckpts: dict[str, float] = {}
    total_cpu = 0.0

    for kind in KINDS:
        ckpts = Q3_CKPTS if (not skip_eval and kind == "q3") else ()
        res = _train_stage(
            cols[kind],
            data,
            kind,
            steps_map[kind],
            seed,
            f"RT_{kind}",
            deadline,
            replay_kinds=(),
            ckpts=ckpts,
        )
        stage_stats[kind] = res
        total_cpu += res["cpu"]
        if kind == "q3":
            q3_ckpts = res["ckpts"]

    if skip_eval:
        return {"stage_stats": stage_stats, "cpu": round(total_cpu, 2)}

    final_eval = {k: evaluate_kind(cols[k], data, "val", k, detail=True) for k in KINDS}
    return {
        "arm": "RT",
        "arch_version": ARCH_VERSION,
        "n_params": sum(c.n_params() for c in cols.values()),
        "q3_ckpts": q3_ckpts,
        "final": final_eval,
        "final_accs": {k: final_eval[k]["acc"] for k in KINDS},
        "stage_stats": stage_stats,
        "cpu": round(total_cpu, 2),
    }


def compute_r1_verdict(s_res: dict[str, Any], rt_res: dict[str, Any]) -> dict[str, Any]:
    if int(s_res.get("replay_batches", 0)) <= 0:
        return {"exit": "engineering_failure", "reason": "S replay_batches == 0"}
    if not s_res.get("mixed_equivalence", {}).get("exact", False):
        return {"exit": "engineering_failure", "reason": "mixed-batch equivalence failed"}

    s_final = s_res["final_accs"]
    g1_min = min(float(s_final[k]) for k in KINDS)

    s_ckpts = s_res["q3_ckpts"]
    rt_ckpts = rt_res["q3_ckpts"]
    diffs = [float(s_ckpts[str(c)]) - float(rt_ckpts[str(c)]) for c in Q3_CKPTS]
    g2 = sum(diffs) / len(diffs)
    ceiling_hit = float(rt_ckpts[str(Q3_CKPTS[0])]) >= CEILING_ACC

    peaks = s_res["peaks_before_q3"]
    bwt_q1 = float(s_final["q1"]) - float(peaks["q1"])
    bwt_q2 = float(s_final["q2"]) - float(peaks["q2"])
    g3_min = min(bwt_q1, bwt_q2)

    if g1_min < GATE_MIN or g2 < G3_MIN or g3_min < G3_MIN:
        exit_label = "pilot_signal_negative"
    elif g1_min >= GATE_POS and g2 >= G2_MIN and not ceiling_hit and g3_min >= G3_MIN:
        exit_label = "pilot_signal_positive"
    else:
        exit_label = "pilot_inconclusive"

    return {
        "exit": exit_label,
        "g1_final": {k: round(float(s_final[k]), 6) for k in KINDS},
        "g1_min": round(g1_min, 6),
        "g2_ckpt_diffs": {str(c): round(d, 6) for c, d in zip(Q3_CKPTS, diffs, strict=True)},
        "g2_mean": round(g2, 6),
        "ceiling_hit": ceiling_hit,
        "rt_first_ckpt": round(float(rt_ckpts[str(Q3_CKPTS[0])]), 6),
        "g3_bwt": {"q1": round(bwt_q1, 6), "q2": round(bwt_q2, 6)},
        "g3_min": round(g3_min, 6),
    }


# --------------------------------------------------------------------------- driver
def smoke(data: dict[str, Any], started: float, encode_cpu: float) -> dict[str, Any]:
    deadline = started + CAP_CPU
    smoke_map = {k: SMOKE_STEPS for k in KINDS}
    s_smoke = run_shared(SCREEN_SEED, data, deadline, smoke_map, skip_eval=True)
    rt_smoke = run_rt(SCREEN_SEED, data, deadline, smoke_map, skip_eval=True)

    probe_model = make_column(SCREEN_SEED, "S")
    e0 = time.process_time()
    eval_cpu = {}
    for k in KINDS:
        t_k = time.process_time()
        evaluate_kind(probe_model, data, "val", k, detail=True)
        eval_cpu[k] = time.process_time() - t_k
    eval_pass_cpu = time.process_time() - e0

    proj_train_cpu = 0.0
    for arm_stats in (s_smoke["stage_stats"], rt_smoke["stage_stats"]):
        for k in KINDS:
            per_step = arm_stats[k]["train_loop_cpu"] / SMOKE_STEPS
            proj_train_cpu += per_step * STAGE_STEPS[k]

    proj_eval_cpu = (
        eval_cpu["q1"] * 4 + eval_cpu["q2"] * 3 + eval_cpu["q3"] * (len(Q3_CKPTS) * 2 + 2)
    )
    proj_1seed = round(encode_cpu + (proj_train_cpu + proj_eval_cpu) * PROJECTION_MARGIN, 2)
    smoke_cpu = round(time.process_time() - started, 2)
    planned = "pilot_1seed" if smoke_cpu + proj_1seed <= CAP_CPU else "budget_insufficient"
    return {
        "encode_cpu": encode_cpu,
        "smoke_cpu": smoke_cpu,
        "eval_pass_cpu": round(eval_pass_cpu, 3),
        "proj_train_cpu": round(proj_train_cpu, 2),
        "proj_eval_cpu": round(proj_eval_cpu, 2),
        "projected_1seed": proj_1seed,
        "margin": PROJECTION_MARGIN,
        "cap_cpu": CAP_CPU,
        "planned": planned,
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
        "schema": "taiji-n32-6-01-v1",
        "card": CARD,
        "arch_version": ARCH_VERSION,
        "seed": SCREEN_SEED,
        "stage_steps": STAGE_STEPS,
        "q3_ckpts": list(Q3_CKPTS),
        "replay": {"batch": REPLAY_BATCH, "weight": REPLAY_WEIGHT},
        "design": [design],
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "data": {
            k: {
                "train_rows": int(data["train"][k]["y"].numel()),
                "val_rows": int(data["val"][k]["y"].numel()),
            }
            for k in KINDS
        }
        | {"q3_digests": data["q3_digests"]},
        "gates": {"g1_min": GATE_MIN, "g1_pos": GATE_POS, "g2_min": G2_MIN, "g3_min": G3_MIN},
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
        proj = smoke_doc["projection"]
        if proj["planned"] != "pilot_1seed":
            report.update({"exit": "budget_insufficient", "planned": proj["planned"]})
            status = 3
        else:
            budget = CAP_CPU - float(proj["smoke_cpu"])
            deadline = started + budget
            try:
                s_res = run_shared(SCREEN_SEED, data, deadline)
                rt_res = run_rt(SCREEN_SEED, data, deadline)
                verdict = compute_r1_verdict(s_res, rt_res)
                report.update({"planned": "pilot_1seed", "S": s_res, "RT": rt_res, **verdict})
            except CapExceeded as exc:
                report.update({"exit": "engineering_failure", "reason": f"cap exceeded: {exc}"})
            except NonFiniteLoss as exc:
                report.update({"exit": "engineering_failure", "reason": f"non-finite: {exc}"})
            if report.get("exit") in ("engineering_failure", "budget_insufficient"):
                status = 3
        out = FULL_REPORT

    report["cost"] = {"cpu": round(time.process_time() - started, 2), "cap_cpu": CAP_CPU}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: report.get(k) for k in ("exit", "g1_final", "g2_mean", "g3_bwt", "cost")}
    if args.smoke:
        summary["projection"] = report["projection"]
    print(json.dumps(summary, ensure_ascii=False))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
