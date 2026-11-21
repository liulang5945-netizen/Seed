"""Stage-2 cognitive loop verification on SelectivePersistentGraph (D-20261121-228, 0 check / 0 final).

Loops verified across 3 fixed seeds without touching ``taiji/u2_neural_units.py``:
1. ``step_mask_bidirectional_chinese``: 32-dim legal-channel char projection (zero pos emb) + padding-freeze ``step_mask``.
2. ``p24_cross_session_consolidation_4type``: 2-entity 5-session continuous stream (`N=48`) with stale/revocation/
   misleading/contradiction, trusted-only unprotected overwrite control, Session 3->4 state reset, 0 old-skill damage,
   and cross-session ``capsule_v2`` + state disk roundtrip.
3. ``p26_autonomous_epistemic_choice``: 2-turn partial-observability task with pure utility-grounded Q-value regression
   (zero oracle regime supervision), fixed-policy baselines, permuted-reveal ablation, and confidence Brier score.
4. ``p27_p28_causal_physics_and_t40_recovery``: multi-step occluded 1D friction physics + no-probe shortcut check,
   and long-gap (``t_target < 20``) horizon ``T=40`` selective hold vs ``static_c`` with mid-episode ``t=20`` recovery.
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
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import SEEDS, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.run_s2_cn_vocab3_20261121 import CLASSES as CN_CLASSES  # noqa: E402
from scripts.training.run_s2_cn_vocab3_20261121 import FIT as CN_FIT  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
    _clip_and_step,
    _skill_scores,
    _trainable_params,
)
from scripts.training.verify_taiji_selective_bridge_20261121 import (  # noqa: E402
    BI_CN_ADV_FIRST,
    BI_CN_ADV_SECOND,
    BI_CN_CHAR_INDEX,
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    BI_CN_VOCAB,
    CN_COLOURS,
    CN_MAX_LEN,
)
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_cognitive_loop_20261121.json"
CAP_CPU = 1200.0
FIT_OFFSET = 10_000_000
BATCH = 64
FIT = 256

_P32_GEN = torch.Generator().manual_seed(2026112109)
_P32_RAW = torch.randn(len(BI_CN_VOCAB), 32, generator=_P32_GEN)
BI_CN_PROJ32 = _P32_RAW / _P32_RAW.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Loop 1: 32-dim Legal-Channel Char Projection + Step-Masked Bidirectional CN
# ---------------------------------------------------------------------------


def _build_bidirectional_cn_32dim(
    count: int,
    seed: int,
    first_templates: tuple[str, ...],
    second_templates: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    all_tpls = [*[(t, True) for t in first_templates], *[(t, False) for t in second_templates]]
    rows_ticks: list[torch.Tensor] = []
    rows_labels: list[int] = []
    for idx in range(count):
        gen = torch.Generator().manual_seed(seed + idx)
        c1 = int(torch.randint(0, CN_CLASSES, (1,), generator=gen))
        c2 = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=gen))
        if c2 >= c1:
            c2 += 1
        tpl, pick_first = all_tpls[idx % len(all_tpls)]
        txt = tpl.format(c1=CN_COLOURS[c1], c2=CN_COLOURS[c2])
        seq = torch.zeros(CN_MAX_LEN, 48, dtype=torch.float32)
        for ti, ch in enumerate(txt[:CN_MAX_LEN]):
            char_idx = BI_CN_CHAR_INDEX.get(ch)
            if char_idx is not None:
                seq[ti, :32] = BI_CN_PROJ32[char_idx]
        rows_ticks.append(seq)
        rows_labels.append(c1 if pick_first else c2)
    return {"ticks": torch.stack(rows_ticks), "labels": torch.tensor(rows_labels, dtype=torch.long)}


def _eval_cn_masked(
    graph: SelectivePersistentGraph,
    ticks: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_step_mask: bool,
) -> float:
    n_units = graph.selective_spec.n_units
    gates = torch.ones(ticks.shape[0], ticks.shape[1], n_units)
    step_mask = (ticks.abs().sum(dim=-1) > 0).float() if use_step_mask else None
    with torch.no_grad():
        out = graph(
            _channels_seq(ticks),
            gates,
            graph.initial_state(batch_size=ticks.shape[0]),
            step_mask=step_mask,
        )
        preds = out["readout"].argmax(dim=-1)
    return float((preds == labels).float().mean())


def _train_cn_loop1_arm(
    base_graph: Any,
    train_data: dict[str, torch.Tensor],
    seed: int,
    *,
    use_step_mask: bool,
    steps: int = 1200,
    lr: float = 6e-3,
) -> SelectivePersistentGraph:
    spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        output_dim=CN_CLASSES,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed,
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    graph.freeze_base_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 17)
        graph._build_selective_branch(spec)
    optimizer = torch.optim.Adam(_trainable_params(graph), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 31)
    n_train = train_data["ticks"].shape[0]
    for step in range(steps):
        cur_lr = max(lr * 0.05, lr_at(step, steps) * (lr / 3e-3))
        for group in optimizer.param_groups:
            group["lr"] = min(lr, cur_lr)
        idx = torch.randint(0, n_train, (BATCH,), generator=generator)
        seq = train_data["ticks"][idx]
        gates = torch.ones(BATCH, seq.shape[1], spec.n_units)
        step_mask = (seq.abs().sum(dim=-1) > 0).float() if use_step_mask else None
        out = graph(
            _channels_seq(seq),
            gates,
            graph.initial_state(batch_size=BATCH),
            step_mask=step_mask,
        )
        loss = loss_fn(out["readout"], train_data["labels"][idx])
        _clip_and_step(optimizer, graph, loss)
    return graph


def run_loop_1_step_mask_cn(seed: int, base_graph: Any) -> dict[str, Any]:
    train_chars = {
        ch
        for tpl in (*BI_CN_TRAIN_SECOND, *BI_CN_TRAIN_FIRST)
        for ch in tpl.replace("{c1}", "").replace("{c2}", "")
    }
    fit_chars = {
        ch
        for tpl in (*BI_CN_FIT_SECOND, *BI_CN_FIT_FIRST, *BI_CN_ADV_FIRST, *BI_CN_ADV_SECOND)
        for ch in tpl.replace("{c1}", "").replace("{c2}", "")
    }
    unseen_chars = sorted(fit_chars - train_chars)

    bi_train = _build_bidirectional_cn_32dim(
        1600, seed + 600, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    bi_fit = _build_bidirectional_cn_32dim(
        CN_FIT, seed + 600 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    adv_fit = _build_bidirectional_cn_32dim(
        CN_FIT, seed + 700 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )

    unmasked = _train_cn_loop1_arm(base_graph, bi_train, seed + 101, use_step_mask=False)
    masked = _train_cn_loop1_arm(base_graph, bi_train, seed + 101, use_step_mask=True)

    return {
        "unmasked_bi_acc": _eval_cn_masked(
            unmasked, bi_fit["ticks"], bi_fit["labels"], use_step_mask=False
        ),
        "masked_bi_acc": _eval_cn_masked(
            masked, bi_fit["ticks"], bi_fit["labels"], use_step_mask=True
        ),
        "unmasked_adv_acc": _eval_cn_masked(
            unmasked, adv_fit["ticks"], adv_fit["labels"], use_step_mask=False
        ),
        "masked_adv_acc": _eval_cn_masked(
            masked, adv_fit["ticks"], adv_fit["labels"], use_step_mask=True
        ),
        "unseen_chars": unseen_chars,
    }


# ---------------------------------------------------------------------------
# Loop 2: Package 24 Cross-Session Consolidation & 4-Type Epistemic Correction
# ---------------------------------------------------------------------------


def _make_p24_stream(
    count: int,
    seed: int,
    *,
    force_trusted: bool = False,
    trusted_regimes_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Build 2-entity 5-session T=14 stream (S1-2 BASE, S3 4-type revision, S4-5 noise, QUERY)."""
    horizon = 14
    generator = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    ticks[:, : horizon - 1, 16:24] = torch.randn(count, horizon - 1, 8, generator=generator) * 0.2

    base_vals = torch.randint(0, 4, (count, 2), generator=generator)
    e_queries = torch.randint(0, 2, (count,), generator=generator)
    pool = (
        torch.tensor([0, 1, 3], dtype=torch.long)
        if trusted_regimes_only
        else torch.arange(4, dtype=torch.long)
    )
    regimes = pool[torch.arange(count, dtype=torch.long) % pool.numel()]
    regimes = regimes[torch.randperm(count, generator=generator)]

    val_labels = torch.zeros(count, dtype=torch.long)
    conflict_labels = torch.zeros(count, dtype=torch.float32)

    for b in range(count):
        eq = int(e_queries[b])
        v_base = int(base_vals[b, eq])
        other = 1 - eq

        # Sessions 1-2 (ticks 0..3): 2 base entities + 2 untrusted distractors on other entity
        order_s12 = torch.randperm(4, generator=generator).tolist()
        for idx_pos, t in enumerate(order_s12):
            if idx_pos < 2:
                ticks[b, t, idx_pos] = 1.0
                ticks[b, t, 4 + int(base_vals[b, idx_pos])] = 1.0
                ticks[b, t, 8] = 1.0  # BASE
                ticks[b, t, 12] = 1.0  # trusted
            else:
                ticks[b, t, other] = 1.0
                ticks[b, t, 4 + int(torch.randint(0, 4, (1,), generator=generator))] = 1.0
                ticks[b, t, 9] = 1.0
                ticks[b, t, 13] = 1.0  # untrusted distractor

        # Session 3 (ticks 4..8): epistemic correction regimes on eq
        reg = int(regimes[b])
        s3_perm = torch.randperm(5, generator=generator).tolist()
        t_first, t_second = sorted((4 + s3_perm[0], 4 + s3_perm[1]))
        if reg == 0:  # stale -> trusted UPDATE(v_new != v_base)
            v_new = (v_base + 1 + int(torch.randint(0, 3, (1,), generator=generator))) % 4
            ticks[b, t_first, eq] = 1.0
            ticks[b, t_first, 4 + v_new] = 1.0
            ticks[b, t_first, 9] = 1.0
            ticks[b, t_first, 12] = 1.0
            val_labels[b] = v_new
        elif reg == 1:  # revocation -> UPDATE(v_temp) then REVOKE
            v_temp = (v_base + 1 + int(torch.randint(0, 3, (1,), generator=generator))) % 4
            ticks[b, t_first, eq] = 1.0
            ticks[b, t_first, 4 + v_temp] = 1.0
            ticks[b, t_first, 9] = 1.0
            ticks[b, t_first, 12] = 1.0
            ticks[b, t_second, eq] = 1.0
            ticks[b, t_second, 10] = 1.0
            ticks[b, t_second, 12] = 1.0
            val_labels[b] = v_base
        elif reg == 2:  # misleading -> untrusted UPDATE(v_fake != v_base)
            v_fake = (v_base + 1 + int(torch.randint(0, 3, (1,), generator=generator))) % 4
            ticks[b, t_first, eq] = 1.0
            ticks[b, t_first, 4 + v_fake] = 1.0
            ticks[b, t_first, 9] = 1.0
            ticks[b, t_first, 12 if force_trusted else 13] = 1.0
            val_labels[b] = v_base
        else:  # contradiction -> two conflicting trusted UPDATEs on eq
            off1 = 1 + int(torch.randint(0, 3, (1,), generator=generator))
            off2 = 1 + ((off1 + int(torch.randint(0, 2, (1,), generator=generator))) % 3)
            ticks[b, t_first, eq] = 1.0
            ticks[b, t_first, 4 + ((v_base + off1) % 4)] = 1.0
            ticks[b, t_first, 9] = 1.0
            ticks[b, t_first, 12] = 1.0
            ticks[b, t_second, eq] = 1.0
            ticks[b, t_second, 4 + ((v_base + off2) % 4)] = 1.0
            ticks[b, t_second, 9] = 1.0
            ticks[b, t_second, 12] = 1.0
            val_labels[b] = v_base
            conflict_labels[b] = 1.0

        # Sessions 4-5 (ticks 9..12): Gaussian noise only (no replay of S1-3); tick 13: QUERY
        ticks[b, 13, eq] = 1.0
        ticks[b, 13, 11] = 1.0

    return {
        "ticks": ticks,
        "val_labels": val_labels,
        "conflict_labels": conflict_labels,
        "regimes": regimes,
    }


def _train_p24_graph(
    base_graph: Any,
    seed: int,
    *,
    force_trusted: bool = False,
    trusted_regimes_only: bool = False,
    steps: int = 1400,
) -> SelectivePersistentGraph:
    spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=6,
        output_dim=5,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed,
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    graph.freeze_base_()
    optimizer = torch.optim.Adam(_trainable_params(graph), lr=7e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    for step in range(steps):
        batch = _make_p24_stream(
            BATCH,
            seed * 25000 + step,
            force_trusted=force_trusted,
            trusted_regimes_only=trusted_regimes_only,
        )
        gates = torch.ones(BATCH, 14, spec.n_units)
        out = graph(_channels_seq(batch["ticks"]), gates, graph.initial_state(batch_size=BATCH))
        logits = out["readout"]
        loss = ce(logits[:, :4], batch["val_labels"]) + bce(logits[:, 4], batch["conflict_labels"])
        _clip_and_step(optimizer, graph, loss)
    return graph


def run_loop_2_p24_consolidation(
    seed: int, base_graph: Any, skill_data: dict[str, torch.Tensor]
) -> dict[str, Any]:
    fit_data = _make_p24_stream(FIT, seed + 25000 + FIT_OFFSET)
    base_before = _skill_scores(SelectivePersistentGraph(base_graph), skill_data)

    main_graph = _train_p24_graph(
        base_graph, seed + 211, force_trusted=False, trusted_regimes_only=False
    )
    unprotected_graph = _train_p24_graph(
        base_graph, seed + 212, force_trusted=True, trusted_regimes_only=True
    )

    with torch.no_grad():
        gates_full = torch.ones(FIT, 14, main_graph.selective_spec.n_units)
        out_full = main_graph(
            _channels_seq(fit_data["ticks"]), gates_full, main_graph.initial_state(batch_size=FIT)
        )["readout"]
        val_preds = out_full[:, :4].argmax(dim=-1)
        val_correct = (val_preds == fit_data["val_labels"]).float()
        conflict_preds = (out_full[:, 4] > 0.0).float()
        reg = fit_data["regimes"]

        stale_acc = float(val_correct[reg == 0].mean())
        revocation_acc = float(val_correct[reg == 1].mean())
        misleading_acc = float(val_correct[reg == 2].mean())
        contradiction_val_acc = float(val_correct[reg == 3].mean())
        contradiction_detect_acc = float(
            (conflict_preds == fit_data["conflict_labels"]).float().mean()
        )
        contradiction_pos_recall = float((conflict_preds[reg == 3] == 1.0).float().mean())
        overall_acc = float(val_correct.mean())

        # Unprotected control (trained on trusted regimes, evaluated on trust-masked misleading)
        fit_unprot = _make_p24_stream(
            FIT, seed + 25000 + FIT_OFFSET, force_trusted=True, trusted_regimes_only=False
        )
        out_unprot = unprotected_graph(
            _channels_seq(fit_unprot["ticks"]),
            gates_full,
            unprotected_graph.initial_state(batch_size=FIT),
        )["readout"]
        unprot_mask = fit_unprot["regimes"] == 2
        unprot_misleading_acc = float(
            (out_unprot[:, :4].argmax(dim=-1)[unprot_mask] == fit_unprot["val_labels"][unprot_mask])
            .float()
            .mean()
        )

        # No-consolidation reset control at tick=9 (boundary between Session 3 and Session 4)
        gates_tail = torch.ones(FIT, 5, main_graph.selective_spec.n_units)
        out_reset = main_graph(
            _channels_seq(fit_data["ticks"][:, 9:]),
            gates_tail,
            main_graph.initial_state(batch_size=FIT),
        )["readout"]
        no_consolidation_acc = float(
            (out_reset[:, :4].argmax(dim=-1) == fit_data["val_labels"]).float().mean()
        )

    after_scores = _skill_scores(main_graph, skill_data)
    old_skill_damage = max(max(0.0, base_before[k] - after_scores[k]) for k in base_before)

    # Cross-session capsule_v2 + state disk roundtrip at tick=9
    with torch.no_grad():
        pre_s4 = main_graph(
            _channels_seq(fit_data["ticks"][:, :9]),
            torch.ones(FIT, 9, main_graph.selective_spec.n_units),
            main_graph.initial_state(batch_size=FIT),
        )
    cap = build_capsule_v2(main_graph, clocks=(3, 2))
    saved_state = {k: v.detach().clone() for k, v in pre_s4["next_state"].items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        cap_path = Path(tmpdir) / f"p24_cap_{seed}.pt"
        state_path = Path(tmpdir) / f"p24_state_{seed}.pt"
        save_capsule_v2(cap, cap_path)
        torch.save(saved_state, state_path)
        loaded_cap = load_capsule_v2(cap_path)
        loaded_state = cast(dict[str, torch.Tensor], torch.load(state_path, weights_only=True))
    fresh = SelectivePersistentGraph(copy.deepcopy(base_graph))
    apply_capsule_v2(fresh, loaded_cap)
    with torch.no_grad():
        resumed = fresh(_channels_seq(fit_data["ticks"][:, 9:]), gates_tail, loaded_state)[
            "readout"
        ]
    capsule_diff = float((out_full - resumed).abs().max())

    return {
        "overall_acc": overall_acc,
        "stale_acc": stale_acc,
        "revocation_acc": revocation_acc,
        "misleading_acc": misleading_acc,
        "unprotected_misleading_acc": unprot_misleading_acc,
        "contradiction_val_acc": contradiction_val_acc,
        "contradiction_detect_acc": contradiction_detect_acc,
        "contradiction_pos_recall": contradiction_pos_recall,
        "no_consolidation_acc": no_consolidation_acc,
        "consolidation_gain": overall_acc - no_consolidation_acc,
        "old_skill_damage": old_skill_damage,
        "capsule_v2_max_abs_diff": capsule_diff,
    }


# ---------------------------------------------------------------------------
# Loop 3: Package 22.6/26 Autonomous Epistemic Choice (Q-Value Utility)
# ---------------------------------------------------------------------------


def _make_p26_batch(count: int, seed: int) -> dict[str, torch.Tensor]:
    """Build 3 balanced partial-observability regimes (known, ambiguous, unresolvable), zero regime label."""
    generator = torch.Generator().manual_seed(seed)
    turn0 = torch.zeros(count, 4, 48, dtype=torch.float32)
    reveal = torch.zeros(count, 1, 48, dtype=torch.float32)
    regimes = (torch.arange(count, dtype=torch.long) % 3)[
        torch.randperm(count, generator=generator)
    ]

    c1 = torch.randint(0, 4, (count,), generator=generator)
    c2 = (c1 + 1 + torch.randint(0, 3, (count,), generator=generator)) % 4
    rule = torch.randint(0, 2, (count,), generator=generator)
    labels = torch.where(rule == 0, c1, c2)

    for b in range(count):
        reg = int(regimes[b])
        if reg in (0, 1):
            turn0[b, :, 16:24] = torch.randn(4, 8, generator=generator) * 0.1
            turn0[b, 0, int(c1[b])] = 1.0
            turn0[b, 1, int(c2[b])] = 1.0
            if reg == 0:  # known
                turn0[b, 2, 8 + int(rule[b])] = 1.0
                reveal[b, 0, 34] = 1.0
            else:  # ambiguous
                turn0[b, 2, 10] = 1.0
                reveal[b, 0, 8 + int(rule[b])] = 1.0
                reveal[b, 0, 34] = 1.0
            turn0[b, 3, 33] = 1.0
        else:  # unresolvable: corrupt colour noise + high-variance sensor static
            labels[b] = int(torch.randint(0, 4, (1,), generator=generator))
            turn0[b, :3, :4] = torch.rand(3, 4, generator=generator)
            turn0[b, :, 16:24] = torch.randn(4, 8, generator=generator) * 1.5
            turn0[b, 3, 33] = 1.0
            reveal[b, 0, :4] = torch.rand(4, generator=generator)
            reveal[b, 0, 16:24] = torch.randn(8, generator=generator) * 1.5
            reveal[b, 0, 34] = 1.0

    return {"turn0": turn0, "reveal": reveal, "labels": labels, "regimes": regimes}


def run_loop_3_p26_epistemic_choice(seed: int, base_graph: Any) -> dict[str, Any]:
    fit_count = 384
    spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        output_dim=8,  # 0..3: ans, 4..6: expected utility Q(ACT/QUERY/WAIT), 7: conf
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 311,
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    graph.freeze_base_()
    optimizer = torch.optim.Adam(_trainable_params(graph), lr=6e-3)
    steps = 1000

    for step in range(steps):
        batch = _make_p26_batch(BATCH, seed * 30000 + step)
        out0 = graph(
            _channels_seq(batch["turn0"]),
            torch.ones(BATCH, 4, spec.n_units),
            graph.initial_state(batch_size=BATCH),
        )
        ro0 = out0["readout"]
        logits_ans_0 = ro0[:, 0:4]
        q_pred = ro0[:, 4:7]
        logit_conf_0 = ro0[:, 7]
        a_0 = logits_ans_0.argmax(dim=-1)

        out1 = graph(
            _channels_seq(batch["reveal"]),
            torch.ones(BATCH, 1, spec.n_units),
            out0["next_state"],
        )
        logits_ans_1 = out1["readout"][:, 0:4]
        a_1 = logits_ans_1.argmax(dim=-1)

        y = batch["labels"]
        u_act = torch.where(a_0 == y, 1.0, -1.0)
        u_query = torch.where(a_1 == y, 1.0, -1.0) - 0.20
        u_wait = torch.full_like(u_act, -0.05)
        utils = torch.stack([u_act, u_query, u_wait], dim=-1).detach()

        loss = (
            F.cross_entropy(logits_ans_0, y)
            + F.cross_entropy(logits_ans_1, y)
            + F.mse_loss(q_pred, utils)
            + F.binary_cross_entropy_with_logits(logit_conf_0, (a_0 == y).float().detach())
        )
        _clip_and_step(optimizer, graph, loss)

    fit_data = _make_p26_batch(fit_count, seed + 30000 + FIT_OFFSET)
    with torch.no_grad():
        out0 = graph(
            _channels_seq(fit_data["turn0"]),
            torch.ones(fit_count, 4, spec.n_units),
            graph.initial_state(batch_size=fit_count),
        )
        ro0 = out0["readout"]
        logits_ans_0 = ro0[:, 0:4]
        pol_choice = ro0[:, 4:7].argmax(dim=-1)
        conf_0 = torch.sigmoid(ro0[:, 7])
        a_0 = logits_ans_0.argmax(dim=-1)

        out1_real = graph(
            _channels_seq(fit_data["reveal"]),
            torch.ones(fit_count, 1, spec.n_units),
            out0["next_state"],
        )
        a_1_real = out1_real["readout"][:, 0:4].argmax(dim=-1)

        perm = torch.roll(torch.arange(fit_count), shifts=1)
        out1_perm = graph(
            _channels_seq(fit_data["reveal"][perm]),
            torch.ones(fit_count, 1, spec.n_units),
            out0["next_state"],
        )
        a_1_perm = out1_perm["readout"][:, 0:4].argmax(dim=-1)

        y = fit_data["labels"]
        reg = fit_data["regimes"]
        u_act = torch.where(a_0 == y, 1.0, -1.0)
        u_query = torch.where(a_1_real == y, 1.0, -1.0) - 0.20
        u_wait = torch.full_like(u_act, -0.05)

        realized_u = torch.where(
            pol_choice == 0,
            u_act,
            torch.where(pol_choice == 1, u_query, u_wait),
        )
        amb_mask = reg == 1
        return {
            "net_utility": float(realized_u.mean()),
            "always_act_utility": float(u_act.mean()),
            "always_query_utility": float(u_query.mean()),
            "always_wait_utility": float(u_wait.mean()),
            "known_act_rate": float((pol_choice[reg == 0] == 0).float().mean()),
            "ambiguous_query_rate": float((pol_choice[amb_mask] == 1).float().mean()),
            "unresolvable_wait_rate": float((pol_choice[reg == 2] == 2).float().mean()),
            "ambiguous_turn1_acc": float((a_1_real[amb_mask] == y[amb_mask]).float().mean()),
            "ambiguous_turn1_permuted_acc": float(
                (a_1_perm[amb_mask] == y[amb_mask]).float().mean()
            ),
            "brier_score": float(((conf_0 - (a_0 == y).float()) ** 2).mean()),
        }


# ---------------------------------------------------------------------------
# Loop 4: Package 27 Causal Occluded Physics & Package 28 T=40 Crash Recovery
# ---------------------------------------------------------------------------


def _make_p27_physics_batch(count: int, seed: int) -> dict[str, torch.Tensor]:
    """Balanced 4-class T=14 1D occluded block dynamics (ticks 0..5 probe, ticks 6..13 occluded)."""
    horizon = 14
    generator = torch.Generator().manual_seed(seed)
    per_class = count // 4
    collected: list[list[torch.Tensor]] = [[], [], [], []]

    while any(len(c) < per_class for c in collected):
        mu = 0.08 if int(torch.randint(0, 2, (1,), generator=generator)) == 0 else 0.42
        x_wall = float(2.5 + 2.5 * torch.rand((), generator=generator))
        f_probe = float(0.9 + 0.5 * torch.rand((), generator=generator))
        f_test = float(0.6 + 2.0 * torch.rand((), generator=generator))

        x_o, v_o = 0.0, 0.0
        v_hit = 0.0
        hit = False
        for k in range(8):
            f_k = f_test if k == 0 else 0.0
            v_o = v_o * (1.0 - mu) + f_k
            x_o = x_o + v_o
            if not hit and x_o >= x_wall:
                hit = True
                v_hit = v_o
        d_free = x_o
        if d_free < x_wall - 0.6:
            cls = 0
        elif d_free < x_wall:
            cls = 1
        elif v_hit < 0.70:
            cls = 2
        else:
            cls = 3

        if len(collected[cls]) >= per_class:
            continue

        seq = torch.zeros(horizon, 48, dtype=torch.float32)
        colours = torch.randint(0, 8, (horizon,), generator=generator)
        seq.scatter_(1, colours.unsqueeze(-1), 1.0)

        x_p, v_p = 0.0, 0.0
        for t in range(6):
            f_t = f_probe if t == 0 else 0.0
            v_p = v_p * (1.0 - mu) + f_t
            x_p = x_p + v_p
            seq[t, 8] = x_p / 5.0
            seq[t, 9] = v_p / 2.0
            seq[t, 10] = f_t / 2.0
            seq[t, 11] = x_wall / 5.0
            seq[t, 12] = 0.0

        seq[6, 10] = f_test / 2.0
        seq[6, 11] = x_wall / 5.0
        seq[6, 12] = 1.0

        seq[7:, 12] = 1.0
        seq[13, 11] = x_wall / 5.0
        seq[13, 33] = 1.0
        collected[cls].append(seq)

    all_ticks = torch.cat([torch.stack(c[:per_class]) for c in collected], dim=0)
    all_labels = torch.cat(
        [torch.full((per_class,), cls, dtype=torch.long) for cls in range(4)], dim=0
    )
    perm = torch.randperm(all_ticks.shape[0], generator=generator)
    return {"ticks": all_ticks[perm], "labels": all_labels[perm]}


def _make_m1_long_gap_data(
    count: int, seed: int, *, horizon: int = 40, max_target_tick: int = 20
) -> dict[str, torch.Tensor]:
    """Horizon T=40 hold with target strictly in [0, 19] (forced >=20-tick gap to query)."""
    generator = torch.Generator().manual_seed(seed)
    ticks = torch.zeros(count, horizon, 48, dtype=torch.float32)
    labels = torch.randint(0, 8, (count,), generator=generator)
    target_ticks = torch.randint(0, max_target_tick, (count,), generator=generator)

    distractors = torch.randint(0, 8, (count, horizon), generator=generator)
    ticks.scatter_(2, distractors.unsqueeze(-1), 1.0)
    ticks[:, :, 8:16] = torch.randn(count, horizon, 8, generator=generator) * 0.25

    row_idx = torch.arange(count)
    ticks[row_idx, target_ticks, :8] = 0.0
    ticks[row_idx, target_ticks, labels] = 1.0
    ticks[row_idx, target_ticks, 32] = 1.0
    ticks[:, horizon - 1, 33] = 1.0
    return {"ticks": ticks, "labels": labels, "target_ticks": target_ticks}


def run_loop_4_physics_and_t40(seed: int, base_graph: Any) -> dict[str, Any]:
    # Part 4A: Multi-step occluded causal physics
    phys_spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=4,
        output_dim=4,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 411,
    )
    phys_graph = SelectivePersistentGraph(copy.deepcopy(base_graph), phys_spec)
    phys_graph.freeze_base_()
    opt_phys = torch.optim.Adam(_trainable_params(phys_graph), lr=7e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    for step in range(1200):
        batch = _make_p27_physics_batch(BATCH, seed * 40000 + step)
        out = phys_graph(
            _channels_seq(batch["ticks"]),
            torch.ones(BATCH, 14, phys_spec.n_units),
            phys_graph.initial_state(batch_size=BATCH),
        )
        loss = ce(out["readout"], batch["labels"])
        _clip_and_step(opt_phys, phys_graph, loss)

    fit_phys = _make_p27_physics_batch(FIT, seed + 40000 + FIT_OFFSET)
    no_probe_ticks = fit_phys["ticks"].clone()
    no_probe_ticks[:, :6, 8:11] = 0.0
    with torch.no_grad():
        gates_p = torch.ones(FIT, 14, phys_spec.n_units)
        preds_phys = phys_graph(
            _channels_seq(fit_phys["ticks"]), gates_p, phys_graph.initial_state(batch_size=FIT)
        )["readout"].argmax(dim=-1)
        preds_noprobe = phys_graph(
            _channels_seq(no_probe_ticks), gates_p, phys_graph.initial_state(batch_size=FIT)
        )["readout"].argmax(dim=-1)
    physics_acc = float((preds_phys == fit_phys["labels"]).float().mean())
    physics_no_probe_acc = float((preds_noprobe == fit_phys["labels"]).float().mean())

    # Part 4B: Long-gap T=40 M1 Selective vs static_c + Mid-episode t=20 crash recovery
    def _train_m1_t40(*, selective_gate: bool, offset_seed: int) -> SelectivePersistentGraph:
        spec = SelectiveUnitSpec(
            groups=6,
            units_per_group=5,
            output_dim=8,
            readout_hidden=64,
            per_sample_state=True,
            selective_gate=selective_gate,
            init_write_bias=-2.5,
            seed=seed + offset_seed,
        )
        g = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
        g.freeze_base_()
        opt = torch.optim.Adam(_trainable_params(g), lr=7e-3)
        for step in range(800):
            batch = _make_m1_long_gap_data(BATCH, seed * 50000 + step)
            out = g(
                _channels_seq(batch["ticks"]),
                torch.ones(BATCH, 40, spec.n_units),
                g.initial_state(batch_size=BATCH),
            )
            loss = ce(out["readout"], batch["labels"])
            _clip_and_step(opt, g, loss)
        return g

    g_sel = _train_m1_t40(selective_gate=True, offset_seed=421)
    g_static = _train_m1_t40(selective_gate=False, offset_seed=422)
    fit_m1 = _make_m1_long_gap_data(FIT, seed + 50000 + FIT_OFFSET)
    with torch.no_grad():
        gates_40 = torch.ones(FIT, 40, g_sel.selective_spec.n_units)
        out_full_sel = g_sel(
            _channels_seq(fit_m1["ticks"]), gates_40, g_sel.initial_state(batch_size=FIT)
        )["readout"]
        out_full_static = g_static(
            _channels_seq(fit_m1["ticks"]), gates_40, g_static.initial_state(batch_size=FIT)
        )["readout"]
        sel_t40 = float((out_full_sel.argmax(dim=-1) == fit_m1["labels"]).float().mean())
        static_t40 = float((out_full_static.argmax(dim=-1) == fit_m1["labels"]).float().mean())

        pre_crash = g_sel(
            _channels_seq(fit_m1["ticks"][:, :20]),
            torch.ones(FIT, 20, g_sel.selective_spec.n_units),
            g_sel.initial_state(batch_size=FIT),
        )
    cap = build_capsule_v2(g_sel, clocks=(4, 2))
    saved_state = {k: v.detach().clone() for k, v in pre_crash["next_state"].items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        cap_path = Path(tmpdir) / f"t40_cap_{seed}.pt"
        state_path = Path(tmpdir) / f"t40_state_{seed}.pt"
        save_capsule_v2(cap, cap_path)
        torch.save(saved_state, state_path)
        loaded_cap = load_capsule_v2(cap_path)
        loaded_state = cast(dict[str, torch.Tensor], torch.load(state_path, weights_only=True))
    recovered = SelectivePersistentGraph(copy.deepcopy(base_graph))
    apply_capsule_v2(recovered, loaded_cap)
    with torch.no_grad():
        resumed = recovered(
            _channels_seq(fit_m1["ticks"][:, 20:]),
            torch.ones(FIT, 20, recovered.selective_spec.n_units),
            loaded_state,
        )["readout"]
    crash_diff = float((out_full_sel - resumed).abs().max())

    return {
        "physics_acc": physics_acc,
        "physics_no_probe_acc": physics_no_probe_acc,
        "physics_probe_drop": physics_acc - physics_no_probe_acc,
        "selective_t40_acc": sel_t40,
        "static_c_t40_acc": static_t40,
        "crash_recovery_max_abs_diff": crash_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    started = time.process_time()
    torch.set_num_threads(1)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        if time.process_time() - started > CAP_CPU:
            raise RuntimeError("cognitive_loop_budget_exceeded")
        base_graph, skill_data = train_joint_parent(seed)
        rows.append(
            {
                "seed": seed,
                "loop1_step_mask_cn": run_loop_1_step_mask_cn(seed, base_graph),
                "loop2_p24_consolidation": run_loop_2_p24_consolidation(
                    seed, base_graph, skill_data
                ),
                "loop3_p26_epistemic": run_loop_3_p26_epistemic_choice(seed, base_graph),
                "loop4_p27_p28": run_loop_4_physics_and_t40(seed, base_graph),
            }
        )

    l1_rows = [r["loop1_step_mask_cn"] for r in rows]
    l2_rows = [r["loop2_p24_consolidation"] for r in rows]
    l3_rows = [r["loop3_p26_epistemic"] for r in rows]
    l4_rows = [r["loop4_p27_p28"] for r in rows]

    s1 = {
        "unmasked_bi_acc": _mean([r["unmasked_bi_acc"] for r in l1_rows]),
        "masked_bi_acc": _mean([r["masked_bi_acc"] for r in l1_rows]),
        "unmasked_adv_acc": _mean([r["unmasked_adv_acc"] for r in l1_rows]),
        "masked_adv_acc": _mean([r["masked_adv_acc"] for r in l1_rows]),
        "unseen_chars": l1_rows[0]["unseen_chars"],
    }

    s2 = {
        "overall_acc": _mean([r["overall_acc"] for r in l2_rows]),
        "stale_acc": _mean([r["stale_acc"] for r in l2_rows]),
        "revocation_acc": _mean([r["revocation_acc"] for r in l2_rows]),
        "misleading_acc": _mean([r["misleading_acc"] for r in l2_rows]),
        "unprotected_misleading_acc": _mean([r["unprotected_misleading_acc"] for r in l2_rows]),
        "contradiction_val_acc": _mean([r["contradiction_val_acc"] for r in l2_rows]),
        "contradiction_detect_acc": _mean([r["contradiction_detect_acc"] for r in l2_rows]),
        "contradiction_pos_recall": _mean([r["contradiction_pos_recall"] for r in l2_rows]),
        "no_consolidation_acc": _mean([r["no_consolidation_acc"] for r in l2_rows]),
        "consolidation_gain": _mean([r["consolidation_gain"] for r in l2_rows]),
        "old_skill_damage_max": max(r["old_skill_damage"] for r in l2_rows),
        "capsule_v2_max_abs_diff": max(r["capsule_v2_max_abs_diff"] for r in l2_rows),
    }

    s3 = {
        k: _mean([r[k] for r in l3_rows])
        for k in (
            "net_utility",
            "always_act_utility",
            "always_query_utility",
            "always_wait_utility",
            "known_act_rate",
            "ambiguous_query_rate",
            "unresolvable_wait_rate",
            "ambiguous_turn1_acc",
            "ambiguous_turn1_permuted_acc",
            "brier_score",
        )
    }

    s4 = {
        "physics_acc": _mean([r["physics_acc"] for r in l4_rows]),
        "physics_no_probe_acc": _mean([r["physics_no_probe_acc"] for r in l4_rows]),
        "physics_probe_drop": _mean([r["physics_probe_drop"] for r in l4_rows]),
        "selective_t40_acc": _mean([r["selective_t40_acc"] for r in l4_rows]),
        "static_c_t40_acc": _mean([r["static_c_t40_acc"] for r in l4_rows]),
        "crash_recovery_max_abs_diff": max(r["crash_recovery_max_abs_diff"] for r in l4_rows),
    }

    best_fixed_u = max(
        float(s3["always_act_utility"]),
        float(s3["always_query_utility"]),
        float(s3["always_wait_utility"]),
    )
    gates = {
        "l1_bi_ge_085": float(s1["masked_bi_acc"]) >= 0.85,
        "l1_adv_beats_no_mask": float(s1["masked_adv_acc"]) >= float(s1["unmasked_adv_acc"]) + 0.05,
        "l1_adv_scope_ge_075": float(s1["masked_adv_acc"]) >= 0.75,
        "l1_unseen_chars_zero": len(cast(list[str], s1["unseen_chars"])) == 0,
        "l2_stale_and_revocation_ge_085": (
            s2["stale_acc"] >= 0.85 and s2["revocation_acc"] >= 0.85
        ),
        "l2_misleading_resistance_ge_085": s2["misleading_acc"] >= 0.85,
        "l2_unprotected_control_fails": s2["unprotected_misleading_acc"] <= 0.40,
        "l2_contradiction_detected_ge_085": (
            s2["contradiction_detect_acc"] >= 0.85 and s2["contradiction_val_acc"] >= 0.80
        ),
        "l2_consolidation_gain_and_retention": (
            s2["consolidation_gain"] >= 0.40
            and s2["old_skill_damage_max"] == 0.0
            and s2["capsule_v2_max_abs_diff"] <= 1e-10
        ),
        "l3_beats_all_fixed_policies": float(s3["net_utility"]) >= best_fixed_u + 0.15,
        "l3_regime_routing_accurate": (
            s3["known_act_rate"] >= 0.85
            and s3["ambiguous_query_rate"] >= 0.85
            and s3["unresolvable_wait_rate"] >= 0.80
        ),
        "l3_turn1_after_query_ge_085": s3["ambiguous_turn1_acc"] >= 0.85,
        "l3_permuted_reveal_drop_ge_035": (
            s3["ambiguous_turn1_acc"] - s3["ambiguous_turn1_permuted_acc"] >= 0.35
        ),
        "l3_calibration_brier_le_015": s3["brier_score"] <= 0.15,
        "l4_physics_acc_ge_085": s4["physics_acc"] >= 0.85,
        "l4_physics_no_probe_drop_ge_030": s4["physics_probe_drop"] >= 0.30,
        "l4_long_horizon_t40_ge_090": (
            s4["selective_t40_acc"] >= 0.90
            and (s4["selective_t40_acc"] - s4["static_c_t40_acc"] >= 0.40)
        ),
        "l4_mid_episode_crash_recovery_exact": s4["crash_recovery_max_abs_diff"] <= 1e-10,
    }
    verdict = "cognitive_loop_verified" if all(gates.values()) else "cognitive_loop_partial"
    summary = {
        "loop1_step_mask_cn": s1,
        "loop2_p24_consolidation": s2,
        "loop3_p26_epistemic": s3,
        "loop4_p27_p28": s4,
    }
    payload: dict[str, Any] = {
        "schema": "taiji-u2-cognitive-loop-v1",
        "card": "D-20261121-228 Stage-2 cognitive loop closure on SelectivePersistentGraph",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "cap_cpu": CAP_CPU,
            "fit_offset": FIT_OFFSET,
        },
        "rows": rows,
        "summary": summary,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "summary": summary,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if verdict == "cognitive_loop_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
