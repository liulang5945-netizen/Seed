"""Four-axis capability bridge on additive selective units (D-20261121-227, 0 check / 0 final).

Bridges ``taiji/u2_selective_units.py`` without editing ``taiji/u2_neural_units.py``:
1. ``equal_flops_arch_compare``: SelectivePersistentGraph vs equal-FLOPs GRU & causal Transformer on M3.
2. ``neutral_growth_and_lesion``: 2-gen neutral growth (N=4->16->32), equal-step G5/G6 controls, H2 dormant lesion,
   old-skill damage=0.0, multi-gen capsule_v2 <= 1e-10.
3. ``a3c_shift_invariant_chinese``: exact A3c 5/5 split + 2-rule bidirectional Chinese (10/10, seen chars)
   comparing causal SelectivePersistentGraph (zero pos emb) vs PosAttentionRead raw baseline.
4. ``closed_loop_action_feedback``: 2-turn colour-ambiguous action task with binary right/wrong feedback
   ONLY (zero answer leakage), checking turn-1 correction & zero/permuted feedback controls.
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

from scripts.training.run_s2_23_0b_joint_20261120 import SEEDS, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.run_s2_action_prior_20261121 import (
    CLASSES as ACTION_CLASSES,
)
from scripts.training.run_s2_action_prior_20261121 import (
    COLOURS as ACTION_COLOURS,
)
from scripts.training.run_s2_action_prior_20261121 import (
    DIRECTIONS as ACTION_DIRECTIONS,
)
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    ACTION_CHAR_INDEX,
    ACTION_FIT_TEMPLATES,
    ACTION_PROJECTION,
    ACTION_TRAIN_TEMPLATES,
    ACTION_VOCAB,
    _channels_seq,
    _clip_and_step,
    _eval_selective_preds,
    _make_m3_data,
    _skill_scores,
    _trainable_params,
)

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_cn_vocab3_20261121 import (  # noqa: E402
    CLASSES as CN_CLASSES,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    COLOURS as CN_COLOURS,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    FIT as CN_FIT,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    FIT_TEMPLATES as CN_FIT_TEMPLATES,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    MAX_LEN as CN_MAX_LEN,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    TRAIN as CN_TRAIN,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    TRAIN_TEMPLATES as CN_TRAIN_TEMPLATES,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    PosAttentionRead,
)
from scripts.training.run_s2_cn_vocab3_20261121 import (
    build as build_cn_vocab3,
)
from taiji.u2_neural_units import LEGAL_INPUT_CHANNELS  # noqa: E402
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    _AntitheticSelectiveCohort,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_selective_bridge_20261121.json"
CAP_CPU = 1200.0
FIT_OFFSET = 10_000_000
BATCH = 64
FIT = 256

BI_CN_TRAIN_SECOND = (
    *CN_TRAIN_TEMPLATES,
    "先说{c1}再说{c2}，要后者",
)
BI_CN_TRAIN_FIRST = (
    "请把{c1}和{c2}里先说的那个拿给我",
    "先提{c1}再提{c2}，要前者",
    "{c1}与{c2}中，先提到的那个",
    "我要{c1}和{c2}里前说的",
    "从{c1}、{c2}里挑先说的那个",
    "先说{c1}再说{c2}，要前者",
)
BI_CN_FIT_SECOND = (
    "请给我{c1}和{c2}中后说的那个",
    "{c1}与{c2}中后说的那个",
    "挑{c1}与{c2}里后说的",
    "从{c1}、{c2}里，我要后说的那个",
)
BI_CN_FIT_FIRST = (
    "请给我{c1}和{c2}中先说的那个",
    "{c1}与{c2}中先说的那个",
    "挑{c1}与{c2}里前说的",
    "从{c1}、{c2}里，我要先说的那个",
)
BI_CN_ADV_FIRST = ("先说{c1}后说{c2}，要前者",)
BI_CN_ADV_SECOND = ("先说{c1}后说{c2}，要后者",)
BI_CN_VOCAB = tuple(
    sorted(
        {
            ch
            for tpl in (
                *BI_CN_TRAIN_SECOND,
                *BI_CN_TRAIN_FIRST,
                *BI_CN_FIT_SECOND,
                *BI_CN_FIT_FIRST,
            )
            for ch in tpl.replace("{c1}", "").replace("{c2}", "")
        }
        | set(CN_COLOURS)
    )
)
BI_CN_CHAR_INDEX = {ch: idx for idx, ch in enumerate(BI_CN_VOCAB)}
_BI_PROJ_GEN = torch.Generator().manual_seed(2026112107)
_BI_PROJ_RAW = torch.randn(len(BI_CN_VOCAB), 8, generator=_BI_PROJ_GEN)
BI_CN_PROJECTION = _BI_PROJ_RAW / _BI_PROJ_RAW.norm(dim=0, keepdim=True).clamp(min=1e-6)


# ---------------------------------------------------------------------------
# Bridge 1: Equal-FLOPs Architecture Comparison (Selective vs GRU vs Transformer)
# ---------------------------------------------------------------------------


def selective_flops_per_seq(
    steps: int,
    *,
    groups: int = 8,
    n_units: int = 32,
    hidden_dim: int = 8,
    base_groups: int = 6,
    input_dim: int = 8,
    readout_hidden: int = 64,
    output_dim: int = 4,
) -> int:
    """Analytic per-sequence forward FLOPs (2 * MACs + elementwise gates) for SelectivePersistentGraph."""
    raw_dim = len(LEGAL_INPUT_CHANNELS) * input_dim
    ctx_dim = groups * hidden_dim
    base_encode = steps * base_groups * (2 * input_dim * hidden_dim + 2 * hidden_dim * hidden_dim)
    cross_mix = steps * (2 * raw_dim * ctx_dim)
    drive = steps * (2 * n_units * hidden_dim)
    per_tick_units = (
        2 * n_units * n_units + 2 * ctx_dim * n_units + 2 * n_units * n_units + 14 * n_units
    )
    readout = (
        2 * raw_dim * (2 * n_units)
        + 2 * (2 * n_units) * readout_hidden
        + 2 * readout_hidden * output_dim
    )
    return int(base_encode + cross_mix + drive + steps * per_tick_units + readout)


def gru_flops_per_seq(
    steps: int,
    hidden_size: int,
    *,
    input_size: int = 48,
    readout_hidden: int = 64,
    output_dim: int = 4,
) -> int:
    per_tick = 6 * (input_size * hidden_size + hidden_size * hidden_size) + 10 * hidden_size
    readout = 2 * hidden_size * readout_hidden + 2 * readout_hidden * output_dim
    return int(steps * per_tick + readout)


def transformer_flops_per_seq(
    steps: int,
    d_model: int,
    dim_ff: int,
    *,
    input_size: int = 48,
    readout_hidden: int = 64,
    output_dim: int = 4,
) -> int:
    proj = steps * (2 * input_size * d_model)
    attn = steps * 2 * (4 * d_model * d_model + 2 * steps * d_model)
    ffn = steps * 2 * (2 * d_model * dim_ff)
    readout = 2 * d_model * readout_hidden + 2 * readout_hidden * output_dim
    return int(proj + attn + ffn + readout)


def match_gru_hidden(target_flops: int, steps: int = 12) -> tuple[int, int]:
    best_h = 32
    best_f = gru_flops_per_seq(steps, best_h)
    best_err = abs(best_f - target_flops)
    for h in range(8, 129):
        f = gru_flops_per_seq(steps, h)
        err = abs(f - target_flops)
        if err < best_err:
            best_h, best_f, best_err = h, f, err
    return best_h, best_f


def match_transformer_dims(target_flops: int, steps: int = 12) -> tuple[int, int, int, int]:
    best = (32, 4, 32, transformer_flops_per_seq(steps, 32, 32))
    best_err = abs(best[3] - target_flops)
    for d_model in (16, 20, 24, 28, 32, 36, 40, 48):
        nhead = 4
        if d_model % nhead != 0:
            continue
        for dim_ff in range(16, 129, 4):
            f = transformer_flops_per_seq(steps, d_model, dim_ff)
            err = abs(f - target_flops)
            if err < best_err:
                best = (d_model, nhead, dim_ff, f)
                best_err = err
    return best


class _GRUSequenceBaseline(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int = 4, readout_hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(48, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, readout_hidden),
            nn.Tanh(),
            nn.Linear(readout_hidden, output_dim),
        )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(ticks)
        return cast(torch.Tensor, self.head(out[:, -1, :]))


class _CausalTransformerBaseline(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        max_len: int = 12,
        output_dim: int = 4,
        readout_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(48, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.05)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, readout_hidden),
            nn.Tanh(),
            nn.Linear(readout_hidden, output_dim),
        )

    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        steps = ticks.shape[1]
        x = self.proj(ticks) + self.pos[:steps].unsqueeze(0)
        mask = torch.triu(
            torch.full((steps, steps), float("-inf"), device=ticks.device), diagonal=1
        )
        out = self.layer(x, src_mask=mask)
        return cast(torch.Tensor, self.head(out[:, -1, :]))


def run_bridge_1_equal_flops(seed: int, base_graph: Any) -> dict[str, Any]:
    """Bridge 1: Equal-FLOPs comparison on M3 T=12 across Selective, GRU, Transformer."""
    steps = 1000
    horizon = 12
    fit_data = _make_m3_data(FIT, seed + 9000 + FIT_OFFSET, horizon=horizon)
    sel_flops = selective_flops_per_seq(
        horizon, groups=8, n_units=32, readout_hidden=64, output_dim=4
    )
    gru_h, gru_flops = match_gru_hidden(sel_flops, steps=horizon)
    tf_d, tf_heads, tf_ff, tf_flops = match_transformer_dims(sel_flops, steps=horizon)

    loss_fn = nn.CrossEntropyLoss()

    # Arm 1: SelectivePersistentGraph (N=32)
    t0 = time.process_time()
    torch.manual_seed(seed + 11)
    spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=4,
        output_dim=4,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 11,
    )
    sel_graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    sel_graph.freeze_base_()
    opt_sel = torch.optim.Adam(_trainable_params(sel_graph), lr=7e-3, weight_decay=1e-4)
    for step in range(steps):
        batch = _make_m3_data(BATCH, seed * 10000 + step, horizon=horizon)
        gates = torch.ones(BATCH, horizon, spec.n_units)
        out = sel_graph(
            _channels_seq(batch["ticks"]), gates, sel_graph.initial_state(batch_size=BATCH)
        )
        loss = loss_fn(out["readout"], batch["labels"])
        _clip_and_step(opt_sel, sel_graph, loss)
    sel_preds = _eval_selective_preds(sel_graph, fit_data["ticks"])
    sel_acc = float((sel_preds == fit_data["labels"]).float().mean())
    sel_cpu = round(time.process_time() - t0, 3)

    # Arm 2: Equal-FLOPs GRU
    t1 = time.process_time()
    torch.manual_seed(seed + 12)
    gru = _GRUSequenceBaseline(gru_h, output_dim=4, readout_hidden=64)
    opt_gru = torch.optim.Adam(gru.parameters(), lr=7e-3, weight_decay=1e-4)
    for step in range(steps):
        batch = _make_m3_data(BATCH, seed * 10000 + step, horizon=horizon)
        loss = loss_fn(gru(batch["ticks"]), batch["labels"])
        opt_gru.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(gru.parameters(), 1.0)
        opt_gru.step()
    with torch.no_grad():
        gru_acc = float((gru(fit_data["ticks"]).argmax(-1) == fit_data["labels"]).float().mean())
    gru_cpu = round(time.process_time() - t1, 3)

    # Arm 3: Equal-FLOPs Causal Transformer
    t2 = time.process_time()
    torch.manual_seed(seed + 13)
    tf = _CausalTransformerBaseline(
        tf_d, tf_heads, tf_ff, max_len=horizon, output_dim=4, readout_hidden=64
    )
    opt_tf = torch.optim.Adam(tf.parameters(), lr=4e-3, weight_decay=1e-4)
    for step in range(steps):
        batch = _make_m3_data(BATCH, seed * 10000 + step, horizon=horizon)
        loss = loss_fn(tf(batch["ticks"]), batch["labels"])
        opt_tf.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
        opt_tf.step()
    with torch.no_grad():
        tf_acc = float((tf(fit_data["ticks"]).argmax(-1) == fit_data["labels"]).float().mean())
    tf_cpu = round(time.process_time() - t2, 3)

    return {
        "selective": {"acc": sel_acc, "flops": sel_flops, "cpu": sel_cpu},
        "gru": {
            "acc": gru_acc,
            "hidden": gru_h,
            "flops": gru_flops,
            "ratio": gru_flops / sel_flops,
            "cpu": gru_cpu,
        },
        "transformer": {
            "acc": tf_acc,
            "d_model": tf_d,
            "nhead": tf_heads,
            "dim_ff": tf_ff,
            "flops": tf_flops,
            "ratio": tf_flops / sel_flops,
            "cpu": tf_cpu,
        },
    }


# ---------------------------------------------------------------------------
# Bridge 2: 2-Gen Neutral Growth + Equal-Step Controls + H2 Lesion
# ---------------------------------------------------------------------------


def _train_cohort_m3_segment(
    graph: SelectivePersistentGraph,
    seed: int,
    step_start: int,
    num_steps: int,
    *,
    lr: float = 7e-3,
    immature_aux_weight: float = 0.5,
) -> None:
    loss_fn = nn.CrossEntropyLoss()
    params = _trainable_params(graph)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    n_units = graph.selective_units.n_units
    for offset in range(num_steps):
        batch = _make_m3_data(BATCH, seed * 20000 + step_start + offset, horizon=12)
        gates = torch.ones(BATCH, 12, n_units)
        out = graph(_channels_seq(batch["ticks"]), gates, graph.initial_state(batch_size=BATCH))
        loss = loss_fn(out["readout"], batch["labels"])
        if immature_aux_weight > 0.0:
            for aux_logits in out.get("immature_cohort_logits", []):
                loss = loss + immature_aux_weight * loss_fn(aux_logits, batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        graph.project_all_parameters_()


def _newborn_cohort_step1_grad(
    graph: SelectivePersistentGraph,
    cohort_index: int,
    fit_ticks: torch.Tensor,
    fit_labels: torch.Tensor,
) -> float:
    n_units = graph.selective_units.n_units
    gates = torch.ones(fit_ticks.shape[0], fit_ticks.shape[1], n_units)
    out = graph(_channels_seq(fit_ticks), gates, graph.initial_state(batch_size=fit_ticks.shape[0]))
    loss = F.cross_entropy(out["readout"], fit_labels)
    graph.zero_grad(set_to_none=True)
    loss.backward()
    cohort = cast(_AntitheticSelectiveCohort, graph.growth_cohorts[cohort_index])
    assert cohort.pos_units.r.grad is not None
    assert cohort.pos_units.write_in.weight.grad is not None
    r_grad = float(cohort.pos_units.r.grad.abs().max())
    w_grad = float(cohort.pos_units.write_in.weight.grad.abs().max())
    graph.zero_grad(set_to_none=True)
    return max(r_grad, w_grad)


def run_bridge_2_growth_and_lesion(
    seed: int, base_graph: Any, skill_data: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Bridge 2: 2-gen growth (4->12->28 functional / 52 total units), equal-step G5/G6, H2 lesion."""
    fit_data = _make_m3_data(FIT, seed + 19000 + FIT_OFFSET, horizon=12)
    base_before = _skill_scores(SelectivePersistentGraph(base_graph), skill_data)

    # Lineage: Gen 0 (N=4, 400 steps) -> Gen 1 (+8 functional / 16 twin, 600 steps) -> Gen 2 (+16 functional / 32 twin, 900 steps; 28 functional / 52 total units)
    spec_g0 = SelectiveUnitSpec(
        groups=1,
        units_per_group=4,
        output_dim=4,
        readout_hidden=16,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 21,
    )
    grown = SelectivePersistentGraph(copy.deepcopy(base_graph), spec_g0)
    grown.freeze_base_()
    _train_cohort_m3_segment(grown, seed, 0, 400)
    with torch.no_grad():
        g0_acc = float(
            (_eval_selective_preds(grown, fit_data["ticks"]) == fit_data["labels"]).float().mean()
        )
        pre_g1_out = grown(
            _channels_seq(fit_data["ticks"][:32]),
            torch.ones(32, 12, 4),
            grown.initial_state(batch_size=32),
        )["readout"]

    c1 = grown.grow_antithetic_cohort_(
        groups=4, units_per_group=2, readout_hidden=32, freeze_mature=True, seed=seed + 101
    )
    with torch.no_grad():
        post_g1_out = grown(
            _channels_seq(fit_data["ticks"][:32]),
            torch.ones(32, 12, 4),
            grown.initial_state(batch_size=32),
        )["readout"]
    birth_diff_1 = float((pre_g1_out - post_g1_out).abs().max())
    grad_g1 = _newborn_cohort_step1_grad(grown, c1, fit_data["ticks"][:32], fit_data["labels"][:32])

    _train_cohort_m3_segment(grown, seed, 400, 600)
    grown.promote_cohorts_to_mature_()
    with torch.no_grad():
        g1_acc = float(
            (_eval_selective_preds(grown, fit_data["ticks"]) == fit_data["labels"]).float().mean()
        )
        pre_g2_out = grown(
            _channels_seq(fit_data["ticks"][:32]),
            torch.ones(32, 12, 4),
            grown.initial_state(batch_size=32),
        )["readout"]

    c2 = grown.grow_antithetic_cohort_(
        groups=4, units_per_group=4, readout_hidden=48, freeze_mature=True, seed=seed + 202
    )
    with torch.no_grad():
        post_g2_out = grown(
            _channels_seq(fit_data["ticks"][:32]),
            torch.ones(32, 12, 4),
            grown.initial_state(batch_size=32),
        )["readout"]
    birth_diff_2 = float((pre_g2_out - post_g2_out).abs().max())
    grad_g2 = _newborn_cohort_step1_grad(grown, c2, fit_data["ticks"][:32], fit_data["labels"][:32])

    _train_cohort_m3_segment(grown, seed, 1000, 900)
    grown.promote_cohorts_to_mature_()
    gen2_acc = float(
        (_eval_selective_preds(grown, fit_data["ticks"]) == fit_data["labels"]).float().mean()
    )

    # H2 Mature Lesion: mark both grown cohorts dormant, then restore mature
    grown.set_cohort_lifecycle_(c1, "dormant")
    grown.set_cohort_lifecycle_(c2, "dormant")
    lesioned_acc = float(
        (_eval_selective_preds(grown, fit_data["ticks"]) == fit_data["labels"]).float().mean()
    )
    grown.set_cohort_lifecycle_(c1, "mature")
    grown.set_cohort_lifecycle_(c2, "mature")
    restored_acc = float(
        (_eval_selective_preds(grown, fit_data["ticks"]) == fit_data["labels"]).float().mean()
    )
    lesion_drop = gen2_acc - lesioned_acc

    # Retention & multi-generation capsule_v2 check
    after_scores = _skill_scores(grown, skill_data)
    old_skill_damage = max(max(0.0, base_before[k] - after_scores[k]) for k in base_before)

    cap = build_capsule_v2(grown, clocks=(4, 2))
    with tempfile.TemporaryDirectory() as tmpdir:
        cap_path = Path(tmpdir) / f"grown_cap_{seed}.pt"
        save_capsule_v2(cap, cap_path)
        loaded = load_capsule_v2(cap_path)
    rebuilt = SelectivePersistentGraph(copy.deepcopy(base_graph))
    apply_capsule_v2(rebuilt, loaded)
    with torch.no_grad():
        gates_full = torch.ones(FIT, 12, 4)
        ro_a = grown(
            _channels_seq(fit_data["ticks"]), gates_full, grown.initial_state(batch_size=FIT)
        )["readout"]
        ro_b = rebuilt(
            _channels_seq(fit_data["ticks"]), gates_full, rebuilt.initial_state(batch_size=FIT)
        )["readout"]
    capsule_diff = float((ro_a - ro_b).abs().max())

    # Equal-step control G5_fixed_n4 (N=4, 1,900 steps)
    g5 = SelectivePersistentGraph(copy.deepcopy(base_graph), spec_g0)
    g5.freeze_base_()
    _train_cohort_m3_segment(g5, seed, 0, 1900)
    g5_acc = float(
        (_eval_selective_preds(g5, fit_data["ticks"]) == fit_data["labels"]).float().mean()
    )

    # Equal-step control G6_prealloc_n32 (N=32, 1,900 steps)
    spec_g6 = SelectiveUnitSpec(
        groups=4,
        units_per_group=8,
        output_dim=4,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 29,
    )
    g6 = SelectivePersistentGraph(copy.deepcopy(base_graph), spec_g6)
    g6.freeze_base_()
    _train_cohort_m3_segment(g6, seed, 0, 1900)
    g6_acc = float(
        (_eval_selective_preds(g6, fit_data["ticks"]) == fit_data["labels"]).float().mean()
    )

    return {
        "g0_acc": g0_acc,
        "g1_acc": g1_acc,
        "gen2_acc": gen2_acc,
        "g5_fixed_n4_acc": g5_acc,
        "g6_prealloc_n32_acc": g6_acc,
        "total_selective_units": grown.total_selective_units,
        "birth_diff_max": max(birth_diff_1, birth_diff_2),
        "newborn_step1_grad_min": min(grad_g1, grad_g2),
        "lesioned_acc": lesioned_acc,
        "lesion_drop": lesion_drop,
        "restored_diff": abs(gen2_acc - restored_acc),
        "old_skill_damage": old_skill_damage,
        "capsule_v2_max_abs_diff": capsule_diff,
    }


# ---------------------------------------------------------------------------
# Bridge 3: Shift-Invariant Causal Selective Reading on A3c + Bidirectional Chinese
# ---------------------------------------------------------------------------


def _bi_char_tick(char: str) -> torch.Tensor:
    tick = torch.zeros(48, dtype=torch.float32)
    index = BI_CN_CHAR_INDEX.get(char)
    if index is not None:
        one_hot = torch.zeros(len(BI_CN_VOCAB), dtype=torch.float32)
        one_hot[index] = 1.0
        tick[:8] = one_hot @ BI_CN_PROJECTION
    return tick


def _build_bidirectional_cn(
    count: int,
    seed: int,
    first_templates: tuple[str, ...],
    second_templates: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    all_tpls = [
        *[(tpl, True) for tpl in first_templates],
        *[(tpl, False) for tpl in second_templates],
    ]
    rows_ticks: list[torch.Tensor] = []
    rows_labels: list[int] = []
    for idx in range(count):
        generator = torch.Generator().manual_seed(seed + idx)
        first = int(torch.randint(0, CN_CLASSES, (1,), generator=generator))
        second = int(torch.randint(0, CN_CLASSES - 1, (1,), generator=generator))
        if second >= first:
            second += 1
        tpl, pick_first = all_tpls[idx % len(all_tpls)]
        text = tpl.format(c1=CN_COLOURS[first], c2=CN_COLOURS[second])
        ticks = [_bi_char_tick(ch) for ch in text[:CN_MAX_LEN]]
        while len(ticks) < CN_MAX_LEN:
            ticks.append(torch.zeros(48, dtype=torch.float32))
        rows_ticks.append(torch.stack(ticks))
        rows_labels.append(first if pick_first else second)
    return {"ticks": torch.stack(rows_ticks), "labels": torch.tensor(rows_labels, dtype=torch.long)}


def _train_cn_selective_graph(
    base_graph: Any,
    train_data: dict[str, torch.Tensor],
    seed: int,
    *,
    steps: int,
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
        out = graph(_channels_seq(seq), gates, graph.initial_state(batch_size=BATCH))
        loss = loss_fn(out["readout"], train_data["labels"][idx])
        _clip_and_step(optimizer, graph, loss)
    return graph


def _eval_cn_selective(graph: SelectivePersistentGraph, split: dict[str, torch.Tensor]) -> float:
    preds = _eval_selective_preds(graph, split["ticks"])
    return float((preds == split["labels"]).float().mean())


def _train_and_eval_pos_attn_raw(
    train_data: dict[str, torch.Tensor],
    fit_data: dict[str, torch.Tensor],
    seed: int,
    *,
    steps: int = 1800,
) -> float:
    torch.manual_seed(seed + 77)
    head = PosAttentionRead(train_data["ticks"].shape[-1], CN_CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 79)
    n_train = train_data["ticks"].shape[0]
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps)
        idx = torch.randint(0, n_train, (BATCH,), generator=generator)
        loss = loss_fn(head(train_data["ticks"][idx]), train_data["labels"][idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = head(fit_data["ticks"]).argmax(dim=-1)
    return float((preds == fit_data["labels"]).float().mean())


def run_bridge_3_a3c_chinese(seed: int, base_graph: Any) -> dict[str, Any]:
    """Bridge 3: exact A3c and unconfounded 2-rule bidirectional Chinese + adversarial scope probe."""
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
    unseen_bi = sorted(fit_chars - train_chars)

    a3c_train = build_cn_vocab3(CN_TRAIN, seed, CN_TRAIN_TEMPLATES)
    a3c_fit = build_cn_vocab3(CN_FIT, seed + FIT_OFFSET, CN_FIT_TEMPLATES)
    a3c_pos_raw = _train_and_eval_pos_attn_raw(a3c_train, a3c_fit, seed, steps=1200)
    a3c_graph = _train_cn_selective_graph(base_graph, a3c_train, seed + 51, steps=1100, lr=6e-3)
    a3c_sel = _eval_cn_selective(a3c_graph, a3c_fit)

    bi_train = _build_bidirectional_cn(1600, seed + 600, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
    bi_fit = _build_bidirectional_cn(
        CN_FIT, seed + 600 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    adv_fit = _build_bidirectional_cn(
        CN_FIT, seed + 700 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    bi_pos_raw = _train_and_eval_pos_attn_raw(bi_train, bi_fit, seed + 61, steps=1400)
    bi_graph = _train_cn_selective_graph(base_graph, bi_train, seed + 62, steps=1500, lr=6e-3)
    bi_sel = _eval_cn_selective(bi_graph, bi_fit)
    adv_sel = _eval_cn_selective(bi_graph, adv_fit)

    return {
        "a3c_exact": {"pos_attn_raw": a3c_pos_raw, "selective_causal": a3c_sel},
        "a3c_bidirectional": {
            "pos_attn_raw": bi_pos_raw,
            "selective_causal": bi_sel,
            "adversarial_scope_acc": adv_sel,
            "unseen_fit_chars": unseen_bi,
        },
    }


# ---------------------------------------------------------------------------
# Bridge 4: Closed-Loop 2-Turn Action + Binary Right/Wrong Feedback (Zero Leakage)
# ---------------------------------------------------------------------------


def _ambiguous_action_episode(seed: int, template_str: str) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    c_target = int(torch.randint(0, 4, (1,), generator=generator))
    other_colours = [c for c in range(4) if c != c_target]
    c_other = other_colours[int(torch.randint(0, 3, (1,), generator=generator))]
    perm_slots = torch.randperm(3, generator=generator).tolist()
    s_a, s_b, s_d = perm_slots[0], perm_slots[1], perm_slots[2]
    true_slot = s_a if int(torch.randint(0, 2, (1,), generator=generator)) == 0 else s_b
    slot_colours = [0, 0, 0]
    slot_colours[s_a] = c_target
    slot_colours[s_b] = c_target
    slot_colours[s_d] = c_other
    positions = torch.randint(0, 4, (3,), generator=generator).tolist()
    dir_idx = int(torch.randint(0, 4, (1,), generator=generator))
    direction = ACTION_DIRECTIONS[dir_idx]

    turn0_ticks = torch.zeros(4, 48, dtype=torch.float32)
    for slot in range(3):
        turn0_ticks[slot, slot_colours[slot]] = 1.0
        turn0_ticks[slot, 6 + positions[slot]] = 1.0
    text = template_str.format(c=ACTION_COLOURS[c_target], d=direction)
    bag = torch.zeros(len(ACTION_VOCAB), dtype=torch.float32)
    for ch in text:
        if ch in ACTION_CHAR_INDEX:
            bag[ACTION_CHAR_INDEX[ch]] += 1.0
    turn0_ticks[3, 24:40] = bag @ ACTION_PROJECTION
    label = true_slot * 4 + dir_idx
    return {"turn0_ticks": turn0_ticks, "label": label}


def _build_ambiguous_split(
    count: int, seed: int, templates: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    rows = [
        _ambiguous_action_episode(seed + idx, templates[idx % len(templates)])
        for idx in range(count)
    ]
    return {
        "turn0_ticks": torch.stack([r["turn0_ticks"] for r in rows]),
        "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
    }


def _make_turn1_feedback_tick(
    actions: torch.Tensor, is_correct: torch.Tensor, *, mode: str = "real", seed: int = 0
) -> torch.Tensor:
    """Build single feedback tick [B, 1, 48]: proprio[0:3]=slot, proprio[3:7]=dir, feedback[0:2]=right/wrong."""
    batch_size = actions.shape[0]
    tick = torch.zeros(batch_size, 1, 48, dtype=torch.float32)
    slots = actions // 4
    dirs = actions % 4
    row_idx = torch.arange(batch_size)
    tick[row_idx, 0, 16 + slots] = 1.0  # proprio 16..18
    tick[row_idx, 0, 19 + dirs] = 1.0  # proprio 19..22

    if mode == "real":
        tick[row_idx, 0, 40] = is_correct.float()
        tick[row_idx, 0, 41] = (~is_correct).float()
    elif mode == "zero":
        pass
    elif mode == "permuted":
        gen = torch.Generator().manual_seed(seed + 888)
        perm = torch.randperm(batch_size, generator=gen)
        tick[row_idx, 0, 40] = is_correct[perm].float()
        tick[row_idx, 0, 41] = (~is_correct[perm]).float()
    else:
        raise ValueError(f"Unknown feedback mode: {mode!r}")
    return tick


def run_bridge_4_closed_loop_feedback(seed: int, base_graph: Any) -> dict[str, float]:
    """Bridge 4: 2-turn ambiguous action task with binary right/wrong feedback and ablations."""
    train_data = _build_ambiguous_split(1024, seed + 15000, ACTION_TRAIN_TEMPLATES)
    fit_data = _build_ambiguous_split(FIT, seed + 15000 + FIT_OFFSET, ACTION_FIT_TEMPLATES)

    spec = SelectiveUnitSpec(
        groups=6,
        units_per_group=5,
        output_dim=ACTION_CLASSES,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-1.5,
        seed=seed + 41,
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base_graph), spec)
    graph.freeze_base_()
    optimizer = torch.optim.Adam(_trainable_params(graph), lr=5e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 43)
    steps = 1200
    for step in range(steps):
        cur_lr = max(5e-4, lr_at(step, steps) * (5e-3 / 3e-3))
        for group in optimizer.param_groups:
            group["lr"] = min(5e-3, cur_lr)
        idx = torch.randint(0, train_data["turn0_ticks"].shape[0], (BATCH,), generator=generator)
        t0_seq = train_data["turn0_ticks"][idx]
        labels = train_data["labels"][idx]
        gates_0 = torch.ones(BATCH, 4, spec.n_units)
        out_0 = graph(_channels_seq(t0_seq), gates_0, graph.initial_state(batch_size=BATCH))
        logits_0 = out_0["readout"]
        a_0 = logits_0.argmax(dim=-1).detach()
        is_correct = a_0 == labels

        t1_tick = _make_turn1_feedback_tick(a_0, is_correct, mode="real")
        gates_1 = torch.ones(BATCH, 1, spec.n_units)
        out_1 = graph(_channels_seq(t1_tick), gates_1, out_0["next_state"])
        logits_1 = out_1["readout"]
        loss = loss_fn(logits_0, labels) + 1.5 * loss_fn(logits_1, labels)
        _clip_and_step(optimizer, graph, loss)

    with torch.no_grad():
        fit_seq = fit_data["turn0_ticks"]
        fit_labels = fit_data["labels"]
        out0_fit = graph(
            _channels_seq(fit_seq),
            torch.ones(FIT, 4, spec.n_units),
            graph.initial_state(batch_size=FIT),
        )
        a0_fit = out0_fit["readout"].argmax(dim=-1)
        correct_0 = a0_fit == fit_labels
        turn0_acc = float(correct_0.float().mean())

        accs: dict[str, float] = {"turn0_acc": turn0_acc}
        for mode_name, mode_key in (
            ("turn1_with_feedback", "real"),
            ("turn1_no_feedback", "zero"),
            ("turn1_permuted_feedback", "permuted"),
        ):
            fb_tick = _make_turn1_feedback_tick(a0_fit, correct_0, mode=mode_key, seed=seed)
            out1_fit = graph(
                _channels_seq(fb_tick),
                torch.ones(FIT, 1, spec.n_units),
                out0_fit["next_state"],
            )
            preds_1 = out1_fit["readout"].argmax(dim=-1)
            accs[mode_name] = float((preds_1 == fit_labels).float().mean())
            if mode_key == "real":
                wrong_mask = ~correct_0
                accs["error_correction_rate"] = (
                    float((preds_1[wrong_mask] == fit_labels[wrong_mask]).float().mean())
                    if bool(wrong_mask.any())
                    else 1.0
                )
    return accs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    started = time.process_time()
    torch.set_num_threads(1)

    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        if time.process_time() - started > CAP_CPU:
            raise RuntimeError("selective_bridge_budget_exceeded")
        base_graph, skill_data = train_joint_parent(seed)
        rows.append(
            {
                "seed": seed,
                "bridge1_equal_flops": run_bridge_1_equal_flops(seed, base_graph),
                "bridge2_growth_lesion": run_bridge_2_growth_and_lesion(
                    seed, base_graph, skill_data
                ),
                "bridge3_a3c_chinese": run_bridge_3_a3c_chinese(seed, base_graph),
                "bridge4_action_feedback": run_bridge_4_closed_loop_feedback(seed, base_graph),
            }
        )

    summary: dict[str, Any] = {
        "bridge1_equal_flops": {
            "selective_acc": _mean([r["bridge1_equal_flops"]["selective"]["acc"] for r in rows]),
            "gru_acc": _mean([r["bridge1_equal_flops"]["gru"]["acc"] for r in rows]),
            "transformer_acc": _mean(
                [r["bridge1_equal_flops"]["transformer"]["acc"] for r in rows]
            ),
            "selective_flops": rows[0]["bridge1_equal_flops"]["selective"]["flops"],
            "gru_flops": rows[0]["bridge1_equal_flops"]["gru"]["flops"],
            "gru_flops_ratio": rows[0]["bridge1_equal_flops"]["gru"]["ratio"],
            "transformer_flops": rows[0]["bridge1_equal_flops"]["transformer"]["flops"],
            "transformer_flops_ratio": rows[0]["bridge1_equal_flops"]["transformer"]["ratio"],
        },
        "bridge2_growth_lesion": {
            "g0_acc": _mean([r["bridge2_growth_lesion"]["g0_acc"] for r in rows]),
            "g1_acc": _mean([r["bridge2_growth_lesion"]["g1_acc"] for r in rows]),
            "gen2_acc": _mean([r["bridge2_growth_lesion"]["gen2_acc"] for r in rows]),
            "g5_fixed_n4_acc": _mean([r["bridge2_growth_lesion"]["g5_fixed_n4_acc"] for r in rows]),
            "g6_prealloc_n32_acc": _mean(
                [r["bridge2_growth_lesion"]["g6_prealloc_n32_acc"] for r in rows]
            ),
            "birth_diff_max": max(r["bridge2_growth_lesion"]["birth_diff_max"] for r in rows),
            "newborn_step1_grad_min": min(
                r["bridge2_growth_lesion"]["newborn_step1_grad_min"] for r in rows
            ),
            "lesioned_acc": _mean([r["bridge2_growth_lesion"]["lesioned_acc"] for r in rows]),
            "lesion_drop": _mean([r["bridge2_growth_lesion"]["lesion_drop"] for r in rows]),
            "restored_diff_max": max(r["bridge2_growth_lesion"]["restored_diff"] for r in rows),
            "old_skill_damage_max": max(
                r["bridge2_growth_lesion"]["old_skill_damage"] for r in rows
            ),
            "capsule_v2_max_abs_diff": max(
                r["bridge2_growth_lesion"]["capsule_v2_max_abs_diff"] for r in rows
            ),
        },
        "bridge3_a3c_chinese": {
            "a3c_exact_pos_attn_raw": _mean(
                [r["bridge3_a3c_chinese"]["a3c_exact"]["pos_attn_raw"] for r in rows]
            ),
            "a3c_exact_selective_causal": _mean(
                [r["bridge3_a3c_chinese"]["a3c_exact"]["selective_causal"] for r in rows]
            ),
            "a3c_bi_pos_attn_raw": _mean(
                [r["bridge3_a3c_chinese"]["a3c_bidirectional"]["pos_attn_raw"] for r in rows]
            ),
            "a3c_bi_selective_causal": _mean(
                [r["bridge3_a3c_chinese"]["a3c_bidirectional"]["selective_causal"] for r in rows]
            ),
            "a3c_bi_adversarial_scope": _mean(
                [
                    r["bridge3_a3c_chinese"]["a3c_bidirectional"]["adversarial_scope_acc"]
                    for r in rows
                ]
            ),
            "a3c_bi_unseen_chars": rows[0]["bridge3_a3c_chinese"]["a3c_bidirectional"][
                "unseen_fit_chars"
            ],
        },
        "bridge4_action_feedback": {
            k: _mean([r["bridge4_action_feedback"][k] for r in rows])
            for k in (
                "turn0_acc",
                "turn1_with_feedback",
                "turn1_no_feedback",
                "turn1_permuted_feedback",
                "error_correction_rate",
            )
        },
    }

    b1 = summary["bridge1_equal_flops"]
    b2 = summary["bridge2_growth_lesion"]
    b3 = summary["bridge3_a3c_chinese"]
    b4 = summary["bridge4_action_feedback"]

    gates = {
        "b1_flops_parity": (
            0.90 <= b1["gru_flops_ratio"] <= 1.10 and 0.90 <= b1["transformer_flops_ratio"] <= 1.10
        ),
        "b1_selective_ge_090": b1["selective_acc"] >= 0.90,
        "b1_non_inferior_to_gru": b1["selective_acc"] >= b1["gru_acc"] - 0.03,
        "b2_birth_neutral": b2["birth_diff_max"] <= 1e-5,
        "b2_newborn_alive": b2["newborn_step1_grad_min"] > 0.0,
        "b2_growth_beats_fixed_n4": b2["gen2_acc"] >= b2["g5_fixed_n4_acc"] + 0.20,
        "b2_gen2_ge_090": b2["gen2_acc"] >= 0.90,
        "b2_lesion_causal": (
            b2["lesion_drop"] >= 0.50 * max(0.0, b2["gen2_acc"] - b2["g5_fixed_n4_acc"])
            and b2["lesion_drop"] >= 0.15
        ),
        "b2_old_skill_damage_zero": b2["old_skill_damage_max"] == 0.0,
        "b2_capsule_v2_exact": b2["capsule_v2_max_abs_diff"] <= 1e-10,
        "b3_a3c_exact_ge_080": b3["a3c_exact_selective_causal"] >= 0.80,
        "b3_beats_pos_attn": b3["a3c_exact_selective_causal"]
        >= b3["a3c_exact_pos_attn_raw"] + 0.15,
        "b3_bidirectional_ge_080": b3["a3c_bi_selective_causal"] >= 0.80,
        "b4_turn1_ge_085": b4["turn1_with_feedback"] >= 0.85,
        "b4_gain_over_no_fb": b4["turn1_with_feedback"] - b4["turn1_no_feedback"] >= 0.25,
        "b4_gain_over_permuted_fb": b4["turn1_with_feedback"] - b4["turn1_permuted_feedback"]
        >= 0.25,
        "b4_correction_rate_ge_080": b4["error_correction_rate"] >= 0.80,
    }
    verdict = "selective_bridge_verified" if all(gates.values()) else "selective_bridge_partial"

    payload: dict[str, Any] = {
        "schema": "taiji-u2-selective-bridge-v1",
        "card": "D-20261121-227 Four-axis capability bridge on SelectivePersistentGraph",
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
            "m3_steps": 1000,
            "growth_steps": "400+600+900=1900",
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
    return 0 if verdict == "selective_bridge_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
