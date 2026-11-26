"""S1-R restructure: 48-unit subject + per-phase readout heads + developmental growth (0 check).

Diagnosis (D-20261121-233) showed P-B is capacity/generalization bound and plain growth leaves
new units unused.  S1-R keeps ONE subject (48 units, shared selective state) but gives each
phase its own small readout head over [z, a], so tasks do not fight over one readout; growth
re-initialises the new-unit head columns and trains the newborn units through the P-D head, so
lesion can actually measure their contribution.  Same 16 gates as S1, <=1,500 CPU, 0 check.
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
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
)

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_ADV_FIRST,
    BI_CN_ADV_SECOND,
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_m1_long_gap_data,
    _make_p24_stream,
    _make_p26_batch,
    _make_p27_physics_batch,
)
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_s1r2_rework_20261121.json"
CAP_CPU = 1000.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 160
UNITS_GROUPS = 8
UNITS_PER_GROUP = 6
STEPS_A = 800
STEPS_B = 1000
STEPS_C = 500
STEPS_D = 400
STEPS_GROW = 300
STEPS_STAB = 80
REPLAY_FRAC = 0.5


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _step(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
    optimizer.step()


class PhaseHead(nn.Module):
    def __init__(self, state_dim: int, out_dim: int, width: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, width), nn.Tanh(), nn.Dropout(0.2), nn.Linear(width, out_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class SubjectR:
    def __init__(self, seed: int) -> None:
        base, _ = train_joint_parent(seed)
        spec = SelectiveUnitSpec(
            groups=UNITS_GROUPS,
            units_per_group=UNITS_PER_GROUP,
            output_dim=13,
            readout_hidden=64,
            per_sample_state=True,
            selective_gate=True,
            init_write_bias=-2.0,
            seed=seed + 1000,
        )
        self.graph = SelectivePersistentGraph(copy.deepcopy(base), spec)
        self.graph.freeze_base_()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1017)
            self.graph._build_selective_branch(spec)
        self.n_units = spec.n_units
        self.phase_heads: dict[str, PhaseHead] = {}
        self._make_head("cn", 8, seed, width=128)
        self._make_head("p24", 5, seed, width=128)
        self._make_head("p26", 8, seed)
        self._make_head("p27", 4, seed)
        self.replay: list[dict[str, Any]] = []

    def _make_head(self, name: str, out_dim: int, seed: int, width: int = 64) -> None:
        head = PhaseHead(2 * self.n_units, out_dim, width)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 2000 + len(self.phase_heads))
            for parameter in head.parameters():
                nn.init.normal_(parameter, std=0.05)
        self.phase_heads[name] = head

    def state_of(self, ticks: torch.Tensor, step_mask: torch.Tensor | None = None) -> torch.Tensor:
        out = self.graph(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], self.n_units),
            self.graph.initial_state(batch_size=ticks.shape[0]),
            step_mask=step_mask,
        )
        return out["next_state"]["z"], out["next_state"]["a"]

    def logits(
        self, name: str, ticks: torch.Tensor, step_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        z, a = self.state_of(ticks, step_mask)
        head = self.phase_heads[name]
        width = head.net[0].in_features
        return head(torch.cat([z, a], dim=-1)[..., :width])

    def params(self, extra: list[nn.Parameter] | None = None) -> list[nn.Parameter]:
        values = [p for p in self.graph.parameters() if p.requires_grad]
        for head in self.phase_heads.values():
            values.extend([p for p in head.parameters() if p.requires_grad])
        if extra:
            values.extend(extra)
        return values

    def train_mode(self) -> None:
        for head in self.phase_heads.values():
            head.train()

    def eval_mode(self) -> None:
        for head in self.phase_heads.values():
            head.eval()


def _loss_cn(
    subject: SubjectR, batch: dict[str, torch.Tensor], rng: torch.Generator
) -> torch.Tensor:
    idx = torch.randint(0, batch["ticks"].shape[0], (BATCH,), generator=rng)
    seq = batch["ticks"][idx]
    logits = subject.logits("cn", seq, step_mask=(seq.abs().sum(-1) > 0).float())
    return F.cross_entropy(logits, batch["labels"][idx])


def _loss_p24(
    subject: SubjectR, batch: dict[str, torch.Tensor], rng: torch.Generator
) -> torch.Tensor:
    idx = torch.randint(0, batch["ticks"].shape[0], (BATCH,), generator=rng)
    logits = subject.logits("p24", batch["ticks"][idx])
    return F.cross_entropy(
        logits[:, :4], batch["val_labels"][idx]
    ) + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"][idx])


def _loss_p27(
    subject: SubjectR, batch: dict[str, torch.Tensor], rng: torch.Generator
) -> torch.Tensor:
    idx = torch.randint(0, batch["ticks"].shape[0], (BATCH,), generator=rng)
    return F.cross_entropy(subject.logits("p27", batch["ticks"][idx]), batch["labels"][idx])


def _replay_loss(subject: SubjectR, rng: torch.Generator) -> torch.Tensor:
    if not subject.replay:
        return torch.zeros(())
    item = subject.replay[int(torch.randint(0, len(subject.replay), (1,), generator=rng))]
    if item["kind"] == "cn":
        return _loss_cn(subject, item, rng)
    if item["kind"] == "p24":
        return _loss_p24(subject, item, rng)
    return _loss_p27(subject, item, rng)


def _eval_cn(subject: SubjectR, fit: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = subject.logits(
            "cn", fit["ticks"], step_mask=(fit["ticks"].abs().sum(-1) > 0).float()
        )
    return float((logits.argmax(-1) == fit["labels"]).float().mean())


def phase_a(subject: SubjectR, seed: int) -> dict[str, float]:
    subject.train_mode()
    train = _build_bidirectional_cn_32dim(1400, seed + 600, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)
    fit = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    adv = _build_bidirectional_cn_32dim(
        256, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    rng = torch.Generator().manual_seed(seed + 1101)
    optimizer = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    for _ in range(STEPS_A):
        loss = _loss_cn(subject, train, rng)
        _step(optimizer, loss)
    subject.replay.append({"kind": "cn", **train})
    subject.eval_mode()
    with torch.no_grad():
        adv_acc = _eval_cn(subject, adv)
        head = subject.phase_heads["cn"]
        z, a = subject.state_of(
            train["ticks"], step_mask=(train["ticks"].abs().sum(-1) > 0).float()
        )
        x = torch.cat([z, a], dim=-1)[..., : head.net[0].in_features]
        train_acc = float((head(x).argmax(-1) == train["labels"]).float().mean())
    return {
        "masked_bi_acc": _eval_cn(subject, fit),
        "masked_adv_acc": adv_acc,
        "train_acc": train_acc,
    }


def phase_b(subject: SubjectR, seed: int) -> dict[str, float]:
    subject.train_mode()
    train = _make_p24_stream(480, seed + 211)
    fit = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    snapshot_head = copy.deepcopy(subject.phase_heads["p24"])
    rng = torch.Generator().manual_seed(seed + 1201)
    optimizer = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for _ in range(STEPS_B):
        loss = _loss_p24(subject, train, rng)
        if subject.replay:
            loss = loss + REPLAY_FRAC * _replay_loss(subject, rng)
        _step(optimizer, loss)
    subject.replay.append({"kind": "p24", **train})
    subject.eval_mode()
    with torch.no_grad():
        z, a = subject.state_of(fit["ticks"])
        logits = subject.phase_heads["p24"](torch.cat([z, a], dim=-1))
        preds = logits[:, :4].argmax(-1)
        overall = float((preds == fit["val_labels"]).float().mean())
        base_logits = snapshot_head(torch.cat([z, a], dim=-1))
        unprotected = float((base_logits[:, :4].argmax(-1) == fit["val_labels"]).float().mean())
    return {"overall_acc": overall, "unprotected_acc": unprotected, "gain": overall - unprotected}


def phase_c(subject: SubjectR, seed: int) -> dict[str, float]:
    subject.train_mode()
    fit_count = FIT
    rng = torch.Generator().manual_seed(seed + 1301)
    optimizer = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    for _ in range(STEPS_C):
        batch = _make_p26_batch(
            BATCH, seed * 31000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        first = subject.graph(
            _channels_seq(batch["turn0"]),
            torch.ones(BATCH, 4, subject.n_units),
            subject.graph.initial_state(batch_size=BATCH),
        )
        ro0 = subject.phase_heads["p26"](
            torch.cat([first["next_state"]["z"], first["next_state"]["a"]], dim=-1)
        )
        second = subject.graph(
            _channels_seq(batch["reveal"]),
            torch.ones(BATCH, 1, subject.n_units),
            first["next_state"],
        )
        ro1 = subject.phase_heads["p26"](
            torch.cat([second["next_state"]["z"], second["next_state"]["a"]], dim=-1)
        )
        y = batch["labels"]
        a0 = ro0[:, :4].argmax(-1)
        a1 = ro1[:, :4].argmax(-1)
        u_act = torch.where(a0 == y, 1.0, -1.0)
        u_query = torch.where(a1 == y, 1.0, -1.0) - 0.20
        u_wait = torch.full_like(u_act, -0.05)
        utils = torch.stack([u_act, u_query, u_wait], dim=-1)
        loss = F.cross_entropy(ro0[:, :4], y)
        loss = loss + F.cross_entropy(ro1[:, :4], y)
        loss = loss + F.mse_loss(ro0[:, 4:7], utils)
        loss = loss + F.binary_cross_entropy_with_logits(ro0[:, 7], (a0 == y).float())
        if subject.replay:
            loss = loss + REPLAY_FRAC * _replay_loss(subject, rng)
        _step(optimizer, loss)

    fit = _make_p26_batch(fit_count, seed + 61 + FIT_OFFSET)
    subject.eval_mode()
    with torch.no_grad():
        first = subject.graph(
            _channels_seq(fit["turn0"]),
            torch.ones(fit_count, 4, subject.n_units),
            subject.graph.initial_state(batch_size=fit_count),
        )
        ro0 = subject.phase_heads["p26"](
            torch.cat([first["next_state"]["z"], first["next_state"]["a"]], dim=-1)
        )
        second = subject.graph(
            _channels_seq(fit["reveal"]),
            torch.ones(fit_count, 1, subject.n_units),
            first["next_state"],
        )
        ro1 = subject.phase_heads["p26"](
            torch.cat([second["next_state"]["z"], second["next_state"]["a"]], dim=-1)
        )
        policy = ro0[:, 4:7].argmax(-1)
        a0 = ro0[:, :4].argmax(-1)
        a1 = ro1[:, :4].argmax(-1)
        conf = torch.sigmoid(ro0[:, 7])
        y = fit["labels"]
        reg = fit["regimes"]
        u_act = torch.where(a0 == y, 1.0, -1.0)
        u_query = torch.where(a1 == y, 1.0, -1.0) - 0.20
        u_wait = torch.full_like(u_act, -0.05)
        realized = torch.where(policy == 0, u_act, torch.where(policy == 1, u_query, u_wait))
        return {
            "net_utility": float(realized.mean()),
            "always_act_utility": float(u_act.mean()),
            "known_act_rate": float((policy[reg == 0] == 0).float().mean()),
            "ambiguous_query_rate": float((policy[reg == 1] == 1).float().mean()),
            "unresolvable_wait_rate": float((policy[reg == 2] == 2).float().mean()),
            "brier_score": float(((conf - (a0 == y).float()) ** 2).mean()),
        }


def phase_d(subject: SubjectR, seed: int) -> dict[str, float]:
    subject.train_mode()
    rng = torch.Generator().manual_seed(seed + 1401)
    optimizer = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    last = None
    for _ in range(STEPS_D):
        last = _make_p27_physics_batch(
            BATCH, seed * 41000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        loss = _loss_p27(subject, last, rng)
        if subject.replay:
            loss = loss + REPLAY_FRAC * _replay_loss(subject, rng)
        _step(optimizer, loss)
    subject.replay.append({"kind": "p27", **last})
    fit = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)
    subject.eval_mode()
    with torch.no_grad():
        acc = float(
            (subject.logits("p27", fit["ticks"]).argmax(-1) == fit["labels"]).float().mean()
        )
        no_probe = fit["ticks"].clone()
        no_probe[:, :6, 8:11] = 0.0
        acc_np = float((subject.logits("p27", no_probe).argmax(-1) == fit["labels"]).float().mean())
    return {"accuracy": acc, "ablation_drop": acc - acc_np}


def _stabilize(subject: SubjectR, seed: int) -> None:
    rng = torch.Generator().manual_seed(seed + 1501)
    optimizer = torch.optim.Adam(subject.params(), lr=3e-3)
    for _ in range(STEPS_STAB):
        loss = _replay_loss(subject, rng)
        if float(loss.detach()) == 0.0:
            return
        _step(optimizer, loss * 2.0)


def _old_skill_scores(subject: SubjectR, seed: int) -> dict[str, float]:
    subject.eval_mode()
    cn_fit = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    p24_fit = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    with torch.no_grad():
        logits = subject.logits("p24", p24_fit["ticks"])
        p24 = float((logits[:, :4].argmax(-1) == p24_fit["val_labels"]).float().mean())
    return {"cn_masked_bi": _eval_cn(subject, cn_fit), "p24_overall": p24}


def _mask_new_unit_grads(graph, old_units: int, new_units: int) -> None:
    """Keep mature-unit parameters frozen; only newborn unit rows receive gradient."""
    for _name, parameter in graph.selective_units.named_parameters():
        if parameter.grad is None or parameter.ndim == 0:
            continue
        if parameter.shape[0] == new_units:
            parameter.grad[:old_units] = 0.0


def phase_e_growth(subject: SubjectR, seed: int) -> dict[str, float]:
    fit = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)
    subject.eval_mode()
    with torch.no_grad():
        before_logits = subject.logits("p27", fit["ticks"])
        before = before_logits.argmax(-1)
    old_units = subject.graph.selective_units.n_units
    subject.graph.grow_units_neutral_(1)
    subject.graph.grow_units_neutral_(1)
    subject.n_units = subject.graph.selective_units.n_units
    new_units = subject.n_units
    head = subject.phase_heads["p27"]
    old_net = head.net[0]
    new_net = nn.Linear(2 * new_units, old_net.out_features)
    with torch.no_grad():
        new_net.weight.zero_()
        new_net.bias.copy_(old_net.bias)
        new_net.weight[:, : 2 * old_units].copy_(old_net.weight)
    head.net[0] = new_net
    subject.train_mode()
    for parameter in subject.graph.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    for parameter in subject.graph.selective_units.parameters():
        parameter.requires_grad_(True)
    trainable = [p for p in subject.graph.selective_units.parameters() if p.requires_grad]
    trainable += [p for p in head.parameters() if p.requires_grad]
    subject.eval_mode()
    with torch.no_grad():
        z, a = subject.state_of(fit["ticks"])
        birth_logits = head(torch.cat([z, a], dim=-1)[..., : 2 * new_units])
        birth_consistency = float((birth_logits.argmax(-1) == before).float().mean())
        birth_diff = float((birth_logits - before_logits).abs().max())
    # stage 1: temporary head over the newborn-unit slice learns their input path
    temp_head = nn.Sequential(
        nn.Linear(2 * (new_units - old_units), 64), nn.Tanh(), nn.Linear(64, 4)
    )
    stage1_params = [
        p for p in subject.graph.selective_units.parameters() if p.requires_grad
    ] + list(temp_head.parameters())
    stage1_opt = torch.optim.Adam(stage1_params, lr=4e-3)
    rng = torch.Generator().manual_seed(seed + 1601)
    for _ in range(200):
        batch = _make_p27_physics_batch(
            BATCH, seed * 51000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        z, a = subject.state_of(batch["ticks"])
        x = torch.cat([z, a], dim=-1)[:, 2 * old_units :]
        loss = F.cross_entropy(temp_head(x), batch["labels"])
        stage1_opt.zero_grad(set_to_none=True)
        loss.backward()
        _mask_new_unit_grads(subject.graph, old_units, new_units)
        torch.nn.utils.clip_grad_norm_(stage1_params, 1.0)
        stage1_opt.step()
    # stage 2: zero-init p27 new columns then joint training
    with torch.no_grad():
        head.net[0].weight[:, 2 * old_units :] = 0.0
    stage2_opt = torch.optim.Adam(trainable, lr=4e-3)
    for _ in range(STEPS_GROW):
        batch = _make_p27_physics_batch(
            BATCH, seed * 52000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        loss = F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"])
        if subject.replay:
            loss = loss + REPLAY_FRAC * _replay_loss(subject, rng)
        stage2_opt.zero_grad(set_to_none=True)
        loss.backward()
        _mask_new_unit_grads(subject.graph, old_units, new_units)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        stage2_opt.step()
    subject.eval_mode()
    with torch.no_grad():
        full = subject.logits("p27", fit["ticks"])
        final_acc = float((full.argmax(-1) == fit["labels"]).float().mean())
        weight = head.net[0].weight.clone()
        head.net[0].weight[:, 2 * old_units :] = 0.0
        les = head(torch.cat(subject.state_of(fit["ticks"]), dim=-1)[..., : 2 * new_units]).argmax(
            -1
        )
        lesion_drop = final_acc - float((les == fit["labels"]).float().mean())
        head.net[0].weight.copy_(weight)
    subject.graph.selective_units.promote_immature_to_mature_()
    return {
        "final_accuracy": final_acc,
        "birth_consistency": birth_consistency,
        "birth_diff": birth_diff,
        "lesion_drop": lesion_drop,
        "units": float(new_units),
    }


def phase_f_continuity(subject: SubjectR, seed: int) -> dict[str, float]:
    batch = _make_m1_long_gap_data(32, seed + 101 + FIT_OFFSET, horizon=40, max_target_tick=20)
    ticks = batch["ticks"]
    subject.eval_mode()
    with torch.no_grad():
        full = subject.logits("cn", ticks, step_mask=(ticks.abs().sum(-1) > 0).float())
        pred_full = full.argmax(-1)
        state = subject.graph.initial_state(batch_size=ticks.shape[0])
        for t in range(20):
            out = subject.graph(
                _channels_seq(ticks[:, t : t + 1]),
                torch.ones(ticks.shape[0], 1, subject.n_units),
                state,
            )
            state = out["next_state"]
        crash_state = {k: v.clone() for k, v in state.items()}
    cap = build_capsule_v2(subject.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "capsule.json")
        loaded = load_capsule_v2(path)
        head_path = Path(tmp) / "heads.pt"
        torch.save({k: v.state_dict() for k, v in subject.phase_heads.items()}, head_path)
        heads_state = torch.load(head_path, weights_only=True)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subject.graph.base_graph), subject.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        out_fresh = fresh(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], subject.n_units),
            fresh.initial_state(batch_size=ticks.shape[0]),
        )
        fz, fa = out_fresh["next_state"]["z"], out_fresh["next_state"]["a"]
        head = PhaseHead(
            subject.phase_heads["cn"].net[0].in_features,
            8,
            subject.phase_heads["cn"].net[0].out_features,
        )
        head.load_state_dict(heads_state["cn"])
        width = head.net[0].in_features
        logits = head(torch.cat([fz, fa], dim=-1)[..., :width])
        diff = float((logits - full).abs().max())
        resume = fresh(
            _channels_seq(ticks[:, 20:]),
            torch.ones(ticks.shape[0], 20, subject.n_units),
            crash_state,
        )
        logits_resume = head(
            torch.cat([resume["next_state"]["z"], resume["next_state"]["a"]], dim=-1)[..., :width]
        )
        reconnect_acc = float((logits_resume.argmax(-1) == pred_full).float().mean())
    return {"capsule_forward_diff": diff, "reconnect_acc": reconnect_acc}


def run_seed(seed: int) -> dict[str, Any]:
    subject = SubjectR(seed)
    a = phase_a(subject, seed)
    a_after = _old_skill_scores(subject, seed)
    b = phase_b(subject, seed)
    _stabilize(subject, seed)
    b_after = _old_skill_scores(subject, seed)
    c = phase_c(subject, seed)
    _stabilize(subject, seed)
    d = phase_d(subject, seed)
    _stabilize(subject, seed)
    e = phase_e_growth(subject, seed)
    _stabilize(subject, seed)
    e_after = _old_skill_scores(subject, seed)
    f = phase_f_continuity(subject, seed)
    damage = max(
        0.0,
        a_after["cn_masked_bi"] - e_after["cn_masked_bi"],
        b_after["p24_overall"] - e_after["p24_overall"],
    )
    return {
        "seed": seed,
        "phase_a": a,
        "phase_b": b,
        "phase_c": c,
        "phase_d": d,
        "phase_e": e,
        "phase_f": f,
        "old_skill_damage": damage,
        "skill_trace": {"after_a": a_after, "after_b": b_after, "after_e": e_after},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {
        "a_bi": _mean([r["phase_a"]["masked_bi_acc"] for r in rows]),
        "a_adv": _mean([r["phase_a"]["masked_adv_acc"] for r in rows]),
        "b_overall": _mean([r["phase_b"]["overall_acc"] for r in rows]),
        "b_unprotected": _mean([r["phase_b"]["unprotected_acc"] for r in rows]),
        "b_gain": _mean([r["phase_b"]["gain"] for r in rows]),
        "c_utility": _mean([r["phase_c"]["net_utility"] for r in rows]),
        "c_always_act": _mean([r["phase_c"]["always_act_utility"] for r in rows]),
        "c_known_act": _mean([r["phase_c"]["known_act_rate"] for r in rows]),
        "c_amb_query": _mean([r["phase_c"]["ambiguous_query_rate"] for r in rows]),
        "c_unres_wait": _mean([r["phase_c"]["unresolvable_wait_rate"] for r in rows]),
        "c_brier": _mean([r["phase_c"]["brier_score"] for r in rows]),
        "d_acc": _mean([r["phase_d"]["accuracy"] for r in rows]),
        "d_ablation": _mean([r["phase_d"]["ablation_drop"] for r in rows]),
        "e_final": _mean([r["phase_e"]["final_accuracy"] for r in rows]),
        "e_birth": _mean([r["phase_e"]["birth_consistency"] for r in rows]),
        "e_birth_diff": max(r["phase_e"]["birth_diff"] for r in rows),
        "e_lesion": _mean([r["phase_e"]["lesion_drop"] for r in rows]),
        "f_capsule": max(r["phase_f"]["capsule_forward_diff"] for r in rows),
        "f_reconnect": _mean([r["phase_f"]["reconnect_acc"] for r in rows]),
        "damage": max(r["old_skill_damage"] for r in rows),
    }
    gates = {
        "a_bi": means["a_bi"] >= 0.95,
        "a_adv": means["a_adv"] >= 0.90,
        "b_overall": means["b_overall"] >= 0.95,
        "b_unprotected": means["b_unprotected"] <= 0.40,
        "b_gain": means["b_gain"] >= 0.20,
        "c_utility_gain": means["c_utility"] - means["c_always_act"] >= 0.05,
        "c_routes": min(means["c_known_act"], means["c_amb_query"], means["c_unres_wait"]) >= 0.80,
        "c_brier": means["c_brier"] <= 0.20,
        "d_acc": means["d_acc"] >= 0.80,
        "d_ablation": means["d_ablation"] >= 0.20,
        "e_final": means["e_final"] >= 0.90,
        "e_birth": means["e_birth"] >= 1.0 - 1e-6 and means["e_birth_diff"] <= 1e-6,
        "e_lesion": means["e_lesion"] >= 0.20,
        "f_capsule": means["f_capsule"] <= 1e-6,
        "f_reconnect": means["f_reconnect"] >= 0.90,
        "old_skill_damage": means["damage"] <= 0.02,
    }
    passed = sum(1 for value in gates.values() if value)
    verdict = "s1r_verified" if passed == len(gates) else "s1r_partial"
    payload = {
        "schema": "taiji-u2-s1r2-rework-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "subject_units": UNITS_GROUPS * UNITS_PER_GROUP,
            "heads": ["cn", "p24", "p26", "p27"],
            "growth": "48 -> 56 -> 64 (+1/group twice), zero-init head columns + masked newborn two-stage",
            "budget_cpu": CAP_CPU,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "one shared 48-unit subject with per-phase readout heads; growth trains newborn units via the p27 head; external heads serialized separately from capsule_v2",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "passed": f"{passed}/{len(gates)}",
                "means": means,
                "gates": gates,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "s1r_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
