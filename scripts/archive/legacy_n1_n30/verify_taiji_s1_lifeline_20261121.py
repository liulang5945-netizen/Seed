"""S1 lifeline: one SelectivePersistentGraph subject runs P-A..P-F sequentially (0 check / 0 final).

Contract: MAINLINE §3A.1 / DESIGN-TAIJI_U2_S1_LIFELINE_20261121.md.
Phases (same instance, carried state, replay buffer, no per-phase new subject):
  P-A bidirectional Chinese (zero pos-embed, step_mask)
  P-B 2-entity 5-session four-type correction
  P-C two-turn epistemic choice (ACT/QUERY/WAIT)
  P-D occluded 1D friction physics
  P-E two-generation neutral growth on the same subject + lesion
  P-F capsule_v2 cold restore + mid-episode reconnect
Gates are pre-registered in the design card; a failed gate yields s1_partial, never retuned.
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

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
    _clip_and_step,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_ADV_FIRST,
    BI_CN_ADV_SECOND,
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _eval_cn_masked,
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

OUT = ROOT / "reports/taiji_u2_s1_lifeline_20261121.json"
CAP_CPU = 2000.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 160
ANSWER_DIM = 8
CONFLICT_IDX = 8
Q_SLICE = slice(9, 12)
CONF_IDX = 12
OUTPUT_DIM = 13
STEPS_A = 600
STEPS_B = 500
STEPS_C = 450
STEPS_D = 450
STEPS_GROW = 180
REPLAY_FRAC = 0.25
CN_FIT_N = 256
P24_FIT_N = FIT
P26_FIT_N = FIT
P27_FIT_N = FIT


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class Subject:
    """Single subject wrapper holding one graph plus replay buffers."""

    def __init__(self, seed: int) -> None:
        base, self.skill_data = train_joint_parent(seed)
        spec = SelectiveUnitSpec(
            groups=6,
            units_per_group=5,
            output_dim=OUTPUT_DIM,
            readout_hidden=64,
            per_sample_state=True,
            selective_gate=True,
            init_write_bias=-2.0,
            seed=seed + 900,
        )
        self.graph = SelectivePersistentGraph(copy.deepcopy(base), spec)
        self.graph.freeze_base_()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 917)
            self.graph._build_selective_branch(spec)
        self.optimizer = torch.optim.Adam(self._params(), lr=7e-3, weight_decay=1e-4)
        self.replay: list[dict[str, Any]] = []
        self.n_units = spec.n_units

    def _params(self) -> list[nn.Parameter]:
        return [p for p in self.graph.parameters() if p.requires_grad]

    def forward(
        self,
        ticks: torch.Tensor,
        gates: torch.Tensor | None = None,
        state: dict[str, torch.Tensor] | None = None,
        step_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        batch = ticks.shape[0]
        if gates is None:
            gates = torch.ones(batch, ticks.shape[1], self.n_units)
        if state is None:
            state = self.graph.initial_state(batch_size=batch)
        return self.graph(_channels_seq(ticks), gates, state, step_mask=step_mask)

    def train_step(self, loss: torch.Tensor) -> None:
        _clip_and_step(self.optimizer, self.graph, loss)

    def promote_if_any(self) -> None:
        self.graph.selective_units.promote_immature_to_mature_()

    def add_replay(self, kind: str, payload: dict[str, Any]) -> None:
        self.replay.append({"kind": kind, **payload})


def _cn_batch(seed: int, count: int = 1600) -> dict[str, torch.Tensor]:
    return _build_bidirectional_cn_32dim(count, seed + 600, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND)


def _cn_fit(seed: int) -> dict[str, torch.Tensor]:
    return _build_bidirectional_cn_32dim(
        CN_FIT_N, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )


def _cn_adv(seed: int) -> dict[str, torch.Tensor]:
    return _build_bidirectional_cn_32dim(
        CN_FIT_N, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )


def _p24_batch(seed: int, count: int) -> dict[str, torch.Tensor]:
    return _make_p24_stream(count, seed)


def _loss_replay(subject: Subject, rng: torch.Generator) -> torch.Tensor:
    if not subject.replay:
        return torch.zeros(())
    picks = [
        subject.replay[int(torch.randint(0, len(subject.replay), (1,), generator=rng))]
        for _ in range(2)
    ]
    total = torch.zeros(())
    for item in picks:
        if item["kind"] == "cn":
            idx = torch.randint(0, item["ticks"].shape[0], (BATCH // 4,), generator=rng)
            seq = item["ticks"][idx]
            out = subject.forward(seq, step_mask=(seq.abs().sum(-1) > 0).float())
            total = total + F.cross_entropy(out["readout"][:, :ANSWER_DIM], item["labels"][idx])
        elif item["kind"] == "p24":
            idx = torch.randint(0, item["ticks"].shape[0], (BATCH // 4,), generator=rng)
            out = subject.forward(item["ticks"][idx])
            total = total + F.cross_entropy(out["readout"][:, :4], item["val_labels"][idx])
            total = total + F.binary_cross_entropy_with_logits(
                out["readout"][:, CONFLICT_IDX], item["conflict_labels"][idx]
            )
        elif item["kind"] == "p27":
            idx = torch.randint(0, item["ticks"].shape[0], (BATCH // 4,), generator=rng)
            out = subject.forward(item["ticks"][idx])
            total = total + F.cross_entropy(out["readout"][:, :4], item["labels"][idx])
    return total


def phase_a(subject: Subject, seed: int) -> dict[str, float]:
    train = _cn_batch(seed)
    fit = _cn_fit(seed)
    adv = _cn_adv(seed)
    rng = torch.Generator().manual_seed(seed + 31)
    for _ in range(STEPS_A):
        idx = torch.randint(0, train["ticks"].shape[0], (BATCH,), generator=rng)
        seq = train["ticks"][idx]
        out = subject.forward(seq, step_mask=(seq.abs().sum(-1) > 0).float())
        loss = F.cross_entropy(out["readout"][:, :ANSWER_DIM], train["labels"][idx])
        if subject.replay:
            loss = loss + REPLAY_FRAC * _loss_replay(subject, rng)
        subject.train_step(loss)
    subject.promote_if_any()
    subject.add_replay("cn", {"ticks": train["ticks"], "labels": train["labels"]})
    return {
        "masked_bi_acc": _eval_cn_masked(
            subject.graph, fit["ticks"], fit["labels"], use_step_mask=True
        ),
        "masked_adv_acc": _eval_cn_masked(
            subject.graph, adv["ticks"], adv["labels"], use_step_mask=True
        ),
    }


def phase_b(subject: Subject, seed: int) -> dict[str, float]:
    train = _p24_batch(seed + 211, 320)
    fit = _p24_batch(seed + 212 + FIT_OFFSET, P24_FIT_N)
    snapshot = copy.deepcopy(subject.graph)
    rng = torch.Generator().manual_seed(seed + 41)
    for _ in range(STEPS_B):
        idx = torch.randint(0, train["ticks"].shape[0], (BATCH,), generator=rng)
        out = subject.forward(train["ticks"][idx])
        loss = F.cross_entropy(out["readout"][:, :4], train["val_labels"][idx])
        loss = loss + F.binary_cross_entropy_with_logits(
            out["readout"][:, CONFLICT_IDX], train["conflict_labels"][idx]
        )
        if subject.replay:
            loss = loss + REPLAY_FRAC * _loss_replay(subject, rng)
        subject.train_step(loss)
    subject.promote_if_any()
    subject.add_replay(
        "p24",
        {
            "ticks": train["ticks"],
            "val_labels": train["val_labels"],
            "conflict_labels": train["conflict_labels"],
        },
    )
    with torch.no_grad():
        out = subject.forward(fit["ticks"])["readout"]
        preds = out[:, :4].argmax(-1)
        overall = float((preds == fit["val_labels"]).float().mean())
        conf_pred = (out[:, CONFLICT_IDX] > 0).float()
        conflict_acc = float((conf_pred == fit["conflict_labels"]).float().mean())
    with torch.no_grad():
        base_out = snapshot(
            _channels_seq(fit["ticks"]),
            torch.ones(P24_FIT_N, 14, subject.n_units),
            snapshot.initial_state(batch_size=P24_FIT_N),
        )["readout"]
    unprotected = float((base_out[:, :4].argmax(-1) == fit["val_labels"]).float().mean())
    return {
        "overall_acc": overall,
        "conflict_acc": conflict_acc,
        "unprotected_acc": unprotected,
        "gain": overall - unprotected,
    }


def phase_c(subject: Subject, seed: int) -> dict[str, float]:
    fit_count = P26_FIT_N
    rng = torch.Generator().manual_seed(seed + 51)
    for _ in range(STEPS_C):
        batch = _make_p26_batch(
            BATCH, seed * 30000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        out0 = subject.forward(batch["turn0"])
        out1 = subject.forward(batch["reveal"], state=out0["next_state"])
        y = batch["labels"]
        a0 = out0["readout"][:, :4].argmax(-1)
        a1 = out1["readout"][:, :4].argmax(-1)
        u_act = torch.where(a0 == y, 1.0, -1.0)
        u_query = torch.where(a1 == y, 1.0, -1.0) - 0.20
        u_wait = torch.full_like(u_act, -0.05)
        utils = torch.stack([u_act, u_query, u_wait], dim=-1)
        q_pred = out0["readout"][:, Q_SLICE]
        loss = F.cross_entropy(out0["readout"][:, :4], y)
        loss = loss + F.cross_entropy(out1["readout"][:, :4], y)
        loss = loss + 0.5 * F.mse_loss(q_pred, utils)
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            out0["readout"][:, CONF_IDX], (a0 == y).float()
        )
        if subject.replay:
            loss = loss + REPLAY_FRAC * _loss_replay(subject, rng)
        subject.train_step(loss)
    subject.promote_if_any()

    fit = _make_p26_batch(fit_count, seed + 61 + FIT_OFFSET)
    with torch.no_grad():
        out0 = subject.forward(fit["turn0"])
        ro0 = out0["readout"]
        policy = ro0[:, Q_SLICE].argmax(-1)
        a0 = ro0[:, :4].argmax(-1)
        conf = torch.sigmoid(ro0[:, CONF_IDX])
        out1 = subject.forward(fit["reveal"], state=out0["next_state"])
        a1 = out1["readout"][:, :4].argmax(-1)
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


def phase_d(subject: Subject, seed: int) -> dict[str, float]:
    rng = torch.Generator().manual_seed(seed + 71)
    for _ in range(STEPS_D):
        batch = _make_p27_physics_batch(
            BATCH, seed * 40000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        out = subject.forward(batch["ticks"])
        loss = F.cross_entropy(out["readout"][:, :4], batch["labels"])
        if subject.replay:
            loss = loss + REPLAY_FRAC * _loss_replay(subject, rng)
        subject.train_step(loss)
    subject.promote_if_any()
    subject.add_replay("p27", {"ticks": batch["ticks"], "labels": batch["labels"]})
    fit = _make_p27_physics_batch(P27_FIT_N, seed + 81 + FIT_OFFSET)
    with torch.no_grad():
        acc = float(
            (subject.forward(fit["ticks"])["readout"][:, :4].argmax(-1) == fit["labels"])
            .float()
            .mean()
        )
        no_probe = fit["ticks"].clone()
        no_probe[:, :6, 8:11] = 0.0
        acc_np = float(
            (subject.forward(no_probe)["readout"][:, :4].argmax(-1) == fit["labels"]).float().mean()
        )
    return {"accuracy": acc, "ablation_drop": acc - acc_np}


def _old_skill_scores(subject: Subject, seed: int) -> dict[str, float]:
    fit = _cn_fit(seed)
    p24 = _p24_batch(seed + 212 + FIT_OFFSET, P24_FIT_N)
    with torch.no_grad():
        p24_out = subject.forward(p24["ticks"])["readout"]
        p24_acc = float((p24_out[:, :4].argmax(-1) == p24["val_labels"]).float().mean())
    return {
        "cn_masked_bi": _eval_cn_masked(
            subject.graph, fit["ticks"], fit["labels"], use_step_mask=True
        ),
        "p24_overall": p24_acc,
    }


def phase_e_growth(subject: Subject, seed: int) -> dict[str, float]:
    fixed = _make_p27_physics_batch(64, seed + 91 + FIT_OFFSET)
    with torch.no_grad():
        before = subject.forward(fixed["ticks"])["readout"].clone()
    old_units = subject.graph.selective_units.n_units
    subject.graph.grow_units_neutral_(2)
    subject.graph.grow_units_neutral_(2)
    new_units = subject.graph.selective_units.n_units
    subject.n_units = new_units
    with torch.no_grad():
        after = subject.forward(fixed["ticks"])["readout"].clone()
    birth_diff = float((after - before).abs().max())
    subject.optimizer = torch.optim.Adam(subject._params(), lr=5e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(seed + 81)
    for _ in range(STEPS_GROW):
        batch = _make_p27_physics_batch(
            BATCH, seed * 50000 + int(torch.randint(0, 1_000_000, (1,), generator=rng))
        )
        out = subject.forward(batch["ticks"])
        loss = F.cross_entropy(out["readout"][:, :4], batch["labels"])
        if subject.replay:
            loss = loss + REPLAY_FRAC * _loss_replay(subject, rng)
        subject.train_step(loss)
    subject.graph.selective_units.promote_immature_to_mature_()
    fit = _make_p27_physics_batch(P27_FIT_N, seed + 81 + FIT_OFFSET)
    with torch.no_grad():
        acc = float(
            (subject.forward(fit["ticks"])["readout"][:, :4].argmax(-1) == fit["labels"])
            .float()
            .mean()
        )
        gates = torch.ones(P27_FIT_N, 14, new_units)
        gates[..., old_units:] = 0.0
        lesioned = float(
            (
                subject.forward(fit["ticks"], gates=gates)["readout"][:, :4].argmax(-1)
                == fit["labels"]
            )
            .float()
            .mean()
        )
    return {
        "final_accuracy": acc,
        "birth_diff": birth_diff,
        "lesion_drop": acc - lesioned,
        "units": float(new_units),
    }


def phase_f_continuity(subject: Subject, seed: int) -> dict[str, float]:
    batch = _make_m1_long_gap_data(32, seed + 101 + FIT_OFFSET, horizon=40, max_target_tick=20)
    ticks = batch["ticks"]
    with torch.no_grad():
        out_full = subject.forward(ticks)
        pred_full = out_full["readout"][:, :ANSWER_DIM].argmax(-1)

        st = subject.graph.initial_state(batch_size=ticks.shape[0])
        for t in range(20):
            o = subject.forward(ticks[:, t : t + 1], state=st)
            st = o["next_state"]
        cont = subject.forward(ticks[:, 20:], state={k: v.clone() for k, v in st.items()})
        pred_cont = cont["readout"][:, :ANSWER_DIM].argmax(-1)

    gate = torch.ones(ticks.shape[0], ticks.shape[1], subject.n_units)
    cap = build_capsule_v2(subject.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "s1_capsule.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subject.graph.base_graph), subject.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        out_fresh = fresh(
            _channels_seq(ticks), gate, fresh.initial_state(batch_size=ticks.shape[0])
        )
        fresh_diff = float((out_fresh["readout"] - out_full["readout"]).abs().max())

        cont_state = {k: v.clone() for k, v in st.items()}
        o2 = fresh(
            _channels_seq(ticks[:, 20:]),
            torch.ones(ticks.shape[0], 20, subject.n_units),
            cont_state,
        )
        reconnect_diff = float(
            (o2["readout"][:, :ANSWER_DIM] - cont["readout"][:, :ANSWER_DIM]).abs().max()
        )
        reconnect_acc = float((pred_cont == pred_full).float().mean())

    return {
        "capsule_forward_diff": fresh_diff,
        "reconnect_diff": reconnect_diff,
        "reconnect_acc": reconnect_acc,
        "continuity_ok": float(reconnect_acc >= 0.90 and fresh_diff <= 1e-6),
    }


def run_seed(seed: int) -> dict[str, Any]:
    subject = Subject(seed)
    a = phase_a(subject, seed)
    a_after = _old_skill_scores(subject, seed)
    b = phase_b(subject, seed)
    b_after = _old_skill_scores(subject, seed)
    c = phase_c(subject, seed)
    d = phase_d(subject, seed)
    d_after = _old_skill_scores(subject, seed)
    e = phase_e_growth(subject, seed)
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
        "skill_trace": {
            "after_a": a_after,
            "after_b": b_after,
            "after_d": d_after,
            "after_e": e_after,
        },
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
        "e_birth": max(r["phase_e"]["birth_diff"] for r in rows),
        "e_lesion": _mean([r["phase_e"]["lesion_drop"] for r in rows]),
        "f_capsule": max(r["phase_f"]["capsule_forward_diff"] for r in rows),
        "f_reconnect_acc": _mean([r["phase_f"]["reconnect_acc"] for r in rows]),
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
        "e_birth": means["e_birth"] <= 1e-6,
        "e_lesion": means["e_lesion"] >= 0.20,
        "f_capsule": means["f_capsule"] <= 1e-6,
        "f_reconnect": means["f_reconnect_acc"] >= 0.90,
        "old_skill_damage": means["damage"] <= 0.02,
    }
    passed = sum(1 for v in gates.values() if v)
    verdict = "s1_verified" if passed == len(gates) else "s1_partial"
    payload = {
        "schema": "taiji-u2-s1-lifeline-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "single_subject": True,
            "phases": ["P-A", "P-B", "P-C", "P-D", "P-E", "P-F"],
            "steps": {"a": STEPS_A, "b": STEPS_B, "c": STEPS_C, "d": STEPS_D, "grow": STEPS_GROW},
            "budget_cpu": CAP_CPU,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "same subject instance across phases; replay buffer mixes previous phases; growth 30->42->54 (+2/group twice); synthetic controlled surface",
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
    return 0 if verdict == "s1_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
