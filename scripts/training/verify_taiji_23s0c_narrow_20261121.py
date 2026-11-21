"""23-S.0c narrow contract redesign: frozen shared units + per-phase adapter heads (0 check).

Protocol redesign after D-20261121-239: freeze the shared 48-unit subject after P-A Chinese,
then train only per-phase adapter heads (P-B correction, P-D physics).  This removes the
unit-level interference seen in 23-S.0/0b.  The unprotected control is redefined as the
untrained P-B head (chance-level), since the prior "trusted-only" control was invalid on the
new-core fixture.  Run under standing authorization; zero check/final, <=1,200 CPU.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq  # noqa: E402
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
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r2_rework_20261121 import PhaseHead  # noqa: E402
from scripts.training.verify_taiji_s1r_restructure_20261121 import SubjectR  # noqa: E402
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_23s0c_narrow_20261121.json"
CAP_CPU = 1200.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 160
STEPS_A = 900
STEPS_B = 1500
STEPS_D = 800


def _set_mode(subject: SubjectR, training: bool) -> None:
    for head in subject.phase_heads.values():
        head.train(training)


def _step(optimizer: torch.optim.Optimizer, loss: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]], 1.0
    )
    optimizer.step()


def _cn_acc(subject: SubjectR, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    _set_mode(subject, False)
    with torch.no_grad():
        logits = subject.logits("cn", ticks, step_mask=(ticks.abs().sum(-1) > 0).float())
    return float((logits.argmax(-1) == labels).float().mean())


def _p24_scores(
    subject: SubjectR, ticks: torch.Tensor, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    _set_mode(subject, False)
    with torch.no_grad():
        logits = subject.logits("p24", ticks)
        return {
            "overall": float((logits[:, :4].argmax(-1) == batch["val_labels"]).float().mean()),
            "conflict": float(
                ((logits[:, 4] > 0).float() == batch["conflict_labels"]).float().mean()
            ),
        }


def _p27_acc(subject: SubjectR, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    _set_mode(subject, False)
    with torch.no_grad():
        return float((subject.logits("p27", ticks).argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    subject.phase_heads["cn"] = PhaseHead(2 * subject.n_units, 8, 128)
    subject.phase_heads["p24"] = PhaseHead(2 * subject.n_units, 5, 128)

    fit_a = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    adv_a = _build_bidirectional_cn_32dim(
        256, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    fit_b = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    fit_d = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)

    # P-A: train shared units + cn head
    _set_mode(subject, True)
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    for step in range(STEPS_A):
        batch = _build_bidirectional_cn_32dim(
            BATCH, seed * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        logits = subject.logits(
            "cn", batch["ticks"], step_mask=(batch["ticks"].abs().sum(-1) > 0).float()
        )
        _step(opt, F.cross_entropy(logits, batch["labels"]))
    a_bi = _cn_acc(subject, fit_a["ticks"], fit_a["labels"])
    a_adv = _cn_acc(subject, adv_a["ticks"], adv_a["labels"])

    # freeze shared units: only adapter heads are trained from here on
    for parameter in subject.graph.parameters():
        parameter.requires_grad_(False)
    _set_mode(subject, True)

    # P-B: p24 head only; untrained head is the new unprotected control
    p24_untrained = copy.deepcopy(subject.phase_heads["p24"])
    p24_untrained.eval()
    _set_mode(subject, True)
    opt_b = torch.optim.Adam(subject.phase_heads["p24"].parameters(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_B):
        batch = _make_p24_stream(BATCH, seed * 25000 + step)
        logits = subject.logits("p24", batch["ticks"])
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        _step(opt_b, loss)
    b = _p24_scores(subject, fit_b["ticks"], fit_b)
    with torch.no_grad():
        z, aa = subject.state_of(fit_b["ticks"])
        untrained_logits = p24_untrained(torch.cat([z, aa], dim=-1))
        unprotected = float(
            (untrained_logits[:, :4].argmax(-1) == fit_b["val_labels"]).float().mean()
        )

    # P-D: p27 head only
    _set_mode(subject, True)
    opt_d = torch.optim.Adam(subject.phase_heads["p27"].parameters(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_D):
        batch = _make_p27_physics_batch(BATCH, seed * 41000 + step)
        _step(opt_d, F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"]))
    d = _p27_acc(subject, fit_d["ticks"], fit_d["labels"])
    no_probe = fit_d["ticks"].clone()
    no_probe[:, :6, 8:11] = 0.0
    d_np = _p27_acc(subject, no_probe, fit_d["labels"])
    a_bi_after = _cn_acc(subject, fit_a["ticks"], fit_a["labels"])
    b_after = _p24_scores(subject, fit_b["ticks"], fit_b)["overall"]
    damage = max(0.0, a_bi - a_bi_after, b["overall"] - b_after)

    # capsule + reconnect on the frozen subject
    ticks = _make_m1_long_gap_data(32, seed + 101 + FIT_OFFSET, horizon=40, max_target_tick=20)[
        "ticks"
    ]
    with torch.no_grad():
        out_a = subject.graph(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], subject.n_units),
            subject.graph.initial_state(batch_size=ticks.shape[0]),
        )
    cap = build_capsule_v2(subject.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subject.graph.base_graph), subject.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        out_b = fresh(
            _channels_seq(ticks),
            torch.ones(ticks.shape[0], ticks.shape[1], subject.n_units),
            fresh.initial_state(batch_size=ticks.shape[0]),
        )
        capsule_diff = float((out_a["readout"] - out_b["readout"]).abs().max())
        state2 = fresh.initial_state(batch_size=ticks.shape[0])
        for tt in range(20):
            out = fresh(
                _channels_seq(ticks[:, tt : tt + 1]),
                torch.ones(ticks.shape[0], 1, subject.n_units),
                state2,
            )
            state2 = out["next_state"]
        resume = fresh(
            _channels_seq(ticks[:, 20:]),
            torch.ones(ticks.shape[0], 20, subject.n_units),
            state2,
        )
        reconnect_acc = float(
            (resume["readout"].argmax(-1) == out_a["readout"].argmax(-1)).float().mean()
        )

    return {
        "seed": seed,
        "a_bi": a_bi,
        "a_adv": a_adv,
        "b_overall": b["overall"],
        "b_conflict": b["conflict"],
        "b_unprotected": unprotected,
        "b_gain": b["overall"] - unprotected,
        "d_acc": d,
        "d_ablation": d - d_np,
        "old_skill_damage": damage,
        "capsule_diff": capsule_diff,
        "reconnect_acc": reconnect_acc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    b_overalls = sorted(row["b_overall"] for row in rows)
    gates = {
        "a_bi": min(row["a_bi"] for row in rows) >= 0.95,
        "a_adv": min(row["a_adv"] for row in rows) >= 0.90,
        "b_median": b_overalls[1] >= 0.85,
        "b_mean": sum(b_overalls) / len(b_overalls) >= 0.85,
        "b_min": b_overalls[0] >= 0.80,
        "b_conflict": min(row["b_conflict"] for row in rows) >= 0.95,
        "b_unprotected": max(row["b_unprotected"] for row in rows) <= 0.40,
        "b_gain": min(row["b_gain"] for row in rows) >= 0.20,
        "d_acc": min(row["d_acc"] for row in rows) >= 0.80,
        "d_ablation": min(row["d_ablation"] for row in rows) >= 0.20,
        "old_skill_damage": max(row["old_skill_damage"] for row in rows) <= 0.02,
        "capsule_diff": max(row["capsule_diff"] for row in rows) <= 1e-6,
        "reconnect": min(row["reconnect_acc"] for row in rows) >= 0.90,
    }
    passed = sum(1 for value in gates.values() if value)
    verdict = "23s0c_verified" if passed == len(gates) else "23s0c_partial"
    payload = {
        "schema": "taiji-u2-23s0c-narrow-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "run_under_standing_authorization_pending_owner_ratification",
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "design": "freeze units after P-A; P-B/P-D adapter heads only; untrained head as unprotected control",
            "budget_cpu": CAP_CPU,
        },
        "rows": rows,
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "protocol redesign after 23-S.0 partial; unprotected control redefined as untrained head; zero check/final",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "passed": f"{passed}/{len(gates)}",
                "gates": gates,
                "rows": rows,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "23s0c_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
