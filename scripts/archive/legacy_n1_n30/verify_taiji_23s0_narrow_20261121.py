"""23-S.0 narrow contract: Chinese + four-type correction + occlusion physics on one subject (0 check).

Single 48-unit SelectivePersistentGraph with per-phase readout heads (A1 wording).  P-B uses
online data (fixed-sample training overfits).  Gates are frozen in DESIGN-TAIJI_U2_23S0_NARROW.
Run under the owner's standing "proceed by recommendation" authorization; zero check/final.
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
from scripts.training.verify_taiji_s1r_restructure_20261121 import PhaseHead, SubjectR  # noqa: E402
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_23s0_narrow_20261121.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 160
STEPS_A = 800
STEPS_B = 2000
STEPS_CONTROL = 800
STEPS_D = 800
STEPS_STAB = 80
REPLAY_FRAC = 0.25


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


def _cn_acc(subject: SubjectR, fit: dict[str, torch.Tensor]) -> float:
    _set_mode(subject, False)
    with torch.no_grad():
        logits = subject.logits(
            "cn", fit["ticks"], step_mask=(fit["ticks"].abs().sum(-1) > 0).float()
        )
    return float((logits.argmax(-1) == fit["labels"]).float().mean())


def _p24_acc(subject: SubjectR, fit: dict[str, torch.Tensor]) -> dict[str, float]:
    _set_mode(subject, False)
    with torch.no_grad():
        logits = subject.logits("p24", fit["ticks"])
        overall = float((logits[:, :4].argmax(-1) == fit["val_labels"]).float().mean())
        conflict = float(((logits[:, 4] > 0).float() == fit["conflict_labels"]).float().mean())
    return {"overall": overall, "conflict": conflict}


def _p27_acc(subject: SubjectR, fit: dict[str, torch.Tensor]) -> float:
    _set_mode(subject, False)
    with torch.no_grad():
        return float(
            (subject.logits("p27", fit["ticks"]).argmax(-1) == fit["labels"]).float().mean()
        )


def run_seed(seed: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    fit_a = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    adv_a = _build_bidirectional_cn_32dim(
        256, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    fit_b = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    fit_d = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)

    # P-A online Chinese
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
    _set_mode(subject, False)
    a_bi = _cn_acc(subject, fit_a)
    a_adv = _cn_acc(subject, adv_a)
    a_train = _build_bidirectional_cn_32dim(
        256, seed + 900 + FIT_OFFSET, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
    )
    cn_memory = {"kind": "cn", "ticks": a_train["ticks"], "labels": a_train["labels"]}

    # P-B online correction
    _set_mode(subject, True)
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_B):
        batch = _make_p24_stream(BATCH, seed * 25000 + step)
        logits = subject.logits("p24", batch["ticks"])
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        if step % 4 == 0:
            loss = loss + REPLAY_FRAC * F.cross_entropy(
                subject.logits(
                    "cn",
                    cn_memory["ticks"][:12],
                    step_mask=(cn_memory["ticks"][:12].abs().sum(-1) > 0).float(),
                ),
                cn_memory["labels"][:12],
            )
        _step(opt, loss)
    b = _p24_acc(subject, fit_b)

    # unprotected control: trusted-only head on a frozen copy of the subject
    _set_mode(subject, False)
    control = copy.deepcopy(subject)
    for parameter in control.graph.parameters():
        parameter.requires_grad_(False)
    control.phase_heads["p24"].train()
    copt = torch.optim.Adam(control.phase_heads["p24"].parameters(), lr=7e-3)
    for step in range(STEPS_CONTROL):
        batch = _make_p24_stream(
            BATCH, seed * 26000 + step, force_trusted=True, trusted_regimes_only=True
        )
        z, aa = control.state_of(batch["ticks"])
        logits = control.phase_heads["p24"](torch.cat([z, aa], dim=-1))
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        copt.zero_grad(set_to_none=True)
        loss.backward()
        copt.step()
    _set_mode(control, False)
    with torch.no_grad():
        z, aa = control.state_of(fit_b["ticks"])
        control_logits = control.phase_heads["p24"](torch.cat([z, aa], dim=-1))
        unprotected = float(
            (control_logits[:, :4].argmax(-1) == fit_b["val_labels"]).float().mean()
        )

    # P-D online physics
    _set_mode(subject, True)
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_D):
        batch = _make_p27_physics_batch(BATCH, seed * 41000 + step)
        loss = F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"])
        if step % 4 == 0:
            p24_batch = _make_p24_stream(12, seed * 27000 + step)
            logits = subject.logits("p24", p24_batch["ticks"])
            loss = loss + REPLAY_FRAC * (
                F.cross_entropy(logits[:, :4], p24_batch["val_labels"])
                + F.binary_cross_entropy_with_logits(logits[:, 4], p24_batch["conflict_labels"])
            )
        _step(opt, loss)
    _set_mode(subject, False)
    d = _p27_acc(subject, fit_d)
    no_probe = fit_d["ticks"].clone()
    no_probe[:, :6, 8:11] = 0.0
    _set_mode(subject, False)
    with torch.no_grad():
        d_np = float((subject.logits("p27", no_probe).argmax(-1) == fit_d["labels"]).float().mean())
    a_bi_after = _cn_acc(subject, fit_a)
    b_after = _p24_acc(subject, fit_b)
    damage = max(0.0, a_bi - a_bi_after, b["overall"] - b_after["overall"])

    # capsule exactness + reconnect
    ticks = _make_m1_long_gap_data(32, seed + 101 + FIT_OFFSET, horizon=40, max_target_tick=20)[
        "ticks"
    ]
    _set_mode(subject, False)
    with torch.no_grad():
        full = subject.logits("cn", ticks, step_mask=(ticks.abs().sum(-1) > 0).float())
        state = subject.graph.initial_state(batch_size=ticks.shape[0])
        for t in range(20):
            out = subject.graph(
                _channels_seq(ticks[:, t : t + 1]),
                torch.ones(ticks.shape[0], 1, subject.n_units),
                state,
            )
            state = out["next_state"]
    cap = build_capsule_v2(subject.graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
        head_path = Path(tmp) / "heads.pt"
        torch.save({k: v.state_dict() for k, v in subject.phase_heads.items()}, head_path)
        heads_state = torch.load(head_path, weights_only=True)
    fresh = SelectivePersistentGraph(
        copy.deepcopy(subject.graph.base_graph), subject.graph.selective_spec
    )
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    fresh.phase_heads = {}

    for name, state_dict in heads_state.items():
        dim = state_dict["net.0.weight"].shape[1]
        width = state_dict["net.0.weight"].shape[0]
        weight_keys = [
            key for key in state_dict if key.startswith("net.") and key.endswith(".weight")
        ]
        last_index = max(int(key.split(".")[1]) for key in weight_keys)
        out_dim = state_dict[f"net.{last_index}.weight"].shape[0]
        head = PhaseHead(dim, out_dim)
        head.load_state_dict(state_dict)
        head.eval()
        fresh.phase_heads[name] = head
    with torch.no_grad():
        width = fresh.phase_heads["cn"].net[0].in_features
        state2 = fresh.initial_state(batch_size=ticks.shape[0])
        for t in range(20):
            out = fresh(
                _channels_seq(ticks[:, t : t + 1]),
                torch.ones(ticks.shape[0], 1, subject.n_units),
                state2,
            )
            state2 = out["next_state"]
        resume = fresh(
            _channels_seq(ticks[:, 20:]),
            torch.ones(ticks.shape[0], 20, subject.n_units),
            state2,
        )
        resume_logits = fresh.phase_heads["cn"](
            torch.cat([resume["next_state"]["z"], resume["next_state"]["a"]], dim=-1)[..., :width]
        )
        capsule_diff = float((resume_logits - full).abs().max())
        pred_cont = resume_logits.argmax(-1)
        pred_full = full.argmax(-1)
        reconnect_acc = float((pred_cont == pred_full).float().mean())

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
        "a_bi_after": a_bi_after,
        "b_after": b_after["overall"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    b_overalls = sorted(row["b_overall"] for row in rows)
    b_median = b_overalls[1]
    b_mean = sum(b_overalls) / len(b_overalls)
    gates = {
        "a_bi": min(row["a_bi"] for row in rows) >= 0.95,
        "a_adv": min(row["a_adv"] for row in rows) >= 0.90,
        "b_median": b_median >= 0.85,
        "b_mean": b_mean >= 0.85,
        "b_min": min(b_overalls) >= 0.80,
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
    verdict = "23s0_verified" if passed == len(gates) else "23s0_partial"
    payload = {
        "schema": "taiji-u2-23s0-narrow-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "run_under_standing_authorization_pending_owner_ratification",
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "subject_units": 48,
            "phases": ["P-A Chinese", "P-B online correction", "P-D physics"],
            "budget_cpu": CAP_CPU,
        },
        "rows": rows,
        "means": {
            "a_bi": sum(row["a_bi"] for row in rows) / len(rows),
            "a_adv": sum(row["a_adv"] for row in rows) / len(rows),
            "b_overall_mean": b_mean,
            "b_overall_median": b_median,
            "b_overall_min": min(b_overalls),
            "b_conflict": sum(row["b_conflict"] for row in rows) / len(rows),
            "b_unprotected": sum(row["b_unprotected"] for row in rows) / len(rows),
            "d_acc": sum(row["d_acc"] for row in rows) / len(rows),
            "d_ablation": sum(row["d_ablation"] for row in rows) / len(rows),
            "old_skill_damage": max(row["old_skill_damage"] for row in rows),
            "capsule_diff": max(row["capsule_diff"] for row in rows),
            "reconnect_acc": sum(row["reconnect_acc"] for row in rows) / len(rows),
        },
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "run under standing owner authorization; heads serialized separately from subject capsule; synthetic controlled surface",
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
                "means": payload["means"],
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "23s0_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
