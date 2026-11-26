"""S1-G3: cohort-developmental integration after growth (0 check, <=800 CPU).

Uses grow_antithetic_cohort_ (twin newborn cohort, exact zero birth delta, frozen mature graph),
trains it during the immature phase with the positive-twin auxiliary signal (S0 Bridge2 path),
promotes it, then measures the lesion caused by putting the cohort back to dormant.
Gates: birth diff <= 1e-6, lesion >= 0.20, old-skill damage <= 0.02, capsule diff <= 1e-6.
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
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r_restructure_20261121 import SubjectR  # noqa: E402
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    SelectiveUnitSpec,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_s1g3_cohort_20261121.json"
CAP_CPU = 800.0
SEEDS = (20261120, 20261121)
BATCH = 48
FIT_OFFSET = 10_000_000
STEPS_BASE = 350
STEPS_COHORT = 500
STEPS_HEAD = 150


def _build_graph(seed: int) -> SelectivePersistentGraph:
    base, _ = train_joint_parent(seed)
    spec = SelectiveUnitSpec(
        groups=8,
        units_per_group=6,
        output_dim=13,
        readout_hidden=64,
        per_sample_state=True,
        selective_gate=True,
        init_write_bias=-2.0,
        seed=seed + 1000,
    )
    graph = SelectivePersistentGraph(copy.deepcopy(base), spec)
    graph.freeze_base_()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1017)
        graph._build_selective_branch(spec)
    return graph


def _forward(graph, ticks, units: int) -> dict[str, Any]:
    return graph(_channels_seq(ticks), torch.ones(ticks.shape[0], ticks.shape[1], units), None)


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict:
    torch.set_num_threads(1)
    graph = _build_graph(seed)
    subject = SubjectR(seed)
    p27 = _make_p27_physics_batch(512, seed + 91)
    p27_fit = _make_p27_physics_batch(128, seed + 92 + FIT_OFFSET)
    base_params = [p for p in graph.parameters() if p.requires_grad]
    opt = torch.optim.Adam(base_params, lr=6e-3)
    rng = torch.Generator().manual_seed(seed + 1201)
    units = graph.selective_spec.n_units
    for _ in range(STEPS_BASE):
        idx = torch.randint(0, p27["ticks"].shape[0], (BATCH,), generator=rng)
        out = _forward(graph, p27["ticks"][idx], units)
        loss = F.cross_entropy(out["readout"][:, :4], p27["labels"][idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params, 1.0)
        opt.step()
    with torch.no_grad():
        base_acc = _acc(
            _forward(graph, p27_fit["ticks"], units)["readout"][:, :4], p27_fit["labels"]
        )
        before_readout = _forward(graph, p27_fit["ticks"], units)["readout"]

    # old skills via external heads on shared units (base graph state frozen during cohort training)
    p24 = _make_p24_stream(128, seed + 82 + FIT_OFFSET)
    for _name, head in subject.phase_heads.items():
        head.train()
    cn_ticks = torch.zeros(128, 8, 48)
    gen = torch.Generator().manual_seed(seed + 73)
    cn_labels = torch.randint(0, 8, (128,), generator=gen)
    for i in range(128):
        cn_ticks[i, 0, cn_labels[i]] = 1.0
    for name, ticks, labels in (
        ("cn", cn_ticks, cn_labels),
        ("p24", p24["ticks"], p24["val_labels"]),
    ):
        head_opt = torch.optim.Adam(subject.phase_heads[name].parameters(), lr=6e-3)
        for _ in range(STEPS_HEAD):
            idx = torch.randint(0, ticks.shape[0], (BATCH,), generator=rng)
            loss = F.cross_entropy(subject.logits(name, ticks[idx]), labels[idx])
            head_opt.zero_grad(set_to_none=True)
            loss.backward()
            head_opt.step()

    def _old_skill() -> dict[str, float]:
        with torch.no_grad():
            cn_acc = _acc(subject.logits("cn", cn_ticks), cn_labels)
            p24_acc = _acc(subject.logits("p24", p24["ticks"]), p24["val_labels"])
        return {"cn": cn_acc, "p24": p24_acc}

    old_before = _old_skill()

    # grow a twin cohort; mature graph stays frozen
    cohort_index = graph.grow_antithetic_cohort_(
        graph.selective_spec.groups,
        graph.selective_spec.units_per_group,
        readout_hidden=32,
        freeze_mature=True,
        freeze_anchor=True,
        seed=seed + 1301,
    )
    with torch.no_grad():
        birth_readout = _forward(graph, p27_fit["ticks"], units)["readout"]
    birth_diff = float((birth_readout - before_readout).abs().max())

    cohort_params = [p for p in graph.growth_cohorts[cohort_index].parameters() if p.requires_grad]
    opt_c = torch.optim.Adam(cohort_params, lr=6e-3)
    for _ in range(STEPS_COHORT):
        idx = torch.randint(0, p27["ticks"].shape[0], (BATCH,), generator=rng)
        out = _forward(graph, p27["ticks"][idx], units)
        loss = F.cross_entropy(out["readout"][:, :4], p27["labels"][idx])
        if out["immature_cohort_logits"]:
            aux = out["immature_cohort_logits"][0][:, :4]
            loss = loss + 0.5 * F.cross_entropy(aux, p27["labels"][idx])
        opt_c.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cohort_params, 1.0)
        opt_c.step()
    graph.promote_cohorts_to_mature_()
    with torch.no_grad():
        final_acc = _acc(
            _forward(graph, p27_fit["ticks"], units)["readout"][:, :4], p27_fit["labels"]
        )
    graph.set_cohort_lifecycle_(cohort_index, "dormant")
    with torch.no_grad():
        dormant_acc = _acc(
            _forward(graph, p27_fit["ticks"], units)["readout"][:, :4], p27_fit["labels"]
        )
    graph.set_cohort_lifecycle_(cohort_index, "mature")
    lesion_drop = final_acc - dormant_acc
    old_after = _old_skill()
    damage = max(old_before[key] - old_after[key] for key in old_before)

    cap = build_capsule_v2(graph)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_capsule_v2(cap, Path(tmp) / "cap.json")
        loaded = load_capsule_v2(path)
    fresh = SelectivePersistentGraph(copy.deepcopy(graph.base_graph), graph.selective_spec)
    fresh.freeze_base_()
    apply_capsule_v2(fresh, copy.deepcopy(loaded))
    with torch.no_grad():
        a = _forward(graph, p27_fit["ticks"], units)["readout"]
        b = _forward(fresh, p27_fit["ticks"], units)["readout"]
        capsule_diff = float((a - b).abs().max())
    return {
        "seed": seed,
        "base_acc": base_acc,
        "final_acc": final_acc,
        "lesion_drop": lesion_drop,
        "birth_diff": birth_diff,
        "old_skill_damage": damage,
        "capsule_diff": capsule_diff,
        "cohort_index": cohort_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "birth_neutral": max(row["birth_diff"] for row in rows) <= 1e-6,
        "integration": min(row["lesion_drop"] for row in rows) >= 0.20,
        "old_skill_damage": max(row["old_skill_damage"] for row in rows) <= 0.02,
        "capsule_exact": max(row["capsule_diff"] for row in rows) <= 1e-6,
    }
    verdict = "s1g3_verified" if all(gates.values()) else "s1g3_partial"
    payload = {
        "schema": "taiji-u2-s1g3-cohort-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "cohort": "antithetic twin, frozen anchor, immature aux",
            "steps": [STEPS_BASE, STEPS_COHORT],
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "unit core untouched; cohort is the additive growth interface already exercised in S0 Bridge2",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "rows": rows, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "s1g3_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
