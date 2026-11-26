"""S1-G structure check: growth birth-neutrality and capsule_v2 exactness with correct head remap (0 check).

S1-R2's growth/capsule gates used a wrong head layout after growth: readout input is [z(N), a(N)],
so when N grows the a-offset moves from old_n to new_n.  This card remaps every head correctly,
then verifies: (1) growth is birth-neutral, (2) old skills keep 0 damage, (3) capsule_v2 restores
the same graph exactly.  Light training only, <=500 CPU, 0 check/final.
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

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import (  # noqa: E402
    _channels_seq,
)
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r_restructure_20261121 import (  # noqa: E402
    PhaseHead,
    SubjectR,
)
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_s1g_growth_fix_20261121.json"
CAP_CPU = 500.0
SEEDS = (20261120, 20261121)
STEPS = 200
BATCH = 48
FIT_OFFSET = 10_000_000


def _remap_head_layout(head: PhaseHead, old_units: int, new_units: int) -> None:
    """Rebuild head input from [z(old), a(old)] to [z(new), a(new)] with a-offset shift."""
    old_linear = head.net[0]
    width = old_linear.out_features
    new_linear = nn.Linear(2 * new_units, width)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.bias.copy_(old_linear.bias)
        new_linear.weight[:, :old_units].copy_(old_linear.weight[:, :old_units])
        new_linear.weight[:, new_units : new_units + old_units].copy_(
            old_linear.weight[:, old_units : 2 * old_units]
        )
    head.net[0] = new_linear


def train_head_only(
    subject: SubjectR, name: str, ticks: torch.Tensor, labels: torch.Tensor, seed: int
) -> None:
    for head in subject.phase_heads.values():
        head.train()
    optimizer = torch.optim.Adam(subject.phase_heads[name].parameters(), lr=6e-3)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = torch.randint(0, ticks.shape[0], (BATCH,), generator=rng)
        logits = subject.logits(name, ticks[idx])
        loss = F.cross_entropy(logits, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(subject.phase_heads[name].parameters(), 1.0)
        optimizer.step()


def eval_head(subject: SubjectR, name: str, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    for head in subject.phase_heads.values():
        head.eval()
    with torch.no_grad():
        logits = subject.logits(name, ticks)
    return float((logits.argmax(-1) == labels).float().mean())


def run_seed(seed: int) -> dict:
    torch.set_num_threads(1)
    subject = SubjectR(seed)
    cn = _make_p24_stream
    del cn
    cn_ticks = torch.zeros(128, 8, 48)
    gen = torch.Generator().manual_seed(seed + 71)
    cn_labels = torch.randint(0, 8, (128,), generator=gen)
    for i in range(128):
        cn_ticks[i, 0, cn_labels[i]] = 1.0
    p24 = _make_p24_stream(128, seed + 81)
    p27 = _make_p27_physics_batch(128, seed + 91)
    train_head_only(subject, "cn", cn_ticks, cn_labels, seed + 101)
    train_head_only(subject, "p24", p24["ticks"], p24["val_labels"], seed + 102)
    train_head_only(subject, "p27", p27["ticks"], p27["labels"], seed + 103)

    fit_cn = cn_ticks
    fit_p24 = _make_p24_stream(96, seed + 81 + FIT_OFFSET)
    fit_p27 = _make_p27_physics_batch(96, seed + 91 + FIT_OFFSET)
    before = {
        "cn": eval_head(subject, "cn", fit_cn, cn_labels),
        "p24": eval_head(subject, "p24", fit_p24["ticks"], fit_p24["val_labels"]),
        "p27": eval_head(subject, "p27", fit_p27["ticks"], fit_p27["labels"]),
    }
    fixed_ticks = fit_p27["ticks"]
    with torch.no_grad():
        before_logits = subject.logits("p27", fixed_ticks)

    old_units = subject.graph.selective_units.n_units
    for _ in range(2):
        count_before = subject.graph.selective_units.n_units
        subject.graph.grow_units_neutral_(1)
        subject.n_units = subject.graph.selective_units.n_units
        for _name, head in subject.phase_heads.items():
            _remap_head_layout(head, count_before, subject.n_units)
    final_units = subject.graph.selective_units.n_units

    after = {
        "cn": eval_head(subject, "cn", fit_cn, cn_labels),
        "p24": eval_head(subject, "p24", fit_p24["ticks"], fit_p24["val_labels"]),
        "p27": eval_head(subject, "p27", fit_p27["ticks"], fit_p27["labels"]),
    }
    with torch.no_grad():
        after_logits = subject.logits("p27", fixed_ticks)
    birth_diff = float((after_logits - before_logits).abs().max())
    damage = max(before[key] - after[key] for key in before)

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
        out_a = subject.graph(
            _channels_seq(fixed_ticks),
            torch.ones(96, 14, subject.n_units),
            subject.graph.initial_state(batch_size=96),
        )
        out_b = fresh(
            _channels_seq(fixed_ticks),
            torch.ones(96, 14, subject.n_units),
            fresh.initial_state(batch_size=96),
        )
        capsule_diff = float((out_a["next_state"]["z"] - out_b["next_state"]["z"]).abs().max())
        capsule_a_diff = float((out_a["next_state"]["a"] - out_b["next_state"]["a"]).abs().max())
    return {
        "seed": seed,
        "before": before,
        "after": after,
        "birth_diff": birth_diff,
        "old_skill_damage": damage,
        "capsule_diff": max(capsule_diff, capsule_a_diff),
        "units": [old_units, final_units],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_seed(seed) for seed in SEEDS]
    gates = {
        "birth_neutral": max(row["birth_diff"] for row in rows) <= 1e-6,
        "old_skill_damage": max(row["old_skill_damage"] for row in rows) <= 0.02,
        "capsule_exact": max(row["capsule_diff"] for row in rows) <= 1e-6,
    }
    verdict = "s1g_verified" if all(gates.values()) else "s1g_partial"
    payload = {
        "schema": "taiji-u2-s1g-growth-fix-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "growth": "48 -> 56 -> 64",
            "head_remap": True,
            "steps": STEPS,
        },
        "rows": rows,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "S1-R2 growth/capsule gates were measured with a wrong [z,a] head layout after growth; this card corrects the layout",
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
    return 0 if verdict == "s1g_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
