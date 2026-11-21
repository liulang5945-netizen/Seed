"""S1-RT stage 1b: corrected fixed-partition test on 72 units (24 per task), 0 check, <=800 CPU.

Fixes stage-1 gate mistakes: separate eval sets (CN normal vs adversarial), full 3x3 lesion
matrix, and 24 units per task.  Two arms (shared control vs partitioned), 3 seeds.
Run under standing authorization; zero check/final.
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

OUT = ROOT / "reports/taiji_u2_s1rt_stage1b_20261121.json"
CAP_CPU = 800.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 128
STEPS_CN = 600
STEPS_P24 = 900
STEPS_P27 = 500
BLOCK = 24
N_UNITS = 72


def _step(opt: torch.optim.Optimizer, loss: torch.Tensor, graph) -> None:
    _clip_and_step(opt, graph, loss)


class Subject72(SubjectR):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        base, _ = __import__(
            "scripts.training.run_s2_23_2c_attention_20261120", fromlist=["train_joint_parent"]
        ).train_joint_parent(seed)
        spec = __import__(
            "taiji.u2_selective_units", fromlist=["SelectiveUnitSpec"]
        ).SelectiveUnitSpec(
            groups=12,
            units_per_group=6,
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
        self.phase_heads = {
            "cn": PhaseHead(2 * self.n_units, 8, 128),
            "p24": PhaseHead(2 * self.n_units, 5, 128),
            "p27": PhaseHead(2 * self.n_units, 4, 64),
        }
        self.masks: dict[str, torch.Tensor] = {}

    def train_mode(self) -> None:
        for head in self.phase_heads.values():
            head.train()

    def eval_mode(self) -> None:
        for head in self.phase_heads.values():
            head.eval()

    def set_masks(self, partitioned: bool) -> None:
        n = self.n_units
        if partitioned:
            self.masks = {
                "cn": torch.tensor([1.0 if i < 24 else 0.0 for i in range(n)]),
                "p24": torch.tensor([1.0 if 24 <= i < 48 else 0.0 for i in range(n)]),
                "p27": torch.tensor([1.0 if i >= 48 else 0.0 for i in range(n)]),
            }
        else:
            self.masks = {name: torch.ones(n) for name in ("cn", "p24", "p27")}

    def logits(
        self, name: str, ticks: torch.Tensor, step_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = self.masks[name].view(1, 1, self.n_units)
        gates = mask.expand(ticks.shape[0], ticks.shape[1], self.n_units)
        out = self.graph(
            _channels_seq(ticks),
            gates,
            self.graph.initial_state(batch_size=ticks.shape[0]),
            step_mask=step_mask,
        )
        z, a = out["next_state"]["z"], out["next_state"]["a"]
        head = self.phase_heads[name]
        return head(torch.cat([z, a], dim=-1))


def _eval(subject: Subject72, name: str, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    subject.eval_mode()
    with torch.no_grad():
        logits = subject.logits(
            name, ticks, step_mask=(ticks.abs().sum(-1) > 0).float() if name == "cn" else None
        )
    return float((logits.argmax(-1) == labels).float().mean())


def _eval_p24(subject: Subject72, ticks: torch.Tensor, batch: dict[str, torch.Tensor]) -> float:
    subject.eval_mode()
    with torch.no_grad():
        logits = subject.logits("p24", ticks)
    return float((logits[:, :4].argmax(-1) == batch["val_labels"]).float().mean())


def run_arm(seed: int, partitioned: bool) -> dict[str, Any]:
    torch.set_num_threads(1)
    subject = Subject72(seed)
    subject.set_masks(partitioned)
    fit_cn = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    fit_adv = _build_bidirectional_cn_32dim(
        256, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    fit_p24 = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    fit_p27 = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)
    trace: dict[str, float] = {}

    # P-A
    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=1e-4)
    for step in range(STEPS_CN):
        batch = _build_bidirectional_cn_32dim(
            BATCH, seed * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        loss = F.cross_entropy(
            subject.logits("cn", batch["ticks"], (batch["ticks"].abs().sum(-1) > 0).float()),
            batch["labels"],
        )
        _step(opt, loss, subject.graph)
    trace["after_cn"] = _eval(subject, "cn", fit_cn["ticks"], fit_cn["labels"])
    cn_peak = trace["after_cn"]

    # P-B online
    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_P24):
        batch = _make_p24_stream(BATCH, seed * 25000 + step)
        logits = subject.logits("p24", batch["ticks"])
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        _step(opt, loss, subject.graph)
    trace["after_p24"] = _eval_p24(subject, fit_p24["ticks"], fit_p24)
    p24_peak = trace["after_p24"]

    # P-D
    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=1e-4)
    for step in range(STEPS_P27):
        batch = _make_p27_physics_batch(BATCH, seed * 41000 + step)
        _step(
            opt,
            F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"]),
            subject.graph,
        )
    cn = _eval(subject, "cn", fit_cn["ticks"], fit_cn["labels"])
    adv = _eval(subject, "cn", fit_adv["ticks"], fit_adv["labels"])
    p24 = _eval_p24(subject, fit_p24["ticks"], fit_p24)
    p27 = _eval(subject, "p27", fit_p27["ticks"], fit_p27["labels"])

    # 3x3 lesion matrix
    tasks = {
        "cn": lambda: _eval(subject, "cn", fit_cn["ticks"], fit_cn["labels"]),
        "p24": lambda: _eval_p24(subject, fit_p24["ticks"], fit_p24),
        "p27": lambda: _eval(subject, "p27", fit_p27["ticks"], fit_p27["labels"]),
    }
    matrix = {}
    for les_name in ("cn", "p24", "p27"):
        saved = subject.masks[les_name].clone()
        subject.masks[les_name].zero_()
        row = {name: tasks[name]() for name in tasks}
        subject.masks[les_name].copy_(saved)
        matrix[les_name] = row

    ticks = fit_p27["ticks"][:32]
    with torch.no_grad():
        out_a = subject.graph(
            _channels_seq(ticks),
            torch.ones(32, ticks.shape[1], subject.n_units),
            subject.graph.initial_state(batch_size=32),
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
            torch.ones(32, ticks.shape[1], subject.n_units),
            fresh.initial_state(batch_size=32),
        )
        capsule_diff = float((out_a["readout"] - out_b["readout"]).abs().max())

    return {
        "seed": seed,
        "partitioned": partitioned,
        "trace": trace,
        "cn": cn,
        "cn_adv": adv,
        "p24": p24,
        "p27": p27,
        "cn_damage": max(0.0, cn_peak - cn),
        "p24_damage": max(0.0, p24_peak - p24),
        "lesion_matrix": matrix,
        "capsule_diff": capsule_diff,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    rows = [run_arm(seed, partitioned) for partitioned in (False, True) for seed in SEEDS]
    shared = [r for r in rows if not r["partitioned"]]
    routed = [r for r in rows if r["partitioned"]]
    means = {
        "shared": {
            k: sum(r[k] for r in shared) / len(shared)
            for k in ("cn", "cn_adv", "p24", "p27", "cn_damage", "p24_damage")
        },
        "partitioned": {
            k: sum(r[k] for r in routed) / len(routed)
            for k in ("cn", "cn_adv", "p24", "p27", "cn_damage", "p24_damage")
        },
    }
    g3 = all(
        min(
            r["lesion_matrix"][own][own] - r["lesion_matrix"][own][other]
            for other in ("cn", "p24", "p27")
            if other != own
        )
        >= 0.15
        for r in routed
        for own in ("cn", "p24", "p27")
    )
    gates = {
        "g0_control": any(
            r["p27"] < 0.80 or r["cn_damage"] > 0.05 or r["p24_damage"] > 0.05 for r in shared
        ),
        "g1_coexist": all(
            r["cn"] >= 0.95 and r["cn_adv"] >= 0.90 and r["p24"] >= 0.85 and r["p27"] >= 0.80
            for r in routed
        ),
        "g2_old_skill": all(r["cn_damage"] <= 0.02 and r["p24_damage"] <= 0.02 for r in routed),
        "g3_specificity": g3,
        "g4_capsule": all(r["capsule_diff"] <= 1e-6 for r in routed),
    }
    passed = sum(1 for value in gates.values() if value)
    verdict = "s1rt_stage1b_verified" if passed == len(gates) else "s1rt_stage1b_partial"
    payload = {
        "schema": "taiji-u2-s1rt-stage1b-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "run_under_standing_authorization_pending_owner_ratification",
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "units": N_UNITS,
            "blocks": [24, 24, 24],
            "steps": [STEPS_CN, STEPS_P24, STEPS_P27],
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "stage-1b corrected gates; partition is experiment-level; zero check/final",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "means": means,
                "passed": f"{passed}/{len(gates)}",
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "s1rt_stage1b_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
