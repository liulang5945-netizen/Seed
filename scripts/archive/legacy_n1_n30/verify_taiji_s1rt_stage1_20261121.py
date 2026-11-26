"""S1-RT stage 1: fixed task-partition routing vs shared units (0 check, <=600 CPU).

Root cause from D-20261121-243: the new core passes per-axis but a shared 48-unit substrate is
overwritten by sequential tasks.  This smoke test gives each task a 16-unit subpopulation
(gate mask) and compares with the shared-substrate control, reusing the S1-R2 phase functions
so that per-axis training is known-good.  Run under standing authorization; zero check/final.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import verify_taiji_s1r2_rework_20261121 as s1r2  # noqa: E402
from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_algorithm_holistic_20261121 import _channels_seq  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_ADV_FIRST,
    BI_CN_ADV_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1r_restructure_20261121 import (
    SubjectR as BaseSubject,  # noqa: E402
)
from taiji.u2_selective_units import (  # noqa: E402
    SelectivePersistentGraph,
    apply_capsule_v2,
    build_capsule_v2,
    load_capsule_v2,
    save_capsule_v2,
)

OUT = ROOT / "reports/taiji_u2_s1rt_stage1_20261121.json"
CAP_CPU = 600.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
BLOCK = 16


class RoutedSubject(BaseSubject):
    """SubjectR whose per-task readout uses a fixed unit mask (or all units when shared)."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.masks: dict[str, torch.Tensor] = {}

    def train_mode(self) -> None:
        for head in self.phase_heads.values():
            head.train()

    def eval_mode(self) -> None:
        for head in self.phase_heads.values():
            head.eval()

    def set_masks(self, partitioned: bool) -> None:
        self.partitioned = partitioned
        n = self.n_units
        if partitioned:
            self.masks = {
                "cn": torch.tensor([1.0 if i < 16 else 0.0 for i in range(n)]),
                "p24": torch.tensor([1.0 if 16 <= i < 32 else 0.0 for i in range(n)]),
                "p27": torch.tensor([1.0 if i >= 32 else 0.0 for i in range(n)]),
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
        width = (
            head.net[0].in_features
            if hasattr(head.net[0], "in_features")
            else head.net[0].weight.shape[1]
        )
        return head(torch.cat([z, a], dim=-1)[..., :width])


def _acc(subject: RoutedSubject, name: str, ticks: torch.Tensor, labels: torch.Tensor) -> float:
    subject.eval_mode()
    with torch.no_grad():
        logits = subject.logits(
            name, ticks, step_mask=(ticks.abs().sum(-1) > 0).float() if name == "cn" else None
        )
    return float((logits.argmax(-1) == labels).float().mean())


def _acc_p24(subject: RoutedSubject, ticks: torch.Tensor, batch: dict[str, torch.Tensor]) -> float:
    subject.eval_mode()
    with torch.no_grad():
        logits = subject.logits("p24", ticks)
    return float((logits[:, :4].argmax(-1) == batch["val_labels"]).float().mean())


def run_arm(seed: int, partitioned: bool) -> dict:
    torch.set_num_threads(1)
    subject = RoutedSubject(seed)
    subject.set_masks(partitioned)
    b = s1r2.phase_b(subject, seed)
    d = s1r2.phase_d(subject, seed)

    fit_cn = _build_bidirectional_cn_32dim(
        256, seed + 800 + FIT_OFFSET, BI_CN_ADV_FIRST, BI_CN_ADV_SECOND
    )
    fit_p24 = _make_p24_stream(128, seed + 212 + FIT_OFFSET)
    fit_p27 = _make_p27_physics_batch(128, seed + 81 + FIT_OFFSET)
    cn = _acc(subject, "cn", fit_cn["ticks"], fit_cn["labels"])
    p24 = _acc_p24(subject, fit_p24["ticks"], fit_p24)
    p27 = _acc(subject, "p27", fit_p27["ticks"], fit_p27["labels"])

    # specificity: lesion each task's own mask and measure own/other drops
    lesion = {}
    for name, ticks, labels in (
        ("cn", fit_cn["ticks"], fit_cn["labels"]),
        ("p27", fit_p27["ticks"], fit_p27["labels"]),
    ):
        saved = subject.masks[name].clone()
        subject.masks[name].zero_()
        dropped = _acc(subject, name, ticks, labels)
        subject.masks[name].copy_(saved)
        lesion[name] = float(_acc(subject, name, ticks, labels)) - dropped
    saved24 = subject.masks["p24"].clone()
    subject.masks["p24"].zero_()
    p24_drop = _acc_p24(subject, fit_p24["ticks"], fit_p24)
    subject.masks["p24"].copy_(saved24)
    lesion["p24"] = _acc_p24(subject, fit_p24["ticks"], fit_p24) - p24_drop

    # capsule exactness (internal readout)
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
        "b_overall": b["overall_acc"],
        "d_acc": d["accuracy"],
        "cn_fit": cn,
        "p24_fit": p24,
        "p27_fit": p27,
        "lesion_cn": lesion["cn"],
        "lesion_p24": lesion["p24"],
        "lesion_p27": lesion["p27"],
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
            for k in ("a_bi", "a_adv", "b_overall", "d_acc", "cn_fit", "p24_fit", "p27_fit")
        },
        "routed": {
            k: sum(r[k] for r in routed) / len(routed)
            for k in ("a_bi", "a_adv", "b_overall", "d_acc", "cn_fit", "p24_fit", "p27_fit")
        },
    }
    gates = {
        "g0_control": min(r["d_acc"] for r in shared) < 0.65
        or max(0.0, max(r["a_bi"] for r in shared) - min(r["cn_fit"] for r in routed)) > 0.05,
        "g1_coexist": min(r["cn_fit"] for r in routed) >= 0.95
        and min(r["b_overall"] for r in routed) >= 0.85
        and min(r["d_acc"] for r in routed) >= 0.80,
        "g2_old_skill": max(r["a_bi"] for r in routed) - min(r["cn_fit"] for r in routed) <= 0.02,
        "g3_specificity": min(r["lesion_cn"] for r in routed) >= 0.20
        and min(r["lesion_p24"] for r in routed) >= 0.20
        and min(r["lesion_p27"] for r in routed) >= 0.20,
        "g4_capsule": max(r["capsule_diff"] for r in routed) <= 1e-6,
    }
    passed = sum(1 for value in gates.values() if value)
    verdict = "s1rt_stage1_verified" if passed == len(gates) else "s1rt_stage1_partial"
    payload = {
        "schema": "taiji-u2-s1rt-stage1-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "run_under_standing_authorization_pending_owner_ratification",
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "blocks": [16, 16, 16],
            "arms": ["shared", "partitioned"],
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "passed": passed,
        "total_gates": len(gates),
        "verdict": verdict,
        "disclosure": "stage-1 fixed partition is an experiment-level mechanism; zero check/final",
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
    return 0 if verdict == "s1rt_stage1_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
