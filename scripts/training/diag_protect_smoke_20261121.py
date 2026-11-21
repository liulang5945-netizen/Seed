"""Parameter-level protection smoke test (1 seed, <=300 CPU, 0 check).

Stage-1b showed task gating alone does not protect learned units (peak~1.0 then overwritten).
This test masks gradients so that, during each task's phase, only that task's unit block and
the matching read_gate rows update; other blocks/rows are frozen.  Goal: sequential P-A/P-B/P-D
all >=0.80 with old-skill damage <=0.02.  Run under standing authorization; zero check/final.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.verify_taiji_cognitive_loop_20261121 import (  # noqa: E402
    BI_CN_FIT_FIRST,
    BI_CN_FIT_SECOND,
    BI_CN_TRAIN_FIRST,
    BI_CN_TRAIN_SECOND,
    _build_bidirectional_cn_32dim,
    _make_p24_stream,
    _make_p27_physics_batch,
)
from scripts.training.verify_taiji_s1rt_stage1b_20261121 import Subject72  # noqa: E402

OUT = ROOT / "reports/taiji_u2_protect_smoke_20261121.json"
CAP_CPU = 300.0
SEED = 20261120
FIT_OFFSET = 10_000_000
BATCH = 48
FIT = 128
BLOCKS = {"cn": (0, 24), "p24": (24, 48), "p27": (48, 72)}


def mask_grads(subject: Subject72, keep: str) -> None:
    lo, hi = BLOCKS[keep]
    n = subject.n_units
    for _name, parameter in subject.graph.selective_units.named_parameters():
        if parameter.grad is None or parameter.ndim == 0:
            continue
        if parameter.shape[0] == n:
            parameter.grad[:lo] = 0.0
            parameter.grad[hi:] = 0.0
        if parameter.ndim == 2 and parameter.shape[1] == n:
            parameter.grad[:, :lo] = 0.0
            parameter.grad[:, hi:] = 0.0
    rg = subject.graph.read_gate
    if rg.weight.grad is not None:
        rows = list(range(lo, hi)) + list(range(n + lo, n + hi))
        keep_rows = set(rows)
        for idx in range(rg.weight.grad.shape[0]):
            if idx not in keep_rows:
                rg.weight.grad[idx] = 0.0
                rg.bias.grad[idx] = 0.0


def step_protected(subject: Subject72, opt, loss, keep: str) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    mask_grads(subject, keep)
    torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
    opt.step()


def run(seed: int) -> dict:
    torch.set_num_threads(1)
    subject = Subject72(seed)
    subject.set_masks(True)
    fit_cn = _build_bidirectional_cn_32dim(
        256, seed + 700 + FIT_OFFSET, BI_CN_FIT_FIRST, BI_CN_FIT_SECOND
    )
    fit_p24 = _make_p24_stream(FIT, seed + 212 + FIT_OFFSET)
    fit_p27 = _make_p27_physics_batch(FIT, seed + 81 + FIT_OFFSET)
    trace = {}

    def eval_cn() -> float:
        subject.eval_mode()
        with torch.no_grad():
            sm = (fit_cn["ticks"].abs().sum(-1) > 0).float()
            return float(
                (subject.logits("cn", fit_cn["ticks"], sm).argmax(-1) == fit_cn["labels"])
                .float()
                .mean()
            )

    def eval_p24() -> float:
        subject.eval_mode()
        with torch.no_grad():
            logits = subject.logits("p24", fit_p24["ticks"])
            return float((logits[:, :4].argmax(-1) == fit_p24["val_labels"]).float().mean())

    def eval_p27() -> float:
        subject.eval_mode()
        with torch.no_grad():
            return float(
                (subject.logits("p27", fit_p27["ticks"]).argmax(-1) == fit_p27["labels"])
                .float()
                .mean()
            )

    # unused internal readout stays frozen
    for parameter in subject.graph.selective_readout.parameters():
        parameter.requires_grad_(False)
    for name, head in subject.phase_heads.items():
        for parameter in head.parameters():
            parameter.requires_grad_(name == "cn")
    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=6e-3, weight_decay=0.0)
    for step in range(1000):
        batch = _build_bidirectional_cn_32dim(
            BATCH, seed * 6000 + step, BI_CN_TRAIN_FIRST, BI_CN_TRAIN_SECOND
        )
        sm = (batch["ticks"].abs().sum(-1) > 0).float()
        loss = F.cross_entropy(subject.logits("cn", batch["ticks"], sm), batch["labels"])
        step_protected(subject, opt, loss, "cn")
    trace["cn_after_cn"] = eval_cn()
    # freeze shared input mixing after the first task
    for parameter in subject.graph.cross_channel_mix.parameters():
        parameter.requires_grad_(False)

    for name in ("p24", "p27"):
        for parameter in subject.phase_heads[name].parameters():
            parameter.requires_grad_(True)

    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=0.0)
    for step in range(900):
        batch = _make_p24_stream(BATCH, seed * 25000 + step)
        logits = subject.logits("p24", batch["ticks"])
        loss = F.cross_entropy(logits[:, :4], batch["val_labels"])
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, 4], batch["conflict_labels"])
        step_protected(subject, opt, loss, "p24")
    trace["cn_after_p24"] = eval_cn()
    trace["p24_after_p24"] = eval_p24()

    subject.train_mode()
    opt = torch.optim.Adam(subject.params(), lr=7e-3, weight_decay=0.0)
    for step in range(800):
        batch = _make_p27_physics_batch(BATCH, seed * 41000 + step)
        loss = F.cross_entropy(subject.logits("p27", batch["ticks"]), batch["labels"])
        step_protected(subject, opt, loss, "p27")
    trace["cn_final"] = eval_cn()
    trace["p24_final"] = eval_p24()
    trace["p27_final"] = eval_p27()
    damage = max(
        0.0, trace["cn_after_cn"] - trace["cn_final"], trace["p24_after_p24"] - trace["p24_final"]
    )
    return {"trace": trace, "damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    row = run(SEED)
    passed = (
        row["trace"]["cn_final"] >= 0.95
        and row["trace"]["p24_final"] >= 0.80
        and row["trace"]["p27_final"] >= 0.80
        and row["damage"] <= 0.02
    )
    payload = {
        "schema": "taiji-u2-protect-smoke-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "run_under_standing_authorization_pending_owner_ratification",
        "seed": SEED,
        "trace": row["trace"],
        "damage": row["damage"],
        "verdict": (
            "parameter_protection_smoke_passed" if passed else "parameter_protection_smoke_failed"
        ),
        "disclosure": "one seed; parameter-row gradient masking + read_gate row masking; zero check/final",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
