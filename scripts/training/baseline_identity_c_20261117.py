"""Plan C baseline identity: a parameter-matched, weight-tied compute-heavy transformer.

The frozen transformer block is applied K times without adding parameters. This
keeps the parameter band and raises forward+backward FLOPs toward the U2 price,
so the 256/24 dose envelope can host a compute-matched comparison. The class
lives outside ``taiji/`` and is a new identity version; the old identity manifest
and the U2 arm are untouched.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.baseline_identity_22_4_4 import (  # noqa: E402
    BATCH,
    DEVICE,
    DTYPE,
    GOAL,
    SEED,
    STEPS,
    SharedFrontend,
    apply_frozen_init,
    fragment_inputs,
    normalized_sha256,
    param_counts,
    state_dict_digest,
)

REPORT = ROOT / "reports/taiji_u2_22_5_c_baseline_identity_20261117.json"
MANIFEST = ROOT / "plans/manifests/taiji_u2_22_4_4_baseline_identity_c_20261117.json"
FLOPS_REPORT = ROOT / "reports/taiji_u2_22_5_0_flops_parity_20261117.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_22_4_4_price_20261117.json"
ANCHOR = 22742
TOLERANCE = 0.10
PARAM_BAND = (ANCHOR * (1 - TOLERANCE), ANCHOR * (1 + TOLERANCE))
FLOP_RATIO_CAP = 256 / 24
WIDTH = 28


class WeightTiedTransformerBaseline(nn.Module):
    architecture_id = "weight_tied_transformer_sequence"

    def __init__(self, width: int, applications: int, heads: int = 4) -> None:
        super().__init__()
        if width < 1 or applications < 1:
            raise ValueError("invalid_weight_tied_configuration")
        self.width = width
        self.applications = applications
        self.heads = heads
        self.frontend = SharedFrontend()
        self.record_in = nn.Linear(32, width)
        self.position = nn.Parameter(torch.randn(STEPS, width))
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.byte_head = nn.Linear(width + 4, 257)
        apply_frozen_init(self)

    def forward(self, observations: dict, targets: torch.Tensor, text_context: torch.Tensor):
        sequence = self.frontend.step_tokens(observations, text_context)
        hidden = self.record_in(sequence) + self.position[None]
        mask = torch.triu(torch.ones(STEPS, STEPS, dtype=torch.bool, device=DEVICE), diagonal=1)
        for _ in range(self.applications):
            attended, _ = self.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
            hidden = self.norm_attention(hidden + attended)
            hidden = self.norm_feedforward(hidden + self.feedforward(hidden))
        previous = self.frontend.embedding(torch.full((BATCH,), 255, dtype=torch.long))
        logits = self.byte_head(torch.cat([hidden[:, -1], previous], -1))
        return functional.cross_entropy(logits, targets)


def build(apps: int) -> nn.Module:
    torch.manual_seed(SEED)
    return WeightTiedTransformerBaseline(WIDTH, apps).to(dtype=DTYPE, device=DEVICE).eval()


def measure(apps: int) -> dict:
    model = build(apps)
    observations, targets = fragment_inputs()
    with torch.no_grad():
        text_context = model.frontend.text_units(GOAL)
    counter = FlopCounterMode(display=False)
    with counter:
        loss = model(observations, targets, text_context)
        loss.backward()
    flops = int(counter.get_total_flops())
    total, tensors = param_counts(model)
    # Repeat the identical forward+backward, no optimizer, for a stable wall mean.
    samples = []
    for _ in range(5):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        value = model(observations, targets, text_context)
        value.backward()
        samples.append(time.perf_counter() - started)
        model.zero_grad(set_to_none=True)
    return {
        "applications": apps,
        "param_total": total,
        "param_tensors": tensors,
        "flops": flops,
        "wall_median": round(statistics.median(samples), 6),
        "loss": float(loss.detach()),
    }


def main() -> int:
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    u2_flops = json.loads(FLOPS_REPORT.read_text(encoding="utf-8"))["counts"]["u2_reference"]
    u2_wall = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))["price_table"]["components"][
        "u2_reference"
    ]["single_learning"]["wall"]
    target_flops = u2_flops / FLOP_RATIO_CAP
    scan = [measure(apps) for apps in range(1, 7)]
    eligible = [
        row
        for row in scan
        if row["param_total"] <= PARAM_BAND[1]
        and row["param_total"] >= PARAM_BAND[0]
        and row["flops"] >= target_flops
    ]
    selected = (
        min(eligible, key=lambda row: (row["flops"], row["applications"])) if eligible else None
    )
    model = build(selected["applications"]) if selected else None
    report = {
        "schema": "taiji-u2-22-5-c-baseline-identity-v1",
        "card": "22.5.0.C",
        "status": "selected" if selected else "compute_band_not_met",
        "anchor": ANCHOR,
        "param_band": list(PARAM_BAND),
        "u2_flops": u2_flops,
        "u2_single_learning_wall": u2_wall,
        "target_flops": round(target_flops, 1),
        "flop_ratio_cap": FLOP_RATIO_CAP,
        "scan": scan,
        "selected": selected,
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 6),
            "wall": round(time.perf_counter() - started_wall, 6),
        },
        "discipline": {
            "optimizer_steps": 0,
            "project_data_reads": 0,
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if selected:
        total, tensors = param_counts(model)
        first = state_dict_digest(model)
        second = state_dict_digest(build(selected["applications"]))
        if first != second or total != selected["param_total"]:
            raise RuntimeError("weight-tied identity construction drift")
        manifest = {
            "schema": "taiji-u2-22.4.4-baseline-identity-c-v1",
            "architecture_id": "weight_tied_transformer_sequence",
            "status": "frozen",
            "supersedes": "plans/manifests/taiji_u2_22_4_4_baseline_identity_20261111.json",
            "scope": "plan C comparable-cost baseline; U2 arm unchanged",
            "conditions": {
                "dtype": "float32",
                "device": "cpu",
                "threads": 1,
                "torch_version": torch.__version__,
                "training_runs": 0,
                "optimizer_steps": 0,
            },
            "param_total": total,
            "param_tensors": tensors,
            "param_inventory": [
                [name, list(parameter.shape), str(parameter.dtype).removeprefix("torch.")]
                for name, parameter in model.named_parameters()
            ],
            "init": {
                "rule": "same as baseline_identity_22_4_4; same seed",
                "seed": SEED,
                "post_init_state_dict_sha256": first,
                "second_construction_sha256": second,
                "same_seed_digest_equal": first == second,
            },
            "architecture": {
                "module_path": "scripts/training/baseline_identity_c_20261117.py",
                "module_sha256": normalized_sha256(Path(__file__).resolve()),
                "class_name": "WeightTiedTransformerBaseline",
                "architecture_id": "weight_tied_transformer_sequence",
                "width": WIDTH,
                "applications": selected["applications"],
                "weight_tied": True,
                "layers": 1,
                "attention_heads": 4,
                "mask_type": "causal_upper_triangular_bool",
                "dropout": 0.0,
                "state_tensors": 0,
                "reused_u2_components": [
                    "encoder",
                    "metadata",
                    "modality",
                    "embedding",
                    "text_input",
                    "convolutions",
                ],
                "new_components": [
                    "record_in",
                    "position",
                    "attention",
                    "norm_attention",
                    "norm_feedforward",
                    "feedforward",
                    "byte_head",
                ],
            },
            "recipe": {
                "optimizer": {
                    "type": "Adam",
                    "lr": 0.001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "weight_decay": 0.0,
                },
                "batch": BATCH,
                "sequence_length": STEPS,
                "steps": 1,
                "recipe_delta": [
                    "the same transformer block is applied K times with shared parameters"
                ],
                "same_as_u2_arm": True,
            },
            "information_rights": {
                "visible_input": "same fragment as the frozen identity",
                "feedback_channels": [
                    "observation",
                    "train_label",
                    "online_correction",
                    "help_reply",
                    "oracle_score",
                ],
                "oracle_not_fedback": True,
                "equal_to_u2_arm": True,
            },
            "seeds": {
                "fixed_seeds": [20261110, 20261111, 20261112],
                "mapping": {
                    seed: {"init_seed": seed, "data_order_seed": seed}
                    for seed in (20261110, 20261111, 20261112)
                },
            },
            "parity": {
                "param_delta_pct": round(100 * abs(total - ANCHOR) / ANCHOR, 6),
                "flops": selected["flops"],
                "flops_ratio_to_u2": round(u2_flops / selected["flops"], 6),
                "wall": selected["wall_median"],
                "wall_ratio_to_u2": round(u2_wall / selected["wall_median"], 6),
                "flop_ratio_cap": FLOP_RATIO_CAP,
                "compute_parity_available": u2_flops / selected["flops"] <= FLOP_RATIO_CAP,
            },
            "measurement_report": str(REPORT.relative_to(ROOT)),
        }
        MANIFEST.write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n")
        report["manifest"] = str(MANIFEST.relative_to(ROOT))
        report["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "C baseline status="
        + report["status"]
        + " selected="
        + json.dumps(selected, ensure_ascii=False)
        + " target_flops="
        + str(report["target_flops"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
