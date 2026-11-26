"""Price the pixel-salience readout on a live U2Subject (no training, no source edits).

The design card declares the operator and its wiring contract but leaves two
things open: whether the injection is neutral at birth, and what it costs.  Both
are answered here without touching a pinned file: the readout is attached to a
model *instance* (``add_module`` plus a bound-method replacement of
``branch_context``), so ``taiji/`` keeps its identity and no bundle is invalidated.

Measured:

* ``neutral``: with the zero-initialised projection the first-step log
  probabilities must equal the switch-off ones exactly; after a 1e-3
  perturbation the delta must be non-zero, else the neutral test is decorative.
* ``gradients``: one backward on the first-target-byte loss must leave a non-zero
  gradient on every new tensor.
* ``cost``: forward+backward CPU per example, median of five repeats, switch off
  versus switch on.

The budget table then scales the measured per-example cost by the example-updates
the prototype needed and by the gate's current allowance.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import (  # noqa: E402
    generate_examples,
    make_model,
    output,
    prepare,
)

REPORT = ROOT / "reports/taiji_u2_22_5_a_pixel_salience_price_20261117.json"
SEED = 20261110
REPEATS = 5
PROTOTYPE_EXAMPLE_UPDATES = 4_600_000
GATE_EXAMPLE_UPDATES = 256 * 2
SIDE = 16
MAX_BYTES = 64


class PixelSalienceReadout(nn.Module):
    """Query-conditioned pixel salience with a linear coordinate readout."""

    def __init__(self, width: int = 16, text: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.query = nn.Linear(width, width)
        self.pixel = nn.Linear(3, width)
        self.null = nn.Parameter(torch.zeros(width))
        self.out = nn.Linear(6, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, raster: torch.Tensor, pixel_xy: torch.Tensor, request: torch.Tensor):
        flat = raster.reshape(raster.shape[0], SIDE * SIDE, 3)
        key = torch.tanh(self.pixel(flat))
        text = self.byte(request) + self.position[0, : request.numel()]
        question = torch.tanh(
            self.query(torch.tanh(self.text(torch.cat([text.mean(0), text.amax(0)], -1))))
        )
        score = torch.cat(
            [
                torch.einsum("d,nkd->nk", question, key) / math.sqrt(key.shape[-1]),
                (question @ self.null).reshape(1, 1).expand(raster.shape[0], -1),
            ],
            -1,
        )
        weight = score.softmax(-1)[:, :-1]
        mass = weight.sum(-1)
        x = pixel_xy[..., 0].reshape(1, -1)
        y = pixel_xy[..., 1].reshape(1, -1)
        moment_x = (weight * x).sum(-1)
        moment_y = (weight * y).sum(-1)
        features = torch.stack(
            [
                moment_x / mass.clamp(min=1e-6),
                moment_y / mass.clamp(min=1e-6),
                mass,
                weight.amax(-1),
                moment_x,
                moment_y,
            ],
            -1,
        )
        return self.out(features)


def _pixel_xy(coordinates: torch.Tensor) -> torch.Tensor:
    """Expand the 16 patch centres to the 16x16 pixel grid with fixed sub-offsets."""
    centres = coordinates[:, :2]
    offsets = torch.tensor([-3 / 32, -1 / 32, 1 / 32, 3 / 32])
    rows = []
    for index in range(SIDE):
        span_x = (centres[index][0] + offsets).reshape(-1, 1).expand(4, 4)
        span_y = (centres[index][1] + offsets).reshape(1, -1).expand(4, 4)
        rows.append(torch.stack([span_x, span_y], -1))
    return torch.stack(rows).reshape(4, 4, 4, 4, 2).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 2)


def _install(model, readout: PixelSalienceReadout) -> None:
    model.add_module("pixel_readout", readout)
    base = model.branch_context

    def patched(self, state, goal, action):
        context = base(state, goal, action)
        if state.get("pixel_raster") is None:
            return context
        return context + self.pixel_readout(
            state["pixel_raster"], state["pixel_xy"], state["pixel_request"]
        )

    model.branch_context = types.MethodType(patched, model)


def _state(model, example: dict) -> dict:
    child = dict(prepare(model, example["public"], example["id"]))
    child["pixel_raster"] = example["public"]["observations"]["rgb"][None]
    child["pixel_xy"] = _pixel_xy(example["public"]["coordinates"]["rgb"])
    child["pixel_request"] = torch.tensor(list(example["public"]["request"]), dtype=torch.long)
    return child


def _first(model, example: dict) -> torch.Tensor:
    probability, _, _ = output(model, _state(model, example))
    return probability


def _loss(model, examples: list[dict]) -> torch.Tensor:
    terms = [_first(model, example)[int(example["target"][0])] for example in examples]
    return -torch.stack(terms).mean()


def _cost(model, examples: list[dict]) -> float:
    timings = []
    for _ in range(REPEATS):
        started = time.process_time()
        model.zero_grad(set_to_none=True)
        _loss(model, examples).backward()
        timings.append((time.process_time() - started) / len(examples))
    return sorted(timings)[len(timings) // 2]


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    model = make_model("native", SEED)
    batch = generate_examples("train", 4)[:2]

    before = {name: value.clone() for name, value in model.state_dict().items()}
    off_first = _first(model, batch[0])
    off_cost = _cost(model, batch)

    torch.manual_seed(SEED + 7)
    readout = PixelSalienceReadout()
    _install(model, readout)
    on_first = _first(model, batch[0])
    both = torch.isfinite(on_first) & torch.isfinite(off_first)
    neutral_delta = float((on_first[both] - off_first[both]).abs().max())
    neutral_mismatch = int((torch.isfinite(on_first) != torch.isfinite(off_first)).sum())

    with torch.no_grad():
        readout.out.weight.add_(1e-3)
    shifted = _first(model, batch[0])
    both = torch.isfinite(shifted) & torch.isfinite(off_first)
    perturbed = float((shifted[both] - off_first[both]).abs().max())

    model.zero_grad(set_to_none=True)
    _loss(model, batch).backward()
    gradients = {
        name: None if parameter.grad is None else round(float(parameter.grad.norm()), 10)
        for name, parameter in readout.named_parameters()
    }
    missing = sorted(name for name, norm in gradients.items() if not norm)
    with torch.no_grad():
        readout.out.weight.sub_(1e-3)

    on_cost = _cost(model, batch)
    held = all(
        torch.equal(before[name], value)
        for name, value in model.state_dict().items()
        if name in before
    )
    payload = {
        "schema": "taiji-u2-22-5-a-pixel-salience-price-v1",
        "card": "22.5.0.A",
        "status": "priced_no_training",
        "neutral_at_birth": {
            "max_abs_first_step_delta_zero_init": neutral_delta,
            "entries_with_different_support": neutral_mismatch,
            "max_abs_first_step_delta_after_1e3_perturbation": perturbed,
            "green": neutral_delta == 0.0 and neutral_mismatch == 0,
            "red": perturbed > 0.0,
        },
        "gradients": {"norms": gradients, "zero_grad_tensors": missing},
        "inherited_parameters_untouched": held,
        "readout_parameters": sum(p.numel() for p in readout.parameters()),
        "cost": {
            "cpu_per_example_switch_on": round(on_cost, 6),
            "cpu_per_example_switch_off": round(off_cost, 6),
            "increment": round(on_cost - off_cost, 6),
            "repeats": REPEATS,
            "batch_examples": len(batch),
        },
        "budget": {
            "prototype_example_updates": PROTOTYPE_EXAMPLE_UPDATES,
            "gate_example_updates": GATE_EXAMPLE_UPDATES,
            "cpu_prototype": round(PROTOTYPE_EXAMPLE_UPDATES * on_cost, 1),
            "cpu_gate": round(GATE_EXAMPLE_UPDATES * on_cost, 1),
            "shortfall_ratio": round(PROTOTYPE_EXAMPLE_UPDATES / GATE_EXAMPLE_UPDATES, 1),
            "note": (
                "prototype example-updates are full-batch steps times worlds; this is a scale, "
                "not an optimum"
            ),
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "check_reads": 0,
            "final_reads": 0,
            "source_files_modified": 0,
        },
        "source_pins": {
            "scripts/training/price_a_pixel_salience_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "neutral_delta="
        + str(neutral_delta)
        + " red_delta="
        + str(perturbed)
        + " cpu_on="
        + str(round(on_cost, 6))
        + " cpu_off="
        + str(round(off_cost, 6))
        + " missing_grads="
        + str(missing)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
