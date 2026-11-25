"""Harder two-phase continual-growth task for N34.2 growth-benefit testing (`taiji-continual-growth-v2`).

N33.3's `taiji-continual-growth-v1` saturated: both growth and tuned same-capacity preallocation
reached 0.9995, so the growth net benefit vs preallocation was exactly 0.0. To make a fair
net-benefit measurement possible, v2 raises phase-1 interference:

* Phase 1 (`y in {0..5}`): 6 noisy prototypes sharing the same 8 signal dims (label noise and
  overlapping prototypes), so a 32-unit preallocated net can overfit the noisy P1 signal and a
  32-unit net adapted at high LR for P2 tends to forget P1 — while the growth path freezes mature
  units and adapts only newly born ones.
* Phase 2 (`y in {6..11}`): 4-pair nonlinear parity mod 3 over 8 signal dims, harder than v1's
  3-pair mod 3, keeping genuine capacity pressure on 8 units.

No ``task_id`` routing; identical interface to v1 (`make_split`).
"""

from __future__ import annotations

import torch

TASK_VERSION = "taiji-continual-growth-v2"
IN_DIM = 16
NUM_CLASSES = 12

_P1_PROTOS = torch.tensor(
    [
        [1.5, 1.5, -1.5, 0.0, 1.1, -1.1, 0.0, 0.7],
        [-1.5, 1.5, 1.5, -1.1, 0.0, 1.1, -0.7, 0.0],
        [0.0, -1.5, 1.5, 1.5, -1.1, 0.0, 1.1, -0.7],
        [1.1, 0.0, -1.5, -1.5, 1.1, 1.1, 0.0, -0.9],
        [-1.1, -1.5, 0.0, 1.5, 0.0, -1.1, 1.1, 0.9],
        [0.7, 1.1, 1.1, 0.0, -1.5, 0.0, -1.5, 0.0],
    ],
    dtype=torch.float32,
)


def _sample_phase1(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 6, (n,), generator=gen, dtype=torch.long)
    x_lo = _P1_PROTOS[y] + 0.60 * torch.randn((n, 8), generator=gen)
    # 10% label noise keeps a large preallocated net from simply memorising P1.
    flip = torch.rand((n,), generator=gen) < 0.10
    y_noisy = torch.where(flip, torch.randint(0, 6, (n,), generator=gen, dtype=torch.long), y)
    x_hi = 0.20 * torch.randn((n, 8), generator=gen)
    return torch.cat([x_lo, x_hi], dim=1), y_noisy


def _sample_phase2(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (n, 8), generator=gen)
    signs = bits.to(torch.float32) * 2.0 - 1.0
    pairs = [
        (bits[:, 0] ^ bits[:, 1]).to(torch.long),
        (bits[:, 2] ^ bits[:, 3]).to(torch.long),
        (bits[:, 4] ^ bits[:, 5]).to(torch.long),
        (bits[:, 6] ^ bits[:, 7]).to(torch.long),
    ]
    y = 6 + ((pairs[0] + pairs[1] + pairs[2] + pairs[3]) % 3)

    x_lo = 0.20 * torch.randn((n, 8), generator=gen)
    sig = 1.25 * signs + 0.16 * torch.randn((n, 8), generator=gen)
    return torch.cat([x_lo, sig], dim=1), y


def make_split(
    seed: int,
    *,
    n_train: int = 3072,
    n_val: int = 1024,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    gen = torch.Generator().manual_seed(int(seed) % (2**63))
    p1_tr_x, p1_tr_y = _sample_phase1(n_train, gen)
    p1_va_x, p1_va_y = _sample_phase1(n_val, gen)
    p2_tr_x, p2_tr_y = _sample_phase2(n_train, gen)
    p2_va_x, p2_va_y = _sample_phase2(n_val, gen)
    return {
        "phase1": {
            "train": {"x": p1_tr_x, "y": p1_tr_y},
            "val": {"x": p1_va_x, "y": p1_va_y},
        },
        "phase2": {
            "train": {"x": p2_tr_x, "y": p2_tr_y},
            "val": {"x": p2_va_x, "y": p2_va_y},
        },
    }
