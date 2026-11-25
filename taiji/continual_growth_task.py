"""Two-phase task-agnostic continual-learning benchmark for N33.3 (`taiji-continual-growth-v1`).

All inputs share ``IN_DIM = 16`` and label space ``{0..5}`` with ZERO ``task_id``:
  * Phase 1 (`y in {0, 1, 2}`): well-separated prototypes in dims `0..7` (noise in `8..15`),
    easily mastered by an 8-unit network without triggering growth.
  * Phase 2 (`y in {3, 4, 5}`): 3-pair nonlinear parity mod 3 over dims `8..13` (noise elsewhere),
    creating genuine capacity pressure on an 8-unit network while solvable by 32 units.
"""

from __future__ import annotations

import torch

TASK_VERSION = "taiji-continual-growth-v1"
IN_DIM = 16
NUM_CLASSES = 6


def _sample_phase1(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 3, (n,), generator=gen, dtype=torch.long)
    protos = torch.tensor(
        [
            [1.6, 1.6, -1.6, 0.0, 1.2, -1.2, 0.0, 0.8],
            [-1.6, 1.6, 1.6, -1.2, 0.0, 1.2, -0.8, 0.0],
            [0.0, -1.6, 1.6, 1.6, -1.2, 0.0, 1.2, -0.8],
        ],
        dtype=torch.float32,
    )
    x_lo = protos[y] + 0.35 * torch.randn((n, 8), generator=gen)
    x_hi = 0.15 * torch.randn((n, 8), generator=gen)
    return torch.cat([x_lo, x_hi], dim=1), y


def _sample_phase2(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (n, 6), generator=gen)
    signs = bits.to(torch.float32) * 2.0 - 1.0  # in {-1, +1}
    p0 = (bits[:, 0] ^ bits[:, 1]).to(torch.long)
    p1 = (bits[:, 2] ^ bits[:, 3]).to(torch.long)
    p2 = (bits[:, 4] ^ bits[:, 5]).to(torch.long)
    y = 3 + ((p0 + p1 + p2) % 3)

    x_lo = 0.15 * torch.randn((n, 8), generator=gen)
    sig = 1.25 * signs + 0.18 * torch.randn((n, 6), generator=gen)
    pad = 0.15 * torch.randn((n, 2), generator=gen)
    x_hi = torch.cat([sig, pad], dim=1)
    return torch.cat([x_lo, x_hi], dim=1), y


def make_split(
    seed: int,
    *,
    n_train: int = 3072,
    n_val: int = 1024,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Generate deterministic Phase 1 and Phase 2 train/val splits."""
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
