"""Frozen evaluation framework for the P0 diagnostic line (owner-requested common functions).

All readouts are explicitly placed in eval mode, take fixed seeds, and report accuracy on
held-out data.  Readouts:
  * closed_form_probe  -- least-squares linear readout fitted on train features
  * prototype_acc      -- nearest class mean using a 768-sample support set (16 x 48)
  * logreg_acc         -- Adam-trained linear readout to convergence (different readout)
  * mlp_transfer_acc   -- fresh PhaseHead-like MLP trained `steps` interactions (the old G4b)
  * ideal_reference    -- 16-step MLP on noisy label one-hot features (16-step ceiling)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CLASSES = 8


def eval_all(*modules) -> None:
    for m in modules:
        if m is not None:
            m.eval()


def _aug(feat: torch.Tensor) -> torch.Tensor:
    return torch.cat([feat, torch.ones(feat.shape[0], 1)], dim=-1)


def closed_form_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    classes: int = CLASSES,
) -> float:
    a_tr = _aug(x_train)
    a_te = _aug(x_test)
    y = F.one_hot(y_train, classes).float()
    w = torch.linalg.lstsq(a_tr, y).solution
    return float(((a_te @ w).argmax(-1) == y_test).float().mean())


def prototype_acc(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    classes: int = CLASSES,
) -> float:
    means = torch.stack([x_support[y_support == c].mean(dim=0) for c in range(classes)])
    d = torch.cdist(x_test, means)
    return float((d.argmin(-1) == y_test).float().mean())


def logreg_acc(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    steps: int = 400,
    lr: float = 2e-2,
    classes: int = CLASSES,
    seed: int = 0,
) -> float:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Linear(x_train.shape[-1], classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        loss = F.cross_entropy(head(x_train), y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return float((head(x_test).argmax(-1) == y_test).float().mean())


def mlp_transfer_acc(
    x_support: torch.Tensor,
    y_support: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    steps: int = 16,
    batch: int = 48,
    lr: float = 6e-3,
    width: int = 64,
    classes: int = CLASSES,
    seed: int = 0,
) -> float:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = torch.nn.Sequential(
            torch.nn.Linear(x_support.shape[-1], width),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(width, classes),
        )
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    head.train()
    for _ in range(steps):
        idx = torch.randint(0, x_support.shape[0], (min(batch, x_support.shape[0]),))
        loss = F.cross_entropy(head(x_support[idx]), y_support[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return float((head(x_test).argmax(-1) == y_test).float().mean())


def ideal_reference(
    *, steps: int = 16, dim: int = 48, noise: float = 0.5, classes: int = CLASSES, seed: int = 0
) -> dict:
    """16-step transfer ceiling on ideal (noisy label one-hot) features."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        y_support = torch.randint(0, classes, (steps * 48,))
        y_test = torch.randint(0, classes, (256,))
        proj = torch.randn(classes, dim)  # one shared projection for support and test
        x_support = F.one_hot(y_support, classes).float() @ proj
        x_test = F.one_hot(y_test, classes).float() @ proj
        x_support = x_support + noise * torch.randn_like(x_support)
        x_test = x_test + noise * torch.randn_like(x_test)
    acc = mlp_transfer_acc(x_support, y_support, x_test, y_test, steps=steps, seed=seed + 1)
    lin = closed_form_probe(x_support, y_support, x_test, y_test)
    return {"mlp16": acc, "linear": lin}
