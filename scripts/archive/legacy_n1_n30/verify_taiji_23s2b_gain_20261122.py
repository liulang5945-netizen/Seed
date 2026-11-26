"""23-S.2b: equal-compute GRU/TF baselines vs routed arm (3 seeds, <=1500 CPU, 0 check)."""

import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, r"E:\Seed")
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.training.verify_taiji_cognitive_loop_20261121 import (
    _make_p24_stream,
    _make_p27_physics_batch,
)

CN_STEPS, P24_STEPS, P27_STEPS = 1000, 900, 1300
B = 48
SEEDS = (20261120, 20261121, 20261122)
Q_R = {20261120: 0.8359375, 20261121: 0.8828125, 20261122: 0.859375}


def get_cn():
    from scripts.training.verify_taiji_23s2a_gain_20261122 import cn_split

    return cn_split


class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.cn = nn.Linear(64, 8)
        self.p24 = nn.Linear(64, 5)
        self.p27 = nn.Linear(64, 4)

    def forward(self, ticks, mask, task):
        out, _ = self.gru(ticks)
        idx = mask.float().sum(-1).long().clamp(min=1) - 1
        pooled = out[torch.arange(out.shape[0]), idx]
        return getattr(self, task)(pooled)


class TFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(48, 64)
        layer = nn.TransformerEncoderLayer(64, 2, 128, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.cn = nn.Linear(64, 8)
        self.p24 = nn.Linear(64, 5)
        self.p27 = nn.Linear(64, 4)

    def forward(self, ticks, mask, task):
        x = self.proj(ticks) + _pos(ticks.shape[1], 64, ticks.dtype)
        out = self.enc(x, src_key_padding_mask=~mask.bool())
        pooled = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return getattr(self, task)(pooled)


def _pos(steps, dim, dtype):
    pos = torch.arange(steps, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=dtype) * (-math.log(10000.0) / dim))
    pe = torch.zeros(steps, dim, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


def train_phase(model, task, data, opt, epochs):
    model.train()
    for k in range(epochs):
        b = data(k)
        ticks = b["ticks"]
        mask = (ticks.abs().sum(-1) > 0).float()
        logits = model(ticks, mask, task)
        if task == "p24":
            loss = F.cross_entropy(
                logits[:, :4], b["val_labels"]
            ) + F.binary_cross_entropy_with_logits(logits[:, 4], b["conflict_labels"])
        else:
            loss = F.cross_entropy(logits, b["labels"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def evaluate(model, seed):
    model.eval()
    cn_split = get_cn()
    fit_cn = cn_split(256, seed + 900 + 10_000_000, "unseen")
    m = (fit_cn["ticks"].abs().sum(-1) > 0).float()
    cn = float((model(fit_cn["ticks"], m, "cn").argmax(-1) == fit_cn["labels"]).float().mean())
    fit24 = _make_p24_stream(128, seed + 212 + 10_000_000)
    m24 = (fit24["ticks"].abs().sum(-1) > 0).float()
    lg = model(fit24["ticks"], m24, "p24")
    p24 = float((lg[:, :4].argmax(-1) == fit24["val_labels"]).float().mean())
    fit27 = _make_p27_physics_batch(128, seed + 81 + 10_000_000)
    m27 = (fit27["ticks"].abs().sum(-1) > 0).float()
    p27 = float((model(fit27["ticks"], m27, "p27").argmax(-1) == fit27["labels"]).float().mean())
    return {"cn_unseen": cn, "p24": p24, "p27": p27, "q": min(cn, p24, p27)}


def run(seed, arm):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = GRUModel() if arm == "gru" else TFModel()
    cn_split = get_cn()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0)
    train_phase(model, "cn", lambda k: cn_split(B, seed * 6000 + k, "train"), opt, CN_STEPS)
    train_phase(model, "p24", lambda k: _make_p24_stream(B, seed * 25000 + k), opt, P24_STEPS)
    train_phase(
        model, "p27", lambda k: _make_p27_physics_batch(B, seed * 41000 + k), opt, P27_STEPS
    )
    return {"arm": arm, "seed": seed, **evaluate(model, seed)}


def main():
    started = time.process_time()
    crit = json.loads(
        Path(r"E:\Seed\plans/manifests/taiji_u2_23s2b_criteria_20261122.json").read_text(
            encoding="utf-8"
        )
    )
    body = {k: v for k, v in crit.items() if k != "criteria_sha256"}
    canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canon).hexdigest() == crit["criteria_sha256"]
    rows = [run(seed, arm) for seed in SEEDS for arm in ("gru", "tf")]
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    gains = []
    for seed in SEEDS:
        best_base = max(by_seed[seed]["gru"]["q"], by_seed[seed]["tf"]["q"])
        gains.append(Q_R[seed] - best_base)
    mean_gain = sum(gains) / len(gains)
    verified = mean_gain >= crit["gain_mean_min"] and min(gains) >= crit["gain_min"]
    payload = {
        "schema": "taiji-u2-23s2b-gain-v1",
        "criteria_sha256": crit["criteria_sha256"],
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "q_routed": Q_R,
        "rows": rows,
        "gains": gains,
        "mean_gain": mean_gain,
        "verdict": "23s2b_gain_verified" if verified else "23s2b_gain_absent",
        "run_conditions": {
            "threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
        },
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    Path(r"E:\Seed\reports\taiji_u2_23s2b_gain_20261122.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "gains": [round(g, 4) for g in gains],
                "mean_gain": round(mean_gain, 4),
                "rows": rows,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
