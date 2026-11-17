"""A-gate readout: is the gap lexicon acquisition or query binding?

The ceiling probe showed that a converged wide head sits at 0.510 (three-body
fit) / 0.635 (1-2 body fit) on the frozen 768-world training split, i.e. 0.80 is
not reachable in the current regime by this head family.  That leaves two very
different diagnoses:

* the head has to learn the Chinese colour lexicon ("红/蓝/绿/..." -> a palette
  entry) from 768 worlds at the same time as the reading rule, and it spends its
  capacity on the first; or
* even when the referent is handed to it, the token set does not support binding
  the asked entity to a position, so the reading rule itself is the wall.

This probe holds the head, the tokens (16 patches of 48 rgb values plus 2-d
coordinates), the worlds, the step rule and the seeds fixed, and changes only the
query input:

* ``bytes``  -- the raw request bytes through the same byte/position pool that
  the capacity head used;
* ``entity`` -- a 6-dim one-hot of the asked entity, i.e. the colour-name prior
  that the baseline ``locate_public`` is declared to receive.

Declared exits:

1. ``not_convergent``: any arm misses 0.99 train accuracy inside the step cap;
   then no verdict is read.
2. ``lexicon_is_binding``: ``entity`` median fit >= 0.80 and ``bytes`` < 0.80.
3. ``query_binding_is_binding``: both medians < 0.80.
4. ``query_input_not_binding``: ``bytes`` median fit >= 0.80.

No U2 optimizer step, no check read and no final read occur here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.probe_a_gate_ceiling_20261117 import (  # noqa: E402
    EVAL_EVERY,
    STEP_CAP,
    TARGET_TRAIN,
    _features,
)
from scripts.training.probe_a_query_bind_head_20261117 import (  # noqa: E402
    HEAD_SEEDS,
    MAX_BYTES,
    BindBlock,
    _confusion,
)
from taiji.u2_task_episode import NAMES  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_a_gate_query_binding_20261117.json"
SUFFIX = "现在在哪里？"


class QueryHead(nn.Module):
    """Identical token path for both arms; only the query encoder differs."""

    def __init__(self, token_dim: int, query_dim: int, width: int = 64, text: int = 24) -> None:
        super().__init__()
        self.token = nn.Linear(token_dim, width)
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.entity = nn.Linear(query_dim, width)
        self.null_token = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([BindBlock(width), BindBlock(width)])
        self.output = nn.Linear(width, 4)

    def forward(self, tokens, request, mask, onehot, source: str):
        token = torch.tanh(self.token(tokens))
        null = self.null_token.expand(token.shape[0], 1, -1)
        token = torch.cat([token, null], 1)
        if source == "bytes":
            text = self.byte(request) + self.position[:, : request.shape[1]]
            text = text.masked_fill(~mask[..., None], 0.0)
            mean = text.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            peak = text.masked_fill(~mask[..., None], -1e4).amax(1)
            query = torch.tanh(self.text(torch.cat([mean, peak], -1)))
        else:
            query = torch.tanh(self.entity(onehot))
        for block in self.blocks:
            query = block(token, query)
        return self.output(query)


def _onehot(requests: list[bytes]) -> torch.Tensor:
    rows = torch.zeros(len(requests), len(NAMES))
    for index, raw in enumerate(requests):
        text = raw.decode("utf-8")
        if not text.endswith(SUFFIX):
            raise ValueError("unexpected_request_form")
        rows[index, NAMES.index(text[: -len(SUFFIX)])] = 1.0
    return rows


def _fit(data: dict, seed: int, source: str) -> dict:
    started = time.process_time()
    torch.manual_seed(seed)
    head = QueryHead(data["raw"].shape[2], len(NAMES))
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    steps = 0
    accuracy = 0.0
    while steps < STEP_CAP:
        optimizer.zero_grad(set_to_none=True)
        logits = head(data["raw"], data["request"], data["mask"], data["onehot"], source)
        loss = functional.cross_entropy(logits, data["label"])
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % EVAL_EVERY == 0:
            with torch.no_grad():
                accuracy = (
                    (
                        head(
                            data["raw"], data["request"], data["mask"], data["onehot"], source
                        ).argmax(-1)
                        == data["label"]
                    )
                    .float()
                    .mean()
                    .item()
                )
            if accuracy >= TARGET_TRAIN:
                break
    head.eval()
    with torch.no_grad():
        train_logits = head(data["raw"], data["request"], data["mask"], data["onehot"], source)
    return {
        "seed": seed,
        "steps": steps,
        "train_accuracy": (train_logits.argmax(-1) == data["label"]).float().mean().item(),
        "train_confusion": _confusion(train_logits.argmax(-1), data["label"]),
        "cpu": round(time.process_time() - started, 4),
        "head": head,
    }


def _score(head, data: dict, source: str) -> tuple[float, list[list[int]]]:
    with torch.no_grad():
        logits = head(data["raw"], data["request"], data["mask"], data["onehot"], source)
    return (logits.argmax(-1) == data["label"]).float().mean().item(), _confusion(
        logits.argmax(-1), data["label"]
    )


def _median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def _verdict(arms: dict) -> str:
    if not all(row["train_accuracy"] >= 0.99 for arm in arms.values() for row in arm["results"]):
        return "not_convergent"
    if arms["bytes_fit3"]["median_fit_accuracy"] >= 0.80:
        return "query_input_not_binding"
    if arms["entity_fit3"]["median_fit_accuracy"] >= 0.80:
        return "lexicon_is_binding"
    return "query_binding_is_binding"


def main() -> int:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    rows = _features()
    for key in ("train12", "train3072", "fit3", "fit12"):
        packed = rows[key]["request"].tolist()
        mask = rows[key]["mask"].tolist()
        decoded = [
            bytes([value for value, keep in zip(row, flags, strict=True) if keep])
            for row, flags in zip(packed, mask, strict=True)
        ]
        rows[key]["onehot"] = _onehot(decoded)
    arms = {}
    for name, source, split in (
        ("bytes_fit3", "bytes", "fit3"),
        ("entity_fit3", "entity", "fit3"),
        ("bytes_fit12", "bytes", "fit12"),
        ("entity_fit12", "entity", "fit12"),
    ):
        results = []
        for seed in HEAD_SEEDS:
            fitted = _fit(rows["train12"], seed, source)
            head = fitted.pop("head")
            accuracy, confusion = _score(head, rows[split], source)
            fitted["fit_accuracy"] = accuracy
            fitted["fit_confusion"] = confusion
            results.append(fitted)
        arms[name] = {
            "query": source,
            "fit_split": split,
            "results": results,
            "median_train_accuracy": _median([row["train_accuracy"] for row in results]),
            "median_fit_accuracy": _median([row["fit_accuracy"] for row in results]),
        }
    verdict = _verdict(arms)
    payload = {
        "schema": "taiji-u2-22-5-a-gate-query-binding-v1",
        "card": "22.5.0.A",
        "verdict": verdict,
        "why": (
            "the converged ceiling on the frozen 768-world split is 0.510 fit, so the "
            "next question is which ingredient is missing: the colour lexicon or the "
            "query-to-token binding.  Only the query encoder changes between arms."
        ),
        "conditions": {
            "head": "QueryHead: patch tokens width 64, two BindBlocks, null slot",
            "query_bytes": "byte embedding 24 + position, mean/peak pool",
            "query_entity": "6-dim one-hot of the asked entity (declared prior)",
            "train_split": "train12 (768 worlds, 1-2 bodies)",
            "step_cap": STEP_CAP,
            "target_train_accuracy": TARGET_TRAIN,
            "evaluate_every": EVAL_EVERY,
            "seeds": list(HEAD_SEEDS),
        },
        "arms": arms,
        "medians": {name: arm["median_fit_accuracy"] for name, arm in arms.items()},
        "declared_verdict": {
            "not_convergent": "any arm below 0.99 train accuracy at the step cap",
            "lexicon_is_binding": "entity median fit >= 0.80 and bytes < 0.80",
            "query_binding_is_binding": "both medians < 0.80",
            "query_input_not_binding": "bytes median fit >= 0.80",
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": sum(
                row["steps"] for arm in arms.values() for row in arm["results"]
            ),
            "check_reads": 0,
            "final_reads": 0,
        },
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.perf_counter() - started_wall, 4),
        },
        "source_pins": {
            "scripts/training/probe_a_gate_query_binding_20261117.py": hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
            "scripts/training/probe_a_gate_ceiling_20261117.py": hashlib.sha256(
                (ROOT / "scripts/training/probe_a_gate_ceiling_20261117.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
            "scripts/training/probe_a_query_bind_head_20261117.py": hashlib.sha256(
                (ROOT / "scripts/training/probe_a_query_bind_head_20261117.py")
                .read_text(encoding="utf-8-sig")
                .encode()
            ).hexdigest(),
        },
    }
    REPORT.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "query-binding verdict="
        + verdict
        + " medians="
        + json.dumps({name: round(value, 4) for name, value in payload["medians"].items()})
        + " steps="
        + json.dumps({name: [row["steps"] for row in arm["results"]] for name, arm in arms.items()})
        + " cost="
        + str(payload["cost"]["cpu"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
