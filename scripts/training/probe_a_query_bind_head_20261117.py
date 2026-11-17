"""Query-bound readout heads on the 22.4.3 train/fit partitions.

The earlier first-token probe fed a frozen, untrained 8-dimensional ``goal()``
pool as the query. Six entity names share all but their first character, so that
random projection erased "which entity" and the head only reached the class
prior. This probe instead lets the head encode the raw request bytes and bind
them to patch tokens by attention.

Two token sources are compared under the same head:

* ``raw``: the 16 RGB patches (48 values) with their 2-d patch coordinates.
* ``u2``: the frozen U2 encoder tokens (rgb, audio, proprio, outcome).

Declared verdict, computed on the median over three head seeds:

* ``head_feasible``: train >= 0.95 and fit >= 0.80.
* ``query_pathway_fixed_generalization_gap``: train >= 0.95 and fit < 0.80.
* ``perception_not_separable``: train < 0.95.

No U2 optimizer step, no check read and no final read occur here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.first_a_support_22_5 import generate_examples, make_model  # noqa: E402

REPORT = ROOT / "reports/taiji_u2_22_5_a_query_bind_head_v3_20261117.json"
LABELS = ("左边", "中间", "右边", "未知")
MAX_BYTES = 64
SEED = 20261110
HEAD_SEEDS = (20261110, 20261111, 20261112)
COUNTS = {"train": 768, "fit": 192}
CAPACITY_REPORT = ROOT / "reports/taiji_u2_22_5_a_query_bind_capacity_20261117.json"
CAPACITY_STEPS = 2000


class QueryBindHead(nn.Module):
    """Pool the request bytes, then attend once over the token set."""

    def __init__(self, token_dim: int, width: int = 32, text: int = 16) -> None:
        super().__init__()
        self.token = nn.Linear(token_dim, width)
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.null_token = nn.Parameter(torch.zeros(width))
        self.output = nn.Linear(2 * width, 4)

    def forward(self, tokens, request, mask):
        token = torch.tanh(self.token(tokens))
        null = self.null_token.expand(token.shape[0], 1, -1)
        token = torch.cat([token, null], 1)
        text = self.byte(request) + self.position[:, : request.shape[1]]
        text = text.masked_fill(~mask[..., None], 0.0)
        mean = text.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        peak = text.masked_fill(~mask[..., None], -1e4).amax(1)
        query = self.query(torch.tanh(self.text(torch.cat([mean, peak], -1))))[:, None, :]
        scores = torch.einsum("bqd,bkd->bqk", query, self.key(token)) / math.sqrt(token.shape[-1])
        bound = torch.einsum("bqk,bkd->bqd", scores.softmax(-1), self.value(token))[:, 0]
        return self.output(torch.cat([bound, query[:, 0]], -1))


class BindBlock(nn.Module):
    """One residual attention block with a single learned query."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(self, tokens, query):
        prompt = self.query(query)[:, None, :]
        scores = torch.einsum("bqd,bkd->bqk", prompt, self.key(tokens)) / math.sqrt(
            tokens.shape[-1]
        )
        bound = torch.einsum("bqk,bkd->bqd", scores.softmax(-1), self.value(tokens))[:, 0]
        query = self.norm_attention(query + bound)
        return self.norm_feedforward(query + self.feedforward(query))


class CapacityHead(nn.Module):
    """Wider two-block variant used only for the capacity upper bound."""

    def __init__(self, token_dim: int, width: int = 64, text: int = 24) -> None:
        super().__init__()
        self.token = nn.Linear(token_dim, width)
        self.byte = nn.Embedding(256, text)
        self.position = nn.Parameter(torch.zeros(1, MAX_BYTES, text))
        self.text = nn.Linear(2 * text, width)
        self.null_token = nn.Parameter(torch.zeros(width))
        self.blocks = nn.ModuleList([BindBlock(width), BindBlock(width)])
        self.output = nn.Linear(width, 4)

    def forward(self, tokens, request, mask):
        token = torch.tanh(self.token(tokens))
        null = self.null_token.expand(token.shape[0], 1, -1)
        token = torch.cat([token, null], 1)
        text = self.byte(request) + self.position[:, : request.shape[1]]
        text = text.masked_fill(~mask[..., None], 0.0)
        mean = text.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        peak = text.masked_fill(~mask[..., None], -1e4).amax(1)
        query = torch.tanh(self.text(torch.cat([mean, peak], -1)))
        for block in self.blocks:
            query = block(token, query)
        return self.output(query)


def _examples(split: str) -> list[dict]:
    """Use the formal A-gate fixture so the head is validated on the real distribution.

    The first pass drew 22.4.3 partition seeds directly and skipped every world
    whose requested label was unreachable, which biased the label mix and left
    452 of 768 train worlds. ``generate_examples`` retries attempts per index,
    matching what the A gate actually trains on.
    """
    examples = []
    for example in generate_examples(split, COUNTS[split]):
        examples.append(
            {
                "request": example["public"]["request"],
                "observations": example["public"]["observations"],
                "coordinates": example["public"]["coordinates"],
                "outcome": example["public"]["outcome"],
                "label": LABELS.index(example["target"].decode()),
            }
        )
    return examples


def _pad(requests: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(int(row.numel()) for row in requests)
    if length > MAX_BYTES:
        raise ValueError("request_exceeds_text_budget")
    request = torch.zeros(len(requests), length, dtype=torch.long)
    mask = torch.zeros(len(requests), length, dtype=torch.bool)
    for index, row in enumerate(requests):
        request[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return request, mask


def _extract(model, examples: list[dict], raw_only: bool = False) -> dict:
    raw_tokens, u2_tokens, requests, labels = [], [], [], []
    with torch.no_grad():
        for example in examples:
            rgb = example["observations"]["rgb"]
            coords = example["coordinates"]["rgb"][:, :2]
            raw_tokens.append(torch.cat([rgb, coords], -1))
            if not raw_only:
                encoded = model.encode(
                    example["observations"], example["coordinates"], example["outcome"]
                )
                u2_tokens.append(
                    torch.cat(
                        [encoded["rgb"], encoded["audio"], encoded["proprio"], encoded["outcome"]],
                        0,
                    )
                )
            requests.append(torch.tensor(list(example["request"]), dtype=torch.long))
            labels.append(example["label"])
    request, mask = _pad(requests)
    rows = {
        "raw": torch.stack(raw_tokens),
        "request": request,
        "mask": mask,
        "label": torch.tensor(labels),
    }
    if u2_tokens:
        rows["u2"] = torch.stack(u2_tokens)
    return rows


def _fit(
    tokens: torch.Tensor, request: torch.Tensor, mask: torch.Tensor, label: torch.Tensor, seed: int
):
    torch.manual_seed(seed)
    head = QueryBindHead(tokens.shape[2])
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(600):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(head(tokens, request, mask), label)
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        logits = head(tokens, request, mask)
    return head, (logits.argmax(-1) == label).float().mean().item()


def _confusion(prediction: torch.Tensor, label: torch.Tensor) -> list[list[int]]:
    table = [[0] * 4 for _ in range(4)]
    for row, column in zip(label.tolist(), prediction.tolist(), strict=True):
        table[row][column] += 1
    return table


def _fit_capacity(data: dict, seed: int):
    torch.manual_seed(seed)
    head = CapacityHead(data["raw"].shape[2])
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(CAPACITY_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.cross_entropy(
            head(data["raw"], data["request"], data["mask"]), data["label"]
        )
        loss.backward()
        optimizer.step()
    head.eval()
    with torch.no_grad():
        logits = head(data["raw"], data["request"], data["mask"])
    return head, (logits.argmax(-1) == data["label"]).float().mean().item(), logits


def _capacity(data: dict, fit: dict) -> dict:
    rows = []
    for seed in HEAD_SEEDS:
        head, train_accuracy, train_logits = _fit_capacity(data, seed)
        with torch.no_grad():
            logits = head(fit["raw"], fit["request"], fit["mask"])
        rows.append(
            {
                "seed": seed,
                "train_accuracy": train_accuracy,
                "fit_accuracy": (logits.argmax(-1) == fit["label"]).float().mean().item(),
                "train_confusion": _confusion(train_logits.argmax(-1), data["label"]),
                "fit_confusion": _confusion(logits.argmax(-1), fit["label"]),
            }
        )
    median_train = sorted(row["train_accuracy"] for row in rows)[1]
    median_fit = sorted(row["fit_accuracy"] for row in rows)[1]
    if median_train < 0.95:
        verdict = "capacity_still_short"
    elif median_fit < 0.80:
        verdict = "capacity_feasible_generalization_gap"
    else:
        verdict = "capacity_feasible"
    return {
        "results": rows,
        "median_train_accuracy": median_train,
        "median_fit_accuracy": median_fit,
        "verdict": verdict,
    }


def main() -> int:
    model = make_model("native", SEED)
    train_examples = _examples("train")
    fit_examples = _examples("fit")
    if "--capacity" in sys.argv:
        train = _extract(model, train_examples, raw_only=True)
        fit = _extract(model, fit_examples, raw_only=True)
        capacity = _capacity(train, fit)
        if CAPACITY_REPORT.exists():
            raise FileExistsError(CAPACITY_REPORT)
        payload = {
            "schema": "taiji-u2-22-5-a-query-bind-capacity-v1",
            "card": "22.5.0.A",
            "source": "raw",
            "width": 64,
            "blocks": 2,
            "steps": CAPACITY_STEPS,
            "capacity": capacity,
            "declared_verdict": {
                "capacity_feasible": "median train >= 0.95 and median fit >= 0.80",
                "capacity_feasible_generalization_gap": "median train >= 0.95, fit < 0.80",
                "capacity_still_short": "median train < 0.95",
            },
            "discipline": {
                "u2_optimizer_steps": 0,
                "head_optimizer_steps": CAPACITY_STEPS * len(HEAD_SEEDS),
                "check_reads": 0,
                "final_reads": 0,
            },
            "examples": {"train": len(train_examples), "fit": len(fit_examples)},
        }
        CAPACITY_REPORT.write_bytes(
            json.dumps(payload, ensure_ascii=False, indent=2).encode() + b"\n"
        )
        print(
            "capacity verdict="
            + capacity["verdict"]
            + " train="
            + str(round(capacity["median_train_accuracy"], 4))
            + " fit="
            + str(round(capacity["median_fit_accuracy"], 4))
        )
        return 0
    train = _extract(model, train_examples)
    fit = _extract(model, fit_examples)
    summary = {}
    for source in ("raw", "u2"):
        rows = []
        for seed in HEAD_SEEDS:
            head, train_accuracy = _fit(
                train[source], train["request"], train["mask"], train["label"], seed
            )
            with torch.no_grad():
                train_logits = head(train[source], train["request"], train["mask"])
                logits = head(fit[source], fit["request"], fit["mask"])
            fit_accuracy = (logits.argmax(-1) == fit["label"]).float().mean().item()
            rows.append(
                {
                    "seed": seed,
                    "train_accuracy": train_accuracy,
                    "fit_accuracy": fit_accuracy,
                    "train_confusion": _confusion(train_logits.argmax(-1), train["label"]),
                    "fit_confusion": _confusion(logits.argmax(-1), fit["label"]),
                }
            )
        summary[source] = {
            "results": rows,
            "max_train_accuracy": max(row["train_accuracy"] for row in rows),
            "median_train_accuracy": sorted(row["train_accuracy"] for row in rows)[1],
            "max_fit_accuracy": max(row["fit_accuracy"] for row in rows),
            "median_fit_accuracy": sorted(row["fit_accuracy"] for row in rows)[1],
        }

    primary = summary["raw"]
    if primary["median_train_accuracy"] < 0.95:
        verdict = "perception_not_separable"
    elif primary["median_fit_accuracy"] < 0.80:
        verdict = "query_pathway_fixed_generalization_gap"
    else:
        verdict = "head_feasible"
    report = {
        "schema": "taiji-u2-22-5-a-query-bind-head-probe-v1",
        "card": "22.5.0.A",
        "verdict": verdict,
        "primary_source": "raw",
        "summary": summary,
        "examples": {
            "train": len(train_examples),
            "fit": len(fit_examples),
            "request_width": int(train["request"].shape[1]),
            "fixture": "generate_examples (formal A-gate distribution)",
        },
        "declared_verdict": {
            "head_feasible": "raw median train >= 0.95 and median fit >= 0.80",
            "query_pathway_fixed_generalization_gap": "raw median train >= 0.95, fit < 0.80",
            "perception_not_separable": "raw median train < 0.95",
        },
        "discipline": {
            "u2_optimizer_steps": 0,
            "head_optimizer_steps": 600 * len(HEAD_SEEDS) * len(summary),
            "check_reads": 0,
            "final_reads": 0,
        },
    }
    if REPORT.exists():
        raise FileExistsError(REPORT)
    REPORT.write_bytes(json.dumps(report, ensure_ascii=False, indent=2).encode() + b"\n")
    print(
        "query-bind verdict="
        + verdict
        + " raw_train="
        + str(round(primary["median_train_accuracy"], 4))
        + " raw_fit="
        + str(round(primary["median_fit_accuracy"], 4))
        + " u2_train="
        + str(round(summary["u2"]["median_train_accuracy"], 4))
        + " u2_fit="
        + str(round(summary["u2"]["median_fit_accuracy"], 4))
        + " train_n="
        + str(len(train_examples))
        + " fit_n="
        + str(len(fit_examples))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
