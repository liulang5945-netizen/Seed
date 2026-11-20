"""Upgraded-readout re-tests: Chinese lexical sequence and multimodal action fusion (0 check).

Task A (lexical): the instruction characters are fed as one tick each and a positional
attention readout predicts the 16-way action; the held-out wording template is the fit set.
Task B (fusion): a scene tick and an instruction tick are fed in sequence; the readout must
combine them to predict the 12-way body-slot x direction action.
Both use the frozen 23.0b parent, disjoint fit seeds, and the positional-attention readout.
Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import accuracy as skill_accuracy  # noqa: E402
from scripts.training.run_s2_23_0b_joint_20261120 import (
    build_data as build_skill_data,  # noqa: E402
)
from scripts.training.run_s2_23_0b_joint_20261120 import channels, lr_at  # noqa: E402
from scripts.training.run_s2_23_2c_attention_20261120 import train_joint_parent  # noqa: E402

OUT = ROOT / "reports/taiji_u2_upgrade_retest_20261120.json"
CAP_CPU = 1500.0
SEEDS = (20261120, 20261121, 20261122)
FIT_OFFSET = 10_000_000
STEPS = 1500
BATCH = 64
LR = 3e-3
COLOURS = ("红", "蓝", "绿", "黄")
DIRECTIONS = ("左", "右", "前", "后")
TEMPLATES = ("把{c}球放到{d}边", "{c}球放{d}边", "让{c}球去{d}边", "把{c}球放去{d}边")
VOCAB = "把球放到边让去红蓝绿黄左右前后"
CHAR_INDEX = {char: index for index, char in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PROJECTION = torch.randn(VOCAB_SIZE, 8, generator=torch.Generator().manual_seed(31))
LEX_TRAIN = 512
LEX_FIT = 128
FUSION_TRAIN = 512
FUSION_FIT = 128
JOINT_MIN = 0.60
MARGIN = 0.10


def char_sequence(text: str) -> torch.Tensor:
    ids = [CHAR_INDEX[char] for char in text if char in CHAR_INDEX]
    table = torch.zeros(16, VOCAB_SIZE)
    for position, index in enumerate(ids[:16]):
        table[position, index] = 1.0
    embedded = table @ PROJECTION
    padded = torch.zeros(16, 48)
    padded[:, :8] = embedded
    return padded


def lexical_episode(seed: int, template_index: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colour = COLOURS[int(torch.randint(0, 4, (1,), generator=generator))]
    direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    text = TEMPLATES[template_index].format(c=colour, d=direction)
    sequence = char_sequence(text)
    return {"sequence": sequence, "label": COLOURS.index(colour) * 4 + DIRECTIONS.index(direction)}


def fusion_episode(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    colours = torch.randperm(4, generator=generator)[:3].tolist()
    positions = torch.randint(0, 4, (3,), generator=generator).tolist()
    target_slot = int(torch.randint(0, 3, (1,), generator=generator))
    direction = DIRECTIONS[int(torch.randint(0, 4, (1,), generator=generator))]
    scene = torch.zeros(3, 8)
    for slot in range(3):
        scene[slot, colours[slot]] = 1.0
        scene[slot, 4 + positions[slot]] = 1.0
    text = TEMPLATES[0].format(c=COLOURS[colours[target_slot]], d=direction)
    instruction = char_sequence(text)
    scene_tick = torch.zeros(48)
    scene_tick[:24] = scene.reshape(-1)
    instruction_tick = torch.zeros(48)
    instruction_tick[:8] = instruction[0][:8]
    sequence = torch.stack([scene_tick, instruction_tick])
    return {"sequence": sequence, "label": target_slot * 4 + DIRECTIONS.index(direction)}


class PosAttentionRead(nn.Module):
    def __init__(self, token_dim: int, classes: int, width: int = 64, positions: int = 16) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(positions, token_dim))
        self.token = nn.Sequential(nn.Linear(token_dim, width), nn.Tanh())
        self.query1 = nn.Sequential(nn.Linear(width, width), nn.Tanh())
        self.query2 = nn.Sequential(nn.Linear(width + 8, width), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(2 * width, width), nn.Tanh(), nn.Linear(width, classes))
        self.query_token = nn.Parameter(torch.zeros(8))

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embed = self.token(sequence + self.pos[: sequence.shape[1]])
        key = embed.mean(dim=1)
        q1 = self.query1(key)
        scores = torch.softmax(torch.einsum("btd,bd->bt", embed, q1), dim=-1)
        context1 = torch.einsum("bt,btd->bd", scores, embed)
        q2 = self.query2(
            torch.cat([context1, self.query_token.expand(context1.shape[0], -1)], dim=-1)
        )
        scores2 = torch.softmax(torch.einsum("btd,bd->bt", embed, q2), dim=-1)
        context2 = torch.einsum("bt,btd->bd", scores2, embed)
        return self.out(torch.cat([context2, q1], dim=-1))


def unit_sequence(graph, sequence: torch.Tensor) -> torch.Tensor:
    state = graph.initial_state()
    activities = []
    for tick_index in range(sequence.shape[1]):
        gates = torch.ones(sequence.shape[0], graph.units.n_units)
        with torch.no_grad():
            result = graph(channels(sequence[:, tick_index]), gates, state)
        activities.append(result["unit_result"]["activity"])
        state = result["next_state"]
    return torch.stack(activities, dim=1)


def train_head(head: nn.Module, batch: dict[str, torch.Tensor], seed: int) -> None:
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for step in range(STEPS):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, STEPS)
        index = torch.randint(0, batch["sequence"].shape[0], (BATCH,), generator=generator)
        loss = loss_fn(head(batch["sequence"][index]), batch["labels"][index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def accuracy(head: nn.Module, batch: dict[str, torch.Tensor]) -> float:
    with torch.no_grad():
        logits = head(batch["sequence"])
    return float((logits.argmax(-1) == batch["labels"]).float().mean())


def build_lexical(seed: int, templates: tuple[int, ...], count: int) -> dict[str, torch.Tensor]:
    rows = [
        lexical_episode(seed + index, templates[index % len(templates)]) for index in range(count)
    ]
    return {
        "sequence": torch.stack([row["sequence"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


def build_fusion(seed: int, count: int) -> dict[str, torch.Tensor]:
    rows = [fusion_episode(seed + index) for index in range(count)]
    return {
        "sequence": torch.stack([row["sequence"] for row in rows]).float(),
        "labels": torch.tensor([row["label"] for row in rows]),
    }


def run_seed(seed: int) -> dict:
    graph, _ = train_joint_parent(seed)
    skill_data = build_skill_data(seed)
    baseline = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    for parameter in graph.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    results = {}
    # Task A: lexical grounding with held-out wording template.
    train = build_lexical(seed, (0, 1, 2), LEX_TRAIN)
    fit = build_lexical(seed + FIT_OFFSET, (3,), LEX_FIT)
    train_units = unit_sequence(graph, train["sequence"])
    fit_units = unit_sequence(graph, fit["sequence"])
    head = PosAttentionRead(train_units.shape[-1], 16)
    train_head(head, {"sequence": train_units, "labels": train["labels"]}, seed)
    results["lexical"] = accuracy(head, {"sequence": fit_units, "labels": fit["labels"]})
    raw_head = PosAttentionRead(train["sequence"].shape[-1], 16)
    train_head(raw_head, {"sequence": train["sequence"], "labels": train["labels"]}, seed)
    results["lexical_replay"] = accuracy(
        raw_head, {"sequence": fit["sequence"], "labels": fit["labels"]}
    )
    # Task B: multimodal fusion.
    train = build_fusion(seed, FUSION_TRAIN)
    fit = build_fusion(seed + FIT_OFFSET, FUSION_FIT)
    train_units = unit_sequence(graph, train["sequence"])
    fit_units = unit_sequence(graph, fit["sequence"])
    for arm, train_batch, fit_batch in (
        (
            "fusion",
            {"sequence": train_units, "labels": train["labels"]},
            {"sequence": fit_units, "labels": fit["labels"]},
        ),
        (
            "fusion_scene_only",
            {"sequence": train_units[:, :1], "labels": train["labels"]},
            {"sequence": fit_units[:, :1], "labels": fit["labels"]},
        ),
        (
            "fusion_instruction_only",
            {"sequence": train_units[:, 1:], "labels": train["labels"]},
            {"sequence": fit_units[:, 1:], "labels": fit["labels"]},
        ),
    ):
        head = PosAttentionRead(train_batch["sequence"].shape[-1], 12)
        train_head(head, train_batch, seed)
        results[arm] = accuracy(head, fit_batch)
    after = {
        "s1": skill_accuracy(graph, skill_data["x1f"], skill_data["y1f"], 0),
        "s2": skill_accuracy(graph, skill_data["x2f"], skill_data["y2f"], 1),
        "s3": skill_accuracy(graph, skill_data["x3f"], skill_data["y3f"], 2),
    }
    damage = {skill: max(0.0, baseline[skill] - after[skill]) for skill in baseline}
    return {"seed": seed, "arms": results, "retention_damage": damage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    rows = [run_seed(seed) for seed in SEEDS]
    means = {arm: sum(row["arms"][arm] for row in rows) / len(rows) for arm in rows[0]["arms"]}
    best_single = max(means["fusion_scene_only"], means["fusion_instruction_only"])
    gates = {
        "lexical_learnable": means["lexical_replay"] >= 0.80,
        "lexical_level": means["lexical"] >= 0.60,
        "fusion_gain": means["fusion"] >= best_single + MARGIN,
        "fusion_level": means["fusion"] >= JOINT_MIN,
        "retention": all(max(row["retention_damage"].values()) <= 0.02 for row in rows),
    }
    verdict = "upgrade_retest_verified" if all(gates.values()) else "upgrade_retest_partial"
    payload = {
        "schema": "taiji-u2-upgrade-retest-v1",
        "card": "upgraded positional-attention readout retests: Chinese lexical and multimodal fusion",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "fit_offset": FIT_OFFSET,
            "steps": STEPS,
            "lex_train": LEX_TRAIN,
            "lex_fit": LEX_FIT,
            "fusion_train": FUSION_TRAIN,
            "fusion_fit": FUSION_FIT,
        },
        "rows": rows,
        "means": means,
        "gates": gates,
        "verdict": verdict,
        "disclosure": "character sequence ticks and positional-attention readout; fusion uses scene tick then instruction tick (single character projection)",
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {"verdict": verdict, "gates": gates, "means": means, "cpu": payload["cost"]["cpu"]},
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "upgrade_retest_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
