"""Public-image/raw-byte CNN-GRU reference for first-round learnability calibration."""

from __future__ import annotations

import torch
from torch import nn

from scripts.training.paired_world_22_5 import public_example
from taiji.u2_operators import utf8_mask
from taiji.utf8_state import advance_utf8


class FirstRoundBaseline(nn.Module):
    grid: torch.Tensor

    def __init__(self, seed=2026111309):
        super().__init__()
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(257, 16)
            self.convolution = nn.Sequential(
                nn.Conv2d(5, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 16, 3, padding=1), nn.GELU()
            )
            self.vision = nn.Sequential(nn.Linear(16, 64), nn.LayerNorm(64))
            self.question = nn.Sequential(nn.Linear(16, 64), nn.LayerNorm(64))
            self.mix = nn.Linear(192, 64)
            self.gru = nn.GRU(80, 64, batch_first=True)
            self.output = nn.Linear(64, 257)
        axis = torch.linspace(-1, 1, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack([x, y]))

    def context(self, images, image_index, question_ids, question_valid):
        # The index associates each question with its public image, never a learned ID.
        grid = self.grid[None].expand(images.shape[0], -1, -1, -1)
        visual = self.vision(self.convolution(torch.cat([images, grid], 1)).mean((-1, -2)))
        visual = visual[image_index]
        embedded = self.embedding(question_ids)
        question = (embedded * question_valid[..., None]).sum(1) / question_valid.sum(1).clamp_min(
            1
        )[:, None]
        question = self.question(question)
        return self.mix(torch.cat([visual, question, visual * question], -1)).tanh()

    def teacher_logits(self, batch):
        context = self.context(
            batch["images"], batch["image_index"], batch["question_ids"], batch["question_valid"]
        )
        previous = self.embedding(batch["previous"])
        recurrent_input = torch.cat(
            [previous, context[:, None].expand(-1, previous.shape[1], -1)], -1
        )
        hidden, _ = self.gru(recurrent_input, context[None])
        return self.output(hidden)

    def step(self, previous, context, hidden):
        inputs = torch.cat([self.embedding(previous), context], -1)[:, None]
        output, hidden = self.gru(inputs, hidden)
        return self.output(output[:, 0]), hidden


def make_batch(partition):
    bank, indices, questions, targets, identities = {}, [], [], [], []
    images: list[torch.Tensor] = []
    for episode in partition["episodes"]:
        example = public_example(partition, episode)
        raw = example["public"]["observations"]["rgb"]
        if raw.shape != (16, 48) or not bool(torch.isfinite(raw).all()):
            raise ValueError("calibration_image_shape_or_values")
        if episode["world_id"] not in bank:
            bank[episode["world_id"]] = len(images)
            image = (
                raw.reshape(4, 4, 4, 4, 3)
                .permute(0, 2, 1, 3, 4)
                .reshape(16, 16, 3)
                .permute(2, 0, 1)
            )
            images.append(image.to(dtype=torch.float32))
        indices.append(bank[episode["world_id"]])
        questions.append(episode["requests"][0])
        target = episode["targets"][0]
        if len(target) != 6 or not 1 <= len(questions[-1]) <= 512:
            raise ValueError("calibration_text_outside_frozen_profile")
        targets.append([*target, 256])
        identities.append(episode["id"])
    if not images:
        raise ValueError("empty_calibration_partition")
    length = max(map(len, questions))
    ids = torch.full((len(questions), length), 256, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    masks = []
    cache = {}
    for row, (question, symbols) in enumerate(zip(questions, targets, strict=True)):
        ids[row, : len(question)] = torch.tensor(list(question))
        valid[row, : len(question)] = True
        remaining, lead, used = 0, 0, 0
        allowed = []
        for symbol in symbols:
            key = remaining, lead, 64 - used
            if key not in cache:
                cache[key] = utf8_mask(*key)
            mask = cache[key]
            if not bool(mask[symbol]):
                raise ValueError("invalid_calibration_target")
            allowed.append(mask)
            if symbol != 256:
                remaining, lead = advance_utf8(remaining, lead, symbol)
                used += 1
        masks.append(torch.stack(allowed))
    target_tensor = torch.tensor(targets, dtype=torch.long)
    previous = torch.cat([target_tensor.new_full((len(targets), 1), 256), target_tensor[:, :-1]], 1)
    return {
        "images": torch.stack(images),
        "image_index": torch.tensor(indices),
        "question_ids": ids,
        "question_valid": valid,
        "targets": target_tensor,
        "previous": previous,
        "teacher_masks": torch.stack(masks),
        "question_hex": [item.hex() for item in questions],
        "episode_ids": identities,
    }


def loss_and_logits(model, batch):
    logits = model.teacher_logits(batch)
    logp = logits.masked_fill(~batch["teacher_masks"], -torch.inf).log_softmax(-1)
    loss = -logp.gather(-1, batch["targets"][..., None]).squeeze(-1).mean()
    return loss, logp


def generate(model, batch, guard=lambda: None):
    count = len(batch["question_hex"])
    content = [bytearray() for _ in range(count)]
    ended = [False] * count
    remaining, lead = [0] * count, [0] * count
    cache = {}
    with torch.no_grad():
        context = model.context(
            batch["images"], batch["image_index"], batch["question_ids"], batch["question_valid"]
        )
        hidden = context[None]
        active = list(range(count))
        previous = torch.full((count,), 256, dtype=torch.long)
        for _ in range(65):
            guard()
            logits, hidden = model.step(previous, context, hidden)
            masks = []
            for index in active:
                key = remaining[index], lead[index], 64 - len(content[index])
                if key not in cache:
                    cache[key] = utf8_mask(*key)
                masks.append(cache[key])
            symbols = logits.masked_fill(~torch.stack(masks), -torch.inf).argmax(-1).tolist()
            keep = []
            for row, (index, symbol) in enumerate(zip(active, symbols, strict=True)):
                if symbol == 256:
                    ended[index] = True
                else:
                    content[index].append(symbol)
                    remaining[index], lead[index] = advance_utf8(
                        remaining[index], lead[index], symbol
                    )
                    keep.append(row)
            if not keep:
                break
            context = context[keep]
            hidden = hidden[:, keep]
            previous = torch.tensor([symbols[row] for row in keep], dtype=torch.long)
            active = [active[row] for row in keep]
    rows = []
    for index, raw in enumerate(content):
        try:
            bytes(raw).decode("utf-8", errors="strict")
            valid = True
        except UnicodeDecodeError:
            valid = False
        expected = bytes(batch["targets"][index, :-1].tolist())
        rows.append(
            {
                "episode_id": batch["episode_ids"][index],
                "question_hex": batch["question_hex"][index],
                "target_hex": expected.hex(),
                "output_hex": raw.hex(),
                "eos": ended[index],
                "valid_utf8": valid,
                "correct": ended[index] and valid and raw == expected,
            }
        )
    groups = {
        question: [row for row in rows if row["question_hex"] == question]
        for question in sorted(set(batch["question_hex"]))
    }
    return {
        "correct": sum(row["correct"] for row in rows),
        "denominator": count,
        "accuracy": sum(row["correct"] for row in rows) / count,
        "by_question": {
            key: {
                "correct": sum(row["correct"] for row in values),
                "denominator": len(values),
                "accuracy": sum(row["correct"] for row in values) / len(values),
            }
            for key, values in groups.items()
        },
        "rows": rows,
    }
