"""P4.1: give the body's rgb encoder the column colour objective.

P3.30 and P3.31 both consumed the body's own [16, 8] column tokens and both
stalled: the 7-way colour probe reached only 0.674 on fit while the same 48 raw
values reach 1.000, and the four-way task stopped at 0.31-0.34. Eight dimensions
can separate six colours, so the limit is not width. The encoder has only ever
been trained by byte NLL, which never asked it to name a colour.

This card adds that objective and nothing else. Two arms, identical in every
respect except the extra term:

  A  encoder + readout, four-way cross-entropy only
  B  encoder + readout, four-way cross-entropy plus 7-way cross-entropy on every
     column token against the palette colour of that column

The 7-way head is a declared training scaffold (63 parameters). It never appears
at evaluation, and the served readout is the same 58-parameter module P3.30 used.
Prototypes are recomputed from the current encoder each forward, so at step zero
both arms are byte-identical to P3.30's build and the readout keeps meaning "the
token this encoder gives a column of that colour".

Verdicts, frozen before any reading:
  probe >= 0.95 and fit >= 0.80 -> column_supervision_makes_parent_reachable
  probe >= 0.95 and fit <= 0.55 -> colour_in_tokens_readout_still_blocks
  probe <  0.95                 -> encoder_cannot_carry_colour
  otherwise                     -> inconclusive_column_supervision

Modes: ``probe`` (zero training), ``price``, ``run``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.nn import functional

from scripts.training.check_report_run_conditions import write_report
from scripts.training.first_a_support_22_5 import RGB_COLORS, make_model
from taiji.u2_history import atomic_load, value_digest
from taiji.u2_observation_columns import column_tokens
from taiji.u2_task_episode import NAMES, locate_public

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_1_column_supervision_20261114.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p4_1_price_20261114.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p4_1_column_supervision_20261114.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
COLOURS = 6
COLOUR_CLASSES = COLOURS + 1  # six palette colours plus background
SUFFIX = "现在在哪里？"
MATCH = 0.05


class BudgetStop(RuntimeError):
    pass


def palette_tensor() -> torch.Tensor:
    return torch.tensor([[value / 255 for value in row] for row in RGB_COLORS])


def column_positions() -> torch.Tensor:
    return 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1


def image_of(example: dict) -> torch.Tensor:
    patches = example["public"]["observations"]["rgb"]
    return patches.reshape(4, 4, 4, 4, 3).permute(0, 2, 1, 3, 4).reshape(SIDE, SIDE, 3)


_PALETTE = palette_tensor()


def column_labels(example: dict) -> torch.Tensor:
    """Dominant palette colour per column, or six when no column is painted."""
    image = image_of(example)
    distance = ((image[:, :, None, :] - _PALETTE[None, None, :, :]) ** 2).sum(-1)
    nearest = distance.argmin(-1)
    matched = distance.min(-1).values < MATCH
    labels = torch.full((SIDE,), COLOURS, dtype=torch.long)
    for column in range(SIDE):
        counts = [
            int(((nearest[:, column] == colour) & matched[:, column]).sum())
            for colour in range(COLOURS)
        ]
        best = int(max(range(COLOURS), key=lambda colour: counts[colour]))
        if counts[best] >= 2:
            labels[column] = best
    return labels


def prototype_tokens(subject) -> torch.Tensor:
    """What the current encoder produces for a column filled with each colour."""
    values = _PALETTE[:, None, :].repeat(1, SIDE, 1).reshape(COLOURS, SIDE * 3)
    frame = torch.zeros(COLOURS, 8)
    frame[:, 5] = 1
    frame[:, 7] = 1
    return torch.tanh(
        subject.encoder["rgb"](values) + subject.metadata(frame) + subject.modality[0]
    )


class TokenDetector(nn.Module):
    """P3.30's readout, with the prototypes tied to the live encoder."""

    def __init__(self, subject, tau: float) -> None:
        super().__init__()
        self.subject = [subject]  # not a module; the encoder owns its own parameters
        self.classifier = nn.Linear(QUERY, COLOURS)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_mix = nn.Parameter(torch.zeros(()))
        self.presence = nn.Parameter(torch.tensor(2.0))
        self.abstain = nn.Parameter(torch.tensor(-1.0))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor):
        prototypes = prototype_tokens(self.subject[0])
        squared = ((tokens[None, :, :] - prototypes[:, None, :]) ** 2).sum(-1)
        temperature = self.log_tau.exp().clamp(1e-3, 1.0)
        weights = (-squared / temperature).softmax(-1)
        centroids = (weights * column_positions()).sum(-1)
        nearest = squared.min(-1).values
        logits = self.classifier(query)
        colour = logits.softmax(-1)
        mix = (logits / self.log_mix.exp().clamp(0.05, 20.0)).softmax(-1)
        centre = (mix * centroids).sum()
        null = self.abstain + self.presence * (mix * nearest).sum()
        scale = self.log_scale.exp().clamp(1e-2, 1e2)
        shape = torch.stack([-scale * (centre - c).abs() for c in CENTRES] + [scale * null])
        return shape, colour, mix, logits, centroids, null


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("column_pathway", ROOT / "taiji/u2_observation_columns.py"),
        ("subject", ROOT / "taiji/u2_subject.py"),
    ):
        if pins.get(name) != sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if mode == "run":
        if contract.get("status") != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("missing_training_authority")
        if not contract.get("cpu_cap") or not contract.get("wall_cap"):
            raise ValueError("missing_measured_cap")


def build(contract: dict, arm: str):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    module = TokenDetector(model, contract["candidate"]["tau"])
    scaffold = nn.Linear(8, COLOUR_CLASSES) if arm == "B" else None
    spec = contract["optimizer"]
    parameters = list(module.parameters()) + list(model.encoder["rgb"].parameters())
    if scaffold is not None:
        parameters += list(scaffold.parameters())
    optimizer = torch.optim.Adam(
        parameters,
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, scaffold, optimizer


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


def tokens_of(model, example: dict) -> torch.Tensor:
    return column_tokens(model, example["public"]["observations"], example["public"]["coordinates"])


def colour_index(example: dict) -> int:
    return NAMES.index(example["public"]["request"].decode()[: -len(SUFFIX)])


_TRUTH: dict[str, int | None] = {}


def oracle_column(example: dict) -> int | None:
    key = example["id"]
    if key in _TRUTH:
        return _TRUTH[key]
    image = image_of(example) * 255.0
    position = locate_public(image, example["public"]["request"].decode()[: -len(SUFFIX)])
    column = None if position is None else min(15, max(0, int((position + 1) / 2 * 16)))
    _TRUTH[key] = column
    return column


def objective(model, module, scaffold, batch: list[dict], weight: float):
    shapes, colours = [], []
    for example in batch:
        shape, _, _, logits, _, _ = module(tokens_of(model, example), query_of(model, example))
        shapes.append(shape)
        colours.append(logits)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(shapes), labels)
    if weight:
        targets = torch.tensor([colour_index(example) for example in batch], dtype=torch.long)
        loss = loss + weight * functional.cross_entropy(torch.stack(colours), targets)
    if scaffold is not None:
        column_losses = []
        column_targets = []
        for example in batch:
            logits = scaffold(tokens_of(model, example))
            column_losses.append(logits)
            column_targets.append(column_labels(example))
        loss = loss + weight * functional.cross_entropy(
            torch.cat(column_losses), torch.cat(column_targets)
        )
    return loss


def probe(model, examples: list[dict], lam: float) -> dict:
    """Ridge from column tokens to the 7-way column label.

    The overall figure is dominated by background columns, which are 74.6% of the
    fit split, so a constant "background" predictor scores 0.746. The number that
    answers "can the token name a colour" is the painted-column accuracy, and its
    floor is the painted majority class (0.208), not the 1/6 chance level. Both
    baselines are reported beside the measurements so neither can be mistaken for
    chance.
    """
    with torch.no_grad():
        train = torch.cat([tokens_of(model, example) for example in examples["train"]])
        fit = torch.cat([tokens_of(model, example) for example in examples["fit"]])
    train_y = torch.cat([column_labels(example) for example in examples["train"]])
    fit_y = torch.cat([column_labels(example) for example in examples["fit"]])
    mu, sd = train.mean(0), train.std(0).clamp_min(1e-9)
    a = (train - mu) / sd
    b = (fit - mu) / sd
    onehot = torch.zeros(len(train_y), COLOUR_CLASSES)
    onehot[torch.arange(len(train_y)), train_y] = 1
    weight = torch.linalg.solve(a.T @ a + lam * torch.eye(a.shape[1]), a.T @ onehot)
    predicted = {"train": (a @ weight).argmax(1), "fit": (b @ weight).argmax(1)}
    truth = {"train": train_y, "fit": fit_y}
    metrics: dict = {}
    for split in ("train", "fit"):
        pred, target = predicted[split], truth[split]
        painted, background = target < COLOURS, target >= COLOURS
        counts = torch.bincount(target[painted], minlength=COLOURS) if painted.any() else None
        metrics[split] = {
            "overall": float((pred == target).float().mean()),
            "painted": (
                float((pred[painted] == target[painted]).float().mean()) if painted.any() else 0.0
            ),
            "background": (
                float((pred[background] == target[background]).float().mean())
                if background.any()
                else 0.0
            ),
            "painted_columns": int(painted.sum()),
            "background_columns": int(background.sum()),
            "background_fraction": float(background.float().mean()),
            "painted_majority": (
                float(counts.max()) / float(counts.sum()) if counts is not None else 0.0
            ),
        }
    return metrics


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = colour_correct = hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, colour, mix, _, centroids, _ = module(
                tokens_of(model, example), query_of(model, example)
            )
            label = example["target"].decode()
            correct += int(int(shape.argmax()) == LABELS.index(label))
            colour_correct += int(int(colour.argmax()) == colour_index(example))
            if int(shape.argmax()) == 3:
                abstain_unknown += int(label == "未知")
                abstain_other += int(label != "未知")
            if label != "未知":
                truth = oracle_column(example)
                if truth is not None:
                    total += 1
                    chosen = float((mix * centroids).sum())
                    column = min(15, max(0, int((chosen + 1) / 2 * 16)))
                    hits += int(column == truth)
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / len(examples),
        "colour_accuracy": colour_correct / len(examples),
        "centroid_column_hits": hits,
        "centroid_column_total": total,
        "abstain_on_unknown": abstain_unknown,
        "abstain_on_other": abstain_other,
    }


def batch_at(data, order, batch_size, update):
    return [
        data["train"][order[((update - 1) * batch_size + offset) % len(order)]]
        for offset in range(batch_size)
    ]


def decide(probe_fit: float, fit: float, criteria: dict) -> str:
    if probe_fit >= criteria["probe_gate"] and fit >= criteria["fit_gate"]:
        return "column_supervision_makes_parent_reachable"
    if probe_fit >= criteria["probe_gate"] and fit <= criteria["fit_low"]:
        return "colour_in_tokens_readout_still_blocks"
    if probe_fit < criteria["probe_gate"]:
        return "encoder_cannot_carry_colour"
    return "inconclusive_column_supervision"


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, scaffold, optimizer = build(contract, "B")
    order, size = contract["order"], contract["batch"]
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, order, size, update)
        optimizer.zero_grad(set_to_none=True)
        started = time.process_time()
        loss = objective(model, module, scaffold, batch, float(contract["aux_weight"]))
        loss.backward()
        optimizer.step()
        per_step.append(time.process_time() - started)
    step = statistics.median(per_step)
    curve = (max(contract["dose_points"]) * 2 + 3) * step
    report = {
        "schema": "taiji-u2-p4-1-price-v1",
        "card": "22.5.P4.1",
        "date": "2026-11-14",
        "measurements": {
            "per_step_cpu": round(step, 6),
            "per_step_samples": per_step,
            "curve_two_arms_cpu_lower_bound": round(curve, 6),
            "probe_cpu_estimate": round(24 * step, 6),
        },
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "discipline": {"optimizer_calls": 3, "checkpoints": 0, "ledger_writes": 0},
        "note": "three in-memory steps; nothing is saved apart from this receipt",
    }
    write_report(PRICE_REPORT, report)
    print(json.dumps({"per_step_cpu": round(step, 6), "curve": round(curve, 3)}))
    return 0


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p4_1_run_already_started")
    root.mkdir(parents=True)
    doses = contract["dose_points"]
    weight = float(contract["aux_weight"])
    baseline = probe(make_model("native", int(contract["candidate"]["seed"])), data, 1.0)
    arms: dict[str, dict] = {}
    for arm in ("A", "B"):
        model, module, scaffold, optimizer = build(contract, arm)
        rows = []
        for update in range(1, max(doses) + 1):
            if time.process_time() - cpu_start > contract["cpu_cap"]:
                raise BudgetStop("p4_1_budget_stop")
            batch = batch_at(data, contract["order"], contract["batch"], update)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, module, scaffold, batch, weight)
            loss.backward()
            optimizer.step()
            if update not in doses:
                continue
            started = time.process_time()
            measured = probe(model, data, 1.0)
            train = accuracy(model, module, data["train"])
            fit = accuracy(model, module, data["fit"])
            torch.save(
                {
                    "module": module.state_dict(),
                    "encoder": {
                        name: value.detach().clone()
                        for name, value in model.encoder["rgb"].state_dict().items()
                    },
                },
                root / f"arm{arm}-{update:06d}.pt",
            )
            rows.append(
                {
                    "update": update,
                    "probe_train": measured["train"],
                    "probe_fit": measured["fit"],
                    "probe_painted_fit": measured["fit"]["painted"],
                    "probe_overall_fit": measured["fit"]["overall"],
                    "fit_accuracy": fit["accuracy"],
                    "train_accuracy": train["accuracy"],
                    "fit_correct": fit["correct"],
                    "train_correct": train["correct"],
                    "colour_accuracy_fit": fit["colour_accuracy"],
                    "colour_accuracy_train": train["colour_accuracy"],
                    "centroid_column_hits_fit": fit["centroid_column_hits"],
                    "centroid_column_total_fit": fit["centroid_column_total"],
                    "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                    "abstain_on_other_fit": fit["abstain_on_other"],
                    "cpu": round(time.process_time() - started, 6),
                }
            )
        arms[arm] = {
            "doses": rows,
            "trainable_parameters": sum(parameter.numel() for parameter in module.parameters())
            + sum(parameter.numel() for parameter in model.encoder["rgb"].parameters()),
            "scaffold_parameters": (
                0 if scaffold is None else sum(p.numel() for p in scaffold.parameters())
            ),
        }
    final = arms["B"]["doses"][-1]
    verdict = decide(final["probe_fit"]["painted"], final["fit_accuracy"], contract["criteria"])
    report = {
        "schema": "taiji-u2-p4-1-column-supervision-v1",
        "date": "2026-11-14",
        "card": "22.5.P4.1",
        "phase": "column_supervision",
        "verdict": verdict,
        "baseline_probe": baseline,
        "arms": arms,
        "criteria": contract["criteria"],
        "criteria_sha256": contract["criteria_sha256"],
        "batch": contract["batch"],
        "aux_weight": weight,
        "optimizer": contract["optimizer"],
        "training_authority": contract["training_authority"],
        "cost": {
            "total_cpu": round(time.process_time() - cpu_start, 6),
            "total_wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
            "capability_claims": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "scope": (
            "two arms on the body's own rgb encoder: A trains encoder plus readout on the "
            "four-way answer, B adds a 7-way column colour objective via a declared "
            "training-time head; the served readout is unchanged; no capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(json.dumps({"verdict": verdict, "final": final, "cost": report["cost"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("probe", "price", "run"), default="probe")
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "price":
        return price(contract)
    if args.mode == "run":
        return run(contract)
    guards(contract, "probe")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model = make_model("native", int(contract["candidate"]["seed"]))
    measured = probe(model, data, 1.0)
    print(
        json.dumps(
            {
                "probe": measured,
                "cpu": round(time.process_time() - cpu_start, 6),
                "wall": round(time.perf_counter() - wall_start, 6),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
