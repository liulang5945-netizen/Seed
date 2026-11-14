"""P4.1 card: ask the body's own rgb encoder to learn colour, per column.

P3.30 measured the body's column tokens from outside and found the answer
partially there: a 7-way colour ridge probe per position reached 0.674 on fit
while raw pixels reach 1.000, and the 59-parameter readout topped out at fit
0.3125. The diagnosis was that the encoder had never been asked to encode colour
identity - it has only ever seen byte NLL.

This card asks it, with one variable against P3.30: ``encoder.rgb`` unfreezes and
receives a per-column auxiliary cross-entropy whose logits are the negative
squared distances from each column token to the six palette prototypes. The
readout, the data, the order, the doses and the query-colour head are unchanged.
Nothing else in the body trains, and no new parameter is created: the auxiliary
logits are distances, the background and ambiguous columns are simply not
supervised.

The prototypes are recomputed every step from the current encoder and detached.
They are reference geometry for both the auxiliary loss and the readout, never a
gradient path: with the encoder moving, a gradient through the prototypes would
let the loss be paid by moving the target instead of the token.

Verdicts (frozen before any reading):
  probe_fit >= 0.95 and fit >= 0.80  -> pathway_unlocked
  fit >= 0.80                        -> task_carries_probe_lags
  probe_fit >= 0.95 and fit <= 0.55  -> colour_learned_task_blocked
  probe_fit < 0.95 and fit <= 0.55   -> encoder_dead_on_this_task
  prototype collapse override        -> column_supervision_collapsed
  otherwise                          -> inconclusive_column_supervision

Modes: ``price``, ``run`` and ``trial`` (an in-memory engineering trial that
writes nothing; it exists so a degenerate design is caught before freezing).
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

CONTRACT = ROOT / "plans/manifests/taiji_u2_p4_1_column_supervision_20261115.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_p4_1_price_20261115.json"
RUN_REPORT = ROOT / "reports/taiji_u2_p4_1_column_supervision_20261115.json"
LABELS = ["左边", "中间", "右边", "未知"]
CENTRES = [-2 / 3, 0.0, 2 / 3]
QUERY = 8
SIDE = 16
COLOURS = 6
SUFFIX = "现在在哪里？"
TAU_AUX = 1.0
BACKGROUND = "background_not_supervised"


class BudgetStop(RuntimeError):
    pass


def palette_tensor() -> torch.Tensor:
    return torch.tensor([[value / 255 for value in row] for row in RGB_COLORS])


def column_positions() -> torch.Tensor:
    """One entry per column: the tokens are already columns."""
    return 2 * (torch.arange(SIDE) + 0.5) / SIDE - 1


def prototype_tokens(subject) -> torch.Tensor:
    """Derived reference geometry: encoder output for a palette-filled column.

    Recomputed and detached on every call so the auxiliary loss and the readout
    both measure distance to what the encoder says a pure colour column looks
    like right now, without ever training through the reference itself.
    """
    palette = palette_tensor()
    values = palette[:, None, :].repeat(1, SIDE, 1).reshape(COLOURS, SIDE * 3)
    frame = torch.zeros(COLOURS, 8)
    frame[:, 1] = 0
    frame[:, 5] = 1
    frame[:, 7] = 1
    with torch.no_grad():
        tokens = torch.tanh(
            subject.encoder["rgb"](values) + subject.metadata(frame) + subject.modality[0]
        )
    return tokens.detach()


class TokenDetector(nn.Module):
    """Same 59-parameter readout as P3.30; prototypes arrive per forward."""

    def __init__(self, tau: float) -> None:
        super().__init__()
        self.classifier = nn.Linear(QUERY, COLOURS)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_mix = nn.Parameter(torch.zeros(()))
        self.presence = nn.Parameter(torch.tensor(2.0))
        self.abstain = nn.Parameter(torch.tensor(-1.0))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: torch.Tensor, query: torch.Tensor, prototypes: torch.Tensor):
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
        return shape, colour, mix, logits, centroids, null, squared


def sha_of(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()


def guards(contract: dict, mode: str) -> None:
    if sha_of(Path(__file__)) != contract["source_pins"]["runner"]:
        raise ValueError("runner_source_drift")
    if (
        sha_of(ROOT / "scripts/training/first_a_support_22_5.py")
        != contract["source_pins"]["support"]
    ):
        raise ValueError("support_source_drift")
    if (
        sha_of(ROOT / "taiji/u2_observation_columns.py")
        != contract["source_pins"]["column_pathway"]
    ):
        raise ValueError("column_pathway_source_drift")
    if mode == "run":
        if contract["status"] != "frozen":
            raise ValueError("contract_not_frozen")
        if not contract.get("training_authority"):
            raise ValueError("training_authority_absent")
        if not contract.get("cpu_cap"):
            raise ValueError("cpu_cap_absent")


def trainable_of(model):
    return [value for name, value in model.named_parameters() if name.startswith("encoder.rgb")]


def build(contract: dict):
    seed = int(contract["candidate"]["seed"])
    torch.manual_seed(seed)
    model = make_model("native", seed)
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith("encoder.rgb"))
    module = TokenDetector(contract["candidate"]["tau"])
    spec = contract["optimizer"]
    body_optimizer = torch.optim.Adam(
        trainable_of(model),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    module_optimizer = torch.optim.Adam(
        list(module.parameters()),
        lr=spec["lr"],
        betas=tuple(spec["betas"]),
        eps=spec["eps"],
        weight_decay=spec["weight_decay"],
    )
    return model, module, body_optimizer, module_optimizer


def query_of(model, example: dict) -> torch.Tensor:
    return model.text_units(example["public"]["request"]).mean(0)


def tokens_of(model, example: dict) -> torch.Tensor:
    return column_tokens(model, example["public"]["observations"], example["public"]["coordinates"])


def colour_name(example: dict) -> str:
    return example["public"]["request"].decode()[: -len(SUFFIX)]


def colour_index(example: dict) -> int:
    return NAMES.index(colour_name(example))


def unpatched(example: dict) -> torch.Tensor:
    return (
        example["public"]["observations"]["rgb"]
        .reshape(4, 4, 4, 4, 3)
        .permute(0, 2, 1, 3, 4)
        .reshape(SIDE, SIDE, 3)
        * 255.0
    )


_COLUMN_LABELS: dict[str, dict[int, int]] = {}


def column_labels(example: dict) -> dict[int, int]:
    """Which palette colour occupies each column, from the public locator.

    A column claimed by two colours is marked -1 and excluded from the loss
    rather than given a poisoned target. Columns with no colour are background
    and are not supervised; the auxiliary never invents a background class.
    """
    key = example["id"]
    if key in _COLUMN_LABELS:
        return _COLUMN_LABELS[key]
    image = unpatched(example)
    mapping: dict[int, int] = {}
    for index, name in enumerate(NAMES):
        position = locate_public(image, name)
        if position is None:
            continue
        column = min(15, max(0, int((position + 1) / 2 * 16)))
        if column in mapping and mapping[column] != index:
            mapping[column] = -1
        else:
            mapping[column] = index
    _COLUMN_LABELS[key] = mapping
    return mapping


def oracle_column(example: dict) -> int | None:
    labels = column_labels(example)
    queried = colour_index(example)
    for column, index in labels.items():
        if index == queried:
            return column
    return None


def auxiliary_loss(model, batch: list[dict]) -> tuple[torch.Tensor, dict]:
    """Per-column CE over prototype distances for colour-occupied columns only."""
    total = None
    supervised = 0
    ambiguous = 0
    for example in batch:
        labels = column_labels(example)
        ambiguous += sum(1 for value in labels.values() if value < 0)
        usable = {column: index for column, index in labels.items() if index >= 0}
        if not usable:
            continue
        tokens = tokens_of(model, example)
        prototypes = prototype_tokens(model)
        squared = ((tokens[None, :, :] - prototypes[:, None, :]) ** 2).sum(-1)
        logits = (-squared / TAU_AUX).transpose(0, 1)
        columns = torch.tensor(sorted(usable))
        targets = torch.tensor([usable[int(column)] for column in columns])
        term = functional.cross_entropy(logits[columns], targets, reduction="sum")
        total = term if total is None else total + term
        supervised += len(usable)
    if total is None or supervised == 0:
        zero = torch.zeros(())
        return zero.requires_grad_(), {"supervised": 0, "ambiguous": ambiguous}
    return total / supervised, {"supervised": supervised, "ambiguous": ambiguous}


def objective(model, module, batch: list[dict], lam: float):
    shapes, colours = [], []
    for example in batch:
        shape, _, _, logits, _, _, _ = module(
            tokens_of(model, example), query_of(model, example), prototype_tokens(model)
        )
        shapes.append(shape)
        colours.append(logits)
    labels = torch.tensor([LABELS.index(example["target"].decode()) for example in batch])
    loss = functional.cross_entropy(torch.stack(shapes), labels)
    if lam:
        targets = torch.tensor([colour_index(example) for example in batch])
        loss = loss + lam * functional.cross_entropy(torch.stack(colours), targets)
    aux, stats = auxiliary_loss(model, batch)
    loss = loss + lam * aux
    return loss, aux, stats


def accuracy(model, module, examples: list[dict]) -> dict:
    with torch.no_grad():
        correct = 0
        colour_correct = 0
        hits = total = 0
        abstain_unknown = abstain_other = 0
        for example in examples:
            shape, colour, mix, _, centroids, _, _ = module(
                tokens_of(model, example), query_of(model, example), prototype_tokens(model)
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


def ridge_probe(model, data: dict) -> dict:
    """7-way colour decodability of one column token, ridge on train, test on fit."""
    classes = COLOURS + 1  # six palette colours plus a background class

    def rows(examples: list[dict]):
        features, targets = [], []
        for example in examples:
            with torch.no_grad():
                tokens = tokens_of(model, example)
            labels = column_labels(example)
            for column in range(SIDE):
                features.append(tokens[column])
                targets.append(labels.get(column, COLOURS))
        return torch.stack(features), torch.tensor(targets)

    x_train, y_train = rows(data["train"])
    x_fit, y_fit = rows(data["fit"])
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-9)
    a = (x_train - mean) / std
    b = (x_fit - mean) / std
    one_hot = torch.zeros(len(y_train), classes)
    one_hot[torch.arange(len(y_train)), y_train] = 1
    identity = torch.eye(a.shape[1])
    weights = torch.linalg.solve(a.T @ a + identity, a.T @ one_hot)
    train_accuracy = float(((a @ weights).argmax(1) == y_train).float().mean())
    fit_accuracy = float(((b @ weights).argmax(1) == y_fit).float().mean())
    return {"train": train_accuracy, "fit": fit_accuracy, "classes": classes}


def geometry(model) -> dict:
    prototypes = prototype_tokens(model)
    pairwise = torch.cdist(prototypes, prototypes)
    off_diagonal = pairwise[~torch.eye(COLOURS, dtype=torch.bool)]
    return {
        "prototype_pairwise_min": round(float(off_diagonal.min()), 6),
        "prototype_pairwise_mean": round(float(off_diagonal.mean()), 6),
    }


def batch_at(data, order, batch_size, update):
    return [
        data["train"][order[((update - 1) * batch_size + offset) % len(order)]]
        for offset in range(batch_size)
    ]


def price(contract: dict) -> int:
    guards(contract, "price")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, body_optimizer, module_optimizer = build(contract)
    per_step = []
    for update in range(1, 4):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        started = time.process_time()
        body_optimizer.zero_grad(set_to_none=True)
        module_optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(model, module, batch, float(contract["aux_weight"]))
        loss.backward()
        body_optimizer.step()
        module_optimizer.step()
        per_step.append(time.process_time() - started)
        if time.process_time() - cpu_start > contract["price_cap_cpu"]:
            raise BudgetStop("price_probe_budget_stop")
    started = time.process_time()
    scored = accuracy(model, module, data["fit"])
    baseline = ridge_probe(model, data)
    eval_cpu = time.process_time() - started
    step_cpu = statistics.median(per_step)
    curve_cpu = max(contract["dose_points"]) * step_cpu + 8 * eval_cpu
    cap = round(min(contract["design_cap_cpu"], max(curve_cpu * 1.1, 60.0)), 3)
    expected = contract["criteria"]["baseline_check"]["expected_probe_fit_at_zero"]
    tolerance = contract["criteria"]["baseline_check"]["tolerance"]
    report = {
        "schema": "taiji-u2-p4-1-price-v1",
        "date": "2026-11-15",
        "card": "22.5.P4.1",
        "phase": "price_probe",
        "trainable_parameters": {
            "encoder_rgb": sum(value.numel() for value in trainable_of(model)),
            "module": sum(value.numel() for value in module.parameters()),
        },
        "measurements": {
            "per_step_cpu": [round(value, 6) for value in per_step],
            "per_step_cpu_median": round(step_cpu, 6),
            "eval_cpu": round(eval_cpu, 6),
            "eval_accuracy_at_zero": scored["accuracy"],
            "baseline_probe": baseline,
            "baseline_matches_preflight": abs(baseline["fit"] - expected) <= tolerance,
            "geometry_at_zero": geometry(model),
        },
        "derived": {
            "curve_updates": max(contract["dose_points"]),
            "curve_cpu_estimate": round(curve_cpu, 3),
            "cap_cpu": cap,
            "cap_wall": round(cap * 2, 3),
        },
        "verdict": "price_measured" if cap <= contract["design_cap_cpu"] else "pricing_over_budget",
        "discipline": {
            "checkpoints_written": 0,
            "optimizer_calls": 3,
            "check_reads": 0,
            "final_reads": 0,
            "confirm_reads": 0,
        },
        "run_conditions": {"threads": torch.get_num_threads(), "torch": torch.__version__},
        "measured": {
            "cpu_seconds": round(time.process_time() - cpu_start, 6),
            "wall_seconds": round(time.perf_counter() - wall_start, 6),
        },
        "scope": "pricing only; three in-memory steps are discarded, no capability claim",
    }
    write_report(PRICE_REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cap_cpu": cap,
                "curve_cpu": round(curve_cpu, 3),
                "baseline_probe_fit": baseline["fit"],
                "baseline_matches_preflight": report["measurements"]["baseline_matches_preflight"],
            }
        )
    )
    return 0 if report["verdict"] == "price_measured" else 4


def trial(contract: dict, steps: int) -> int:
    """In-memory engineering trial. Writes nothing; exists to catch degeneracy."""
    guards(contract, "price")
    torch.set_num_threads(1)
    data = atomic_load(Path(contract["data_root"]))
    model, module, body_optimizer, module_optimizer = build(contract)
    lam = float(contract["aux_weight"])
    history = []
    for update in range(1, steps + 1):
        batch = batch_at(data, contract["order"], contract["batch"], update)
        body_optimizer.zero_grad(set_to_none=True)
        module_optimizer.zero_grad(set_to_none=True)
        loss, aux, stats = objective(model, module, batch, lam)
        loss.backward()
        body_optimizer.step()
        module_optimizer.step()
        if update in (1, steps // 3, 2 * steps // 3, steps):
            scored = accuracy(model, module, data["fit"])
            probe = ridge_probe(model, data)
            geo = geometry(model)
            history.append(
                {
                    "update": update,
                    "loss": round(float(loss.detach()), 4),
                    "aux": round(float(aux.detach()), 4),
                    "supervised": stats["supervised"],
                    "fit": scored["accuracy"],
                    "colour": scored["colour_accuracy"],
                    "column_hits": scored["centroid_column_hits"],
                    "probe_train": round(probe["train"], 4),
                    "probe_fit": round(probe["fit"], 4),
                    "proto_min_dist": geo["prototype_pairwise_min"],
                }
            )
            print(json.dumps(history[-1]))
    log6 = math.log(COLOURS)
    last = history[-1]
    collapsed = last["proto_min_dist"] < 1e-4 and abs(last["aux"] - log6) < 0.01
    print(
        json.dumps(
            {
                "trial_verdict": "column_supervision_collapsed" if collapsed else "alive",
                "log6": round(log6, 4),
            }
        )
    )
    return 5 if collapsed else 0


def decide(final: dict, criteria: dict) -> str:
    gate, low, probe_gate = criteria["fit_gate"], criteria["fit_low"], criteria["probe_gate"]
    if final["geometry_collapse"]:
        return "column_supervision_collapsed"
    if final["fit_accuracy"] >= gate and final["probe_fit"] >= probe_gate:
        return "pathway_unlocked"
    if final["fit_accuracy"] >= gate:
        return "task_carries_probe_lags"
    if final["fit_accuracy"] <= low and final["probe_fit"] >= probe_gate:
        return "colour_learned_task_blocked"
    if final["fit_accuracy"] <= low:
        return "encoder_dead_on_this_task"
    return "inconclusive_column_supervision"


def run(contract: dict) -> int:
    guards(contract, "run")
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = atomic_load(Path(contract["data_root"]))
    assert value_digest(data) == contract["data_digest"]
    model, module, body_optimizer, module_optimizer = build(contract)
    doses = contract["dose_points"]
    root = Path(contract["run_root"])
    if root.exists() or RUN_REPORT.exists():
        raise FileExistsError("p4_1_run_already_started")
    root.mkdir(parents=True)
    baseline = ridge_probe(model, data)
    lam = float(contract["aux_weight"])
    log6 = math.log(COLOURS)
    results = []
    for update in range(1, max(doses) + 1):
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("p4_1_budget_stop")
        batch = batch_at(data, contract["order"], contract["batch"], update)
        body_optimizer.zero_grad(set_to_none=True)
        module_optimizer.zero_grad(set_to_none=True)
        loss, aux, stats = objective(model, module, batch, lam)
        loss.backward()
        body_optimizer.step()
        module_optimizer.step()
        if update not in doses:
            continue
        started = time.process_time()
        train = accuracy(model, module, data["train"])
        fit = accuracy(model, module, data["fit"])
        probe = ridge_probe(model, data)
        geo = geometry(model)
        torch.save(
            {
                "module": module.state_dict(),
                "encoder_rgb": {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                    if name.startswith("encoder.rgb")
                },
            },
            root / f"bundle-{update:06d}.pt",
        )
        results.append(
            {
                "update": update,
                "train_accuracy": train["accuracy"],
                "fit_accuracy": fit["accuracy"],
                "train_correct": train["correct"],
                "fit_correct": fit["correct"],
                "colour_accuracy_train": train["colour_accuracy"],
                "colour_accuracy_fit": fit["colour_accuracy"],
                "centroid_column_hits_fit": fit["centroid_column_hits"],
                "centroid_column_total_fit": fit["centroid_column_total"],
                "abstain_on_unknown_fit": fit["abstain_on_unknown"],
                "abstain_on_other_fit": fit["abstain_on_other"],
                "probe_train": round(probe["train"], 6),
                "probe_fit": round(probe["fit"], 6),
                "aux_last": round(float(aux.detach()), 6),
                "aux_supervised_last": stats["supervised"],
                "prototype_pairwise_min": geo["prototype_pairwise_min"],
                "geometry_collapse": bool(
                    geo["prototype_pairwise_min"] < 1e-4 and abs(float(aux.detach()) - log6) < 0.01
                ),
                "cpu": round(time.process_time() - started, 6),
            }
        )
    verdict = decide(results[-1], contract["criteria"])
    report = {
        "schema": "taiji-u2-p4-1-column-supervision-v1",
        "date": "2026-11-15",
        "card": "22.5.P4.1",
        "phase": "column_colour_supervision",
        "verdict": verdict,
        "baseline_probe": baseline,
        "doses": results,
        "trainable_parameters": {
            "encoder_rgb": sum(value.numel() for value in trainable_of(model)),
            "module": sum(value.numel() for value in module.parameters()),
        },
        "batch": contract["batch"],
        "optimizer": contract["optimizer"],
        "control_arms": contract["control_arms"],
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
            "single-variable card against P3.30: encoder.rgb unfreezes and receives a "
            "per-column prototype-distance CE; the 59-parameter readout, data, order, doses "
            "and query-colour head are unchanged; prototypes are per-step derived reference "
            "geometry and never a gradient path; no capability claim"
        ),
    }
    write_report(RUN_REPORT, report)
    print(json.dumps({"verdict": verdict, "last": results[-1], "cost": report["cost"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("price", "run", "trial"), default="price")
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if args.mode == "price":
        return price(contract)
    if args.mode == "trial":
        return trial(contract, int(args.steps))
    try:
        guards(contract, "run")
    except ValueError as error:
        print(json.dumps({"verdict": "refused", "reason": str(error)}))
        return 2
    try:
        return run(contract)
    except BudgetStop as error:
        if not RUN_REPORT.exists():
            write_report(
                RUN_REPORT,
                {
                    "schema": "taiji-u2-p4-1-column-supervision-v1",
                    "date": "2026-11-15",
                    "card": "22.5.P4.1",
                    "verdict": "budget_stop",
                    "reason": str(error),
                },
            )
        return 3
    except Exception as error:  # noqa: BLE001
        if not RUN_REPORT.exists():
            write_report(
                RUN_REPORT,
                {
                    "schema": "taiji-u2-p4-1-column-supervision-v1",
                    "date": "2026-11-15",
                    "card": "22.5.P4.1",
                    "verdict": "engineering_failure",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
