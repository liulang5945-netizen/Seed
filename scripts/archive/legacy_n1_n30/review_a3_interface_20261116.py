"""22.5.A.3: which interface can carry the read's evidence into the byte path?

A.2 handed the frozen four-way answer to the byte path as a fixed code and the
generator wrote it exactly. The open question is how the *evidence* gets there
when the answer is not handed over. This review is zero-training on the
subject: it measures whether the body's own readout input ``x`` can expose the
frozen read's class at all, and compares that with the column tokens that the
read actually uses.

Three interfaces are on the table:

  A  fixed evidence projection   corrected per-column logits -> fixed projection
  B  trainable read adapter      corrected per-column logits -> learned adapter
  C  read-as-teacher distillation the body's own ``x`` must carry the class

C is the only one that keeps the subject as the reader, so its viability is
measured first.  Probes are closed-form linear and RBF ridge fits; there are no
iterative updates and no subject optimizer steps.

Decision rule (pre-registered): with a label-shuffle null at or below 0.35,
max(linear fit, RBF fit) >= 0.80 is ``interface_decodable``, 0.40-0.80 is
``interface_partially_decodable`` and below 0.40 is
``interface_not_decodable``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import run_a2_byte_20261115 as a2
from scripts.training import run_p4_1_22_5 as p4_1
from scripts.training import run_p4_2_22_5 as p4_2
from scripts.training.check_report_run_conditions import write_report

CONTRACT = ROOT / "plans/manifests/taiji_u2_a3_interface_20261116.json"
A2_CONTRACT = ROOT / "plans/manifests/taiji_u2_a2_byte_20261115.json"
A2_RUNNER = ROOT / "scripts/training/run_a2_byte_20261115.py"
REPORT = ROOT / "reports/taiji_u2_a3_interface_20261116.json"
CLASSES = 4


class BudgetStop(RuntimeError):
    pass


def guards(contract: dict) -> None:
    pins = contract.get("source_pins") or {}
    for name, path in (
        ("runner", Path(__file__)),
        ("subject", ROOT / "taiji/u2_subject.py"),
        ("p4_1_runner", ROOT / "scripts/training/run_p4_1_22_5.py"),
        ("p4_2_runner", ROOT / "scripts/training/run_p4_2_22_5.py"),
        ("support", ROOT / "scripts/training/first_a_support_22_5.py"),
        ("columns", ROOT / "taiji/u2_observation_columns.py"),
        ("a2_runner", A2_RUNNER),
        ("a2_contract", A2_CONTRACT),
    ):
        if pins.get(name) != p4_2.sha_of(path):
            raise ValueError(f"source_pin_mismatch:{name}")
    if contract.get("status") != "frozen":
        raise ValueError("contract_not_frozen")


def cache_x(model, examples: list[dict]) -> dict[str, list[torch.Tensor]]:
    """Prefix-order ``joint_readout`` inputs of the frozen body, per example."""
    codes = {
        example["id"]: torch.zeros(int(3 * model.c.d), dtype=torch.float32) for example in examples
    }
    positions = a2.cache_positions(model, examples, codes)
    grouped: dict[str, list[torch.Tensor]] = {}
    for item in positions:
        grouped.setdefault(item["example"], []).append(item["x"].flatten())
    order = {example["id"]: index for index, example in enumerate(examples)}
    return {key: value for key, value in sorted(grouped.items(), key=lambda pair: order[pair[0]])}


def token_features(model, examples: list[dict]) -> torch.Tensor:
    return torch.stack([p4_1.tokens_of(model, example).flatten() for example in examples])


def x_at_position(cache: dict[str, list[torch.Tensor]], examples: list[dict], position: int):
    return torch.stack([cache[example["id"]][position] for example in examples])


def x_mean(cache: dict[str, list[torch.Tensor]], examples: list[dict]):
    return torch.stack([torch.stack(cache[example["id"]]).mean(0) for example in examples])


def standardize(train: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    scale = train.std(0).clamp_min(1e-6)
    return (train - mean) / scale, (test - mean) / scale


def ridge(
    train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, lam: float
) -> list[int]:
    train_x, test_x = standardize(train_x, test_x)
    train_x = torch.cat([train_x, torch.ones(train_x.shape[0], 1)], -1)
    test_x = torch.cat([test_x, torch.ones(test_x.shape[0], 1)], -1)
    target = torch.nn.functional.one_hot(train_y, CLASSES).to(train_x.dtype)
    gram = train_x.T @ train_x + lam * torch.eye(train_x.shape[1])
    weights = torch.linalg.solve(gram, train_x.T @ target)
    return (test_x @ weights).argmax(-1).tolist()


def rbf_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    lam: float,
    gamma: float,
) -> list[int]:
    train_x, test_x = standardize(train_x, test_x)
    squared_train = ((train_x[:, None, :] - train_x[None, :, :]) ** 2).sum(-1)
    squared_test = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(-1)
    kernel = (-gamma * squared_train).exp()
    cross = (-gamma * squared_test).exp()
    target = torch.nn.functional.one_hot(train_y, CLASSES).to(train_x.dtype)
    alpha = torch.linalg.solve(kernel + lam * torch.eye(kernel.shape[0]), target)
    return (cross @ alpha).argmax(-1).tolist()


def accuracy(predicted: list[int], truth: torch.Tensor) -> float:
    return sum(int(a == b) for a, b in zip(predicted, truth.tolist(), strict=True)) / len(truth)


def probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    lam: float = 1.0,
    gamma: float = 1.0 / 24.0,
    shuffles: int = 5,
) -> dict:
    linear_fit = accuracy(ridge(train_x, train_y, test_x, lam), test_y)
    linear_train = accuracy(ridge(train_x, train_y, train_x, lam), train_y)
    rbf_fit = accuracy(rbf_ridge(train_x, train_y, test_x, lam, gamma), test_y)
    rbf_train = accuracy(rbf_ridge(train_x, train_y, train_x, lam, gamma), train_y)
    generator = torch.Generator().manual_seed(20261116)
    nulls = []
    for _ in range(shuffles):
        shuffled = train_y[torch.randperm(len(train_y), generator=generator)]
        nulls.append(
            max(
                accuracy(ridge(train_x, shuffled, test_x, lam), test_y),
                accuracy(rbf_ridge(train_x, shuffled, test_x, lam, gamma), test_y),
            )
        )
    return {
        "linear_train": linear_train,
        "linear_fit": linear_fit,
        "rbf_train": rbf_train,
        "rbf_fit": rbf_fit,
        "best_fit": max(linear_fit, rbf_fit),
        "null_max": max(nulls),
        "null_mean": sum(nulls) / len(nulls),
    }


def class_labels(model, head, prior, examples: list[dict]) -> torch.Tensor:
    return torch.tensor(
        [a2.read_class(model, head, prior, example) for example in examples],
        dtype=torch.long,
    )


def colour_probe(
    train_x: torch.Tensor,
    train_labels: list[torch.Tensor],
    test_x: torch.Tensor,
    test_labels: list[torch.Tensor],
) -> dict:
    train_y = torch.cat(train_labels)
    test_y = torch.cat(test_labels)
    train_x, test_x = standardize(train_x, test_x)
    train_x = torch.cat([train_x, torch.ones(train_x.shape[0], 1)], -1)
    test_x = torch.cat([test_x, torch.ones(test_x.shape[0], 1)], -1)
    target = torch.nn.functional.one_hot(train_y, 7).to(train_x.dtype).reshape(-1, 16 * 7)
    gram = train_x.T @ train_x + 10.0 * torch.eye(train_x.shape[1])
    weights = torch.linalg.solve(gram, train_x.T @ target)
    predicted = (test_x @ weights).reshape(-1, 16, 7).argmax(-1).reshape(-1)
    truth = test_y.reshape(-1)
    painted = truth < 6
    return {
        "overall": float((predicted == truth).float().mean()),
        "painted": (
            float((predicted[painted] == truth[painted]).float().mean())
            if bool(painted.any())
            else 0.0
        ),
        "background": (
            float((predicted[~painted] == truth[~painted]).float().mean())
            if bool((~painted).any())
            else 0.0
        ),
        "painted_columns": int(painted.sum()),
    }


def decide(primary: dict, null_max: float, criteria: dict) -> str:
    if null_max > criteria["null_max"]:
        return "probe_unreliable"
    best = primary["best_fit"]
    if best >= criteria["decodable_min"]:
        return "interface_decodable"
    if best >= criteria["partial_min"]:
        return "interface_partially_decodable"
    return "interface_not_decodable"


def run(contract: dict) -> int:
    guards(contract)
    torch.set_num_threads(1)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    data = a2.atomic_load(ROOT / contract["data"]["root"])
    assert a2.value_digest(data) == contract["data"]["digest"]
    model = a2.build_subject(contract)
    head = a2.load_head(contract)
    prior = a2.prior_of(data)
    state_before = a2.value_digest(model.state_dict())
    read_fit = a2.read_rows(model, head, prior, data["fit"])
    read_train = a2.read_rows(model, head, prior, data["train"])

    def budget() -> None:
        if time.process_time() - cpu_start > contract["cpu_cap"]:
            raise BudgetStop("a3_budget_stop")
        if time.perf_counter() - wall_start > contract["wall_cap"]:
            raise BudgetStop("a3_wall_stop")

    budget()
    train_cache = cache_x(model, data["train"])
    budget()
    fit_cache = cache_x(model, data["fit"])
    budget()
    train_tokens = token_features(model, data["train"])
    fit_tokens = token_features(model, data["fit"])
    budget()

    train_labels = class_labels(model, head, prior, data["train"])
    fit_labels = class_labels(model, head, prior, data["fit"])
    criteria = contract["criteria"]["decision"]
    primary = probe(
        x_at_position(train_cache, data["train"], 0),
        train_labels,
        x_at_position(fit_cache, data["fit"], 0),
        fit_labels,
        gamma=1.0 / 24.0,
    )
    budget()
    mean_pooled = probe(
        x_mean(train_cache, data["train"]),
        train_labels,
        x_mean(fit_cache, data["fit"]),
        fit_labels,
        gamma=1.0 / 24.0,
    )
    budget()
    tokens = probe(train_tokens, train_labels, fit_tokens, fit_labels, gamma=1.0 / 128.0)
    budget()
    by_position = {}
    for position in range(len(next(iter(train_cache.values())))):
        row = probe(
            x_at_position(train_cache, data["train"], position),
            train_labels,
            x_at_position(fit_cache, data["fit"], position),
            fit_labels,
            gamma=1.0 / 24.0,
            shuffles=1,
        )
        by_position[str(position)] = {
            key: row[key] for key in ("linear_fit", "rbf_fit", "best_fit")
        }
    budget()
    columns = colour_probe(
        x_at_position(train_cache, data["train"], 0),
        [p4_1.column_labels(example) for example in data["train"]],
        x_at_position(fit_cache, data["fit"], 0),
        [p4_1.column_labels(example) for example in data["fit"]],
    )
    state_after = a2.value_digest(model.state_dict())
    verdict = decide(primary, primary["null_max"], criteria)
    payload = {
        "schema": "taiji-u2-a3-interface-v1",
        "card": "22.5.A.3",
        "date": "2026-11-16",
        "verdict": verdict,
        "candidate": contract["candidate"],
        "data_digest": contract["data"]["digest"],
        "read": {
            "fit": {key: read_fit[key] for key in ("correct", "denominator", "accuracy")},
            "train": {key: read_train[key] for key in ("correct", "denominator", "accuracy")},
        },
        "features": {
            "x_at_prefix0": int(x_at_position(train_cache, data["train"], 0).shape[1]),
            "x_mean_over_prefixes": int(x_mean(train_cache, data["train"]).shape[1]),
            "column_tokens": int(train_tokens.shape[1]),
            "train": len(data["train"]),
            "fit": len(data["fit"]),
        },
        "primary": primary,
        "mean_pooled": mean_pooled,
        "reference_tokens": tokens,
        "by_position": by_position,
        "column_colours_from_x": columns,
        "decision_rule": criteria,
        "subject_state_unchanged": state_before == state_after,
        "cost": {
            "cpu": round(time.process_time() - cpu_start, 6),
            "wall": round(time.perf_counter() - wall_start, 6),
            "cap_cpu": contract["cpu_cap"],
            "cap_wall": contract["wall_cap"],
        },
        "discipline": {
            "subject_optimizer_steps": 0,
            "iterative_probe_training": 0,
            "probe_fits": "closed-form linear ridge and RBF kernel ridge only",
            "checkpoints_written": 0,
            "capability_claims": 0,
            "accepted_parent": False,
            "check_final_reads": 0,
        },
        "run_conditions": {"threads": 1, "torch": torch.__version__},
        "scope": (
            "zero-training interface review: closed-form probes only; no subject update, "
            "no checkpoint, no capability or parent claim"
        ),
    }
    write_report(REPORT, payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "x_linear": primary["linear_fit"],
                "x_rbf": primary["rbf_fit"],
                "null": primary["null_max"],
                "tokens": tokens["best_fit"],
                "cost": payload["cost"],
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    return run(contract)


if __name__ == "__main__":
    sys.exit(main())
