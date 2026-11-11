"""10.7d margin-objective probe: does a pairwise loss strip the common mode?

10.7b/10.7c showed the six sample gradients agree to 0.999977 and that the small
residual *does* carry the correct-object split. The natural next question is whether
a loss that compares the two actions **inside one sample** behaves differently:

* plain loss (10.7's): ``NLL(yes|correct) + NLL(no|other)``;
* margin loss: ``softplus(-(verdict_other - verdict_correct))`` where
  ``verdict`` is the 10.6 construction ``NLL(no) - NLL(yes)``. The state and goal
  literal prior is shared by both verdicts and cancels inside the difference.

It trains nothing: both gradients are read and discarded, and the plain branch is
also a cross-check that this script reproduces 10.7b/10.7c's numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_subject_selection_10_4 as prior  # noqa: E402
from taiji.action_verdict_scorer import NO, YES, prefix  # noqa: E402
from taiji.constrained_learning import gradient_vector  # noqa: E402
from taiji.goal_conditioned_training import _nll, sample_loss, samples  # noqa: E402

TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
STRUCTURE_FLOOR = 0.5
COLLINEAR_CEILING = 0.9
COHERENCE_FLOOR = 0.5


def cosine(first: torch.Tensor, second: torch.Tensor):
    first = first.double()
    second = second.double()
    norm = float(first.norm()) * float(second.norm())
    if norm == 0.0:
        return None
    return float(torch.dot(first, second)) / norm


def explained_fraction(residuals, labels) -> float:
    total = sum(float(residual.norm() ** 2) for residual in residuals)
    if total == 0.0:
        return 0.0
    groups: dict[str, list[torch.Tensor]] = {}
    for residual, label in zip(residuals, labels):
        groups.setdefault(label, []).append(residual)
    explained = 0.0
    for values in groups.values():
        mean = torch.stack(values).mean(dim=0)
        explained += len(values) * float(mean.norm() ** 2)
    return explained / total


def within_cosine(residuals, labels):
    groups: dict[str, list[torch.Tensor]] = {}
    for residual, label in zip(residuals, labels):
        groups.setdefault(label, []).append(residual)
    out = {}
    for name, values in groups.items():
        pairs = [
            cosine(values[first], values[second])
            for first, second in itertools.combinations(range(len(values)), 2)
        ]
        pairs = [value for value in pairs if value is not None]
        out[name] = statistics.fmean(pairs) if pairs else None
    return out


def structure(gradients, labels) -> dict:
    mean = torch.stack(gradients).mean(dim=0)
    residuals = [gradient - mean for gradient in gradients]
    pairs = [
        cosine(gradients[first], gradients[second])
        for first, second in itertools.combinations(range(len(gradients)), 2)
    ]
    pairs = [value for value in pairs if value is not None]
    coherence = within_cosine(residuals, labels)
    return {
        "norms": [float(gradient.norm()) for gradient in gradients],
        "mean_norm": float(mean.norm()),
        "pairwise_cosine_min": min(pairs) if pairs else None,
        "pairwise_cosine_median": statistics.median(pairs) if pairs else None,
        "pairwise_cosine_max": max(pairs) if pairs else None,
        "explained_fraction": explained_fraction(residuals, labels),
        "within_group_cosine": coherence,
    }


def classify_margin(stats: dict) -> str:
    coherence = [value for value in stats["within_group_cosine"].values() if value is not None]
    if not coherence or stats["pairwise_cosine_median"] is None:
        return "margin_undefined"
    if (
        stats["explained_fraction"] >= STRUCTURE_FLOOR
        and min(coherence) >= COHERENCE_FLOOR
        and stats["pairwise_cosine_median"] < COLLINEAR_CEILING
    ):
        return "margin_isolates_differential"
    if stats["pairwise_cosine_median"] >= COLLINEAR_CEILING:
        return "margin_still_common"
    return "margin_mixed"


def margin_loss(subject, sample: dict, provider):
    correct = prefix(
        sample["state"],
        sample["goal"],
        f"把{sample['correct'][0]}放到{sample['correct'][1]}",
    )
    other = prefix(
        sample["state"],
        sample["goal"],
        f"把{sample['other'][0]}放到{sample['other'][1]}",
    )
    yes_correct = _nll(subject, correct, YES, provider)
    no_correct = _nll(subject, correct, NO, provider)
    yes_other = _nll(subject, other, YES, provider)
    no_other = _nll(subject, other, NO, provider)
    verdict_correct = no_correct - yes_correct
    verdict_other = no_other - yes_other
    margin = verdict_other - verdict_correct
    return torch.nn.functional.softplus(-margin), float(margin.detach())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7d diagnostic card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    parameters = [parameter for parameter in subject.parameters() if parameter.requires_grad]

    rows = samples()
    per_state = len(rows) // 6
    train_rows = rows[: TRAIN_STATES * per_state]

    plain, margin, labels, margins, failure = [], [], [], [], None
    try:
        for sample in train_rows:
            subject.zero_grad(set_to_none=True)
            plain.append(gradient_vector(sample_loss(subject, sample, provider), parameters))
            subject.zero_grad(set_to_none=True)
            loss, value = margin_loss(subject, sample, provider)
            margin.append(gradient_vector(loss, parameters))
            labels.append(sample["correct"][0])
            margins.append(value)
        subject.zero_grad(set_to_none=True)
    except Exception as error:  # noqa: BLE001 - the exit needs the message
        failure = f"{type(error).__name__}: {error}"

    stats = {}
    verdict = "margin_undefined"
    if not failure:
        plain_stats = structure(plain, labels)
        margin_stats = structure(margin, labels)
        stats = {
            "margins": margins,
            "plain": plain_stats,
            "margin": margin_stats,
            "cos_plain_margin": cosine(
                torch.stack(plain).mean(dim=0), torch.stack(margin).mean(dim=0)
            ),
        }
        verdict = classify_margin(margin_stats)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif failure is not None:
        status = "engineering_failure"
    else:
        status = verdict

    report = {
        "schema": "taiji-margin-probe-10-7d-v1",
        "step": "10.7d",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(train_rows),
            "plain_loss": "NLL(yes|correct) + NLL(no|other) (10.7's loss)",
            "margin_loss": "softplus(-(verdict_other - verdict_correct)), verdict = NLL(no)-NLL(yes)",
            "structure_floor": STRUCTURE_FLOOR,
            "collinear_ceiling": COLLINEAR_CEILING,
            "coherence_floor": COHERENCE_FLOOR,
            "plain_cross_check": "plain.explained_fraction should match 10.7c's 0.645858",
        },
        "rebuilt_digest": digest,
        "verdict": verdict,
        "stats": stats,
        "failure": failure,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Diagnostic only: six samples, one deterministic subject, both gradients read "
            "and discarded. A clean margin gradient says the loss has something to aim at, "
            "not that training it moves held-out selection; that would need its own "
            "authorisation. No parameter was updated and no capability is claimed."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_conditioned_training.py",
                "taiji/action_verdict_scorer.py",
                "scripts/training/diagnose_margin_objective_10_7d.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "verdict": verdict,
                "plain_F": stats.get("plain", {}).get("explained_fraction"),
                "margin_F": stats.get("margin", {}).get("explained_fraction"),
                "margin_pairwise_median": stats.get("margin", {}).get("pairwise_cosine_median"),
                "plain_pairwise_median": stats.get("plain", {}).get("pairwise_cosine_median"),
                "margin_within": stats.get("margin", {}).get("within_group_cosine"),
                "cos_plain_margin": stats.get("cos_plain_margin"),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
