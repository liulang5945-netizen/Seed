"""10.7c residual structure: is the 1.8e-5 left after subtracting the mean real or noise?

10.7b showed the six per-sample gradients agree to 0.999977, leaving a residual with
1.75e-5 of the energy. The next question is whether that residual is *about* the
correct-object split or just scatter:

* ``F`` = fraction of residual energy explained by the two groups
  (``sum_g n_g ||r_g||^2 / sum_i ||r_i||^2``). For six vectors split 3/3 at random
  the expectation is ``(k-1)/(n-1) = 0.2``;
* ``F`` under every balanced relabelling gives an exact permutation rank;
* mean cosine of residuals inside each group says whether they point one way.

It trains nothing: gradients are read and discarded.
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
from taiji.constrained_learning import gradient_vector  # noqa: E402
from taiji.goal_conditioned_training import sample_loss, samples  # noqa: E402

TRAIN_STATES = 3
BUDGET_CPU_SECONDS = 300
BUDGET_WALL_SECONDS = 600
EXPLAINED_FLOOR = 0.5
NOISE_CEILING = 0.35
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
    for residual, label in zip(residuals, labels, strict=False):
        groups.setdefault(label, []).append(residual)
    explained = 0.0
    for values in groups.values():
        mean = torch.stack(values).mean(dim=0)
        explained += len(values) * float(mean.norm() ** 2)
    return explained / total


def within_cosine(residuals, labels):
    groups: dict[str, list[torch.Tensor]] = {}
    for residual, label in zip(residuals, labels, strict=False):
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


def classify(fraction, coherence) -> str:
    values = [value for value in coherence.values() if value is not None]
    if fraction is None or not values:
        return "no_structure"
    if fraction >= EXPLAINED_FLOOR and min(values) >= COHERENCE_FLOOR:
        return "residual_group_structure"
    if fraction <= NOISE_CEILING:
        return "residual_noise_like"
    return "residual_mixed"


def flat_parameters(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7c diagnostic card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    parameters = [parameter for parameter in subject.parameters() if parameter.requires_grad]

    rows = samples()
    per_state = len(rows) // 6
    train_rows = rows[: TRAIN_STATES * per_state]

    gradients, labels, losses, failure = [], [], [], None
    try:
        for sample in train_rows:
            subject.zero_grad(set_to_none=True)
            loss = sample_loss(subject, sample, provider)
            gradients.append(gradient_vector(loss, parameters))
            labels.append(sample["correct"][0])
            losses.append(float(loss.detach()))
        subject.zero_grad(set_to_none=True)
    except Exception as error:  # noqa: BLE001 - the exit needs the message
        failure = f"{type(error).__name__}: {error}"

    stats = {}
    verdict = "no_structure"
    if not failure:
        mean = torch.stack(gradients).mean(dim=0)
        residuals = [gradient - mean for gradient in gradients]
        fraction = explained_fraction(residuals, labels)
        coherence = within_cosine(residuals, labels)
        permutations = []
        half = len(residuals) // 2
        for rest in itertools.combinations(range(1, len(residuals)), half - 1):
            chosen = (0, *rest)
            assignment = ["a" if index in chosen else "b" for index in range(len(residuals))]
            permutations.append(explained_fraction(residuals, assignment))
        ordered = sorted(permutations)
        ties = sum(1 for value in permutations if value >= fraction - 1e-12)
        stats = {
            "residual_norms": [float(residual.norm()) for residual in residuals],
            "explained_fraction": fraction,
            "expected_fraction_under_random_split": (len(set(labels)) - 1) / (len(residuals) - 1),
            "permutation_fractions": permutations,
            "permutation_at_least_observed": ties,
            "permutation_total": len(permutations),
            "permutation_p": ties / len(permutations),
            "within_group_cosine": coherence,
            "group_order": sorted(ordered),
        }
        verdict = classify(fraction, coherence)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif failure is not None:
        status = "engineering_failure"
    else:
        status = verdict if verdict != "no_structure" else "residual_no_structure"

    report = {
        "schema": "taiji-residual-structure-10-7c-v1",
        "step": "10.7c",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(train_rows),
            "grouping": "correct object (3 vs 3)",
            "explained_floor": EXPLAINED_FLOOR,
            "noise_ceiling": NOISE_CEILING,
            "coherence_floor": COHERENCE_FLOOR,
        },
        "rebuilt_digest": digest,
        "verdict": verdict,
        "stats": stats,
        "per_sample_losses": losses,
        "failure": failure,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Diagnostic only: six samples, one deterministic subject, gradients read and "
            "discarded. Ten distinct balanced relabelings make the smallest attainable "
            "permutation p 0.1, so this can order the true split against other splits but "
            "cannot certify a signal. No parameter was updated."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                "taiji/goal_conditioned_training.py",
                "scripts/training/diagnose_residual_structure_10_7c.py",
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
                "explained_fraction": stats.get("explained_fraction"),
                "expected": stats.get("expected_fraction_under_random_split"),
                "within_group_cosine": stats.get("within_group_cosine"),
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
