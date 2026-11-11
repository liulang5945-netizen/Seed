"""10.7b gradient alignment: do the six training samples pull the same way?

10.7 showed the trained arms answer like a constant preference. The behavioural
fact is not in doubt; this diagnostic asks the first-order question underneath it:

* read the per-sample gradient of the same loss on the six training samples;
* how aligned are they with each other (pairwise cosine, residual fraction)?
* do the two *correct-object* groups push the same direction or opposite ones?
* where does the saved lr=1e-4 update sit relative to the mean gradient?

If the two groups push the same way, no averaging update can separate them at
first order -- which is what "learned a constant" would look like from below.

It trains nothing: gradients are read, not applied, and the update comes from the
weights 10.7 already saved.
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
TOGETHER_FLOOR = 0.9
OPPOSE_CEILING = -0.9
TRAINING_CARD = "reports/taiji_core_goal_training_10_7_20261101.json"


def cosine(first: torch.Tensor, second: torch.Tensor):
    first = first.double()
    second = second.double()
    norm = float(first.norm()) * float(second.norm())
    if norm == 0.0:
        return None
    return float(torch.dot(first, second)) / norm


def classify(cos_groups) -> str:
    if cos_groups is None:
        return "no_alignment"
    if cos_groups >= TOGETHER_FLOOR:
        return "groups_pull_together"
    if cos_groups <= OPPOSE_CEILING:
        return "groups_oppose"
    return "groups_mixed"


def flat_parameters(parameters) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("10.7b diagnostic card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()
    card = json.loads((ROOT / TRAINING_CARD).read_text(encoding="utf-8"))
    expected_identity = card.get("parent_model_digest")
    fast = [run for run in card.get("runs") or [] if abs(float(run.get("lr")) - 1e-4) <= 1e-12][0]

    parent, subject, provider = prior.build_subject()
    digest = prior.parent_digest(parent)
    parameters = [parameter for parameter in subject.parameters() if parameter.requires_grad]
    initial = flat_parameters(parameters)

    rows = samples()
    per_state = len(rows) // 6
    train_rows = rows[: TRAIN_STATES * per_state]

    per_sample, failure = [], None
    try:
        for sample in train_rows:
            subject.zero_grad(set_to_none=True)
            loss = sample_loss(subject, sample, provider)
            gradient = gradient_vector(loss, parameters)
            per_sample.append(
                {
                    "object": sample["correct"][0],
                    "loss": float(loss.detach()),
                    "gradient": gradient,
                }
            )
        subject.zero_grad(set_to_none=True)
    except Exception as error:  # noqa: BLE001 - the exit needs the message
        failure = f"{type(error).__name__}: {error}"

    alignment = {}
    cos_groups = None
    mean_gradient = None
    if not failure:
        gradients = [row["gradient"] for row in per_sample]
        mean_gradient = torch.stack(gradients).mean(dim=0)
        norms = [float(gradient.norm()) for gradient in gradients]
        pairwise = [
            cosine(gradients[first], gradients[second])
            for first, second in itertools.combinations(range(len(gradients)), 2)
        ]
        pairwise = [value for value in pairwise if value is not None]
        groups = {}
        for row in per_sample:
            groups.setdefault(row["object"], []).append(row["gradient"])
        group_means = {name: torch.stack(values).mean(dim=0) for name, values in groups.items()}
        names = list(group_means)
        cos_groups = cosine(group_means[names[0]], group_means[names[1]])
        mean_norm = float(mean_gradient.norm())
        alignment = {
            "group_sizes": {name: len(values) for name, values in groups.items()},
            "group_names": names,
            "gradient_norms": norms,
            "mean_norm": mean_norm,
            "mean_over_average_norm": (mean_norm / statistics.fmean(norms) if norms else None),
            "pairwise_cosine_min": min(pairwise) if pairwise else None,
            "pairwise_cosine_median": (statistics.median(pairwise) if pairwise else None),
            "pairwise_cosine_max": max(pairwise) if pairwise else None,
            "residual_fraction": (
                statistics.fmean(
                    float((gradient - mean_gradient).norm() ** 2) for gradient in gradients
                )
                / statistics.fmean([norm**2 for norm in norms])
                if norms and statistics.fmean([norm**2 for norm in norms]) > 0
                else None
            ),
            "cos_group_means": cos_groups,
        }

    payload = torch.load(
        ROOT / str(fast.get("weights_path")), map_location="cpu", weights_only=True
    )
    subject.load_state_dict(payload["state_dict"], strict=True)
    delta = flat_parameters(parameters) - initial
    cos_delta_mean = cosine(delta, mean_gradient) if mean_gradient is not None else None
    verdict = classify(cos_groups)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif failure is not None:
        status = "engineering_failure"
    else:
        status = f"training_{verdict}"

    report = {
        "schema": "taiji-training-direction-10-7b-v1",
        "step": "10.7b",
        "date": "2026-11-01",
        "status": status,
        "declared": {
            "samples": len(train_rows),
            "grouping": "correct object (3 vs 3)",
            "loss": "taiji.goal_conditioned_training.sample_loss",
            "delta": f"lr=1e-4 weights ({fast.get('weights_path')}) minus rebuilt subject",
            "together_floor": TOGETHER_FLOOR,
            "oppose_ceiling": OPPOSE_CEILING,
        },
        "parent_model_digest": expected_identity,
        "rebuilt_digest": digest,
        "identity_match": digest == expected_identity,
        "alignment": alignment,
        "delta_norm": float(delta.norm()),
        "cos_delta_mean_gradient": cos_delta_mean,
        "verdict": verdict,
        "per_sample_losses": [row["loss"] for row in per_sample],
        "failure": failure,
        "model_involved": True,
        "capability_claimed": False,
        "training_admitted": False,
        "limits": (
            "Diagnostic only: six samples, one deterministic subject, one saved update. "
            "First-order alignment says what these six gradients agree on, not what a "
            "different loss, more samples or another architecture would do; no parameter "
            "was updated and no capability is claimed."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in (
                TRAINING_CARD,
                "taiji/goal_conditioned_training.py",
                "scripts/training/diagnose_training_direction_10_7b.py",
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
                "identity_match": report["identity_match"],
                "cos_group_means": cos_groups,
                "mean_over_average": alignment.get("mean_over_average_norm"),
                "pairwise_median": alignment.get("pairwise_cosine_median"),
                "residual_fraction": alignment.get("residual_fraction"),
                "cos_delta_mean": cos_delta_mean,
                "delta_norm": report["delta_norm"],
                "cpu_seconds": round(cpu_seconds, 3),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
