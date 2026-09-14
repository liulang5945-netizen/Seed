"""M2.R2.R0.1 context, ordering and simple-reference ablations.

This evaluator keeps the Taiji structure fixed.  It asks whether the small
cross-seed predictive-context signal survives three controlled changes:

* normal order versus a deterministic byte-order shuffle;
* context-only learning versus protected-readout-only learning;
* a post-training lesion of the learned temporal residual.

Every model point uses the same record-disjoint phase-C train/holdout chain,
owner attribution, read-only score checks and checkpoint round trip as R2.R0.
The byte unigram and additive n-gram references are diagnostic baselines, not
provider output and not promotion criteria.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m2r1_phase_c_canary import (  # noqa: E402
    _authorization,
    _process_working_set_bytes,
    build_disjoint_phase_chain,
    load_joint_child,
)
from scripts.training.eval_taiji_m2r2_learning_curve import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_CORPUS,
    _issue_boundary,
    _owner_digests,
    _preflight_checkpoint,
    _score,
    _validate_source_lineage,
)
from taiji import Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-r2-ablation-v1"
DEFAULT_VARIANTS = ("frozen", "normal_context", "shuffled_context", "readout_only")
ALL_VARIANTS = frozenset(DEFAULT_VARIANTS)
SHUFFLE_SEED_OFFSET = 7001


def shuffled_bytes(data: bytes, *, seed: int) -> bytes:
    """Return a deterministic permutation with exactly the same byte counts."""

    shuffled = bytearray(data)
    random.Random(int(seed)).shuffle(shuffled)
    return bytes(shuffled)


def _ngram_bpb(
    train: bytes,
    holdout: bytes,
    *,
    order: int,
    alphabet_size: int = 257,
    boundary_symbol: int = 256,
    alpha: float = 1.0,
) -> float:
    """Score an additive byte n-gram model on the same boundary convention."""

    if order < 0:
        raise ValueError("order must be non-negative")
    if alphabet_size <= 1 or not 0 <= boundary_symbol < alphabet_size:
        raise ValueError("invalid reference alphabet")
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")

    if order == 0:
        counts = [alpha] * int(alphabet_size)
        for symbol in (boundary_symbol, *train):
            counts[int(symbol)] += 1.0
        total = sum(counts)
        targets = (*holdout, boundary_symbol)
        return sum(-math.log2(counts[int(symbol)] / total) for symbol in targets) / len(targets)

    contexts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
    sequence = (boundary_symbol,) * int(order) + tuple(train) + (boundary_symbol,)
    for index in range(order, len(sequence)):
        context = tuple(int(value) for value in sequence[index - order : index])
        contexts[context][int(sequence[index])] += 1

    holdout_sequence = (boundary_symbol,) * int(order) + tuple(holdout) + (boundary_symbol,)
    losses: list[float] = []
    for index in range(order, len(holdout_sequence)):
        context = tuple(int(value) for value in holdout_sequence[index - order : index])
        target = int(holdout_sequence[index])
        counts = contexts.get(context)
        observed = 0 if counts is None else int(sum(counts.values()))
        count = 0 if counts is None else int(counts.get(target, 0))
        probability = (count + alpha) / (observed + alpha * float(alphabet_size))
        losses.append(-math.log2(probability))
    return sum(losses) / max(1, len(losses))


def reference_scores(train: bytes, holdout: bytes, model: Taiji) -> dict[str, float]:
    """Return fixed, auditable byte baselines for one budget point."""

    return {
        "unigram_bpb": float(
            _ngram_bpb(
                train,
                holdout,
                order=0,
                alphabet_size=model.config.alphabet_size,
                boundary_symbol=model.config.boundary_symbol,
            )
        ),
        "ngram2_bpb": float(
            _ngram_bpb(
                train,
                holdout,
                order=2,
                alphabet_size=model.config.alphabet_size,
                boundary_symbol=model.config.boundary_symbol,
            )
        ),
        "ngram3_bpb": float(
            _ngram_bpb(
                train,
                holdout,
                order=3,
                alphabet_size=model.config.alphabet_size,
                boundary_symbol=model.config.boundary_symbol,
            )
        ),
    }


def _training_kwargs(variant: str) -> dict[str, bool]:
    return {
        "learn_fabric": False,
        "learn_predictive_context": variant in {"normal_context", "shuffled_context"},
        "learn_predictive_readout": variant == "readout_only",
    }


def _lesion_temporal_residual(
    model: Taiji,
    holdout: bytes,
    *,
    seed: int,
    budget: int,
) -> dict[str, Any]:
    """Score a clone with only the learned private temporal residual removed."""

    lesioned = Taiji.from_checkpoint(model.checkpoint())
    recurrent_before = content_digest(lesioned.predictive_context.recurrent.to_payload())
    with torch.no_grad():
        lesioned.predictive_context.recurrent.edge_weight.zero_()
    recurrent_after = content_digest(lesioned.predictive_context.recurrent.to_payload())
    before_score = content_digest(lesioned.checkpoint())
    boundary = _issue_boundary(
        arm="temporal_lesion",
        scope="protected",
        seed=seed,
        budget=budget,
    )
    score = _score(
        lesioned,
        holdout,
        boundary=boundary,
        authorization=_authorization(boundary),
    )
    after_score = content_digest(lesioned.checkpoint())
    return {
        "score": score,
        "lesion_applied": recurrent_before != recurrent_after,
        "recurrent_digest_before": recurrent_before,
        "recurrent_digest_after": recurrent_after,
        "clone_digest_before_score": before_score,
        "clone_digest_after_score": after_score,
        "score_read_only_after_lesion": before_score == after_score,
    }


def _run_point(
    *,
    source_payload: Mapping[str, Any],
    train_stream: bytes,
    holdout: bytes,
    variant: str,
    budget: int,
    seed: int,
    baseline_holdout_bpb: float,
) -> dict[str, Any]:
    model = Taiji.from_checkpoint(source_payload)
    before = _owner_digests(model)
    train_data = train_stream[:budget]
    started = time.perf_counter()
    train_metrics: dict[str, float] | None = None
    if variant != "frozen":
        train_metrics = model.learn_bytes(
            train_data,
            epochs=1,
            include_start_boundary=True,
            include_end_boundary=True,
            reset=True,
            use_memory=False,
            **_training_kwargs(variant),
        )
    train_seconds = time.perf_counter() - started
    after = _owner_digests(model)
    attribution = _owner_attribution_for_variant(variant, before, after)
    boundary = _issue_boundary(
        arm=variant,
        scope="protected",
        seed=seed,
        budget=budget,
    )
    authorization = _authorization(boundary)
    train_score = _score(
        model,
        train_data,
        boundary=boundary,
        authorization=authorization,
    )
    holdout_score = _score(
        model,
        holdout,
        boundary=boundary,
        authorization=authorization,
    )
    lesion = None
    if variant in {"normal_context", "shuffled_context"}:
        lesion = _lesion_temporal_residual(
            model,
            holdout,
            seed=seed,
            budget=budget,
        )
    payload = model.checkpoint()
    checkpoint_digest = content_digest(payload)
    restored = Taiji.from_checkpoint(payload)
    restored_digest = content_digest(restored.checkpoint())
    temporary = PROJECT_ROOT / "output" / "taiji-m2r2-ablation-point.pt"
    torch.save(payload, temporary)
    checkpoint_bytes = temporary.stat().st_size
    temporary.unlink(missing_ok=True)
    score_checks = [bool(train_score["read_only"]), bool(holdout_score["read_only"])]
    if lesion is not None:
        score_checks.append(bool(lesion["score"]["read_only"]))
    checks = {
        "owner_contract": bool(attribution["contract_passed"]),
        "scores_are_read_only": all(score_checks),
        "checkpoint_round_trip": checkpoint_digest == restored_digest,
        "temporal_lesion_is_effective": lesion is None or bool(lesion["lesion_applied"]),
        "temporal_lesion_score_is_read_only": lesion is None or bool(lesion["score"]["read_only"]),
    }
    current_working_set = _process_working_set_bytes()
    return {
        "variant": variant,
        "arm": variant_to_arm(variant),
        "budget_bytes": int(budget),
        "train_stream_digest": content_digest(train_stream),
        "train": {
            "metrics": train_metrics,
            "seconds": float(train_seconds),
            "working_set_bytes": current_working_set,
        },
        "train_score": train_score,
        "holdout_score": holdout_score,
        "holdout_gain_bpb": float(baseline_holdout_bpb - holdout_score["bpb"]),
        "references": reference_scores(train_data, holdout, model),
        "temporal_lesion": lesion,
        "owners": {
            "before": before,
            "after": after,
            "attribution": attribution,
        },
        "checkpoint": {
            "digest": checkpoint_digest,
            "restored_digest": restored_digest,
            "bytes": int(checkpoint_bytes),
        },
        "checks": checks,
    }


def variant_to_arm(variant: str) -> str:
    """Map evaluator variants to the stable R2 owner attribution contract."""

    if variant in {"normal_context", "shuffled_context"}:
        return "predictive_context"
    if variant == "readout_only":
        return "protected_readout"
    if variant == "frozen":
        return "frozen"
    raise ValueError(f"unsupported ablation variant: {variant}")


def _owner_attribution_for_variant(
    variant: str,
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    write_sets = {
        "frozen": set(),
        "predictive_context": {"predictive_context"},
        "protected_readout": {"protected_predictive_readout"},
    }
    arm = variant_to_arm(variant)
    if arm not in write_sets:
        raise ValueError(f"unsupported attribution arm: {arm}")
    changed = {
        owner: str(before.get(owner, "")) != str(after.get(owner, ""))
        for owner in sorted(set(before) | set(after))
    }
    expected = write_sets[arm]
    unexpected = sorted(
        owner for owner, value in changed.items() if value and owner not in expected
    )
    expected_changed = sorted(owner for owner in expected if changed.get(owner, False))
    return {
        "changed": changed,
        "expected_writable_owners": sorted(expected),
        "changed_expected_owners": expected_changed,
        "unexpected_changed_owners": unexpected,
        "contract_passed": not unexpected,
        "required_owner_changed": bool(expected) and bool(expected_changed),
    }


def run_ablation(
    checkpoint: Path,
    *,
    corpus_paths: Sequence[Path],
    seed: int,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    eval_bytes: int = 4096,
    variants: Sequence[str] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    normalized_budgets = tuple(sorted(dict.fromkeys(int(value) for value in budgets)))
    normalized_variants = tuple(dict.fromkeys(str(value) for value in variants))
    if not normalized_budgets or any(value <= 0 for value in normalized_budgets):
        raise ValueError("budgets must contain positive values")
    if not normalized_variants or any(value not in ALL_VARIANTS for value in normalized_variants):
        raise ValueError(f"variants must be drawn from {sorted(ALL_VARIANTS)}")

    payload, source_model = load_joint_child(checkpoint, expected_seed=seed)
    chain = build_disjoint_phase_chain(corpus_paths, cohort_seeds=(11, 29, 47))
    phase_a = chain.phase_a_by_seed[int(seed)]
    phase_b = chain.phase_b_by_seed[int(seed)]
    _validate_source_lineage(
        payload,
        phase_a_digest=phase_a.digest,
        phase_b_digest=phase_b.digest,
    )
    if max(normalized_budgets) > len(chain.phase_c.train):
        raise ValueError("largest budget exceeds phase-C training bytes")
    if eval_bytes <= 0 or eval_bytes > len(chain.phase_c.holdout):
        raise ValueError("eval_bytes must fit the phase-C holdout prefix")

    source_payload = source_model.checkpoint()
    source_digest = content_digest(source_payload)
    preflight = _preflight_checkpoint(
        source_model,
        PROJECT_ROOT / "output" / "taiji-m2r2-ablation-preflight.pt",
    )
    if not preflight["matching_digest"]:
        raise RuntimeError("source checkpoint save/restore preflight failed")

    holdout = chain.phase_c.holdout[:eval_bytes]
    baseline_boundary = _issue_boundary(
        arm="baseline",
        scope="protected",
        seed=seed,
        budget=0,
    )
    baseline_score = _score(
        Taiji.from_checkpoint(source_payload),
        holdout,
        boundary=baseline_boundary,
        authorization=_authorization(baseline_boundary),
    )

    streams: dict[str, bytes] = {"normal": chain.phase_c.train}
    if "shuffled_context" in normalized_variants:
        streams["shuffled"] = shuffled_bytes(
            chain.phase_c.train,
            seed=int(seed) + SHUFFLE_SEED_OFFSET,
        )

    points: list[dict[str, Any]] = []
    for variant in normalized_variants:
        stream = streams["shuffled" if variant == "shuffled_context" else "normal"]
        for budget in normalized_budgets:
            points.append(
                _run_point(
                    source_payload=source_payload,
                    train_stream=stream,
                    holdout=holdout,
                    variant=variant,
                    budget=budget,
                    seed=seed,
                    baseline_holdout_bpb=float(baseline_score["bpb"]),
                )
            )

    all_passed = all(bool(value) for point in points for value in point["checks"].values())
    return {
        "format": FORMAT,
        "version": 1,
        "status": "passed" if all_passed else "failed",
        "can_promote": False,
        "source_checkpoint": str(checkpoint),
        "source_digest": source_digest,
        "seed": int(seed),
        "budgets": list(normalized_budgets),
        "eval_bytes": int(eval_bytes),
        "variants": list(normalized_variants),
        "shuffle_seed": int(seed) + SHUFFLE_SEED_OFFSET,
        "preflight": preflight,
        "datasets": {
            "phase_a": {
                "digest": phase_a.digest,
                "selected_record_count": len(phase_a.selected_record_digests),
            },
            "phase_b": {
                "digest": phase_b.digest,
                "selected_record_count": len(phase_b.selected_record_digests),
            },
            "phase_c": {
                "digest": chain.phase_c.digest,
                "selected_record_count": len(chain.phase_c.selected_record_digests),
            },
            "overlap_counts": dict(chain.overlap_counts),
            "record_disjoint": not any(chain.overlap_counts.values()),
        },
        "baseline": {
            "holdout_score": baseline_score,
            "holdout_bpb": float(baseline_score["bpb"]),
        },
        "points": points,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "output"
        / "taiji-m2s-seed11-identity-generation-20260905"
        / "last.pt",
    )
    parser.add_argument("--corpus", type=Path, nargs="+", default=[DEFAULT_CORPUS])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--eval-bytes", type=int, default=4096)
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(ALL_VARIANTS), default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = run_ablation(
        args.checkpoint,
        corpus_paths=args.corpus,
        seed=args.seed,
        budgets=args.budgets,
        eval_bytes=args.eval_bytes,
        variants=args.variants,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "status": report["status"],
                "points": len(report["points"]),
                "technical_checks_passed": sum(
                    bool(value) for point in report["points"] for value in point["checks"].values()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
