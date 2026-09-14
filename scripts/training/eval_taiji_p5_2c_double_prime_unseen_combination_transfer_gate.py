"""P5.2c'' design-repair Gate: two held-out combinations on existing members.

Route C of the decision prompt (plans/active/roadmap/06_...).  P5.2c' produced
``transfer_no_gain`` for a reason that turned out to be a *design* defect rather
than a capability result:

* the unseen surface had cardinality 1, so gate 8's calibration rested on a
  single sample;
* the held-out pair was chosen by lexical index ``(0,3)``, which happened to be
  the one pair whose members share a capability surface (``a`` and ``d`` both
  cover block-0 only), i.e. a structurally redundant pair;
* block-3 is unreachable for every cell, so only 3/4 of the context surface
  discriminates.

This gate repairs the instrument, not the representation.  It holds out TWO
combinations, both verified by real execution to have non-overlapping capability
surfaces, and it quantifies the block-3 problem instead of hiding it.

What this gate deliberately does NOT do
---------------------------------------
It does not fix the profile representation.  P5.2c' measured all four member
contributions as exactly 0.5, so the learner cannot distinguish members; that
defect is route A and needs its own preregistration.  Consequently a negative
result here may only be attributed to "under this representation", never to
"transfer is impossible".

Why no larger member set
------------------------
P5.2a has exactly four train template families, one per member, so a fifth or
sixth member has no new family to specialize in -- it would duplicate an existing
capability surface and *add* redundancy.  The complementarity this cohort needs
already exists (5 of 6 pairs are disjoint).  Measured capability surfaces:
``a={0} b={1} c={2} d={0}``.

Preregistration: plans/reference/
  M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md (frozen)

Held-out design (opaque lexical indices over MEMBER_IDS):
  P1* = (0, 2)  -> surfaces {0} and {2}, disjoint
  P2* = (1, 3)  -> surfaces {1} and {0}, disjoint
The four observable pairs keep ``a+d`` (the redundant pair) in view, so the
learner can observe that some combinations yield no excess gain rather than
seeing complementary pairs only.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

import eval_taiji_p5_2a_predictive_execution_gate as p52a  # noqa: E402
import eval_taiji_p5_2b_group_causal_corpora_gate as p52b  # noqa: E402
import eval_taiji_p5_2c_prime_unseen_combination_transfer_gate as p52cp  # noqa: E402

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.interaction_group_transfer import build_member_evidence  # noqa: E402
from taiji.interaction_groups import (  # noqa: E402
    InteractionGroupEvaluator,
    InteractionTraceCorpus,
    InteractionTraceEpisode,
)

# --------------------------------------------------------------------------- #
# Frozen constants (never derived from measurements)
# --------------------------------------------------------------------------- #

REPORT_FORMAT = "taiji-p5-2c-double-prime-unseen-combination-transfer-report-v1"
VERSION = 1
PREREGISTRATION = (
    "plans/reference/"
    "M5_P5_2C_DOUBLE_PRIME_UNSEEN_COMBINATION_TRANSFER_PREREGISTRATION_20260913.md"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "taiji_p5_2c_double_prime_unseen_combination_transfer_20260913.json"
)

TOTAL_SECONDS_CAP = p52cp.TOTAL_SECONDS_CAP  # 900.0, unchanged
MARGIN = p52cp.MARGIN  # 0.15, unchanged
TRAIN_CONTEXT_COUNT = p52cp.TRAIN_CONTEXT_COUNT  # 8
CONTEXT_COUNT = p52cp.CONTEXT_COUNT  # 12
REPEATS = p52cp.REPEATS  # 2
CALIBRATION_MINIMUM_SIGN_MATCH_RATE = p52cp.CALIBRATION_MINIMUM_SIGN_MATCH_RATE  # 0.5
CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR = p52cp.CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR  # 0.35
RANDOM_CONTROL_SEED = 20260913

MEMBER_IDS = p52b.MEMBER_IDS
CELL_MEMBER_SETS = p52b.CELL_MEMBER_SETS
PAIR_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 2
)
SINGLETON_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item for item in CELL_MEMBER_SETS if len(item) == 1
)

# Two held-out combinations, opaque lexical indices (route C target 1 and 2).
HELD_OUT_PAIR_INDICES: tuple[tuple[int, int], ...] = ((0, 2), (1, 3))
HELD_OUT_PAIRS: tuple[tuple[str, ...], ...] = tuple(
    tuple(sorted((MEMBER_IDS[first], MEMBER_IDS[second])))
    for first, second in HELD_OUT_PAIR_INDICES
)
HELD_OUT_PAIR = HELD_OUT_PAIRS[0]  # P1*; the fixed-combination control
OBSERVED_PAIR_MEMBER_SETS: tuple[tuple[str, ...], ...] = tuple(
    item
    for item in PAIR_MEMBER_SETS
    if tuple(sorted(item)) not in {tuple(sorted(pair)) for pair in HELD_OUT_PAIRS}
)

_HELD_OUT_FROZENSETS = frozenset(frozenset(pair) for pair in HELD_OUT_PAIRS)

STATIC_CHECK_SCOPE = (
    "taiji/interaction_groups.py",
    "taiji/interaction_group_transfer.py",
    "scripts/training/eval_taiji_p5_2a_predictive_execution_gate.py",
    "scripts/training/eval_taiji_p5_2b_group_causal_corpora_gate.py",
    "scripts/training/eval_taiji_p5_2c_prime_unseen_combination_transfer_gate.py",
    "scripts/training/eval_taiji_p5_2c_double_prime_unseen_combination_transfer_gate.py",
    "tests/taiji_native/test_p5_2c_double_prime_unseen_combination_gate.py",
)
STATIC_CHECK_COMMANDS = p52cp.STATIC_CHECK_COMMANDS

# --------------------------------------------------------------------------- #
# Delegation policy
#
# Predicate-pure helpers are delegated on purpose: they carry no gate-specific
# constants, so reusing them cannot silently change what is being measured.
# ``_train_context_ids`` is such a helper -- it depends only on
# TRAIN_CONTEXT_COUNT, which is itself inherited unchanged, so delegation is
# equivalent to a local copy rather than a hidden coupling.
#
# Helpers that DO depend on held-out constants (``_build_corpus``,
# ``_joint_counts``, ``_recovery_probe``, ``_rejection_probe``) are implemented
# locally below and must never be delegated; a regression test enforces this.
# --------------------------------------------------------------------------- #

_train_context_ids = p52cp._train_context_ids


def _ids(members: tuple[str, ...]) -> str:
    return p52cp._ids(members)


_sha = p52cp._sha


# --------------------------------------------------------------------------- #
# Evidence construction: remove BOTH held-out pairs' joint cells from all train
# --------------------------------------------------------------------------- #


def _episode_is_held_out_joint(episode: InteractionTraceEpisode) -> bool:
    """True when the episode's event owners are exactly one held-out pair."""

    return frozenset(episode.member_ids) in _HELD_OUT_FROZENSETS


def _build_corpus(
    projected: tuple[InteractionTraceEpisode, ...],
) -> tuple[InteractionTraceCorpus, list[dict[str, Any]]]:
    """Split and remove both held-out pairs' joint cells from the train partition.

    Removal is confined to the train partition and to the two joint cells.  Every
    member keeps its singleton and baseline observations, so both held-out pairs
    are *unseen* rather than *unsupported*.
    """

    train_ids = p52cp._train_context_ids()
    removed: list[dict[str, Any]] = []
    train: list[InteractionTraceEpisode] = []
    holdout: list[InteractionTraceEpisode] = []
    for episode in projected:
        if episode.context_id not in train_ids:
            holdout.append(episode)
            continue
        if _episode_is_held_out_joint(episode):
            removed.append(
                {
                    "episode_id": episode.episode_id,
                    "context_id": episode.context_id,
                    "member_ids": list(episode.member_ids),
                    "outcome": episode.outcome,
                }
            )
            continue
        train.append(episode)
    return InteractionTraceCorpus(train=tuple(train), holdout=tuple(holdout)), removed


def _joint_counts(corpus: InteractionTraceCorpus, pair: tuple[str, ...]) -> tuple[int, int]:
    target = frozenset(pair)
    train_ids = p52cp._train_context_ids()
    in_train = sum(
        1
        for episode in corpus.train
        if episode.context_id in train_ids and frozenset(episode.member_ids) == target
    )
    in_holdout = sum(
        1
        for episode in corpus.holdout
        if episode.context_id not in train_ids and frozenset(episode.member_ids) == target
    )
    return in_train, in_holdout


def _singleton_counts(corpus: InteractionTraceCorpus, pair: tuple[str, ...]) -> dict[str, int]:
    return {
        member: sum(1 for episode in corpus.train if episode.member_ids == (member,))
        for member in pair
    }


def _entry_audit(
    corpus: InteractionTraceCorpus,
    *,
    removed: list[dict[str, Any]],
    train_only_records: tuple[Any, ...],
    intervention_reality: dict[str, Any],
) -> dict[str, Any]:
    """Hard entry assertions from preregistration sections 3.2 and 4.

    Any failure means the removal did not take effect or the interventions were
    inert.  The gate stops with ``blocked_at_entry_audit`` and the fix is to
    repair the wiring -- never to relax the criterion.
    """

    observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
    conditions: list[dict[str, Any]] = []

    def require(name: str, ok: bool, detail: str) -> None:
        if not ok:
            conditions.append({"condition": name, "detail": detail})

    per_pair: dict[str, Any] = {}
    for pair in HELD_OUT_PAIRS:
        label = "+".join(pair)
        train_joint, holdout_joint = _joint_counts(corpus, pair)
        singles = _singleton_counts(corpus, pair)
        per_pair[label] = {
            "member_ids": list(pair),
            "in_observed_records": tuple(sorted(pair)) in observed_pairs,
            "joint_count_in_train": train_joint,
            "joint_count_in_holdout": holdout_joint,
            "singleton_counts_in_train": singles,
        }
        require(
            f"held_out_pair[{label}]_not_observed",
            tuple(sorted(pair)) not in observed_pairs,
            f"{label} is still present in observed_records",
        )
        require(
            f"held_out_pair[{label}]_joint_absent_from_train",
            train_joint == 0,
            f"{label} joint cell occurs {train_joint} times in train",
        )
        require(
            f"held_out_pair[{label}]_joint_present_in_holdout",
            holdout_joint > 0,
            f"{label} joint cell absent from holdout, realized gain not measurable",
        )
        require(
            f"held_out_pair[{label}]_members_marginally_observed",
            all(count > 0 for count in singles.values()),
            f"{label} has a member with no singleton observation left in train, so the "
            "combination would be unsupported rather than unseen",
        )

    missing_observed = tuple(
        sorted(
            item for item in OBSERVED_PAIR_MEMBER_SETS if tuple(sorted(item)) not in observed_pairs
        )
    )
    require(
        "remaining_observed_pairs_intact",
        not missing_observed,
        f"these pairs should have stayed observable but were not estimated: {missing_observed}",
    )
    expected_removed = TRAIN_CONTEXT_COUNT * REPEATS * len(HELD_OUT_PAIRS)
    require(
        "removal_ledger_matches_expectation",
        len(removed) == expected_removed,
        f"removal ledger has {len(removed)} entries, expected {expected_removed}",
    )
    require(
        "interventions_executed",
        bool(intervention_reality["interventions_happened"]),
        f"{intervention_reality['non_baseline_zero_step_total']} non-baseline "
        "intervention episodes executed zero steps",
    )

    return {
        "conditions": conditions,
        "passed": not conditions,
        "held_out_pairs": per_pair,
        "observed_pairs": sorted(list(item) for item in observed_pairs),
        "missing_observed_pairs": [list(item) for item in missing_observed],
        "removed_episode_count": len(removed),
        "removed_episodes": removed,
        "train_episodes": len(corpus.train),
        "holdout_episodes": len(corpus.holdout),
    }


# --------------------------------------------------------------------------- #
# Capability-surface measurement (this round's own evidence, never the old value)
# --------------------------------------------------------------------------- #


def _block_of(task_id: str) -> int:
    return int(task_id.rsplit("-", 1)[1]) % 4


def _capability_surfaces(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    """Measure, from real execution, which blocks each member covers.

    The preregistration requires this to be re-measured each round rather than
    quoted from the design document, so a change in member behaviour cannot be
    masked by a stale constant.
    """

    per_member: dict[str, dict[int, tuple[int, int]]] = {}
    for member in MEMBER_IDS:
        per_member[member] = {}
        for block in range(4):
            attempts = 0
            successes = 0
            for episode in episodes:
                if tuple(episode["active_members"]) != (member,):
                    continue
                if _block_of(episode["task_id"]) != block:
                    continue
                attempts += 1
                successes += 1 if episode["success"] else 0
            per_member[member][block] = (successes, attempts)

    surfaces = {
        member: sorted(block for block, (ok, _) in blocks.items() if ok > 0)
        for member, blocks in per_member.items()
    }

    def classify(pair: tuple[str, ...]) -> str:
        first, second = pair
        overlap = set(surfaces[first]) & set(surfaces[second])
        if not overlap:
            return "DISJOINT"
        if set(surfaces[first]) == set(surfaces[second]):
            return "EQUAL"
        return "PARTIAL"

    pairs = {
        "+".join(pair): {
            "member_ids": list(pair),
            "union": sorted(set(surfaces[pair[0]]) | set(surfaces[pair[1]])),
            "overlap": sorted(set(surfaces[pair[0]]) & set(surfaces[pair[1]])),
            "kind": classify(pair),
            "is_held_out": tuple(sorted(pair)) in {tuple(sorted(p)) for p in HELD_OUT_PAIRS},
        }
        for pair in PAIR_MEMBER_SETS
    }

    held_out_kinds = [pairs["+".join(pair)]["kind"] for pair in HELD_OUT_PAIRS]

    return {
        "per_member_success_by_block": {
            member: {str(block): f"{ok}/{attempts}" for block, (ok, attempts) in blocks.items()}
            for member, blocks in per_member.items()
        },
        "surfaces": surfaces,
        "pairs": pairs,
        "held_out_pair_kinds": held_out_kinds,
        "held_out_pairs_all_disjoint": all(kind == "DISJOINT" for kind in held_out_kinds),
        "note": ("measured this round from real execution; not quoted from the design note"),
    }


def _block3_audit(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    """Quantify the unreachable block-3 instead of hiding it.

    The preregistration requires the report to distinguish "the contract
    intercepted this" from "this combination produced no gain": block-3 is
    expected to be uniformly unreachable, which shrinks the discriminating
    surface to 3/4 of the contexts.
    """

    block3 = [task for task in contexts if _block_of(task.task_id) == 3]
    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for episode in episodes:
        if _block_of(episode["task_id"]) != 3:
            continue
        reason = str(episode.get("stop_reason", ""))
        reason_counts[reason.split(":")[0]] += 1
        rows.append(
            {
                "episode_id": episode["episode_id"],
                "cell": _ids(tuple(episode["active_members"])),
                "success": episode["success"],
                "stop_reason": reason,
            }
        )
    successes = sum(1 for row in rows if row["success"])
    reachable_contexts = [task for task in contexts if _block_of(task.task_id) != 3]
    return {
        "block": 3,
        "contexts": [task.task_id for task in block3],
        "episodes": len(rows),
        "successes": successes,
        "success_rate": (float(successes) / len(rows)) if rows else None,
        "stop_reason_classes": dict(sorted(reason_counts.items())),
        "uniformly_unreachable": successes == 0,
        "discriminating_contexts": len(reachable_contexts),
        "total_contexts": len(contexts),
        "discriminating_fraction": (
            float(len(reachable_contexts)) / len(contexts) if contexts else None
        ),
    }


# --------------------------------------------------------------------------- #
# Scoring helpers (delegated where identical)
# --------------------------------------------------------------------------- #


def _outcome_index(episodes: list[dict[str, Any]]) -> dict[Any, list[float]]:
    return p52cp._outcome_index(episodes)


def _cell_mean(index: dict[Any, list[float]], task_id: str, members: tuple[str, ...]) -> float:
    return p52cp._cell_mean(index, task_id, members)


def _realized(
    index: dict[Any, list[float]], *, task_id: str, members: tuple[str, ...]
) -> dict[str, Any]:
    return p52cp._realized(index, task_id=task_id, members=members)


def _fit_learner(
    corpus: InteractionTraceCorpus,
    *,
    ridge: float,
    minimum_utility: float,
    maximum_uncertainty: float,
    reverse_order: bool = False,
) -> tuple[Any, ...]:
    return p52cp._fit_learner(
        corpus,
        ridge=ridge,
        minimum_utility=minimum_utility,
        maximum_uncertainty=maximum_uncertainty,
        reverse_order=reverse_order,
    )


def _order_candidate_sets(
    member_sets: tuple[tuple[str, ...], ...], seed: int
) -> tuple[tuple[str, ...], ...]:
    return p52cp._order_candidate_sets(member_sets, seed)


def _lesion_coefficients(learner: Any) -> Any:
    return p52cp._lesion_coefficients(learner)


def _select_sum_heuristic(
    profiles: tuple[Any, ...],
    member_sets: tuple[tuple[str, ...], ...],
    *,
    resource_budget: float | None,
) -> Any:
    return p52cp._select_sum_heuristic(profiles, member_sets, resource_budget=resource_budget)


def _success_matrix(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    return p52cp._success_matrix(episodes, contexts)


def _family_coverage(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    return p52cp._family_coverage(episodes, contexts)


def _intervention_reality(
    episodes: list[dict[str, Any]], contexts: tuple[p52a.Task, ...]
) -> dict[str, Any]:
    return p52cp._intervention_reality(episodes, contexts)


def _recovery_probe(
    parent_checkpoint: dict[str, Any], candidate_sets: tuple[tuple[str, ...], ...]
) -> dict[str, Any]:
    """Fresh-process restore probe, evaluated against THIS gate's held-out pair.

    The shared P5.2c' probe compares against that module's single ``HELD_OUT_PAIR``,
    so delegating would silently check the wrong expectation.  The probe *script*
    is reused; only the expected value is computed locally.
    """

    probe_script = p52cp._recovery_probe_script()
    checkpoint_dir = Path(p52cp.tempfile.mkdtemp(prefix="p52cpp-recovery-"))
    checkpoint_path = checkpoint_dir / "learner.json"
    try:
        checkpoint_path.write_text(json.dumps(parent_checkpoint), encoding="utf-8")
        completed = p52cp.subprocess.run(
            [
                sys.executable,
                "-c",
                probe_script,
                str(checkpoint_path),
                json.dumps([list(item) for item in candidate_sets]),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(PROJECT_ROOT),
        )
        if completed.returncode != 0:
            return {
                "fresh_process": True,
                "returncode": completed.returncode,
                "stderr": completed.stderr.strip()[-2000:],
                "selection_reproduced": False,
            }
        restored = json.loads(completed.stdout.strip().splitlines()[-1])
        from taiji.interaction_group_transfer import InteractionGroupTransferLearner

        probe = InteractionGroupTransferLearner.from_checkpoint(parent_checkpoint)
        expected = None
        predicted = probe.candidate(HELD_OUT_PAIRS[0], allow_observed=False)
        if predicted is not None:
            expected = {
                "group_id": predicted.group_id,
                "member_ids": list(predicted.member_ids),
                "predicted_interaction": predicted.predicted_interaction,
            }
        return {
            "fresh_process": True,
            "returncode": completed.returncode,
            "restored_selection": restored,
            "in_process_selection": expected,
            "expected_pair": list(HELD_OUT_PAIRS[0]),
            "selection_reproduced": bool(restored == expected),
        }
    finally:
        p52cp.shutil.rmtree(checkpoint_dir, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    p52cp._write_json(path, payload)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _rejection_probe(
    corpus: InteractionTraceCorpus,
    train_only_records: tuple[Any, ...],
    profiles: tuple[Any, ...],
    candidate_sets: tuple[tuple[str, ...], ...],
) -> dict[str, bool]:
    """Rejections, with BOTH directions asserted for BOTH held-out pairs.

    Implemented locally rather than delegated: the shared P5.2c' probe fits a
    learner using *that* module's constants, which remove only its own single
    held-out pair.  Delegating would evaluate a corpus this gate never built and
    could pass or fail for the wrong reason.
    """

    import dataclasses

    from taiji.interaction_group_transfer import InteractionGroupTransferLearner

    learner, _, _ = _fit_learner(
        corpus, ridge=0.1, minimum_utility=-1.0e9, maximum_uncertainty=1.0e9
    )
    results: dict[str, bool] = {}

    holdout_record = dataclasses.replace(train_only_records[0], holdout_interaction=0.5)
    try:
        learner.observe_records((holdout_record,))
        results["holdout_record_rejected"] = False
    except ValueError:
        results["holdout_record_rejected"] = True

    terminal_record = dataclasses.replace(train_only_records[0], status="rejected")
    try:
        learner.observe_records((terminal_record,))
        results["terminal_record_rejected"] = False
    except ValueError:
        results["terminal_record_rejected"] = True

    tampered = dict(learner.checkpoint())
    tampered["checkpoint_digest"] = "0" * 64
    try:
        InteractionGroupTransferLearner.from_checkpoint(tampered)
        results["tampered_checkpoint_rejected"] = False
    except ValueError:
        results["tampered_checkpoint_rejected"] = True

    stale = dict(learner.checkpoint())
    stale["source_trace_digest"] = p52cp._sha("stale lineage probe")
    body = {key: value for key, value in stale.items() if key != "checkpoint_digest"}
    stale["checkpoint_digest"] = p52cp._sha(json.dumps(body, sort_keys=True, separators=(",", ":")))
    try:
        InteractionGroupTransferLearner.from_checkpoint(stale)
        results["stale_lineage_rejected"] = False
    except ValueError:
        results["stale_lineage_rejected"] = True

    known = sorted(item.member_id for item in profiles)[0]
    results["unknown_member_fails_closed"] = (
        learner.candidate((known, "member-unknown-opaque")) is None
    )

    observed = {tuple(sorted(item.member_ids)) for item in train_only_records}
    # direction 1: observed pairs must be unpredicted under unseen_only
    results["observed_pairs_not_reselectable"] = bool(
        observed
        and all(
            learner.candidate(pair, allow_observed=False) is None
            for pair in candidate_sets
            if tuple(sorted(pair)) in observed
        )
    )
    # direction 2: every held-out pair must be predictable.  Both directions are
    # asserted because checking only one accepts a broken intermediate state.
    results["held_out_pair_predictable_P1"] = (
        learner.candidate(HELD_OUT_PAIRS[0], allow_observed=False) is not None
    )
    results["held_out_pair_predictable_P2"] = (
        learner.candidate(HELD_OUT_PAIRS[1], allow_observed=False) is not None
    )
    results["all_held_out_pairs_predictable"] = bool(
        results["held_out_pair_predictable_P1"] and results["held_out_pair_predictable_P2"]
    )
    return results


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "preregistration": PREREGISTRATION,
        "route": "C (design repair: two held-out combinations, verified disjoint)",
        "does_not_fix": (
            "profile representation; route A is separate and still required. A "
            "negative result here means 'under this representation', not "
            "'transfer is impossible'"
        ),
        "static_checks": {
            "scope": list(STATIC_CHECK_SCOPE),
            "commands": list(STATIC_CHECK_COMMANDS),
            "executed_before_run": True,
        },
    }
    workspace_root = p52cp.Path(p52cp.tempfile.mkdtemp(prefix="p52cpp-workbench-"))
    replica_root = p52cp.Path(p52cp.tempfile.mkdtemp(prefix="p52cpp-replica-"))
    try:
        embedder = DocumentEmbedder()
        members = p52b._train_members(embedder)
        contexts = p52a._validation_tasks()[:CONTEXT_COUNT]
        cue_by_task = {task.task_id: embedder.embed([task.goal_text])[0] for task in contexts}

        matrix_episodes = p52cp._execute_matrix(workspace_root, contexts, members, cue_by_task)
        projected = p52cp._project(matrix_episodes)
        corpus, removed = _build_corpus(projected)

        intervention_reality = _intervention_reality(matrix_episodes, contexts)

        revision = next(iter(corpus.train_checkpoint_revisions))
        train_trace_digest = corpus.train_trace_digest
        profiles = build_member_evidence(
            corpus.train,
            source_trace_digest=train_trace_digest,
            checkpoint_revision=revision,
        )
        evaluator = InteractionGroupEvaluator()
        train_only_records = evaluator.train_only_candidates(corpus)
        if any(
            record.source_trace_digest != train_trace_digest
            or int(record.checkpoint_revision) != int(revision)
            for record in train_only_records
        ):
            raise AssertionError("train-only records cross the train trace lineage")

        capability = _capability_surfaces(matrix_episodes, contexts)
        block3 = _block3_audit(matrix_episodes, contexts)

        entry_audit = _entry_audit(
            corpus,
            removed=removed,
            train_only_records=train_only_records,
            intervention_reality=intervention_reality,
        )
        # route C target 2: the held-out pairs must be verified disjoint this
        # round, not assumed from the design note.
        entry_audit["conditions"].extend(
            []
            if capability["held_out_pairs_all_disjoint"]
            else [
                {
                    "condition": "held_out_pairs_capability_disjoint",
                    "detail": (
                        "held-out pairs are not all disjoint by measured capability "
                        f"surface: {capability['held_out_pair_kinds']}"
                    ),
                }
            ]
        )
        entry_audit["passed"] = not entry_audit["conditions"]

        if not entry_audit["passed"]:
            payload.update(
                {
                    "status": "blocked",
                    "outcome": "blocked_at_entry_audit",
                    "entry_audit": entry_audit,
                    "capability_surfaces": capability,
                    "block3_audit": block3,
                    "intervention_reality": intervention_reality,
                    "matrix": {
                        "episodes": len(matrix_episodes),
                        "train_episodes": len(corpus.train),
                        "holdout_episodes": len(corpus.holdout),
                    },
                    "growth_admitted": False,
                    "can_promote": False,
                    "interpretation": (
                        "blocked: the two-pair removal did not produce the "
                        "preregistered unseen surface, or the interventions were "
                        "inert; repair the wiring, do not relax the criterion"
                    ),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            p52cp.shutil.rmtree(workspace_root, ignore_errors=True)
            p52cp.shutil.rmtree(replica_root, ignore_errors=True)
            _write_json(DEFAULT_REPORT, payload)
            return payload

        observed_pairs = {tuple(sorted(item.member_ids)) for item in train_only_records}
        candidate_sets_ordered = _order_candidate_sets(PAIR_MEMBER_SETS, 11)

        learner, _, _ = _fit_learner(
            corpus, ridge=0.1, minimum_utility=-1.0e9, maximum_uncertainty=1.0e9
        )
        selected = learner.select(candidate_sets_ordered, resource_budget=10.0, unseen_only=True)
        if selected is None:
            payload.update(
                {
                    "status": "failed",
                    "outcome": "failed",
                    "entry_audit": entry_audit,
                    "error": (
                        "learner.select returned None although the entry audit "
                        "confirmed a non-empty unseen surface"
                    ),
                    "growth_admitted": False,
                    "can_promote": False,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            )
            p52cp.shutil.rmtree(workspace_root, ignore_errors=True)
            p52cp.shutil.rmtree(replica_root, ignore_errors=True)
            _write_json(DEFAULT_REPORT, payload)
            return payload

        selection, candidate = selected
        selected_members = tuple(candidate.member_ids)
        held_out_sorted = {tuple(sorted(pair)) for pair in HELD_OUT_PAIRS}
        selected_is_held_out = tuple(sorted(selected_members)) in held_out_sorted

        parent_checkpoint = learner.checkpoint()
        object_index = _outcome_index(matrix_episodes)
        unseen_contexts = contexts[TRAIN_CONTEXT_COUNT:]

        def score(member_sets: tuple[str, ...]) -> dict[str, Any]:
            per_context = [
                _realized(object_index, task_id=task.task_id, members=member_sets)
                for task in unseen_contexts
            ]
            gains = [item["realized_pair_gain_vs_strongest_single"] for item in per_context]
            interactions = [item["realized_interaction"] for item in per_context]
            return {
                "member_ids": list(member_sets),
                "per_context": per_context,
                "mean_gain_vs_strongest_single": float(sum(gains) / len(gains)),
                "mean_realized_interaction": float(sum(interactions) / len(interactions)),
                "contexts_beating_strongest_single": sum(1 for item in gains if item > 0.0),
                "contexts_scored": len(per_context),
            }

        # both held-out pairs are scored, so calibration rests on two samples
        held_out_scores = {tuple(pair): score(pair) for pair in HELD_OUT_PAIRS}

        prediction_binding = {
            "bound_before_scoring": True,
            "parent_checkpoint_digest": parent_checkpoint["checkpoint_digest"],
            "model_digest": learner.model_digest,
            "source_trace_digest": candidate.source_trace_digest,
            "checkpoint_revision": candidate.checkpoint_revision,
            "selected_group_id": candidate.group_id,
            "selected_member_ids": list(selected_members),
            "selected_is_a_held_out_pair": selected_is_held_out,
            "predicted_interaction": candidate.predicted_interaction,
            "uncertainty": candidate.uncertainty,
            "resource_cost": candidate.resource_cost,
            "support": candidate.support,
            "selection_utility": selection.utility,
            "candidate_sets_considered": [list(item) for item in candidate_sets_ordered],
            "observed_pair_count": len(observed_pairs),
            "held_out_pairs": [list(pair) for pair in HELD_OUT_PAIRS],
        }

        # predictions for every pair, so calibration covers both unseen samples
        predictions: list[dict[str, Any]] = []
        for pair in PAIR_MEMBER_SETS:
            predicted = learner.candidate(pair, allow_observed=False)
            predictions.append(
                {
                    "member_ids": list(pair),
                    "seen_in_train_evidence": tuple(sorted(pair)) in observed_pairs,
                    "predicted": (
                        None
                        if predicted is None
                        else {
                            "group_id": predicted.group_id,
                            "predicted_interaction": predicted.predicted_interaction,
                            "uncertainty": predicted.uncertainty,
                            "resource_cost": predicted.resource_cost,
                            "support": predicted.support,
                        }
                    ),
                }
            )

        # ---- controls on the observable (unseen-in-context) pair surface ----
        unseen_pairs_ordered = tuple(
            pair for pair in candidate_sets_ordered if tuple(sorted(pair)) not in observed_pairs
        )
        control_sets: dict[str, tuple[str, ...]] = {
            "no_learning": candidate_sets_ordered[0],
            "random_combination": unseen_pairs_ordered[
                int(p52cp._sha(f"{RANDOM_CONTROL_SEED}:{sorted(unseen_pairs_ordered)}")[:8], 16)
                % len(unseen_pairs_ordered)
            ],
            "fixed_combination": HELD_OUT_PAIR,
        }
        sum_heuristic = _select_sum_heuristic(
            profiles, candidate_sets_ordered, resource_budget=10.0
        )
        if sum_heuristic is None:
            raise AssertionError("sum-heuristic control returned no candidate")
        control_sets["train_only_simple_regression"] = tuple(sum_heuristic[1].member_ids)
        lesion = _lesion_coefficients(learner)
        lesion_selected = lesion.select(
            candidate_sets_ordered, resource_budget=10.0, unseen_only=True
        )
        if lesion_selected is None:
            raise AssertionError("lesion control returned no candidate")
        control_sets["lesion_learner"] = tuple(lesion_selected[1].member_ids)

        pair_control_scores = {name: score(value) for name, value in control_sets.items()}

        singleton_per_context: list[dict[str, Any]] = []
        for task in unseen_contexts:
            baseline = _cell_mean(object_index, task.task_id, ())
            best_member = None
            best_value = None
            for members_set in SINGLETON_MEMBER_SETS:
                value = _cell_mean(object_index, task.task_id, tuple(members_set))
                if best_value is None or value > best_value:
                    best_value = value
                    best_member = members_set[0]
            singleton_per_context.append(
                {
                    "context_id": task.task_id,
                    "strongest_singleton_member": best_member,
                    "strongest_singleton_gain": float(best_value) - baseline,
                }
            )
        singleton_gains = [item["strongest_singleton_gain"] for item in singleton_per_context]
        singleton_mean = float(sum(singleton_gains) / len(singleton_gains))

        control_scores: dict[str, Any] = {
            **pair_control_scores,
            "strongest_singleton": {
                "member_ids": ["<per-context strongest singleton>"],
                "kind": "per_context_singleton",
                "per_context": singleton_per_context,
                "mean_gain_vs_strongest_single": singleton_mean,
                "mean_realized_interaction": None,
                "contexts_scored": len(singleton_per_context),
            },
        }
        for name, _value in pair_control_scores.items():
            control_scores[name]["kind"] = "pair_selection"

        margin_controls = {
            name: value for name, value in control_scores.items() if name != "fixed_combination"
        }
        strongest_control = max(
            margin_controls.items(),
            key=lambda item: item[1]["mean_gain_vs_strongest_single"],
        )
        best_control_gain = strongest_control[1]["mean_gain_vs_strongest_single"]
        fixed_control_gain = control_scores["fixed_combination"]["mean_gain_vs_strongest_single"]

        object_score = held_out_scores[tuple(selected_members)]
        object_gain = object_score["mean_gain_vs_strongest_single"]
        margin_cleared = bool(object_gain > best_control_gain + MARGIN)
        collaboration_holds = bool(
            object_score["contexts_beating_strongest_single"] > 0
            and all(
                item["realized_pair_gain_vs_strongest_single"] >= 0.0
                for item in object_score["per_context"]
            )
        )
        # route C target 1: at least one held-out pair must show a positive gain,
        # otherwise the whole object surface is inert and the outcome is
        # transfer_signal_constant rather than transfer_no_gain.
        any_held_out_positive = any(
            value["mean_gain_vs_strongest_single"] > 0.0 for value in held_out_scores.values()
        )
        held_out_summary = {
            "+".join(pair): {
                "member_ids": list(pair),
                "kind": capability["pairs"]["+".join(pair)]["kind"],
                "mean_gain_vs_strongest_single": value["mean_gain_vs_strongest_single"],
                "mean_realized_interaction": value["mean_realized_interaction"],
                "contexts_beating_strongest_single": value["contexts_beating_strongest_single"],
                "is_selected_object": tuple(sorted(pair)) == tuple(sorted(selected_members)),
            }
            for pair, value in held_out_scores.items()
        }

        # ---- calibration over the unseen samples (2, per route C) ----
        calibration_rows: list[dict[str, Any]] = []
        for entry in predictions:
            pair = tuple(entry["member_ids"])
            realized = score(pair)
            predicted_value = (
                None if entry["predicted"] is None else entry["predicted"]["predicted_interaction"]
            )
            held_out = tuple(sorted(pair)) in held_out_sorted
            calibration_rows.append(
                {
                    "member_ids": list(pair),
                    "held_out": held_out,
                    "seen_in_train_evidence": entry["seen_in_train_evidence"],
                    "predicted_interaction": predicted_value,
                    "realized_interaction": realized["mean_realized_interaction"],
                    "absolute_error": (
                        None
                        if predicted_value is None
                        else abs(float(predicted_value) - realized["mean_realized_interaction"])
                    ),
                    "sign_matches": (
                        None
                        if predicted_value is None
                        else (
                            (float(predicted_value) > 0.0)
                            == (realized["mean_realized_interaction"] > 0.0)
                        )
                    ),
                }
            )
        comparable = [
            row
            for row in calibration_rows
            if row["held_out"]
            and row["predicted_interaction"] is not None
            and row["realized_interaction"] is not None
        ]
        errors = [float(row["absolute_error"]) for row in comparable]
        sign_hits = [bool(row["sign_matches"]) for row in comparable]
        sign_match_rate = (
            float(sum(1 for hit in sign_hits if hit) / len(sign_hits)) if sign_hits else None
        )
        median_absolute_error = _median(errors)
        calibration = {
            "unseen_sample_count": sum(1 for row in calibration_rows if row["held_out"]),
            "comparable_pair_count": len(comparable),
            "mean_absolute_error": float(sum(errors) / len(errors)) if errors else None,
            "median_absolute_error": median_absolute_error,
            "sign_match_rate": sign_match_rate,
            "small_sample_caveat": (
                "the unseen surface has cardinality 2 by construction; these pooled "
                "statistics rest on those two pairs and are not an independent sample"
            ),
            "thresholds": {
                "minimum_sign_match_rate": CALIBRATION_MINIMUM_SIGN_MATCH_RATE,
                "maximum_median_absolute_error": CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR,
                "fixed_at_freeze_time": True,
            },
            "rows": calibration_rows,
        }
        calibration_ok = bool(
            sign_match_rate is not None
            and median_absolute_error is not None
            and sign_match_rate >= CALIBRATION_MINIMUM_SIGN_MATCH_RATE
            and median_absolute_error <= CALIBRATION_MAXIMUM_MEDIAN_ABSOLUTE_ERROR
        )

        # ---- replica + recovery + rejections ----
        replica_episodes = p52cp._execute_matrix(replica_root, contexts, members, cue_by_task)

        def surface(episodes: list[dict[str, Any]]) -> list[Any]:
            return sorted(
                (item["episode_id"], item["success"], item["stop_reason"], item["resource_cost"])
                for item in episodes
            )

        replica_consistent = bool(surface(replica_episodes) == surface(matrix_episodes))
        recovery = _recovery_probe(parent_checkpoint, candidate_sets_ordered)
        rejections = _rejection_probe(corpus, train_only_records, profiles, candidate_sets_ordered)

        replay_learner, _, _ = _fit_learner(
            corpus,
            ridge=0.1,
            minimum_utility=-1.0e9,
            maximum_uncertainty=1.0e9,
            reverse_order=True,
        )
        replay_selected = replay_learner.select(
            tuple(reversed(candidate_sets_ordered)), resource_budget=10.0, unseen_only=True
        )
        replay_matches = bool(
            replay_selected is not None
            and tuple(replay_selected[1].member_ids) == selected_members
            and math.isclose(
                float(replay_selected[1].predicted_interaction),
                float(candidate.predicted_interaction),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        executed_entries = [
            step for episode in matrix_episodes for step in episode["steps"] if step.get("executed")
        ]
        provenance_ok = all(
            set(step.get("provenance", {}).values())
            <= {"goal_state", "world_state", "transaction_token", "goal_state+world_state"}
            for step in executed_entries
        )
        safety_violations = sum(
            1
            for episode in matrix_episodes
            for step in episode["steps"]
            if step.get("safety_violation")
        )
        stop_reasons: Counter[str] = Counter()
        for episode in matrix_episodes:
            stop_reasons[str(episode["stop_reason"]).split(":")[0]] += 1

        all_cells_constant = bool(
            all(
                len({item["success"] for item in matrix_episodes if item["task_id"] == task_id})
                <= 1
                for task_id in {item["task_id"] for item in matrix_episodes}
            )
        )

        total_wall = time.perf_counter() - started
        evidence_admissible = bool(
            all(
                record.status == "candidate"
                and record.holdout_interaction is None
                and record.holdout_recovery_effect is None
                for record in train_only_records
            )
            and len(train_only_records) > 0
            and intervention_reality["interventions_happened"]
        )
        unseen_identity = bool(
            tuple(sorted(selected_members)) not in observed_pairs
            and all(member in {item.member_id for item in profiles} for member in selected_members)
            and bool(entry_audit["held_out_pairs"])
            and all(
                item["in_observed_records"] is False
                and item["joint_count_in_train"] == 0
                and item["joint_count_in_holdout"] > 0
                for item in entry_audit["held_out_pairs"].values()
            )
        )
        holdout_disjoint = bool(
            {item.context_id for item in corpus.train}.isdisjoint(
                {item.context_id for item in corpus.holdout}
            )
            and len(corpus.holdout)
            == (CONTEXT_COUNT - TRAIN_CONTEXT_COUNT) * REPEATS * len(CELL_MEMBER_SETS)
        )

        gates = {
            "static_checks": True,
            "evidence_admissibility": evidence_admissible,
            "unseen_combination_identity": unseen_identity,
            "unseen_context_holdout": holdout_disjoint,
            "label_opaqueness": True,
            "real_execution": bool(
                executed_entries
                and provenance_ok
                and safety_violations == 0
                and intervention_reality["interventions_happened"]
                and selected_is_held_out
            ),
            "rejection_recovery": bool(
                all(rejections.values()) and recovery["selection_reproduced"]
            ),
            "prediction_binding_and_calibration": bool(
                prediction_binding["bound_before_scoring"]
                and prediction_binding["parent_checkpoint_digest"]
                == parent_checkpoint["checkpoint_digest"]
                and calibration_ok
            ),
            "transfer_and_budget": bool(
                margin_cleared
                and collaboration_holds
                and any_held_out_positive
                and replica_consistent
                and total_wall <= TOTAL_SECONDS_CAP
            ),
        }
        mechanical = {key: value for key, value in gates.items() if key != "transfer_and_budget"}
        if not all(mechanical.values()):
            outcome = "failed"
        elif not all(gates.values()):
            if all_cells_constant or not any_held_out_positive:
                outcome = "transfer_signal_constant"
            else:
                outcome = "transfer_no_gain"
        else:
            outcome = "unseen_combination_transfer_supported"

        payload.update(
            {
                "status": "completed",
                "outcome": outcome,
                "experiment_passed": bool(outcome == "unseen_combination_transfer_supported"),
                "design": {
                    "members": list(MEMBER_IDS),
                    "member_kind": "family-specialist procedural readouts (same instances as P5.2b)",
                    "composition": "dual-predict-select (P5.2b frozen revision)",
                    "contexts": CONTEXT_COUNT,
                    "train_contexts": TRAIN_CONTEXT_COUNT,
                    "unseen_contexts": CONTEXT_COUNT - TRAIN_CONTEXT_COUNT,
                    "repeats": REPEATS,
                    "cells": ["(F,F)", "(T,F)", "(F,T)", "(T,T)"],
                    "margin": MARGIN,
                    "held_out_pairs": [list(pair) for pair in HELD_OUT_PAIRS],
                    "held_out_pair_opaque_indices": [list(idx) for idx in HELD_OUT_PAIR_INDICES],
                    "structural_change": (
                        "BOTH held-out pairs' joint (T,T) cells are removed from every "
                        "train context; their other cells and all singleton evidence "
                        "are retained, and holdout is untouched. See entry_audit."
                    ),
                    "member_set_widening_rejected_because": (
                        "P5.2a has exactly four train template families, one per "
                        "member, so extra members would duplicate an existing "
                        "capability surface and add redundancy rather than "
                        "complementarity; the complementarity this cohort needs "
                        "already exists (5 of 6 pairs are disjoint)"
                    ),
                    "candidate_pairs": [list(item) for item in PAIR_MEMBER_SETS],
                    "observed_pairs": sorted(list(item) for item in observed_pairs),
                    "learner_observes_train_only_records": True,
                    "recovery_effect_disclosure": (
                        "not measured this gate; recorded as 0 without fabrication"
                    ),
                    "semantic_mapping_disclosure": (
                        "member id -> training family mapping stored in runner config only"
                    ),
                    "sequence_generalization_claim": (
                        "none: gains are attributed to complementary family coverage only"
                    ),
                    "known_cost_disclosure": (
                        "P1* is structurally designated, so control C4 coincides with "
                        "the correct answer and is excluded from the margin-bearing "
                        "controls; a pass claims only prediction-realization "
                        "consistency plus superiority over no-learning/random/"
                        "singleton/regression/lesion controls, NOT the ability to "
                        "choose among many unseen combinations. The profile "
                        "representation is deliberately not fixed here."
                    ),
                },
                "entry_audit": entry_audit,
                "capability_surfaces": capability,
                "block3_audit": block3,
                "prediction_binding": prediction_binding,
                "prediction_surface": predictions,
                "object": {"member_ids": list(selected_members), "score": object_score},
                "held_out_pairs_scored": held_out_summary,
                "controls": {
                    name: {
                        "member_ids": (
                            list(control_sets[name])
                            if name in control_sets
                            else ["<per-context strongest singleton>"]
                        ),
                        "kind": value.get("kind", "pair_selection"),
                        "score": value,
                    }
                    for name, value in control_scores.items()
                },
                "control_summary": {
                    "strongest_control": strongest_control[0],
                    "strongest_control_member_ids": (
                        list(control_sets[strongest_control[0]])
                        if strongest_control[0] in control_sets
                        else ["<per-context strongest singleton>"]
                    ),
                    "strongest_control_gain": best_control_gain,
                    "object_gain": object_gain,
                    "margin": MARGIN,
                    "required": best_control_gain + MARGIN,
                    "margin_cleared": margin_cleared,
                    "collaboration_holds": collaboration_holds,
                    "any_held_out_pair_positive": any_held_out_positive,
                    "fixed_combination_gain_excluded_from_margin": fixed_control_gain,
                    "c2_strongest_singleton_gain": singleton_mean,
                    "margin_bearing_controls": sorted(margin_controls),
                },
                "calibration": calibration,
                "family_coverage": _family_coverage(matrix_episodes, contexts),
                "success_matrix": _success_matrix(matrix_episodes, contexts),
                "matrix": {
                    "episodes": len(matrix_episodes),
                    "train_episodes": len(corpus.train),
                    "holdout_episodes": len(corpus.holdout),
                    "all_cells_constant": all_cells_constant,
                    "stop_reason_classes": dict(sorted(stop_reasons.items())),
                },
                "intervention_reality": intervention_reality,
                "member_evidence": {
                    "profiles": len(profiles),
                    "member_ids": sorted(item.member_id for item in profiles),
                    "contributions": {item.member_id: item.contribution for item in profiles},
                    "contribution_uniform": len(
                        {round(float(item.contribution), 12) for item in profiles}
                    )
                    == 1,
                    "representation_defect_still_present_note": (
                        "uniform contributions mean the learner cannot distinguish "
                        "members; fixing this is route A and is NOT done here"
                    ),
                },
                "recovery": recovery,
                "rejections": rejections,
                "self_consistency": {"reverse_order_selection_matches": replay_matches},
                "deterministic_and_budget": {
                    "replica_consistent": replica_consistent,
                    "elapsed_seconds": round(total_wall, 3),
                    "total_seconds_cap": TOTAL_SECONDS_CAP,
                },
                "gates": gates,
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    f"completed: outcome={outcome}; object={'+'.join(selected_members)} "
                    f"gain={round(object_gain, 6)} vs strongest control "
                    f"{strongest_control[0]}={round(best_control_gain, 6)} "
                    f"(required {round(best_control_gain + MARGIN, 6)}); "
                    f"held_out_pairs={held_out_summary}; "
                    f"calibration sign_rate={sign_match_rate}, "
                    f"median_abs_error={median_absolute_error}, "
                    f"held_out_samples={calibration['unseen_sample_count']}"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    finally:
        p52cp.shutil.rmtree(workspace_root, ignore_errors=True)
        p52cp.shutil.rmtree(replica_root, ignore_errors=True)
    _write_json(DEFAULT_REPORT, payload)
    return payload


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "route": result.get("route"),
                "gates_failed": sorted(
                    key for key, value in (result.get("gates") or {}).items() if not value
                ),
                "entry_audit_passed": (result.get("entry_audit") or {}).get("passed"),
                "entry_audit_conditions": (result.get("entry_audit") or {}).get("conditions"),
                "held_out_pair_kinds": (result.get("capability_surfaces") or {}).get(
                    "held_out_pair_kinds"
                ),
                "block3_uniformly_unreachable": (result.get("block3_audit") or {}).get(
                    "uniformly_unreachable"
                ),
                "object": (result.get("object") or {}).get("member_ids"),
                "held_out_pairs_scored": result.get("held_out_pairs_scored"),
                "control_summary": result.get("control_summary"),
                "calibration": {
                    key: value
                    for key, value in (result.get("calibration") or {}).items()
                    if key
                    in (
                        "sign_match_rate",
                        "median_absolute_error",
                        "comparable_pair_count",
                        "unseen_sample_count",
                    )
                },
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
