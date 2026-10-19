"""Build the approved PLAN-N1-123 sixteen-observation v13 face."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_hard_neighbor_v9 as v9  # noqa: E402
from scripts.training import build_taiji_n1_v12_after_n121 as v12_builder  # noqa: E402
from scripts.training import eval_taiji_n1_v12_after_n121 as v12_eval  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-123_v13_long_context_range_probe_draft_20261019.md"
PLAN_SHA256 = "421daf945e87af1c85d1a0b135d124b83a5f3c404f619c537f289200d34962eb"
N122_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-122_execution_closeout_20261019.md"
N122_CLOSEOUT_SHA256 = "93a560e08eb992b82e4a69b6f5a858011ea08e3bbe0de1150718902d312754dc"
N122_PLAN_SHA256 = "762eec7cf9b6abc514b487dbb3f31d8497b53cee476d5501258c67f6c30ca84a"
N122_SOURCE_EXCLUSIONS = v12_builder.SOURCE_EXCLUSIONS
N122_SOURCE_EXCLUSIONS_SHA256 = "32b7b69c8c6e0a033964e85c59653558fb6d675a67d005b3b1464f7f2f15c4d4"
N122_READINESS = v12_builder.READINESS
N122_READINESS_SHA256 = "fa659e3ab59685bfc3ec3273698ca58d9537f3fbf2ef5f205af1b33634733717"
N122_FACE = v12_builder.FACE
N122_FACE_SHA256 = "dc8daeecc190a40b308fc8f7f79b432a021970649de782c7799dd995ce0c6a11"
N122_FACE_CHECKS = v12_builder.FACE_CHECKS
N122_FACE_CHECKS_SHA256 = "f0bf8ce3979f1f352b4c2885e8c231e9c3961815f615f157ce40eeb2dbd0d7d8"
N122_RANGE = v12_builder.RANGE_REPORT
N122_RANGE_SHA256 = "3f3e68cdddd692daac3d3ba51e2573ce71195469b693388d42b9c215f696853b"
N122_RANGE_ATTEMPT = v12_builder.RANGE_ATTEMPT
N122_RANGE_ATTEMPT_SHA256 = "59874cd5410b57aef0979f6e1feadbc6b59a93421a9ab0b456e8714aa6594f2a"
N122_RANGE_ATTRIBUTION = ROOT / "reports/taiji_n1_v12_after_n121_range_attribution_20261019.json"
N122_RANGE_ATTRIBUTION_SHA256 = "aca64b0169214366dfcee014e530f941c1f6cbaaafed1a3bf26d4034e02fddf0"
N122_CPU_RECONCILIATION = ROOT / "reports/taiji_n1_v12_after_n121_cpu_reconciliation_20261019.json"
N122_CPU_RECONCILIATION_SHA256 = "9337c43c8cf9c877c081edaa6d8615dc2888166da95ec2af940f0b55a5acb2a6"

SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v13_after_n122_source_exclusions_20261019.json"
READINESS = ROOT / "reports/taiji_n1_v13_after_n122_confirmation_readiness_20261019.json"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json"
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_v13_after_n122_face_checks_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v13_after_n122_range_20261019.json"
RANGE_ATTEMPT = RANGE_REPORT.with_name(RANGE_REPORT.stem + "_attempt.json")
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v13_after_n122_confirmation_20261019.json"
CONFIRM_ATTEMPT = CONFIRM_REPORT.with_name(CONFIRM_REPORT.stem + "_attempt.json")

SEED = 113
ITEMS = 40
ITEMS_PER_KIND = 10
OBSERVATIONS = 16
ENTITY_LENGTH = 10
ANSWER_LENGTHS = (6, 7)
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
ABSTENTION = "unknown"
CPU_LIMIT_SECONDS = 600
EXPECTED_EXCLUDED_ENTITIES = 6048
EXPECTED_EXCLUDED_WORDS = 4848


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _face_entities_and_words(face: dict) -> tuple[set[str], set[str], set[str]]:
    names: set[str] = set()
    answer_values: set[str] = set()
    observed_materials: set[str] = set()
    for episode in face.get("episodes", []):
        names.update(episode["entities"])
        answer_values.add(episode["answer"])
        observed_materials.update(
            row.removesuffix(".").split()[-1] for row in episode["observations"] if " uses " in row
        )
    return names, observed_materials, answer_values


def source_inventory() -> tuple[list[dict], set[str], set[str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-123 changed after approval")
    fixed_sources = {
        N122_CLOSEOUT: N122_CLOSEOUT_SHA256,
        v12_builder.PLAN: N122_PLAN_SHA256,
        N122_SOURCE_EXCLUSIONS: N122_SOURCE_EXCLUSIONS_SHA256,
        N122_READINESS: N122_READINESS_SHA256,
        N122_FACE: N122_FACE_SHA256,
        N122_FACE_CHECKS: N122_FACE_CHECKS_SHA256,
        N122_RANGE: N122_RANGE_SHA256,
        N122_RANGE_ATTEMPT: N122_RANGE_ATTEMPT_SHA256,
        N122_RANGE_ATTRIBUTION: N122_RANGE_ATTRIBUTION_SHA256,
        N122_CPU_RECONCILIATION: N122_CPU_RECONCILIATION_SHA256,
    }
    for path, expected in fixed_sources.items():
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen N1-122 source changed: {path}")
    if v12_builder.PLAN_SHA256 != N122_PLAN_SHA256:
        raise ValueError("N1-122 plan identity is inconsistent")
    if v12_builder.CONFIRM_REPORT.exists() or v12_builder.CONFIRM_ATTEMPT.exists():
        raise ValueError("N1-122 confirmation must remain unqueried")

    readiness = read_json(N122_READINESS)
    exclusions = read_json(N122_SOURCE_EXCLUSIONS)
    range_report = read_json(N122_RANGE)
    range_attempt = read_json(N122_RANGE_ATTEMPT)
    attribution = read_json(N122_RANGE_ATTRIBUTION)
    cpu = read_json(N122_CPU_RECONCILIATION)
    if (
        readiness.get("complete") is not True
        or readiness.get("status") != "ready_for_approval"
        or readiness.get("model_loads") != 0
        or readiness.get("model_queries") != 0
        or readiness.get("parameter_updates") != 0
        or readiness.get("checkpoint_hash_count") != 36
        or readiness.get("independent_full_bundle_restores_reused") != 18
        or exclusions.get("status") != "ready_for_approval"
        or exclusions.get("entity_count") != 5898
        or exclusions.get("word_value_count") != 4728
        or readiness.get("source_exclusion_manifest_sha256") != sha256(N122_SOURCE_EXCLUSIONS)
        or range_report.get("complete") is not True
        or range_report.get("status") != "range_not_met"
        or range_report.get("range_gate", {}).get("range_check_passed") is not False
        or range_report.get("range_gate", {}).get("usable_endpoints") != 3
        or range_report.get("model_queries") != 180
        or range_report.get("confirmation_queries") != 0
        or range_report.get("source_hashes_unchanged") is not True
        or range_report.get("face_sha256") != N122_FACE_SHA256
        or range_report.get("readiness_sha256") != N122_READINESS_SHA256
        or range_attempt.get("status") != "started"
        or attribution.get("status") != "complete"
        or attribution.get("confirmation_queries") != 0
        or attribution.get("face_sha256") != N122_FACE_SHA256
        or cpu.get("status") != "complete_budget_remains_valid"
        or cpu.get("range_result_unchanged") is not True
        or cpu.get("reconciled_cumulative_cpu_seconds", CPU_LIMIT_SECONDS + 1) > CPU_LIMIT_SECONDS
        or cpu.get("source_hashes", {}).get("range_report") != N122_RANGE_SHA256
        or cpu.get("source_hashes", {}).get("face_checks") != N122_FACE_CHECKS_SHA256
        or cpu.get("source_hashes", {}).get("readiness") != N122_READINESS_SHA256
    ):
        raise ValueError("N1-122 closeout evidence is incomplete or inconsistent")

    # This revalidates the original v3/v5/v6/v7-v12 and scratch-face exclusions,
    # all bound sources, the checkpoint hashes, and the 18 independent restore receipts.
    train_items, names, words, bound = v12_builder.source_inventory()
    face, checks = v12_eval._face_and_checks()
    n122_names, n122_materials, n122_answers = _face_entities_and_words(face)
    if (
        checks.get("accepted") is not True
        or len(n122_names) != 150
        or len(n122_materials) != 120
        or len(n122_answers) != 25
        or exclusions.get("plan_sha256") != N122_PLAN_SHA256
        or set(exclusions.get("excluded_entities", [])) != names
        or set(exclusions.get("excluded_word_values", [])) != words
    ):
        raise ValueError("frozen N1-122 face or exclusion inventory differs")
    names.update(n122_names)
    words.update(n122_materials)
    words.update(n122_answers)
    if len(names) != EXPECTED_EXCLUDED_ENTITIES or len(words) != EXPECTED_EXCLUDED_WORDS:
        raise ValueError(
            f"N1-123 exclusion union changed: entities={len(names)}, words={len(words)}"
        )

    source_hashes = dict(bound["source_hashes"])
    source_hashes.update({str(path.resolve()): sha256(path) for path in fixed_sources})
    source_hashes[str(PLAN.resolve())] = PLAN_SHA256
    source_hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    source_hashes[str((ROOT / "scripts/training/eval_taiji_n1_v13_after_n122.py").resolve())] = (
        sha256(ROOT / "scripts/training/eval_taiji_n1_v13_after_n122.py")
    )
    test_paths = (
        ROOT / "tests/taiji_native/test_n1_plan123_v13_face_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan123_scoring_contract.py",
    )
    for path in test_paths:
        if not path.is_file():
            raise FileNotFoundError(f"N1-123 bound test is required before readiness: {path}")
        source_hashes[str(path.resolve())] = sha256(path)

    provenance = {
        "n122_plan_sha256": N122_PLAN_SHA256,
        "n122_closeout_sha256": N122_CLOSEOUT_SHA256,
        "n122_face_sha256": N122_FACE_SHA256,
        "n122_source_exclusions_sha256": N122_SOURCE_EXCLUSIONS_SHA256,
        "n122_readiness_sha256": N122_READINESS_SHA256,
        "n122_face_checks_sha256": N122_FACE_CHECKS_SHA256,
        "n122_range_report_sha256": N122_RANGE_SHA256,
        "n122_range_attempt_sha256": N122_RANGE_ATTEMPT_SHA256,
        "n122_range_attribution_sha256": N122_RANGE_ATTRIBUTION_SHA256,
        "n122_cpu_reconciliation_sha256": N122_CPU_RECONCILIATION_SHA256,
        "n122_face_entities_excluded": len(n122_names),
        "n122_face_observed_materials_excluded": len(n122_materials),
        "n122_face_answer_values_excluded": len(n122_answers),
        "excluded_entity_count": len(names),
        "excluded_word_value_count": len(words),
        "source_hashes": source_hashes,
    }
    return (
        train_items,
        names,
        words,
        {
            **bound,
            "provenance": provenance,
            "source_hashes": source_hashes,
            "n122_face": face,
            "n122_face_checks": checks,
            "n122_range": range_report,
        },
    )


def source_readiness() -> dict:
    output_paths = (
        SOURCE_EXCLUSIONS,
        READINESS,
        FACE,
        FACE_ATTEMPT,
        FACE_CHECKS,
        RANGE_REPORT,
        RANGE_ATTEMPT,
        CONFIRM_REPORT,
        CONFIRM_ATTEMPT,
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("PLAN-N1-123 paths are write-once; no redraw or retry")
    started = time.process_time()
    train_items, names, words, bound = source_inventory()
    if len(names) != EXPECTED_EXCLUDED_ENTITIES or len(words) != EXPECTED_EXCLUDED_WORDS:
        raise ValueError("complete PLAN-N1-123 exclusion union required")
    write_new(
        SOURCE_EXCLUSIONS,
        {
            "schema": "n1-plan123-v13-source-exclusions-v1",
            "plan_sha256": PLAN_SHA256,
            "status": "ready_for_approved_execution",
            "excluded_entities": sorted(names),
            "excluded_word_values": sorted(words),
            "entity_count": len(names),
            "word_value_count": len(words),
            "provenance": bound["provenance"],
            "source_hashes": bound["source_hashes"],
            "can_promote": False,
        },
    )
    source_hashes = dict(bound["source_hashes"])
    source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
    # Finish source-hash verification before freezing the CPU reading into the report.
    if any(
        not Path(path).is_file() or sha256(Path(path)) != expected
        for path, expected in source_hashes.items()
    ):
        raise ValueError("a frozen source changed during PLAN-N1-123 readiness")
    total = time.process_time() - started
    report = {
        "schema": "n1-plan123-v13-zero-query-readiness-v1",
        "status": "ready_for_approved_execution" if total <= CPU_LIMIT_SECONDS else "stopped",
        "complete": total <= CPU_LIMIT_SECONDS,
        "plan_sha256": PLAN_SHA256,
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": source_hashes,
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "baseline_endpoints": len(bound["baseline"].get("models", [])),
        "candidate_endpoints": len(bound["candidate"].get("models", [])),
        "independent_full_bundle_restores_reused": 18,
        "all_full_bundles_equal": True,
        "train_items": len(train_items),
        "excluded_entity_count": len(names),
        "excluded_word_value_count": len(words),
        "planned_face_seed": SEED,
        "planned_items": ITEMS,
        "planned_items_per_kind": ITEMS_PER_KIND,
        "planned_observations_per_item": OBSERVATIONS,
        "planned_name_neighbor_distance": DISTANCE,
        "face_generated": False,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "planned_range_queries": 180,
        "planned_confirmation_queries_if_range_passes": 360,
        "planned_max_total_queries": 540,
        "cpu_seconds": round(total, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
    }
    write_new(READINESS, report)
    return report


def build_v13_face(
    train_items: list[dict], excluded_names: set[str], excluded_words: set[str]
) -> dict:
    del train_items  # The frozen v3 corpus is bound for exclusion; no training data are sampled.
    rng = random.Random(SEED)
    used_names = set(excluded_names)
    used_words = set(excluded_words) | set(DIRECTIONS) | {ABSTENTION}
    seen = Counter()
    episodes = []
    for index in range(ITEMS):
        kind = KINDS[index % len(KINDS)]
        ordinal = seen[kind]
        seen[kind] += 1
        subject = v9._novel(ENTITY_LENGTH, used_names, rng)
        distractor_count = OBSERVATIONS - SUBJECT_ROWS[kind]
        distractors = [v9._confusable(subject, used_names, rng) for _ in range(distractor_count)]
        materials = [v9._novel(6, used_words, rng) for _ in range(distractor_count)]
        answer_length = ANSWER_LENGTHS[ordinal % len(ANSWER_LENGTHS)]
        target_rows: list[str] = []
        if kind == "lookup":
            answer = v9._novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {answer}."]
        elif kind == "revision":
            stale = v9._novel(answer_length, used_words, rng)
            answer = v9._novel(answer_length, used_words, rng)
            question = f"material of {subject}?"
            target_rows = [f"{subject} uses {stale}.", f"{subject} now uses {answer}."]
        elif kind == "transition":
            answer = DIRECTIONS[ordinal % len(DIRECTIONS)]
            turn = rng.choice((1, 3))
            start = (DIRECTIONS.index(answer) - turn) % len(DIRECTIONS)
            question = f"direction of {subject}?"
            turn_word = "right" if turn == 1 else "left"
            target_rows = [f"{subject} faces {DIRECTIONS[start]}.", f"{subject} turns {turn_word}."]
        else:
            answer = ABSTENTION
            question = f"material of {subject}?"

        if kind == "transition":
            filler = [
                f"{name} faces {DIRECTIONS[(ordinal + offset) % len(DIRECTIONS)]}."
                for offset, name in enumerate(distractors)
            ]
        else:
            filler = [
                f"{name} uses {value}." for name, value in zip(distractors, materials, strict=True)
            ]
        rows: list[str | None] = [None] * OBSERVATIONS
        within_subset_ordinal = ordinal % (ITEMS_PER_KIND // 2)
        target_start = (
            within_subset_ordinal % (OBSERVATIONS - len(target_rows) + 1) if target_rows else 0
        )
        for offset, row in enumerate(target_rows):
            rows[target_start + offset] = row
        filler_iter = iter(filler)
        for position, row in enumerate(rows):
            if row is None:
                rows[position] = next(filler_iter)
        subset = "sacrifice" if ordinal < ITEMS_PER_KIND // 2 else "confirm"
        episodes.append(
            {
                "id": f"v13-hard-{index:03d}",
                "split": "holdout",
                "subset": subset,
                "kind": kind,
                "entities": [subject, *distractors],
                "observations": [str(row) for row in rows],
                "question": question,
                "answer": answer,
            }
        )
    subsets = {
        subset: [episode["id"] for episode in episodes if episode["subset"] == subset]
        for subset in ("sacrifice", "confirm")
    }
    return {
        "schema": "n1-hard-neighbor-v13-confirmation-v1",
        "seed": SEED,
        "items": ITEMS,
        "items_per_kind": ITEMS_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": ENTITY_LENGTH,
        "name_neighbor_distance": DISTANCE,
        "answer_lengths": list(ANSWER_LENGTHS),
        "subset_sizes": {name: len(ids) for name, ids in subsets.items()},
        "subsets": subsets,
        "exclusion_sources": {},
        "episodes": episodes,
    }


def audit_v13_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    names, observed_materials, answers = _face_entities_and_words(face)
    kind_counts = Counter(item.get("kind") for item in episodes)
    id_set = {item.get("id") for item in episodes}
    subsets = face.get("subsets", {})
    checks = {
        "schema_and_seed": face.get("schema") == "n1-hard-neighbor-v13-confirmation-v1"
        and face.get("seed") == SEED,
        "face_size_and_classes": len(episodes) == ITEMS
        and kind_counts == Counter({kind: ITEMS_PER_KIND for kind in KINDS}),
        "unique_holdout_ids": len(id_set) == ITEMS
        and all(item.get("split") == "holdout" for item in episodes)
        and all(item.get("id") == f"v13-hard-{index:03d}" for index, item in enumerate(episodes)),
        "observation_count": all(
            len(item.get("observations", [])) == OBSERVATIONS for item in episodes
        ),
        "confusable_name_distance": all(
            sum(a != b for a, b in zip(item["entities"][0], other, strict=True)) == DISTANCE
            for item in episodes
            for other in item["entities"][1:]
        ),
        "disjoint_entity_names": not (names & excluded_names),
        "disjoint_observation_materials": not (observed_materials & excluded_words),
        "subset_partition": set(subsets) == {"sacrifice", "confirm"}
        and len(subsets.get("sacrifice", [])) == ITEMS // 2
        and len(subsets.get("confirm", [])) == ITEMS // 2
        and set(subsets.get("sacrifice", [])).isdisjoint(subsets.get("confirm", []))
        and set(subsets.get("sacrifice", [])) | set(subsets.get("confirm", [])) == id_set,
        "subset_class_balance": all(
            sum(item["kind"] == kind and item["id"] in subsets.get(subset, []) for item in episodes)
            == ITEMS_PER_KIND // 2
            for subset in ("sacrifice", "confirm")
            for kind in KINDS
        ),
    }
    length_checks = {}
    for kind in ("lookup", "revision"):
        length_checks[kind] = {}
        for subset, expected in (("sacrifice", {6: 3, 7: 2}), ("confirm", {6: 2, 7: 3})):
            selected = [
                item
                for item in episodes
                if item["kind"] == kind and item["id"] in subsets.get(subset, [])
            ]
            lengths = Counter(len(item["answer"]) for item in selected)
            revision_matches = True
            if kind == "revision":
                for item in selected:
                    subject = item["entities"][0]
                    target_rows = [
                        row
                        for row in item["observations"]
                        if row.startswith(subject + " uses ")
                        or row.startswith(subject + " now uses ")
                    ]
                    revision_matches &= len(target_rows) == 2 and len(
                        target_rows[0].removesuffix(".").split()[-1]
                    ) == len(target_rows[1].removesuffix(".").split()[-1]) == len(item["answer"])
            length_checks[kind][subset] = {
                "counts": dict(lengths),
                "ok": lengths == Counter(expected) and revision_matches,
            }
    checks["answer_length_split"] = all(
        length_checks[kind][subset]["ok"]
        for kind in length_checks
        for subset in ("sacrifice", "confirm")
    )
    return {
        "schema": "n1-plan123-v13-face-checks-v1",
        "checks": checks,
        "length_split": length_checks,
        "entity_count": len(names),
        "observation_material_count": len(observed_materials),
        "answer_value_count": len(answers),
        "accepted": all(checks.values()),
    }


def _verify_readiness(report: dict | None = None) -> dict:
    if report is None:
        if not READINESS.is_file():
            raise FileNotFoundError("PLAN-N1-123 zero-query readiness is required")
        report = read_json(READINESS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("schema") != "n1-plan123-v13-zero-query-readiness-v1"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
        or report.get("checkpoint_hash_count") != 36
        or len(report.get("checkpoint_hashes", {})) != 36
        or report.get("independent_full_bundle_restores_reused") != 18
        or report.get("all_full_bundles_equal") is not True
        or report.get("excluded_entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or report.get("excluded_word_value_count") != EXPECTED_EXCLUDED_WORDS
        or report.get("model_loads") != 0
        or report.get("checkpoint_loads") != 0
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("optimizer_steps") != 0
        or report.get("face_generated") is not False
    ):
        raise ValueError("PLAN-N1-123 zero-query readiness is stale or incomplete")
    for raw_path, expected in report.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"PLAN-N1-123 source changed after readiness: {path}")
    for raw_path, expected in report.get("checkpoint_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"endpoint checkpoint changed after readiness: {path}")
    return report


def build_face_after_approval() -> dict:
    downstream_paths = (
        FACE,
        FACE_ATTEMPT,
        FACE_CHECKS,
        RANGE_REPORT,
        RANGE_ATTEMPT,
        CONFIRM_REPORT,
        CONFIRM_ATTEMPT,
    )
    if any(path.exists() for path in downstream_paths):
        raise FileExistsError(
            "PLAN-N1-123 face and query stages are write-once; no retry or redraw"
        )
    if not READINESS.is_file():
        raise FileNotFoundError("zero-query PLAN-N1-123 readiness is required")
    readiness = _verify_readiness()
    inventory = read_json(SOURCE_EXCLUSIONS)
    train_items, excluded_names, excluded_words, bound = source_inventory()
    if (
        inventory.get("plan_sha256") != PLAN_SHA256
        or set(inventory.get("excluded_entities", [])) != excluded_names
        or set(inventory.get("excluded_word_values", [])) != excluded_words
        or sha256(SOURCE_EXCLUSIONS) != readiness.get("source_exclusion_manifest_sha256")
    ):
        raise ValueError("PLAN-N1-123 source-exclusion manifest is stale")
    started = time.process_time()
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan123-v13-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "source_readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    face = build_v13_face(train_items, excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "plan_sha256": PLAN_SHA256,
        "source_readiness_sha256": sha256(READINESS),
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "n122_plan_sha256": N122_PLAN_SHA256,
        "n122_face_sha256": N122_FACE_SHA256,
        "n122_range_report_sha256": N122_RANGE_SHA256,
        "n122_cpu_reconciliation_sha256": N122_CPU_RECONCILIATION_SHA256,
        "excluded_entity_count": len(excluded_names),
        "excluded_word_value_count": len(excluded_words),
        "source_hashes": bound["source_hashes"],
    }
    checks = audit_v13_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
    checks.update(
        {
            "face": str(FACE),
            "face_sha256": None,
            "plan_sha256": PLAN_SHA256,
            "source_readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "build_cpu_seconds": round(time.process_time() - started, 6),
        }
    )
    if not checks["accepted"]:
        write_new(FACE_CHECKS, checks)
        raise ValueError("PLAN-N1-123 face failed its frozen contract; do not retry")
    write_new(FACE, face)
    checks["face_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--build-face", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.build_face:
        if not args.owner_approved:
            raise PermissionError("PLAN-N1-123 face generation requires owner approval")
        result = build_face_after_approval()
    else:
        result = source_readiness()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "schema",
                    "status",
                    "complete",
                    "accepted",
                    "excluded_entity_count",
                    "excluded_word_value_count",
                    "face_sha256",
                    "model_loads",
                    "checkpoint_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                    "cpu_seconds",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("accepted", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
