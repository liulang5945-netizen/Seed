"""Build the PLAN-N1-122 v12 face from the approved four-observation specification."""

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
from scripts.training import build_taiji_n1_v11_after_n120 as v11_builder  # noqa: E402
from scripts.training import eval_taiji_n1_v11_after_n120 as v11_eval  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from scripts.training import run_taiji_n1_hard_neighbor_training as baseline_training  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-122_v12_hard_neighbor_range_recovery_draft_20261019.md"
PLAN_SHA256 = "762eec7cf9b6abc514b487dbb3f31d8497b53cee476d5501258c67f6c30ca84a"
N121_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-121_execution_closeout_20261019.md"
N121_CLOSEOUT_SHA256 = "a50abc1b359be0e21010864864b11c0daf6296a03d5217d0419aa8be339aac90"
N121_CORRECTION_SHA256 = "896dad354e6c81b7a1157746fe4e4bc5dcc8fcd266a7707dc09f476d7bb7aba7"
N121_FACE = ROOT / "plans/manifests/n1_hard_neighbor_v11_confirmation_seed111_20261019.json"
N121_FACE_SHA256 = "1d9236c5a258493ace574af74ae6b032632cf53011b009ad47660f862ea87f5e"
N121_RANGE = ROOT / "reports/taiji_n1_v11_after_n120_range_20261019.json"
N121_RANGE_SHA256 = "f1ce49385c1ee6001385a93a5bf65a7de08546c094afac697f3e6fee46fc8cf2"
N121_SOURCE_AUDIT = ROOT / "reports/taiji_n1_v11_after_n120_source_overlap_audit_20261019.json"
N121_SOURCE_AUDIT_SHA256 = "d96b51e2094af19736d804d06f605802c586ee2c214028ca3a441084f0938bdd"
V6_PREFLIGHT_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v6_preflight_consumed_face_exclusions_20261019.json"
)
V6_PREFLIGHT_EXCLUSIONS_SHA256 = "0fd51c841ddc4625b4ca3d70a021860be0108ea7c063ab342a5d5f9472f48de2"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v12_after_n121_source_exclusions_20261019.json"
READINESS = ROOT / "reports/taiji_n1_v12_after_n121_confirmation_readiness_20261019.json"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v12_confirmation_seed112_20261019.json"
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_v12_after_n121_face_checks_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v12_after_n121_range_20261019.json"
RANGE_ATTEMPT = RANGE_REPORT.with_name(RANGE_REPORT.stem + "_attempt.json")
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v12_after_n121_confirmation_20261019.json"
CONFIRM_ATTEMPT = CONFIRM_REPORT.with_name(CONFIRM_REPORT.stem + "_attempt.json")

SEED = 112
ITEMS = 40
ITEMS_PER_KIND = 10
OBSERVATIONS = 4
ENTITY_LENGTH = 10
ANSWER_LENGTHS = (6, 7)
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
DIRECTIONS = ("north", "east", "south", "west")
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
ABSTENTION = "unknown"
CPU_LIMIT_SECONDS = 600
EXPECTED_EXCLUDED_ENTITIES = 5898
EXPECTED_EXCLUDED_WORDS = 4728


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


def _check_v6_preflight_inventory(inventory: dict) -> tuple[set[str], set[str]]:
    if (
        inventory.get("schema") != "n1-v6-preflight-consumed-face-exclusions-v2"
        or inventory.get("status") != "complete"
        or inventory.get("unique_face_count") != 16
        or inventory.get("source_run_record_count") != 17
        or inventory.get("entity_count") != 3660
        or inventory.get("observed_material_count") != 2855
        or inventory.get("target_answer_value_count") != 324
        or inventory.get("all_excluded_word_value_count") != 2860
        or len(inventory.get("faces", [])) != 16
        or len(inventory.get("entity_names", [])) != 3660
        or len(inventory.get("observed_material_values", [])) != 2855
        or len(inventory.get("all_excluded_word_values", [])) != 2860
    ):
        raise ValueError("N1-95 consumed scratch-face exclusion inventory differs")
    if (
        len(set(inventory["entity_names"])) != 3660
        or len(set(inventory["observed_material_values"])) != 2855
        or len(set(inventory["all_excluded_word_values"])) != 2860
        or not set(inventory["observed_material_values"]).issubset(
            inventory["all_excluded_word_values"]
        )
        or not set(inventory["target_answer_values"]).issubset(
            inventory["all_excluded_word_values"]
        )
    ):
        raise ValueError("N1-95 exclusion inventory values are inconsistent")
    for raw_path, expected in inventory.get("source_hashes", {}).items():
        path = ROOT / raw_path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"N1-95 source report or generator changed: {path}")
    if len({item["face_sha256"] for item in inventory["faces"]}) != 16:
        raise ValueError("N1-95 inventory must contain 16 unique face hashes")
    run_face_hashes = set()
    run_record_count = 0
    for source_path in inventory.get("source_hashes", {}):
        if "taiji_n1_v6_preflight_range_sweep.json" not in source_path and (
            "taiji_n1_v6_preflight_confusability_sweep.json" not in source_path
        ):
            continue
        report = read_json(ROOT / source_path)
        run_record_count += len(report.get("runs", []))
        run_face_hashes.update(run.get("face_sha256") for run in report.get("runs", []))
    if run_record_count != 17 or run_face_hashes != {
        item["face_sha256"] for item in inventory["faces"]
    }:
        raise ValueError("N1-95 run records do not map to the frozen face set")
    # The committed inventory contains the exclusion values; verify the external source bytes
    # while they remain available, but do not use those files as evaluation inputs.
    for face in inventory["faces"]:
        source_path = Path(face["scratch_source_path"])
        if not source_path.is_file() or sha256(source_path) != face["face_sha256"]:
            raise ValueError(f"N1-95 scratch face source changed: {source_path}")
    return set(inventory["entity_names"]), set(inventory["all_excluded_word_values"])


def source_inventory() -> tuple[list[dict], set[str], set[str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-122 draft changed after approval")
    if sha256(N121_CLOSEOUT) != N121_CLOSEOUT_SHA256:
        raise ValueError("N1-121 closeout changed after this plan was drafted")
    if sha256(N121_FACE) != N121_FACE_SHA256 or sha256(N121_RANGE) != N121_RANGE_SHA256:
        raise ValueError("N1-121 face or range report changed")
    if sha256(N121_SOURCE_AUDIT) != N121_SOURCE_AUDIT_SHA256:
        raise ValueError("N1-121 source-overlap audit changed")
    if sha256(V6_PREFLIGHT_EXCLUSIONS) != V6_PREFLIGHT_EXCLUSIONS_SHA256:
        raise ValueError("N1-95 scratch exclusion inventory changed")
    n121_face = read_json(N121_FACE)
    range_report = read_json(N121_RANGE)
    overlap_audit = read_json(N121_SOURCE_AUDIT)
    if (
        n121_face.get("schema") != "n1-hard-neighbor-v11-confirmation-v1"
        or n121_face.get("seed") != 111
        or range_report.get("complete") is not True
        or range_report.get("status") != "range_not_met"
        or range_report.get("model_queries") != 180
        or range_report.get("confirmation_queries") != 0
        or range_report.get("source_hashes_unchanged") is not True
        or overlap_audit.get("status") != "complete_shared_answer_vocabulary_only"
        or overlap_audit.get("overlap_with_plan121_generated_face")
        != {"entities": 0, "observed_material_values": 0, "answer_vocabulary_values": 5}
    ):
        raise ValueError("N1-121 one-shot range failure or source boundary differs")

    train_items, names, words, prior = v11_builder._source_exclusions()
    n121_names, n121_materials, n121_answers = _face_entities_and_words(n121_face)
    inventory = read_json(V6_PREFLIGHT_EXCLUSIONS)
    scratch_names, scratch_words = _check_v6_preflight_inventory(inventory)
    names.update(n121_names)
    words.update(n121_materials)
    words.update(n121_answers)
    names.update(scratch_names)
    words.update(scratch_words)
    if len(names) != EXPECTED_EXCLUDED_ENTITIES or len(words) != EXPECTED_EXCLUDED_WORDS:
        raise ValueError(
            f"N1-122 source exclusion counts changed: entities={len(names)}, words={len(words)}"
        )
    if len(train_items) != 192:
        raise ValueError("frozen v3 training source must contain 192 items")

    # Reuse only the already-restored PLAN-N1-100/120 bundles if their complete receipts still
    # bind all bytes. This static readiness phase does not load a checkpoint.
    baseline, candidate, revalidation, prequery, restore, n121_readiness = (
        v11_eval._read_bound_inputs()
    )
    scorer_readiness = v11_eval._verify_scorer_readiness()
    checkpoint_hashes = {
        **prequery["baseline_checkpoint_hashes"],
        **prequery["candidate_checkpoint_hashes"],
    }
    if len(checkpoint_hashes) != 36 or any(
        not Path(path).is_file() or sha256(Path(path)) != digest
        for path, digest in checkpoint_hashes.items()
    ):
        raise ValueError("all 36 unchanged PLAN-N1-100/120 checkpoint hashes are required")
    if (
        restore.get("independent_full_bundle_restores") != 18
        or restore.get("all_full_bundles_equal") is not True
        or revalidation.get("training_sufficient") is not True
        or len(baseline.get("models", [])) != 9
        or len(candidate.get("models", [])) != 9
        or scorer_readiness.get("checkpoint_hash_count") != 36
        or scorer_readiness.get("model_queries") != 0
        or scorer_readiness.get("parameter_updates") != 0
    ):
        raise ValueError("previous final-bundle restore and training-gate evidence is stale")
    _, baseline_globals, _ = baseline_training.make_runner(load_inputs=False)
    n120.configure()
    _, candidate_globals, _ = n120._BASE.make_runner(load_inputs=False)
    if (
        not callable(baseline_globals.get("load"))
        or not callable(candidate_globals.get("load"))
        or not callable(v11_eval.scorer.evaluate)
    ):
        raise TypeError("baseline/candidate final checkpoint loaders must be available")

    source_paths = {
        PLAN,
        N121_CLOSEOUT,
        N121_FACE,
        N121_RANGE,
        N121_SOURCE_AUDIT,
        V6_PREFLIGHT_EXCLUSIONS,
        v11_builder.N121_PLAN,
        v11_builder.N121_SOURCE_CORRECTION,
        v11_builder.N110_SCHEDULE,
        v11_eval.BASELINE_REPORT,
        v11_eval.CANDIDATE_REPORT,
        v11_eval.CANDIDATE_REVALIDATION,
        v11_eval.PREQUERY_READINESS,
        v11_eval.RESTORE_READINESS,
        v11_eval.FACE_READINESS,
        v11_eval.SCORER_READINESS,
        Path(v11_builder.__file__).resolve(),
        Path(v11_eval.__file__).resolve(),
        Path(v11_eval.scorer.__file__).resolve(),
        Path(v11_builder.v9.__file__).resolve(),
        Path(v11_builder.train_data.__file__).resolve(),
        Path(baseline_training.__file__).resolve(),
        Path(n120.__file__).resolve(),
        Path(n120.expanded.__file__).resolve(),
        Path(n120.plan120_receipt.__file__).resolve(),
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v12_after_n121.py",
        ROOT / "tests/taiji_native/test_n1_plan122_v12_face_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan122_scoring_contract.py",
    }
    source_paths.update(Path(face["scratch_source_path"]) for face in inventory["faces"])
    source_hashes = {str(Path(path).resolve()): sha256(Path(path)) for path in source_paths}
    # Carry forward the complete v11 readiness provenance, including the old training, source,
    # adapter, and restore identities it already checked.
    source_hashes.update(scorer_readiness["source_hashes"])
    provenance = {
        "n121_plan_sha256": v11_eval.N121_PLAN_SHA256,
        "n121_correction_sha256": N121_CORRECTION_SHA256,
        "n121_face_sha256": N121_FACE_SHA256,
        "n121_range_report_sha256": N121_RANGE_SHA256,
        "n121_source_audit_sha256": N121_SOURCE_AUDIT_SHA256,
        "v6_preflight_inventory_sha256": V6_PREFLIGHT_EXCLUSIONS_SHA256,
        "n110_material_map_sha256": v11_builder.N110_VALUE_MAP_SHA256,
        "n110_material_map_entries": v11_builder.N110_VALUE_MAP_ENTRIES,
        "n121_face_entities_excluded": len(n121_names),
        "n121_face_observed_materials_excluded": len(n121_materials),
        "n121_face_answer_values_excluded": len(n121_answers),
        "preflight_faces_excluded": inventory["unique_face_count"],
        "preflight_entities_excluded": len(scratch_names),
        "preflight_observed_materials_excluded": inventory["observed_material_count"],
        "preflight_answer_values_excluded": inventory["target_answer_value_count"],
        "excluded_entity_count": len(names),
        "excluded_word_value_count": len(words),
        "source_hashes": source_hashes,
    }
    return (
        train_items,
        names,
        words,
        {
            "provenance": provenance,
            "source_hashes": source_hashes,
            "checkpoint_hashes": checkpoint_hashes,
            "baseline": baseline,
            "candidate": candidate,
            "revalidation": revalidation,
            "prequery": prequery,
            "restore": restore,
            "n121_readiness": n121_readiness,
            "v11_scorer_readiness": scorer_readiness,
        },
    )


def source_readiness() -> dict:
    if any(
        path.exists()
        for path in (
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
    ):
        raise FileExistsError("PLAN-N1-122 paths are write-once; no redraw or retry")
    started = time.process_time()
    train_items, names, words, bound = source_inventory()
    if len(names) != EXPECTED_EXCLUDED_ENTITIES or len(words) != EXPECTED_EXCLUDED_WORDS:
        raise ValueError("complete source exclusion union required")
    source_exclusion_manifest = {
        "schema": "n1-plan122-v12-source-exclusions-v1",
        "plan_sha256": PLAN_SHA256,
        "status": "ready_for_approval",
        "excluded_entities": sorted(names),
        "excluded_word_values": sorted(words),
        "entity_count": len(names),
        "word_value_count": len(words),
        "provenance": bound["provenance"],
        "source_hashes": bound["source_hashes"],
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, source_exclusion_manifest)
    exclusion_sha = sha256(SOURCE_EXCLUSIONS)
    total = time.process_time() - started
    source_paths = dict(bound["source_hashes"])
    source_paths[str(SOURCE_EXCLUSIONS.resolve())] = exclusion_sha
    source_paths[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    report = {
        "schema": "n1-plan122-v12-zero-query-readiness-v1",
        "status": "ready_for_approval" if total <= CPU_LIMIT_SECONDS else "stopped",
        "complete": total <= CPU_LIMIT_SECONDS,
        "plan_sha256": PLAN_SHA256,
        "source_exclusion_manifest_sha256": exclusion_sha,
        "source_hashes": source_paths,
        "checkpoint_hashes": bound["checkpoint_hashes"],
        "checkpoint_hash_count": len(bound["checkpoint_hashes"]),
        "baseline_endpoints": len(bound["baseline"].get("models", [])),
        "candidate_endpoints": len(bound["candidate"].get("models", [])),
        "candidate_position_training_gates_passed": sum(
            model.get("training_sufficient") is True
            for model in bound["revalidation"].get("models", [])
            if model.get("arm") == "position"
        ),
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


def build_v12_face(
    train_items: list[dict], excluded_names: set[str], excluded_words: set[str]
) -> dict:
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
                "id": f"v12-hard-{index:03d}",
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
        "schema": "n1-hard-neighbor-v12-confirmation-v1",
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


def audit_v12_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    names, observed_materials, answers = _face_entities_and_words(face)
    kind_counts = Counter(item.get("kind") for item in episodes)
    id_set = {item.get("id") for item in episodes}
    subsets = face.get("subsets", {})
    checks = {
        "schema_and_seed": face.get("schema") == "n1-hard-neighbor-v12-confirmation-v1"
        and face.get("seed") == SEED,
        "face_size_and_classes": len(episodes) == ITEMS
        and kind_counts == Counter({kind: ITEMS_PER_KIND for kind in KINDS}),
        "unique_holdout_ids": len(id_set) == ITEMS
        and all(item.get("split") == "holdout" for item in episodes)
        and all(item.get("id") == f"v12-hard-{index:03d}" for index, item in enumerate(episodes)),
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
                        row for row in item["observations"] if row.startswith(subject + " ")
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
        "schema": "n1-plan122-v12-face-checks-v1",
        "checks": checks,
        "length_split": length_checks,
        "entity_count": len(names),
        "observation_material_count": len(observed_materials),
        "answer_value_count": len(answers),
        "accepted": all(checks.values()),
    }


def build_face_after_approval() -> dict:
    if any(
        path.exists()
        for path in (
            FACE,
            FACE_ATTEMPT,
            FACE_CHECKS,
            RANGE_REPORT,
            RANGE_ATTEMPT,
            CONFIRM_REPORT,
            CONFIRM_ATTEMPT,
        )
    ):
        raise FileExistsError("PLAN-N1-122 paths are one-shot; no retry or redraw")
    if not READINESS.is_file():
        raise FileNotFoundError("zero-query PLAN-N1-122 readiness is required")
    readiness = read_json(READINESS)
    _verify_readiness(readiness)
    inventory = read_json(SOURCE_EXCLUSIONS)
    train_items, excluded_names, excluded_words, bound = source_inventory()
    if (
        inventory.get("plan_sha256") != PLAN_SHA256
        or set(inventory.get("excluded_entities", [])) != excluded_names
        or set(inventory.get("excluded_word_values", [])) != excluded_words
        or sha256(SOURCE_EXCLUSIONS) != readiness.get("source_exclusion_manifest_sha256")
    ):
        raise ValueError("PLAN-N1-122 source-exclusion manifest is stale")
    started = time.process_time()
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan122-v12-face-attempt-v1",
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
    face = build_v12_face(train_items, excluded_names, excluded_words)
    face["exclusion_sources"] = {
        "plan_sha256": PLAN_SHA256,
        "source_readiness_sha256": sha256(READINESS),
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "n121_face_sha256": N121_FACE_SHA256,
        "v6_preflight_inventory_sha256": V6_PREFLIGHT_EXCLUSIONS_SHA256,
        "excluded_entity_count": len(excluded_names),
        "excluded_word_value_count": len(excluded_words),
        "source_hashes": bound["source_hashes"],
    }
    checks = audit_v12_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
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
        raise ValueError("PLAN-N1-122 face failed its frozen contract; do not retry")
    write_new(FACE, face)
    checks["face_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return checks


def _verify_readiness(report: dict | None = None) -> dict:
    if report is None:
        if not READINESS.is_file():
            raise FileNotFoundError("PLAN-N1-122 zero-query readiness is required")
        report = read_json(READINESS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approval"
        or report.get("schema") != "n1-plan122-v12-zero-query-readiness-v1"
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
        raise ValueError("PLAN-N1-122 zero-query readiness is stale or incomplete")
    for raw_path, expected in report.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"PLAN-N1-122 source changed after readiness: {path}")
    for raw_path, expected in report.get("checkpoint_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"endpoint checkpoint changed after readiness: {path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness", action="store_true")
    mode.add_argument("--build-face", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.build_face:
        if not args.owner_approved:
            raise PermissionError("PLAN-N1-122 face generation requires owner approval")
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
