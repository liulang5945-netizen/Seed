"""Build the frozen PLAN-N1-105 counterbalanced diagnostic face."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from scripts.training import build_taiji_n1_hard_neighbor_training_schedule as train_data
from scripts.training import build_taiji_n1_hard_neighbor_v9 as v9
from scripts.training import run_taiji_n1_hard_neighbor_training as training

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-105_counterbalanced_input_feature_diagnostic_20261017.md"
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v10_factorial_diagnostic_20261017.json"
CHECKS = ROOT / "reports/taiji_n1_hard_neighbor_v10_factorial_diagnostic_checks_20261017.json"
EVALUATOR = ROOT / "scripts/training/eval_taiji_n1_input_factorial_diagnostic.py"
CONTRACT_TEST = ROOT / "tests/taiji_native/test_n1_run105_input_factorial_diagnostic_contract.py"

SEED = 110
KINDS = ("lookup", "revision", "transition", "unknown")
ANSWER_LENGTHS = (6, 7)
TEMPLATES_PER_KIND = 4
OBSERVATIONS = 12
UNKNOWN_ITEMS = 20
EXPECTED_ITEMS = {"lookup": 96, "revision": 88, "transition": 44, "unknown": 20}
V9_FACE_SHA256 = "d93dd6c6e0d98125ef1c91ed7ad8aec07a46f6621fb597dbffa6e889f29ac60f"
PLAN_SHA256 = "af6eebc3e3fb01fc8c409f762176aefa8b91f246e9ea7325f42c5fa3267537ca"
SCHEMA = "n1-hard-neighbor-v10-factorial-diagnostic-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def _source_paths() -> list[Path]:
    runtime_paths = [
        ROOT / "scripts/training/eval_taiji_n1_hard_neighbor_v9.py",
        ROOT / "scripts/training/eval_taiji_n1_transition_factorial_confirmation.py",
        ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py",
        ROOT / "scripts/training/run_taiji_n1_relation_paired.py",
        ROOT / "scripts/training/run_taiji_n1_native_evidence_paired.py",
        ROOT / "scripts/training/audit_taiji_n1_transition_factorial_paired.py",
        ROOT / "scripts/training/build_taiji_n1_fresh_paired_admission.py",
        ROOT / "taiji/contracts.py",
        ROOT / "taiji/fabric.py",
        ROOT / "taiji/native_evidence_candidate.py",
    ]
    return [
        PLAN,
        Path(__file__).resolve(),
        EVALUATOR,
        CONTRACT_TEST,
        Path(v9.__file__).resolve(),
        Path(train_data.__file__).resolve(),
        Path(training.__file__).resolve(),
        *training.source_paths(),
        *runtime_paths,
        train_data.MANIFEST,
        training.REPORT,
        training.SCHEDULE,
        v9.V5,
        v9.V6,
        v9.V7,
        v9.V7_RANGE,
        v9.V7_RANGE_ATTEMPT,
        v9.V8,
        v9.V8_RANGE,
        v9.V8_RANGE_ATTEMPT,
        v9.FACE,
    ]


def collect_exclusions() -> tuple[set[str], set[str], dict, dict[str, str]]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-105 bytes differ from the frozen approved protocol")
    if sha256(v9.FACE) != V9_FACE_SHA256:
        raise ValueError("the consumed v9 face differs from the approved exclusion source")
    if sha256(v9.V5) != v9.EXPECTED_SHA256["v5"] or sha256(v9.V6) != v9.EXPECTED_SHA256["v6"]:
        raise ValueError("v5/v6 face bytes differ from the frozen exclusion sources")

    training_report = v9._training_precondition()
    train_items, _ = train_data.train_items_from_frozen_manifest()
    v5 = v9.read_json(v9.V5)
    v6 = v9.read_json(v9.V6)
    excluded_names, excluded_words = v9.exclusions_from_faces(v5, v6)
    excluded_names.update(name for item in train_items for name in item["entities"])
    excluded_words.update(
        line.removesuffix(".").split()[-1]
        for item in train_items
        for line in item["observations"]
        if " uses " in line
    )

    v7_names, v7_words, v7_provenance = v9.exclusions_from_consumed_sacrifice(
        label="v7",
        path=v9.V7,
        range_report_path=v9.V7_RANGE,
        range_attempt_path=v9.V7_RANGE_ATTEMPT,
        confirm_report_path=v9.V7_CONFIRM,
        confirm_attempt_path=v9.V7_CONFIRM_ATTEMPT,
    )
    v8_names, v8_words, v8_provenance = v9.exclusions_from_consumed_sacrifice(
        label="v8",
        path=v9.V8,
        range_report_path=v9.V8_RANGE,
        range_attempt_path=v9.V8_RANGE_ATTEMPT,
        confirm_report_path=v9.V8_CONFIRM,
        confirm_attempt_path=v9.V8_CONFIRM_ATTEMPT,
    )
    excluded_names.update(v7_names | v8_names)
    excluded_words.update(v7_words | v8_words)

    v9_face = v9.read_json(v9.FACE)
    if v9_face.get("schema") != "n1-hard-neighbor-v9-confirmation-v1":
        raise ValueError("the consumed v9 exclusion face has an unexpected schema")
    for episode in v9_face.get("episodes", []):
        excluded_names.update(episode["entities"])
        excluded_words.add(episode["answer"])
        excluded_words.update(
            line.removesuffix(".").split()[-1]
            for line in episode["observations"]
            if " uses " in line
        )

    provenance = {
        "training_report_sha256": sha256(training.REPORT),
        "training_sufficient": training_report["training_sufficient"],
        "training_model_receipts": len(training_report["models"]),
        "v3_train": {
            "manifest_sha256": sha256(train_data.MANIFEST),
            "items": len(train_items),
        },
        "v5_sha256": sha256(v9.V5),
        "v6_sha256": sha256(v9.V6),
        "v7_sacrifice": v7_provenance,
        "v8_sacrifice": v8_provenance,
        "v9_full_consumed_face": {
            "manifest_sha256": sha256(v9.FACE),
            "items": len(v9_face.get("episodes", [])),
            "payload_fields_decoded": True,
        },
        "v7_v8_confirmation_payload_fields_decoded": False,
        "v7_v8_confirmation_artifacts_absent": all(
            not path.exists()
            for path in (v9.V7_CONFIRM, v9.V7_CONFIRM_ATTEMPT, v9.V8_CONFIRM, v9.V8_CONFIRM_ATTEMPT)
        ),
    }
    guards = {str(path.resolve()): sha256(path) for path in _source_paths()}
    return excluded_names, excluded_words, provenance, guards


def _make_episode(
    *,
    item_id: str,
    kind: str,
    template_index: int | None,
    subject: str,
    question: str,
    answer: str,
    target_rows: list[str],
    target_position: int | None,
    answer_length: int | None,
    distractors: list[str],
    filler_rows: list[str],
    replicate_index: int | None = None,
) -> dict:
    if len(distractors) != OBSERVATIONS - len(target_rows) or len(filler_rows) != len(distractors):
        raise ValueError("matched-template filler count differs from the target row count")

    rows: list[str | None] = [None] * OBSERVATIONS
    if target_rows:
        if (
            target_position is None
            or not 1 <= target_position <= OBSERVATIONS - len(target_rows) + 1
        ):
            raise ValueError("target row position is outside the twelve-row face")
        start = target_position - 1
        for offset, row in enumerate(target_rows):
            rows[start + offset] = row
    filler_iter = iter(filler_rows)
    for index, row in enumerate(rows):
        if row is None:
            rows[index] = next(filler_iter)

    episode = {
        "id": item_id,
        "split": "diagnostic",
        "kind": kind,
        "template_index": template_index,
        "replicate_index": replicate_index,
        "entities": [subject, *distractors],
        "observations": [str(row) for row in rows],
        "question": question,
        "answer": answer,
        "target_row_position_1based": target_position,
        "material_answer_length_chars": answer_length,
    }
    if len(episode["observations"]) != OBSERVATIONS:
        raise AssertionError("diagnostic episode does not contain twelve observations")
    return episode


def build_face(*, excluded_names: set[str], excluded_words: set[str], seed: int = SEED) -> dict:
    rng = random.Random(seed)
    used_names = set(excluded_names)
    used_words = set(excluded_words) | {v9.ABSTENTION}
    episodes: list[dict] = []

    for kind in KINDS:
        if kind == "unknown":
            for ordinal in range(UNKNOWN_ITEMS):
                subject = v9._novel(v9.ENTITY_LENGTH, used_names, rng)
                template_index = ordinal // (UNKNOWN_ITEMS // TEMPLATES_PER_KIND)
                distractors = [
                    v9._confusable(subject, used_names, rng) for _ in range(OBSERVATIONS)
                ]
                values = [v9._novel(6, used_words, rng) for _ in distractors]
                filler_rows = [
                    f"{name} uses {value}." for name, value in zip(distractors, values, strict=True)
                ]
                episode = _make_episode(
                    item_id=f"v10-{kind}-{ordinal:02d}",
                    kind=kind,
                    template_index=template_index,
                    replicate_index=ordinal % (UNKNOWN_ITEMS // TEMPLATES_PER_KIND),
                    subject=subject,
                    question=f"material of {subject}?",
                    answer=v9.ABSTENTION,
                    target_rows=[],
                    target_position=None,
                    answer_length=None,
                    distractors=distractors,
                    filler_rows=filler_rows,
                )
                episodes.append(episode)
            continue

        for template_index in range(TEMPLATES_PER_KIND):
            subject = v9._novel(v9.ENTITY_LENGTH, used_names, rng)
            question = (
                f"direction of {subject}?" if kind == "transition" else f"material of {subject}?"
            )
            answers = (
                {length: v9._novel(length, used_words, rng) for length in ANSWER_LENGTHS}
                if kind in {"lookup", "revision"}
                else {}
            )
            stale = v9._novel(6, used_words, rng) if kind == "revision" else None
            target_row_count = {"lookup": 1, "revision": 2, "transition": 2}[kind]
            distractor_count = OBSERVATIONS - target_row_count
            distractors = [
                v9._confusable(subject, used_names, rng) for _ in range(distractor_count)
            ]
            if kind == "transition":
                answer = v9.DIRECTIONS[rng.randrange(len(v9.DIRECTIONS))]
                turn_word = rng.choice(("right", "left"))
                turn = 1 if turn_word == "right" else -1
                start_direction = v9.DIRECTIONS[
                    (v9.DIRECTIONS.index(answer) - turn) % len(v9.DIRECTIONS)
                ]
                filler_rows = [
                    f"{name} faces {v9.DIRECTIONS[rng.randrange(len(v9.DIRECTIONS))]}."
                    for name in distractors
                ]
            else:
                filler_values = [v9._novel(6, used_words, rng) for _ in distractors]
                filler_rows = [
                    f"{name} uses {value}."
                    for name, value in zip(distractors, filler_values, strict=True)
                ]

            positions = range(1, OBSERVATIONS + 1) if kind == "lookup" else range(1, OBSERVATIONS)
            for position in positions:
                lengths: tuple[int | None, ...] = (
                    ANSWER_LENGTHS if kind in {"lookup", "revision"} else (None,)
                )
                for answer_length in lengths:
                    if kind == "lookup":
                        answer = answers[answer_length]
                        target_rows = [f"{subject} uses {answer}."]
                    elif kind == "revision":
                        answer = answers[answer_length]
                        target_rows = [
                            f"{subject} uses {stale}.",
                            f"{subject} now uses {answer}.",
                        ]
                    else:
                        target_rows = [
                            f"{subject} faces {start_direction}.",
                            f"{subject} turns {turn_word}.",
                        ]
                    length_tag = f"l{answer_length}" if answer_length is not None else "lna"
                    item_id = f"v10-{kind}-t{template_index:02d}-p{position:02d}-{length_tag}"
                    episode = _make_episode(
                        item_id=item_id,
                        kind=kind,
                        template_index=template_index,
                        replicate_index=None,
                        subject=subject,
                        question=question,
                        answer=answer,
                        target_rows=target_rows,
                        target_position=position,
                        answer_length=answer_length,
                        distractors=distractors,
                        filler_rows=filler_rows,
                    )
                    episodes.append(episode)

    counts = Counter(episode["kind"] for episode in episodes)
    return {
        "schema": SCHEMA,
        "seed": seed,
        "items": len(episodes),
        "templates_per_kind": TEMPLATES_PER_KIND,
        "observations_per_item": OBSERVATIONS,
        "entity_length": v9.ENTITY_LENGTH,
        "name_neighbor_distance": 1,
        "split": "diagnostic",
        "kind_counts": dict(sorted(counts.items())),
        "factor_design": {
            "lookup_positions_1based": list(range(1, OBSERVATIONS + 1)),
            "revision_positions_1based": list(range(1, OBSERVATIONS)),
            "transition_positions_1based": list(range(1, OBSERVATIONS)),
            "material_answer_lengths_chars": list(ANSWER_LENGTHS),
            "items_per_lookup_revision_cell": TEMPLATES_PER_KIND,
            "items_per_transition_position": TEMPLATES_PER_KIND,
            "unknown_items": UNKNOWN_ITEMS,
        },
        "episodes": episodes,
    }


def audit_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    episodes = face.get("episodes", [])
    ids = [episode.get("id") for episode in episodes]
    kind_counts = Counter(episode.get("kind") for episode in episodes)
    errors: list[str] = []
    target_row_counts = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}

    if face.get("schema") != SCHEMA or face.get("seed") != SEED:
        errors.append("schema_or_seed")
    if face.get("split") != "diagnostic" or face.get("items") != sum(EXPECTED_ITEMS.values()):
        errors.append("face_metadata")
    if len(episodes) != sum(EXPECTED_ITEMS.values()) or len(set(ids)) != len(episodes):
        errors.append("item_count_or_unique_ids")
    if kind_counts != Counter(EXPECTED_ITEMS):
        errors.append("kind_counts")

    expected_order = []
    for kind in KINDS:
        if kind == "unknown":
            expected_order.extend(f"v10-unknown-{ordinal:02d}" for ordinal in range(UNKNOWN_ITEMS))
            continue
        for template_index in range(TEMPLATES_PER_KIND):
            positions = range(1, OBSERVATIONS + 1) if kind == "lookup" else range(1, OBSERVATIONS)
            lengths = ANSWER_LENGTHS if kind in {"lookup", "revision"} else (None,)
            for position in positions:
                for answer_length in lengths:
                    length_tag = f"l{answer_length}" if answer_length is not None else "lna"
                    expected_order.append(
                        f"v10-{kind}-t{template_index:02d}-p{position:02d}-{length_tag}"
                    )
    if ids != expected_order:
        errors.append("item_order")

    seen_distractors: set[str] = set()
    distractor_owner: dict[str, tuple[str, int | str]] = {}
    template_distractors: dict[tuple[str, int | str], tuple[str, ...]] = {}
    template_filler_rows: dict[tuple[str, int | str], tuple[str, ...]] = {}
    seen_filler_values: set[str] = set()
    filler_value_owner: dict[str, tuple[str, int | str]] = {}
    subject_owner: dict[str, tuple[str, int | str]] = {}
    target_subjects: dict[tuple[str, int], str] = {}
    template_questions: dict[tuple[str, int], str] = {}
    answer_by_cell: dict[tuple[str, int, int], str] = {}
    stale_by_template: dict[tuple[str, int], str] = {}
    target_material_values: set[str] = set()
    pair_counts: Counter[tuple[str, int | None, int | None]] = Counter()
    unknown_template_counts: Counter[int] = Counter()

    for episode in episodes:
        item_id = episode.get("id", "<unknown>")
        kind = episode.get("kind")
        if kind not in KINDS or episode.get("split") != "diagnostic":
            errors.append(f"bad_kind_or_split:{item_id}")
            continue
        entities = episode.get("entities")
        observations = episode.get("observations")
        if (
            not isinstance(entities, list)
            or not entities
            or any(not isinstance(name, str) for name in entities)
            or not isinstance(observations, list)
            or len(observations) != OBSERVATIONS
            or any(not isinstance(row, str) for row in observations)
        ):
            errors.append(f"bad_input_shape:{item_id}")
            continue
        template_index = episode.get("template_index")
        if type(template_index) is not int or not 0 <= template_index < TEMPLATES_PER_KIND:
            errors.append(f"bad_template_index:{item_id}")
            continue
        subject, *distractors = entities
        owner_key: tuple[str, int | str] = (
            (kind, template_index) if kind != "unknown" else (kind, item_id)
        )
        if len(entities) != 1 + OBSERVATIONS - target_row_counts[kind]:
            errors.append(f"bad_entity_count:{item_id}")
        if len(set(entities)) != len(entities):
            errors.append(f"duplicate_entity_in_item:{item_id}")
        if any(name in excluded_names for name in entities):
            errors.append(f"excluded_entity:{item_id}")
        previous_owner = subject_owner.setdefault(subject, owner_key)
        if previous_owner != owner_key:
            errors.append(f"subject_reused_across_templates:{item_id}")
        if len(subject) != v9.ENTITY_LENGTH or any(
            len(name) != v9.ENTITY_LENGTH for name in distractors
        ):
            errors.append(f"wrong_entity_length:{item_id}")
        if any(
            sum(left != right for left, right in zip(subject, name, strict=True)) != 1
            for name in distractors
        ):
            errors.append(f"non_d1_distractor:{item_id}")
        distractor_tuple = tuple(distractors)
        if template_distractors.setdefault(owner_key, distractor_tuple) != distractor_tuple:
            errors.append(f"template_distractors_changed:{item_id}")
        for name in distractors:
            previous_distractor_owner = distractor_owner.setdefault(name, owner_key)
            if previous_distractor_owner != owner_key:
                errors.append(f"distractor_reused_between_templates:{item_id}")
            seen_distractors.add(name)

        question = episode.get("question")
        if not isinstance(question, str):
            errors.append(f"bad_question:{item_id}")
        expected_question = (
            f"direction of {subject}?" if kind == "transition" else f"material of {subject}?"
        )
        if question != expected_question:
            errors.append(f"question_semantics_mismatch:{item_id}")
        target_positions = [
            index
            for index, row in enumerate(observations, start=1)
            if row.startswith(subject + " ")
        ]
        position = episode.get("target_row_position_1based")
        answer_length = episode.get("material_answer_length_chars")
        if len(target_positions) != target_row_counts[kind]:
            errors.append(f"wrong_target_row_count:{item_id}")
        if kind != "unknown" and (
            type(position) is not int
            or not 1 <= position <= OBSERVATIONS - target_row_counts[kind] + 1
            or not target_positions
            or target_positions[0] != position
        ):
            errors.append(f"target_position_mismatch:{item_id}")
            continue
        target_indices = set(target_positions)
        current_filler_rows = tuple(
            row for index, row in enumerate(observations, start=1) if index not in target_indices
        )
        if template_filler_rows.setdefault(owner_key, current_filler_rows) != current_filler_rows:
            errors.append(f"template_filler_facts_changed:{item_id}")
        local_filler_values = []
        for row in current_filler_rows:
            if " uses " not in row:
                continue
            value = row.removesuffix(".").split()[-1]
            local_filler_values.append(value)
            previous_value_owner = filler_value_owner.setdefault(value, owner_key)
            if previous_value_owner != owner_key:
                errors.append(f"filler_material_reused_between_templates:{item_id}")
            if len(value) != 6 or value in excluded_words:
                errors.append(f"excluded_or_wrong_length_filler_material:{item_id}")
            seen_filler_values.add(value)
        if len(local_filler_values) != len(set(local_filler_values)):
            errors.append(f"duplicate_filler_value_within_item:{item_id}")

        if kind == "unknown":
            ordinal = int(item_id.rsplit("-", 1)[-1]) if item_id.startswith("v10-unknown-") else -1
            expected_template = ordinal // (UNKNOWN_ITEMS // TEMPLATES_PER_KIND)
            expected_replicate = ordinal % (UNKNOWN_ITEMS // TEMPLATES_PER_KIND)
            if (
                position is not None
                or answer_length is not None
                or episode.get("answer") != v9.ABSTENTION
                or template_index != expected_template
                or episode.get("replicate_index") != expected_replicate
                or question != f"material of {subject}?"
            ):
                errors.append(f"bad_unknown_features:{item_id}")
            unknown_template_counts[template_index] += 1
            continue

        template_key = (kind, template_index)
        if target_subjects.setdefault(template_key, subject) != subject:
            errors.append(f"template_subject_changed:{item_id}")
        if template_questions.setdefault(template_key, question) != question:
            errors.append(f"template_question_changed:{item_id}")
        if episode.get("replicate_index") is not None:
            errors.append(f"unexpected_replicate_index:{item_id}")

        if kind in {"lookup", "revision"}:
            answer = episode.get("answer")
            if (
                not isinstance(answer, str)
                or type(answer_length) is not int
                or len(answer) != answer_length
                or answer_length not in ANSWER_LENGTHS
            ):
                errors.append(f"wrong_answer_length:{item_id}")
                continue
            cell = (kind, template_index, answer_length)
            if answer_by_cell.setdefault(cell, answer) != answer:
                errors.append(f"template_answer_changed:{item_id}")
            if answer in excluded_words:
                errors.append(f"excluded_target_material:{item_id}")
            target_material_values.add(answer)
            pair_counts[(kind, position, answer_length)] += 1
            if kind == "lookup":
                if position is None or observations[position - 1] != f"{subject} uses {answer}.":
                    errors.append(f"lookup_target_mismatch:{item_id}")
            else:
                target_rows = observations[position - 1 : position + 1] if position else []
                if len(target_rows) != 2:
                    errors.append(f"revision_rows_missing:{item_id}")
                    continue
                stale = target_rows[0].removesuffix(".").split()[-1]
                if (
                    target_rows[0] != f"{subject} uses {stale}."
                    or target_rows[1] != f"{subject} now uses {answer}."
                    or stale == answer
                    or len(stale) != 6
                ):
                    errors.append(f"revision_target_mismatch:{item_id}")
                if stale_by_template.setdefault(template_key, stale) != stale:
                    errors.append(f"revision_stale_changed:{item_id}")
                if stale in excluded_words:
                    errors.append(f"excluded_stale_material:{item_id}")
                target_material_values.add(stale)
        else:
            if answer_length is not None or position is None:
                errors.append(f"bad_transition_features:{item_id}")
            pair_counts[(kind, position, None)] += 1
            target_rows = observations[position - 1 : position + 1] if position else []
            if (
                len(target_rows) != 2
                or not target_rows[0].startswith(subject + " faces ")
                or not target_rows[1].startswith(subject + " turns ")
            ):
                errors.append(f"transition_order_mismatch:{item_id}")
            else:
                start_direction = target_rows[0].removesuffix(".").split()[-1]
                turn = target_rows[1].removesuffix(".").split()[-1]
                if start_direction not in v9.DIRECTIONS or turn not in {"left", "right"}:
                    errors.append(f"transition_fact_invalid:{item_id}")
                else:
                    delta = 1 if turn == "right" else -1
                    final = v9.DIRECTIONS[
                        (v9.DIRECTIONS.index(start_direction) + delta) % len(v9.DIRECTIONS)
                    ]
                    if final != episode.get("answer"):
                        errors.append(f"transition_answer_mismatch:{item_id}")

    expected_filler_materials = (
        TEMPLATES_PER_KIND * (OBSERVATIONS - target_row_counts["lookup"])
        + TEMPLATES_PER_KIND * (OBSERVATIONS - target_row_counts["revision"])
        + UNKNOWN_ITEMS * OBSERVATIONS
    )
    if len(seen_filler_values) != expected_filler_materials:
        errors.append("distractor_materials_not_unique")
    if target_material_values & seen_filler_values:
        errors.append("target_and_distractor_material_overlap")
    if target_material_values & excluded_words or seen_filler_values & excluded_words:
        errors.append("material_source_overlap")
    if len(set(target_subjects.values())) != len(target_subjects):
        errors.append("target_templates_share_subject")
    if len(target_subjects) != 3 * TEMPLATES_PER_KIND:
        errors.append("target_template_count")
    if len(subject_owner) != 3 * TEMPLATES_PER_KIND + UNKNOWN_ITEMS:
        errors.append("subject_owner_count")
    if len(target_material_values) != 20:
        errors.append("target_material_values_not_unique")
    if unknown_template_counts != Counter({index: 5 for index in range(TEMPLATES_PER_KIND)}):
        errors.append("unknown_template_balance")

    for kind in ("lookup", "revision"):
        positions = range(1, OBSERVATIONS + 1) if kind == "lookup" else range(1, OBSERVATIONS)
        for position in positions:
            for length in ANSWER_LENGTHS:
                if pair_counts[(kind, position, length)] != TEMPLATES_PER_KIND:
                    errors.append(f"incomplete_cell:{kind}:{position}:{length}")
    for position in range(1, OBSERVATIONS):
        if pair_counts[("transition", position, None)] != TEMPLATES_PER_KIND:
            errors.append(f"incomplete_transition_position:{position}")

    expected_distractor_occurrences = sum(
        EXPECTED_ITEMS[kind] * (OBSERVATIONS - target_row_counts[kind]) for kind in KINDS
    )
    expected_unique_distractors = (
        TEMPLATES_PER_KIND
        * sum(
            OBSERVATIONS - target_row_counts[kind] for kind in ("lookup", "revision", "transition")
        )
        + UNKNOWN_ITEMS * OBSERVATIONS
    )
    checks = {
        "schema_and_seed": face.get("schema") == SCHEMA and face.get("seed") == SEED,
        "diagnostic_split": face.get("split") == "diagnostic",
        "item_count": len(episodes) == sum(EXPECTED_ITEMS.values()),
        "kind_counts": kind_counts == Counter(EXPECTED_ITEMS),
        "unique_ids_and_canonical_order": len(set(ids)) == len(episodes) and ids == expected_order,
        "twelve_observations": all(
            len(item.get("observations", [])) == OBSERVATIONS for item in episodes
        ),
        "target_template_identity": len(target_subjects) == 3 * TEMPLATES_PER_KIND
        and len(set(target_subjects.values())) == len(target_subjects),
        "source_disjointness": not (
            set(excluded_names) & {name for item in episodes for name in item["entities"]}
        )
        and not ((target_material_values | seen_filler_values) & excluded_words),
        "d1_distractor_count": len(seen_distractors) == expected_unique_distractors,
        "d1_distractor_occurrences": expected_distractor_occurrences
        == sum(OBSERVATIONS - target_row_counts[episode["kind"]] for episode in episodes),
        "full_factorial_cells": all(
            pair_counts[(kind, position, length)] == TEMPLATES_PER_KIND
            for kind in ("lookup", "revision")
            for position in (
                range(1, OBSERVATIONS + 1) if kind == "lookup" else range(1, OBSERVATIONS)
            )
            for length in ANSWER_LENGTHS
        ),
        "transition_position_cells": all(
            pair_counts[("transition", position, None)] == TEMPLATES_PER_KIND
            for position in range(1, OBSERVATIONS)
        ),
        "unknown_items": kind_counts["unknown"] == UNKNOWN_ITEMS,
    }
    return {
        "accepted": not errors and all(checks.values()),
        "checks": checks,
        "errors": errors,
        "item_count": len(episodes),
        "kind_counts": dict(sorted(kind_counts.items())),
        "unique_target_templates": len(target_subjects),
        "unique_d1_distractors": len(seen_distractors),
        "d1_distractor_occurrences": expected_distractor_occurrences,
        "target_cell_count": len(pair_counts),
        "source_overlap_names": 0 if checks["source_disjointness"] else None,
        "source_overlap_materials": 0 if checks["source_disjointness"] else None,
        "v7_v8_confirmation_payload_fields_decoded": False,
    }


def create_face() -> dict:
    if FACE.exists() or CHECKS.exists():
        raise FileExistsError("PLAN-N1-105 diagnostic face and checks are write-once")
    started = time.process_time()
    excluded_names, excluded_words, provenance, source_guards = collect_exclusions()
    face = build_face(excluded_names=excluded_names, excluded_words=excluded_words)
    face["exclusion_sources"] = provenance
    face["construction"] = {
        "plan_sha256": sha256(PLAN),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_guards": source_guards,
    }
    report = audit_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
    if not report["accepted"]:
        raise ValueError(f"PLAN-N1-105 construction checks failed: {report['errors']}")
    if not provenance["v7_v8_confirmation_artifacts_absent"]:
        raise ValueError("a v7/v8 confirmation artifact appeared during source exclusion")
    if not all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in source_guards.items()
    ):
        raise ValueError("an exclusion source changed during diagnostic face construction")
    cpu_seconds = time.process_time() - started
    encoded_face = (
        json.dumps(face, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    checks = {
        "schema": "n1-hard-neighbor-v10-factorial-diagnostic-checks-v1",
        "status": "complete",
        "complete": True,
        "accepted": True,
        "face_sha256": hashlib.sha256(encoded_face.encode("utf-8")).hexdigest(),
        "plan_sha256": sha256(PLAN),
        "builder_sha256": sha256(Path(__file__).resolve()),
        "source_guards": source_guards,
        "exclusion_sources": provenance,
        "audit": report,
        "cpu_seconds": round(cpu_seconds, 6),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "can_promote": False,
    }
    with FACE.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded_face)
    checks["face_file_sha256"] = sha256(FACE)
    write_new(CHECKS, checks)
    return checks


def main() -> int:
    result = create_face()
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete": result["complete"],
                "face": str(FACE),
                "face_sha256": result["face_file_sha256"],
                "checks": str(CHECKS),
                "cpu_seconds": result["cpu_seconds"],
                "model_queries": 0,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
