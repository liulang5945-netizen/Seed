"""构造PLAN-N1-150 train-only长度混合续训schedule。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v26_abstention_length_after_n148 as n149_builder
from scripts.training import eval_taiji_n1_v26_abstention_length_after_n148 as n149_eval
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import seal_ast

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-150_matched_long_context_training_continuation_draft_20261020.md"
)
PLAN_SHA256 = "21416667c74656a50f21261d2764794910a6366e25fbdeac86eb87981cc768ed"
SCORING_ADDENDUM = (
    ROOT / "plans/reference/PLAN-N1-150A_train_only_scoring_denominator_freeze_20261020.md"
)
SCORING_ADDENDUM_SHA256 = "13bdec809f39820b94fdc8773db091a1d08091404037c703ed628d8748bf8d09"
RESOURCE_ADDENDUM = ROOT / "plans/reference/PLAN-N1-150B_training_cpu_ceiling_revision_20261020.md"
RESOURCE_ADDENDUM_SHA256 = "e690f71df597b5e16d3b53a48229fc686c3e12a0b081cfff1d65330ebe12ff7e"
PRECHECK = ROOT / "reports/taiji_n1_plan150_length_source_precheck_20261020.json"
SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_plan150_v27_training_exclusions_20261020.json"
SCHEDULE = ROOT / "plans/manifests/n1_plan150_v27_length_schedule_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan150_v27_length_readiness_20261020.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_plan150_v27_length_readiness_attempt_20261020.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan150_v27_length_source_binding_20261020.json"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/n1_plan150_v27_length_robust_training_seed126_20261020"
PREFLIGHT = ROOT / "reports/taiji_n1_plan150_v27_length_checkpoint_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan150_v27_length_checkpoint_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan150_v27_length_checkpoint_preflight_binding_20261020.json"
)
ATTEMPT = ROOT / "reports/taiji_n1_plan150_v27_length_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan150_v27_length_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan150_v27_length_result_20261020.json"

N149_MANIFEST = n149_builder.SOURCE_EXCLUSIONS
N149_FACE = n149_builder.FACE
N149_RESULT = n149_eval.RESULT
N149_RESULT_SHA256 = "878e2457072f81f508312993a10b35e3bb70151b709fa752ede79550d57a4ea8"
N132_RESULT = n132.RESULT

SEEDS = (11, 12, 13)
BRANCHES = ("length_mixed", "eight_row_control")
LENGTHS = (8, 16, 32)
SHORT_CONTROL_LENGTH = 8
UPDATES_PER_PARENT = 48
UPDATES_PER_MODEL = n132.ADDITIONAL_UPDATES
EXPECTED_TOTAL_UPDATES = UPDATES_PER_MODEL * len(SEEDS) * len(BRANCHES)
TRAIN_KINDS = tuple(n132.TRAIN_KINDS)
ENTITY_LENGTH = 10
DISTANCE = 1
CPU_LIMIT_SECONDS = 18_000.0
LETTERS = "abcdefghijklmnopqrstuvwxyz"
ABSTENTION = "unknown"
DIRECTIONS = ("north", "east", "south", "west")


class Refuse(RuntimeError):
    """train/source/schedule合同不成立。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_strings(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _extract_observation_values(item: dict) -> tuple[set[str], set[str]]:
    names = set(str(name) for name in item.get("entities", []))
    words = set()
    for row in item.get("observations", []):
        parts = row.removesuffix(".").split()
        if parts:
            names.add(parts[0])
        if len(parts) == 3 and parts[1] == "uses":
            words.add(parts[2])
        elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
            words.add(parts[3])
        elif len(parts) == 3 and parts[1] in ("faces", "turns"):
            words.add(parts[2])
    if item.get("answer") not in (None, ABSTENTION, *DIRECTIONS):
        words.add(str(item["answer"]))
    return names, words


def _context_key(item: dict) -> str:
    payload = {
        "id": item.get("id"),
        "kind": item.get("kind"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "split": item.get("split"),
        "observations": item.get("observations"),
    }
    return canonical_sha(payload)


def _parent_and_route(item: dict) -> tuple[str, str]:
    match = re.match(r"(raw|hard)-(train-\d+)", str(item.get("id", "")))
    if not match:
        raise Refuse(f"cannot identify raw/hard train parent: {item.get('id')}")
    route, parsed_parent = match.groups()
    parent = str(item.get("parent_id") or parsed_parent)
    if parent != parsed_parent:
        raise Refuse(f"scheduled parent ID mismatch: {item.get('id')}")
    return parent, route


def _base_context_rows(item: dict) -> list[str]:
    observations = list(item.get("observations", []))
    target_indices = n132._target_observation_indices(item)
    if target_indices is None:
        return observations
    target_rows = [observations[index] for index in target_indices]
    other_rows = [row for index, row in enumerate(observations) if index not in set(target_indices)]
    return [*target_rows, *other_rows]


def _generate_context_bank(
    source_items: dict[str, dict],
    excluded_entities: set[str],
    excluded_words: set[str],
    seed: int = 150126,
) -> tuple[dict[str, dict[int, list[str]]], dict]:
    used_entities = set(excluded_entities)
    used_words = set(excluded_words) | {ABSTENTION} | set(DIRECTIONS)
    for item in source_items.values():
        names, words = _extract_observation_values(item)
        used_entities.update(names)
        used_words.update(words)
    initial_entities = set(used_entities)
    initial_words = set(used_words)
    descriptors = {}
    required_fillers = Counter()
    for key in sorted(source_items):
        item = source_items[key]
        if item.get("split") != "train" or item.get("kind") not in TRAIN_KINDS:
            raise Refuse(f"context source is not a supported train item: {item.get('id')}")
        subject = n132._query_subject(item["question"])
        if len(subject) != ENTITY_LENGTH or not subject.isascii() or not subject.isalpha():
            raise Refuse(
                f"train subject is not a {ENTITY_LENGTH}-letter lowercase entity: {item.get('id')}"
            )
        base_rows = _base_context_rows(item)
        if not base_rows or len(base_rows) >= max(LENGTHS):
            raise Refuse(
                f"unexpected original context length for {item.get('id')}: {len(base_rows)}"
            )
        descriptors[key] = (item, subject, base_rows)
        required_fillers[subject] = max(required_fillers[subject], max(LENGTHS) - len(base_rows))

    filler_banks = {}
    for subject in sorted(required_fillers):
        subject_seed = int(hashlib.sha256(subject.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed ^ subject_seed)
        filler_rows = []
        for filler_index in range(required_fillers[subject]):
            entity_count_before = len(used_entities)
            neighbor = n149_builder.n148_builder.n147_builder._confusable(
                subject, used_entities, rng
            )
            if (
                len(neighbor) != ENTITY_LENGTH
                or sum(first != second for first, second in zip(subject, neighbor, strict=True))
                != DISTANCE
                or len(used_entities) != entity_count_before + 1
            ):
                raise Refuse(
                    f"generated distractor is not a unique d={DISTANCE} neighbor: {subject}"
                )
            material_length = 6 if filler_index % 2 == 0 else 7
            word_count_before = len(used_words)
            material = n149_builder.n148_builder.n147_builder._novel(
                material_length, used_words, rng
            )
            if len(material) != material_length or len(used_words) != word_count_before + 1:
                raise Refuse(
                    f"generated training material is not unique or has wrong length: {subject}"
                )
            filler_rows.append(f"{neighbor} uses {material}.")
        filler_banks[subject] = filler_rows

    views = {}
    source_digests = {}
    for key, (_item, subject, base_rows) in descriptors.items():
        filler_rows = filler_banks[subject]
        views[key] = {
            length: [*base_rows, *filler_rows[: length - len(base_rows)]] for length in LENGTHS
        }
        source_digests[key] = {str(length): canonical_sha(views[key][length]) for length in LENGTHS}
    generated_entities = used_entities - initial_entities
    generated_words = used_words - initial_words
    if generated_entities & excluded_entities or generated_words & excluded_words:
        raise Refuse("generated training distractors overlap an excluded evaluation source")
    summary = {
        "context_source_count": len(source_items),
        "unique_question_subject_count": len(filler_banks),
        "generated_distractor_entity_count": len(generated_entities),
        "generated_distractor_word_count": len(generated_words),
        "context_view_count": len(views) * len(LENGTHS),
        "context_bank_sha256": canonical_sha(source_digests),
        "generated_entity_set_sha256": digest_strings(generated_entities),
        "generated_word_set_sha256": digest_strings(generated_words),
    }
    return views, summary


def build_schedule(
    g: dict, inputs: dict, excluded_entities: set[str], excluded_words: set[str]
) -> dict:
    if len(inputs.get("items", [])) != 192 or any(
        item.get("split") != "train" for item in inputs["items"]
    ):
        raise Refuse("N1-150 requires exactly 192 train parents")
    if "length_ledger" not in g:
        raise Refuse("base schedule must expose its length ledger")
    g["length_ledger"].clear()
    slot_counts = Counter()
    route_length_counts = Counter()
    parent_counts = Counter()
    kind_counts = Counter()
    slots = []
    source_items = {_context_key(item): item for item in inputs["items"]}
    binding_queries = 0
    for step in range(UPDATES_PER_MODEL):
        item, second = g["scheduled"](step, inputs["items"], inputs["curriculum"])
        if item.get("split") != "train" or item.get("kind") not in TRAIN_KINDS:
            raise Refuse(f"base scheduler emitted non-train task at step {step}")
        if second is not None and (
            second.get("split") != "train" or second.get("observations") != item.get("observations")
        ):
            raise Refuse(f"paired second query changed source tape at step {step}")
        specs = n132._query_specs(item, second)
        binding_queries += sum(
            n132._target_observation_indices(spec["query"]) is not None for spec in specs
        )
        parent, route = _parent_and_route(item)
        route_key = (parent, route)
        occurrence = slot_counts[route_key]
        slot_counts[route_key] += 1
        parent_counts[parent] += 1
        kind_counts[item["kind"]] += 1
        candidate_length = LENGTHS[occurrence % len(LENGTHS)]
        route_length_counts[(parent, route, candidate_length)] += 1
        item_key = _context_key(item)
        source_items.setdefault(item_key, item)
        second_key = None
        if second is not None:
            second_key = _context_key(second)
        slots.append(
            {
                "step": step,
                "parent_id": parent,
                "route": route,
                "kind": item["kind"],
                "item_key": item_key,
                "item": item,
                "item_id": item["id"],
                "second": second,
                "second_key": second_key,
                "second_id": None if second is None else second["id"],
                "candidate_length": candidate_length,
                "control_length": SHORT_CONTROL_LENGTH,
            }
        )
    if (
        len(parent_counts) != 192
        or set(parent_counts.values()) != {UPDATES_PER_PARENT}
        or set(slot_counts.values()) != {24}
        or any(
            route_length_counts[(parent, route, length)] != 8
            for parent, route in slot_counts
            for length in LENGTHS
        )
        or kind_counts != Counter({kind: 2_304 for kind in TRAIN_KINDS})
        or binding_queries != 11_520
    ):
        raise Refuse("N1-150 fixed parent/route/kind/length slot counts differ")
    views, view_summary = _generate_context_bank(source_items, excluded_entities, excluded_words)
    for slot in slots:
        if slot["item_key"] not in views:
            raise Refuse(f"scheduled item has no context views: {slot['item_id']}")
    slot_rows = [
        {
            "step": row["step"],
            "parent_id": row["parent_id"],
            "route": row["route"],
            "kind": row["kind"],
            "item_key": row["item_key"],
            "item_id": row["item_id"],
            "second_key": row["second_key"],
            "second_id": row["second_id"],
            "second_kind": None if row["second"] is None else row["second"].get("kind"),
            "second_question_sha256": (
                None
                if row["second"] is None
                else hashlib.sha256(row["second"]["question"].encode("utf-8")).hexdigest()
            ),
            "candidate_length": row["candidate_length"],
            "control_length": row["control_length"],
        }
        for row in slots
    ]
    context_hashes = {
        key: {str(length): canonical_sha(view) for length, view in sorted(value.items())}
        for key, value in sorted(views.items())
    }
    manifest = {
        "schema": "n1-plan150-v27-length-schedule-v1",
        "plan_sha256": PLAN_SHA256,
        "seeds": list(SEEDS),
        "branches": list(BRANCHES),
        "updates_per_model": UPDATES_PER_MODEL,
        "updates_by_task_kind": dict(sorted(kind_counts.items())),
        "answerable_binding_queries_per_model": binding_queries,
        "parent_count": len(parent_counts),
        "parent_updates": dict(sorted(parent_counts.items())),
        "route_updates_per_parent": 24,
        "candidate_updates_by_length": {str(length): 3_072 for length in LENGTHS},
        "control_updates_by_length": {"8": UPDATES_PER_MODEL, "16": 0, "32": 0},
        "candidate_parent_route_length_slots": {
            f"{parent}/{route}/{length}": count
            for (parent, route, length), count in sorted(route_length_counts.items())
        },
        "schedule_rows": slot_rows,
        "schedule_sha256": canonical_sha(slot_rows),
        "context_source_hashes": context_hashes,
        "context_bank": view_summary,
        "can_promote": False,
    }
    return {"manifest": manifest, "slots": slots, "views": views, "source_items": source_items}


def validate_schedule_manifest(manifest: dict) -> dict:
    rows = manifest.get("schedule_rows", [])
    if manifest.get("schedule_sha256") != canonical_sha(rows):
        raise Refuse("schedule manifest SHA mismatch")
    if len(rows) != UPDATES_PER_MODEL:
        raise Refuse("N1-150 schedule must have exactly 9,216 slots")
    counts = Counter(row.get("candidate_length") for row in rows)
    if counts != Counter({8: 3_072, 16: 3_072, 32: 3_072}):
        raise Refuse("candidate length slot balance differs")
    if any(row.get("control_length") != 8 for row in rows):
        raise Refuse("control branch must remain at eight rows")
    return {
        "accepted": True,
        "schedule_rows": len(rows),
        "schedule_sha256": manifest["schedule_sha256"],
        "candidate_updates_by_length": dict(counts),
        "control_updates_by_length": manifest["control_updates_by_length"],
        "can_promote": False,
    }


def verify_hashes(source_hashes: dict[str, str]) -> bool:
    """S3 切换: 经 seal_ast 三通道校验（在位字节 / 在位 AST / git 历史取证）。"""
    return seal_ast.verify_hash_dict(source_hashes)[0]


def parent_checkpoints() -> tuple[dict[int, dict], dict[str, str]]:
    result = read_json(N132_RESULT)
    if (
        result.get("complete") is not True
        or result.get("training_sufficient_all_seeds") is not True
        or result.get("training_updates") != 27_648
        or result.get("parameter_updates") != 27_648
        or result.get("dev_queries") != 0
        or result.get("v13_queries") != 0
        or result.get("v14_queries") != 0
        or result.get("confirmation_queries") != 0
        or result.get("can_promote") is not False
    ):
        raise Refuse("N1-132 full-bundle position parents are not all train-sufficient")
    parents = {
        int(row["seed"]): row
        for row in result.get("models", [])
        if row.get("arm") == "position" and row.get("training_sufficient") is True
    }
    if set(parents) != set(SEEDS):
        raise Refuse("all three N1-132 position parent checkpoints are required")
    hashes = {}
    for seed, row in parents.items():
        path = Path(row["final_checkpoint_path"]).resolve()
        digest = str(row["final_checkpoint_sha256"])
        if not path.is_file() or sha256(path) != digest:
            raise Refuse(f"N1-132 seed{seed} checkpoint changed: {path}")
        hashes[str(path)] = digest
    return parents, hashes


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        previous = target.get(path)
        if previous is not None and previous != digest:
            raise Refuse(f"conflicting source hashes: {path}")
        target[path] = digest


def collect_source_hashes(train_context: dict, parent_hashes: dict[str, str]) -> dict[str, str]:
    n132_result = read_json(N132_RESULT)
    n149_result = read_json(N149_RESULT)
    hashes = dict(train_context.get("source_hashes", {}))
    _merge_hashes(hashes, n132_result.get("source_hashes", {}))
    _merge_hashes(hashes, n149_result.get("source_hashes", {}))
    _merge_hashes(hashes, parent_hashes)
    paths = (
        PLAN,
        SCORING_ADDENDUM,
        RESOURCE_ADDENDUM,
        Path(__file__).resolve(),
        Path(__file__).resolve().with_name("run_taiji_n1_v27_length_robust_training_after_n149.py"),
        ROOT / "tests/taiji_native/test_n1_plan150_v27_length_robust_training_contract.py",
        N132_RESULT,
        n132.ATTEMPT,
        n132.PROGRESS,
        n132.READINESS,
        n132.SOURCE_BINDING,
        n132.PREFLIGHT,
        n132.PREFLIGHT_BINDING,
        N149_MANIFEST,
        N149_FACE,
        N149_RESULT,
        n149_eval.ATTEMPT,
        n149_eval.PROGRESS,
        n149_eval.READINESS,
        n149_eval.SOURCE_BINDING,
        ROOT / "plans/reference/PLAN-N1-149_execution_closeout_20261020.md",
        ROOT / "reports/taiji_n1_v26_abstention_length_diagnostic_20261019.json",
    )
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise Refuse(f"frozen N1-132/N1-149 source missing: {path}")
        _merge_hashes(hashes, {str(path): sha256(path)})
    if not verify_hashes(n132_result.get("source_hashes", {})):
        raise Refuse("N1-132 source lineage changed")
    if not verify_hashes(n149_result.get("source_hashes", {})):
        raise Refuse("N1-149 source/checkpoint lineage changed")
    return hashes


def prepare_readiness() -> tuple[dict, dict, dict, dict]:
    """Read train-only source and build deterministic context/slot contracts; no model load."""
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-150 plan hash changed")
    require(
        sha256(SCORING_ADDENDUM) == SCORING_ADDENDUM_SHA256,
        "PLAN-N1-150A scoring addendum hash changed",
    )
    require(
        sha256(RESOURCE_ADDENDUM) == RESOURCE_ADDENDUM_SHA256,
        "PLAN-N1-150B resource addendum hash changed",
    )
    require(sha256(N149_RESULT) == N149_RESULT_SHA256, "N1-149 result hash changed")
    n149_manifest = read_json(N149_MANIFEST)
    n149_result = read_json(N149_RESULT)
    if (
        n149_manifest.get("schema") != "n1-plan149-v26-source-exclusions-v1"
        or n149_manifest.get("entity_count") != 39_442
        or n149_manifest.get("word_value_count") != 16_608
        or n149_result.get("complete") is not True
        or n149_result.get("diagnostic_queries") != 2_880
        or len(n149_result.get("diagnostic_rows", [])) != 2_880
        or n149_result.get("source_hashes_unchanged") is not True
        or n149_result.get("can_promote") is not False
    ):
        raise Refuse("N1-149 diagnostic/source exclusion is incomplete")
    parents, parent_hashes = parent_checkpoints()
    bound = n132._build_context(load_inputs=True)
    g, inputs = bound["g"], bound["inputs"]
    schedule = build_schedule(
        g,
        inputs,
        set(n149_manifest["excluded_entities"]),
        set(n149_manifest["excluded_word_values"]),
    )
    validate_schedule_manifest(schedule["manifest"])
    hashes = collect_source_hashes(bound, parent_hashes)
    require(verify_hashes(hashes), "source/checkpoint hash changed during train schedule readiness")
    return bound, parents, schedule, hashes


def source_readiness() -> dict:
    outputs = (
        PRECHECK,
        SOURCE_EXCLUSIONS,
        SCHEDULE,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        ATTEMPT,
        PROGRESS,
        RESULT,
    )
    if any(path.exists() for path in outputs) or CHECKPOINT_ROOT.exists():
        raise FileExistsError(
            "PLAN-N1-150 readiness, schedule, checkpoint, and train outputs are write-once"
        )
    started = time.process_time()
    bound, parents, schedule, source_hashes = prepare_readiness()
    n149_manifest = read_json(N149_MANIFEST)
    precheck = {
        "schema": "n1-plan150-v27-length-source-precheck-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "source_hashes": source_hashes,
        "source_hashes_unchanged": True,
        "n132_parent_checkpoint_hashes": {
            str(Path(row["final_checkpoint_path"]).resolve()): row["final_checkpoint_sha256"]
            for row in parents.values()
        },
        "n149_exclusion_counts": {
            "entities": len(n149_manifest["excluded_entities"]),
            "word_values": len(n149_manifest["excluded_word_values"]),
        },
        "schedule": schedule["manifest"],
        "cpu_seconds": round(time.process_time() - started, 6),
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(PRECHECK, precheck)
    exclusions = {
        "schema": "n1-plan150-v27-training-exclusions-v1",
        "status": "ready_for_train_only_context_generation",
        "plan_sha256": PLAN_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "seed": 126,
        "entity_count": len(n149_manifest["excluded_entities"]),
        "word_value_count": len(n149_manifest["excluded_word_values"]),
        "entity_exclusion_sha256": n149_manifest["entity_exclusion_sha256"],
        "word_exclusion_sha256": n149_manifest["word_exclusion_sha256"],
        "excluded_entities": n149_manifest["excluded_entities"],
        "excluded_word_values": n149_manifest["excluded_word_values"],
        "source_hashes": source_hashes,
        "can_promote": False,
    }
    write_new(SOURCE_EXCLUSIONS, exclusions)
    schedule_manifest = dict(
        schedule["manifest"],
        source_exclusion_sha256=sha256(SOURCE_EXCLUSIONS),
        resource_addendum_sha256=RESOURCE_ADDENDUM_SHA256,
    )
    write_new(SCHEDULE, schedule_manifest)
    complete_sources = dict(source_hashes)
    for artifact in (PRECHECK, SOURCE_EXCLUSIONS, SCHEDULE):
        complete_sources[str(artifact.resolve())] = sha256(artifact)
    sources_unchanged = verify_hashes(complete_sources)
    cpu = time.process_time() - started
    complete = sources_unchanged and cpu <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan150-v27-length-source-readiness-v1",
        "status": "ready_for_zero_checkpoint_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
        "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
        "precheck_sha256": sha256(PRECHECK),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "schedule_sha256": sha256(SCHEDULE),
        "schedule_identity_sha256": schedule_manifest["schedule_sha256"],
        "source_hashes": complete_sources,
        "source_hashes_unchanged": sources_unchanged,
        "n132_parent_checkpoint_hashes": {
            str(Path(row["final_checkpoint_path"]).resolve()): row["final_checkpoint_sha256"]
            for row in parents.values()
        },
        "n132_parent_count": len(parents),
        "schedule_summary": {
            "slots_per_model": UPDATES_PER_MODEL,
            "candidate_updates_by_length": schedule_manifest["candidate_updates_by_length"],
            "control_updates_by_length": schedule_manifest["control_updates_by_length"],
            "parent_count": schedule_manifest["parent_count"],
            "updates_by_task_kind": schedule_manifest["updates_by_task_kind"],
            "answerable_binding_queries_per_model": schedule_manifest[
                "answerable_binding_queries_per_model"
            ],
            "total_updates": EXPECTED_TOTAL_UPDATES,
            "train_score_queries_planned": 7_488,
        },
        "cpu_seconds": round(cpu, 6),
        "cumulative_cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "summary_zh": (
            "192条train父项的长度schedule合同通过；candidate 8/16/32等量、control固定8行，0模型/checkpoint加载、0训练、0查询，可进入zero checkpoint preflight。"
            if complete
            else "schedule/source readiness停止；不训练、不查询、不重试。"
        ),
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan150-v27-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "cpu_seconds": round(cpu, 6),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan150-v27-length-readiness-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "scoring_addendum_sha256": SCORING_ADDENDUM_SHA256,
            "resource_addendum_sha256": RESOURCE_ADDENDUM_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": complete_sources,
            "parent_checkpoint_hashes": report["n132_parent_checkpoint_hashes"],
            "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
            "schedule_sha256": sha256(SCHEDULE),
            "can_promote": False,
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="PLAN-N1-150 train-only length schedule readiness")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        parser.error("one-shot invocation requires --validate-only")
    result = source_readiness()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "cpu_seconds",
                    "parent_count",
                    "schedule_summary",
                    "model_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
