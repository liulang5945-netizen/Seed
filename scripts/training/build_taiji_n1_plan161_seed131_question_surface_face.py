"""构造PLAN-N1-161 seed131未见问句句式迁移诊断面。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan153_v28_fresh_context_length_face as n153
from scripts.training import build_taiji_n1_plan160_seed130_candidate_parent_face as n160
from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-161_unseen_question_surface_diagnostic_20261020.md"
PLAN_SHA256 = "07ac97d0b96dc8dc0e17f3e7df7aa85b7df8211b2e31ff75f35872119ca567d3"
N160A_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-160A_execution_closeout_20261020.md"
N160A_RESULT = n160.RESULT
N160A_RESULT_SHA256 = "e499fb690e11334a7e798629ea06f8275febb21b67bb2cc1bccb4543add579dc"
N160A_ATTEMPT = n160.ATTEMPT
N160A_PROGRESS = n160.PROGRESS
N160A_PROGRESS_SHA256 = "bc928dc7cdda33a90e146d722f7c888b9e65bbdc9310ce35f1cbdfd8e7d0d1dc"
N150_RESULT = n160.n159_builder.N150_RESULT
TRAIN_MANIFESTS = (
    ROOT / "plans/manifests/n1_native_evidence_curriculum_v3_20261015.json",
    ROOT / "plans/manifests/n1_paired_questions_20261015.json",
)
SOURCE_AUDIT = ROOT / "reports/taiji_n1_plan161_question_surface_source_audit_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan161_question_surface_readiness_20261020.json"
READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan161_question_surface_readiness_attempt_20261020.json"
)
SOURCE_BINDING = ROOT / "reports/taiji_n1_plan161_question_surface_source_binding_20261020.json"
PREFLIGHT = ROOT / "reports/taiji_n1_plan161_question_surface_preflight_20261020.json"
PREFLIGHT_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan161_question_surface_preflight_attempt_20261020.json"
)
PREFLIGHT_BINDING = (
    ROOT / "reports/taiji_n1_plan161_question_surface_preflight_binding_20261020.json"
)
FACE = ROOT / "plans/manifests/n1_plan161_v33_question_surface_seed131_20261020.json"
FACE_AUDIT = ROOT / "reports/taiji_n1_plan161_question_surface_face_audit_20261020.json"
FACE_CHECKS = ROOT / "reports/taiji_n1_plan161_question_surface_face_checks_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan161_question_surface_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan161_question_surface_progress_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan161_question_surface_result_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan161_seed131_question_surface_diagnostic_contract.py"

SEED = 131
ITEMS_PER_KIND = 40
LENGTHS = (8, 16, 32)
KINDS = ("lookup", "revision", "transition", "unknown")
SEEDS = (11, 12, 13)
BRANCHES = ("unknown8_rebalanced", "length_mixed_parent")
ENDPOINTS = tuple((seed, branch) for seed in SEEDS for branch in BRANCHES)
QUERIES_PER_ENDPOINT = ITEMS_PER_KIND * len(KINDS) * len(LENGTHS)
TOTAL_QUERIES = QUERIES_PER_ENDPOINT * len(ENDPOINTS)
QUESTION_TEMPLATES = {
    "material_active": "what material does {entity} use?",
    "material_passive": "which material is used by {entity}?",
    "direction_after_turning": "which direction does {entity} face after turning?",
    "direction_after_turn": "what direction is {entity} facing after its turn?",
}
MATERIAL_KINDS = frozenset(("lookup", "revision", "unknown"))
CPU_LIMIT_SECONDS = 3_600.0
LINKED_CPU_LIMIT_SECONDS = 18_000.0
PRIOR_LINKED_CPU_SECONDS = 10_700.171875


class Refuse(RuntimeError):
    """PLAN-N1-161来源、题面或合同错误。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def write_jsonl(stream, value: dict) -> None:
    json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _merge_hashes(target: dict[str, str], incoming: dict[str, str]) -> None:
    for raw_path, digest in incoming.items():
        path = str(Path(raw_path).resolve())
        if (
            path in target
            and target[path] != digest
            and not seal_ast.recorded_value_attested(path, target[path])
        ):
            raise Refuse(f"conflicting source SHA: {path}")
        target[path] = digest


def question_template_id(kind: str, index: int) -> str:
    if kind in MATERIAL_KINDS:
        return "material_active" if index % 2 == 0 else "material_passive"
    if kind == "transition":
        return "direction_after_turning" if index % 2 == 0 else "direction_after_turn"
    raise Refuse(f"unsupported question-surface kind: {kind}")


def render_question(template_id: str, entity: str) -> str:
    try:
        return QUESTION_TEMPLATES[template_id].format(entity=entity)
    except KeyError as exc:
        raise Refuse(f"unknown question template: {template_id}") from exc


def _questions(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "question" and isinstance(child, str):
                yield child
            yield from _questions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _questions(child)


def _template_regexes() -> dict[str, re.Pattern]:
    marker = "__ENTITY__"
    return {
        template_id: re.compile(
            "^"
            + re.escape(template.format(entity=marker)).replace(re.escape(marker), r"[a-z]{10}")
            + "$"
        )
        for template_id, template in QUESTION_TEMPLATES.items()
    }


def audit_training_question_surface(source_hashes: dict[str, str]) -> dict:
    n150 = read_json(N150_RESULT)
    manifest_hashes = {}
    questions = []
    regexes = _template_regexes()
    for path in TRAIN_MANIFESTS:
        resolved = str(path.resolve())
        expected = n150.get("source_hashes", {}).get(resolved)
        require(expected is not None, f"training manifest is not bound by N1-150: {path}")
        require(sha256(path) == expected, f"training manifest SHA changed: {path}")
        require(
            source_hashes.get(resolved) == expected,
            f"training manifest missing from source union: {path}",
        )
        manifest_hashes[resolved] = expected
        questions.extend(_questions(read_json(path)))
    collisions = {
        template_id: sum(regex.fullmatch(question) is not None for question in questions)
        for template_id, regex in regexes.items()
    }
    require(
        not any(collisions.values()), f"question template overlaps training questions: {collisions}"
    )
    return {
        "complete": True,
        "training_manifest_hashes": manifest_hashes,
        "training_question_count": len(questions),
        "unseen_question_template_matches": collisions,
    }


def source_context() -> tuple[list[dict], dict[str, str], dict]:
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-161 SHA changed")
    n160a = read_json(N160A_RESULT)
    require(
        sha256(N160A_RESULT) == N160A_RESULT_SHA256
        and sha256(N160A_PROGRESS) == N160A_PROGRESS_SHA256
        and n160a.get("complete") is True
        and n160a.get("fresh_queries") == 2_880
        and n160a.get("new_fresh_queries") == 2_400
        and n160a.get("source_hashes_unchanged") is True
        and n160a.get("parameter_updates") == 0
        and n160a.get("dev_queries") == 0
        and n160a.get("holdout_queries") == 0
        and n160a.get("confirmation_queries") == 0
        and n160a.get("can_promote") is False,
        "N1-160A completed source evidence is invalid",
    )
    endpoints, sources, exclusions = n160.source_context()
    require(len(endpoints) == len(ENDPOINTS), "N1-161 requires the frozen six endpoints")
    _merge_hashes(sources, n160a.get("source_hashes", {}))
    for path in (
        PLAN,
        N160A_CLOSEOUT,
        N160A_RESULT,
        N160A_ATTEMPT,
        N160A_PROGRESS,
        n160.FACE,
        n160.FACE_AUDIT,
        n160.FACE_CHECKS,
        *TRAIN_MANIFESTS,
        Path(__file__).resolve(),
        ROOT / "scripts/training/run_taiji_n1_plan161_seed131_question_surface_diagnostic.py",
        TEST,
    ):
        path = Path(path).resolve()
        require(path.is_file(), f"PLAN-N1-161 source missing: {path}")
        _merge_hashes(sources, {str(path): sha256(path)})

    excluded_entities = set(exclusions["excluded_entities"])
    excluded_words = set(exclusions["excluded_words"])
    face130_entities, face130_words = n160.n159_builder._face_values(read_json(n160.FACE))
    excluded_entities.update(face130_entities)
    excluded_words.update(face130_words)
    training_audit = audit_training_question_surface(sources)
    if not n160.n150_builder.verify_hashes(sources):
        raise Refuse("N1-161 source/checkpoint SHA changed")
    summary = {
        "prior_face_seeds": [128, 129, 130],
        "seed130_face_sha256": sha256(n160.FACE),
        "seed130_entity_count": len(face130_entities),
        "seed130_word_value_count": len(face130_words),
        "excluded_entity_count": len(excluded_entities),
        "excluded_word_value_count": len(excluded_words),
        "entity_exclusion_sha256": n153.digest_strings(excluded_entities),
        "word_exclusion_sha256": n153.digest_strings(excluded_words),
        "training_question_surface_audit": training_audit,
    }
    return (
        endpoints,
        dict(sorted(sources.items())),
        {
            "excluded_entities": excluded_entities,
            "excluded_words": excluded_words,
            "summary": summary,
        },
    )


def source_readiness() -> dict:
    outputs = (
        SOURCE_AUDIT,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        PREFLIGHT,
        PREFLIGHT_ATTEMPT,
        PREFLIGHT_BINDING,
        FACE,
        FACE_AUDIT,
        FACE_CHECKS,
        ATTEMPT,
        PROGRESS,
        RESULT,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-161 outputs are write-once")
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    cpu = time.process_time() - started
    linked = PRIOR_LINKED_CPU_SECONDS + cpu
    complete = (
        len(endpoints) == len(ENDPOINTS)
        and n160.n150_builder.verify_hashes(sources)
        and cpu <= CPU_LIMIT_SECONDS
        and linked <= LINKED_CPU_LIMIT_SECONDS
    )
    audit = {
        "schema": "n1-plan161-question-surface-source-audit-v1",
        "status": "complete" if complete else "stopped",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": sources,
        "source_hashes_unchanged": n160.n150_builder.verify_hashes(sources),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "question_templates": QUESTION_TEMPLATES,
        "question_surface_audit": exclusions["summary"]["training_question_surface_audit"],
        "source_exclusion_summary": exclusions["summary"],
        "face_spec": {
            "seed": SEED,
            "items_per_kind": ITEMS_PER_KIND,
            "kinds": list(KINDS),
            "lengths": list(LENGTHS),
            "template_ids": {
                kind: [question_template_id(kind, 0), question_template_id(kind, 1)]
                for kind in KINDS
            },
            "total_queries": TOTAL_QUERIES,
        },
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(linked, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(SOURCE_AUDIT, audit)
    prepared = dict(sources)
    prepared[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    readiness = {
        "schema": "n1-plan161-question-surface-readiness-v1",
        "status": "ready_for_six_bundle_preflight" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "source_audit_sha256": sha256(SOURCE_AUDIT),
        "source_hashes": prepared,
        "source_hashes_unchanged": n160.n150_builder.verify_hashes(prepared),
        "endpoint_count": len(endpoints),
        "endpoints": endpoints,
        "question_templates": QUESTION_TEMPLATES,
        "question_surface_audit": exclusions["summary"]["training_question_surface_audit"],
        "source_exclusion_summary": exclusions["summary"],
        "face_spec": audit["face_spec"],
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cpu_seconds": round(cpu, 6),
        "cumulative_linked_cpu_seconds": round(linked, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "can_promote": False,
    }
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan161-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "cpu_seconds": round(cpu, 6),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
        },
    )
    write_new(READINESS, readiness)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan161-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": prepared,
            "can_promote": False,
        },
    )
    return readiness


def _make_face() -> tuple[dict, dict]:
    endpoints, sources, exclusions = source_context()
    face, generator = n153.build_face(
        exclusions["excluded_entities"], exclusions["excluded_words"], seed=SEED
    )
    per_template = Counter()
    for item in face["items"]:
        original_id = item["id"]
        index = int(original_id.rsplit("-", 1)[1])
        kind = str(item["kind"])
        template_id = question_template_id(kind, index)
        target = item["entities"][0]
        item["id"] = original_id.replace("v28-", "v33-", 1)
        item["parent_id"] = item["id"]
        item["question_template_id"] = template_id
        item["question"] = render_question(template_id, target)
        per_template[(kind, template_id)] += 1
    manifest = {
        **face,
        "schema": "n1-plan161-v33-question-surface-face-v1",
        "plan161_schema": "n1-plan161-v33-question-surface-face-v1",
        "plan_sha256": PLAN_SHA256,
        "seed": SEED,
        "question_templates": QUESTION_TEMPLATES,
        "question_template_counts": {
            f"{kind}/{template_id}": count
            for (kind, template_id), count in sorted(per_template.items())
        },
        "source_exclusion_summary": exclusions["summary"],
        "prior_seed128_face_sha256": n160.FACE128_SHA256,
        "prior_seed129_face_sha256": n160.FACE129_SHA256,
        "prior_seed130_face_sha256": n160.PRIOR_FACE_SHA256,
        "endpoint_count": len(endpoints),
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    contract = validate_face(
        manifest, exclusions["excluded_entities"], exclusions["excluded_words"]
    )
    return manifest, {"base_generator": generator, "face_contract": contract}


def validate_face(manifest: dict, excluded_entities: set[str], excluded_words: set[str]) -> dict:
    items = manifest.get("items", [])
    require(
        manifest.get("plan161_schema") == "n1-plan161-v33-question-surface-face-v1",
        "N1-161 face schema differs",
    )
    require(
        manifest.get("plan_sha256") == PLAN_SHA256 and manifest.get("seed") == SEED,
        "N1-161 face plan/seed differs",
    )
    require(len(items) == ITEMS_PER_KIND * len(KINDS) == 160, "N1-161 requires 160 items")
    require(manifest.get("total_queries") == TOTAL_QUERIES, "N1-161 query denominator differs")
    kind_counts = Counter(item.get("kind") for item in items)
    require(
        kind_counts == Counter({kind: ITEMS_PER_KIND for kind in KINDS}),
        "N1-161 kind balance differs",
    )
    template_counts = Counter()
    all_entities = []
    all_words = []
    ids = set()
    for item in items:
        item_id = str(item["id"])
        require(item_id not in ids, f"duplicate N1-161 item id: {item_id}")
        ids.add(item_id)
        require(item.get("split") == "diagnostic", f"invalid split: {item_id}")
        target = item["entities"][0]
        index = int(item_id.rsplit("-", 1)[1])
        template_id = question_template_id(str(item["kind"]), index)
        require(
            item.get("question_template_id") == template_id,
            f"template assignment differs: {item_id}",
        )
        require(
            item.get("question") == render_question(template_id, target),
            f"question template differs: {item_id}",
        )
        require(
            item["question"].count(target) == 1, f"target entity is not uniquely named: {item_id}"
        )
        require(
            not any(name != target and name in item["question"] for name in item["entities"]),
            f"distractor leaks into question: {item_id}",
        )
        template_counts[(str(item["kind"]), template_id)] += 1
        require(
            item.get("target_observation_indices")
            == (
                {"lookup": [0], "revision": [0, 1], "transition": [0, 1], "unknown": []}[
                    item["kind"]
                ]
            ),
            f"target position contract differs: {item_id}",
        )
        rows8, rows16, rows32 = (item["observations_by_length"][str(length)] for length in LENGTHS)
        require(
            len(rows8) == 8 and len(rows16) == 16 and len(rows32) == 32,
            f"context lengths differ: {item_id}",
        )
        require(rows8 == rows16[:8] == rows32[:8], f"contexts are not nested: {item_id}")
        target_rows = item["target_observation_indices"]
        if item["kind"] == "lookup":
            require(
                rows32[0] == f"{target} uses {item['answer']}.",
                f"lookup answer evidence differs: {item_id}",
            )
        elif item["kind"] == "revision":
            require(
                rows32[0].startswith(f"{target} uses ")
                and rows32[1] == f"{target} now uses {item['answer']}.",
                f"revision evidence differs: {item_id}",
            )
        elif item["kind"] == "transition":
            require(
                rows32[0].startswith(f"{target} faces ")
                and rows32[1].startswith(f"{target} turns "),
                f"transition evidence differs: {item_id}",
            )
        else:
            require(
                not any(row.startswith(target + " ") for row in rows32),
                f"unknown target unexpectedly appears in evidence: {item_id}",
            )
        require(
            len(target_rows)
            == (0 if item["kind"] == "unknown" else (1 if item["kind"] == "lookup" else 2)),
            f"target evidence count differs: {item_id}",
        )
        for entity in item["entities"]:
            all_entities.append(entity)
        for row in rows32:
            entity = row.removesuffix(".").split()[0]
            if entity != target:
                require(
                    n153._hamming(target, entity) == 1,
                    f"distractor distance differs: {item_id}/{entity}",
                )
            parts = row.removesuffix(".").split()
            if len(parts) == 3 and parts[1] == "uses":
                all_words.append(parts[2])
            elif len(parts) == 4 and parts[1:3] == ["now", "uses"]:
                all_words.append(parts[3])
    expected_templates = {
        (kind, question_template_id(kind, index)): 20 for kind in KINDS for index in (0, 1)
    }
    require(template_counts == Counter(expected_templates), "N1-161 template balance differs")
    serialized_template_counts = {
        f"{kind}/{template_id}": count
        for (kind, template_id), count in sorted(template_counts.items())
    }
    require(
        manifest.get("question_template_counts") == serialized_template_counts,
        "N1-161 manifest template counts differ",
    )
    require(len(all_entities) == len(set(all_entities)), "N1-161 face repeats an entity")
    require(len(all_words) == len(set(all_words)), "N1-161 face repeats an observation value")
    face_entities = set(all_entities)
    face_words = set(all_words)
    require(not face_entities & excluded_entities, "N1-161 face entity overlaps prior source")
    require(not face_words & excluded_words, "N1-161 face word overlaps prior source")
    return {
        "accepted": True,
        "base_item_count": len(items),
        "items_by_kind": dict(sorted(kind_counts.items())),
        "question_template_counts": {
            f"{k}/{t}": n for (k, t), n in sorted(template_counts.items())
        },
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "total_queries": TOTAL_QUERIES,
        "face_entity_count": len(face_entities),
        "face_word_count": len(face_words),
        "nested_contexts": True,
        "target_evidence_positions_preserved": True,
        "source_overlap": False,
        "can_promote": False,
    }


def prepare_face() -> dict:
    if any(path.exists() for path in (FACE, FACE_AUDIT, FACE_CHECKS)):
        raise FileExistsError("N1-161 face artifacts are write-once")
    for path in (READINESS, PREFLIGHT, PREFLIGHT_BINDING, PREFLIGHT_ATTEMPT):
        if not path.is_file():
            raise FileNotFoundError(f"N1-161 readiness/preflight required: {path}")
    readiness = read_json(READINESS)
    preflight = read_json(PREFLIGHT)
    pf_binding = read_json(PREFLIGHT_BINDING)
    require(
        readiness.get("complete") is True
        and readiness.get("plan_sha256") == PLAN_SHA256
        and readiness.get("source_hashes_unchanged") is True,
        "N1-161 readiness is stale",
    )
    require(
        preflight.get("complete") is True
        and preflight.get("plan_sha256") == PLAN_SHA256
        and seal_ast.receipt_binding_matches(preflight.get("readiness_sha256"), READINESS)
        and preflight.get("source_hashes_unchanged") is True
        and preflight.get("independent_restore_count") == 6,
        "N1-161 six-bundle preflight is incomplete",
    )
    require(
        pf_binding.get("status") == "complete"
        and pf_binding.get("plan_sha256") == PLAN_SHA256
        and seal_ast.receipt_binding_matches(pf_binding.get("readiness_sha256"), READINESS)
        and seal_ast.receipt_binding_matches(pf_binding.get("preflight_sha256"), PREFLIGHT),
        "N1-161 preflight binding is stale",
    )
    require(
        pf_binding.get("source_hashes") == preflight.get("source_hashes"),
        "N1-161 preflight source map differs",
    )
    started = time.process_time()
    endpoints, sources, exclusions = source_context()
    ready_sources = dict(sources)
    ready_sources[str(SOURCE_AUDIT.resolve())] = sha256(SOURCE_AUDIT)
    require(readiness.get("source_hashes") == ready_sources, "N1-161 readiness source map changed")
    face, details = _make_face()
    write_new(FACE, face)
    bound = dict(ready_sources)
    for path in (READINESS, SOURCE_BINDING, PREFLIGHT, PREFLIGHT_BINDING, FACE):
        bound[str(Path(path).resolve())] = sha256(path)
    audit = {
        "schema": "n1-plan161-question-surface-face-audit-v1",
        "status": "complete",
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_hashes": bound,
        "source_hashes_unchanged": n160.n150_builder.verify_hashes(bound),
        "question_templates": QUESTION_TEMPLATES,
        "source_exclusion_summary": exclusions["summary"],
        "face_contract": details["face_contract"],
        "base_generator_summary": details["base_generator"],
        "cpu_seconds": round(time.process_time() - started, 6),
        "can_promote": False,
    }
    write_new(FACE_AUDIT, audit)
    checks = {
        "schema": "n1-plan161-question-surface-face-checks-v1",
        "status": (
            "complete"
            if audit["source_hashes_unchanged"] and details["face_contract"]["accepted"]
            else "stopped"
        ),
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_audit_sha256": sha256(FACE_AUDIT),
        "face_contract": details["face_contract"],
        "source_hashes_unchanged": audit["source_hashes_unchanged"],
        "total_queries": TOTAL_QUERIES,
        "can_promote": False,
    }
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-161 source readiness and seed131 question-surface face"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--prepare-face", action="store_true")
    args = parser.parse_args()
    result = source_readiness() if args.validate_only else prepare_face()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "source_hashes_unchanged",
                    "endpoint_count",
                    "cpu_seconds",
                    "cumulative_linked_cpu_seconds",
                    "model_loads",
                    "model_queries",
                    "parameter_updates",
                    "can_promote",
                )
                if key in result
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete", result.get("status") == "complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
