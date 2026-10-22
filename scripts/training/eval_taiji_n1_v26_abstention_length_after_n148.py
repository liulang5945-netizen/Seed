"""PLAN-N1-149：仅接受diagnostic split的matched abstention scorer与runner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_v25_abstention_after_n147 as n148_builder
from scripts.training import build_taiji_n1_v26_abstention_length_after_n148 as face_builder
from scripts.training import eval_taiji_n1_v23_confirmation_after_n143 as frozen_eval
from scripts.training import eval_taiji_n1_v25_abstention_after_n147 as n148_eval

PLAN = face_builder.PLAN
PLAN_SHA256 = face_builder.PLAN_SHA256
CPU_LIMIT_SECONDS = 7_200.0
SOURCE_PRECHECK = face_builder.SOURCE_PRECHECK
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
FACE = face_builder.FACE
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
PREPARED = face_builder.PREPARED

READINESS = ROOT / "reports/taiji_n1_v26_abstention_length_readiness_20261019.json"
READINESS_ATTEMPT = ROOT / "reports/taiji_n1_v26_abstention_length_readiness_attempt_20261019.json"
SOURCE_BINDING = ROOT / "reports/taiji_n1_v26_abstention_length_source_binding_20261019.json"
ATTEMPT = ROOT / "reports/taiji_n1_v26_abstention_length_attempt_20261019.json"
PROGRESS = ROOT / "reports/taiji_n1_v26_abstention_length_progress_20261019.jsonl"
RESULT = ROOT / "reports/taiji_n1_v26_abstention_length_diagnostic_20261019.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan149_v26_abstention_length_contract.py"

CONDITIONS = ("known", "unknown")
LENGTHS = (8, 16, 32)
CONTEXT_LENGTHS = LENGTHS
LOGICAL_PAIRS = 40
ITEMS = LOGICAL_PAIRS * len(LENGTHS) * len(CONDITIONS)


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


def append_jsonl(path: Path, value: dict) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


class DurableRows(list):
    """Write each scored diagnostic row before the next model call."""

    def __init__(self, subset: str, spec: dict, progress_path: Path):
        super().__init__()
        self.subset = subset
        self.spec = spec
        self.progress_path = Path(progress_path)

    def append(self, row: dict) -> None:
        durable = dict(
            row,
            subset=self.subset,
            endpoint_id=self.spec["endpoint_id"],
            source=self.spec["source"],
            branch=self.spec["branch"],
            seed=self.spec["seed"],
        )
        super().append(durable)
        append_jsonl(self.progress_path, {"event": "query_row", "row": durable})


def validate_diagnostic_items(items: list[dict]) -> dict:
    if len(items) != ITEMS or len({item.get("id") for item in items}) != ITEMS:
        raise ValueError("exactly 240 unique diagnostic items are required")
    if any(item.get("split") != "diagnostic" for item in items):
        raise ValueError("diagnostic split required; do not relabel as holdout")
    if any(item.get("arm", "position") != "position" for item in items):
        raise ValueError("frozen position arm required")
    if any(
        item.get("context_length") not in LENGTHS
        or len(item.get("observations", [])) != item.get("context_length")
        for item in items
    ):
        raise ValueError("diagnostic context length must be exactly 8, 16, or 32 rows")
    length_groups: dict[int, dict[str, dict[str, dict]]] = {
        length: defaultdict(dict) for length in LENGTHS
    }
    logical_groups: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for item in items:
        condition = item.get("pair_condition")
        if condition not in CONDITIONS or item.get("kind") != (
            "lookup" if condition == "known" else "unknown"
        ):
            raise ValueError(f"invalid matched diagnostic condition: {item.get('id')}")
        length_groups[int(item["context_length"])][str(item["pair_id"])][condition] = item
        logical_groups[str(item["logical_pair_id"])][int(item["context_length"])][condition] = item
    if len(logical_groups) != LOGICAL_PAIRS:
        raise ValueError("exactly 40 logical matched pairs are required")
    for length in LENGTHS:
        pairs = length_groups[length]
        if len(pairs) != LOGICAL_PAIRS or any(
            set(group) != set(CONDITIONS) for group in pairs.values()
        ):
            raise ValueError(f"40 complete known/unknown pairs are required at length {length}")
        known_lengths = Counter(len(group["known"]["answer"]) for group in pairs.values())
        if any(known_lengths[value] != 20 for value in (6, 7)):
            raise ValueError(f"6/7-character answers must balance at length {length}")
        for pair_id, group in pairs.items():
            known = group["known"]
            unknown = group["unknown"]
            target = known.get("entities", [None])[0]
            if (
                known.get("target_start_index") != 0
                or unknown.get("target_start_index") != 0
                or known.get("logical_pair_id") != unknown.get("logical_pair_id")
                or known.get("pair_id") != unknown.get("pair_id")
                or unknown.get("entities", [None])[0] != target
                or known.get("question") != unknown.get("question")
                or known.get("target_fact_indices") != [0]
                or unknown.get("target_fact_indices") != []
                or known["observations"][0] != f"{target} uses {known.get('answer')}."
                or any(
                    known["observations"][index] != unknown["observations"][index]
                    for index in range(1, length)
                )
            ):
                raise ValueError(f"matched pair contract mismatch: {pair_id}")
    for logical_id, by_length in logical_groups.items():
        if set(by_length) != set(LENGTHS):
            raise ValueError(f"logical pair missing a context length: {logical_id}")
        prior_prefix: dict[str, list[str]] = {}
        prior_answer = None
        prior_question = None
        for length in LENGTHS:
            conditions = by_length[length]
            if set(conditions) != set(CONDITIONS):
                raise ValueError(f"known/unknown condition missing: {logical_id}/{length}")
            known = conditions["known"]
            unknown = conditions["unknown"]
            if prior_answer is not None and known["answer"] != prior_answer:
                raise ValueError(f"known answer changes across lengths: {logical_id}")
            if prior_question is not None and known["question"] != prior_question:
                raise ValueError(f"question changes across lengths: {logical_id}")
            prior_answer = known["answer"]
            prior_question = known["question"]
            if prior_prefix:
                for condition in CONDITIONS:
                    if (
                        by_length[length][condition]["observations"][: len(prior_prefix[condition])]
                        != prior_prefix[condition]
                    ):
                        raise ValueError(
                            f"contexts are not nested: {logical_id}/{condition}/{length}"
                        )
            prior_prefix = {
                condition: list(by_length[length][condition]["observations"])
                for condition in CONDITIONS
            }
    return {
        "items": len(items),
        "logical_pairs": len(logical_groups),
        "pair_groups": sum(len(length_groups[length]) for length in LENGTHS),
        "items_by_length": {
            length: sum(item["context_length"] == length for item in items) for length in LENGTHS
        },
        "conditions": dict(Counter(item["pair_condition"] for item in items)),
    }


@torch.no_grad()
def measure_diagnostic(model, item: dict, arm: str, session: str) -> dict:
    """A diagnostic-only scorer; preserves the diagnostic split and never steps an optimizer."""
    if item.get("split") != "diagnostic" or arm != "position":
        raise ValueError("diagnostic split and frozen position arm are required")
    paired = frozen_eval.n132.e11.scorer.paired
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode("utf-8")
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return {
        "id": item["id"],
        "logical_pair_id": item["logical_pair_id"],
        "pair_id": item["pair_id"],
        "pair_condition": item["pair_condition"],
        "kind": item["kind"],
        "split": "diagnostic",
        "context_length": item["context_length"],
        "target_start_index": item["target_start_index"],
        "paired_answer_length": item["paired_answer_length"],
        "answer_length": item["answer_length"],
        "teacher_exact": bool(torch.equal(predicted, labels)),
        "teacher_tokens_correct": int((predicted == labels).sum()),
        "teacher_token_count": len(labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "teacher_symbols": predicted.tolist(),
        "target_symbols": labels.tolist(),
        "output_hex": trace.output.hex(),
        "raw_hex": trace.raw.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def score_endpoint_diagnostic(
    model,
    optimizer,
    spec: dict,
    items: list[dict],
    *,
    fingerprint,
    budget_guard,
    output: DurableRows,
    accounting: dict,
) -> None:
    before = fingerprint(model.bundle(optimizer))
    _ = validate_diagnostic_items(items)
    try:
        for item in items:
            budget_guard()
            session = f"n1-v26-diagnostic-{spec['seed']}-{spec['branch']}-{item['id']}"
            if session in model._observations:
                raise ValueError("diagnostic session collides with frozen model history")
            accounting["model_queries"] += 1
            try:
                row = measure_diagnostic(model, item, "position", session)
            finally:
                model._observations.pop(session, None)
            if row.get("id") != item["id"] or row.get("pair_id") != item["pair_id"]:
                raise ValueError("diagnostic scorer returned a row for the wrong item")
            output.append(row)
    finally:
        if fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("diagnostic changed model, optimizer, RNG, or history bundle")


def _read_prepared_face() -> tuple[dict, dict, dict]:
    if not PLAN.is_file() or sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-149 frozen SHA-256 mismatch")
    paths = (SOURCE_PRECHECK, SOURCE_EXCLUSIONS, FACE, FACE_ATTEMPT, FACE_CHECKS, PREPARED)
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("N1-149 face preparation is incomplete")
    precheck = read_json(SOURCE_PRECHECK)
    manifest = read_json(SOURCE_EXCLUSIONS)
    face = read_json(FACE)
    attempt = read_json(FACE_ATTEMPT)
    checks = read_json(FACE_CHECKS)
    prepared = read_json(PREPARED)
    if (
        precheck.get("status") != "complete"
        or precheck.get("plan_sha256") != PLAN_SHA256
        or precheck.get("source_union", {}).get("entity_exclusion_count") != 39_442
        or precheck.get("source_union", {}).get("word_exclusion_count") != 16_608
        or precheck.get("source_hashes_unchanged") is not True
        or precheck.get("model_queries") != 0
        or precheck.get("parameter_updates") != 0
        or manifest.get("schema") != "n1-plan149-v26-source-exclusions-v1"
        or manifest.get("status") != "ready_for_one_time_face_generation"
        or manifest.get("plan_sha256") != PLAN_SHA256
        or manifest.get("entity_count") != 39_442
        or manifest.get("word_value_count") != 16_608
        or len(set(manifest.get("excluded_entities", []))) != 39_442
        or len(set(manifest.get("excluded_word_values", []))) != 16_608
        or face_builder.digest_strings(set(manifest.get("excluded_entities", [])))
        != manifest.get("entity_exclusion_sha256")
        or face_builder.digest_strings(set(manifest.get("excluded_word_values", [])))
        != manifest.get("word_exclusion_sha256")
        or face.get("schema") != "n1-v26-matched-abstention-length-diagnostic-face-v1"
        or face.get("seed") != 126
        or face.get("plan_sha256") != PLAN_SHA256
        or len(face.get("episodes", [])) != 240
        or attempt.get("plan_sha256") != PLAN_SHA256
        or attempt.get("model_queries") != 0
        or attempt.get("parameter_updates") != 0
        or sha256(FACE) != checks.get("face_sha256")
        or checks.get("accepted") is not True
        or checks.get("plan_sha256") != PLAN_SHA256
        or checks.get("source_exclusion_sha256") != sha256(SOURCE_EXCLUSIONS)
        or checks.get("source_hashes_unchanged") is not True
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or prepared.get("complete") is not True
        or prepared.get("plan_sha256") != PLAN_SHA256
        or prepared.get("face_sha256") != sha256(FACE)
        or prepared.get("face_checks_sha256") != sha256(FACE_CHECKS)
        or prepared.get("source_exclusion_sha256") != sha256(SOURCE_EXCLUSIONS)
        or prepared.get("source_precheck_sha256") != sha256(SOURCE_PRECHECK)
        or prepared.get("source_hashes_unchanged") is not True
        or prepared.get("model_queries") != 0
        or prepared.get("parameter_updates") != 0
    ):
        raise ValueError("v26 face/source preparation differs from frozen contract")
    if not n148_builder.n144_builder._verify_hashes(manifest.get("source_hashes", {})):
        raise ValueError("N1-149 source exclusion hashes changed")
    if not n148_builder.n144_builder._verify_hashes(prepared.get("source_hashes", {})):
        raise ValueError("N1-149 prepared-face source hashes changed")
    verified = face_builder.validate_face(
        face,
        set(manifest["excluded_entities"]),
        set(manifest["excluded_word_values"]),
    )
    if verified.get("accepted") is not True:
        raise ValueError("saved v26 paired face no longer passes its contract")
    return manifest, face, prepared


def _endpoint_specs() -> list[dict]:
    return frozen_eval._endpoint_specs()


def _source_hashes(prepared: dict, specs: list[dict]) -> dict[str, str]:
    hashes = dict(prepared.get("source_hashes", {}))
    paths = (
        SOURCE_PRECHECK,
        SOURCE_EXCLUSIONS,
        FACE,
        FACE_ATTEMPT,
        FACE_CHECKS,
        PREPARED,
        READINESS,
        READINESS_ATTEMPT,
        SOURCE_BINDING,
        ATTEMPT,
        Path(__file__).resolve(),
        Path(face_builder.__file__).resolve(),
        Path(n148_builder.__file__).resolve(),
        Path(n148_eval.__file__).resolve(),
        Path(frozen_eval.__file__).resolve(),
        TEST,
    )
    for path in paths:
        path = Path(path).resolve()
        if path.is_file():
            hashes[str(path)] = sha256(path)
    for spec in specs:
        hashes[str(Path(spec["path"]).resolve())] = spec["sha256"]
    return hashes


def _restore(spec: dict):
    return frozen_eval._restore_endpoint(spec)


def validate_only() -> dict:
    outputs = (READINESS, READINESS_ATTEMPT, SOURCE_BINDING, ATTEMPT, PROGRESS, RESULT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("PLAN-N1-149 readiness/query phase is write-once")
    started = time.process_time()
    frozen_eval.n120_eval.n120.configure()
    _manifest, _face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan149-v26-readiness-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "planned_endpoints": 12,
            "planned_diagnostic_queries": 2880,
            "model_queries": 0,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    restores = []
    errors = []
    for spec in specs:
        before = time.process_time()
        model = optimizer = None
        try:
            model, optimizer, payload, fingerprint = _restore(spec)
            restores.append(
                {
                    "endpoint_id": spec["endpoint_id"],
                    "source": spec["source"],
                    "branch": spec["branch"],
                    "seed": spec["seed"],
                    "checkpoint_sha256": spec["sha256"],
                    "bundle_fingerprint": fingerprint,
                    "history_count": len(model._observations),
                    "optimizer_parameters": len(optimizer.state),
                    "equal": True,
                    "complete": True,
                    "cpu_seconds": round(time.process_time() - before, 6),
                }
            )
        except BaseException as exc:
            errors.append(f"{spec['endpoint_id']}: {type(exc).__name__}: {exc}")
            break
        finally:
            if model is not None:
                del model
            if optimizer is not None:
                del optimizer
    hashes = _source_hashes(prepared, specs)
    unchanged = n148_builder.n144_builder._verify_hashes(hashes)
    if not unchanged:
        errors.append("N1-149 source/checkpoint hash changed during readiness")
    cpu = time.process_time() - started
    prior_cpu = float(prepared["cpu_seconds"])
    cumulative = prior_cpu + cpu
    complete = not errors and len(restores) == 12 and cumulative <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-plan149-v26-checkpoint-readiness-v1",
        "status": "ready_for_diagnostic" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": hashes,
        "source_hashes_unchanged": unchanged,
        "endpoints": restores,
        "endpoint_count": len(restores),
        "independent_full_bundle_restores": len(restores),
        "all_full_bundles_equal": len(restores) == 12 and all(row["equal"] for row in restores),
        "checkpoint_loads": len(restores),
        "model_loads": len(restores),
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "prior_face_preparation_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": (
            "12端点诊断bundle readiness完成；0查询/更新，可进入一次性v26诊断。"
            if complete
            else "bundle readiness停止；0查询，不重试。"
        ),
    }
    write_new(READINESS, report)
    write_new(
        SOURCE_BINDING,
        {
            "schema": "n1-plan149-v26-readiness-source-binding-v1",
            "status": "complete" if complete else "stopped",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_hashes": hashes,
            "checkpoint_hashes": {
                str(Path(spec["path"]).resolve()): spec["sha256"] for spec in specs
            },
            "can_promote": False,
        },
    )
    return report


def _verify_readiness() -> tuple[dict, dict, list[dict]]:
    if not READINESS.is_file() or not SOURCE_BINDING.is_file():
        raise FileNotFoundError("run --validate-only and pass readiness before N1-149 queries")
    readiness = read_json(READINESS)
    binding = read_json(SOURCE_BINDING)
    _manifest, face, prepared = _read_prepared_face()
    specs = _endpoint_specs()
    if (
        readiness.get("status") != "ready_for_diagnostic"
        or readiness.get("complete") is not True
        or readiness.get("plan_sha256") != PLAN_SHA256
        or readiness.get("face_sha256") != sha256(FACE)
        or readiness.get("endpoint_count") != 12
        or readiness.get("independent_full_bundle_restores") != 12
        or readiness.get("all_full_bundles_equal") is not True
        or readiness.get("source_hashes_unchanged") is not True
        or readiness.get("model_queries") != 0
        or readiness.get("parameter_updates") != 0
        or binding.get("status") != "complete"
        or binding.get("plan_sha256") != PLAN_SHA256
        or binding.get("readiness_sha256") != sha256(READINESS)
        or binding.get("source_hashes") != readiness.get("source_hashes")
    ):
        raise ValueError("N1-149 readiness/source binding mismatch")
    hashes = dict(readiness["source_hashes"])
    hashes[str(READINESS.resolve())] = sha256(READINESS)
    hashes[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
    if not n148_builder.n144_builder._verify_hashes(hashes):
        raise ValueError("N1-149 query source/checkpoint hash mismatch")
    return readiness, face, specs


def _paired_summary(rows: list[dict], face: dict) -> dict:
    face_by_id = {item["id"]: item for item in face["episodes"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["endpoint_id"]].append(row)
    by_endpoint = {}
    for endpoint_id, endpoint_rows in sorted(grouped.items()):
        by_id = {row["id"]: row for row in endpoint_rows}
        if len(by_id) != len(endpoint_rows):
            raise ValueError(f"duplicate scored item within endpoint: {endpoint_id}")
        per_length = {}
        for length in CONTEXT_LENGTHS:
            by_pair: dict[str, dict[str, dict]] = defaultdict(dict)
            for item_id, row in by_id.items():
                item = face_by_id[item_id]
                if int(item["context_length"]) == length:
                    by_pair[item["pair_id"]][item["pair_condition"]] = row
            if len(by_pair) != LOGICAL_PAIRS or any(
                set(pair) != set(CONDITIONS) for pair in by_pair.values()
            ):
                raise ValueError(
                    f"endpoint lacks 40 complete pairs at length {length}: {endpoint_id}"
                )
            counts = Counter()
            known_exact = known_abstain = unknown_abstain = unknown_false = 0
            for _pair_id, conditions in by_pair.items():
                known = conditions["known"]
                unknown = conditions["unknown"]
                known_correct = known.get("free_exact") is True
                known_literal_abstain = known.get("output_hex") == b"unknown".hex()
                unknown_literal_abstain = unknown.get("output_hex") == b"unknown".hex()
                known_exact += int(known_correct)
                known_abstain += int(known_literal_abstain)
                unknown_abstain += int(unknown_literal_abstain)
                unknown_false += int(not unknown_literal_abstain)
                pattern = (
                    "known_exact_unknown_abstain"
                    if known_correct and unknown_literal_abstain
                    else (
                        "known_abstains_unknown_abstain"
                        if known_literal_abstain and unknown_literal_abstain
                        else (
                            "known_exact_unknown_false_answer"
                            if known_correct
                            else "other_pair_outcome"
                        )
                    )
                )
                counts[pattern] += 1
            per_length[str(length)] = {
                "pairs": len(by_pair),
                "known_exact": known_exact,
                "known_abstain_literal": known_abstain,
                "unknown_abstain_literal": unknown_abstain,
                "unknown_false_answer": unknown_false,
                "paired_outcomes": dict(sorted(counts.items())),
            }

        logical_pairs: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
        for item_id, row in by_id.items():
            item = face_by_id[item_id]
            logical_pairs[item["logical_pair_id"]][int(item["context_length"])][
                item["pair_condition"]
            ] = row
        if len(logical_pairs) != LOGICAL_PAIRS:
            raise ValueError(f"endpoint logical pair count differs: {endpoint_id}")
        curves = []
        for logical_id, by_length in sorted(logical_pairs.items()):
            if set(by_length) != set(CONTEXT_LENGTHS) or any(
                set(by_length[length]) != set(CONDITIONS) for length in CONTEXT_LENGTHS
            ):
                raise ValueError(f"incomplete nested pair curve: {endpoint_id}/{logical_id}")
            curves.append(
                {
                    "logical_pair_id": logical_id,
                    "by_length": {
                        str(length): {
                            "known_exact": by_length[length]["known"].get("free_exact") is True,
                            "known_abstain_literal": by_length[length]["known"].get("output_hex")
                            == b"unknown".hex(),
                            "unknown_abstain_literal": by_length[length]["unknown"].get(
                                "output_hex"
                            )
                            == b"unknown".hex(),
                        }
                        for length in CONTEXT_LENGTHS
                    },
                }
            )
        by_endpoint[endpoint_id] = {"by_context_length": per_length, "logical_pair_curves": curves}
    return by_endpoint


def execute_approved() -> dict:
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-149 v26 face is one-shot; no retry")
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan149-v26-diagnostic-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS) if READINESS.is_file() else None,
            "source_binding_sha256": sha256(SOURCE_BINDING) if SOURCE_BINDING.is_file() else None,
            "face_sha256": sha256(FACE) if FACE.is_file() else None,
            "planned_endpoints": 12,
            "planned_diagnostic_queries": 2880,
            "parameter_updates": 0,
            "dev_queries": 0,
            "v3_holdout_queries": 0,
            "can_promote": False,
        },
    )
    errors = []
    receipts = []
    result_rows = []
    queries = 0
    model_loads = 0
    readiness = {}
    face = {}
    specs = []
    prior_cpu = 0.0
    sources = {}
    started = time.process_time()
    try:
        frozen_eval.n120_eval.n120.configure()
        readiness, face, specs = _verify_readiness()
        prior_cpu = float(readiness["cumulative_cpu_seconds"])
        budget = frozen_eval.CpuDeadline(
            CPU_LIMIT_SECONDS - prior_cpu - (time.process_time() - started)
        )
        items = list(face["episodes"])
        validation = validate_diagnostic_items(items)
        if validation != {
            "items": 240,
            "logical_pairs": 40,
            "pair_groups": 120,
            "items_by_length": {8: 80, 16: 80, 32: 80},
            "conditions": {"known": 120, "unknown": 120},
        }:
            raise ValueError("v26 diagnostic face counts changed after readiness")
        sources = dict(readiness["source_hashes"])
        sources[str(READINESS.resolve())] = sha256(READINESS)
        sources[str(SOURCE_BINDING.resolve())] = sha256(SOURCE_BINDING)
        sources[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
        if not n148_builder.n144_builder._verify_hashes(sources):
            raise ValueError("N1-149 source/checkpoint hash changed before first diagnostic call")
        PROGRESS.open("x", encoding="utf-8", newline="\n").close()
        for spec in specs:
            budget.guard(reserve=10)
            model = optimizer = None
            before = None
            accounting = {"model_queries": 0}
            output = DurableRows("diagnostic", spec, PROGRESS)
            receipt = {
                "endpoint_id": spec["endpoint_id"],
                "source": spec["source"],
                "branch": spec["branch"],
                "seed": spec["seed"],
                "queries": 0,
                "completed_rows": 0,
                "model_loaded": False,
                "checkpoint_sha256": spec["sha256"],
                "error": None,
            }
            endpoint_started = time.process_time()
            try:
                model, optimizer, payload, before = _restore(spec)
                receipt["model_loaded"] = True
                score_endpoint_diagnostic(
                    model,
                    optimizer,
                    spec,
                    items,
                    fingerprint=frozen_eval.n132.e11.scorer.paired.fingerprint,
                    budget_guard=lambda: budget.guard(reserve=5),
                    output=output,
                    accounting=accounting,
                )
            except BaseException as exc:
                receipt["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if model is not None and optimizer is not None and before is not None:
                    try:
                        if (
                            frozen_eval.n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
                            != before
                        ):
                            receipt["error"] = (
                                "ValueError: frozen diagnostic changed bundle fingerprint"
                            )
                    except BaseException as exc:
                        receipt["error"] = f"{type(exc).__name__}: {exc}"
                if model is not None:
                    del model
                if optimizer is not None:
                    del optimizer
            receipt["queries"] = accounting["model_queries"]
            receipt["completed_rows"] = len(output)
            receipt["bundle_fingerprint"] = before
            receipt["checkpoint_unchanged"] = (
                Path(spec["path"]).is_file() and sha256(spec["path"]) == spec["sha256"]
            )
            receipt["cpu_seconds"] = round(time.process_time() - endpoint_started, 6)
            if not receipt["checkpoint_unchanged"]:
                receipt["error"] = (
                    receipt["error"] or "ValueError: checkpoint bytes changed during diagnostic"
                )
            result_rows.extend(output)
            receipts.append(receipt)
            queries += int(receipt["queries"])
            model_loads += int(receipt["model_loaded"])
            append_jsonl(PROGRESS, {"event": "endpoint_progress", **receipt})
            if receipt["error"]:
                errors.append(f"{spec['endpoint_id']}: {receipt['error']}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    unchanged = bool(sources) and n148_builder.n144_builder._verify_hashes(sources)
    if not unchanged:
        errors.append("N1-149 diagnostic source/checkpoint hash changed")
    execution_cpu = time.process_time() - started
    cumulative = prior_cpu + execution_cpu
    complete = (
        not errors
        and queries == 2880
        and len(result_rows) == 2880
        and len(receipts) == 12
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    paired = _paired_summary(result_rows, face) if result_rows else {}
    report = {
        "schema": "n1-plan149-v26-matched-abstention-length-diagnostic-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_binding_sha256": sha256(SOURCE_BINDING),
        "attempt_sha256": sha256(ATTEMPT),
        "progress_sha256": sha256(PROGRESS) if PROGRESS.is_file() else None,
        "face_sha256": sha256(FACE),
        "source_exclusion_sha256": sha256(SOURCE_EXCLUSIONS),
        "source_hashes": sources,
        "source_hashes_unchanged": unchanged,
        "endpoint_count": 12,
        "diagnostic_queries": queries,
        "diagnostic_rows_completed": len(result_rows),
        "endpoint_receipts": receipts,
        "paired_summary": paired,
        "diagnostic_rows": result_rows,
        "model_loads": model_loads + int(readiness.get("model_loads", 0)),
        "checkpoint_loads": model_loads + int(readiness.get("checkpoint_loads", 0)),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "cpu_seconds": round(execution_cpu, 6),
        "prior_cpu_seconds": round(prior_cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": f"v26 diagnostic split弃答诊断{('完整' if complete else '停止')}：{queries}/2880条；0训练/更新/dev/旧v3 holdout，can_promote=false。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-149 diagnostic-split matched abstention scorer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    result = validate_only() if args.validate_only else execute_approved()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "independent_full_bundle_restores",
                    "diagnostic_queries",
                    "parameter_updates",
                    "can_promote",
                    "errors",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
