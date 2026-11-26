"""Execute the approved PLAN-N1-99 known-face abstention mechanism probe.

The v6 4-observation/d=1 cell is reused from its published report. Only B/C/D
are queried, for 135 rows total. This is a diagnostic on transformed known
items, never an unseen confirmation or capability gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    build_taiji_n1_fresh_confirmation_v6 as builder,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation as base85,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as bind,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_native_evidence_paired as paired,
)
from scripts.training import (  # noqa: E402
    verify_taiji_n1_v6_range_check as range_gate,
)

PLAN = ROOT / "plans/reference/PLAN-N1-99_unknown_abstention_factorial_diagnostic_20261017.md"
FACE = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
BASELINE = ROOT / "reports/taiji_n1_transition_factorial_confirmation_v6.json"
PLAN90 = ROOT / "plans/reference/PLAN-N1-90_three_family_binding_v5_confirmation_20261017.md"
PLAN93 = ROOT / "plans/reference/PLAN-N1-93_v6_unseen_face_construction_20261017.md"
REPORT = ROOT / "reports/taiji_n1_unknown_abstention_factorial_diagnostic_20261017.json"
ATTEMPT = ROOT / "reports/taiji_n1_unknown_abstention_factorial_diagnostic_20261017_attempt.json"

SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
NEW_CELLS = ("B", "C", "D")
FACE_SHA256 = "7dddf00f3f2e24381728855b218e67bc8160fd6badeb1a7a933937d999766d49"
BASELINE_SHA256 = "3d924e19d97b59ea12505111603b7d562ada8e8d9bdac71351a51c24c5918849"
NAME_SEED = 9902
CPU_LIMIT_SECONDS = 300
ITEMS_PER_CELL = 5
ENDPOINTS = len(SEEDS) * len(ARMS)
NEW_QUERY_COUNT = len(NEW_CELLS) * ITEMS_PER_CELL * ENDPOINTS
Refuse = range_gate.Refuse


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def hamming(left: str, right: str) -> int:
    require(len(left) == len(right), "Hamming distance requires equal-length entity names")
    return sum(a != b for a, b in zip(left, right, strict=True))


def candidate_value(observation: str, entity: str) -> str:
    prefix = f"{entity} uses "
    require(observation.count(prefix) == 1, "unknown observation must bind one listed entity")
    require(observation.endswith("."), "unknown observation must retain its terminal period")
    value = observation[len(prefix) : -1]
    require(bool(value), "candidate observation value cannot be empty")
    return value


def validate_base_item(item: dict) -> None:
    require(item.get("kind") == "unknown", "PLAN-N1-99 only accepts unknown items")
    require(item.get("split") == "holdout", "diagnostic source items must retain holdout labels")
    require(item.get("answer") == "unknown", "unknown target text changed")
    require(
        len(item.get("observations", [])) == 4, "v6 unknown base item must have four observations"
    )
    entities = item.get("entities", [])
    require(len(entities) == 5, "v6 unknown base item must have five entities")
    require(all(name.isascii() and name.islower() for name in entities), "entity alphabet changed")
    require(all(len(name) == len(entities[0]) for name in entities), "entity name lengths differ")
    require(len(set(entities)) == len(entities), "base item contains duplicate entity names")
    subject = entities[0]
    require(item["question"].count(subject) == 1, "question must name its subject exactly once")
    require(
        all(entity not in item["question"] for entity in entities[1:]),
        "distractor entity leaked into the question",
    )
    require(
        all(subject not in observation for observation in item["observations"]),
        "unknown subject unexpectedly occurs in evidence",
    )
    require(
        [hamming(subject, name) for name in entities[1:]] == [1, 1, 1, 1],
        "v6 base distractors no longer match the signed d=1 shape",
    )
    for name, observation in zip(entities[1:], item["observations"], strict=True):
        candidate_value(observation, name)


def _confusable(name: str, pool: set[str], distance: int, rng: random.Random) -> str:
    """Use PLAN-N1-93's exact-character replacement method with a frozen RNG."""
    letters = builder.LETTERS
    for _attempt in range(200):
        chars = list(name)
        for position in rng.sample(range(len(name)), distance):
            chars[position] = rng.choice(letters.replace(chars[position], ""))
        value = "".join(chars)
        if value not in pool:
            pool.add(value)
            return value
    raise Refuse("could not draw a unique d=2 entity name after 200 attempts")


def _variant(item: dict, cell: str, entities: list[str], observations: list[str]) -> dict:
    variant = copy.deepcopy(item)
    variant["id"] = f"n1-99-{cell}-{item['id']}"
    variant["entities"] = list(entities)
    variant["observations"] = list(observations)
    variant["plan99_cell"] = cell
    variant["base_id"] = item["id"]
    return variant


def build_variants(items: list[dict], *, name_seed: int = NAME_SEED) -> dict[str, list[dict]]:
    require(len(items) == ITEMS_PER_CELL, f"expected five base items, got {len(items)}")
    ids = [item.get("id") for item in items]
    require(len(set(ids)) == ITEMS_PER_CELL, "duplicate v6 base item IDs")
    pool = {name for item in items for name in item["entities"]}
    rng = random.Random(name_seed)
    variants = {cell: [] for cell in NEW_CELLS}
    for item in items:
        validate_base_item(item)
        entities = item["entities"]
        subject = entities[0]
        d2_names = [_confusable(subject, pool, 2, rng) for _ in range(4)]
        require(
            all(hamming(subject, name) == 2 for name in d2_names),
            "generated d=2 entity failed exact Hamming check",
        )
        d2_observations = []
        for old_name, new_name, observation in zip(
            entities[1:], d2_names, item["observations"], strict=True
        ):
            value = candidate_value(observation, old_name)
            require(observation.count(old_name) == 1, "base observation name is ambiguous")
            rewritten = observation.replace(old_name, new_name, 1)
            require(candidate_value(rewritten, new_name) == value, "candidate value changed")
            d2_observations.append(rewritten)

        variants["B"].append(
            _variant(item, "B", [subject, *entities[1:3]], item["observations"][:2])
        )
        variants["C"].append(_variant(item, "C", [subject, *d2_names], d2_observations))
        variants["D"].append(_variant(item, "D", [subject, *d2_names[:2]], d2_observations[:2]))

    all_ids = [item["id"] for cell_items in variants.values() for item in cell_items]
    require(len(all_ids) == NEW_QUERY_COUNT // ENDPOINTS, "wrong number of diagnostic variants")
    require(len(set(all_ids)) == len(all_ids), "diagnostic variant IDs collide")
    return variants


def _baseline_rows(report: dict, items: list[dict]) -> list[dict]:
    ids = {item["id"] for item in items}
    rows = [row for row in report["rows"] if row.get("id") in ids and row.get("kind") == "unknown"]
    expected_keys = {(item_id, seed, arm) for item_id in ids for seed in SEEDS for arm in ARMS}
    keys = [(row["id"], row["seed"], row["arm"]) for row in rows]
    require(len(rows) == len(expected_keys), "v6 A baseline must contain exactly 45 rows")
    require(
        len(keys) == len(set(keys)) and set(keys) == expected_keys,
        "v6 A baseline join is not bijective",
    )
    for row in rows:
        require(row.get("split") == "holdout", "v6 A baseline split changed")
        require(
            row.get("target_symbols") == list(b"unknown") + [256], "v6 A baseline target changed"
        )
        exact = bytes.fromhex(row["output_hex"]) == b"unknown"
        require(exact == bool(row["free_exact"]), "v6 A exact-output field disagrees with bytes")
    return [
        dict(row, condition="A", base_id=row["id"], query_source="published_v6") for row in rows
    ]


def _summaries(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for cell in ("A", *NEW_CELLS):
        selected = [row for row in rows if row.get("condition") == cell]
        by_model = {}
        for seed in SEEDS:
            for arm in ARMS:
                model_rows = [row for row in selected if row["seed"] == seed and row["arm"] == arm]
                by_model[f"{seed}_{arm}"] = {
                    "rows": len(model_rows),
                    "exact_unknown": sum(
                        bytes.fromhex(row["output_hex"]) == b"unknown" for row in model_rows
                    ),
                }
        result[cell] = {"rows": len(selected), "by_seed_arm": by_model}
    return result


def _code_files() -> list[Path]:
    return [
        Path(__file__).resolve(),
        Path(base85.__file__).resolve(),
        Path(bind.__file__).resolve(),
        Path(builder.__file__).resolve(),
        Path(paired.__file__).resolve(),
        Path(range_gate.__file__).resolve(),
    ]


def preconditions() -> dict:
    """Bind the approved surface and all nine endpoints without loading models."""
    require(PLAN.is_file(), f"approved PLAN-N1-99 is missing: {PLAN}")
    plan_text = PLAN.read_text(encoding="utf-8")
    require("owner 于 2026-10-17 已明确批准" in plan_text, "PLAN-N1-99 approval marker is absent")
    require(not ATTEMPT.exists(), f"one-shot attempt already exists: {ATTEMPT}")
    require(not REPORT.exists(), f"diagnostic report path already exists: {REPORT}")
    require(
        FACE.is_file() and BASELINE.is_file(), "v6 input manifest or baseline report is missing"
    )
    require(sha_raw(FACE) == FACE_SHA256, "v6 manifest bytes differ from approved PLAN-N1-99")
    require(sha_raw(BASELINE) == BASELINE_SHA256, "published v6 baseline report changed")
    manifest = read_json(FACE)
    confirm_ids = manifest.get("subsets", {}).get("confirm", [])
    require(
        len(confirm_ids) == 20 and len(set(confirm_ids)) == 20,
        "v6 confirm subset must retain 20 IDs",
    )
    confirm_set = set(confirm_ids)
    confirm_episodes = [
        episode for episode in manifest.get("episodes", []) if episode.get("id") in confirm_set
    ]
    require(
        len(confirm_episodes) == 20 and {item["id"] for item in confirm_episodes} == confirm_set,
        "confirm subset IDs do not bind to episodes",
    )
    items = [item for item in confirm_episodes if item.get("kind") == "unknown"]
    require(len(items) == ITEMS_PER_CELL, "v6 confirm subset must contain five unknown items")
    for item in items:
        validate_base_item(item)

    baseline_report = read_json(BASELINE)
    require(
        baseline_report.get("status") == "published"
        and baseline_report.get("retry_allowed") is False
        and baseline_report.get("consumed_face") == "v6-confirm-20"
        and baseline_report.get("confirmation_queries") == 180
        and len(baseline_report.get("rows", [])) == 180,
        "v6 baseline report is not the consumed published A cell source",
    )
    baseline_rows = _baseline_rows(baseline_report, items)
    variants = build_variants(items)
    for cell, cell_items in variants.items():
        require(len(cell_items) == ITEMS_PER_CELL, f"condition {cell} must contain five items")

    # PLAN-N1-90 owns cross-family endpoint identity; it performs only receipt/hash checks here.
    bound_guards, seals, endpoints, _v5_episodes = bind.readiness()
    expected_endpoints = {(seed, arm) for seed in SEEDS for arm in ARMS}
    require(
        set(endpoints) == expected_endpoints, "the exact nine PLAN-N1-90 endpoints are required"
    )
    v6_guards = baseline_report.get("guards", {})
    for (seed, arm), path in endpoints.items():
        path = Path(path).resolve()
        require(path.is_file(), f"frozen endpoint is missing: {path.name}")
        guard = v6_guards.get(str(path))
        require(isinstance(guard, dict), f"v6 baseline does not bind endpoint: {path.name}")
        require(guard.get("sha_raw") == sha_raw(path), f"v6 endpoint bytes changed: {path.name}")
        require(path.name == f"s{seed}_{arm}_final.pt", f"wrong endpoint path for {seed}/{arm}")
        endpoints[(seed, arm)] = path

    guards = dict(bound_guards)
    additional_sources = [PLAN, FACE, BASELINE, PLAN90, PLAN93, *_code_files()]
    for path in additional_sources:
        require(path.is_file(), f"guarded source is missing: {path}")
        resolved = path.resolve()
        guards[str(resolved)] = bind.lr88.guard_sha256(resolved)

    code_identity = {str(path): sha_raw(path) for path in _code_files()}
    return {
        "items": items,
        "variants": variants,
        "baseline_rows": baseline_rows,
        "baseline_report_sha256": sha_raw(BASELINE),
        "face_sha256": sha_raw(FACE),
        "plan_sha256": sha_raw(PLAN),
        "endpoints": endpoints,
        "guards": guards,
        "code_identity": code_identity,
        "binding_seals": seals,
    }


def _guards_unchanged(guards: dict[str, str]) -> bool:
    for raw_path, expected in guards.items():
        path = Path(raw_path)
        if not path.is_file() or bind.lr88.guard_sha256(path) != expected:
            return False
    return True


def _code_unchanged(identity: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha_raw(Path(path)) == expected
        for path, expected in identity.items()
    )


def _annotate_rows(rows: list[dict], cell: str, id_to_base: dict[str, str]) -> list[dict]:
    annotated = []
    for row in rows:
        require(row["id"] in id_to_base, f"unexpected generated item ID: {row['id']}")
        annotated.append(
            dict(row, condition=cell, base_id=id_to_base[row["id"]], query_source="new_diagnostic")
        )
    return annotated


def write_new(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def run_readiness() -> int:
    started = time.process_time()
    context = preconditions()
    print(
        json.dumps(
            {
                "mode": "readiness",
                "plan_sha256": context["plan_sha256"],
                "v6_sha256": context["face_sha256"],
                "baseline_report_sha256": context["baseline_report_sha256"],
                "unknown_items": len(context["items"]),
                "endpoints": len(context["endpoints"]),
                "baseline_rows_reused": len(context["baseline_rows"]),
                "new_cells": list(NEW_CELLS),
                "new_queries": NEW_QUERY_COUNT,
                "cpu_limit_seconds": CPU_LIMIT_SECONDS,
                "model_loads": 0,
                "checkpoint_loads": 0,
                "new_diagnostic_queries": 0,
                "readiness_cpu_seconds": round(time.process_time() - started, 6),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_approved() -> int:
    started = time.process_time()
    context = preconditions()
    budget = range_gate.Deadline(started, CPU_LIMIT_SECONDS)
    result = {
        "schema": "taiji-n1-unknown-abstention-factorial-diagnostic-v1",
        "plan": str(PLAN),
        "plan_sha256": context["plan_sha256"],
        "status": "running",
        "diagnostic_kind": "known_face_intervention_not_unseen_confirmation",
        "v6_manifest": str(FACE),
        "v6_manifest_sha256_raw": context["face_sha256"],
        "baseline_report": str(BASELINE),
        "baseline_report_sha256_raw": context["baseline_report_sha256"],
        "baseline_rows_reused": len(context["baseline_rows"]),
        "new_cells": list(NEW_CELLS),
        "new_diagnostic_query_limit": NEW_QUERY_COUNT,
        "new_diagnostic_queries": 0,
        "confirmation_queries": 0,
        "dev_queries": 0,
        "v3_holdout_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "training_runs": 0,
        "can_promote": False,
        "unseen_claim_allowed": False,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "name_rng_seed": NAME_SEED,
        "source_guards": context["guards"],
        "code_identity": context["code_identity"],
        "binding_seals": context["binding_seals"],
        "rows": list(context["baseline_rows"]),
        "restore_checks": [],
    }
    write_new(
        ATTEMPT,
        {
            "schema": "taiji-n1-unknown-abstention-factorial-diagnostic-attempt-v1",
            "started_process_time": started,
            "consumed_on_write": True,
            "plan_sha256": context["plan_sha256"],
            "face_sha256_raw": context["face_sha256"],
            "baseline_report_sha256_raw": context["baseline_report_sha256"],
            "source_guards": context["guards"],
            "code_identity": context["code_identity"],
            "new_cells": list(NEW_CELLS),
            "expected_new_diagnostic_queries": NEW_QUERY_COUNT,
        },
    )
    status = "complete"
    error = None
    accounting = {"confirmation_queries": 0}
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=10)
                path = context["endpoints"][seed, arm]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                require(
                    payload.get("seed") == seed and payload.get("arm") == arm,
                    "checkpoint seed/arm mismatch",
                )
                model, optimizer = bind.load_with_family_protocol(payload)
                before = paired.fingerprint(model.bundle(optimizer))
                require(
                    before == paired.fingerprint(payload["candidate"])
                    and before == payload["expected_signature"]["bundle"],
                    f"full model/Adam/RNG/history restore differs for {seed}/{arm}",
                )
                for cell in NEW_CELLS:
                    budget.guard(reserve=10)
                    items = context["variants"][cell]
                    id_to_base = {item["id"]: item["base_id"] for item in items}
                    partial_rows: list[dict] = []
                    try:
                        base85.evaluate(
                            model,
                            optimizer,
                            seed,
                            arm,
                            items,
                            budget,
                            output=partial_rows,
                            accounting=accounting,
                        )
                    except Exception:
                        result["rows"].extend(_annotate_rows(partial_rows, cell, id_to_base))
                        result["new_diagnostic_queries"] = accounting["confirmation_queries"]
                        raise
                    result["rows"].extend(_annotate_rows(partial_rows, cell, id_to_base))
                    result["new_diagnostic_queries"] = accounting["confirmation_queries"]
                    print(
                        f"n1_unknown99 cell={cell} seed={seed} arm={arm} "
                        f"exact={sum(bytes.fromhex(row['output_hex']) == b'unknown' for row in partial_rows)}/5 "
                        f"cpu={time.process_time() - started:.2f}",
                        flush=True,
                    )
                unchanged = before == paired.fingerprint(model.bundle(optimizer))
                result["restore_checks"].append({"seed": seed, "arm": arm, "unchanged": unchanged})
                require(
                    unchanged, f"diagnostic changed model/optimizer/RNG/history for {seed}/{arm}"
                )
    except KeyboardInterrupt:
        status = "interrupted"
        error = "diagnostic attempt consumed; partial rows preserved and no retry"
    except Exception as failure:
        status = "stopped"
        error = f"{type(failure).__name__}: {failure}"

    result["status"] = status
    result["error"] = error
    result["sources_unchanged"] = _guards_unchanged(context["guards"])
    result["code_unchanged"] = _code_unchanged(context["code_identity"])
    result["endpoint_neutrality_all_true"] = bool(
        len(result["restore_checks"]) == ENDPOINTS
        and all(check["unchanged"] for check in result["restore_checks"])
    )
    result["query_rows_match_count"] = result["new_diagnostic_queries"] == len(
        [row for row in result["rows"] if row.get("query_source") == "new_diagnostic"]
    )
    result["cell_summary"] = _summaries(result["rows"])
    result["cpu_seconds_before_report_close"] = round(time.process_time() - started, 6)
    result["report_close_reserve_seconds"] = 5
    result["cpu_budget_guard_passed"] = (
        result["cpu_seconds_before_report_close"] + result["report_close_reserve_seconds"]
        <= CPU_LIMIT_SECONDS
    )
    if status == "complete" and not result["cpu_budget_guard_passed"]:
        result["status"] = "stopped"
        result["error"] = "insufficient CPU reserve for report close"
    result["complete"] = bool(
        result["status"] == "complete"
        and result["new_diagnostic_queries"] == NEW_QUERY_COUNT
        and len(result["rows"]) == 4 * ITEMS_PER_CELL * ENDPOINTS
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["endpoint_neutrality_all_true"]
        and result["query_rows_match_count"]
        and result["cpu_budget_guard_passed"]
    )
    result["capability_pass"] = False
    write_new(REPORT, result)
    actual_cpu_after_close = round(time.process_time() - started, 6)
    print(
        json.dumps(
            {
                "mode": "approved",
                "status": result["status"],
                "complete": result["complete"],
                "new_diagnostic_queries": result["new_diagnostic_queries"],
                "baseline_rows_reused": result["baseline_rows_reused"],
                "cpu_seconds_before_report_close": result["cpu_seconds_before_report_close"],
                "actual_cpu_after_report_close": actual_cpu_after_close,
                "cpu_limit_seconds": CPU_LIMIT_SECONDS,
                "out": str(REPORT),
                "error": result["error"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["complete"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--readiness-only", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.readiness_only:
            return run_readiness()
        return run_approved()
    except Refuse as refusal:
        print(f"REFUSE: {refusal}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
