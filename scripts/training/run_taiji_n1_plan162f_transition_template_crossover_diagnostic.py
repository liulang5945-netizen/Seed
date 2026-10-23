"""执行PLAN-N1-162F：九个冻结endpoint上的train-only transition问句交叉探针。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import analyze_taiji_n1_plan162b_train_only_itemwise_attribution as n162c
from scripts.training import analyze_taiji_n1_plan162d_transition_output_type_attribution as n162d
from scripts.training import build_taiji_n1_v27_length_robust_training_after_n149 as n150_builder
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132
from scripts.training import run_taiji_n1_plan155_short_unknown_context_length_training as n155
from scripts.training import seal_ast  # noqa: E402

PLAN = (
    ROOT / "plans/reference/PLAN-N1-162F_transition_template_crossover_train_diagnostic_20261020.md"
)
PLAN_SHA256 = "2593df635b5def1d9f24a95fa409fabab0df60008e148aab71a6aec1bd2d19ec"
N155A_RESULT = n155.RESULT
N155A_RESULT_SHA256 = "e577d4dda753d016a46e11ab2e0202b2845206af107c54f87aff7f9fdf1ebb11"
N155A_PROGRESS = ROOT / "reports/taiji_n1_plan155a_unknown8_rebalance_progress_20261020.jsonl"
N155A_PROGRESS_SHA256 = "83e951930272d30ab3b70a0164dba0ef628d8876a8138708be08a97d3a7acf73"
N162B_RESULT = n162c.N162B_RESULT
N162B_RESULT_SHA256 = n162c.N162B_RESULT_SHA256
N162B_PROGRESS = n162c.PROGRESS
N162B_PROGRESS_SHA256 = n162c.PROGRESS_SHA256
N162D_REPORT = ROOT / "reports/taiji_n1_plan162d_transition_output_type_attribution_20261020.json"
N162D_REPORT_SHA256 = "27fc0b88ea53fb3d9f211cbaee9997b2372e2bf56dc9f04c93190e98380ef6a2"
N162E_REPORT = ROOT / "reports/taiji_n1_plan162e_parent_control_surface_drift_20261020.json"
N162E_REPORT_SHA256 = "7a2351050e55c98af4160ff0f5b23cf9feabcd49f10abeba2a7b9db86044f39c"
N162F_ATTEMPT = (
    ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_attempt_20261020.json"
)
N162F_PROGRESS = ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_20261020.jsonl"
N162F_RESULT = ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_20261020.json"
ATTEMPT = ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_attempt_20261020.json"
PROGRESS = ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_20261020.jsonl"
RESULT = ROOT / "reports/taiji_n1_plan162f_transition_template_crossover_20261020.json"
TEST = ROOT / "tests/taiji_native/test_n1_plan162f_transition_template_crossover_contract.py"

SEEDS = (11, 12, 13)
ARMS = ("surface_control", "surface_augmented")
LINEAGES = ("n155a_candidate", *ARMS)
ENDPOINTS = tuple((seed, lineage) for lineage in LINEAGES[:1] for seed in SEEDS) + tuple(
    (seed, arm) for seed in SEEDS for arm in ARMS
)
TEMPLATES = {
    "direction_after_turning": "which direction does {entity} face after turning?",
    "direction_after_turn": "what direction is {entity} facing after its turn?",
}
VIEWS = ("original", "rename")
CONDITIONS = ("original", "direction_opposite", "turn_flip", "both")
KINDS = ("lookup", "revision", "transition", "unknown")
PARENTS_PER_VIEW = 48
CASES_PER_ENDPOINT = PARENTS_PER_VIEW * len(VIEWS) * len(CONDITIONS)
TRANSITION_GATE_ROWS_PER_ENDPOINT = 480
QUERIES_PER_ENDPOINT = CASES_PER_ENDPOINT * len(TEMPLATES)
TOTAL_QUERIES = len(ENDPOINTS) * QUERIES_PER_ENDPOINT
PRIOR_LINKED_CPU_SECONDS = 21_266.234375
LINKED_CPU_LIMIT_SECONDS = 36_000.0
CPU_LIMIT_SECONDS = 1_200.0
DIRECTIONS = ("north", "east", "south", "west")


class Refuse(RuntimeError):
    """PLAN-N1-162F来源、checkpoint或train-only query合同错误。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        previous = target.get(path)
        if (
            previous is not None
            and previous != digest
            and not seal_ast.recorded_value_attested(path, previous)
        ):
            raise Refuse(f"conflicting PLAN-N1-162F source hash: {path}")
        target[path] = digest


def _fingerprint(model, optimizer) -> str:
    return n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))


def _restore(path: Path):
    return n155._restore_checkpoint(Path(path))


def _restore_worker(path: Path) -> dict:
    started = time.process_time()
    model, optimizer, payload, fingerprint = _restore(path)
    result = {
        "complete": True,
        "equal": fingerprint == payload["bundle_fingerprint"],
        "fingerprint": fingerprint,
        "bundle_fingerprint": payload["bundle_fingerprint"],
        "seed": payload.get("seed"),
        "branch": payload.get("branch"),
        "total_updates": payload.get("total_updates"),
        "history_count": len(model._observations),
        "optimizer_parameters": len(optimizer.state),
        "cpu_seconds": round(time.process_time() - started, 6),
    }
    del model, optimizer, payload
    return result


def _independent_restore(path: Path, budget: n132.CpuBudget) -> dict:
    budget.guard(reserve=5)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise Refuse(f"independent N1-162F restore failed: {process.stderr[-1200:]}")
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines:
        raise Refuse("independent N1-162F restore returned no payload")
    row = json.loads(lines[-1])
    budget.children += float(row.get("cpu_seconds", 0.0))
    row["path"] = str(Path(path).resolve())
    row["sha256"] = sha256(path)
    require(
        row.get("complete") is True and row.get("equal") is True,
        f"independent restore mismatch: {path}",
    )
    return row


def _source_and_endpoints() -> tuple[dict, list[dict], dict]:
    require(sha256(PLAN) == PLAN_SHA256, "PLAN-N1-162F plan SHA changed")
    require(
        sha256(N155A_RESULT) == N155A_RESULT_SHA256
        and sha256(N155A_PROGRESS) == N155A_PROGRESS_SHA256,
        "N1-155A source evidence SHA changed",
    )
    require(
        sha256(N162B_RESULT) == N162B_RESULT_SHA256
        and sha256(N162B_PROGRESS) == N162B_PROGRESS_SHA256,
        "N1-162B source evidence SHA changed",
    )
    require(
        sha256(N162D_REPORT) == N162D_REPORT_SHA256 and sha256(N162E_REPORT) == N162E_REPORT_SHA256,
        "N1-162D/E attribution report SHA changed",
    )
    n155a = read_json(N155A_RESULT)
    n162b = read_json(N162B_RESULT)
    n162d = read_json(N162D_REPORT)
    n162e = read_json(N162E_REPORT)
    require(
        n155a.get("complete") is True and n155a.get("source_hashes_unchanged") is True,
        "N1-155A candidate source result incomplete",
    )
    require(
        n162b.get("complete") is True
        and n162b.get("source_hashes_unchanged") is True
        and n162b.get("fresh_queries") == 0,
        "N1-162B source result incomplete or consumed a new face",
    )
    require(
        n162d.get("complete") is True and n162e.get("complete") is True,
        "N1-162D/E attribution sources incomplete",
    )
    source_hashes: dict[str, str] = {}
    _merge_hashes(source_hashes, n155a.get("source_hashes", {}))
    _merge_hashes(source_hashes, n162b.get("source_hashes", {}))
    _merge_hashes(source_hashes, n162d.get("source_hashes", {}))
    _merge_hashes(source_hashes, n162e.get("source_hashes", {}))
    for path, digest in (
        (N155A_RESULT, N155A_RESULT_SHA256),
        (N155A_PROGRESS, N155A_PROGRESS_SHA256),
        (N162B_RESULT, N162B_RESULT_SHA256),
        (N162B_PROGRESS, N162B_PROGRESS_SHA256),
        (N162D_REPORT, N162D_REPORT_SHA256),
        (N162E_REPORT, N162E_REPORT_SHA256),
        (PLAN, PLAN_SHA256),
        (
            Path(__file__).resolve(),
            sha256(Path(__file__)),
        ),
        (TEST, sha256(TEST)),
    ):
        _merge_hashes(source_hashes, {str(Path(path).resolve()): digest})
    require(
        n150_builder.verify_hashes(source_hashes), "N1-162F bound source/checkpoint SHA changed"
    )
    n155_candidates = {
        int(row["seed"]): row
        for row in n155a.get("models", [])
        if row.get("branch") == "unknown8_rebalanced" and row.get("training_sufficient") is True
    }
    n162b_models = {(int(row["seed"]), str(row["arm"])): row for row in n162b.get("models", [])}
    require(
        set(n155_candidates) == set(SEEDS), "N1-162F requires three N1-155A candidate endpoints"
    )
    require(
        set(n162b_models) == {(seed, arm) for seed in SEEDS for arm in ARMS},
        "N1-162F requires six N1-162B endpoints",
    )
    endpoints = []
    for seed in SEEDS:
        row = n155_candidates[seed]
        endpoints.append(
            {
                "seed": seed,
                "lineage": "n155a_candidate",
                "path": str(Path(row["final_checkpoint_path"]).resolve()),
                "sha256": row["final_checkpoint_sha256"],
                "fingerprint": row["final_bundle_fingerprint"],
                "expected_total_updates": row["total_updates"],
            }
        )
    for seed in SEEDS:
        for arm in ARMS:
            row = n162b_models[(seed, arm)]
            endpoints.append(
                {
                    "seed": seed,
                    "lineage": arm,
                    "path": str(Path(row["final_checkpoint_path"]).resolve()),
                    "sha256": row["final_checkpoint_sha256"],
                    "fingerprint": row["final_bundle_fingerprint"],
                    "expected_total_updates": row["total_updates"],
                }
            )
    require(len(endpoints) == 9, "N1-162F endpoint denominator differs")
    return (
        source_hashes,
        endpoints,
        {
            "n155a_result": n155a,
            "n162b_result": n162b,
            "n162d_report": n162d,
            "n162e_report": n162e,
        },
    )


def _build_cases(inputs: dict, factorial_helper) -> dict[tuple, dict]:
    cases = {}
    for view in VIEWS:
        transitions = [item for item in inputs["views"][view] if item.get("kind") == "transition"]
        require(len(transitions) == PARENTS_PER_VIEW, f"transition parent count differs in {view}")
        for item in transitions:
            for entry in factorial_helper.variants(item):
                condition = str(entry["condition"])
                case_id = str(entry["case_id"])
                variant = entry["item"]
                initial = str(entry["initial_direction"])
                turn = str(entry["turn"])
                answer = str(variant["answer"])
                require(
                    answer == n162d._turned(initial, turn),
                    f"factorial expected direction differs: {case_id}",
                )
                key = (view, str(item["id"]), condition, case_id)
                if key in cases:
                    raise Refuse(f"duplicate N1-162F transition case: {key}")
                cases[key] = {
                    "view": view,
                    "parent_id": str(item["id"]),
                    "condition": condition,
                    "case_id": case_id,
                    "initial_direction": initial,
                    "turn": turn,
                    "expected_answer": answer,
                    "item": variant,
                }
    require(len(cases) == CASES_PER_ENDPOINT, "N1-162F transition case count differs")
    return cases


def _baseline_outputs(context: dict, cases: dict[tuple, dict]) -> dict[tuple, dict]:
    n155a_rows = []
    with N155A_PROGRESS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row.get("event") == "original_train_gate_row"
                and row.get("branch") == "unknown8_rebalanced"
                and row.get("kind") == "transition"
            ):
                n155a_rows.append(row)
    require(
        len(n155a_rows) == 3 * TRANSITION_GATE_ROWS_PER_ENDPOINT,
        "N1-155A candidate transition gate denominator differs",
    )
    baseline = {}
    for row in n155a_rows:
        view = str(row["evaluation_set"])
        base_view = view.removeprefix("factorial_") if view.startswith("factorial_") else view
        key = (
            int(row["seed"]),
            "n155a_candidate",
            base_view,
            str(row["id"]),
            str(row.get("condition") or "original"),
            str(row.get("case_id") or f"{row['id']}:original"),
        )
        case_key = (
            base_view,
            str(row["id"]),
            str(row.get("condition") or "original"),
            str(row.get("case_id") or f"{row['id']}:original"),
        )
        case = cases.get(case_key)
        require(case is not None, f"N1-155A baseline transition key missing: {case_key}")
        row_baseline = {
            "raw_hex": row["raw_hex"],
            "stop_reason": row["stop_reason"],
            "output_type": n162d._classify_output(
                row["raw_hex"],
                row["stop_reason"],
                case["expected_answer"],
                case["initial_direction"],
                case["turn"],
            ),
            "free_exact": row["free_exact"],
        }
        if key in baseline:
            require(
                baseline[key] == row_baseline,
                f"N1-155A base/factorial original outputs differ: {key}",
            )
        baseline[key] = row_baseline
    n162d_rows = context["n162d_report"]["transition_rows_by_endpoint"]
    for row in n162d_rows:
        view = str(row["evaluation_set"])
        base_view = view.removeprefix("factorial_") if view.startswith("factorial_") else view
        key = (
            int(row["seed"]),
            str(row["arm"]),
            base_view,
            str(row["id"]),
            str(row.get("condition") or "original"),
            str(row.get("case_id") or f"{row['id']}:original"),
        )
        case_key = (
            base_view,
            str(row["id"]),
            str(row.get("condition") or "original"),
            str(row.get("case_id") or f"{row['id']}:original"),
        )
        case = cases.get(case_key)
        require(case is not None, f"N1-162B baseline transition key missing: {case_key}")
        row_baseline = {
            "raw_hex": row["raw_hex"],
            "stop_reason": row["stop_reason"],
            "output_type": row["output_type"],
            "free_exact": row["free_exact_stored"],
        }
        if key in baseline:
            require(
                baseline[key] == row_baseline,
                f"N1-162B base/factorial original outputs differ: {key}",
            )
        baseline[key] = row_baseline
    require(
        len(baseline) == len(ENDPOINTS) * CASES_PER_ENDPOINT,
        "N1-162F original-question baselines do not cover nine endpoints",
    )
    return baseline


def _summarize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(Counter)
    paired: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (
            row["lineage"],
            row["seed"],
            row["view"],
            row["parent_id"],
            row["condition"],
            row["case_id"],
        )
        pair = paired.setdefault(key, {})
        template = str(row["question_template_id"])
        require(template not in pair, f"duplicate template probe row: {key}/{template}")
        pair[template] = row
        group = (row["lineage"], row["seed"], row["view"], row["condition"], template)
        counts = groups[group]
        counts["items"] += 1
        counts["free_exact"] += row["free_exact"] is True
        counts[str(row["output_type"])] += 1
    require(
        all(set(pair) == set(TEMPLATES) for pair in paired.values()),
        "N1-162F two-template pair coverage differs",
    )
    summary = []
    for key, counts in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        lineage, seed, view, condition, template = key
        row = {
            "lineage": lineage,
            "seed": seed,
            "view": view,
            "condition": condition,
            "question_template_id": template,
        }
        row.update({name: int(value) for name, value in sorted(counts.items())})
        summary.append(row)
    paired_summary = []
    type_transitions = Counter()
    for key, pair in paired.items():
        a = pair["direction_after_turning"]
        b = pair["direction_after_turn"]
        type_transitions[(key[0], key[1], key[2], a["output_type"], b["output_type"])] += 1
        paired_summary.append(
            {
                "lineage": key[0],
                "seed": key[1],
                "view": key[2],
                "parent_id": key[3],
                "condition": key[4],
                "case_id": key[5],
                "direction_after_turning_type": a["output_type"],
                "direction_after_turn_type": b["output_type"],
                "direction_after_turning_exact": a["free_exact"],
                "direction_after_turn_exact": b["free_exact"],
                "raw_output_changed": a["raw_hex"] != b["raw_hex"],
                "original_question_type": a["original_question_output_type"],
                "original_question_exact": a["original_question_free_exact"],
            }
        )
    return summary, paired_summary


def analyze() -> dict:
    started = time.process_time()
    if any(path.exists() for path in (ATTEMPT, PROGRESS, RESULT)):
        raise FileExistsError("PLAN-N1-162F outputs are write-once")
    source_hashes, endpoints, context = _source_and_endpoints()
    source_hashes[str(TEST.resolve())] = sha256(TEST)
    write_new(
        ATTEMPT,
        {
            "schema": "n1-plan162f-transition-template-crossover-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "n155a_result_sha256": N155A_RESULT_SHA256,
            "n162b_result_sha256": N162B_RESULT_SHA256,
            "planned_endpoints": len(endpoints),
            "planned_queries": TOTAL_QUERIES,
            "model_queries": 0,
            "parameter_updates": 0,
            "checkpoint_writes": 0,
            "can_promote": False,
        },
    )
    source_hashes[str(ATTEMPT.resolve())] = sha256(ATTEMPT)
    budget = n132.CpuBudget(CPU_LIMIT_SECONDS)
    progress = PROGRESS.open("x", encoding="utf-8", newline="\n")
    restores = []
    output_rows = []
    baseline = {}
    queries = 0
    errors = []
    inputs = None
    try:
        require(
            n150_builder.verify_hashes(source_hashes),
            "N1-162F bound sources/checkpoints changed before restore",
        )
        for endpoint in endpoints:
            path = Path(endpoint["path"]).resolve()
            require(
                path.is_file() and sha256(path) == endpoint["sha256"],
                f"N1-162F endpoint checkpoint SHA changed: {path}",
            )
            restored = _independent_restore(path, budget)
            require(
                restored["fingerprint"] == endpoint["fingerprint"]
                and restored["seed"] == endpoint["seed"],
                f"N1-162F independent restore identity mismatch: {endpoint['lineage']}/{endpoint['seed']}",
            )
            restores.append({**endpoint, "restore": restored})
        require(len(restores) == 9, "N1-162F independent restore count differs")
        bound = n132._build_context(load_inputs=True)
        inputs = bound["inputs"]
        require(
            inputs is not None and len(inputs.get("items", [])) == 192,
            "N1-162F requires the frozen v3 train inputs",
        )
        g = bound["g"]
        factorial = g["calibration"].engineering.factorial
        cases = _build_cases(inputs, factorial)
        baseline = _baseline_outputs(context, cases)
        for endpoint in endpoints:
            path = Path(endpoint["path"]).resolve()
            require(
                path.is_file() and sha256(path) == endpoint["sha256"],
                f"N1-162F checkpoint changed before scoring: {path}",
            )
            model, optimizer, payload, fingerprint = _restore(path)
            require(
                fingerprint == endpoint["fingerprint"],
                f"N1-162F model bundle fingerprint changed: {endpoint['lineage']}/{endpoint['seed']}",
            )
            before = _fingerprint(model, optimizer)
            endpoint_rows = []
            for case in cases.values():
                item = case["item"]
                entity = n132._query_subject(item["question"])
                source_key = (
                    int(endpoint["seed"]),
                    str(endpoint["lineage"]),
                    str(case["view"]),
                    str(case["parent_id"]),
                    str(case["condition"]),
                    str(case["case_id"]),
                )
                base = baseline.get(source_key)
                require(
                    base is not None,
                    f"N1-162F sealed original-question baseline missing: {source_key}",
                )
                for template_id, template in TEMPLATES.items():
                    budget.guard(reserve=4)
                    query_item = dict(item)
                    query_item["question"] = template.format(entity=entity)
                    session = f"n1-plan162f-{endpoint['lineage']}-{endpoint['seed']}-{case['view']}-{case['parent_id']}-{case['condition']}-{template_id}"
                    if session in model._observations:
                        raise Refuse(f"N1-162F session collides with frozen history: {session}")
                    queries += 1
                    try:
                        measured = g["calibration"].runner.core.measure(
                            model, query_item, "position", session
                        )
                    finally:
                        model._observations.pop(session, None)
                    output_type = n162d._classify_output(
                        measured["raw_hex"],
                        measured["stop_reason"],
                        case["expected_answer"],
                        case["initial_direction"],
                        case["turn"],
                    )
                    require(
                        bool(measured["free_exact"]) == (output_type == "correct_exact"),
                        f"N1-162F measured free_exact differs from output type: {session}",
                    )
                    row = {
                        "lineage": endpoint["lineage"],
                        "seed": endpoint["seed"],
                        "view": case["view"],
                        "parent_id": case["parent_id"],
                        "condition": case["condition"],
                        "case_id": case["case_id"],
                        "question_template_id": template_id,
                        "question": query_item["question"],
                        "initial_direction": case["initial_direction"],
                        "turn": case["turn"],
                        "expected_answer": case["expected_answer"],
                        "raw_hex": measured["raw_hex"],
                        "output_hex": measured["output_hex"],
                        "stop_reason": measured["stop_reason"],
                        "free_exact": measured["free_exact"],
                        "teacher_exact": measured["teacher_exact"],
                        "output_type": output_type,
                        "original_question_raw_hex": base["raw_hex"],
                        "original_question_free_exact": base["free_exact"],
                        "original_question_output_type": base["output_type"],
                    }
                    endpoint_rows.append(row)
                    output_rows.append(row)
                    write_jsonl(progress, {"event": "transition_template_query", **row})
                    if queries % 64 == 0:
                        progress.flush()
            require(
                len(endpoint_rows) == QUERIES_PER_ENDPOINT,
                f"N1-162F endpoint query count differs: {endpoint['lineage']}/{endpoint['seed']}",
            )
            require(
                _fingerprint(model, optimizer) == before,
                f"N1-162F read-only query mutated bundle: {endpoint['lineage']}/{endpoint['seed']}",
            )
            del model, optimizer, payload
            if not budget.accounting_complete:
                raise Refuse("N1-162F child CPU accounting incomplete")
        require(
            queries == TOTAL_QUERIES and len(output_rows) == TOTAL_QUERIES,
            "N1-162F total query denominator differs",
        )
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        progress.flush()
        progress.close()

    summary, template_pairs = _summarize(output_rows) if output_rows else ([], [])
    input_hashes = dict(source_hashes)
    for path in (ATTEMPT, PROGRESS):
        if path.is_file():
            input_hashes[str(path.resolve())] = sha256(path)
    unchanged = n150_builder.verify_hashes(input_hashes)
    if not unchanged:
        errors.append("N1-162F source/checkpoint hash changed")
    cpu = time.process_time() - started + budget.children
    stage_cpu = cpu
    linked_cpu = PRIOR_LINKED_CPU_SECONDS + stage_cpu
    complete = (
        not errors
        and queries == TOTAL_QUERIES
        and len(restores) == 9
        and stage_cpu <= CPU_LIMIT_SECONDS
        and linked_cpu <= LINKED_CPU_LIMIT_SECONDS
        and unchanged
    )
    counts = Counter()
    malformed = []
    line_count = 0
    for number, line in enumerate(PROGRESS.open(encoding="utf-8"), 1):
        line_count += 1
        try:
            counts[str(json.loads(line).get("event"))] += 1
        except Exception as exc:
            malformed.append({"line": number, "error": type(exc).__name__})
    if malformed or counts["transition_template_query"] != queries or line_count != queries:
        complete = False
        errors.append(f"N1-162F progress count/malformed mismatch: {dict(counts)}/{len(malformed)}")
    report = {
        "schema": "taiji-n1-plan162f-transition-template-crossover-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "plan_sha256": PLAN_SHA256,
        "n155a_result_sha256": N155A_RESULT_SHA256,
        "n155a_progress_sha256": N155A_PROGRESS_SHA256,
        "n162b_result_sha256": N162B_RESULT_SHA256,
        "n162b_progress_sha256": N162B_PROGRESS_SHA256,
        "n162d_report_sha256": N162D_REPORT_SHA256,
        "n162e_report_sha256": N162E_REPORT_SHA256,
        "source_hashes": {
            path: value
            for path, value in input_hashes.items()
            if value != "not_yet_written" and value is not None
        },
        "source_hashes_unchanged": unchanged,
        "checkpoint_restores": restores,
        "endpoint_count": len(endpoints),
        "queries": queries,
        "queries_per_endpoint": QUERIES_PER_ENDPOINT,
        "output_rows": output_rows,
        "summary_by_endpoint_view_condition_template": summary,
        "paired_template_type_transitions": template_pairs,
        "progress_sha256": sha256(PROGRESS),
        "progress_line_count": line_count,
        "progress_event_counts": dict(sorted(counts.items())),
        "independent_restore_count": len(restores),
        "scoring_model_loads": len(endpoints) if not errors else None,
        "checkpoint_loads": len(restores)
        + (len(endpoints) if len(restores) == len(endpoints) else 0),
        "inference_calls": queries,
        "model_queries": queries,
        "parameter_updates": 0,
        "checkpoint_writes": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "execution_cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "prior_linked_cpu_seconds": PRIOR_LINKED_CPU_SECONDS,
        "cumulative_linked_cpu_seconds": round(linked_cpu, 6),
        "linked_cpu_limit_seconds": LINKED_CPU_LIMIT_SECONDS,
        "errors": errors,
        "can_promote": False,
        "summary_zh": f"N1-162F transition模板交叉诊断{'完整' if complete else '停止'}：{queries}/{TOTAL_QUERIES}条train-only查询、九checkpoint恢复{len(restores)}/9，CPU{cpu:.3f}/{CPU_LIMIT_SECONDS:.0f}秒，0更新/持久checkpoint写入/新面晋级。",
    }
    write_new(RESULT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLAN-N1-162F train-only transition template crossover"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--execute-approved", action="store_true")
    modes.add_argument("--restore-worker", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker is not None:
        result = _restore_worker(args.restore_worker)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0 if result.get("complete") is True and result.get("equal") is True else 1
    result = analyze()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "execution_cpu_seconds",
                    "cumulative_linked_cpu_seconds",
                    "endpoint_count",
                    "queries",
                    "queries_per_endpoint",
                    "parameter_updates",
                    "can_promote",
                    "errors",
                    "summary_zh",
                )
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
