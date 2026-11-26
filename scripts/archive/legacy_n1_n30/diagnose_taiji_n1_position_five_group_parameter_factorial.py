"""PLAN-N1-115: five-group parameter factorial on the consumed train face."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = (
    ROOT / "plans/reference/PLAN-N1-115_five_group_parameter_factorial_diagnostic_draft_20261018.md"
)
REPORT = ROOT / "reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
READINESS = (
    ROOT / "reports/taiji_n1_position_five_group_parameter_factorial_s13_20261018_readiness.json"
)
BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_length_balanced_training_20261018.json"
BASELINE_CHECKPOINT = (
    ROOT / "data/checkpoints/n1_hard_neighbor_training_v1_20261017/s13_position_final.pt"
)
CANDIDATE_CHECKPOINT = (
    ROOT
    / "data/checkpoints/n1_hard_neighbor_length_balanced_training_20261018/s13_position_final.pt"
)
FACTORIAL_MANIFEST = ROOT / "plans/manifests/n1_fresh_transition_factorial_20261016.json"
R1_REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json"
CELL_CORRECTION = (
    ROOT
    / "reports/taiji_n1_position_weight_history_crossover_s13_r1_cell_summary_correction_20261018.json"
)
SESSION_SCOPE = (
    ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_session_scope_20261018.json"
)
R1_SOURCE_ERRATUM = (
    ROOT
    / "reports/taiji_n1_position_weight_history_crossover_s13_r1_postrun_source_erratum_20261018.json"
)

EXPECTED = {
    BASELINE_REPORT: "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909",
    CANDIDATE_REPORT: "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c",
    BASELINE_CHECKPOINT: "09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653",
    CANDIDATE_CHECKPOINT: "d215a51d571976f216af43d0b7372f3cb702b6258237a7144252df4609f3b881",
    FACTORIAL_MANIFEST: "266afd908ee6ae09ac26a1b208a1a16c4d9f793d1f10443888862a2f353a63a3",
    R1_REPORT: "e38b08a857ffe69acb8fbf57e040ed672df538b88988dc333e1182e58ffb16bf",
    CELL_CORRECTION: "cf5e18243411c7314beeb8fac8d1f7e44cc3e6c7cdd5577de5d89ad5d709203e",
    SESSION_SCOPE: "0453d8154fb70e9dbd2cbfc2e1ee7b7813a205c7ae91fee5b225b698bd566811",
    R1_SOURCE_ERRATUM: "8aa10e996e2897ff1212d3a59976bb368f7f593a18bd438c744a6bd268af8b5c",
}
CPU_LIMIT_SECONDS = 600.0
SEED = 13
ARM = "position"
PARENTS = 48
UPDATES = 24_576
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
VIEWS = ("original", "rename")
GROUPS = ("reader", "writer", "local", "context", "relation")
MASKS = tuple(range(32))
HYBRID_MASKS = tuple(range(1, 31))
REUSED_MASKS = (0, 31)
QUERIES_PER_STATE = PARENTS * len(CONDITIONS) * len(VIEWS)
NEW_QUERIES = len(HYBRID_MASKS) * QUERIES_PER_STATE
TOTAL_STATE_ROWS = len(MASKS) * QUERIES_PER_STATE

PROTOCOL = {
    "schema": "n1-position-five-group-parameter-factorial-v1",
    "seed": SEED,
    "arm": ARM,
    "parameter_groups": list(GROUPS),
    "group_sources": ["run100_baseline", "n1_110_candidate"],
    "group_masks": len(MASKS),
    "reused_r1_anchor_masks": list(REUSED_MASKS),
    "new_hybrid_masks": list(HYBRID_MASKS),
    "parents_per_state": PARENTS,
    "conditions": list(CONDITIONS),
    "views": list(VIEWS),
    "rows_per_state": QUERIES_PER_STATE,
    "reused_anchor_rows": len(REUSED_MASKS) * QUERIES_PER_STATE,
    "model_queries": NEW_QUERIES,
    "total_rows_in_report": TOTAL_STATE_ROWS,
    "checkpoint_payload_reads": 2,
    "model_loads": 2,
    "hybrid_model_constructions": 1,
    "parameter_updates": 0,
    "optimizer_steps": 0,
    "checkpoint_saves": 0,
    "dev_queries": 0,
    "holdout_queries": 0,
    "confirmation_queries": 0,
    "cpu_limit_seconds": CPU_LIMIT_SECONDS,
    "retry_allowed": False,
    "can_promote": False,
}

CODE_PATHS = (
    ROOT / "scripts/training/diagnose_taiji_n1_position_weight_history_crossover.py",
    ROOT / "scripts/training/diagnose_taiji_n1_fresh_transition_factorial.py",
    ROOT / "scripts/training/run_taiji_n1_hard_neighbor_training.py",
    ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py",
    ROOT / "scripts/training/run_taiji_n1_transition_factorial_calibration.py",
    ROOT / "scripts/training/audit_taiji_n1_transition_factorial_paired.py",
    ROOT / "scripts/training/_taiji_n1_evidence_shard.py",
    ROOT / "scripts/training/_taiji_n1_group_relative_adam.py",
    ROOT / "scripts/training/run_taiji_n1_relation_paired.py",
    ROOT / "scripts/training/run_taiji_n1_local_relation.py",
    ROOT / "scripts/training/run_taiji_n1_contextual_question.py",
    ROOT / "scripts/training/run_taiji_n1_local_question.py",
    ROOT / "taiji/native_local_relation_evidence_candidate.py",
    ROOT / "taiji/native_contextual_question_evidence_candidate.py",
    ROOT / "taiji/native_local_question_evidence_candidate.py",
    ROOT / "taiji/native_question_aligned_evidence_candidate.py",
    ROOT / "taiji/native_evidence_candidate.py",
    ROOT / "taiji/position_evidence_reference.py",
    ROOT / "taiji/contracts.py",
    ROOT / "taiji/utf8_state.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def source_guards(extra_paths: list[Path] | None = None) -> dict[str, str]:
    paths = [*EXPECTED, *CODE_PATHS, Path(__file__).resolve(), *(extra_paths or [])]
    result = {}
    for path in dict.fromkeys(Path(path).resolve() for path in paths):
        if not path.is_file():
            raise FileNotFoundError(f"PLAN-N1-115 source missing: {path}")
        result[str(path)] = sha256_file(path)
    for path, expected in EXPECTED.items():
        if result[str(path.resolve())] != expected:
            raise ValueError(f"PLAN-N1-115 frozen source hash mismatch: {path}")
    return result


def guards_unchanged(guards: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in guards.items()
    )


def selected_model(report: dict, *, expected_sufficient: bool) -> dict:
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("can_promote") is not False
        or report.get("training_sufficient") is not expected_sufficient
        or any(
            report.get(key) != 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("complete train-only non-promotion endpoint report required")
    selected = [
        item for item in report["models"] if (item.get("seed"), item.get("arm")) == (SEED, ARM)
    ]
    if len(selected) != 1:
        raise ValueError("one seed13/position endpoint required")
    model = selected[0]
    if (
        model.get("complete") is not True
        or model.get("completed_updates") != UPDATES
        or model.get("training_sufficient") is not expected_sufficient
        or len(model.get("restore_checks", [])) != 2
        or any(check.get("equal") is not True for check in model["restore_checks"])
    ):
        raise ValueError("full independently restored endpoint required")
    return model


def validate_r1_inputs(
    r1: dict, cell_correction: dict, session_scope: dict, r1_source_erratum: dict
) -> dict:
    if (
        r1.get("complete") is not True
        or r1.get("status") != "complete"
        or r1.get("model_queries") != 1_536
        or r1.get("native_reproductions") != 768
        or r1.get("parameter_updates") != 0
        or r1.get("checkpoint_saves") != 0
        or r1.get("can_promote") is not False
        or any(
            r1.get(key) != 0 for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        or len(r1.get("rows", [])) != 1_536
        or len(r1.get("anchors", [])) != 2
        or any(anchor.get("passed") is not True for anchor in r1.get("anchors", []))
    ):
        raise ValueError("complete train-only r1 report required")
    if (
        cell_correction.get("status") != "complete"
        or cell_correction.get("source_report", {}).get("sha256") != EXPECTED[R1_REPORT]
        or cell_correction.get("model_queries") != 0
    ):
        raise ValueError("valid r1 cell correction required")
    predictions = session_scope.get("row_comparison", {})
    if any(
        not predictions.get(weight, {}).get("all_rows_identical")
        or predictions.get(weight, {}).get("identical_prediction_rows_by_view")
        != {"original": 192, "rename": 192}
        for weight in ("baseline_weights", "candidate_weights")
    ):
        raise ValueError("history-source invariance must be sealed before fixing baseline history")
    if r1_source_erratum.get("source_r1_report", {}).get("sha256") != EXPECTED[R1_REPORT]:
        raise ValueError("r1 post-run source erratum mismatch")
    return r1


def validate_anchor_rows(r1: dict, r1_module) -> dict[int, list[dict]]:
    expected_combinations = {
        "baseline_weights+baseline_history",
        "baseline_weights+candidate_history",
        "candidate_weights+baseline_history",
        "candidate_weights+candidate_history",
    }
    if r1["model_queries"] != 1_536 or len(r1["rows"]) != 1_536:
        raise ValueError("r1 report must contain the exact1536 completed output rows")
    if {row.get("combination") for row in r1["rows"]} != expected_combinations:
        raise ValueError("r1 report must contain all four frozen weight/history combinations")
    if any(
        sum(row.get("combination") == combination for row in r1["rows"]) != QUERIES_PER_STATE
        for combination in expected_combinations
    ):
        raise ValueError("each r1 combination must contain384 rows")
    anchors = {}
    for mask, combination in (
        (0, "baseline_weights+baseline_history"),
        (31, "candidate_weights+baseline_history"),
    ):
        rows = [row for row in r1["rows"] if row.get("combination") == combination]
        if len(rows) != QUERIES_PER_STATE:
            raise ValueError(f"r1 anchor {combination} must contain384 sealed rows")
        summary = r1_module.summarize_combination(rows)
        if any(summary[view]["items"] != 192 or summary[view]["parents"] != 48 for view in VIEWS):
            raise ValueError("r1 anchors must cover 48 parents in both views")
        anchors[mask] = rows
    return anchors


def report_rows(rows: list[dict], *, mask: int, source: str) -> list[dict]:
    return [
        dict(
            row,
            parameter_state_mask=mask,
            state_id=f"mask-{mask:02d}",
            group_sources=group_sources(mask),
            query_origin=source,
        )
        for row in rows
    ]


def group_sources(mask: int) -> dict[str, str]:
    return {
        group: "candidate" if mask & (1 << index) else "baseline"
        for index, group in enumerate(GROUPS)
    }


def compose_state(
    baseline_state: dict, candidate_state: dict, group_by_name: dict[str, str], mask: int
) -> dict:
    if set(baseline_state) != set(candidate_state) or set(baseline_state) != set(group_by_name):
        raise ValueError("mixed group state requires identical complete parameter keys")
    sources = group_sources(mask)
    for name, group in group_by_name.items():
        if group not in sources:
            raise ValueError(f"parameter maps outside the five frozen groups: {name} -> {group}")
        baseline = baseline_state[name]
        candidate = candidate_state[name]
        if baseline.shape != candidate.shape or baseline.dtype != candidate.dtype:
            raise ValueError(f"checkpoint parameter tensor shape/dtype differs: {name}")
    return {
        name: candidate_state[name] if sources[group] == "candidate" else baseline_state[name]
        for name, group in group_by_name.items()
    }


def factorial_attribution(summaries: dict[int, dict]) -> dict:
    def shapley(scores: dict[int, float], factor_index: int) -> float:
        contribution = 0.0
        bit = 1 << factor_index
        for mask in MASKS:
            if mask & bit:
                continue
            subset_size = mask.bit_count()
            weight = (
                math.factorial(subset_size)
                * math.factorial(len(GROUPS) - subset_size - 1)
                / math.factorial(len(GROUPS))
            )
            contribution += weight * (scores[mask | bit] - scores[mask])
        return contribution

    result = {}
    for view in VIEWS:
        metrics = {
            "joint_four_exact": {mask: summaries[mask][view]["joint_four_exact"] for mask in MASKS}
        }
        for condition in CONDITIONS:
            metrics[f"condition/{condition}"] = {
                mask: summaries[mask][view]["by_condition"][condition] for mask in MASKS
            }
        result[view] = {}
        for metric, scores in metrics.items():
            result[view][metric] = {
                "baseline_all_groups": scores[0],
                "candidate_all_groups": scores[31],
                "total_delta": scores[31] - scores[0],
                "groups": {
                    group: {
                        "mean_edge_delta": statistics.fmean(
                            scores[mask | (1 << index)] - scores[mask]
                            for mask in MASKS
                            if not mask & (1 << index)
                        ),
                        "shapley_delta": shapley(scores, index),
                    }
                    for index, group in enumerate(GROUPS)
                },
            }
    return result


class CpuBudget:
    def __init__(self):
        self.started = time.process_time()

    def used(self) -> float:
        return time.process_time() - self.started

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= CPU_LIMIT_SECONDS:
            raise TimeoutError("PLAN-N1-115 CPU limit reached")


def validate_only() -> dict:
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-115 report/attempt path already occupied")
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen PLAN-N1-115 input mismatch: {path}")
    plan_text = PLAN.read_text(encoding="utf-8")
    if "PLAN-N1-115" not in plan_text or "提案，待 owner 裁定" not in plan_text:
        raise ValueError("frozen pending PLAN-N1-115 draft required for preflight")
    from scripts.training import diagnose_taiji_n1_position_weight_history_crossover as r1_module
    from scripts.training import run_taiji_n1_hard_neighbor_training as base

    r1 = json.loads(R1_REPORT.read_text(encoding="utf-8"))
    correction = json.loads(CELL_CORRECTION.read_text(encoding="utf-8"))
    session_scope = json.loads(SESSION_SCOPE.read_text(encoding="utf-8"))
    source_erratum = json.loads(R1_SOURCE_ERRATUM.read_text(encoding="utf-8"))
    validate_r1_inputs(r1, correction, session_scope, source_erratum)
    anchors = validate_anchor_rows(r1, r1_module)
    loaded, globals_dict, inputs = base.make_runner(load_inputs=True)
    factorial = globals_dict["calibration"].engineering.factorial
    views = inputs["views"]
    entries_by_view = {
        view: [
            entry
            for item in views[view]
            if item["kind"] == "transition"
            for entry in factorial.variants(item)
        ]
        for view in VIEWS
    }
    if any(len(entries_by_view[view]) != 192 for view in VIEWS):
        raise ValueError("each frozen train view must have192 factorial entries")
    manifest = json.loads(FACTORIAL_MANIFEST.read_text(encoding="utf-8"))
    expected_entries = [
        entry
        for item in views["original"]
        if item["kind"] == "transition"
        for entry in factorial.variants(item)
    ]
    if (
        manifest.get("entries") != expected_entries
        or manifest.get("protocol") != factorial.PROTOCOL
    ):
        raise ValueError("factorial manifest differs from the frozen train transition face")
    baseline_record = selected_model(
        json.loads(BASELINE_REPORT.read_text(encoding="utf-8")), expected_sufficient=True
    )
    candidate_record = selected_model(
        json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8")), expected_sufficient=False
    )
    sealed = {
        "baseline": r1_module.read_sealed_factorial_rows(
            baseline_record,
            json.loads(BASELINE_REPORT.read_text(encoding="utf-8")),
            views=views,
            factorial=factorial,
        ),
        "candidate": r1_module.read_sealed_factorial_rows(
            candidate_record,
            json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8")),
            views=views,
            factorial=factorial,
        ),
    }
    anchor_checks = (
        r1_module.check_anchor(
            anchors[0],
            sealed["baseline"][0],
            weight_source="baseline_weights",
            history_source="baseline_history",
        ),
        r1_module.check_anchor(
            anchors[31],
            sealed["candidate"][0],
            weight_source="candidate_weights",
            history_source="baseline_history",
        ),
    )
    if any(check["exact_reproduction"] is not True for check in anchor_checks):
        raise ValueError(
            "reused PLAN-N1-112 all-baseline/all-candidate weight anchors differ from sealed rows"
        )
    guards = source_guards([sealed["baseline"][1], sealed["candidate"][1]])
    if set(anchors) != {0, 31}:
        raise ValueError("two fixed endpoint parameter anchors must be reused")
    readiness = {
        "schema": "n1-position-five-group-parameter-factorial-readiness-v1",
        "ready": True,
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256_at_preflight": sha256_file(PLAN),
        "protocol": PROTOCOL,
        "r1_report_sha256": EXPECTED[R1_REPORT],
        "anchor_rows_reused": 768,
        "anchor_rows_exact_to_sealed_reports": 768,
        "hybrid_states": 30,
        "queries_per_state": QUERIES_PER_STATE,
        "new_model_queries": NEW_QUERIES,
        "checkpoint_payload_reads": 0,
        "model_loads": 0,
        "hybrid_model_constructions": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "train_views": {view: len(entries_by_view[view]) for view in VIEWS},
        "sealed_source_rows": {
            "baseline": len(sealed["baseline"][0]),
            "candidate": len(sealed["candidate"][0]),
        },
        "source_guards": guards,
    }
    if READINESS.exists():
        existing = json.loads(READINESS.read_text(encoding="utf-8"))
        if existing != readiness:
            raise FileExistsError("existing PLAN-N1-115 readiness differs; never overwrite")
    else:
        write_new(READINESS, readiness)
    return readiness


def execute() -> dict:
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-115 is one-shot; report or attempt already exists")
    if "状态：已批准执行" not in PLAN.read_text(encoding="utf-8"):
        raise PermissionError("PLAN-N1-115 query face requires explicit owner approval")
    if not READINESS.is_file():
        raise FileNotFoundError("successful zero-query PLAN-N1-115 readiness is required")
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    if (
        readiness.get("schema") != "n1-position-five-group-parameter-factorial-readiness-v1"
        or readiness.get("ready") is not True
        or readiness.get("new_model_queries") != NEW_QUERIES
        or any(
            readiness.get(key) != 0
            for key in (
                "checkpoint_payload_reads",
                "model_loads",
                "hybrid_model_constructions",
                "model_queries",
                "parameter_updates",
                "optimizer_steps",
                "checkpoint_saves",
            )
        )
        or not guards_unchanged(readiness.get("source_guards", {}))
    ):
        raise ValueError("PLAN-N1-115 readiness source guards changed")
    budget = CpuBudget()
    start_plan_sha = sha256_file(PLAN)
    result = {
        "schema": PROTOCOL["schema"],
        "protocol": PROTOCOL,
        "status": "running",
        "approved_plan_sha256": start_plan_sha,
        "plan_sha256_at_execution_start": start_plan_sha,
        "source_guards": readiness["source_guards"],
        "rows": [],
        "summaries": {},
        "anchors": [],
        "factorial_attribution": {},
        "reused_anchor_rows": 0,
        "model_queries": 0,
        "checkpoint_payload_reads": 0,
        "model_loads": 0,
        "hybrid_model_constructions": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
        "checkpoint_hashes_before": {},
        "checkpoint_hashes_after": {},
    }
    try:
        from scripts.training import run_taiji_n1_hard_neighbor_training as base

        r1_module = __import__(
            "scripts.training.diagnose_taiji_n1_position_weight_history_crossover",
            fromlist=["REPORT"],
        )
        loaded, globals_dict, inputs = base.make_runner(load_inputs=True)
        torch = loaded["torch"]
        torch.set_num_threads(1)
        runner = globals_dict["runner"]
        factorial = globals_dict["calibration"].engineering.factorial
        views = inputs["views"]
        entries_by_view = {
            view: [
                entry
                for item in views[view]
                if item["kind"] == "transition"
                for entry in factorial.variants(item)
            ]
            for view in VIEWS
        }
        if any(len(entries_by_view[view]) != 192 for view in VIEWS):
            raise ValueError("all192 frozen transition variants required per view")
        manifest = json.loads(FACTORIAL_MANIFEST.read_text(encoding="utf-8"))
        expected_entries = [
            entry
            for item in views["original"]
            if item["kind"] == "transition"
            for entry in factorial.variants(item)
        ]
        if (
            manifest.get("entries") != expected_entries
            or manifest.get("protocol") != factorial.PROTOCOL
        ):
            raise ValueError("frozen transition manifest differs from source train items")

        baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        candidate_report = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
        baseline_model_record = selected_model(baseline_report, expected_sufficient=True)
        candidate_model_record = selected_model(candidate_report, expected_sufficient=False)
        baseline_sealed, baseline_shard = r1_module.read_sealed_factorial_rows(
            baseline_model_record, baseline_report, views=views, factorial=factorial
        )
        candidate_sealed, candidate_shard = r1_module.read_sealed_factorial_rows(
            candidate_model_record, candidate_report, views=views, factorial=factorial
        )
        guards = source_guards([baseline_shard, candidate_shard])
        if guards != readiness["source_guards"]:
            raise ValueError("PLAN-N1-115 preflight source guards differ at execution")
        result["source_guards"] = guards
        write_new(
            ATTEMPT,
            {
                "schema": "n1-position-five-group-parameter-factorial-attempt-v1",
                "status": "started",
                "immutable_start_record": True,
                "approved_plan_sha256": start_plan_sha,
                "protocol": PROTOCOL,
                "source_guards": guards,
                "confirmation_face_consumed": False,
            },
        )
        result["checkpoint_hashes_before"] = {
            "baseline": sha256_file(BASELINE_CHECKPOINT),
            "candidate": sha256_file(CANDIDATE_CHECKPOINT),
        }
        r1_report = json.loads(R1_REPORT.read_text(encoding="utf-8"))
        validate_r1_inputs(
            r1_report,
            json.loads(CELL_CORRECTION.read_text(encoding="utf-8")),
            json.loads(SESSION_SCOPE.read_text(encoding="utf-8")),
            json.loads(R1_SOURCE_ERRATUM.read_text(encoding="utf-8")),
        )
        reused = validate_anchor_rows(r1_report, r1_module)
        anchor_checks = [
            r1_module.check_anchor(
                reused[0],
                baseline_sealed,
                weight_source="baseline_weights",
                history_source="baseline_history",
            ),
            r1_module.check_anchor(
                reused[31],
                candidate_sealed,
                weight_source="candidate_weights",
                history_source="baseline_history",
            ),
        ]
        if any(check["exact_reproduction"] is not True for check in anchor_checks):
            raise ValueError("reused weight-state anchors do not reproduce sealed source rows")
        baseline_reused = report_rows(
            reused[0], mask=0, source="r1_reused_baseline_weights_baseline_history"
        )
        candidate_reused = report_rows(
            reused[31], mask=31, source="r1_reused_candidate_weights_baseline_history"
        )
        result["rows"].extend(baseline_reused)
        result["rows"].extend(candidate_reused)
        result["reused_anchor_rows"] = len(baseline_reused) + len(candidate_reused)
        summaries_by_mask = {
            0: r1_module.summarize_combination(baseline_reused),
            31: r1_module.summarize_combination(candidate_reused),
        }
        result["anchors"] = [
            {
                "mask": mask,
                "group_sources": group_sources(mask),
                "reused_rows": len(rows),
                "joint_four_exact": {
                    view: summaries_by_mask[mask][view]["joint_four_exact"] for view in VIEWS
                },
            }
            for mask, rows in ((0, baseline_reused), (31, candidate_reused))
        ]
        result["anchor_reproductions"] = anchor_checks

        budget.guard(reserve=20)
        baseline_payload, baseline_model = r1_module.load_model(
            BASELINE_CHECKPOINT,
            EXPECTED[BASELINE_CHECKPOINT],
            base=globals_dict,
            torch=torch,
            report_model=baseline_model_record,
            result=result,
        )
        candidate_payload, candidate_model = r1_module.load_model(
            CANDIDATE_CHECKPOINT,
            EXPECTED[CANDIDATE_CHECKPOINT],
            base=globals_dict,
            torch=torch,
            report_model=candidate_model_record,
            result=result,
        )
        if type(baseline_model) is not type(candidate_model):
            raise ValueError("frozen endpoint model classes differ")
        baseline_state = baseline_model.state_dict()
        candidate_state = candidate_model.state_dict()
        bounded = globals_dict["engineering"].bounded
        parameter_groups = bounded.grouped_parameters(baseline_model)
        if list(parameter_groups) != list(GROUPS):
            raise ValueError("frozen parameter group order differs from PLAN-N1-115")
        group_by_name = {
            name: group for group, values in parameter_groups.items() for name, _ in values
        }
        if (
            set(group_by_name) != set(baseline_state)
            or set(candidate_state) != set(baseline_state)
            or sum(
                parameter.numel() for values in parameter_groups.values() for _, parameter in values
            )
            != 62_056
        ):
            raise ValueError("all41 updater-group parameters must cover the checkpoint state")

        rng_before = torch.get_rng_state().clone()
        with torch.random.fork_rng():
            hybrid_model = type(baseline_model)(baseline_model.config)
        torch.set_rng_state(rng_before)
        result["hybrid_model_constructions"] = 1
        hybrid_model._observations = baseline_model._observations.copy()
        source_history_keys = frozenset(hybrid_model._observations)
        for mask in HYBRID_MASKS:
            budget.guard(reserve=15)
            hybrid_model.load_state_dict(
                compose_state(baseline_state, candidate_state, group_by_name, mask), strict=True
            )
            state_before = globals_dict["paired"].fingerprint(hybrid_model.state_dict())
            rng_state_before = torch.get_rng_state().clone()
            rows = []
            for view in VIEWS:
                for entry in entries_by_view[view]:
                    budget.guard(reserve=3)
                    session = f"n1-groupfactorial115-mask{mask:02d}-{view}-{entry['case_id']}"
                    if session in hybrid_model._observations:
                        raise ValueError(
                            "factorial measurement session collides with baseline history"
                        )
                    result["model_queries"] += 1
                    try:
                        measured = runner.core.measure(hybrid_model, entry["item"], ARM, session)
                        rows.append(
                            dict(
                                measured,
                                seed=SEED,
                                arm=ARM,
                                view=view,
                                parameter_state_mask=mask,
                                state_id=f"mask-{mask:02d}",
                                group_sources=group_sources(mask),
                                query_origin="new_hybrid_query",
                                **{
                                    key: entry[key]
                                    for key in (
                                        "case_id",
                                        "parent_id",
                                        "condition",
                                        "initial_direction",
                                        "turn",
                                    )
                                },
                            )
                        )
                    finally:
                        hybrid_model._observations.pop(session, None)
            if (
                globals_dict["paired"].fingerprint(hybrid_model.state_dict()) != state_before
                or torch.equal(torch.get_rng_state(), rng_state_before) is not True
                or frozenset(hybrid_model._observations) != source_history_keys
            ):
                raise ValueError("factorial measurement changed parameters, RNG or history keys")
            if len(rows) != QUERIES_PER_STATE:
                raise ValueError("each hybrid state must emit384 rows")
            state_rows = report_rows(rows, mask=mask, source="new_hybrid_query")
            result["rows"].extend(state_rows)
            summaries_by_mask[mask] = r1_module.summarize_combination(state_rows)

        result["summaries"] = {str(mask): summaries_by_mask[mask] for mask in MASKS}
        result["factorial_attribution"] = factorial_attribution(summaries_by_mask)
        result["status"] = "complete"
        result["interpretation_scope"] = (
            "train_only_group_intervention_descriptive_no_generalization_or_promotion"
        )
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "stopped"
        result["error"] = f"{type(error).__name__}: {error}"
    result["cpu_seconds"] = budget.used()
    result["checkpoint_hashes_after"] = {
        "baseline": sha256_file(BASELINE_CHECKPOINT) if BASELINE_CHECKPOINT.is_file() else None,
        "candidate": sha256_file(CANDIDATE_CHECKPOINT) if CANDIDATE_CHECKPOINT.is_file() else None,
    }
    result["checkpoints_unchanged"] = (
        result["checkpoint_hashes_before"] == result["checkpoint_hashes_after"]
    )
    result["sources_unchanged"] = guards_unchanged(result["source_guards"])
    result["unique_state_case_rows"] = len(
        {
            (row.get("parameter_state_mask"), row.get("view"), row.get("case_id"))
            for row in result["rows"]
        }
    )
    code_paths = {Path(__file__).resolve(), *(path.resolve() for path in CODE_PATHS)}
    result["code_unchanged"] = all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in result["source_guards"].items()
        if Path(path) in code_paths
    )
    result["plan_unchanged_during_execution"] = sha256_file(PLAN) == start_plan_sha
    result["complete"] = (
        result["status"] == "complete"
        and result["reused_anchor_rows"] == 768
        and len(result["rows"]) == TOTAL_STATE_ROWS
        and result["unique_state_case_rows"] == TOTAL_STATE_ROWS
        and result["model_queries"] == NEW_QUERIES
        and result["checkpoint_payload_reads"] == 2
        and result["model_loads"] == 2
        and result["hybrid_model_constructions"] == 1
        and len(result.get("anchor_reproductions", [])) == 2
        and all(check.get("exact_reproduction") is True for check in result["anchor_reproductions"])
        and set(result["summaries"]) == {str(mask) for mask in MASKS}
        and result["parameter_updates"] == 0
        and result["optimizer_steps"] == 0
        and result["checkpoint_saves"] == 0
        and all(
            result[key] == 0 for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
        and result["checkpoints_unchanged"]
        and result["sources_unchanged"]
        and result["code_unchanged"]
        and result["plan_unchanged_during_execution"]
        and result["cpu_seconds"] <= CPU_LIMIT_SECONDS
    )
    result["confirmation_face_consumed"] = False
    write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    result = validate_only() if args.validate_only else execute()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
