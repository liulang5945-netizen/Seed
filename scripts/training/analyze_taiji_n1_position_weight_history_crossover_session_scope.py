"""Offline audit of history-factor invariance and session isolation in PLAN-N1-112."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json"
CELL_CORRECTION = (
    ROOT
    / "reports/taiji_n1_position_weight_history_crossover_s13_r1_cell_summary_correction_20261018.json"
)
OUTPUT = (
    ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_session_scope_20261018.json"
)
SOURCE_REPORT_SHA256 = "e38b08a857ffe69acb8fbf57e040ed672df538b88988dc333e1182e58ffb16bf"
PREDICTION_FIELDS = (
    "target_symbols",
    "teacher_symbols",
    "teacher_tokens_correct",
    "teacher_exact",
    "teacher_ce",
    "raw_hex",
    "output_hex",
    "stop_reason",
    "generated_steps",
    "free_exact",
)
WEIGHT_SOURCES = ("baseline_weights", "candidate_weights")
HISTORY_SOURCES = ("baseline_history", "candidate_history")
VIEWS = ("original", "rename")
SOURCE_PATHS = (
    ROOT / "taiji/native_evidence_candidate.py",
    ROOT / "scripts/training/audit_taiji_n1_native_evidence_train_fit.py",
    ROOT / "scripts/training/diagnose_taiji_n1_position_weight_history_crossover.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_history_predictions(rows: list[dict]) -> dict:
    result = {}
    for weight_source in WEIGHT_SOURCES:
        by_history = {}
        for history_source in HISTORY_SOURCES:
            combination = f"{weight_source}+{history_source}"
            selected = [row for row in rows if row.get("combination") == combination]
            mapping = {(row["view"], row["case_id"]): row for row in selected}
            if len(mapping) != 384 or {view for view, _ in mapping} != set(VIEWS):
                raise ValueError("each history source needs384 unique output cases")
            by_history[history_source] = mapping
        left, right = by_history[HISTORY_SOURCES[0]], by_history[HISTORY_SOURCES[1]]
        if set(left) != set(right):
            raise ValueError("history variants must cover the same frozen cases")
        matched = {
            view: sum(
                all(left[key][field] == right[key][field] for field in PREDICTION_FIELDS)
                for key in left
                if key[0] == view
            )
            for view in VIEWS
        }
        result[weight_source] = {
            "cases": 384,
            "identical_prediction_rows_by_view": matched,
            "all_rows_identical": all(matched[view] == 192 for view in VIEWS),
        }
    return result


def session_scope_facts() -> dict:
    candidate_path = SOURCE_PATHS[0]
    candidate_tree = ast.parse(candidate_path.read_text(encoding="utf-8"))
    source_method = next(
        node
        for node in ast.walk(candidate_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "sources"
    )
    source_text = ast.get_source_segment(candidate_path.read_text(encoding="utf-8"), source_method)
    evidence_method = next(
        node
        for node in ast.walk(candidate_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "evidence"
    )
    evidence_text = ast.get_source_segment(
        candidate_path.read_text(encoding="utf-8"), evidence_method
    )

    measure_path = SOURCE_PATHS[1]
    measure_tree = ast.parse(measure_path.read_text(encoding="utf-8"))
    measure_method = next(
        node
        for node in ast.walk(measure_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "measure"
    )
    measure_text = ast.get_source_segment(measure_path.read_text(encoding="utf-8"), measure_method)

    crossover_text = SOURCE_PATHS[2].read_text(encoding="utf-8")
    facts = {
        "sources_select_exact_session_key": "self._observations.get(session, ())" in source_text,
        "evidence_reads_sources_for_same_session": "self.write(self.sources(session), session"
        in evidence_text,
        "evaluator_observes_then_reads_same_session": (
            "paired.observe(model, data, session)" in measure_text
            and "paired.evidence(model, session, arm)" in measure_text
        ),
        "crossover_allocates_fresh_session_and_rejects_collision": (
            "n1-crossover112-" in crossover_text
            and "if session in model._observations" in crossover_text
        ),
    }
    if not all(facts.values()):
        raise ValueError("current session isolation path differs from audited contract")
    return facts


def build_record() -> dict:
    if sha256_file(SOURCE_REPORT) != SOURCE_REPORT_SHA256:
        raise ValueError("frozen r1 report digest differs")
    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    correction = json.loads(CELL_CORRECTION.read_text(encoding="utf-8"))
    if (
        report.get("complete") is not True
        or report.get("model_queries") != 1_536
        or correction.get("source_report", {}).get("sha256") != SOURCE_REPORT_SHA256
    ):
        raise ValueError("complete r1 report and matching cell-summary correction required")
    predictions = compare_history_predictions(report["rows"])
    if not all(entry["all_rows_identical"] for entry in predictions.values()):
        raise ValueError("history swap changed prediction rows; inert-history conclusion invalid")
    source_hashes = {str(path): sha256_file(path) for path in SOURCE_PATHS}
    facts = session_scope_facts()
    if any(sha256_file(Path(path)) != digest for path, digest in source_hashes.items()):
        raise ValueError("session-scope source changed during offline audit")
    return {
        "schema": "n1-position-weight-history-session-scope-audit-v1",
        "status": "complete",
        "source_report": {
            "path": str(SOURCE_REPORT.relative_to(ROOT)),
            "sha256": SOURCE_REPORT_SHA256,
        },
        "cell_correction_report": {
            "path": str(CELL_CORRECTION.relative_to(ROOT)),
            "sha256": sha256_file(CELL_CORRECTION),
        },
        "row_comparison": predictions,
        "session_scope_source_facts": facts,
        "source_hashes": source_hashes,
        "interpretation_boundary": "The r1 outcome tracks checkpoint weights in this fresh-session evaluator. Stored training observation dictionaries are not retrieved across session keys on this path; this does not establish that persistent history is irrelevant in other runtime/session designs.",
        "report_rows_read": len(report["rows"]),
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "can_promote": False,
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("session-scope audit is write-once")
    record = build_record()
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "report": str(OUTPUT.relative_to(ROOT)),
                "model_queries": 0,
                "parameter_updates": 0,
                "history_rows_identical": {
                    weight: item["identical_prediction_rows_by_view"]
                    for weight, item in record["row_comparison"].items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
