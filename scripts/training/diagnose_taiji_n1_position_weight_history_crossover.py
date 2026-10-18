"""PLAN-N1-112: frozen weight/history crossover with format-level restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-112_position_weight_history_crossover_restore_revision_draft_20261018.md"
)
REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_20261018.json"
ATTEMPT = REPORT.with_name(REPORT.stem + "_attempt.json")
READINESS = (
    ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_r1_readiness_20261018.json"
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
ATTRIBUTION_REPORT = ROOT / "reports/taiji_n1_length_balance_failure_attribution_20261018.json"
PREVIOUS_REPORT = ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_20261018.json"
PREVIOUS_ATTEMPT = (
    ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_20261018_attempt.json"
)
PREVIOUS_ERRATUM = (
    ROOT / "reports/taiji_n1_position_weight_history_crossover_s13_20261018_restore_erratum.json"
)

BASELINE_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
CANDIDATE_REPORT_SHA256 = "e848433de597d378e9e1dc52bf488f296f90c5ce67f3855e19bcfe49268aaa0c"
BASELINE_CHECKPOINT_SHA256 = "09df1d47eb490ef9afc5d563d08b40eca6b51951096875d016a67f133f7ed653"
CANDIDATE_CHECKPOINT_SHA256 = "d215a51d571976f216af43d0b7372f3cb702b6258237a7144252df4609f3b881"
FACTORIAL_MANIFEST_SHA256 = "266afd908ee6ae09ac26a1b208a1a16c4d9f793d1f10443888862a2f353a63a3"
ATTRIBUTION_REPORT_SHA256 = "08e80702f0b217e117094c28879f4a62e6f99d6a2315ee38556383afa7d31f4a"
PREVIOUS_REPORT_SHA256 = "388e8ebe7717e5086e1dc8feb746c69854dad88411d152a10fe5c84f731d3a00"
PREVIOUS_ATTEMPT_SHA256 = "3dcca6b75656ff1fec061db1296160f6ed783ccd3ca86920d7e7edbe7d56d61e"
PREVIOUS_ERRATUM_SHA256 = "2ae46031052e5cc5498c427126b8f04591cac97dc0cc9ffd6cd11d74a4c41991"

CPU_LIMIT_SECONDS = 600.0
SEED = 13
ARM = "position"
UPDATES = 24_576
TRAINING_SESSIONS = 43_008
PARENTS = 48
CONDITIONS = ("original", "turn_flip", "direction_opposite", "both")
VIEWS = ("original", "rename")
ANCHOR_COMBINATIONS = (
    ("baseline_weights", "baseline_history"),
    ("candidate_weights", "candidate_history"),
)
HYBRID_COMBINATIONS = (
    ("baseline_weights", "candidate_history"),
    ("candidate_weights", "baseline_history"),
)
ALL_COMBINATIONS = (
    *ANCHOR_COMBINATIONS[:1],
    *HYBRID_COMBINATIONS[:1],
    *HYBRID_COMBINATIONS[1:],
    *ANCHOR_COMBINATIONS[1:],
)
EXPECTED_JOINT = {
    "baseline_weights+baseline_history": {"original": 48, "rename": 48},
    "candidate_weights+candidate_history": {"original": 5, "rename": 4},
}
BASE_FIELDS = (
    "id",
    "kind",
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
ROW_IDENTITY_FIELDS = ("parent_id", "condition", "initial_direction", "turn")

PROTOCOL = {
    "schema": "n1-position-weight-history-crossover-diagnostic-r1-v1",
    "seed": SEED,
    "arm": ARM,
    "updates": UPDATES,
    "parents": PARENTS,
    "views": list(VIEWS),
    "conditions": list(CONDITIONS),
    "combinations": [
        {"weights": weights, "history": history} for weights, history in ALL_COMBINATIONS
    ],
    "queries_per_combination": 384,
    "model_queries": 1_536,
    "native_anchor_queries": 768,
    "native_reproductions_required": 768,
    "checkpoint_payload_reads": 2,
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
    ROOT / "scripts/training/run_taiji_n1_hard_neighbor_training.py",
    ROOT / "scripts/training/run_taiji_n1_transition_factorial_paired.py",
    ROOT / "scripts/training/run_taiji_n1_transition_factorial_calibration.py",
    ROOT / "scripts/training/diagnose_taiji_n1_fresh_transition_factorial.py",
    ROOT / "scripts/training/audit_taiji_n1_transition_factorial_paired.py",
    ROOT / "scripts/training/run_taiji_n1_local_relation.py",
    ROOT / "scripts/training/_taiji_n1_evidence_shard.py",
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
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


class CpuBudget:
    def __init__(self, limit: float):
        self.limit = limit
        self.started = time.process_time()

    def used(self) -> float:
        return time.process_time() - self.started

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-112 cumulative CPU limit reached")


def expected_file_hashes() -> dict[Path, str]:
    return {
        BASELINE_REPORT: BASELINE_REPORT_SHA256,
        CANDIDATE_REPORT: CANDIDATE_REPORT_SHA256,
        BASELINE_CHECKPOINT: BASELINE_CHECKPOINT_SHA256,
        CANDIDATE_CHECKPOINT: CANDIDATE_CHECKPOINT_SHA256,
        FACTORIAL_MANIFEST: FACTORIAL_MANIFEST_SHA256,
        ATTRIBUTION_REPORT: ATTRIBUTION_REPORT_SHA256,
        PREVIOUS_REPORT: PREVIOUS_REPORT_SHA256,
        PREVIOUS_ATTEMPT: PREVIOUS_ATTEMPT_SHA256,
        PREVIOUS_ERRATUM: PREVIOUS_ERRATUM_SHA256,
    }


def source_guards(extra_paths: list[Path] | None = None) -> dict[str, str]:
    paths = [Path(__file__).resolve(), *[path.resolve() for path in CODE_PATHS]]
    paths.extend(path.resolve() for path in expected_file_hashes())
    paths.extend(path.resolve() for path in (extra_paths or []))
    guards = {}
    for path in dict.fromkeys(paths):
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {path}")
        guards[str(path)] = sha256_file(path)
    for path, expected in expected_file_hashes().items():
        if guards[str(path.resolve())] != expected:
            raise ValueError(f"frozen PLAN-N1-112 input changed: {path}")
    return guards


def guards_unchanged(guards: dict[str, str]) -> bool:
    return all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in guards.items()
    )


def selected_model(report: dict, *, sufficient: bool) -> dict:
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("can_promote") is not False
        or any(
            report.get(key) != 0
            for key in ("dev_queries", "holdout_queries", "confirmation_queries")
        )
    ):
        raise ValueError("complete non-promotion source training report required")
    matches = [
        model
        for model in report.get("models", [])
        if (model.get("seed"), model.get("arm")) == (SEED, ARM)
    ]
    if len(matches) != 1:
        raise ValueError("one seed13/position endpoint required")
    model = matches[0]
    if (
        model.get("completed_updates") != UPDATES
        or model.get("complete") is not True
        or model.get("training_sufficient") is not sufficient
        or model.get("new_session_count") != TRAINING_SESSIONS
        or len(model.get("restore_checks", [])) != 2
    ):
        raise ValueError("expected frozen seed13/position training endpoint required")
    final = [
        check
        for check in model["restore_checks"]
        if Path(check["path"]).name == "s13_position_final.pt"
    ]
    if len(final) != 1 or final[0].get("equal") is not True:
        raise ValueError("independently restored final checkpoint required")
    expected_joint = EXPECTED_JOINT[
        "baseline_weights+baseline_history" if sufficient else "candidate_weights+candidate_history"
    ]
    milestone = model.get("milestones", [])[-1]
    if milestone.get("completed_updates") != UPDATES:
        raise ValueError("sealed final training milestone required")
    for view, count in expected_joint.items():
        if milestone.get("factorial", {}).get(view, {}).get("joint_four_exact") != count:
            raise ValueError("sealed anchor joint-four score differs from PLAN-N1-112")
    return model


def read_sealed_factorial_rows(model: dict, report: dict, *, views, factorial):
    from scripts.training import _taiji_n1_evidence_shard as shard

    seal = model.get("shard")
    if not isinstance(seal, dict) or seal.get("counts") != {
        "losses": 24_576,
        "measurements": 24_576,
        "rows": 4_800,
    }:
        raise ValueError("full frozen training evidence shard required")
    identity = {"seed": SEED, "arm": ARM, "protocol": report["protocol"], "phase": "complete"}
    if seal.get("identity") != identity:
        raise ValueError("sealed shard identity does not match its report endpoint")
    shard_path = Path(seal["path"]).resolve()
    if not shard_path.is_relative_to(ROOT.resolve()):
        raise ValueError("training shard must resolve inside the repository")
    record = shard.read_checked(shard_path, seal, identity=identity)
    expected = {
        (f"factorial_{view}", entry["case_id"])
        for view in VIEWS
        for item in views[view]
        if item["kind"] == "transition"
        for entry in factorial.variants(item)
    }
    rows = {
        (row["evaluation_set"], row["case_id"]): row
        for row in record["rows"]
        if row.get("completed_updates") == UPDATES
        and row.get("evaluation_set") in {"factorial_original", "factorial_rename"}
        and row.get("kind") == "transition"
    }
    if set(rows) != expected or len(rows) != 384:
        raise ValueError("sealed source shard lacks all384 final transition factorial rows")
    summary = model["milestones"][-1]["factorial"]
    for view in VIEWS:
        view_rows = [
            rows[(f"factorial_{view}", case_id)]
            for key_view, case_id in expected
            if key_view == f"factorial_{view}"
        ]
        observed = factorial_quartet_summary(view_rows)
        expected_summary = summary[view]
        if any(
            observed[key] != expected_summary[key]
            for key in ("items", "parents", "by_condition", "joint_four_exact")
        ):
            raise ValueError("sealed shard factorial rows do not reproduce the report summary")
    return rows, shard_path


def factorial_quartet_summary(rows: list[dict]) -> dict:
    if len(rows) != 192 or len({(row["parent_id"], row["condition"]) for row in rows}) != 192:
        raise ValueError("exactly192 unique four-condition rows required")
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row["parent_id"], {})[row["condition"]] = row
    if len(grouped) != PARENTS or any(set(group) != set(CONDITIONS) for group in grouped.values()):
        raise ValueError("all48 parents must contain the four frozen conditions")
    return {
        "items": len(rows),
        "parents": len(grouped),
        "by_condition": {
            condition: sum(bool(row["free_exact"]) for row in rows if row["condition"] == condition)
            for condition in CONDITIONS
        },
        "joint_four_exact": sum(
            all(group[condition]["free_exact"] for condition in CONDITIONS)
            for group in grouped.values()
        ),
    }


def summarize_combination(rows: list[dict]) -> dict:
    if len(rows) != 384:
        raise ValueError("each weight/history combination must have384 rows")
    keys = {(row["view"], row["case_id"]) for row in rows}
    if len(keys) != len(rows) or {view for view, _ in keys} != set(VIEWS):
        raise ValueError(
            "each combination requires all384 unique cases across original/rename views"
        )
    summaries = {}
    for view in VIEWS:
        view_rows = [row for row in rows if row["view"] == view]
        summary = factorial_quartet_summary(view_rows)
        parent_rows: dict[str, dict[str, dict]] = {}
        for row in view_rows:
            parent_rows.setdefault(row["parent_id"], {})[row["condition"]] = row
        if len(parent_rows) != PARENTS or any(
            set(group) != set(CONDITIONS) for group in parent_rows.values()
        ):
            raise ValueError("all48 parent condition groups required for cell attribution")
        source_cells = {}
        for parent, group in parent_rows.items():
            source = group["original"]
            direction, turn = source["initial_direction"], source["turn"]
            if direction not in ("north", "east", "south", "west") or turn not in (
                "left",
                "right",
            ):
                raise ValueError("original condition must define each parent direction-turn cell")
            source_cells.setdefault(f"{direction}/{turn}", set()).add(parent)
        cell_summaries = {}
        for direction in ("north", "east", "south", "west"):
            for turn in ("left", "right"):
                cell_parents = source_cells.get(f"{direction}/{turn}", set())
                cell_summaries[f"{direction}/{turn}"] = {
                    "items": len(cell_parents) * len(CONDITIONS),
                    "parents": len(cell_parents),
                    "by_condition": {
                        condition: sum(
                            parent_rows[parent][condition]["free_exact"]
                            for parent in cell_parents
                        )
                        for condition in CONDITIONS
                    },
                    "joint_four_exact": sum(
                        all(
                            parent_rows[parent][condition]["free_exact"]
                            for condition in CONDITIONS
                        )
                        for parent in cell_parents
                    ),
                }
        summary["cells"] = cell_summaries
        summaries[view] = summary
    return summaries


def is_sufficient(summary: dict, gate: int = 44) -> bool:
    return all(summary[view]["joint_four_exact"] >= gate for view in VIEWS)


def interpretation(summaries: dict[str, dict]) -> str:
    baseline_weights_candidate_history = is_sufficient(
        summaries["baseline_weights+candidate_history"]
    )
    candidate_weights_baseline_history = is_sufficient(
        summaries["candidate_weights+baseline_history"]
    )
    if baseline_weights_candidate_history and not candidate_weights_baseline_history:
        return "evidence_toward_candidate_network_weights"
    if candidate_weights_baseline_history and not baseline_weights_candidate_history:
        return "evidence_toward_candidate_observation_history"
    return "interaction_or_2x2_not_separating_sources"


def report_entry(rows, *, weight_source: str, history_source: str):
    combination = f"{weight_source}+{history_source}"
    return [
        dict(
            row, combination=combination, weight_source=weight_source, history_source=history_source
        )
        for row in rows
    ]


def validate_checkpoint_envelope(payload: dict, report_model: dict) -> None:
    protocol = report_model.get("shard", {}).get("identity", {}).get("protocol")
    if (
        payload.get("schema") != "n1-transition-factorial-paired-training-checkpoint-v1"
        or payload.get("seed") != SEED
        or payload.get("arm") != ARM
        or payload.get("completed_updates") != UPDATES
        or payload.get("phase") != "final"
        or not isinstance(protocol, dict)
        or payload.get("protocol") != protocol
        or payload.get("initialization") != "full-fresh-no-inheritance-v1"
        or not isinstance(payload.get("candidate"), dict)
        or not isinstance(payload.get("expected_signature"), dict)
    ):
        raise ValueError("checkpoint envelope does not match its own sealed endpoint report")


def load_model(
    path: Path,
    expected_sha: str,
    *,
    base,
    torch,
    report_model: dict,
    result: dict,
):
    if sha256_file(path) != expected_sha:
        raise ValueError(f"frozen endpoint checkpoint hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    result["checkpoint_payload_reads"] += 1
    validate_checkpoint_envelope(payload, report_model)
    model_type = base["engineering"].LocalRelationEvidenceCandidate
    model = model_type.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    restored_bundle = base["paired"].fingerprint(model.bundle(optimizer))
    if (
        len(model._observations) != TRAINING_SESSIONS
        or len(optimizer.state) != 41
        or any(int(state["step"]) != UPDATES for state in optimizer.state.values())
        or restored_bundle != base["paired"].fingerprint(payload["candidate"])
        or restored_bundle != payload["expected_signature"].get("bundle")
    ):
        raise ValueError("complete checkpoint bundle/history restoration differs")
    result["model_loads"] += 1
    return payload, model


def crossover_model(model_type, weights_model, history_model, torch):
    rng_before = torch.get_rng_state().clone()
    model = model_type(weights_model.config)
    torch.set_rng_state(rng_before)
    model.load_state_dict(weights_model.state_dict(), strict=True)
    model._observations = history_model._observations.copy()
    return model


def evaluate_combination(
    model,
    entries_by_view: dict[str, list[dict]],
    *,
    weight_source: str,
    history_source: str,
    runner,
    base,
    budget: CpuBudget,
    result: dict,
):
    before = base["paired"].fingerprint(model.bundle())
    rows = []
    try:
        for view in VIEWS:
            for entry in entries_by_view[view]:
                budget.guard(reserve=2)
                session = (
                    f"n1-crossover112-{weight_source}-{history_source}-{view}-{entry['case_id']}"
                )
                if session in model._observations:
                    raise ValueError("crossover measurement session collides with frozen history")
                result["model_queries"] += 1
                try:
                    row = runner.core.measure(model, entry["item"], ARM, session)
                    rows.append(
                        dict(
                            row,
                            seed=SEED,
                            arm=ARM,
                            view=view,
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
                    model._observations.pop(session, None)
    finally:
        if base["paired"].fingerprint(model.bundle()) != before:
            raise ValueError("crossover measurement changed model weights, observations or RNG")
    return report_entry(rows, weight_source=weight_source, history_source=history_source)


def check_anchor(
    rows: list[dict], sealed_rows: dict, *, weight_source: str, history_source: str
) -> dict:
    mismatches = []
    matches = 0
    for row in rows:
        key = (f"factorial_{row['view']}", row["case_id"])
        expected = sealed_rows[key]
        exact = all(
            row.get(field) == expected.get(field) for field in (*BASE_FIELDS, *ROW_IDENTITY_FIELDS)
        )
        if exact:
            matches += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "case_id": row["case_id"],
                    "view": row["view"],
                    "fields": [
                        field
                        for field in (*BASE_FIELDS, *ROW_IDENTITY_FIELDS)
                        if row.get(field) != expected.get(field)
                    ],
                }
            )
    return {
        "combination": f"{weight_source}+{history_source}",
        "rows_compared": len(rows),
        "exact_rows": matches,
        "exact_reproduction": len(rows) == 384 and matches == 384,
        "mismatches_first20": mismatches,
    }


def execute() -> dict:
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-112 is one-shot; report or attempt already exists")
    if "状态：已批准执行" not in PLAN.read_text(encoding="utf-8"):
        raise PermissionError("a frozen, explicitly approved PLAN-N1-112 snapshot is required")
    if not READINESS.is_file():
        raise FileNotFoundError("successful PLAN-N1-112 zero-query readiness report is required")
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    if (
        readiness.get("schema") != "n1-position-weight-history-crossover-r1-readiness-v1"
        or readiness.get("ready") is not True
        or any(
            readiness.get(key) != 0
            for key in (
                "checkpoint_payload_reads",
                "model_loads",
                "model_queries",
                "parameter_updates",
            )
        )
        or not isinstance(readiness.get("source_guards"), dict)
        or not guards_unchanged(readiness["source_guards"])
    ):
        raise ValueError("complete zero-query PLAN-N1-112 readiness evidence required")
    budget = CpuBudget(CPU_LIMIT_SECONDS)
    current_plan_sha = sha256_file(PLAN)
    guards = source_guards()
    attempt = {
        "schema": "n1-position-weight-history-crossover-r1-attempt-v1",
        "status": "running",
        "approved_plan_sha256": current_plan_sha,
        "plan_sha256_at_execution_start": current_plan_sha,
        "protocol": PROTOCOL,
        "source_guards": guards,
        "confirmation_face_consumed": False,
    }
    write_new(ATTEMPT, attempt)
    result = {
        "schema": PROTOCOL["schema"],
        "status": "running",
        "protocol": PROTOCOL,
        "approved_plan_sha256": current_plan_sha,
        "plan_sha256_at_execution_start": current_plan_sha,
        "source_guards": guards,
        "rows": [],
        "summaries": {},
        "anchors": [],
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_queries": 0,
        "native_reproductions": 0,
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
        for path, expected in expected_file_hashes().items():
            if sha256_file(path) != expected:
                raise ValueError(f"frozen input changed before evaluation: {path}")

        from scripts.training import run_taiji_n1_hard_neighbor_training as base

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
            raise ValueError("frozen original/renamed transition views must each contain192 cases")
        manifest = json.loads(FACTORIAL_MANIFEST.read_text(encoding="utf-8"))
        expected_manifest_entries = [
            entry
            for item in views["original"]
            if item["kind"] == "transition"
            for entry in factorial.variants(item)
        ]
        if (
            manifest.get("schema") != "n1-fresh-transition-factorial-manifest-v1"
            or manifest.get("protocol") != factorial.PROTOCOL
            or manifest.get("entries") != expected_manifest_entries
        ):
            raise ValueError(
                "approved train transition factorial manifest no longer matches source items"
            )

        baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        candidate_report = json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8"))
        baseline_model_record = selected_model(baseline_report, sufficient=True)
        candidate_model_record = selected_model(candidate_report, sufficient=False)
        baseline_sealed_rows, baseline_shard = read_sealed_factorial_rows(
            baseline_model_record, baseline_report, views=views, factorial=factorial
        )
        candidate_sealed_rows, candidate_shard = read_sealed_factorial_rows(
            candidate_model_record, candidate_report, views=views, factorial=factorial
        )
        guards = source_guards([baseline_shard, candidate_shard])
        result["source_guards"] = guards
        result["training_reports"] = {
            "baseline": {"path": str(BASELINE_REPORT), "sha256": BASELINE_REPORT_SHA256},
            "candidate": {"path": str(CANDIDATE_REPORT), "sha256": CANDIDATE_REPORT_SHA256},
        }
        result["sealed_training_shards"] = {
            "baseline": {"path": str(baseline_shard), **baseline_model_record["shard"]},
            "candidate": {"path": str(candidate_shard), **candidate_model_record["shard"]},
        }
        result["manifest"] = {
            "path": str(FACTORIAL_MANIFEST),
            "sha256": FACTORIAL_MANIFEST_SHA256,
            "entries": len(manifest["entries"]),
        }
        result["checkpoint_hashes_before"] = {
            "baseline": sha256_file(BASELINE_CHECKPOINT),
            "candidate": sha256_file(CANDIDATE_CHECKPOINT),
        }

        budget.guard(reserve=10)
        baseline_payload, baseline_model = load_model(
            BASELINE_CHECKPOINT,
            BASELINE_CHECKPOINT_SHA256,
            base=globals_dict,
            torch=torch,
            report_model=baseline_model_record,
            result=result,
        )
        candidate_payload, candidate_model = load_model(
            CANDIDATE_CHECKPOINT,
            CANDIDATE_CHECKPOINT_SHA256,
            base=globals_dict,
            torch=torch,
            report_model=candidate_model_record,
            result=result,
        )
        if list(baseline_model.state_dict()) != list(candidate_model.state_dict()) or any(
            baseline_model.state_dict()[key].shape != candidate_model.state_dict()[key].shape
            or baseline_model.state_dict()[key].dtype != candidate_model.state_dict()[key].dtype
            for key in baseline_model.state_dict()
        ):
            raise ValueError("frozen baseline and candidate parameter structures differ")
        model_type = type(baseline_model)
        if type(candidate_model) is not model_type:
            raise ValueError("frozen checkpoint model classes differ")

        model_sources = {
            "baseline": baseline_model,
            "candidate": candidate_model,
        }
        sealed_sources = {
            "baseline": baseline_sealed_rows,
            "candidate": candidate_sealed_rows,
        }
        native_ok = {}
        for weight_source, history_source in ANCHOR_COMBINATIONS:
            budget.guard(reserve=10)
            source_model = model_sources[weight_source.removesuffix("_weights")]
            rows = evaluate_combination(
                source_model,
                entries_by_view,
                weight_source=weight_source,
                history_source=history_source,
                runner=runner,
                base=globals_dict,
                budget=budget,
                result=result,
            )
            result["rows"].extend(rows)
            summary = summarize_combination(rows)
            combination = f"{weight_source}+{history_source}"
            result["summaries"][combination] = summary
            anchor = check_anchor(
                rows,
                sealed_sources[history_source.removesuffix("_history")],
                weight_source=weight_source,
                history_source=history_source,
            )
            anchor["joint_four_exact"] = {view: summary[view]["joint_four_exact"] for view in VIEWS}
            anchor["expected_joint_four_exact"] = EXPECTED_JOINT[combination]
            anchor["joint_four_reproduced"] = (
                anchor["joint_four_exact"] == anchor["expected_joint_four_exact"]
            )
            anchor["passed"] = anchor["exact_reproduction"] and anchor["joint_four_reproduced"]
            result["anchors"].append(anchor)
            result["native_reproductions"] += anchor["exact_rows"]
            native_ok[combination] = anchor["passed"]

        if not all(native_ok.values()):
            raise ValueError(
                "native anchors did not reproduce; both hybrid combinations were withheld"
            )

        for weight_source, history_source in HYBRID_COMBINATIONS:
            budget.guard(reserve=10)
            model = crossover_model(
                model_type,
                model_sources[weight_source.removesuffix("_weights")],
                model_sources[history_source.removesuffix("_history")],
                torch,
            )
            rows = evaluate_combination(
                model,
                entries_by_view,
                weight_source=weight_source,
                history_source=history_source,
                runner=runner,
                base=globals_dict,
                budget=budget,
                result=result,
            )
            result["rows"].extend(rows)
            combination = f"{weight_source}+{history_source}"
            result["summaries"][combination] = summarize_combination(rows)
            del model

        result["interpretation"] = interpretation(result["summaries"])
        result["interpretation_scope"] = (
            "exploratory_train_only_mechanism_attribution_no_generalization_or_capability_promotion"
        )
        result["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        result["status"] = "stopped"
        result["error"] = f"{type(error).__name__}: {error}"
    result["cpu_seconds"] = budget.used()
    result["plan_unchanged_during_execution"] = sha256_file(PLAN) == current_plan_sha
    result["checkpoint_hashes_after"] = {
        "baseline": sha256_file(BASELINE_CHECKPOINT) if BASELINE_CHECKPOINT.is_file() else None,
        "candidate": sha256_file(CANDIDATE_CHECKPOINT) if CANDIDATE_CHECKPOINT.is_file() else None,
    }
    result["checkpoints_unchanged"] = (
        result["checkpoint_hashes_after"] == result["checkpoint_hashes_before"]
    )
    result["sources_unchanged"] = guards_unchanged(result["source_guards"])
    result["code_unchanged"] = all(
        Path(path).is_file() and sha256_file(Path(path)) == digest
        for path, digest in result["source_guards"].items()
        if Path(path) == Path(__file__).resolve() or Path(path) in {p.resolve() for p in CODE_PATHS}
    )
    result["complete"] = (
        result["status"] == "complete"
        and len(result["rows"]) == 1_536
        and result["model_queries"] == 1_536
        and result["checkpoint_payload_reads"] == 2
        and result["model_loads"] == 2
        and result["native_reproductions"] == 768
        and len(result["anchors"]) == 2
        and all(anchor["passed"] for anchor in result["anchors"])
        and set(result["summaries"]) == {f"{w}+{h}" for w, h in ALL_COMBINATIONS}
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


def validate_only() -> dict:
    for path, expected in expected_file_hashes().items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen source check failed: {path}")
    if REPORT.exists() or ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-112 diagnostic output path is already occupied")
    plan_text = PLAN.read_text(encoding="utf-8")
    if "PLAN-N1-112" not in plan_text:
        raise ValueError("PLAN-N1-112 proposal is missing")
    predecessor = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    erratum = json.loads(PREVIOUS_ERRATUM.read_text(encoding="utf-8"))
    if (
        predecessor.get("status") != "stopped"
        or predecessor.get("model_queries") != 0
        or predecessor.get("parameter_updates") != 0
        or predecessor.get("checkpoint_hashes_before") != predecessor.get("checkpoint_hashes_after")
        or erratum.get("query_surface_consumed") is not False
        or erratum.get("execution", {}).get("model_queries") != 0
    ):
        raise ValueError("PLAN-N1-111 predecessor must prove a pre-query stop")
    from scripts.training import run_taiji_n1_hard_neighbor_training as base

    loaded, globals_dict, inputs = base.make_runner(load_inputs=True)
    factorial = globals_dict["calibration"].engineering.factorial
    views = inputs["views"]
    counts = {
        view: sum(
            len(factorial.variants(item)) for item in views[view] if item["kind"] == "transition"
        )
        for view in VIEWS
    }
    if counts != {"original": 192, "rename": 192}:
        raise ValueError("frozen train transition views are incomplete")
    manifest = json.loads(FACTORIAL_MANIFEST.read_text(encoding="utf-8"))
    expected_entries = [
        entry
        for item in views["original"]
        if item["kind"] == "transition"
        for entry in factorial.variants(item)
    ]
    if (
        manifest.get("schema") != "n1-fresh-transition-factorial-manifest-v1"
        or manifest.get("protocol") != factorial.PROTOCOL
        or manifest.get("entries") != expected_entries
    ):
        raise ValueError("frozen factorial manifest no longer matches train transition inputs")
    reports = {
        "baseline": json.loads(BASELINE_REPORT.read_text(encoding="utf-8")),
        "candidate": json.loads(CANDIDATE_REPORT.read_text(encoding="utf-8")),
    }
    records = {
        "baseline": selected_model(reports["baseline"], sufficient=True),
        "candidate": selected_model(reports["candidate"], sufficient=False),
    }
    sealed = {
        source: read_sealed_factorial_rows(
            records[source], reports[source], views=views, factorial=factorial
        )
        for source in ("baseline", "candidate")
    }
    guards = source_guards([sealed[source][1] for source in sealed])
    readiness = {
        "schema": "n1-position-weight-history-crossover-r1-readiness-v1",
        "ready": True,
        "report": str(REPORT.relative_to(ROOT)),
        "readiness_report": str(READINESS.relative_to(ROOT)),
        "plan_path": str(PLAN.relative_to(ROOT)),
        "plan_sha256_at_preflight": sha256_file(PLAN),
        "predecessor_report_sha256": PREVIOUS_REPORT_SHA256,
        "protocol": PROTOCOL,
        "model_loads": 0,
        "checkpoint_payload_reads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "checkpoint_saves": 0,
        "view_entries": counts,
        "sealed_anchor_joint_four": {
            "baseline": EXPECTED_JOINT["baseline_weights+baseline_history"],
            "candidate": EXPECTED_JOINT["candidate_weights+candidate_history"],
        },
        "sealed_shards": {
            source: {"path": str(sealed[source][1]), "rows": len(sealed[source][0])}
            for source in sealed
        },
        "source_guard_count": len(guards),
        "source_guards": guards,
        "torch_version": loaded["torch"].__version__,
    }
    if READINESS.exists():
        existing = json.loads(READINESS.read_text(encoding="utf-8"))
        if existing != readiness:
            raise FileExistsError("existing readiness report differs; never overwrite")
    else:
        write_new(READINESS, readiness)
    return readiness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_only(), ensure_ascii=False, sort_keys=True))
        raise SystemExit(0)
    result = execute()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "status",
                    "cpu_seconds",
                    "model_queries",
                    "native_reproductions",
                    "interpretation",
                    "error",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["complete"] else 1)
