"""Authorized-only range and confirmation evaluator for PLAN-N1-121."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import build_taiji_n1_v11_after_n120 as face_builder  # noqa: E402
from scripts.training import eval_taiji_n1_transition_factorial_confirmation as scorer  # noqa: E402
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)  # noqa: E402
from scripts.training import run_taiji_n1_hard_neighbor_training as baseline_training  # noqa: E402
from scripts.training import seal_ast  # noqa: E402

BASELINE_REPORT = ROOT / "reports/taiji_n1_hard_neighbor_training_20261017.json"
BASELINE_REPORT_SHA256 = "dad10ec72c605367f99da035bab07d522ce1793154915ebcc3c7f1022f411909"
CANDIDATE_REPORT = ROOT / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json"
CANDIDATE_REPORT_SHA256 = "873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f"
CANDIDATE_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json"
)
CANDIDATE_REVALIDATION_SHA256 = "22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d"
N110_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-110_length_balanced_training_and_v11_confirmation_draft_20261017.md"
)
N110_PLAN_SHA256 = "16d0cecb325af6463c107973fc0e741478140134a7642cd64c86e045c6069783"
N121_PLAN = (
    ROOT / "plans/reference/PLAN-N1-121_v11_confirmation_after_n120_training_draft_20261019.md"
)
N121_PLAN_SHA256 = "5c2ac778e8e205e1574a2bd93d77dfc1c597490aced4db7a0b0f6d2efbbfefb8"
PREQUERY_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_prequery_readiness_20261019.json"
PREQUERY_READINESS_SHA256 = "5a9883ec5dd924f13279c59c23ecdd7eaf89f54062ba237d60a85e03e33da82b"
RESTORE_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_restore_readiness_20261019.json"
RESTORE_READINESS_SHA256 = "6212e0413153e2d22d7cde86cc8eb97414d0467ee52ac56cee1d04027ca5f2a3"
FACE_READINESS = face_builder.SOURCE_READINESS
FACE = face_builder.FACE
FACE_CHECKS = face_builder.FACE_CHECKS
FACE_ATTEMPT = face_builder.ATTEMPT
RANGE_REPORT = face_builder.RANGE_REPORT
RANGE_ATTEMPT = face_builder.RANGE_ATTEMPT
CONFIRM_REPORT = face_builder.CONFIRM_REPORT
CONFIRM_ATTEMPT = face_builder.CONFIRM_ATTEMPT
SCORER_READINESS_V1 = ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_20261019.json"
SCORER_READINESS_V1_SHA256 = "91334829059890c528a3d83a224fbd8ec57827b188d5d75e7398e7f1705ff782"
SCORER_READINESS_V2 = (
    ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_v2_20261019.json"
)
SCORER_READINESS_V2_SHA256 = "02bb7985813d87d44e7fe99489ef9b01640bfd505e0fab0cc73cc26a610fb5a7"
SCORER_READINESS_V3 = (
    ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_v3_20261019.json"
)
SCORER_READINESS_V3_SHA256 = "83c232557b9c86da9d4c157a25a61116aa1870fa53d765de3f021040dba6cff9"
SCORER_READINESS_V4 = (
    ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_v4_20261019.json"
)
SCORER_READINESS_V4_SHA256 = "1c548d397e981b4fdea956f184d1d0f7ede4e47dfe1c00df0531630bfdb00252"
SCORER_READINESS_V5 = (
    ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_v5_20261019.json"
)
SCORER_READINESS_V5_SHA256 = "d023f5e6409aef512bbd77cea8f24fdf073a9aa5d32d3128b7c7fcec6f55985b"
SCORER_READINESS = ROOT / "reports/taiji_n1_v11_after_n120_confirmation_readiness_v6_20261019.json"
RANGE_CPU_LIMIT_SECONDS = 600
CONFIRM_CPU_LIMIT_SECONDS = 600
MAX_CUMULATIVE_CPU_SECONDS = 600
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
MIN_USABLE_ENDPOINTS = 6
TRIPLE_ERROR_FLOOR = 6
CONFIRM_ITEMS_PER_ENDPOINT = 20
MIN_POSITION_SEED_EXACT = 10
MIN_POSITION_KIND_EXACT = 2
MIN_ADVANTAGE_EACH_CONTROL = 12


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


class CpuDeadline:
    def __init__(self, limit: float):
        self.start = time.process_time()
        self.limit = limit

    def used(self) -> float:
        return time.process_time() - self.start

    def guard(self, reserve: float = 0.0) -> None:
        if self.used() + reserve >= self.limit:
            raise TimeoutError("PLAN-N1-121 cumulative CPU ceiling reached")


def _read_bound_inputs(
    *, require_face_absent: bool = False
) -> tuple[dict, dict, dict, dict, dict, dict]:
    baseline = read_json(BASELINE_REPORT)
    candidate = read_json(CANDIDATE_REPORT)
    revalidation = read_json(CANDIDATE_REVALIDATION)
    prequery = read_json(PREQUERY_READINESS)
    restore = read_json(RESTORE_READINESS)
    face_readiness = read_json(FACE_READINESS)
    for path, expected in (
        (BASELINE_REPORT, BASELINE_REPORT_SHA256),
        (CANDIDATE_REPORT, CANDIDATE_REPORT_SHA256),
        (CANDIDATE_REVALIDATION, CANDIDATE_REVALIDATION_SHA256),
        (PREQUERY_READINESS, PREQUERY_READINESS_SHA256),
        (RESTORE_READINESS, RESTORE_READINESS_SHA256),
        (N110_PLAN, N110_PLAN_SHA256),
        (N121_PLAN, N121_PLAN_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"PLAN-N1-121 source report changed: {path}")
    if (
        baseline.get("complete") is not True
        or baseline.get("training_sufficient") is not True
        or candidate.get("status") != "stopped"
        or revalidation.get("complete") is not True
        or revalidation.get("training_sufficient") is not True
        or revalidation.get("independent_restore_checks") != 18
        or prequery.get("complete") is not True
        or restore.get("complete") is not True
        or restore.get("independent_full_bundle_restores") != 18
        or restore.get("all_full_bundles_equal") is not True
        or any(
            report.get(key) != 0
            for report in (baseline, candidate, revalidation, prequery, restore)
            for key in ("dev_queries", "holdout_queries", "confirmation_queries", "model_queries")
            if key in report
        )
    ):
        raise ValueError("complete training-gate and restore readiness required")
    if (
        face_readiness.get("complete") is not True
        or face_readiness.get("schema") != "n1-plan121-v11-face-source-readiness-v3"
        or face_readiness.get("status") != "ready_for_owner_approval"
        or face_readiness.get("face_generated") is not False
        or face_readiness.get("face_consumed") is not False
        or face_readiness.get("plan_sha256") != N121_PLAN_SHA256
        or face_readiness.get("excluded_entity_count") != 1918
        or face_readiness.get("excluded_material_count") != 1633
        or face_readiness.get("exclusion_provenance", {}).get("n110_schedule_sha256")
        != "b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c"
        or face_readiness.get("exclusion_provenance", {}).get("n110_material_value_map_sha256")
        != "dba1d9eb8fec6d60e6b1ceca496bce3b3386b05d1539ad88696177db0b9e9ba8"
        or face_readiness.get("exclusion_provenance", {}).get("n110_material_value_map_entries")
        != 288
        or face_readiness.get("exclusion_provenance", {}).get(
            "n110_mapped_training_values_excluded"
        )
        != 288
        or face_readiness.get("exclusion_provenance", {}).get("n121_source_correction_sha256")
        != "896dad354e6c81b7a1157746fe4e4bc5dcc8fcd266a7707dc09f476d7bb7aba7"
        or face_readiness.get("exclusion_provenance", {}).get("v7_confirm_payload_fields_decoded")
        is not False
        or face_readiness.get("exclusion_provenance", {}).get("v8_confirm_payload_fields_decoded")
        is not False
        or face_readiness.get("exclusion_provenance", {})
        .get("v7_sacrifice", {})
        .get("confirm_model_queries")
        != 0
        or face_readiness.get("exclusion_provenance", {})
        .get("v8_sacrifice", {})
        .get("confirm_model_queries")
        != 0
    ):
        raise ValueError("v11 seed111 face must remain absent until owner approval")
    for raw_path, expected in face_readiness.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v11 face source changed after source readiness: {path}")
    if require_face_absent and any(
        path.exists()
        for path in (
            FACE,
            FACE_CHECKS,
            FACE_ATTEMPT,
            RANGE_REPORT,
            RANGE_ATTEMPT,
            CONFIRM_REPORT,
            CONFIRM_ATTEMPT,
        )
    ):
        raise FileExistsError("v11 face or one-shot evaluator artifacts already exist")
    return baseline, candidate, revalidation, prequery, restore, face_readiness


def validate_only() -> dict:
    if SCORER_READINESS.exists():
        raise FileExistsError("N1-121 scorer readiness is write-once")
    if (
        not SCORER_READINESS_V1.is_file()
        or sha256(SCORER_READINESS_V1) != SCORER_READINESS_V1_SHA256
    ):
        raise ValueError("prior scorer readiness receipt changed")
    if (
        not SCORER_READINESS_V2.is_file()
        or sha256(SCORER_READINESS_V2) != SCORER_READINESS_V2_SHA256
    ):
        raise ValueError("intermediate scorer readiness receipt changed")
    if (
        not SCORER_READINESS_V3.is_file()
        or sha256(SCORER_READINESS_V3) != SCORER_READINESS_V3_SHA256
    ):
        raise ValueError("third scorer readiness receipt changed")
    if (
        not SCORER_READINESS_V4.is_file()
        or sha256(SCORER_READINESS_V4) != SCORER_READINESS_V4_SHA256
    ):
        raise ValueError("fourth scorer readiness receipt changed")
    if (
        not SCORER_READINESS_V5.is_file()
        or sha256(SCORER_READINESS_V5) != SCORER_READINESS_V5_SHA256
    ):
        raise ValueError("fifth scorer readiness receipt changed")
    started = time.process_time()
    baseline, candidate, revalidation, prequery, restore, face_readiness = _read_bound_inputs(
        require_face_absent=True
    )
    _, baseline_globals, _ = baseline_training.make_runner(load_inputs=False)
    n120.configure()
    _, candidate_globals, _ = n120._BASE.make_runner(load_inputs=False)
    if (
        not callable(baseline_globals.get("load"))
        or not callable(candidate_globals.get("load"))
        or not callable(scorer.evaluate)
    ):
        raise TypeError("baseline and candidate final checkpoint loaders are required")
    source_paths = (
        BASELINE_REPORT,
        CANDIDATE_REPORT,
        CANDIDATE_REVALIDATION,
        PREQUERY_READINESS,
        RESTORE_READINESS,
        N110_PLAN,
        N121_PLAN,
        FACE_READINESS,
        SCORER_READINESS_V5,
        SCORER_READINESS_V4,
        SCORER_READINESS_V3,
        Path(face_builder.__file__).resolve(),
        Path(face_builder.v9.__file__).resolve(),
        Path(face_builder.train_data.__file__).resolve(),
        Path(scorer.__file__).resolve(),
        ROOT / "tests/taiji_native/test_n1_plan121_v11_face_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan121_scoring_contract.py",
        Path(__file__).resolve(),
        Path(baseline_training.__file__).resolve(),
        Path(n120.__file__).resolve(),
        Path(n120.expanded.__file__).resolve(),
        Path(n120.plan120_receipt.__file__).resolve(),
    )
    source_hashes = {str(Path(path).resolve()): sha256(Path(path)) for path in source_paths}
    total = time.process_time() - started
    checkpoint_hashes = {
        **prequery["baseline_checkpoint_hashes"],
        **prequery["candidate_checkpoint_hashes"],
    }
    if len(checkpoint_hashes) != 36 or any(
        not Path(path).is_file() or sha256(Path(path)) != digest
        for path, digest in checkpoint_hashes.items()
    ):
        raise ValueError("36 unchanged baseline/candidate zero/final checkpoint hashes required")
    report = {
        "schema": "n1-plan121-scorer-zero-query-readiness-v6",
        "status": "ready_for_owner_approval",
        "complete": total <= MAX_CUMULATIVE_CPU_SECONDS,
        "baseline_report_sha256": BASELINE_REPORT_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_revalidation_sha256": CANDIDATE_REVALIDATION_SHA256,
        "source_readiness_sha256": sha256(FACE_READINESS),
        "checkpoint_restore_readiness_sha256": RESTORE_READINESS_SHA256,
        "supersedes_readiness_sha256": SCORER_READINESS_V1_SHA256,
        "supersedes_readiness_v2_sha256": SCORER_READINESS_V2_SHA256,
        "supersedes_readiness_v3_sha256": SCORER_READINESS_V3_SHA256,
        "supersedes_readiness_v4_sha256": SCORER_READINESS_V4_SHA256,
        "supersedes_readiness_v5_sha256": SCORER_READINESS_V5_SHA256,
        "plan_sha256": N121_PLAN_SHA256,
        "excluded_entity_count": face_readiness["excluded_entity_count"],
        "excluded_material_count": face_readiness["excluded_material_count"],
        "n110_material_value_map_sha256": face_readiness["exclusion_provenance"][
            "n110_material_value_map_sha256"
        ],
        "n110_schedule_sha256": face_readiness["exclusion_provenance"]["n110_schedule_sha256"],
        "n110_material_value_map_entries": face_readiness["exclusion_provenance"][
            "n110_material_value_map_entries"
        ],
        "n110_mapped_training_values_excluded": face_readiness["exclusion_provenance"][
            "n110_mapped_training_values_excluded"
        ],
        "n121_source_correction_sha256": face_readiness["exclusion_provenance"][
            "n121_source_correction_sha256"
        ],
        "v7_confirm_payload_fields_decoded": face_readiness["exclusion_provenance"][
            "v7_confirm_payload_fields_decoded"
        ],
        "v8_confirm_payload_fields_decoded": face_readiness["exclusion_provenance"][
            "v8_confirm_payload_fields_decoded"
        ],
        "source_hashes": source_hashes,
        "baseline_endpoints": len(baseline["models"]),
        "candidate_endpoints": len(candidate["models"]),
        "candidate_position_gates_passed": sum(
            model.get("training_sufficient") is True
            for model in revalidation["models"]
            if model.get("arm") == "position"
        ),
        "restores_equal": restore["all_full_bundles_equal"],
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint_hash_count": len(checkpoint_hashes),
        "face_generated": FACE.exists(),
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
        "can_promote": False,
        "cpu_seconds": round(total, 6),
        "cpu_limit_seconds": MAX_CUMULATIVE_CPU_SECONDS,
    }
    write_new(SCORER_READINESS, report)
    return report


def _verify_scorer_readiness() -> dict:
    if not SCORER_READINESS.is_file():
        raise FileNotFoundError("completed zero-query scorer readiness is required")
    report = read_json(SCORER_READINESS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_owner_approval"
        or report.get("schema") != "n1-plan121-scorer-zero-query-readiness-v6"
        or report.get("plan_sha256") != N121_PLAN_SHA256
        or report.get("source_readiness_sha256") != sha256(FACE_READINESS)
        or report.get("excluded_entity_count") != 1918
        or report.get("excluded_material_count") != 1633
        or report.get("n110_material_value_map_sha256")
        != "dba1d9eb8fec6d60e6b1ceca496bce3b3386b05d1539ad88696177db0b9e9ba8"
        or report.get("n110_schedule_sha256")
        != "b2d637fcc1e096d40bba35bbc91bc66b0c27257cd74a4bcac5f5f488f55b6d4c"
        or report.get("n110_material_value_map_entries") != 288
        or report.get("n110_mapped_training_values_excluded") != 288
        or report.get("n121_source_correction_sha256")
        != "896dad354e6c81b7a1157746fe4e4bc5dcc8fcd266a7707dc09f476d7bb7aba7"
        or report.get("v7_confirm_payload_fields_decoded") is not False
        or report.get("v8_confirm_payload_fields_decoded") is not False
        or report.get("checkpoint_hash_count") != 36
        or report.get("model_loads") != 0
        or report.get("checkpoint_loads") != 0
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("optimizer_steps") != 0
        or report.get("dev_queries") != 0
        or report.get("holdout_queries") != 0
        or report.get("confirmation_queries") != 0
        or report.get("face_generated") is not False
        or len(report.get("checkpoint_hashes", {})) != 36
    ):
        raise ValueError("zero-query scorer readiness is stale or incomplete")
    for raw_path, expected in report.get("source_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"scorer source changed after readiness: {path}")
    for raw_path, expected in report.get("checkpoint_hashes", {}).items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checkpoint changed after scorer readiness: {path}")
    return report


def _load_face() -> tuple[dict, dict]:
    if not FACE.is_file() or not FACE_CHECKS.is_file():
        raise FileNotFoundError("owner-approved v11 seed111 face and checks are required")
    face = read_json(FACE)
    checks = read_json(FACE_CHECKS)
    if (
        checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), FACE)
        or face.get("schema") != "n1-hard-neighbor-v11-confirmation-v1"
        or face.get("seed") != 111
        or face.get("items") != 40
        or face.get("observations_per_item") != 12
        or face.get("name_neighbor_distance") != 1
        or len(face.get("episodes", [])) != 40
        or len(face.get("subsets", {}).get("sacrifice", [])) != 20
        or len(face.get("subsets", {}).get("confirm", [])) != 20
        or set(face.get("subsets", {})) != {"sacrifice", "confirm"}
        or set(face.get("subsets", {}).get("sacrifice", []))
        & set(face.get("subsets", {}).get("confirm", []))
        or {item.get("split") for item in face.get("episodes", [])} != {"holdout"}
        or len({item.get("id") for item in face.get("episodes", [])}) != 40
        or Counter(item.get("kind") for item in face.get("episodes", []))
        != Counter({kind: 10 for kind in KINDS})
        or set(face.get("subsets", {}).get("sacrifice", []))
        & set(face.get("subsets", {}).get("confirm", []))
        or set(face.get("subsets", {}).get("sacrifice", []))
        | set(face.get("subsets", {}).get("confirm", []))
        != {item.get("id") for item in face.get("episodes", [])}
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or checks.get("plan_sha256") != N121_PLAN_SHA256
        or checks.get("source_readiness_sha256") != sha256(FACE_READINESS)
        or checks.get("checks", {}).get("sacrifice_confirm_length_split") is not True
        or face.get("exclusion_sources", {}).get("plan_sha256") != N121_PLAN_SHA256
        or face.get("exclusion_sources", {}).get("source_readiness_sha256")
        != sha256(FACE_READINESS)
    ):
        raise ValueError("v11 face/check report is stale or differs from the frozen contract")
    face_ids = {item["id"] for item in face["episodes"]}
    if face_ids != set(face["subsets"]["sacrifice"]) | set(face["subsets"]["confirm"]):
        raise ValueError("v11 sacrifice/confirm subsets must partition the face")
    for subset in ("sacrifice", "confirm"):
        selected = [item for item in face["episodes"] if item["id"] in face["subsets"][subset]]
        if any(sum(item["kind"] == kind for item in selected) != 5 for kind in KINDS):
            raise ValueError(f"v11 subset class balance differs: {subset}")
    source_hashes = face.get("exclusion_sources", {}).get("source_hashes", {})
    if not source_hashes:
        raise ValueError("v11 face source-hash chain is required")
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"v11 face source changed after construction: {path}")
    subsets = {name: set(values) for name, values in face["subsets"].items()}
    for subset, ids in subsets.items():
        selected = [item for item in face["episodes"] if item["id"] in ids]
        if any(sum(item["kind"] == kind for item in selected) != 5 for kind in KINDS):
            raise ValueError(f"v11 face class balance differs in {subset}")
    if checks.get("checks", {}).get("sacrifice_confirm_length_split") is not True:
        raise ValueError("v11 length/partition contract did not pass")
    return face, checks


def _final_paths(report: dict, *, cohort: str) -> dict[tuple[int, str], tuple[Path, str]]:
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    paths = {}
    for model in report.get("models", []):
        key = model.get("seed"), model.get("arm")
        if key not in expected or key in paths:
            raise ValueError(f"unexpected/duplicate {cohort} endpoint")
        checks = [
            item
            for item in model.get("restore_checks", [])
            if Path(item.get("path", "")).name == f"s{key[0]}_{key[1]}_final.pt"
        ]
        if len(checks) != 1 or checks[0].get("equal") is not True:
            raise ValueError(f"full restored final checkpoint required for {cohort}/{key}")
        path = Path(checks[0]["path"])
        digest = checks[0].get("sha256")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"final checkpoint changed: {path}")
        paths[key] = path, digest
    if set(paths) != expected:
        raise ValueError(f"all nine final checkpoints required for {cohort}")
    return paths


def _load_model(g: dict, path: Path, expected_sha: str, seed: int, arm: str, protocol: dict):
    if sha256(path) != expected_sha:
        raise ValueError(f"checkpoint bytes changed before scoring: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload.get("protocol") != protocol
        or payload.get("seed") != seed
        or payload.get("arm") != arm
        or payload.get("phase") != "final"
        or payload.get("completed_updates") != 24576
    ):
        raise ValueError(f"checkpoint protocol mismatch: {seed}/{arm}")
    model, optimizer = g["load"](payload)
    paired = g["paired"]
    fingerprint = paired.fingerprint(model.bundle(optimizer))
    expected_bundle = payload.get("expected_signature", {}).get("bundle")
    if not expected_bundle or fingerprint != expected_bundle:
        raise ValueError(f"checkpoint full-bundle restore mismatch: {seed}/{arm}")
    return model, optimizer, fingerprint, paired


def _score_endpoint(
    g: dict,
    path: Path,
    digest: str,
    protocol: dict,
    seed: int,
    arm: str,
    items: list[dict],
    budget: CpuDeadline,
) -> list[dict]:
    model, optimizer, before, paired = _load_model(g, path, digest, seed, arm, protocol)
    try:
        rows = scorer.evaluate(model, optimizer, seed, arm, items, budget)
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("frozen evaluation changed model, optimizer, RNG or history")
        if len(rows) != len(items) or {row["id"] for row in rows} != {item["id"] for item in items}:
            raise ValueError("one exact score row per selected face item required")
        return rows
    finally:
        del model, optimizer


def _range_summary(endpoint_rows: list[dict]) -> dict:
    if len(endpoint_rows) != 9 or any(len(item["rows"]) != 20 for item in endpoint_rows):
        raise ValueError("range needs exactly nine endpoints and twenty sacrifice items")
    expected_ids = None
    for endpoint in endpoint_rows:
        rows = endpoint["rows"]
        ids = {row["id"] for row in rows}
        if (
            len(ids) != 20
            or any(row.get("split") != "holdout" for row in rows)
            or any(sum(row.get("kind") == kind for row in rows) != 5 for kind in KINDS)
            or (expected_ids is not None and ids != expected_ids)
        ):
            raise ValueError(
                "same twenty holdout sacrifice items and class balance required per endpoint"
            )
        expected_ids = ids
    errors = {
        (item["seed"], item["arm"]): sum(
            row["kind"] != "unknown" and not row["free_exact"] for row in item["rows"]
        )
        for item in endpoint_rows
    }
    usable = sum(value >= 6 for value in errors.values())
    return {
        "usable_endpoints": usable,
        "endpoint_count": 9,
        "minimum_usable_endpoints": 6,
        "decodable_items_per_endpoint": 15,
        "triple_error_floor": 6,
        "endpoint_errors": {f"{seed}_{arm}": value for (seed, arm), value in errors.items()},
        "range_check_passed": usable >= 6,
    }


def _cohort_gates(rows: list[dict], cohort: str) -> dict:
    if len(rows) != 180 or len({(row["seed"], row["arm"], row["id"]) for row in rows}) != 180:
        raise ValueError(f"180 unique confirm rows required for {cohort}")
    summary = {}
    expected_ids = None
    for seed in SEEDS:
        summary[seed] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["seed"] == seed and row["arm"] == arm]
            ids = {row["id"] for row in selected}
            if (
                len(selected) != 20
                or len(ids) != 20
                or any(row.get("split") != "holdout" for row in selected)
                or any(sum(row["kind"] == kind for row in selected) != 5 for kind in KINDS)
                or (expected_ids is not None and ids != expected_ids)
            ):
                raise ValueError(
                    f"twenty items and five of each kind required for {cohort}/{seed}/{arm}"
                )
            expected_ids = ids
            summary[seed][arm] = {
                "exact": sum(row["free_exact"] for row in selected),
                "exact_by_kind": {
                    kind: sum(row["kind"] == kind and row["free_exact"] for row in selected)
                    for kind in KINDS
                },
            }
    positions = [summary[seed]["position"] for seed in SEEDS]
    totals = {arm: sum(summary[seed][arm]["exact"] for seed in SEEDS) for arm in ARMS}
    gates = {
        "each_position_seed": all(value["exact"] >= 10 for value in positions),
        "each_position_seed_kind": all(
            all(value["exact_by_kind"][kind] >= 2 for kind in KINDS) for value in positions
        ),
        "advantage_over_terminal": totals["position"] >= totals["terminal"] + 12,
        "advantage_over_cut": totals["position"] >= totals["cut"] + 12,
    }
    return {
        "cohort": cohort,
        "summary": summary,
        "aggregate_exact": totals,
        "gates": gates,
        "all_four_gates": all(gates.values()),
    }


def _prior_cpu(
    face_checks: dict,
    face_readiness: dict,
    prequery: dict,
    restore: dict,
    scorer_readiness: dict,
) -> float:
    return sum(
        float(value)
        for value in (
            face_readiness.get("cpu_seconds", 0),
            prequery.get("cpu_seconds", 0),
            restore.get("cpu_seconds", 0),
            face_checks.get("build_cpu_seconds", 0),
            scorer_readiness.get("cpu_seconds", 0),
        )
    )


def run_range_check(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-121 range queries require owner approval")
    if RANGE_REPORT.exists() or RANGE_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-121 range check is one-shot")
    scorer_readiness = _verify_scorer_readiness()
    face, checks = _load_face()
    baseline, candidate, revalidation, prequery, restore, face_readiness = _read_bound_inputs()
    candidate_paths = _final_paths(candidate, cohort="candidate")
    sacrifice_ids = set(face["subsets"]["sacrifice"])
    items = [item for item in face["episodes"] if item["id"] in sacrifice_ids]
    if len(items) != 20:
        raise ValueError("exact twenty-item sacrifice subset required")
    prior_cpu = _prior_cpu(checks, face_readiness, prequery, restore, scorer_readiness)
    if prior_cpu >= MAX_CUMULATIVE_CPU_SECONDS:
        raise TimeoutError("PLAN-N1-121 cumulative CPU budget already exhausted")
    budget = CpuDeadline(min(RANGE_CPU_LIMIT_SECONDS, MAX_CUMULATIVE_CPU_SECONDS - prior_cpu))
    n120.configure()
    _, g, _ = n120._BASE.make_runner(load_inputs=False)
    write_new(
        RANGE_ATTEMPT,
        {
            "schema": "n1-plan121-v11-range-attempt-v1",
            "status": "started",
            "face_sha256": sha256(FACE),
            "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
            "scorer_readiness_sha256": sha256(SCORER_READINESS),
            "candidate_checkpoints": {f"{s}_{a}": x[1] for (s, a), x in candidate_paths.items()},
            "queried_subset": "sacrifice",
            "confirm_queried": False,
            "parameter_updates": 0,
            "can_promote": False,
        },
    )
    endpoint_rows = []
    error = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=3)
                path, digest = candidate_paths[seed, arm]
                rows = _score_endpoint(
                    g, path, digest, candidate["protocol"], seed, arm, items, budget
                )
                endpoint_rows.append({"seed": seed, "arm": arm, "rows": rows})
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    cumulative_cpu = prior_cpu + budget.used()
    summary = _range_summary(endpoint_rows) if error is None else None
    range_source_hashes = {
        **scorer_readiness["source_hashes"],
        **scorer_readiness["checkpoint_hashes"],
        **face["exclusion_sources"]["source_hashes"],
        str(SCORER_READINESS.resolve()): sha256(SCORER_READINESS),
        str(FACE.resolve()): sha256(FACE),
        str(FACE_CHECKS.resolve()): sha256(FACE_CHECKS),
        **{str(path.resolve()): digest for path, digest in candidate_paths.values()},
    }
    sources_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in range_source_hashes.items()
    )
    complete = (
        error is None
        and len(endpoint_rows) == 9
        and cumulative_cpu <= MAX_CUMULATIVE_CPU_SECONDS
        and sources_unchanged
    )
    report = {
        "schema": "n1-plan121-v11-sacrifice-range-v1",
        "status": (
            "complete"
            if complete and summary["range_check_passed"]
            else "range_not_met" if complete else "stopped"
        ),
        "complete": complete,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "scorer_readiness_sha256": sha256(SCORER_READINESS),
        "source_hashes_unchanged": sources_unchanged,
        "queried_subset": "sacrifice",
        "range_gate": summary,
        "endpoint_results": endpoint_rows,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_receipt_revalidation_sha256": CANDIDATE_REVALIDATION_SHA256,
        "candidate_checkpoint_hashes": {f"{s}_{a}": x[1] for (s, a), x in candidate_paths.items()},
        "model_queries": sum(len(value["rows"]) for value in endpoint_rows),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cumulative_cpu_seconds": cumulative_cpu,
        "cumulative_cpu_limit_seconds": MAX_CUMULATIVE_CPU_SECONDS,
        "cpu_seconds": round(budget.used(), 6),
        "error": error,
        "can_promote": False,
    }
    write_new(RANGE_REPORT, report)
    return report


def run_confirmation(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-121 confirmation queries require owner approval")
    if CONFIRM_REPORT.exists() or CONFIRM_ATTEMPT.exists():
        raise FileExistsError("PLAN-N1-121 confirmation is one-shot")
    if not RANGE_REPORT.is_file():
        raise FileNotFoundError("a completed sacrifice-only range report is required")
    range_report = read_json(RANGE_REPORT)
    scorer_readiness = _verify_scorer_readiness()
    if (
        range_report.get("complete") is not True
        or range_report.get("status") != "complete"
        or range_report.get("range_gate", {}).get("range_check_passed") is not True
        or range_report.get("model_queries") != 180
        or range_report.get("queried_subset") != "sacrifice"
        or range_report.get("confirmation_queries") != 0
        or not seal_ast.receipt_binding_matches(range_report.get("face_sha256"), FACE)
        or range_report.get("scorer_readiness_sha256") != sha256(SCORER_READINESS)
        or range_report.get("source_hashes_unchanged") is not True
    ):
        raise ValueError("range gate must pass on an intact face before confirmation")
    face, checks = _load_face()
    baseline, candidate, _, prequery, restore, face_readiness = _read_bound_inputs()
    baseline_paths = _final_paths(baseline, cohort="baseline")
    candidate_paths = _final_paths(candidate, cohort="candidate")
    confirm_ids = set(face["subsets"]["confirm"])
    items = [item for item in face["episodes"] if item["id"] in confirm_ids]
    if len(items) != 20 or confirm_ids & set(face["subsets"]["sacrifice"]):
        raise ValueError("exact disjoint twenty-item confirm subset required")
    prior_cpu = (
        _prior_cpu(checks, face_readiness, prequery, restore, scorer_readiness)
        + range_report["cpu_seconds"]
    )
    if prior_cpu >= MAX_CUMULATIVE_CPU_SECONDS:
        raise TimeoutError("PLAN-N1-121 cumulative CPU budget already exhausted")
    budget = CpuDeadline(min(CONFIRM_CPU_LIMIT_SECONDS, MAX_CUMULATIVE_CPU_SECONDS - prior_cpu))
    write_new(
        CONFIRM_ATTEMPT,
        {
            "schema": "n1-plan121-v11-confirm-attempt-v1",
            "status": "started",
            "face_sha256": sha256(FACE),
            "range_report_sha256": sha256(RANGE_REPORT),
            "scorer_readiness_sha256": sha256(SCORER_READINESS),
            "baseline_report_sha256": BASELINE_REPORT_SHA256,
            "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
            "queried_subset": "confirm",
            "parameter_updates": 0,
            "optimizer_steps": 0,
            "can_promote": False,
        },
    )

    def score_cohort(label: str, paths: dict, protocol: dict, make_runner) -> list[dict]:
        _, g, _ = make_runner()
        rows = []
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=4)
                path, digest = paths[seed, arm]
                rows.extend(_score_endpoint(g, path, digest, protocol, seed, arm, items, budget))
        return rows

    error = None
    baseline_rows = []
    candidate_rows = []
    try:
        # Restore/evaluate the frozen baseline before the candidate adapter changes the shared policy globals.
        baseline_rows = score_cohort(
            "baseline",
            baseline_paths,
            baseline["protocol"],
            lambda: baseline_training.make_runner(load_inputs=False),
        )
        n120.configure()
        candidate_rows = score_cohort(
            "candidate",
            candidate_paths,
            candidate["protocol"],
            lambda: n120._BASE.make_runner(load_inputs=False),
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    cumulative_cpu = prior_cpu + budget.used()
    baseline_gates = _cohort_gates(baseline_rows, "baseline") if error is None else None
    candidate_gates = _cohort_gates(candidate_rows, "candidate") if error is None else None
    confirm_source_hashes = {
        **scorer_readiness["source_hashes"],
        **scorer_readiness["checkpoint_hashes"],
        **face["exclusion_sources"]["source_hashes"],
        str(SCORER_READINESS.resolve()): sha256(SCORER_READINESS),
        str(FACE.resolve()): sha256(FACE),
        str(FACE_CHECKS.resolve()): sha256(FACE_CHECKS),
        str(RANGE_REPORT.resolve()): sha256(RANGE_REPORT),
        **{str(path.resolve()): digest for path, digest in baseline_paths.values()},
        **{str(path.resolve()): digest for path, digest in candidate_paths.values()},
    }
    sources_unchanged = all(
        Path(path).is_file() and sha256(Path(path)) == digest
        for path, digest in confirm_source_hashes.items()
    )
    complete = (
        error is None
        and len(baseline_rows) == 180
        and len(candidate_rows) == 180
        and cumulative_cpu <= MAX_CUMULATIVE_CPU_SECONDS
        and sources_unchanged
    )
    report = {
        "schema": "n1-plan121-v11-confirmation-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "range_report_sha256": sha256(RANGE_REPORT),
        "scorer_readiness_sha256": sha256(SCORER_READINESS),
        "baseline_report_sha256": BASELINE_REPORT_SHA256,
        "candidate_report_sha256": CANDIDATE_REPORT_SHA256,
        "candidate_receipt_revalidation_sha256": CANDIDATE_REVALIDATION_SHA256,
        "baseline_gates": baseline_gates,
        "candidate_gates": candidate_gates,
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "confirmation_queries": len(baseline_rows) + len(candidate_rows),
        "range_queries": range_report["model_queries"],
        "total_model_queries": range_report["model_queries"]
        + len(baseline_rows)
        + len(candidate_rows),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "cumulative_cpu_seconds": cumulative_cpu,
        "cumulative_cpu_limit_seconds": MAX_CUMULATIVE_CPU_SECONDS,
        "source_hashes_unchanged": sources_unchanged,
        "cpu_seconds": round(budget.used(), 6),
        "can_promote": False,
        "error": error,
    }
    write_new(CONFIRM_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--range-check", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_only()
    elif args.range_check:
        result = run_range_check(owner_approved=args.owner_approved)
    else:
        result = run_confirmation(owner_approved=args.owner_approved)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "range_queries",
                    "confirmation_queries",
                    "total_model_queries",
                    "baseline_gates",
                    "candidate_gates",
                    "confirm_face_generated",
                    "confirm_face_consumed",
                    "cpu_seconds",
                    "cumulative_cpu_seconds",
                    "can_promote",
                    "error",
                )
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
