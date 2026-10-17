"""Offline verification and input-hash reconstruction for PLAN-N1-99.

This tool reads the completed diagnostic, its one-shot attempt, source manifests
and endpoint receipts. It never loads a checkpoint or issues a model query.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as bind,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_unknown_abstention_factorial_diagnostic as diag,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_native_evidence_paired as paired,
)

PRIMARY = diag.REPORT
ATTEMPT = diag.ATTEMPT
OUT = ROOT / "reports/taiji_n1_unknown_abstention_factorial_binding_verification_20261017.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise diag.Refuse(message)


def sha_raw(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def verify() -> dict:
    require(
        PRIMARY.is_file() and ATTEMPT.is_file(), "completed diagnostic and attempt are required"
    )
    require(not OUT.exists(), f"verification report path already exists: {OUT}")
    primary_raw = PRIMARY.read_bytes()
    attempt_raw = ATTEMPT.read_bytes()
    primary = json.loads(primary_raw)
    attempt = json.loads(attempt_raw)
    require(
        primary.get("complete") is True and primary.get("status") == "complete",
        "primary diagnostic is incomplete",
    )
    require(primary.get("new_diagnostic_queries") == diag.NEW_QUERY_COUNT, "query count is not 135")
    require(
        primary.get("confirmation_queries") == 0 and primary.get("can_promote") is False,
        "diagnostic was mislabeled",
    )
    require(attempt.get("consumed_on_write") is True, "one-shot attempt receipt is missing")
    require(
        attempt.get("plan_sha256") == primary.get("plan_sha256"),
        "attempt and report plan hashes differ",
    )
    require(
        attempt.get("code_identity") == primary.get("code_identity"),
        "attempt and report code identities differ",
    )
    require(
        attempt.get("source_guards") == primary.get("source_guards"),
        "attempt and report source guards differ",
    )
    require(
        attempt.get("expected_new_diagnostic_queries") == diag.NEW_QUERY_COUNT,
        "attempt does not bind the approved 135-query scope",
    )
    require(sha_raw(diag.PLAN) == primary.get("plan_sha256"), "PLAN-N1-99 changed after execution")
    require(sha_raw(diag.FACE) == diag.FACE_SHA256, "v6 manifest changed")
    require(sha_raw(diag.BASELINE) == diag.BASELINE_SHA256, "v6 baseline report changed")
    require(
        diag._guards_unchanged(primary["source_guards"]),
        "one or more execution source guards changed",
    )
    require(diag._code_unchanged(primary["code_identity"]), "execution code identity changed")

    manifest = diag.read_json(diag.FACE)
    confirm_ids = set(manifest["subsets"]["confirm"])
    items = [
        episode
        for episode in manifest["episodes"]
        if episode.get("id") in confirm_ids and episode.get("kind") == "unknown"
    ]
    require(len(items) == diag.ITEMS_PER_CELL, "five v6 confirm unknown base items are required")
    variants = diag.build_variants(items)
    by_cell: dict[str, dict[str, dict]] = {"A": {item["id"]: item for item in items}}
    for cell in diag.NEW_CELLS:
        by_cell[cell] = {item["base_id"]: item for item in variants[cell]}

    _guards, _seals, endpoints, _bound_episodes = bind.readiness()
    require(
        set(endpoints) == {(seed, arm) for seed in diag.SEEDS for arm in diag.ARMS},
        "endpoint set changed",
    )
    endpoint_sha = {(seed, arm): sha_raw(Path(path)) for (seed, arm), path in endpoints.items()}

    baseline = diag.read_json(diag.BASELINE)
    baseline_by_key = {
        (row["id"], row["seed"], row["arm"]): row
        for row in baseline["rows"]
        if row["kind"] == "unknown" and row["id"] in by_cell["A"]
    }
    report_rows = primary.get("rows", [])
    require(
        len(report_rows) == 4 * diag.ITEMS_PER_CELL * diag.ENDPOINTS,
        "primary report row count differs from 180",
    )
    row_keys = [(row["condition"], row["base_id"], row["seed"], row["arm"]) for row in report_rows]
    expected_keys = {
        (cell, base_id, seed, arm)
        for cell, cell_items in by_cell.items()
        for base_id in cell_items
        for seed in diag.SEEDS
        for arm in diag.ARMS
    }
    require(
        len(row_keys) == len(set(row_keys)) and set(row_keys) == expected_keys,
        "report rows are not a complete one-to-one join",
    )

    item_bindings = []
    item_sha: dict[tuple[str, str], str] = {}
    for cell, cell_items in by_cell.items():
        for base_id, item in cell_items.items():
            projected = paired.project(item)
            model_input = {
                "observations": list(projected.observations),
                "question_utf8_hex": projected.question.hex(),
                "teacher_target_utf8_hex": item["answer"].encode("utf-8").hex(),
            }
            digest = canonical_sha(item)
            item_sha[(cell, base_id)] = digest
            item_bindings.append(
                {
                    "condition": cell,
                    "base_id": base_id,
                    "variant_id": item["id"],
                    "input_sha256": digest,
                    "model_input_sha256": canonical_sha(model_input),
                    "question_utf8_bytes": len(projected.question),
                    "observation_count": len(projected.observations),
                    "entity_count": len(item["entities"]),
                    "hamming_distances": [
                        diag.hamming(item["entities"][0], name) for name in item["entities"][1:]
                    ],
                    "observations": list(projected.observations),
                    "entities": list(item["entities"]),
                }
            )

    bound_rows = []
    for row in report_rows:
        cell = row["condition"]
        base_id = row["base_id"]
        item = by_cell[cell][base_id]
        expected_id = item["id"]
        require(row["id"] == expected_id, f"row ID does not match reconstructed {cell} input")
        require(
            row.get("query_source") == ("published_v6" if cell == "A" else "new_diagnostic"),
            "row query source is mislabeled",
        )
        require(row["target_symbols"] == list(b"unknown") + [256], "diagnostic target changed")
        exact_unknown = bytes.fromhex(row["output_hex"]) == b"unknown"
        require(
            exact_unknown == bool(row["free_exact"]), "row exactness disagrees with output bytes"
        )
        if cell == "A":
            source = baseline_by_key.get((base_id, row["seed"], row["arm"]))
            require(source is not None, "A baseline row is absent from the v6 report")
            for key in ("output_hex", "free_exact", "target_symbols", "teacher_symbols", "raw_hex"):
                require(row.get(key) == source.get(key), f"A baseline field {key} differs from v6")
        endpoint_path = Path(endpoints[row["seed"], row["arm"]]).resolve()
        bound_rows.append(
            {
                "condition": cell,
                "base_id": base_id,
                "variant_id": row["id"],
                "seed": row["seed"],
                "arm": row["arm"],
                "endpoint_path": str(endpoint_path),
                "endpoint_sha256": endpoint_sha[row["seed"], row["arm"]],
                "input_sha256": item_sha[(cell, base_id)],
                "output_hex": row["output_hex"],
                "exact_unknown": exact_unknown,
                "free_exact": bool(row["free_exact"]),
            }
        )

    return {
        "schema": "taiji-n1-unknown-abstention-factorial-binding-verification-v1",
        "status": "verified",
        "primary_report": str(PRIMARY),
        "primary_report_sha256": sha_raw(PRIMARY),
        "attempt_report_sha256": sha_raw(ATTEMPT),
        "plan_sha256": primary["plan_sha256"],
        "v6_manifest_sha256": sha_raw(diag.FACE),
        "v6_baseline_sha256": sha_raw(diag.BASELINE),
        "execution_report_had_per_row_input_sha256": False,
        "input_hashes_reconstructed_offline_from_frozen_code_and_name_seed": primary[
            "name_rng_seed"
        ],
        "new_model_queries_during_verification": 0,
        "model_loads_during_verification": 0,
        "checkpoint_loads_during_verification": 0,
        "source_guards_unchanged": True,
        "endpoint_count": len(endpoints),
        "item_bindings": item_bindings,
        "row_bindings": bound_rows,
    }


def main() -> int:
    result = verify()
    with OUT.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary_report_sha256": result["primary_report_sha256"],
                "bound_rows": len(result["row_bindings"]),
                "bound_items": len(result["item_bindings"]),
                "endpoints": result["endpoint_count"],
                "new_model_queries": 0,
                "output": str(OUT),
                "output_sha256": sha_raw(OUT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
