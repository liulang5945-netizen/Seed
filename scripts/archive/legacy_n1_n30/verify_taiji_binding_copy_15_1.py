"""15.1 rev2 stage 0a: data, raw-byte scoring, logs and interface preflight.

This stage builds the 24 train / 12 dev worlds, declares the sealed confirm
specification, freezes the scoring/logging/intervention/state-schema protocols,
checks the 14.6 parent identity and writes the data payload.  It does not train,
does not build a model, does not call Adam and does not generate confirm data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from taiji.binding_copy_data import (  # noqa: E402
    ACTION_BYTES,
    AXIS_TARGET,
    CONFIRM_ORDER_SEED,
    CONFIRM_SEED,
    DATA_SEED,
    DIAGONAL_COMBINATIONS,
    GOAL_BYTES,
    LAYER_TARGET,
    OBJECT_NAMES,
    ORDER_SEED,
    PLACE_NAMES,
    SCHEMA,
    SPLIT_WORLDS,
    STYLE_CODES,
    STYLES,
    TRAIN_COMBINATIONS,
    BindingWorld,
    DataError,
    axes_counts,
    build_split,
    confirm_spec,
    fixture_digest,
    fixture_payload,
    rotation_counts,
)
from taiji.binding_copy_scoring import (  # noqa: E402
    MAX_OUTPUT_BYTES,
    SCORING_SCHEMA,
    score_14_6_rows,
    self_test,
)

DESIGN = "plans/reference/SPEC-N15-01_BINDING_COPY_V2_20261102.md"
MAINLINE = "plans/MAINLINE.md"
OLD_PROPOSAL = "plans/reference/PLAN-N15-01_SPAN_BOUND_CONTINUOUS_20261102.md"
REPORT = "reports/taiji_binding_copy_15_1_data_20261102.json"
MANIFEST = "plans/manifests/taiji_binding_copy_15_1_data_20261102.json"
DATA_PAYLOAD = "output/taiji_core_binding_copy_15_1/data_v2.json"
DATA_SOURCE = "scripts/training/verify_taiji_binding_copy_15_1.py"
DATA_MODULE = "taiji/binding_copy_data.py"
SCORING_MODULE = "taiji/binding_copy_scoring.py"
PARENT_CARD = "reports/taiji_core_record_relation_14_6_20261102.json"
PARENT_MANIFEST = "plans/manifests/taiji_core_record_relation_14_6_20261102.json"
PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PREVIOUS_AUDIT = "reports/taiji_post14_algorithm_review_20261102.json"
CLOSEOUT_14_8 = "reports/taiji_core_record_relation_14_8_closeout_20261102.json"

STAGE_CPU_SECONDS = 60.0
STAGE_WALL_SECONDS = 120.0
STAGE_ADAM_CALLS = 0
PRODUCT_BYTES = 100 * 1024**2
REPORT_BYTES_UPPER = 1024**2

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(
    rows: list,
    check_id: str,
    passed: bool,
    observed,
    note: str = "",
    *,
    required: bool = True,
) -> None:
    rows.append(
        {
            "id": check_id,
            "passed": bool(passed),
            "required": bool(required),
            "observed": observed,
            "note": note,
        }
    )


def write_json(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"path": str(path), "sha256": sha256_text(path), "bytes": path.stat().st_size}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_json(relative_path: str) -> dict:
    return cast(dict, json.loads((ROOT / relative_path).read_text(encoding="utf-8")))


def run_type_check(paths: list[str]) -> dict:
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--follow-imports=skip",
        "--ignore-missing-imports",
        *paths,
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        wall = time.perf_counter() - start
        errors = 0
        for line in result.stdout.splitlines():
            if line.startswith("Found ") and " error" in line:
                errors = int(line.split()[1])
        return {
            "returncode": result.returncode,
            "errors": errors,
            "output": result.stdout.strip(),
            "wall_seconds": wall,
            "passed": result.returncode == 0 and errors == 0,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "returncode": None,
            "errors": None,
            "output": f"{type(exc).__name__}: {exc}",
            "wall_seconds": time.perf_counter() - start,
            "passed": False,
        }


def run_lint(paths: list[str]) -> dict:
    start = time.perf_counter()
    ruff = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    ruff_format = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return {
        "ruff": {"returncode": ruff.returncode, "output": ruff.stdout.strip()},
        "ruff_format": {
            "returncode": ruff_format.returncode,
            "output": ruff_format.stdout.strip(),
        },
        "black": {
            "returncode": None,
            "status": "not_run",
            "reason": (
                "the installed black 26.5.1 does not terminate on the stage runners "
                "in this environment; ruff format is the declared formatter check"
            ),
        },
        "wall_seconds": time.perf_counter() - start,
        "passed": ruff.returncode == 0 and ruff_format.returncode == 0,
    }


def aggregate_layers(worlds: tuple[BindingWorld, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for world in worlds:
        for example in world.examples:
            counts[example.layer] = counts.get(example.layer, 0) + 1
    return counts


def aggregate_axes(worlds: tuple[BindingWorld, ...]) -> dict[str, int]:
    counts = {key: 0 for key in AXIS_TARGET}
    for world in worlds:
        for key, value in axes_counts(world.examples).items():
            counts[key] += value
    return counts


def aggregate_styles(worlds: tuple[BindingWorld, ...]) -> dict[str, int]:
    counts = {STYLES[index]: 0 for index in range(len(STYLES))}
    for world in worlds:
        for record in world.records:
            counts[STYLES[record.style]] += 1
    return counts


def source_ids(worlds: tuple[BindingWorld, ...]) -> set[str]:
    return {record.source_id for world in worlds for record in world.records}


def record_bytes(worlds: tuple[BindingWorld, ...]) -> list[int]:
    return [sum(len(record.raw) for record in world.records) for world in worlds]


def parent_record(card: dict, manifest: dict) -> dict:
    candidates = [
        item for item in card["candidates"] if item["seed"] == 27031 and item["arm"] == "A_full"
    ]
    if len(candidates) != 1:
        raise DataError("14.6 parent candidate is not unique")
    retained = [
        item
        for item in manifest["retained"]
        if item["seed"] == 27031 and item["arm"] == "A_full" and item["update_index"] == 300
    ]
    if len(retained) != 1:
        raise DataError("14.6 parent bundle is not unique")
    candidate = candidates[0]
    bundle = retained[0]
    path = ROOT / bundle["path"]
    if not path.exists():
        raise DataError("14.6 parent bundle is missing")
    actual_sha = sha256_text(path)
    actual_bytes = path.stat().st_size
    return {
        "bundle": bundle["path"],
        "expected_sha256": bundle["expected_sha256"],
        "actual_sha256": actual_sha,
        "expected_bytes": int(bundle["expected_bytes"]),
        "actual_bytes": actual_bytes,
        "candidate": candidate["candidate_final"],
        "parameter_count": candidate["parameter_count"],
        "trainable_count": candidate["trainable_count"],
        "source": "14.6 seed 27031 A_full step300",
        "local_behavior_asset": True,
        "qualified_hold_parent": False,
        "new_adam_from_zero": True,
        "old_adam_archived": True,
        "file_verified": bool(
            actual_sha == bundle["expected_sha256"]
            and actual_bytes == int(bundle["expected_bytes"])
            and bundle["verified"] is True
        ),
    }


def protocol() -> dict:
    return {
        "scoring": {
            "main": (
                "completed_action_exact: strict UTF-8, one canonical 27-byte action "
                "raw bytes exactly equal and EOS terminated; max 64 bytes/65 predictions; "
                "early EOS, parse failure and truncation stay in the denominator"
            ),
            "normalization": "bytes or integer tuple/list -> bytes; no display-text inference",
            "priors": [
                "oracle",
                "constant",
                "list",
                "wrong_name",
                "truncation",
                "byte_type_swap",
                "empty",
                "early_eos",
                "invalid_utf8",
                "not_terminated",
            ],
            "axes": [
                "overall",
                "unseen_input_description",
                "unseen_output_name",
                "both_new",
                "unseen_combination",
                "per_world",
            ],
            "reverse": "accuracy plus paired raw-byte/EOS consistency",
            "entity_stage": (
                "object byte 3-14 and place byte 21-26 for the 27-byte canonical action; "
                "teacher-prefix probe under the correct history; free-prefix trajectory "
                "and first divergence; Q record/position/null/gamma/R; argmax source/position"
            ),
            "trace_identity": [
                "parent",
                "parameters",
                "state",
                "source",
                "schema",
                "data_order",
            ],
        },
        "logs": {
            "per_step": [
                "raw_byte",
                "eos_reason",
                "gamma",
                "lambda",
                "q_null",
                "total_copy_posterior",
                "argmax_source",
                "argmax_position",
            ],
            "full_q_examples": 24,
            "short_trace": "all output steps for every evaluated example",
            "teacher_prefix": "correct history probe at the entity stage",
            "free_prefix": "full trajectory and first divergence",
            "identity": ["parent", "parameters", "state", "source", "schema", "data_order"],
            "source": "protocol only; 0a emits no model log",
        },
        "interventions": {
            "separate": ["record", "position", "gamma"],
            "rescue_identity": (
                "remove the intervention and restore the original content/state; the "
                "distribution returns bitwise"
            ),
            "oracle": "entity-stage oracle; gamma=0 forced copy; gamma=1 generator_only",
            "examples_per_candidate": 24,
            "fixed_names": "per-world target 0 and 15; names frozen in the data manifest",
            "conditions": [
                "no_source",
                "wrong_content",
                "wrong_position",
                "continuation_zero",
                "identity_rescue",
            ],
            "full_q_examples": 24,
            "source": "protocol only; 0a runs no model",
        },
        "state_schema": {
            "m": (
                "sub-probability over (source_id, version, byte_offset), mass h<=1; "
                "zero at answer start; shift within the same record by +1 byte"
            ),
            "coverage": "output copy posterior, not the 14 soft-read coverage",
            "source_version": "registered per copied byte",
            "H": "inherited recursive state",
            "F": "inherited fast/plastic state",
            "optimizer": "new Adam from zero; old Adam archived",
            "mode_rng": "separate saved random stream",
            "prefix_cursor": "per-example prefix mode and cursor",
            "budget": "charged per stage",
            "receipt": "legal restore receipt",
            "restore": "at least one non-empty cursor boundary mid-output",
            "replay": "next BPTT from stored original/actual prefix and random stream",
            "revision": "new schema/revision; no old R0 value equality signature",
        },
        "interface": {
            "model_interface": "256 raw bytes plus EOS; no name list or style id",
            "style_not_forward": True,
            "source_id_not_forward": True,
            "gold_only_loss_or_oracle": True,
            "parent": "inherited 14.6 A_full step300, rebuilt with new Adam from zero",
            "new_reader": "declared for 15.1.0b; not built in 0a",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="data", choices=("data",))
    args = parser.parse_args()
    if args.stage != "data":
        raise DataError("only the data stage is implemented in 15.1.0a")
    report_path = ROOT / REPORT
    manifest_path = ROOT / MANIFEST
    data_path = ROOT / DATA_PAYLOAD
    # The data payload is deterministic and may be rewritten while the stage has
    # no successful card yet; the report and manifest remain write-once.
    for path in (report_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"write-once 15.1.0a path exists: {path}")

    design_sha = sha256_text(ROOT / DESIGN)
    mainline_sha = sha256_text(ROOT / MAINLINE)
    old_sha = sha256_text(ROOT / OLD_PROPOSAL)
    data_module_sha = sha256_text(ROOT / DATA_MODULE)
    scoring_module_sha = sha256_text(ROOT / SCORING_MODULE)
    data_source_sha = sha256_text(ROOT / DATA_SOURCE)

    train_worlds = build_split("train")
    dev_worlds = build_split("dev")
    confirm = confirm_spec()
    confirm_raise: str | None = None
    try:
        build_split("confirm")
    except DataError as exc:
        confirm_raise = str(exc)
    if confirm_raise is None:
        raise DataError("the public confirm build did not stay sealed")

    train_examples = sum(len(world.examples) for world in train_worlds)
    dev_examples = sum(len(world.examples) for world in dev_worlds)
    train_records = sum(len(world.records) for world in train_worlds)
    dev_records = sum(len(world.records) for world in dev_worlds)
    train_sources = source_ids(train_worlds)
    dev_sources = source_ids(dev_worlds)
    train_layers = aggregate_layers(train_worlds)
    dev_layers = aggregate_layers(dev_worlds)
    train_axes = aggregate_axes(train_worlds)
    dev_axes = aggregate_axes(dev_worlds)
    train_styles = aggregate_styles(train_worlds)
    dev_styles = aggregate_styles(dev_worlds)
    train_bytes = record_bytes(train_worlds)
    dev_bytes = record_bytes(dev_worlds)
    train_digest = fixture_digest(train_worlds)
    dev_digest = fixture_digest(dev_worlds)
    train_dev_digest = fixture_digest(train_worlds + dev_worlds)
    rotation = rotation_counts(dev_worlds)

    payload = fixture_payload(train_worlds + dev_worlds, confirm)
    data_info = write_json(data_path, payload)
    data_payload_bytes = int(data_info["bytes"])

    expected_counts = {
        "train": {
            "worlds": 24,
            "examples": 288,
            "layers": {"seen_combination": 288},
            "records": 192,
            "source_ids": 192,
        },
        "dev": {
            "worlds": 12,
            "examples": 240,
            "layers": {
                "seen_combination": 144,
                "unseen_combination": 36,
                "new_description": 60,
            },
            "records": 108,
            "source_ids": 108,
        },
        "total": {"examples": 528, "records": 300, "source_ids": 300},
    }
    observed_counts: dict[str, Any] = {
        "train": {
            "worlds": len(train_worlds),
            "examples": train_examples,
            "layers": train_layers,
            "records": train_records,
            "source_ids": len(train_sources),
        },
        "dev": {
            "worlds": len(dev_worlds),
            "examples": dev_examples,
            "layers": dev_layers,
            "records": dev_records,
            "source_ids": len(dev_sources),
        },
        "total": {
            "examples": train_examples + dev_examples,
            "records": train_records + dev_records,
            "source_ids": len(train_sources | dev_sources),
        },
    }
    counts_ok = observed_counts == expected_counts

    axes_ok = (
        train_axes
        == {
            "seen_description_seen_name": 288,
            "new_description_seen_name": 0,
            "seen_description_new_name": 0,
            "new_description_new_name": 0,
            "new_description": 0,
            "unseen_output_name": 0,
            "both_new": 0,
            "unseen_combination": 0,
            "seen_combination": 288,
        }
        and dev_axes == AXIS_TARGET
        and dev_layers == LAYER_TARGET
        and rotation["new_description_name_counts"]
        == {"青釉瓷瓶": 3, "铁灰木匣": 3, "素白陶罐": 3, "赭红石臼": 3}
        and rotation["seen_name_on_new_description"] == 9
        and rotation["new_name_on_seen_description"] == 9
        and rotation["both_new_worlds"] == 3
    )

    styles_ok = (
        sorted(train_styles) == sorted(STYLES)
        and sorted(dev_styles) == sorted(STYLES)
        and all(
            record.style in (0, 1, 2)
            for world in train_worlds + dev_worlds
            for record in world.records
        )
        and max(train_bytes) <= 243
        and max(dev_bytes) <= 279
        and min(train_bytes) >= 219
        and min(dev_bytes) >= 252
        and set(train_sources).isdisjoint(dev_sources)
        and len(train_sources | dev_sources) == 300
        and all(
            len(record.raw) <= 512
            for world in train_worlds + dev_worlds
            for record in world.records
        )
    )

    contract_ok = (
        len(train_worlds) == 24
        and len(dev_worlds) == 12
        and all(validate_for_check(world) for world in train_worlds + dev_worlds)
        and counts_ok
        and styles_ok
    )

    confirm_ok = (
        confirm["generated"] is False
        and confirm["sealed"] is True
        and confirm["worlds"] == 12
        and confirm["examples"] == 240
        and confirm["examples_per_world"] == 20
        and confirm["objects"] == 4
        and confirm["object_indices"] == [0, 1, 2, 4]
        and confirm["record_lines"] == 9
        and confirm["record_bytes_max"] == 279
        and confirm["axes"] == AXIS_TARGET
        and confirm["layers"] == LAYER_TARGET
        and confirm["data_seed"] == CONFIRM_SEED
        and confirm["order_seed"] == CONFIRM_ORDER_SEED
        and confirm_raise is not None
        and len(payload["examples"]) == 528
        and len(payload["worlds"]) == 36
    )

    card = load_json(PARENT_CARD)
    manifest = load_json(PARENT_MANIFEST)
    parent = parent_record(card, manifest)
    parent_ok = (
        parent["file_verified"] is True
        and parent["candidate"]
        == "a3104b3e17c4259cc49347855f53ebc437cf2ec445f6965e857a685e5b8ab756"
        and parent["expected_sha256"]
        == "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
        and parent["expected_bytes"] == 64983135
        and card["status"] == "native_byte_task_failed"
        and card["release_14_7"] is False
        and parent["local_behavior_asset"] is True
        and parent["qualified_hold_parent"] is False
        and parent["new_adam_from_zero"] is True
    )

    # The scoring self-check re-scores all 4,032 stored 14.6 rows.
    rows14 = []
    for candidate in card["candidates"]:
        rows14.extend(candidate["step0"]["rows"])
        for condition in ("real", "permuted", "reverse"):
            rows14.extend(candidate["step300"][condition]["rows"])
        for name in ("no_source", "wrong_record", "wrong_position", "rescue"):
            rows14.extend(candidate["step300"]["structural"]["conditions"][name]["rows"])
    rescored = score_14_6_rows(rows14)
    first_example = dev_worlds[0].examples[0]
    prior = self_test(
        first_example.expected,
        dev_worlds[0].candidate_actions[0].encode(),
        dev_worlds[0].candidate_actions,
    )
    scoring_ok = (
        rescored["n"] == 4032
        and rescored["matched"] == 4032
        and len(rescored["mismatches"]) == 0
        and rescored["counts"] == rescored["stored_counts"]
        and prior["passed"] is True
    )

    protocol_data = protocol()
    logs_ok = (
        protocol_data["logs"]["full_q_examples"] == 24
        and protocol_data["logs"]["per_step"]
        == [
            "raw_byte",
            "eos_reason",
            "gamma",
            "lambda",
            "q_null",
            "total_copy_posterior",
            "argmax_source",
            "argmax_position",
        ]
        and protocol_data["logs"]["source"].startswith("protocol only")
    )
    intervention_ok = (
        protocol_data["interventions"]["separate"] == ["record", "position", "gamma"]
        and protocol_data["interventions"]["examples_per_candidate"] == 24
        and protocol_data["interventions"]["full_q_examples"] == 24
        and "bitwise" in protocol_data["interventions"]["rescue_identity"]
    )
    interface_ok = (
        protocol_data["interface"]["model_interface"]
        == "256 raw bytes plus EOS; no name list or style id"
        and protocol_data["interface"]["style_not_forward"] is True
        and protocol_data["interface"]["source_id_not_forward"] is True
        and protocol_data["state_schema"]["optimizer"] == "new Adam from zero; old Adam archived"
        and protocol_data["state_schema"]["restore"].startswith("at least one non-empty cursor")
    )

    checks: list[dict] = []
    check(
        checks,
        "data_counts",
        counts_ok,
        {
            "expected": expected_counts,
            "observed": observed_counts,
            "record_bytes": {
                "train_min": min(train_bytes),
                "train_max": max(train_bytes),
                "dev_min": min(dev_bytes),
                "dev_max": max(dev_bytes),
            },
            "styles": {"train": train_styles, "dev": dev_styles},
        },
    )
    check(
        checks,
        "data_contract",
        contract_ok,
        {
            "worlds_validated": 36,
            "train_dev_examples": train_examples + dev_examples,
            "train_dev_records": train_records + dev_records,
            "source_ids": {
                "train": len(train_sources),
                "dev": len(dev_sources),
                "union": len(train_sources | dev_sources),
                "disjoint": set(train_sources).isdisjoint(dev_sources),
            },
            "goal_bytes": GOAL_BYTES,
            "action_bytes": ACTION_BYTES,
            "train_dev_digest": train_dev_digest,
            "styles": {"train": train_styles, "dev": dev_styles},
        },
        "Every train/dev world was validated by the generator and re-checked here; "
        "the confirm split has no public build.",
    )
    check(
        checks,
        "axes",
        axes_ok,
        {
            "train": train_axes,
            "dev": dev_axes,
            "target": AXIS_TARGET,
            "layers": LAYER_TARGET,
            "rotation": rotation,
            "per_world": {
                "seen_combination": 12,
                "unseen_combination": 3,
                "new_description": 5,
            },
            "worlds": 12,
        },
        "The new input description and the new output name are separate axes; the "
        "cross is 135/45/45/15, not one combined unseen entity layer.",
    )
    check(
        checks,
        "styles",
        styles_ok,
        {
            "styles": list(STYLES),
            "style_codes": STYLE_CODES,
            "style_index": "world*5 + record_index*7 + split_code*11 mod 3",
            "record_lines": {"train": 8, "dev": 9},
            "record_bytes": {
                "train_min": min(train_bytes),
                "train_max": max(train_bytes),
                "dev_min": min(dev_bytes),
                "dev_max": max(dev_bytes),
                "train_cap": 243,
                "dev_cap": 279,
                "source_cap": 512,
            },
            "train": train_styles,
            "dev": dev_styles,
        },
    )
    check(
        checks,
        "confirm_spec",
        confirm_ok,
        {
            "public_build_raises": confirm_raise,
            "spec": confirm,
            "payload_examples": len(payload["examples"]),
            "payload_worlds": len(payload["worlds"]),
            "payload_confirm": confirm,
        },
        "The confirm specification is sealed; no confirm example, record or source "
        "id is written, and the public confirm build refuses.",
    )
    check(
        checks,
        "scoring",
        scoring_ok,
        {
            "self_test": prior,
            "14_6": {
                "rows": rescored["n"],
                "matched": rescored["matched"],
                "mismatches": len(rescored["mismatches"]),
                "counts": rescored["counts"],
                "stored_counts": rescored["stored_counts"],
            },
            "entity_exact_stored": rescored["stored_counts"]["entity_exact"],
            "normalization": "bytes or integer tuple/list -> bytes; no display text",
            "prior_cases": sorted(prior["cases"]),
        },
        "The new scorer re-scores all 4,032 stored 14.6 rows and matches every "
        "stored completed/full/entity/utf8/byte count exactly.",
    )
    check(
        checks,
        "logs",
        logs_ok,
        {
            "protocol": protocol_data["logs"],
            "full_q_examples": 24,
            "short_trace": "all output steps",
            "source": "protocol only; 0a emits no model log",
        },
    )
    check(
        checks,
        "interventions",
        intervention_ok,
        {
            "protocol": protocol_data["interventions"],
            "source": "protocol only; 0a runs no model",
        },
    )
    check(
        checks,
        "parent",
        parent_ok,
        parent,
        "The 14.6 A_full step300 bundle is a local behavior asset; it is not a "
        "qualified long-term hold parent, and the new Adam starts from zero.",
    )
    check(
        checks,
        "interface",
        interface_ok,
        {
            "state_schema": protocol_data["state_schema"],
            "interface": protocol_data["interface"],
            "data_fields": [
                "goal",
                "records",
                "expected",
                "action",
                "target",
                "actions",
                "flags",
            ],
            "no_model": True,
        },
    )
    check(
        checks,
        "no_model_forward",
        True,
        {
            "model_forwards": 0,
            "adam_calls": 0,
            "parameter_changes": 0,
            "note": "0a builds data, scoring and protocol declarations only",
        },
    )

    type_check = run_type_check([DATA_MODULE, SCORING_MODULE, DATA_SOURCE])
    check(checks, "type_check", type_check["passed"], type_check)
    lint = run_lint([DATA_MODULE, SCORING_MODULE, DATA_SOURCE])
    check(checks, "lint", lint["passed"], lint)

    tool_wall = float(type_check.get("wall_seconds", 0.0)) + float(lint.get("wall_seconds", 0.0))
    stage_cpu = (time.process_time() - STARTED_CPU) + tool_wall
    stage_wall = time.perf_counter() - STARTED_WALL
    product_upper = data_payload_bytes + REPORT_BYTES_UPPER
    within = {
        "cpu": stage_cpu <= STAGE_CPU_SECONDS,
        "wall": stage_wall <= STAGE_WALL_SECONDS,
        "adam": STAGE_ADAM_CALLS <= STAGE_ADAM_CALLS,
        "product": product_upper <= PRODUCT_BYTES,
        "all": (
            stage_cpu <= STAGE_CPU_SECONDS
            and stage_wall <= STAGE_WALL_SECONDS
            and product_upper <= PRODUCT_BYTES
        ),
        "limits": {
            "cpu_seconds": STAGE_CPU_SECONDS,
            "wall_seconds": STAGE_WALL_SECONDS,
            "adam_calls": STAGE_ADAM_CALLS,
            "product_bytes": PRODUCT_BYTES,
        },
        "observed": {
            "cpu_seconds": stage_cpu,
            "wall_seconds": stage_wall,
            "adam_calls": 0,
            "data_payload_bytes": data_payload_bytes,
            "report_bytes_upper_bound": REPORT_BYTES_UPPER,
            "product_bytes_upper_bound": product_upper,
        },
    }
    check(
        checks,
        "cost",
        bool(within["all"]),
        within,
        "The data payload is exact; the report is charged at its declared 1 MiB "
        "upper bound instead of a self-referential size.",
    )

    authorization = {
        "spec": DESIGN,
        "spec_sha256": design_sha,
        "mainline": MAINLINE,
        "mainline_sha256": mainline_sha,
        "old_n15_proposal": OLD_PROPOSAL,
        "old_n15_proposal_sha256": old_sha,
        "old_n15_proposal_not_executed": True,
        "approved_scope": "15.1.0a",
        "user_instruction": "15.1.0a沿继续推进",
    }
    authorization_ok = (
        old_sha == "51cfe08a20cb87af16c3a10939ccf487db1af63d4b693007fc98921102df6897"
        and (ROOT / DESIGN).exists()
        and (ROOT / MAINLINE).exists()
        and authorization["old_n15_proposal_not_executed"] is True
    )
    check(
        checks,
        "authorization",
        authorization_ok,
        authorization,
        "The old N15.1 proposal remains the 14.8-bound historical bytes; the "
        "current execution follows the 15.1 rev2 specification only.",
    )
    source_hashes = {
        DATA_MODULE: data_module_sha,
        SCORING_MODULE: scoring_module_sha,
        DATA_SOURCE: data_source_sha,
        DESIGN: design_sha,
        MAINLINE: mainline_sha,
        OLD_PROPOSAL: old_sha,
        PARENT_CARD: sha256_text(ROOT / PARENT_CARD),
        PARENT_MANIFEST: sha256_text(ROOT / PARENT_MANIFEST),
        PARENT_BUNDLE: sha256_text(ROOT / PARENT_BUNDLE),
        PREVIOUS_AUDIT: sha256_text(ROOT / PREVIOUS_AUDIT),
        CLOSEOUT_14_8: sha256_text(ROOT / CLOSEOUT_14_8),
    }
    data_summary = {
        "schema": SCHEMA,
        "split_worlds": SPLIT_WORLDS,
        "split_seeds": {
            "train": [DATA_SEED, ORDER_SEED],
            "dev": [DATA_SEED, ORDER_SEED],
            "confirm": [CONFIRM_SEED, CONFIRM_ORDER_SEED],
        },
        "train": {
            **observed_counts["train"],
            "record_bytes_min": min(train_bytes),
            "record_bytes_max": max(train_bytes),
            "styles": train_styles,
            "digest": train_digest,
            "world_digests": 24,
        },
        "dev": {
            **observed_counts["dev"],
            "record_bytes_min": min(dev_bytes),
            "record_bytes_max": max(dev_bytes),
            "styles": dev_styles,
            "digest": dev_digest,
            "world_digests": 12,
        },
        "total": {
            **observed_counts["total"],
            "record_bytes": sum(train_bytes) + sum(dev_bytes),
            "world_digests": 36,
        },
        "axes": {"train": train_axes, "dev": dev_axes, "target": AXIS_TARGET},
        "layers": LAYER_TARGET,
        "rotation": rotation,
        "styles": {"train": train_styles, "dev": dev_styles},
        "digests": {
            "train": train_digest,
            "dev": dev_digest,
            "train_dev": train_dev_digest,
        },
        "vocabulary": {
            "object_names": list(OBJECT_NAMES),
            "place_names": list(PLACE_NAMES),
        },
        "confirm_spec": confirm,
        "data_payload": data_info,
        "train_combinations": [list(pair) for pair in TRAIN_COMBINATIONS],
        "diagonal_combinations": [list(pair) for pair in DIAGONAL_COMBINATIONS],
    }
    scoring_summary = {
        "protocol": protocol_data["scoring"],
        "self_test": prior,
        "14_6": {
            "rows": rescored["n"],
            "matched": rescored["matched"],
            "mismatches": len(rescored["mismatches"]),
            "counts": rescored["counts"],
            "stored_counts": rescored["stored_counts"],
        },
        "entity": {
            "object_bytes": [3, 15],
            "place_bytes": [21, 27],
            "stored_entity_exact": rescored["stored_counts"]["entity_exact"],
            "current_entity_exact": rescored["counts"]["entity_exact"],
        },
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "schema": SCORING_SCHEMA,
    }
    cost = {
        "stage": "15.1.0a",
        "stage_cpu_seconds": stage_cpu,
        "stage_wall_seconds": stage_wall,
        "stage_adam_calls": 0,
        "model_forwards": 0,
        "parameter_changes": 0,
        "data_payload_bytes": data_payload_bytes,
        "report_bytes_upper_bound": REPORT_BYTES_UPPER,
        "product_bytes_upper_bound": product_upper,
        "product_bytes_cap": PRODUCT_BYTES,
        "within": within,
    }
    report: dict[str, Any] = {
        "schema": "taiji-binding-copy-15-1-data-v1",
        "step": "15.1.0a",
        "date": "2026-11-02",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "passed_15_1_0a_data_scoring_interface",
        "passed": True,
        "failed_required": [],
        "next": "15.1.0b graph engineering",
        "owner_approval_required_for_next": True,
        "authorization": authorization,
        "parent": parent,
        "data": data_summary,
        "scoring": scoring_summary,
        "logs": protocol_data["logs"],
        "interventions": protocol_data["interventions"],
        "interface": protocol_data["interface"],
        "state_schema": protocol_data["state_schema"],
        "confirm_spec": confirm,
        "confirm_read": False,
        "confirm_generated": False,
        "no_model_forward": True,
        "no_adam_call": True,
        "no_weight_change": True,
        "checks": checks,
        "cost": cost,
        "within": within,
        "source_hashes": source_hashes,
        "limits": (
            "0a data/scoring/logging/interface only: 0 model forward, 0 Adam, 0 "
            "weight change, no confirm data; no task capability, general Chinese, "
            "L3/L4, continuous state, cross-session or growth is claimed."
        ),
        "type_check": type_check,
        "lint": lint,
    }
    manifest_data: dict[str, Any] = {
        "schema": "taiji-binding-copy-15-1-data-manifest-v1",
        "step": "15.1.0a",
        "date": "2026-11-02",
        "status": "passed_15_1_0a_data_scoring_interface",
        "authorization": authorization,
        "parent": parent,
        "data": data_summary,
        "scoring": scoring_summary,
        "logs": protocol_data["logs"],
        "interventions": protocol_data["interventions"],
        "interface": protocol_data["interface"],
        "state_schema": protocol_data["state_schema"],
        "confirm_spec": confirm,
        "confirm_read": False,
        "next": "15.1.0b graph engineering after owner approval",
        "owner_approval_required_for_next": True,
        "cost": cost,
        "within": within,
        "source_hashes": source_hashes,
    }
    manifest_info = write_json(manifest_path, manifest_data)
    report["manifest"] = {
        "path": relative(manifest_path),
        "sha256": manifest_info["sha256"],
        "bytes": manifest_info["bytes"],
    }
    check(
        checks,
        "manifest_identity",
        manifest_info["sha256"] == sha256_text(manifest_path),
        {"path": report["manifest"]["path"], "sha256": manifest_info["sha256"]},
    )
    required = [row for row in checks if row.get("required", True)]
    report["required_checks"] = len(required)
    report["failed_required"] = [row["id"] for row in required if not row["passed"]]
    report["passed"] = not report["failed_required"]
    if not report["passed"]:
        report["status"] = "instrument_invalid"
        report["next"] = (
            "correct the declared 15.1.0a material; do not lower the gate or delete rows"
        )
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    info = write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "failed_required": report["failed_required"],
                "stage_cpu_seconds": stage_cpu,
                "stage_wall_seconds": stage_wall,
                "data_payload_bytes": data_payload_bytes,
                "train_examples": train_examples,
                "dev_examples": dev_examples,
                "axes_dev": dev_axes,
                "rotation": rotation,
                "confirm_spec": confirm,
                "14_6_scoring": {
                    "rows": rescored["n"],
                    "matched": rescored["matched"],
                },
                "parent": parent["candidate"],
                "next": report["next"],
                "report": info["path"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


def validate_for_check(world: BindingWorld) -> bool:
    """The generator already validates; this re-runs the declared check."""
    from taiji.binding_copy_data import validate_world

    validate_world(world)
    return True


if __name__ == "__main__":
    raise SystemExit(main())
