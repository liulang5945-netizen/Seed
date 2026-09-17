"""Freeze and judge the preregistered H3.7 matched development matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji.internalization import content_digest

EXPECTED_DATASET_DIGEST = "0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691"
EXPECTED_PREREGISTRATION = (
    "plans/reference/M5_R2_H3_7_FACTORIZED_RESPONSE_WORKSPACE_CONTRACT_20260917.md"
)
EXPECTED_SEEDS = (20260917, 20260918, 20260919)
EXPECTED_EPOCHS = 10
EXPECTED_EPISODES = 120
EXPECTED_GLOBAL_STEP = 5370
EXPECTED_PARAMETER_BUDGET = 300_000
EXPECTED_PLAN_WIDTH = 48
EXPECTED_PLAN_SLOTS = 4
EXPECTED_PHASE_STRIDE = 16
EXPECTED_GEOMETRIES = {
    "control": "signed_hash_span",
    "treatment": "h3_7_factorized_response_chunks",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-reports", nargs=3, required=True, type=Path)
    parser.add_argument("--treatment-reports", nargs=3, required=True, type=Path)
    parser.add_argument("--ablation-reports", nargs=3, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"report must be a JSON object: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _seed_from_path(path: Path, report: dict[str, Any]) -> int:
    reported = report.get("model_seed")
    if reported is not None:
        return int(reported)
    match = re.search(r"seed(\d+)", path.stem)
    if match is None:
        raise SystemExit(f"cannot recover model seed from report: {path}")
    return int(match.group(1))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _load_checkpoint(report: dict[str, Any], report_path: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint_value = report.get("checkpoint")
    _require(isinstance(checkpoint_value, str), f"report has no checkpoint path: {report_path}")
    checkpoint_path = _resolve_repo_path(checkpoint_value)
    _require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict), f"checkpoint must be a mapping: {checkpoint_path}")
    expected_digest = content_digest(
        {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    )
    _require(
        payload.get("checkpoint_digest") == expected_digest,
        f"checkpoint digest mismatch: {checkpoint_path}",
    )
    return checkpoint_path, payload


def _target_map_digest(payload: dict[str, Any]) -> str:
    return content_digest(
        {str(key): value.detach().cpu().clone() for key, value in sorted(payload.items())}
    )


def _validate_common_run(
    report: dict[str, Any],
    report_path: Path,
    *,
    arm: str,
    expected_seed: int,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    seed = _seed_from_path(report_path, report)
    _require(seed == expected_seed, f"seed mismatch in {report_path}: {seed} != {expected_seed}")
    _require(report.get("status") == "completed", f"run did not complete: {report_path}")
    _require(
        report.get("pre_registration") == EXPECTED_PREREGISTRATION,
        f"wrong preregistration in {report_path}",
    )
    _require(report.get("final_deferred") is True, f"final was not deferred: {report_path}")

    dataset = report.get("dataset")
    _require(isinstance(dataset, dict), f"dataset manifest missing: {report_path}")
    _require(
        dataset.get("digest") == EXPECTED_DATASET_DIGEST,
        f"dataset digest mismatch: {report_path}",
    )
    config = report.get("config")
    _require(isinstance(config, dict), f"config missing: {report_path}")
    _require(
        config.get("response_plan_target_geometry") == EXPECTED_GEOMETRIES[arm],
        f"target geometry mismatch: {report_path}",
    )
    if arm == "control":
        _require(
            config.get("response_phase_readout") is True,
            f"control phase readout missing: {report_path}",
        )
        _require(
            config.get("response_plan_readout") is False,
            f"control owns a plan readout: {report_path}",
        )
        _require(
            config.get("response_plan_variant") == "single",
            f"control variant mismatch: {report_path}",
        )
    else:
        _require(
            config.get("response_phase_readout") is False,
            f"treatment owns phase readout: {report_path}",
        )
        _require(
            config.get("response_plan_readout") is True,
            f"treatment plan readout missing: {report_path}",
        )
        _require(
            config.get("response_plan_variant") == "factorized_v1",
            f"treatment variant mismatch: {report_path}",
        )
        _require(
            config.get("response_plan_width") == EXPECTED_PLAN_WIDTH,
            f"plan width mismatch: {report_path}",
        )
        _require(
            config.get("response_plan_slots") == EXPECTED_PLAN_SLOTS,
            f"plan slot mismatch: {report_path}",
        )
        _require(
            config.get("response_plan_phase_stride") == EXPECTED_PHASE_STRIDE,
            f"phase stride mismatch: {report_path}",
        )

    capacity = report.get("capacity")
    _require(isinstance(capacity, dict), f"capacity record missing: {report_path}")
    _require(
        capacity.get("target_active_parameters") == EXPECTED_PARAMETER_BUDGET,
        f"parameter budget mismatch: {report_path}",
    )
    _require(
        isinstance(capacity.get("effective_active_parameters"), int)
        and capacity["effective_active_parameters"] <= EXPECTED_PARAMETER_BUDGET,
        f"effective budget mismatch: {report_path}",
    )
    _require(capacity.get("within_target") is True, f"run exceeded parameter budget: {report_path}")

    preflight = report.get("preflight")
    _require(isinstance(preflight, dict), f"preflight missing: {report_path}")
    _require(preflight.get("status") == "passed", f"preflight did not pass: {report_path}")
    for key in ("zero_step_roundtrip", "caller_model_unchanged", "atomic_save"):
        _require(preflight.get(key) is True, f"preflight {key} failed: {report_path}")
    update = preflight.get("one_response_update")
    _require(
        isinstance(update, dict) and update.get("status") == "completed",
        f"child update preflight failed: {report_path}",
    )
    if arm == "treatment":
        factorized = preflight.get("factorized_response_plan")
        _require(isinstance(factorized, dict), f"H3.7 preflight missing: {report_path}")
        _require(factorized.get("status") == "passed", f"H3.7 preflight failed: {report_path}")
        for key in (
            "plan_target_candidate_changed",
            "bridge_changed_after_byte_update",
            "slot_planner_changed_after_byte_update",
            "reset_cleared_plan",
            "protected_readout_unchanged",
            "caller_restore_verified_in_finally",
        ):
            _require(factorized.get(key) is True, f"H3.7 preflight {key} failed: {report_path}")

    training = report.get("training")
    _require(isinstance(training, dict), f"training record missing: {report_path}")
    _require(training.get("status") == "completed", f"training did not complete: {report_path}")
    _require(training.get("epochs") == EXPECTED_EPOCHS, f"epoch count mismatch: {report_path}")
    _require(
        training.get("episodes") == EXPECTED_EPISODES, f"episode count mismatch: {report_path}"
    )
    _require(
        training.get("global_step") == EXPECTED_GLOBAL_STEP, f"update count mismatch: {report_path}"
    )

    dev = report.get("dev")
    _require(isinstance(dev, dict), f"dev record missing: {report_path}")
    _require(
        dev.get("checkpoint_read_only") is True, f"dev checkpoint was not read-only: {report_path}"
    )
    _require(dev.get("native_mode_only") is True, f"dev was not native-only: {report_path}")
    paired = report.get("paired_diagnostic")
    _require(isinstance(paired, dict), f"paired diagnostic missing: {report_path}")
    _require(paired.get("status") == "completed", f"paired diagnostic failed: {report_path}")
    _require(
        paired.get("native_mode_only") is True,
        f"paired diagnostic was not native-only: {report_path}",
    )
    _require(paired.get("external_provider") is False, f"external provider was used: {report_path}")

    checkpoint_path, checkpoint = _load_checkpoint(report, report_path)
    _require(
        checkpoint.get("global_step") == training.get("global_step"),
        f"checkpoint global step mismatch: {checkpoint_path}",
    )
    _require(
        checkpoint.get("episode_count") == training.get("episodes"),
        f"checkpoint episode count mismatch: {checkpoint_path}",
    )
    model_payload = checkpoint.get("model")
    _require(isinstance(model_payload, dict), f"model payload missing: {checkpoint_path}")
    if arm == "control":
        _require(
            isinstance(model_payload.get("response_phase_readout"), dict),
            f"control phase payload missing: {checkpoint_path}",
        )
        _require(
            "response_plan_readout" not in model_payload,
            f"control checkpoint contains a plan: {checkpoint_path}",
        )
    else:
        plan_payload = model_payload.get("response_plan_readout")
        _require(
            isinstance(plan_payload, dict), f"treatment plan payload missing: {checkpoint_path}"
        )
        _require(
            plan_payload.get("variant") == "factorized_v1",
            f"checkpoint plan variant mismatch: {checkpoint_path}",
        )
        _require(
            plan_payload.get("plan_slots") == EXPECTED_PLAN_SLOTS,
            f"checkpoint plan slot mismatch: {checkpoint_path}",
        )
        _require(
            plan_payload.get("phase_stride") == EXPECTED_PHASE_STRIDE,
            f"checkpoint phase stride mismatch: {checkpoint_path}",
        )

    target_lineage = report.get("response_plan_target")
    _require(isinstance(target_lineage, dict), f"target lineage missing: {report_path}")
    _require(
        target_lineage.get("geometry") == EXPECTED_GEOMETRIES[arm],
        f"target lineage geometry mismatch: {report_path}",
    )
    if arm == "control":
        _require(
            target_lineage.get("encoder_digest") is None,
            f"control carries an encoder: {report_path}",
        )
    else:
        _require(target_lineage.get("encoder_digest"), f"treatment encoder missing: {report_path}")
        _require(
            target_lineage.get("encoder_parent_checkpoint_digest"),
            f"treatment encoder parent missing: {report_path}",
        )
        _require(
            target_lineage.get("encoder_corpus_digest") == EXPECTED_DATASET_DIGEST,
            f"treatment encoder corpus mismatch: {report_path}",
        )
        _require(
            target_lineage.get("train_target_map_digest"),
            f"treatment target map missing: {report_path}",
        )
        target_payload = checkpoint.get("response_plan_target_encoder")
        target_map = checkpoint.get("response_plan_targets")
        _require(
            isinstance(target_payload, dict),
            f"checkpoint target encoder missing: {checkpoint_path}",
        )
        _require(
            isinstance(target_map, dict), f"checkpoint train targets missing: {checkpoint_path}"
        )
        _require(
            content_digest(target_payload) == target_lineage["encoder_digest"],
            f"target encoder digest mismatch: {report_path}",
        )
        _require(
            _target_map_digest(target_map) == target_lineage["train_target_map_digest"],
            f"target map digest mismatch: {report_path}",
        )

    model_seed = report.get("model_seed")
    code_revision = report.get("code_revision")
    _require(int(model_seed) == seed, f"model seed metadata mismatch: {report_path}")
    _require(
        isinstance(code_revision, str) and code_revision, f"code revision missing: {report_path}"
    )

    summary = {
        "seed": seed,
        "arm": arm,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "model_seed": seed,
        "code_revision": code_revision,
        "dataset_digest": dataset["digest"],
        "target_geometry": target_lineage,
        "capacity": capacity,
        "training": {
            key: training[key]
            for key in (
                "epochs",
                "episodes",
                "global_step",
                "response_accuracy",
                "response_mean_surprise",
            )
        },
        "dev": {
            key: dev[key]
            for key in (
                "episodes",
                "teacher_forced_accuracy",
                "teacher_forced_mean_surprise",
                "exact_response_rate",
                "utf8_valid_rate",
                "no_replacement_rate",
                "response_boundary_rate",
                "sequence_valid_rate",
                "sequence_criterion_pass_rate",
                "required_term_coverage",
                "semantic_criteria_pass_rate",
                "unique_generated_texts",
                "generated_text_collision_rate",
                "checkpoint_read_only",
                "native_mode_only",
            )
        },
        "paired": {
            key: paired[key]
            for key in (
                "episodes",
                "teacher_forced_mean_surprise_delta",
                "exact_response_delta",
                "baseline_prompt_sensitivity_rate",
                "child_prompt_sensitivity_rate",
                "child_output_changed_after_update_rate",
                "checkpoint_read_only",
                "native_mode_only",
                "external_provider",
            )
        },
    }
    return summary, checkpoint_path, checkpoint


def _validate_ablation(
    report: dict[str, Any],
    report_path: Path,
    *,
    expected_seed: int,
    treatment: dict[str, Any],
    treatment_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    _require(report.get("status") == "completed", f"ablation did not complete: {report_path}")
    _require(
        report.get("format") == "taiji-r2-h3-7-response-plan-ablation-v1",
        f"wrong ablation format: {report_path}",
    )
    _require(
        report.get("pre_registration") == EXPECTED_PREREGISTRATION,
        f"wrong ablation preregistration: {report_path}",
    )
    _require(report.get("split") == "dev", f"ablation split is not dev: {report_path}")
    _require(
        report.get("dataset_digest") == EXPECTED_DATASET_DIGEST,
        f"ablation dataset mismatch: {report_path}",
    )
    _require(
        report.get("checkpoint_read_only") is True, f"ablation was not read-only: {report_path}"
    )
    _require(
        report.get("checkpoint_digest") == treatment_checkpoint.get("checkpoint_digest"),
        f"ablation source digest mismatch: {report_path}",
    )
    target_lineage = report.get("target_lineage")
    treatment_lineage = treatment["target_geometry"]
    _require(isinstance(target_lineage, dict), f"ablation target lineage missing: {report_path}")
    _require(
        isinstance(treatment_lineage, dict), f"treatment target lineage missing: {report_path}"
    )
    # The ablation report may carry the H3.7 architecture annotations
    # (variant/slots/stride) in addition to the runner's canonical target
    # lineage.  Compare the bound target identity field-by-field and validate
    # those extra annotations against the frozen contract instead of requiring
    # byte-for-byte equality between two report schemas.
    for key in (
        "geometry",
        "encoder_digest",
        "encoder_parent_checkpoint_digest",
        "encoder_corpus_digest",
        "train_target_map_digest",
        "fit_episode_ids",
    ):
        _require(
            target_lineage.get(key) == treatment_lineage.get(key),
            f"ablation target lineage {key} mismatch: {report_path}",
        )
    _require(
        target_lineage.get("geometry") == EXPECTED_GEOMETRIES["treatment"],
        f"ablation target geometry mismatch: {report_path}",
    )
    _require(
        target_lineage.get("variant") == "factorized_v1",
        f"ablation target variant mismatch: {report_path}",
    )
    _require(
        target_lineage.get("plan_slots") == EXPECTED_PLAN_SLOTS,
        f"ablation target slot mismatch: {report_path}",
    )
    _require(
        target_lineage.get("phase_stride") == EXPECTED_PHASE_STRIDE,
        f"ablation target stride mismatch: {report_path}",
    )
    _require(
        report.get("target_available_count") == 0,
        f"H3.7 dev ablation read a dev target: {report_path}",
    )
    _require(report.get("episodes") == 8, f"ablation episode count mismatch: {report_path}")
    for name in ("normal", "plan_bridge_ablated", "slot_credit_ablated"):
        arm = report.get(name)
        _require(isinstance(arm, dict), f"ablation arm missing: {report_path}")
        _require(
            arm.get("checkpoint_read_only") is True,
            f"ablation {name} was not read-only: {report_path}",
        )
        _require(
            arm.get("native_mode_only") is True,
            f"ablation {name} was not native-only: {report_path}",
        )
    normal = report["normal"]
    treatment_dev = treatment["dev"]
    for key in ("sequence_criterion_pass_rate", "required_term_coverage", "exact_response_rate"):
        _require(
            abs(float(normal[key]) - float(treatment_dev[key])) <= 1e-12,
            f"ablation normal does not match treatment dev for {key}: {report_path}",
        )
    return {
        "seed": expected_seed,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "checkpoint_digest": report["checkpoint_digest"],
        "checkpoint_read_only": True,
        "normal": {
            key: report["normal"][key]
            for key in (
                "sequence_criterion_pass_rate",
                "exact_response_rate",
                "required_term_coverage",
                "generated_text_collision_rate",
            )
        },
        "bridge_ablated": {
            key: report["plan_bridge_ablated"][key]
            for key in (
                "sequence_criterion_pass_rate",
                "exact_response_rate",
                "required_term_coverage",
                "generated_text_collision_rate",
            )
        },
        "slot_credit_ablated": {
            key: report["slot_credit_ablated"][key]
            for key in (
                "sequence_criterion_pass_rate",
                "exact_response_rate",
                "required_term_coverage",
                "generated_text_collision_rate",
            )
        },
    }


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _metric(rows: list[dict[str, Any]], arm: str, key: str) -> list[float]:
    return [float(row[arm]["dev"][key]) for row in rows]


def _paired_metric(rows: list[dict[str, Any]], arm: str, key: str) -> list[float]:
    return [float(row[arm]["paired"][key]) for row in rows]


def _direction_consistent(deltas: Iterable[float]) -> bool:
    values = tuple(float(value) for value in deltas)
    return all(value >= -1e-12 for value in values) and any(value > 1e-12 for value in values)


def _ablation_removes_core_gain(
    rows: list[dict[str, Any]],
    name: str,
) -> bool:
    treatment_gain_observed = False
    core_gain_removed = False
    for row in rows:
        control = row["control"]["dev"]
        treatment = row["treatment"]["dev"]
        ablated = row["ablation"][name]
        for key in ("sequence_criterion_pass_rate", "required_term_coverage"):
            control_value = float(control[key])
            treatment_value = float(treatment[key])
            ablated_value = float(ablated[key])
            if ablated_value > treatment_value + 1e-12:
                return False
            if treatment_value > control_value + 1e-12:
                treatment_gain_observed = True
                # A causal ablation must remove the observed treatment gain,
                # not merely make an already-tied treatment score smaller.
                if ablated_value <= control_value + 1e-12:
                    core_gain_removed = True
    return treatment_gain_observed and core_gain_removed


def main() -> int:
    args = _parse_args()
    control_paths = tuple(path.resolve() for path in args.control_reports)
    treatment_paths = tuple(path.resolve() for path in args.treatment_reports)
    ablation_paths = tuple(path.resolve() for path in args.ablation_reports)
    _require(len(set(control_paths)) == 3, "control reports must be distinct")
    _require(len(set(treatment_paths)) == 3, "treatment reports must be distinct")
    _require(len(set(ablation_paths)) == 3, "ablation reports must be distinct")

    control_by_seed: dict[int, dict[str, Any]] = {}
    treatment_by_seed: dict[int, dict[str, Any]] = {}
    treatment_checkpoints: dict[int, dict[str, Any]] = {}
    for expected_seed, path in zip(EXPECTED_SEEDS, control_paths, strict=True):
        summary, _, _ = _validate_common_run(
            _load_json(path), path, arm="control", expected_seed=expected_seed
        )
        control_by_seed[expected_seed] = summary
    for expected_seed, path in zip(EXPECTED_SEEDS, treatment_paths, strict=True):
        summary, _, checkpoint = _validate_common_run(
            _load_json(path), path, arm="treatment", expected_seed=expected_seed
        )
        treatment_by_seed[expected_seed] = summary
        treatment_checkpoints[expected_seed] = checkpoint
    _require(set(control_by_seed) == set(EXPECTED_SEEDS), "control seed set mismatch")
    _require(set(treatment_by_seed) == set(EXPECTED_SEEDS), "treatment seed set mismatch")

    ablation_by_seed: dict[int, dict[str, Any]] = {}
    for expected_seed, path in zip(EXPECTED_SEEDS, ablation_paths, strict=True):
        ablation_by_seed[expected_seed] = _validate_ablation(
            _load_json(path),
            path,
            expected_seed=expected_seed,
            treatment=treatment_by_seed[expected_seed],
            treatment_checkpoint=treatment_checkpoints[expected_seed],
        )

    rows = []
    for seed in EXPECTED_SEEDS:
        control = control_by_seed[seed]
        treatment = treatment_by_seed[seed]
        _require(
            control["code_revision"] == treatment["code_revision"],
            f"code revision mismatch at seed {seed}",
        )
        _require(
            control["dataset_digest"] == treatment["dataset_digest"] == EXPECTED_DATASET_DIGEST,
            f"dataset mismatch at seed {seed}",
        )
        _require(
            control["training"]["global_step"] == treatment["training"]["global_step"],
            f"update mismatch at seed {seed}",
        )
        rows.append(
            {
                "seed": seed,
                "control": control,
                "treatment": treatment,
                "ablation": ablation_by_seed[seed],
            }
        )

    control_sequence = _metric(rows, "control", "sequence_criterion_pass_rate")
    treatment_sequence = _metric(rows, "treatment", "sequence_criterion_pass_rate")
    control_terms = _metric(rows, "control", "required_term_coverage")
    treatment_terms = _metric(rows, "treatment", "required_term_coverage")
    control_exact = _metric(rows, "control", "exact_response_rate")
    treatment_exact = _metric(rows, "treatment", "exact_response_rate")
    control_utf8 = _metric(rows, "control", "utf8_valid_rate")
    treatment_utf8 = _metric(rows, "treatment", "utf8_valid_rate")
    control_boundary = _metric(rows, "control", "response_boundary_rate")
    treatment_boundary = _metric(rows, "treatment", "response_boundary_rate")
    control_sensitivity = _paired_metric(rows, "control", "child_prompt_sensitivity_rate")
    treatment_sensitivity = _paired_metric(rows, "treatment", "child_prompt_sensitivity_rate")
    sequence_deltas = [
        treatment - control
        for treatment, control in zip(treatment_sequence, control_sequence, strict=True)
    ]
    term_deltas = [
        treatment - control
        for treatment, control in zip(treatment_terms, control_terms, strict=True)
    ]
    exact_deltas = [
        treatment - control
        for treatment, control in zip(treatment_exact, control_exact, strict=True)
    ]

    machine_gates = {
        "all_runs_completed": True,
        "all_runs_within_budget": all(
            row[arm]["capacity"]["within_target"]
            for row in rows
            for arm in ("control", "treatment")
        ),
        "all_preflights_passed": all(
            row[arm]["dev"]["checkpoint_read_only"] and row[arm]["dev"]["native_mode_only"]
            for row in rows
            for arm in ("control", "treatment")
        ),
        "all_final_reads_deferred": True,
        "all_ablations_read_only": all(row["ablation"]["checkpoint_read_only"] for row in rows),
        "native_only": True,
        "runtime_oracle_detected": False,
        "protected_parent_failures": False,
        "target_lineage_failures": False,
    }
    analytical_gates = {
        "sequence_direction_consistent": _direction_consistent(sequence_deltas),
        "required_term_direction_consistent": _direction_consistent(term_deltas),
        "non_proxy_content_direction_consistent": _direction_consistent(sequence_deltas)
        or _direction_consistent(term_deltas),
        "utf8_and_boundary_not_degraded": all(
            treatment >= control
            for treatment, control in zip(treatment_utf8, control_utf8, strict=True)
        )
        and all(
            treatment >= control
            for treatment, control in zip(treatment_boundary, control_boundary, strict=True)
        ),
        "paired_sensitivity_not_degraded": all(
            treatment >= control
            for treatment, control in zip(treatment_sensitivity, control_sensitivity, strict=True)
        ),
        "bridge_ablation_removes_core_gain": _ablation_removes_core_gain(rows, "bridge_ablated"),
        "slot_credit_ablation_removes_core_gain": _ablation_removes_core_gain(
            rows, "slot_credit_ablated"
        ),
    }
    required_keys = (
        "all_runs_completed",
        "all_runs_within_budget",
        "all_preflights_passed",
        "all_final_reads_deferred",
        "all_ablations_read_only",
        "native_only",
        "non_proxy_content_direction_consistent",
        "utf8_and_boundary_not_degraded",
        "paired_sensitivity_not_degraded",
        "bridge_ablation_removes_core_gain",
        "slot_credit_ablation_removes_core_gain",
    )
    gates = {**machine_gates, **analytical_gates}
    development_gate_passed = all(gates[key] for key in required_keys)

    stop_reasons = []
    if not analytical_gates["non_proxy_content_direction_consistent"]:
        stop_reasons.append("no_three_seed_non_proxy_content_direction")
    if not analytical_gates["utf8_and_boundary_not_degraded"]:
        stop_reasons.append("utf8_or_boundary_degraded")
    if not analytical_gates["paired_sensitivity_not_degraded"]:
        stop_reasons.append("paired_prompt_sensitivity_degraded")
    if not analytical_gates["bridge_ablation_removes_core_gain"]:
        stop_reasons.append("bridge_ablation_does_not_remove_core_gain")
    if not analytical_gates["slot_credit_ablation_removes_core_gain"]:
        stop_reasons.append("slot_credit_ablation_does_not_remove_core_gain")

    report = {
        "format": "taiji-r2-h3-7-matched-dev-result-v1",
        "status": "completed" if development_gate_passed else "stopped_before_final",
        "pre_registration": EXPECTED_PREREGISTRATION,
        "dataset_digest": EXPECTED_DATASET_DIGEST,
        "evaluation_code": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
        "protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "max_episodes_per_epoch": 12,
            "episodes_per_run": EXPECTED_EPISODES,
            "global_step": EXPECTED_GLOBAL_STEP,
            "parameter_budget": EXPECTED_PARAMETER_BUDGET,
            "final_deferred_for_all_runs": True,
            "final_capability_read_performed": False,
            "control_geometry": EXPECTED_GEOMETRIES["control"],
            "treatment_geometry": EXPECTED_GEOMETRIES["treatment"],
            "plan_width": EXPECTED_PLAN_WIDTH,
            "plan_slots": EXPECTED_PLAN_SLOTS,
            "phase_stride": EXPECTED_PHASE_STRIDE,
        },
        "source_artifacts": [
            {
                "seed": seed,
                "control": control_by_seed[seed],
                "treatment": treatment_by_seed[seed],
                "causal_ablation": ablation_by_seed[seed],
            }
            for seed in EXPECTED_SEEDS
        ],
        "aggregate": {
            "control_dev_sequence_rate_by_seed": control_sequence,
            "treatment_dev_sequence_rate_by_seed": treatment_sequence,
            "treatment_minus_control_dev_sequence_rate_by_seed": sequence_deltas,
            "control_dev_required_term_coverage_by_seed": control_terms,
            "treatment_dev_required_term_coverage_by_seed": treatment_terms,
            "treatment_minus_control_dev_required_term_coverage_by_seed": term_deltas,
            "control_dev_exact_rate_by_seed": control_exact,
            "treatment_dev_exact_rate_by_seed": treatment_exact,
            "treatment_minus_control_dev_exact_rate_by_seed": exact_deltas,
            "control_dev_utf8_rate_by_seed": control_utf8,
            "treatment_dev_utf8_rate_by_seed": treatment_utf8,
            "control_dev_boundary_rate_by_seed": control_boundary,
            "treatment_dev_boundary_rate_by_seed": treatment_boundary,
            "control_paired_sensitivity_by_seed": control_sensitivity,
            "treatment_paired_sensitivity_by_seed": treatment_sensitivity,
            "bridge_normal_sequence_rate_by_seed": [
                row["ablation"]["normal"]["sequence_criterion_pass_rate"] for row in rows
            ],
            "bridge_ablated_sequence_rate_by_seed": [
                row["ablation"]["bridge_ablated"]["sequence_criterion_pass_rate"] for row in rows
            ],
            "slot_credit_ablated_sequence_rate_by_seed": [
                row["ablation"]["slot_credit_ablated"]["sequence_criterion_pass_rate"]
                for row in rows
            ],
            "bridge_normal_required_term_coverage_by_seed": [
                row["ablation"]["normal"]["required_term_coverage"] for row in rows
            ],
            "bridge_ablated_required_term_coverage_by_seed": [
                row["ablation"]["bridge_ablated"]["required_term_coverage"] for row in rows
            ],
            "slot_credit_ablated_required_term_coverage_by_seed": [
                row["ablation"]["slot_credit_ablated"]["required_term_coverage"] for row in rows
            ],
            "mean_control_dev_sequence_rate": _mean(control_sequence),
            "mean_treatment_dev_sequence_rate": _mean(treatment_sequence),
            "mean_treatment_minus_control_dev_sequence_rate": _mean(sequence_deltas),
            "mean_treatment_minus_control_dev_required_term_coverage": _mean(term_deltas),
        },
        "gates": gates,
        "decision": {
            "development_gate_passed": development_gate_passed,
            "stop_reason": stop_reasons,
            "no_more_epochs": not development_gate_passed,
            "do_not_read_final": not development_gate_passed,
            "final_access_allowed": development_gate_passed,
            "next": (
                "H3.7 dev gates passed; freeze this matrix and perform the one-time final capability read"
                if development_gate_passed
                else "H3.7 negative or incomplete development result; do not read final or add epochs; return to the frozen attribution order"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_gate_passed": development_gate_passed,
                "mean_control_dev_sequence_rate": report["aggregate"][
                    "mean_control_dev_sequence_rate"
                ],
                "mean_treatment_dev_sequence_rate": report["aggregate"][
                    "mean_treatment_dev_sequence_rate"
                ],
                "treatment_minus_control_dev_sequence_rate_by_seed": sequence_deltas,
                "bridge_ablation_removes_core_gain": analytical_gates[
                    "bridge_ablation_removes_core_gain"
                ],
                "slot_credit_ablation_removes_core_gain": analytical_gates[
                    "slot_credit_ablation_removes_core_gain"
                ],
                "final_access_allowed": report["decision"]["final_access_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
