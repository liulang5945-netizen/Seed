"""Freeze and judge the preregistered H3.6-B matched dev matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji.internalization import content_digest

EXPECTED_DATASET_DIGEST = "0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691"
EXPECTED_PREREGISTRATION = "plans/reference/M5_R2_H3_6B_MATCHED_RUN_PREREGISTRATION_20260916.md"
EXPECTED_SEEDS = (20260916, 20260917, 20260918)
EXPECTED_EPOCHS = 10
EXPECTED_EPISODES = 120
EXPECTED_PARAMETER_BUDGET = 300_000
EXPECTED_GEOMETRIES = {
    "control": "signed_hash_span",
    "treatment": "h3_6_whitened_native_compositional",
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
    _require(isinstance(payload, dict), f"checkpoint must be a JSON-like mapping: {checkpoint_path}")
    _require(
        isinstance(payload.get("checkpoint_digest"), str),
        f"checkpoint digest is missing: {checkpoint_path}",
    )
    return checkpoint_path, payload


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
    _require(config.get("response_plan_readout") is True, f"plan readout missing: {report_path}")
    _require(config.get("response_plan_width") == 32, f"plan width mismatch: {report_path}")

    capacity = report.get("capacity")
    _require(isinstance(capacity, dict), f"capacity record missing: {report_path}")
    _require(
        capacity.get("target_active_parameters") == EXPECTED_PARAMETER_BUDGET,
        f"parameter budget mismatch: {report_path}",
    )
    _require(capacity.get("effective_active_parameters") == 276610, f"effective budget mismatch: {report_path}")
    _require(capacity.get("within_target") is True, f"run exceeded parameter budget: {report_path}")

    preflight = report.get("preflight")
    _require(isinstance(preflight, dict), f"preflight missing: {report_path}")
    for key in ("zero_step_roundtrip", "caller_model_unchanged", "atomic_save"):
        _require(preflight.get(key) is True, f"preflight {key} failed: {report_path}")
    _require(preflight.get("status") == "passed", f"preflight did not pass: {report_path}")
    update = preflight.get("one_response_update")
    _require(isinstance(update, dict) and update.get("status") == "completed", f"child update preflight failed: {report_path}")

    training = report.get("training")
    _require(isinstance(training, dict), f"training record missing: {report_path}")
    _require(training.get("status") == "completed", f"training did not complete: {report_path}")
    _require(training.get("epochs") == EXPECTED_EPOCHS, f"epoch count mismatch: {report_path}")
    _require(training.get("episodes") == EXPECTED_EPISODES, f"episode count mismatch: {report_path}")
    _require(training.get("global_step") == 5370, f"update count mismatch: {report_path}")

    dev = report.get("dev")
    _require(isinstance(dev, dict), f"dev record missing: {report_path}")
    _require(dev.get("checkpoint_read_only") is True, f"dev checkpoint was not read-only: {report_path}")
    _require(dev.get("native_mode_only") is True, f"dev was not native-only: {report_path}")
    paired = report.get("paired_diagnostic")
    _require(isinstance(paired, dict), f"paired diagnostic missing: {report_path}")
    _require(paired.get("status") == "completed", f"paired diagnostic failed: {report_path}")
    _require(paired.get("native_mode_only") is True, f"paired diagnostic was not native-only: {report_path}")
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
    report_geometry = report["response_plan_target"]
    _require(isinstance(report_geometry, dict), f"target lineage missing: {report_path}")
    _require(
        report_geometry.get("geometry") == EXPECTED_GEOMETRIES[arm],
        f"target lineage geometry mismatch: {report_path}",
    )
    if arm == "control":
        _require(report_geometry.get("encoder_digest") is None, f"control carries an encoder: {report_path}")
    else:
        _require(isinstance(report_geometry.get("encoder_digest"), str), f"treatment encoder missing: {report_path}")
        _require(isinstance(report_geometry.get("encoder_parent_checkpoint_digest"), str), f"treatment encoder parent missing: {report_path}")
        _require(report_geometry.get("encoder_corpus_digest") == EXPECTED_DATASET_DIGEST, f"treatment encoder corpus mismatch: {report_path}")
        _require(isinstance(report_geometry.get("train_target_map_digest"), str), f"treatment target map missing: {report_path}")
        target_payload = checkpoint.get("response_plan_target_encoder")
        _require(isinstance(target_payload, dict), f"checkpoint target encoder missing: {checkpoint_path}")
        _require(
            content_digest(target_payload) == report_geometry.get("encoder_digest"),
            f"target encoder digest mismatch: {report_path}",
        )

    model_seed = report.get("model_seed")
    code_revision = report.get("code_revision")
    metadata_source = "report"
    if model_seed is None:
        model_seed = seed
        metadata_source = "checkpoint_and_report_filename"
    if code_revision is None:
        code_revision = checkpoint.get("code_revision")
        metadata_source = "checkpoint_and_report_filename"
    _require(int(model_seed) == seed, f"model seed metadata mismatch: {report_path}")
    _require(isinstance(code_revision, str) and code_revision, f"code revision missing: {report_path}")

    summary = {
        "seed": seed,
        "arm": arm,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "model_seed": int(model_seed),
        "code_revision": code_revision,
        "metadata_source": metadata_source,
        "dataset_digest": dataset["digest"],
        "target_geometry": report_geometry,
        "capacity": capacity,
        "training": {
            "epochs": training["epochs"],
            "episodes": training["episodes"],
            "global_step": training["global_step"],
            "response_accuracy": training["response_accuracy"],
            "response_mean_surprise": training["response_mean_surprise"],
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
    _require(report.get("format") == "taiji-r2-h3-6b-response-plan-ablation-v1", f"wrong ablation format: {report_path}")
    _require(report.get("pre_registration") == EXPECTED_PREREGISTRATION, f"wrong ablation preregistration: {report_path}")
    _require(report.get("split") == "dev", f"ablation split is not dev: {report_path}")
    _require(report.get("dataset_digest") == EXPECTED_DATASET_DIGEST, f"ablation dataset mismatch: {report_path}")
    _require(report.get("checkpoint_read_only") is True, f"ablation was not read-only: {report_path}")
    _require(report.get("checkpoint_digest") == treatment_checkpoint.get("checkpoint_digest"), f"ablation source digest mismatch: {report_path}")
    _require(report.get("target_lineage") == treatment["target_geometry"], f"ablation target lineage mismatch: {report_path}")
    _require(report.get("target_available_count") == 0, f"H3.6 dev ablation read a dev target: {report_path}")
    for name in ("normal", "plan_bridge_ablated"):
        arm = report.get(name)
        _require(isinstance(arm, dict), f"ablation arm missing: {report_path}")
        _require(arm.get("checkpoint_read_only") is True, f"ablation {name} was not read-only: {report_path}")
        _require(arm.get("native_mode_only") is True, f"ablation {name} was not native-only: {report_path}")
    _require(report.get("episodes") == 8, f"ablation episode count mismatch: {report_path}")
    return {
        "seed": expected_seed,
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "checkpoint_digest": report["checkpoint_digest"],
        "checkpoint_read_only": True,
        "normal": {
            key: report["normal"][key]
            for key in ("sequence_criterion_pass_rate", "exact_response_rate", "required_term_coverage", "generated_text_collision_rate")
        },
        "ablated": {
            key: report["plan_bridge_ablated"][key]
            for key in ("sequence_criterion_pass_rate", "exact_response_rate", "required_term_coverage", "generated_text_collision_rate")
        },
    }


def _mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    return sum(values) / len(values)


def _metric(rows: list[dict[str, Any]], arm: str, key: str) -> list[float]:
    return [float(row[arm]["dev"][key]) for row in rows]


def _paired_metric(rows: list[dict[str, Any]], arm: str, key: str) -> list[float]:
    return [float(row[arm]["paired"][key]) for row in rows]


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
    for expected_seed, path in zip(EXPECTED_SEEDS, control_paths):
        summary, _, _ = _validate_common_run(
            _load_json(path), path, arm="control", expected_seed=expected_seed
        )
        control_by_seed[expected_seed] = summary
    for expected_seed, path in zip(EXPECTED_SEEDS, treatment_paths):
        summary, _, checkpoint = _validate_common_run(
            _load_json(path), path, arm="treatment", expected_seed=expected_seed
        )
        treatment_by_seed[expected_seed] = summary
        treatment_checkpoints[expected_seed] = checkpoint
    _require(set(control_by_seed) == set(EXPECTED_SEEDS), "control seed set mismatch")
    _require(set(treatment_by_seed) == set(EXPECTED_SEEDS), "treatment seed set mismatch")

    ablation_by_seed: dict[int, dict[str, Any]] = {}
    for expected_seed, path in zip(EXPECTED_SEEDS, ablation_paths):
        report = _load_json(path)
        ablation_by_seed[expected_seed] = _validate_ablation(
            report,
            path,
            expected_seed=expected_seed,
            treatment=treatment_by_seed[expected_seed],
            treatment_checkpoint=treatment_checkpoints[expected_seed],
        )

    rows = []
    for seed in EXPECTED_SEEDS:
        control = control_by_seed[seed]
        treatment = treatment_by_seed[seed]
        ablation = ablation_by_seed[seed]
        _require(control["code_revision"] == treatment["code_revision"], f"code revision mismatch at seed {seed}")
        _require(control["dataset_digest"] == treatment["dataset_digest"] == EXPECTED_DATASET_DIGEST, f"dataset mismatch at seed {seed}")
        _require(control["training"]["global_step"] == treatment["training"]["global_step"], f"update mismatch at seed {seed}")
        rows.append({"seed": seed, "control": control, "treatment": treatment, "ablation": ablation})

    control_sequence = _metric(rows, "control", "sequence_criterion_pass_rate")
    treatment_sequence = _metric(rows, "treatment", "sequence_criterion_pass_rate")
    control_exact = _metric(rows, "control", "exact_response_rate")
    treatment_exact = _metric(rows, "treatment", "exact_response_rate")
    control_terms = _metric(rows, "control", "required_term_coverage")
    treatment_terms = _metric(rows, "treatment", "required_term_coverage")
    control_surprise = _metric(rows, "control", "teacher_forced_mean_surprise")
    treatment_surprise = _metric(rows, "treatment", "teacher_forced_mean_surprise")
    control_collision = _metric(rows, "control", "generated_text_collision_rate")
    treatment_collision = _metric(rows, "treatment", "generated_text_collision_rate")
    control_sensitivity = _paired_metric(rows, "control", "child_prompt_sensitivity_rate")
    treatment_sensitivity = _paired_metric(rows, "treatment", "child_prompt_sensitivity_rate")
    ablated_sequence = [float(row["ablation"]["ablated"]["sequence_criterion_pass_rate"]) for row in rows]
    ablated_collision = [float(row["ablation"]["ablated"]["generated_text_collision_rate"]) for row in rows]
    normal_ablation_sequence = [float(row["ablation"]["normal"]["sequence_criterion_pass_rate"]) for row in rows]

    sequence_deltas = [treatment - control for treatment, control in zip(treatment_sequence, control_sequence)]
    exact_deltas = [treatment - control for treatment, control in zip(treatment_exact, control_exact)]
    term_deltas = [treatment - control for treatment, control in zip(treatment_terms, control_terms)]
    surprise_deltas = [treatment - control for treatment, control in zip(treatment_surprise, control_surprise)]
    ablation_sequence_deltas = [ablated - normal for ablated, normal in zip(ablated_sequence, normal_ablation_sequence)]
    ablation_collision_deltas = [
        float(row["ablation"]["ablated"]["generated_text_collision_rate"])
        - float(row["ablation"]["normal"]["generated_text_collision_rate"])
        for row in rows
    ]

    machine_gates = {
        "all_runs_completed": True,
        "all_runs_within_budget": all(row[arm]["capacity"]["within_target"] for row in rows for arm in ("control", "treatment")),
        "all_preflights_passed": all(
            row[arm]["dev"]["checkpoint_read_only"] and row[arm]["dev"]["native_mode_only"]
            for row in rows
            for arm in ("control", "treatment")
        ),
        "all_final_reads_deferred": True,
        "all_bridge_ablations_read_only": all(row["ablation"]["checkpoint_read_only"] for row in rows),
        "runtime_oracle_detected": False,
        "protected_parent_failures": False,
        "target_lineage_failures": False,
    }
    analytical_gates = {
        "three_seed_dev_sequence_direction_consistent": all(delta >= 0.0 for delta in sequence_deltas)
        and any(delta > 0.0 for delta in sequence_deltas),
        "non_proxy_sequence_improvement": _mean(treatment_sequence) > _mean(control_sequence),
        "non_proxy_exact_improvement": _mean(treatment_exact) > _mean(control_exact),
        "non_proxy_required_term_improvement": _mean(treatment_terms) > _mean(control_terms),
        "paired_sensitivity_not_degraded": all(
            treatment >= control
            for treatment, control in zip(treatment_sensitivity, control_sensitivity)
        ),
        "bridge_effect_present": any(
            delta != 0.0
            for delta in ablation_sequence_deltas + ablation_collision_deltas
        ),
        "bridge_ablation_removes_core_treatment_gain": all(
            ablated <= normal
            for ablated, normal in zip(ablated_sequence, treatment_sequence)
        )
        and any(
            ablated < normal
            for ablated, normal in zip(ablated_sequence, treatment_sequence)
        ),
    }
    required_gates = {
        **machine_gates,
        **analytical_gates,
    }
    development_gate_passed = all(
        required_gates[key]
        for key in (
            "all_runs_completed",
            "all_runs_within_budget",
            "all_preflights_passed",
            "all_final_reads_deferred",
            "all_bridge_ablations_read_only",
            "three_seed_dev_sequence_direction_consistent",
            "non_proxy_sequence_improvement",
            "paired_sensitivity_not_degraded",
            "bridge_ablation_removes_core_treatment_gain",
        )
    )

    stop_reasons = []
    if not analytical_gates["three_seed_dev_sequence_direction_consistent"]:
        stop_reasons.append("seed_direction_inconsistent")
    if not analytical_gates["non_proxy_sequence_improvement"]:
        stop_reasons.append("no_non_proxy_sequence_improvement")
    if not analytical_gates["bridge_ablation_removes_core_treatment_gain"]:
        stop_reasons.append("bridge_does_not_remove_a_treatment_sequence_gain")

    report = {
        "format": "taiji-r2-h3-6b-matched-dev-result-v1",
        "status": "completed" if development_gate_passed else "stopped_before_final",
        "pre_registration": EXPECTED_PREREGISTRATION,
        "dataset_digest": EXPECTED_DATASET_DIGEST,
        "protocol": {
            "seeds": list(EXPECTED_SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "max_episodes_per_epoch": 12,
            "episodes_per_run": EXPECTED_EPISODES,
            "parameter_budget": EXPECTED_PARAMETER_BUDGET,
            "final_deferred_for_all_runs": True,
            "final_capability_read_performed": False,
            "control_geometry": EXPECTED_GEOMETRIES["control"],
            "treatment_geometry": EXPECTED_GEOMETRIES["treatment"],
        },
        "source_artifacts": [
            {
                "seed": seed,
                "control": control_by_seed[seed],
                "treatment": treatment_by_seed[seed],
                "bridge_ablation": ablation_by_seed[seed],
            }
            for seed in EXPECTED_SEEDS
        ],
        "aggregate": {
            "control_effective_parameters": control_by_seed[EXPECTED_SEEDS[0]]["capacity"]["effective_active_parameters"],
            "treatment_effective_parameters": treatment_by_seed[EXPECTED_SEEDS[0]]["capacity"]["effective_active_parameters"],
            "control_dev_sequence_rate_by_seed": control_sequence,
            "treatment_dev_sequence_rate_by_seed": treatment_sequence,
            "treatment_minus_control_dev_sequence_rate_by_seed": sequence_deltas,
            "control_dev_exact_rate_by_seed": control_exact,
            "treatment_dev_exact_rate_by_seed": treatment_exact,
            "treatment_minus_control_dev_exact_rate_by_seed": exact_deltas,
            "control_dev_required_term_coverage_by_seed": control_terms,
            "treatment_dev_required_term_coverage_by_seed": treatment_terms,
            "treatment_minus_control_dev_required_term_coverage_by_seed": term_deltas,
            "control_dev_surprise_by_seed": control_surprise,
            "treatment_dev_surprise_by_seed": treatment_surprise,
            "treatment_minus_control_dev_surprise_by_seed": surprise_deltas,
            "control_dev_collision_rate_by_seed": control_collision,
            "treatment_dev_collision_rate_by_seed": treatment_collision,
            "control_paired_sensitivity_by_seed": control_sensitivity,
            "treatment_paired_sensitivity_by_seed": treatment_sensitivity,
            "bridge_normal_sequence_rate_by_seed": normal_ablation_sequence,
            "bridge_ablated_sequence_rate_by_seed": ablated_sequence,
            "bridge_ablated_minus_normal_sequence_rate_by_seed": ablation_sequence_deltas,
            "bridge_normal_collision_rate_by_seed": [
                float(row["ablation"]["normal"]["generated_text_collision_rate"]) for row in rows
            ],
            "bridge_ablated_collision_rate_by_seed": ablated_collision,
            "bridge_ablated_minus_normal_collision_rate_by_seed": ablation_collision_deltas,
            "mean_control_dev_sequence_rate": _mean(control_sequence),
            "mean_treatment_dev_sequence_rate": _mean(treatment_sequence),
            "mean_treatment_minus_control_dev_sequence_rate": _mean(sequence_deltas),
            "mean_treatment_minus_control_dev_surprise": _mean(surprise_deltas),
        },
        "gates": required_gates,
        "decision": {
            "development_gate_passed": development_gate_passed,
            "stop_reason": stop_reasons,
            "no_more_epochs": not development_gate_passed,
            "do_not_read_final": not development_gate_passed,
            "final_access_allowed": development_gate_passed,
            "next": (
                "H3.6-B mechanism review may proceed after the one-time final read"
                if development_gate_passed
                else "H3.6-B negative result; do not read final or add epochs; return to target, representation, or readout design"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_gate_passed": development_gate_passed,
                "mean_control_dev_sequence_rate": report["aggregate"]["mean_control_dev_sequence_rate"],
                "mean_treatment_dev_sequence_rate": report["aggregate"]["mean_treatment_dev_sequence_rate"],
                "treatment_minus_control_dev_sequence_rate_by_seed": sequence_deltas,
                "bridge_ablated_minus_normal_sequence_rate_by_seed": ablation_sequence_deltas,
                "final_access_allowed": report["decision"]["final_access_allowed"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
