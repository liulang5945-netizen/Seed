"""Run a bounded, read-only attribution audit for the stopped H3.7 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from taiji import LanguageAlignmentTrainer, LanguageEpisodeCorpus
from taiji.internalization import content_digest

EXPECTED_DATASET_DIGEST = "0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691"
EXPECTED_GEOMETRY = "h3_7_factorized_response_chunks"
EXPECTED_VARIANT = "factorized_v1"
EXPECTED_SLOTS = 4
EXPECTED_SLOT_WIDTH = 12
EXPECTED_PHASE_STRIDE = 16


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--aggregate-report", required=True, type=Path)
    parser.add_argument("--treatment-reports", nargs=3, required=True, type=Path)
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(dtype=torch.float32)
    right = right.to(dtype=torch.float32)
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _phase_probe(
    trainer: LanguageAlignmentTrainer,
    episode: Any,
) -> dict[str, Any]:
    trainer._prime(episode)
    plan = trainer.model.begin_response_plan().detach().cpu()
    encoder = trainer.response_plan_target_encoder
    if encoder is None:
        raise RuntimeError("H3.7 treatment is missing its target encoder")
    target = encoder.encode_response(str(episode.response)).detach().cpu()
    readout = trainer.model.response_plan_readout
    _require(readout.variant == EXPECTED_VARIANT, "unexpected H3.7 readout variant")
    _require(readout.plan_slots == EXPECTED_SLOTS, "unexpected H3.7 slot count")
    _require(readout.plan_slot_width == EXPECTED_SLOT_WIDTH, "unexpected H3.7 slot width")
    _require(readout.phase_stride == EXPECTED_PHASE_STRIDE, "unexpected H3.7 phase stride")

    plan_slots = plan.reshape(EXPECTED_SLOTS, EXPECTED_SLOT_WIDTH)
    target_slots = target.reshape(EXPECTED_SLOTS, EXPECTED_SLOT_WIDTH)
    phase_probabilities: list[torch.Tensor] = []
    phase_digests: list[str] = []
    for phase in range(EXPECTED_SLOTS):
        if phase > 0:
            for _ in range(EXPECTED_PHASE_STRIDE):
                trainer.model.advance_response_plan_phase()
        probabilities = trainer.model.response_plan_probabilities().detach().cpu()
        phase_probabilities.append(probabilities)
        phase_digests.append(content_digest(probabilities.tolist()))

    phase_zero = phase_probabilities[0]
    phase_js = [
        0.0,
        *[
            trainer._distribution_js_divergence(phase_zero, current)
            for current in phase_probabilities[1:]
        ],
    ]
    phase_argmax_changed = [
        False,
        *[int(current.argmax()) != int(phase_zero.argmax()) for current in phase_probabilities[1:]],
    ]
    return {
        "episode_id": str(episode.episode_id),
        "plan_digest": content_digest(plan.tolist()),
        "target_digest": content_digest(target.tolist()),
        "plan_norm": float(torch.linalg.vector_norm(plan)),
        "target_norm": float(torch.linalg.vector_norm(target)),
        "plan_target_cosine": _cosine(plan, target),
        "plan_target_l2": float(torch.linalg.vector_norm(plan - target)),
        "slot_plan_norms": [float(torch.linalg.vector_norm(slot)) for slot in plan_slots],
        "slot_target_norms": [float(torch.linalg.vector_norm(slot)) for slot in target_slots],
        "slot_target_cosines": [
            _cosine(plan_slot, target_slot)
            for plan_slot, target_slot in zip(plan_slots, target_slots, strict=True)
        ],
        "phase_probability_digests": phase_digests,
        "phase_js_vs_phase_zero": phase_js,
        "phase_argmax_changed_vs_phase_zero": phase_argmax_changed,
    }


def _audit_treatment(report_path: Path, corpus: LanguageEpisodeCorpus) -> dict[str, Any]:
    report = _load_json(report_path)
    _require(report.get("status") == "completed", f"treatment run is not complete: {report_path}")
    _require(report.get("final_deferred") is True, f"final was not deferred: {report_path}")
    _require(
        report.get("dataset", {}).get("digest") == EXPECTED_DATASET_DIGEST,
        f"dataset mismatch: {report_path}",
    )
    config = report.get("config", {})
    _require(
        config.get("response_plan_variant") == EXPECTED_VARIANT, f"variant mismatch: {report_path}"
    )
    _require(config.get("response_plan_slots") == EXPECTED_SLOTS, f"slot mismatch: {report_path}")
    _require(
        config.get("response_plan_phase_stride") == EXPECTED_PHASE_STRIDE,
        f"stride mismatch: {report_path}",
    )
    checkpoint_value = report.get("checkpoint")
    _require(isinstance(checkpoint_value, str), f"checkpoint missing: {report_path}")
    checkpoint_path = _resolve_repo_path(checkpoint_value)
    _require(checkpoint_path.is_file(), f"checkpoint missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    _require(
        trainer.config.response_plan_target_geometry == EXPECTED_GEOMETRY,
        f"geometry mismatch: {report_path}",
    )
    _require(
        trainer.config.response_plan_variant == EXPECTED_VARIANT,
        f"restored variant mismatch: {report_path}",
    )
    before_digest = str(trainer.checkpoint()["checkpoint_digest"])
    bridge_norm = float(torch.linalg.matrix_norm(trainer.model.response_plan_readout.plan_bridge))
    records = [_phase_probe(trainer, episode) for episode in corpus.for_split("dev")]
    trainer.model.restore(payload["model"])
    after_digest = str(trainer.checkpoint()["checkpoint_digest"])
    _require(after_digest == before_digest, f"attribution audit mutated checkpoint: {report_path}")

    return {
        "seed": int(report["model_seed"]),
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_digest": payload["checkpoint_digest"],
        "code_revision": report["code_revision"],
        "effective_active_parameters": report["capacity"]["effective_active_parameters"],
        "bridge_norm": bridge_norm,
        "records": records,
        "checkpoint_read_only": True,
        "final_read": False,
        "native_mode_only": True,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    args = _parse_args()
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    _require(corpus.digest == EXPECTED_DATASET_DIGEST, "dataset digest does not match H3.7")
    aggregate = _load_json(args.aggregate_report)
    _require(
        aggregate.get("status") == "stopped_before_final",
        "aggregate is not the stopped H3.7 result",
    )
    _require(
        aggregate.get("decision", {}).get("final_access_allowed") is False,
        "aggregate permits final access",
    )

    treatments = [_audit_treatment(path.resolve(), corpus) for path in args.treatment_reports]
    all_records = [record for treatment in treatments for record in treatment["records"]]
    phase_js = [
        _mean([float(record["phase_js_vs_phase_zero"][phase]) for record in all_records])
        for phase in range(EXPECTED_SLOTS)
    ]
    phase_changes = [
        sum(bool(record["phase_argmax_changed_vs_phase_zero"][phase]) for record in all_records)
        / len(all_records)
        for phase in range(EXPECTED_SLOTS)
    ]
    slot_cosines = [
        _mean([float(record["slot_target_cosines"][slot]) for record in all_records])
        for slot in range(EXPECTED_SLOTS)
    ]
    report = {
        "format": "taiji-r2-h3-7-attribution-audit-v1",
        "status": "completed",
        "scope": {
            "dataset_digest": corpus.digest,
            "split": "dev",
            "final_read": False,
            "training_performed": False,
            "read_only": True,
            "native_mode_only": True,
            "order": [
                "target_sketch",
                "slot_phase_schedule",
                "bridge_credit",
                "renderer_readout",
                "native_prefix_representation",
                "capacity_data",
            ],
        },
        "source": {
            "aggregate_report": str(args.aggregate_report.resolve()),
            "aggregate_sha256": _sha256(args.aggregate_report.resolve()),
            "treatment_reports": treatments,
        },
        "target_sketch": {
            "mean_plan_target_cosine": _mean(
                [float(record["plan_target_cosine"]) for record in all_records]
            ),
            "mean_plan_target_l2": _mean(
                [float(record["plan_target_l2"]) for record in all_records]
            ),
            "mean_slot_target_cosine_by_slot": slot_cosines,
            "slot_target_norms_are_one": all(
                abs(float(norm) - 1.0) <= 1e-6
                for record in all_records
                for norm in record["slot_target_norms"]
            ),
        },
        "slot_phase_schedule": {
            "mean_probability_js_vs_phase_zero_by_phase": phase_js,
            "argmax_change_rate_vs_phase_zero_by_phase": phase_changes,
            "nonzero_phase_surface_rate": sum(value > 1e-12 for value in phase_js[1:])
            / (EXPECTED_SLOTS - 1),
        },
        "bridge_credit": {
            "bridge_norm_by_seed": [item["bridge_norm"] for item in treatments],
            "aggregate_gate": aggregate["gates"]["bridge_ablation_removes_core_gain"],
            "aggregate_stop_reason_present": "bridge_ablation_does_not_remove_core_gain"
            in aggregate["decision"]["stop_reason"],
        },
        "renderer_readout": {
            "dev_sequence_by_seed": aggregate["aggregate"]["treatment_dev_sequence_rate_by_seed"],
            "dev_required_term_coverage_by_seed": aggregate["aggregate"][
                "treatment_dev_required_term_coverage_by_seed"
            ],
            "dev_exact_by_seed": aggregate["aggregate"]["treatment_dev_exact_rate_by_seed"],
        },
        "native_prefix_representation": {
            "paired_child_prompt_sensitivity_by_seed": aggregate["aggregate"][
                "treatment_paired_sensitivity_by_seed"
            ],
            "paired_sensitivity_gate": aggregate["gates"]["paired_sensitivity_not_degraded"],
            "content_direction_gate": aggregate["gates"]["non_proxy_content_direction_consistent"],
        },
        "capacity_data": {
            "effective_active_parameters_by_seed": [
                item["effective_active_parameters"] for item in treatments
            ],
            "parameter_budget": aggregate["protocol"]["parameter_budget"],
            "sample_counts": corpus.sample_counts,
            "within_budget_gate": aggregate["gates"]["all_runs_within_budget"],
        },
        "checkpoint_read_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "mean_plan_target_cosine": report["target_sketch"]["mean_plan_target_cosine"],
                "mean_phase_js_vs_zero": report["slot_phase_schedule"][
                    "mean_probability_js_vs_phase_zero_by_phase"
                ],
                "argmax_change_rate_by_phase": report["slot_phase_schedule"][
                    "argmax_change_rate_vs_phase_zero_by_phase"
                ],
                "content_direction_gate": report["native_prefix_representation"][
                    "content_direction_gate"
                ],
                "checkpoint_read_only": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
