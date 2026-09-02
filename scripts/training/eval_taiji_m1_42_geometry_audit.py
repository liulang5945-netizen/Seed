"""Audit native event geometry after phase-A association formation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_m1_36_cue_curriculum import (  # noqa: E402
    _config,
    _curriculum,
)
from taiji import DelayedMemoryTask, Taiji  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FORMAT = "taiji-native-m1-42-geometry-audit-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_42_geometry_audit_20260902.json"
SEEDS = (11, 29, 47)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("geometry summary cannot be empty")
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _event_patterns(model: Taiji, episode: Any) -> dict[str, torch.Tensor | float | int]:
    model.reset_dynamics(episode_id=f"m1-42-train-{episode.memory_id}")
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        learn_motor=False,
        use_memory=False,
    )
    model.observe(
        episode.cue,
        learn=False,
        learn_motor=False,
        use_memory=False,
    )
    state = model.snapshot()
    context = model.fabric.cortical_context(state.regions).detach()
    memory = model.memory
    cue_pattern = memory._cue_pattern(context, state.memory.threshold)
    action_drive = memory._normalize_drive(
        memory.action_encoder.forward(memory._one_hot(episode.action))
    )
    outcome_drive = memory._normalize_drive(
        memory.outcome_encoder.forward(memory._one_hot(episode.outcome))
    )
    time_code = memory._time_code(2)
    episode_code = memory._episode_code(f"m0-b2-train-{episode.memory_id}")
    provenance_code = memory._provenance_code("experienced")
    time_drive = memory._normalize_drive(memory.time_encoder.forward(time_code))
    episode_drive = memory._normalize_drive(memory.episode_encoder.forward(episode_code))
    provenance_drive = memory._normalize_drive(
        memory.provenance_encoder.forward(provenance_code)
    )
    components = (
        action_drive,
        outcome_drive,
        memory.reward_code,
        time_drive,
        episode_drive,
        provenance_drive,
    )
    event_scale = model.config.memory_event_gain / len(components) ** 0.5
    event_drive = cue_pattern + event_scale * torch.stack(components).sum(dim=0)
    event_pattern, _ = memory._activate(event_drive, state.memory.threshold)
    completion = memory.association.forward(cue_pattern)
    context_for_readout = memory.readout_receptors.forward(event_pattern)
    cue_energy = cue_pattern.square().sum().clamp_min(1e-8)
    association_row_capture = (
        cue_pattern[memory.association.pre_index].square().sum(dim=1) / cue_energy
    )
    readout_energy = context_for_readout.square().sum().clamp_min(1e-8)
    readout_row_capture = (
        context_for_readout[memory.action_readout.pre_index].square().sum(dim=1)
        / readout_energy
    )
    component_energy = torch.tensor(
        [component.square().sum().item() for component in components],
        dtype=torch.float32,
    )
    component_energy = component_energy / component_energy.sum().clamp_min(1e-8)
    event_norm = event_pattern.norm().clamp_min(1e-8)
    completion_error = event_pattern - completion
    return {
        "cue_pattern": cue_pattern.detach().clone(),
        "event_pattern": event_pattern.detach().clone(),
        "action_drive": action_drive.detach().clone(),
        "outcome_drive": outcome_drive.detach().clone(),
        "completion": completion.detach().clone(),
        "cue_active_support": int((cue_pattern.abs() > 1e-6).sum().item()),
        "event_active_support": int((event_pattern.abs() > 1e-6).sum().item()),
        "cue_event_cosine": _cosine(cue_pattern, event_pattern),
        "cue_action_cosine": _cosine(cue_pattern, action_drive),
        "cue_outcome_cosine": _cosine(cue_pattern, outcome_drive),
        "action_outcome_cosine": _cosine(action_drive, outcome_drive),
        "association_completion_ratio": float(completion.norm().item() / event_norm.item()),
        "association_error_ratio": float(completion_error.norm().item() / event_norm.item()),
        "association_row_capture_ratio": float(association_row_capture.mean().item()),
        "readout_row_capture_ratio": float(readout_row_capture.mean().item()),
        "action_energy_share": float(component_energy[0].item()),
        "outcome_energy_share": float(component_energy[1].item()),
        "event_component_energy_shares": [float(value) for value in component_energy.tolist()],
    }


def _seed_record(corpus: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    model = Taiji(_config(seed), episode_id=f"m1-42-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(model, episode)
    phase_a_digest = content_digest(deepcopy(model.checkpoint()))
    phase_a_patterns = [_event_patterns(model, episode) for episode in corpus.phase_a_train]
    phase_b_patterns = [_event_patterns(model, episode) for episode in corpus.phase_b_train]
    phase_a_cues = torch.stack([item["cue_pattern"] for item in phase_a_patterns])
    phase_b_cues = torch.stack([item["cue_pattern"] for item in phase_b_patterns])
    phase_a_events = torch.stack([item["event_pattern"] for item in phase_a_patterns])
    phase_b_events = torch.stack([item["event_pattern"] for item in phase_b_patterns])
    cue_cross = F.normalize(phase_a_cues, dim=1) @ F.normalize(phase_b_cues, dim=1).T
    event_cross = F.normalize(phase_a_events, dim=1) @ F.normalize(phase_b_events, dim=1).T
    all_patterns = (*phase_a_patterns, *phase_b_patterns)
    checkpoint_digest = content_digest(deepcopy(model.checkpoint()))
    restored = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
    restored_digest = content_digest(deepcopy(restored.checkpoint()))

    def values(name: str) -> list[float]:
        return [float(item[name]) for item in all_patterns]

    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "post_a_checkpoint_digest": checkpoint_digest,
        "checkpoint_roundtrip_exact": restored_digest == checkpoint_digest,
        "active_parameter_count": model.parameter_count(),
        "planned_active_parameter_count": model.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            model.parameter_count() == model.config.planned_active_parameter_count
        ),
        "phase_a_metrics": {
            "cue_active_support": _summary(
                [float(item["cue_active_support"]) for item in phase_a_patterns]
            ),
            "event_active_support": _summary(
                [float(item["event_active_support"]) for item in phase_a_patterns]
            ),
            "cue_event_cosine": _summary(
                [float(item["cue_event_cosine"]) for item in phase_a_patterns]
            ),
            "association_completion_ratio": _summary(
                [float(item["association_completion_ratio"]) for item in phase_a_patterns]
            ),
            "association_error_ratio": _summary(
                [float(item["association_error_ratio"]) for item in phase_a_patterns]
            ),
            "association_row_capture_ratio": _summary(
                [float(item["association_row_capture_ratio"]) for item in phase_a_patterns]
            ),
            "readout_row_capture_ratio": _summary(
                [float(item["readout_row_capture_ratio"]) for item in phase_a_patterns]
            ),
        },
        "all_phase_metrics": {
            "cue_action_cosine": _summary(values("cue_action_cosine")),
            "cue_outcome_cosine": _summary(values("cue_outcome_cosine")),
            "action_outcome_cosine": _summary(values("action_outcome_cosine")),
            "action_energy_share": _summary(values("action_energy_share")),
            "outcome_energy_share": _summary(values("outcome_energy_share")),
        },
        "cross_phase": {
            "cue_cosine_mean": float(cue_cross.mean().item()),
            "cue_cosine_max": float(cue_cross.max().item()),
            "cue_near_collision_count": int((cue_cross >= 0.90).sum().item()),
            "event_cosine_mean": float(event_cross.mean().item()),
            "event_cosine_max": float(event_cross.max().item()),
            "event_near_collision_count": int((event_cross >= 0.90).sum().item()),
        },
        "phase_a_sample_count": len(phase_a_patterns),
        "phase_b_sample_count": len(phase_b_patterns),
        "holdout_updates": 0,
        "cpu_seconds": round(time.perf_counter() - started, 3),
    }


def run_audit() -> dict[str, Any]:
    corpus = _curriculum(phase_a_start=0, phase_b_start=192)
    records = [_seed_record(corpus, seed) for seed in SEEDS]
    return {
        "format": FORMAT,
        "version": 1,
        "status": "diagnostic",
        "architecture_unchanged": True,
        "variable_changed": "none; post-phase-A geometry observation only",
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "records": records,
        "diagnostic_boundary": {
            "does_not_promote": True,
            "interpretation": (
                "Use the recorded support, cosine, association completion and readout capture "
                "to choose one next repair; this report is not a B5 capability claim."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_audit()
    result["report_path"] = str(args.report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
