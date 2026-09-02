"""Audit geometry of the six native episodic event components."""

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

FORMAT = "taiji-native-m1-45-component-geometry-v1"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_m1_45_component_geometry_20260902.json"
SEEDS = (11, 29, 47)
COMPONENTS = ("action", "outcome", "reward", "time", "episode", "provenance")


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _patterns(model: Taiji, episode: Any) -> dict[str, Any]:
    model.reset_dynamics(episode_id=f"m1-45-audit-{episode.memory_id}")
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
    memory = model.memory
    context = model.fabric.cortical_context(state.regions).detach()
    cue = memory._cue_pattern(context, state.memory.threshold)
    components = {
        "action": memory._normalize_drive(
            memory.action_encoder.forward(memory._one_hot(episode.action))
        ),
        "outcome": memory._normalize_drive(
            memory.outcome_encoder.forward(memory._one_hot(episode.outcome))
        ),
        "reward": memory.reward_code.detach().clone(),
        "time": memory._normalize_drive(memory.time_encoder.forward(memory._time_code(2))),
        "episode": memory._normalize_drive(
            memory.episode_encoder.forward(
                memory._episode_code(f"m0-b2-train-{episode.memory_id}")
            )
        ),
        "provenance": memory._normalize_drive(
            memory.provenance_encoder.forward(memory._provenance_code("experienced"))
        ),
    }
    component_stack = torch.stack([components[name] for name in COMPONENTS])
    event_scale = model.config.memory_event_gain / len(COMPONENTS) ** 0.5
    component_gains = torch.tensor(
        model.config.memory_event_component_gains,
        device=model.device,
        dtype=component_stack.dtype,
    ).unsqueeze(1)
    event_drive = cue + event_scale * (
        component_stack * component_gains
    ).sum(dim=0)
    event_pattern, _ = memory._activate(event_drive, state.memory.threshold)
    completion = memory.association.forward(cue)
    event_norm = event_pattern.norm().clamp_min(1e-8)
    component_energy = component_stack.square().sum(dim=1)
    component_energy = component_energy / component_energy.sum().clamp_min(1e-8)
    return {
        "cue": cue.detach().clone(),
        "components": {name: components[name].detach().clone() for name in COMPONENTS},
        "event": event_pattern.detach().clone(),
        "cue_event_cosine": _cosine(cue, event_pattern),
        "event_active_support": int((event_pattern.abs() > 1e-6).sum().item()),
        "component_active_support": {
            name: int((components[name].abs() > 1e-6).sum().item())
            for name in COMPONENTS
        },
        "component_cue_cosine": {
            name: _cosine(cue, components[name]) for name in COMPONENTS
        },
        "component_energy_share": {
            name: float(component_energy[index].item())
            for index, name in enumerate(COMPONENTS)
        },
        "component_pair_cosine": [
            [_cosine(components[left], components[right]) for right in COMPONENTS]
            for left in COMPONENTS
        ],
        "association_completion_ratio": float(
            completion.norm().item() / event_norm.item()
        ),
        "association_error_ratio": float(
            (event_pattern - completion).norm().item() / event_norm.item()
        ),
    }

def _component_cross(phase_a: list[dict[str, Any]], phase_b: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name in COMPONENTS:
        left = torch.stack([item["components"][name] for item in phase_a])
        right = torch.stack([item["components"][name] for item in phase_b])
        similarities = F.normalize(left, dim=1) @ F.normalize(right, dim=1).T
        records[name] = {
            "mean": float(similarities.mean().item()),
            "max": float(similarities.max().item()),
            "near_collision_count": int((similarities >= 0.90).sum().item()),
        }
    return records


def _seed_record(corpus: Any, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    model = Taiji(_config(seed), episode_id=f"m1-45-phase-a-{seed}")
    for episode in corpus.phase_a_train:
        DelayedMemoryTask._write_episode(model, episode)
    phase_a = [_patterns(model, episode) for episode in corpus.phase_a_train]
    phase_b = [_patterns(model, episode) for episode in corpus.phase_b_train]
    phase_a_digest = content_digest(deepcopy(model.checkpoint()))
    restored = Taiji.from_checkpoint(deepcopy(model.checkpoint()))
    restored_digest = content_digest(deepcopy(restored.checkpoint()))
    pair_matrix = torch.tensor(
        [
            [float(item["component_pair_cosine"][left][right]) for item in phase_a]
            for left in range(len(COMPONENTS))
            for right in range(len(COMPONENTS))
        ],
        dtype=torch.float32,
    ).reshape(len(COMPONENTS), len(COMPONENTS), len(phase_a))
    component_metrics: dict[str, Any] = {}
    for index, name in enumerate(COMPONENTS):
        component_metrics[name] = {
            "active_support": _summary(
                [float(item["component_active_support"][name]) for item in (*phase_a, *phase_b)]
            ),
            "cue_cosine": _summary(
                [float(item["component_cue_cosine"][name]) for item in (*phase_a, *phase_b)]
            ),
            "energy_share": _summary(
                [float(item["component_energy_share"][name]) for item in (*phase_a, *phase_b)]
            ),
            "pair_cosine_mean": _summary(
                [float(value) for value in pair_matrix[index].flatten().tolist()]
            ),
        }
    return {
        "seed": seed,
        "phase_a_checkpoint_digest": phase_a_digest,
        "checkpoint_roundtrip_exact": restored_digest == phase_a_digest,
        "active_parameter_count": model.parameter_count(),
        "planned_active_parameter_count": model.config.planned_active_parameter_count,
        "parameter_count_matches_plan": (
            model.parameter_count() == model.config.planned_active_parameter_count
        ),
        "event_pattern": {
            "active_support": _summary(
                [float(item["event_active_support"]) for item in (*phase_a, *phase_b)]
            ),
            "cue_event_cosine": _summary(
                [float(item["cue_event_cosine"]) for item in (*phase_a, *phase_b)]
            ),
            "association_completion_ratio": _summary(
                [
                    float(item["association_completion_ratio"])
                    for item in phase_a
                ]
            ),
            "association_error_ratio": _summary(
                [float(item["association_error_ratio"]) for item in phase_a]
            ),
        },
        "components": component_metrics,
        "phase_cross_component_cosine": _component_cross(phase_a, phase_b),
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
        "variable_changed": "none; component geometry observation only",
        "components": list(COMPONENTS),
        "cue_curriculum": "maximally separated byte cues",
        "action_curriculum": "phase-A/phase-B overlap on 48/49",
        "memory_units": _config(SEEDS[0]).memory_units,
        "memory_event_gain": _config(SEEDS[0]).memory_event_gain,
        "memory_action_decoder": "shared",
        "identity_organ_enabled": False,
        "corpus_digest": corpus.digest,
        "records": records,
        "diagnostic_boundary": {
            "does_not_promote": True,
            "interpretation": (
                "Component correlations identify a candidate nuisance or collision; "
                "this report is not a B5 capability claim."
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
