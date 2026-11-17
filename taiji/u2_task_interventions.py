"""22.4.2.5 fixed intervention matrix and raw trajectory receipts.

Zero-training fixture validation. Position arms use the frozen task contract;
audio/proprio arms use a local cross-modal scorer schema and do not modify the
frozen contract. Raw media are recorded as SHA-256 digests and actions as compact
tuples, so every trajectory is regenerable from seed plus actions.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch

from .u2_task_episode import (
    NAMES,
    PALETTE,
    PhysicalWorld,
    locate_public,
    position_word,
    structured_world,
)

TICKS = 48
UNKNOWN = "未知"
VERIFY = "求证"
CONSISTENT = "一致"
INCONSISTENT = "不一致"
AUDIO_FREQS = (400, 700, 1000)
AUDIO_RATE = 16000
AUDIO_ENERGY_MIN = 0.05
AUDIO_TOLERANCE = 100.0
INTENSITY_TOLERANCE = 1e-6
POSITION_ARMS = ("action_causality", "history_necessity", "appearance_dynamics", "occlusion")


def _action(target_x: float, force: float, intensity: float, duration: float) -> torch.Tensor:
    action = torch.zeros(15, dtype=torch.float64)
    action[0] = 1
    action[9] = target_x
    action[10] = 0.0
    action[11] = force
    action[12] = 0.0
    action[13] = intensity
    action[14] = duration
    return action


def _zero_action() -> torch.Tensor:
    action = torch.zeros(15, dtype=torch.float64)
    action[0] = 1
    return action


def _compact(action: torch.Tensor) -> list[float]:
    return [float(action[:9].argmax())] + [float(value) for value in action[9:15]]


def _digest_tensor(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    array = value.detach().cpu().contiguous()
    header = f"{array.dtype}|{tuple(array.shape)}|".encode()
    return hashlib.sha256(header + array.numpy().tobytes()).hexdigest()


def _digest_actions(actions: list[list[float]]) -> str:
    body = json.dumps(actions, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(body).hexdigest()


def _positions(state: dict[str, Any], names: list[str]) -> dict[str, str]:
    return {
        name: position_word(state["bodies"][index]["position"][0])
        for index, name in enumerate(names)
    }


def _run(
    seed: int,
    initial_x: float,
    *,
    phase_steps: int,
    direction: float,
    intensity: float,
    duration: float,
    ticks: int = TICKS,
    mass: float = 1.0,
    drag: float = 1.0,
    fixed_target: float | None = None,
) -> dict[str, Any]:
    world, names = structured_world(seed, [(initial_x, 0.0)])
    state = world.snapshot()
    for body in state["bodies"]:
        body["mass"] = mass
        body["drag"] = drag
    world = PhysicalWorld.restore(state)
    first = world.elapse("first")
    first_state = world.snapshot()
    actions: list[list[float]] = []
    result = None
    audio_peak = None
    audio_peak_energy = 0.0
    audio_peak_step = None
    for index in range(ticks - 1):
        if index < phase_steps:
            if fixed_target is None:
                position = world.snapshot()["bodies"][0]["position"][0]
                target = position + 0.15 * direction
            else:
                target = fixed_target
            action = _action(target, direction, intensity, duration)
        else:
            action = _zero_action()
        actions.append(_compact(action))
        result = world.step(f"step-{index}", action)
        audio = result["raw_media"].get("audio")
        if audio is not None:
            energy = float(audio.abs().max())
            if energy > audio_peak_energy:
                audio_peak_energy = energy
                audio_peak = audio
                audio_peak_step = index
    if result is None:
        raise ValueError("trajectory_requires_at_least_two_ticks")
    last_state = world.snapshot()
    return {
        "seed": seed,
        "initial_x": initial_x,
        "mass": mass,
        "drag": drag,
        "ticks": ticks,
        "names": names,
        "body_frequency": last_state["bodies"][0]["frequency"],
        "actions": actions,
        "first_frame": first["raw_media"]["rgb"],
        "last_frame": result["raw_media"]["rgb"],
        "last_audio": result["raw_media"].get("audio"),
        "audio_peak": audio_peak,
        "audio_peak_energy": audio_peak_energy,
        "audio_peak_step": audio_peak_step,
        "last_proprio": result["raw_media"].get("proprio"),
        "last_proprio_valid": result["raw_media"].get("proprio_valid"),
        "first_positions": _positions(first_state, names),
        "last_positions": _positions(last_state, names),
        "world_versions": [first["world_after"], result["world_after"]],
    }


def _receipt(run: dict[str, Any], trajectory_id: str) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "seed": run["seed"],
        "initial_x": run["initial_x"],
        "mass": run["mass"],
        "drag": run["drag"],
        "ticks": run["ticks"],
        "actions": run["actions"],
        "first_frame_sha256": _digest_tensor(run["first_frame"]),
        "last_frame_sha256": _digest_tensor(run["last_frame"]),
        "last_audio_sha256": _digest_tensor(run["last_audio"]),
        "audio_peak_sha256": _digest_tensor(run["audio_peak"]),
        "audio_peak_step": run["audio_peak_step"],
        "audio_peak_energy": run["audio_peak_energy"],
        "last_proprio_sha256": _digest_tensor(run["last_proprio"]),
        "first_positions": run["first_positions"],
        "last_positions": run["last_positions"],
        "body_frequency": run["body_frequency"],
        "world_versions": run["world_versions"],
    }


def _blank_entity(image: torch.Tensor, entity: str) -> torch.Tensor:
    pixels = image.detach().cpu().numpy().astype(np.float64) / 255
    distances = np.linalg.norm(pixels[..., None, :] - PALETTE, axis=-1)
    mask = torch.from_numpy(distances.argmin(-1) == NAMES.index(entity))
    out = image.clone()
    out[mask] = 0
    return out


def _tone_ring(frequency: int) -> torch.Tensor:
    times = (np.arange(256) + 1) / AUDIO_RATE
    ring = 0.6 * np.sin(2 * np.pi * frequency * times) * np.exp(-times / 0.012)
    return torch.tensor(ring, dtype=torch.float64)


def detect_frequency(ring: torch.Tensor | None) -> tuple[int | None, float]:
    if ring is None:
        return None, 0.0
    values = ring.detach().cpu().numpy()
    if not np.isfinite(values).all() or float(np.max(np.abs(values))) < AUDIO_ENERGY_MIN:
        return None, float(np.max(np.abs(values))) if values.size else 0.0
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values), 1 / AUDIO_RATE)
    peak = float(freqs[int(spectrum.argmax())])
    nearest = min(AUDIO_FREQS, key=lambda value: abs(value - peak))
    if abs(nearest - peak) > AUDIO_TOLERANCE:
        return None, float(spectrum.max())
    return nearest, float(spectrum.max())


def _visible_word(frame: torch.Tensor, name: str) -> str:
    x = locate_public(frame, name)
    return UNKNOWN if x is None else position_word(x)


def _legal_digest(
    run: dict[str, Any],
    frame: torch.Tensor | None,
    *,
    audio: torch.Tensor | None = None,
    proprio: torch.Tensor | None = None,
    proprio_valid: torch.Tensor | None = None,
) -> str:
    body = {
        "first_frame": _digest_tensor(run["first_frame"]),
        "last_frame": _digest_tensor(frame),
        "audio": _digest_tensor(audio),
        "proprio": _digest_tensor(proprio),
        "proprio_valid": _digest_tensor(proprio_valid),
        "actions": _digest_actions(run["actions"]),
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _case(
    case_id: str,
    question: str,
    target: str,
    status: str,
    run: dict[str, Any],
    *,
    frame: torch.Tensor,
    audio: torch.Tensor | None = None,
    proprio: torch.Tensor | None = None,
    proprio_valid: torch.Tensor | None = None,
    trajectory_id: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "question": question,
        "target": target,
        "status": status,
        "legal_digest": _legal_digest(
            run, frame, audio=audio, proprio=proprio, proprio_valid=proprio_valid
        ),
        "trajectory_id": trajectory_id,
        "receipt": _receipt(run, trajectory_id),
    }


def _pair_checks(
    cases: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    current_only: list[str] | None = None,
) -> dict[str, Any]:
    targets = [(case["target"], case["status"]) for case in cases]
    checks: dict[str, Any] = {
        "targets_differ": len(set(targets)) == 2,
        "legal_inputs_differ": cases[0]["legal_digest"] != cases[1]["legal_digest"],
        "first_frames_equal": torch.equal(runs[0]["first_frame"], runs[1]["first_frame"]),
        "last_frames_equal": torch.equal(runs[0]["last_frame"], runs[1]["last_frame"]),
        "actions_equal": runs[0]["actions"] == runs[1]["actions"],
    }
    if current_only is not None:
        checks["current_view_only_targets"] = current_only
        checks["current_view_only_distinguishable"] = len(set(current_only)) == 2
    return checks


def build_matrix(seed: int) -> dict[str, Any]:
    right18 = _run(
        seed, -0.55, phase_steps=18, direction=1.0, intensity=0.02, duration=0.02, fixed_target=-0.4
    )
    left18 = _run(
        seed,
        -0.55,
        phase_steps=18,
        direction=-1.0,
        intensity=0.02,
        duration=0.02,
        fixed_target=-0.7,
    )
    heavy18 = _run(
        seed,
        -0.55,
        phase_steps=18,
        direction=1.0,
        intensity=0.02,
        duration=0.02,
        mass=2.0,
        drag=2.0,
        fixed_target=-0.4,
    )
    history_a = _run(seed, -0.55, phase_steps=40, direction=1.0, intensity=0.02, duration=0.02)
    history_b = _run(seed, -0.25, phase_steps=9, direction=1.0, intensity=0.02, duration=0.02)
    name = right18["names"][0]
    question_current = f"{name}现在在哪里？"
    question_previous = f"{name}上一次在哪里？"

    action_cases = [
        _case(
            "action-right",
            question_current,
            right18["last_positions"][name],
            "answer",
            right18,
            frame=right18["last_frame"],
            trajectory_id=f"{seed}-action-right",
        ),
        _case(
            "action-left",
            question_current,
            left18["last_positions"][name],
            "answer",
            left18,
            frame=left18["last_frame"],
            trajectory_id=f"{seed}-action-left",
        ),
    ]
    action_checks = _pair_checks(
        action_cases,
        [right18, left18],
        current_only=[
            _visible_word(right18["last_frame"], name),
            _visible_word(left18["last_frame"], name),
        ],
    )

    history_cases = [
        _case(
            "history-left",
            question_previous,
            history_a["first_positions"][name],
            "answer",
            history_a,
            frame=history_a["last_frame"],
            trajectory_id=f"{seed}-history-left",
        ),
        _case(
            "history-middle",
            question_previous,
            history_b["first_positions"][name],
            "answer",
            history_b,
            frame=history_b["last_frame"],
            trajectory_id=f"{seed}-history-middle",
        ),
    ]
    history_checks = _pair_checks(
        history_cases,
        [history_a, history_b],
        current_only=[UNKNOWN, UNKNOWN],
    )
    history_checks["previous_view_targets"] = [
        history_a["first_positions"][name],
        history_b["first_positions"][name],
    ]

    dynamics_cases = [
        _case(
            "dynamics-light",
            question_current,
            right18["last_positions"][name],
            "answer",
            right18,
            frame=right18["last_frame"],
            trajectory_id=f"{seed}-dynamics-light",
        ),
        _case(
            "dynamics-heavy",
            question_current,
            heavy18["last_positions"][name],
            "answer",
            heavy18,
            frame=heavy18["last_frame"],
            trajectory_id=f"{seed}-dynamics-heavy",
        ),
    ]
    dynamics_checks = _pair_checks(
        dynamics_cases,
        [right18, heavy18],
        current_only=[
            _visible_word(right18["last_frame"], name),
            _visible_word(heavy18["last_frame"], name),
        ],
    )

    occluded_frame = _blank_entity(history_a["last_frame"], name)
    occlusion_cases = [
        _case(
            "occlusion-visible",
            question_current,
            history_a["last_positions"][name],
            "answer",
            history_a,
            frame=history_a["last_frame"],
            trajectory_id=f"{seed}-occlusion-visible",
        ),
        _case(
            "occlusion-hidden",
            question_current,
            UNKNOWN,
            "unknown",
            history_a,
            frame=occluded_frame,
            trajectory_id=f"{seed}-occlusion-hidden",
        ),
    ]
    occlusion_checks = _pair_checks(
        occlusion_cases,
        [history_a, history_a],
        current_only=[
            _visible_word(history_a["last_frame"], name),
            UNKNOWN,
        ],
    )
    occlusion_checks["last_frames_equal"] = False

    audio_run = _run(
        seed, 0.5, phase_steps=24, direction=1.0, intensity=0.2, duration=0.04, ticks=30
    )
    audio_name = audio_run["names"][0]
    audio_question = "哪个实体发出了接触声？"
    detected, energy = detect_frequency(audio_run["audio_peak"])
    present_target = audio_name if detected == audio_run["body_frequency"] else VERIFY
    present_status = "answer" if detected == audio_run["body_frequency"] else "verify"
    audio_present = _case(
        "audio-present",
        audio_question,
        present_target,
        present_status,
        audio_run,
        frame=audio_run["last_frame"],
        audio=audio_run["audio_peak"],
        trajectory_id=f"{seed}-audio-present",
    )
    audio_missing = _case(
        "audio-missing",
        audio_question,
        UNKNOWN,
        "unknown",
        audio_run,
        frame=audio_run["last_frame"],
        audio=None,
        trajectory_id=f"{seed}-audio-missing",
    )
    conflict_freq = next(value for value in AUDIO_FREQS if value != audio_run["body_frequency"])
    conflict_ring = _tone_ring(conflict_freq)
    audio_conflict = _case(
        "audio-conflict",
        audio_question,
        VERIFY,
        "verify",
        audio_run,
        frame=audio_run["last_frame"],
        audio=conflict_ring,
        trajectory_id=f"{seed}-audio-conflict",
    )
    audio_missing_checks = _pair_checks([audio_present, audio_missing], [audio_run, audio_run])
    audio_conflict_checks = _pair_checks([audio_present, audio_conflict], [audio_run, audio_run])
    audio_missing_checks["audio_detected"] = detected
    audio_missing_checks["audio_energy"] = energy

    proprio_run = _run(
        seed, -0.55, phase_steps=47, direction=1.0, intensity=0.2, duration=0.04, ticks=48
    )
    proprio_question = "本体观测的动作强度与记录的动作一致吗？"
    action_intensity = proprio_run["actions"][-1][5]
    proprio_value = proprio_run["last_proprio"]
    proprio_valid = proprio_run["last_proprio_valid"]
    if proprio_value is None or proprio_valid is None:
        raise ValueError("missing_proprio_observation")
    present_consistent = abs(float(proprio_value[4]) - action_intensity) <= INTENSITY_TOLERANCE
    present_target = CONSISTENT if present_consistent else INCONSISTENT
    present_status = "answer" if present_consistent else "verify"
    proprio_present = _case(
        "proprio-present",
        proprio_question,
        present_target,
        present_status,
        proprio_run,
        frame=proprio_run["last_frame"],
        proprio=proprio_value,
        proprio_valid=proprio_valid,
        trajectory_id=f"{seed}-proprio-present",
    )
    missing_valid = torch.zeros_like(proprio_valid)
    missing_value = torch.zeros_like(proprio_value)
    proprio_missing = _case(
        "proprio-missing",
        proprio_question,
        UNKNOWN,
        "unknown",
        proprio_run,
        frame=proprio_run["last_frame"],
        proprio=missing_value,
        proprio_valid=missing_valid,
        trajectory_id=f"{seed}-proprio-missing",
    )
    conflict_value = proprio_value.clone()
    conflict_value[4] = 0.8
    proprio_conflict = _case(
        "proprio-conflict",
        proprio_question,
        INCONSISTENT,
        "verify",
        proprio_run,
        frame=proprio_run["last_frame"],
        proprio=conflict_value,
        proprio_valid=proprio_valid,
        trajectory_id=f"{seed}-proprio-conflict",
    )
    proprio_missing_checks = _pair_checks(
        [proprio_present, proprio_missing], [proprio_run, proprio_run]
    )
    proprio_conflict_checks = _pair_checks(
        [proprio_present, proprio_conflict], [proprio_run, proprio_run]
    )
    proprio_missing_checks["action_intensity"] = action_intensity
    proprio_missing_checks["proprio_intensity"] = float(proprio_value[4])

    return {
        "seed": seed,
        "name": name,
        "arms": {
            "action_causality": {
                "cases": action_cases,
                "checks": action_checks,
            },
            "history_necessity": {
                "cases": history_cases,
                "checks": history_checks,
            },
            "appearance_dynamics": {
                "cases": dynamics_cases,
                "checks": dynamics_checks,
            },
            "occlusion": {
                "cases": occlusion_cases,
                "checks": occlusion_checks,
            },
            "audio_missing": {
                "cases": [audio_present, audio_missing],
                "checks": audio_missing_checks,
            },
            "audio_conflict": {
                "cases": [audio_present, audio_conflict],
                "checks": audio_conflict_checks,
            },
            "proprio_missing": {
                "cases": [proprio_present, proprio_missing],
                "checks": proprio_missing_checks,
            },
            "proprio_conflict": {
                "cases": [proprio_present, proprio_conflict],
                "checks": proprio_conflict_checks,
            },
        },
    }
