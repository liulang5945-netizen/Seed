"""Data-driven homeostatic regulation for the Taiji runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import HomeostaticState

HOMEOSTASIS_CHECKPOINT_FORMAT = "taiji-homeostasis-v1"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class HomeostaticConfig:
    """Event-driven control gains and resource thresholds."""

    curiosity_gain: float = 0.65
    curiosity_decay: float = 0.04
    fatigue_gain: float = 0.24
    fatigue_recovery: float = 0.08
    stress_gain: float = 0.50
    stress_recovery: float = 0.10
    reward_relief: float = 0.18
    sleep_recovery: float = 0.35
    sleep_stress_recovery: float = 0.40
    play_fatigue_cost: float = 0.08
    play_stress_relief: float = 0.20
    rest_threshold: float = 0.65
    play_threshold: float = 0.55

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if float(value) < 0.0:
                raise ValueError(f"homeostatic {name} cannot be negative")
        bounded = (
            "curiosity_decay",
            "fatigue_recovery",
            "stress_recovery",
            "sleep_recovery",
            "sleep_stress_recovery",
            "rest_threshold",
            "play_threshold",
        )
        for name in bounded:
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"homeostatic {name} must be at most one")

    def to_payload(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> HomeostaticConfig:
        return cls(**{name: float(value) for name, value in payload.items()})


@dataclass(frozen=True)
class HomeostaticDrive:
    """Current internal drives exposed to action, replay, and rest policies."""

    exploration: float
    replay: float
    rest: float
    play: float


class HomeostaticController:
    """Update internal drives from prediction, novelty, resource, and outcome signals."""

    def __init__(self, config: HomeostaticConfig | None = None) -> None:
        self.config = config or HomeostaticConfig()

    def drive(self, state: HomeostaticState) -> HomeostaticDrive:
        exploration = _clamp(state.curiosity * (1.0 - state.fatigue))
        replay = _clamp(max(state.fatigue, state.stress))
        rest = _clamp(state.fatigue + 0.5 * state.stress)
        play = _clamp(state.curiosity * (1.0 - state.stress) * (1.0 - state.fatigue))
        return HomeostaticDrive(exploration, replay, rest, play)

    def select_mode(self, state: HomeostaticState) -> str:
        """Select sleep/play/wake from current drives without a fixed schedule."""

        drive = self.drive(state)
        if drive.rest >= self.config.rest_threshold:
            return "sleep"
        if drive.play >= self.config.play_threshold:
            return "play"
        return "wake"

    def update(
        self,
        state: HomeostaticState,
        *,
        prediction_error: float = 0.0,
        novelty: float = 0.0,
        reward: float = 0.0,
        resource_cost: float = 0.0,
        mode: str = "wake",
    ) -> HomeostaticState:
        """Apply one event-driven homeostatic transition."""

        if mode not in {"wake", "sleep", "play"}:
            raise ValueError("homeostatic mode must be wake, sleep, or play")
        error = _clamp(abs(float(prediction_error)))
        novelty = _clamp(abs(float(novelty)))
        positive_reward = _clamp(max(0.0, float(reward)))
        negative_reward = _clamp(max(0.0, -float(reward)))
        cost = _clamp(abs(float(resource_cost)))
        config = self.config
        if mode == "sleep":
            curiosity = _clamp(state.curiosity * (1.0 - config.curiosity_decay))
            fatigue = _clamp(state.fatigue * (1.0 - config.sleep_recovery))
            stress = _clamp(state.stress * (1.0 - config.sleep_stress_recovery))
        elif mode == "play":
            curiosity = _clamp(
                state.curiosity
                + config.curiosity_gain * novelty
                - config.curiosity_decay * state.curiosity
            )
            fatigue = _clamp(
                state.fatigue * (1.0 - config.fatigue_recovery)
                + config.play_fatigue_cost
                + config.fatigue_gain * cost
            )
            stress = _clamp(
                state.stress * (1.0 - config.stress_recovery)
                + config.stress_gain * error
                - config.play_stress_relief
            )
        else:
            curiosity = _clamp(
                state.curiosity * (1.0 - config.curiosity_decay)
                + config.curiosity_gain * max(novelty, error)
            )
            fatigue = _clamp(
                state.fatigue * (1.0 - config.fatigue_recovery)
                + config.fatigue_gain * (cost + 0.5 * error)
            )
            stress = _clamp(
                state.stress * (1.0 - config.stress_recovery)
                + config.stress_gain * (error + negative_reward)
                - config.reward_relief * positive_reward
            )
        return HomeostaticState(
            tick=state.tick + 1,
            curiosity=curiosity,
            fatigue=fatigue,
            stress=stress,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": HOMEOSTASIS_CHECKPOINT_FORMAT,
            "config": self.config.to_payload(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> HomeostaticController:
        if payload.get("format") != HOMEOSTASIS_CHECKPOINT_FORMAT:
            raise ValueError("unsupported homeostasis checkpoint format")
        return cls(HomeostaticConfig.from_payload(dict(payload.get("config", {}))))
