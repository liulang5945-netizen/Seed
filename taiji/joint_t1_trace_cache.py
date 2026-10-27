"""Source-bound cache of frozen TSK-v8 InputTrace/per-turn owner snapshots.

Each cache contains one split at a time and only deterministic Adapter outputs.
It never stores V4 TaskState, producer parses, semantic candidates, or language
answers, all of which must be rebuilt with the current candidate parameters.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .adapter import TSKV8Adapter
from .contracts import CognitiveState
from .input_boundary import InputTrace
from .internalization import content_digest
from .joint_t1_course import JointT1EpisodeV1, joint_t1_course_contract, joint_t1_split_digest
from .persistence import atomic_save

JOINT_T1_TRACE_CACHE_FORMAT = "taiji-joint-t1-trace-cache-v1"
JOINT_T1_TRACE_CACHE_ENTRY_FORMAT = "taiji-joint-t1-trace-cache-entry-v1"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class JointT1TraceCacheEntryV1:
    episode_id: str
    family_id: str
    input_id: str
    turn_index: int
    input_sha256: str
    trace: InputTrace
    cognitive_state: CognitiveState
    entry_sha256: str

    FORMAT = JOINT_T1_TRACE_CACHE_ENTRY_FORMAT

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.episode_id, self.family_id, self.input_id, self.input_sha256)
        ):
            raise ValueError("T1 trace cache entry requires stable episode/source identities")
        if self.turn_index < 0:
            raise ValueError("T1 trace cache turn index cannot be negative")
        if (
            self.trace.input_id != self.input_id
            or self.cognitive_state.episode_id != self.episode_id
        ):
            raise ValueError("T1 trace cache entry crossed source or episode identity")
        raw = bytes(item.value for item in self.trace.observations)
        if _sha256(raw) != self.input_sha256:
            raise ValueError("T1 cached InputTrace bytes differ from the train source")
        if self.cognitive_state.observation is None or self.cognitive_state.percept is None:
            raise ValueError("T1 cached CognitiveState lacks the final input observation/percept")
        if content_digest(self.cognitive_state.observation.to_payload()) != content_digest(
            self.trace.observations[-1].to_payload()
        ) or content_digest(self.cognitive_state.percept.to_payload()) != content_digest(
            self.trace.percepts[-1].to_payload()
        ):
            raise ValueError("T1 cached CognitiveState is not bound to its final InputTrace event")
        if self.entry_sha256 != content_digest(self._payload_without_digest()):
            raise ValueError("T1 trace cache entry digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "input_id": self.input_id,
            "turn_index": self.turn_index,
            "input_sha256": self.input_sha256,
            "trace": self.trace.to_payload(),
            "cognitive_state": self.cognitive_state.to_payload(),
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "entry_sha256": self.entry_sha256}

    @classmethod
    def create(
        cls,
        *,
        episode_id: str,
        family_id: str,
        input_id: str,
        turn_index: int,
        input_sha256: str,
        trace: InputTrace,
        cognitive_state: CognitiveState,
    ) -> JointT1TraceCacheEntryV1:
        payload = {
            "format": cls.FORMAT,
            "version": 1,
            "episode_id": episode_id,
            "family_id": family_id,
            "input_id": input_id,
            "turn_index": turn_index,
            "input_sha256": input_sha256,
            "trace": trace.to_payload(),
            "cognitive_state": cognitive_state.to_payload(),
        }
        return cls(
            episode_id=episode_id,
            family_id=family_id,
            input_id=input_id,
            turn_index=turn_index,
            input_sha256=input_sha256,
            trace=trace,
            cognitive_state=cognitive_state,
            entry_sha256=content_digest(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointT1TraceCacheEntryV1:
        required = {
            "format",
            "version",
            "episode_id",
            "family_id",
            "input_id",
            "turn_index",
            "input_sha256",
            "trace",
            "cognitive_state",
            "entry_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("T1 trace cache entry schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported T1 trace cache entry version")
        return cls(
            episode_id=str(payload["episode_id"]),
            family_id=str(payload["family_id"]),
            input_id=str(payload["input_id"]),
            turn_index=int(payload["turn_index"]),
            input_sha256=str(payload["input_sha256"]),
            trace=InputTrace.from_payload(payload["trace"], device="cpu"),
            cognitive_state=CognitiveState.from_payload(payload["cognitive_state"], device="cpu"),
            entry_sha256=str(payload["entry_sha256"]),
        )


@dataclass(frozen=True)
class JointT1TraceCacheV1:
    adapter_checkpoint_sha256: str
    course_sha256: str
    train_split_sha256: str
    episode_initial_states: tuple[tuple[str, CognitiveState], ...]
    entries: tuple[JointT1TraceCacheEntryV1, ...]
    cache_sha256: str

    FORMAT = JOINT_T1_TRACE_CACHE_FORMAT

    def __post_init__(self) -> None:
        for name in ("adapter_checkpoint_sha256", "course_sha256", "train_split_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"T1 trace cache {name} must be a SHA-256 digest")
        if not self.entries:
            raise ValueError("T1 trace cache cannot be empty")
        identities = tuple((item.episode_id, item.turn_index) for item in self.entries)
        if len(set(identities)) != len(identities):
            raise ValueError("T1 trace cache repeats an episode/turn identity")
        initial_episode_ids = tuple(episode_id for episode_id, _ in self.episode_initial_states)
        if len(set(initial_episode_ids)) != len(initial_episode_ids):
            raise ValueError("T1 trace cache repeats an episode initial-state identity")
        if any(
            state.episode_id != episode_id
            or state.observation is not None
            or state.percept is not None
            for episode_id, state in self.episode_initial_states
        ):
            raise ValueError("T1 cached episode start state is not a fresh cognitive owner")
        if self.cache_sha256 != content_digest(self._payload_without_digest()):
            raise ValueError("T1 trace cache digest differs")

    def _payload_without_digest(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "version": 1,
            "adapter_checkpoint_sha256": self.adapter_checkpoint_sha256,
            "course_sha256": self.course_sha256,
            "train_split_sha256": self.train_split_sha256,
            "episode_initial_states": [
                {"episode_id": episode_id, "cognitive_state": state.to_payload()}
                for episode_id, state in self.episode_initial_states
            ],
            "entries": [item.to_payload() for item in self.entries],
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_digest(), "cache_sha256": self.cache_sha256}

    @property
    def turn_count(self) -> int:
        return len(self.entries)

    @property
    def input_byte_count(self) -> int:
        return sum(len(item.trace.observations) for item in self.entries)

    def get(self, episode_id: str, turn_index: int) -> JointT1TraceCacheEntryV1:
        found = next(
            (
                item
                for item in self.entries
                if item.episode_id == episode_id and item.turn_index == turn_index
            ),
            None,
        )
        if found is None:
            raise KeyError("T1 input trace cache lacks the requested train episode turn")
        return found

    def get_initial_state(self, episode_id: str) -> CognitiveState:
        found = next(
            (state for item_id, state in self.episode_initial_states if item_id == episode_id),
            None,
        )
        if found is None:
            raise KeyError("T1 input trace cache lacks the requested fresh episode state")
        return found

    def validate_for(
        self,
        *,
        base_adapter_checkpoint: Mapping[str, Any],
        train_episodes: Sequence[JointT1EpisodeV1],
    ) -> None:
        if content_digest(dict(base_adapter_checkpoint)) != self.adapter_checkpoint_sha256:
            raise ValueError("T1 trace cache belongs to another Adapter checkpoint")
        if self.course_sha256 != joint_t1_course_contract()["course_sha256"]:
            raise ValueError("T1 trace cache belongs to another course contract")
        if self.train_split_sha256 != joint_t1_split_digest(train_episodes):
            raise ValueError("T1 trace cache belongs to another train split")
        expected = tuple(
            (
                episode.episode_id,
                episode.family_id,
                turn.frame.input_id,
                index,
                _sha256(turn.frame.payload),
            )
            for episode in train_episodes
            for index, turn in enumerate(episode.turns)
        )
        actual = tuple(
            (item.episode_id, item.family_id, item.input_id, item.turn_index, item.input_sha256)
            for item in self.entries
        )
        if actual != expected:
            raise ValueError("T1 trace cache sources/order differ from the selected split manifest")
        expected_episode_ids = tuple(item.episode_id for item in train_episodes)
        if tuple(item_id for item_id, _ in self.episode_initial_states) != expected_episode_ids:
            raise ValueError("T1 trace cache episode-start states differ from train manifest order")

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        if target.exists():
            raise ValueError("T1 trace cache paths are write-once")
        return atomic_save(self.to_payload(), target)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointT1TraceCacheV1:
        required = {
            "format",
            "version",
            "adapter_checkpoint_sha256",
            "course_sha256",
            "train_split_sha256",
            "episode_initial_states",
            "entries",
            "cache_sha256",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("T1 trace cache schema differs")
        if (payload["format"], payload["version"]) != (cls.FORMAT, 1):
            raise ValueError("unsupported T1 trace cache version")
        return cls(
            adapter_checkpoint_sha256=str(payload["adapter_checkpoint_sha256"]),
            course_sha256=str(payload["course_sha256"]),
            train_split_sha256=str(payload["train_split_sha256"]),
            episode_initial_states=tuple(
                (
                    str(item["episode_id"]),
                    CognitiveState.from_payload(item["cognitive_state"], device="cpu"),
                )
                for item in payload["episode_initial_states"]
            ),
            entries=tuple(
                JointT1TraceCacheEntryV1.from_payload(item) for item in payload["entries"]
            ),
            cache_sha256=str(payload["cache_sha256"]),
        )


def build_joint_t1_trace_cache(
    *,
    base_adapter_checkpoint: Mapping[str, Any],
    train_episodes: Sequence[JointT1EpisodeV1],
) -> JointT1TraceCacheV1:
    """Build a frozen Adapter trace cache from one and only one source split."""

    partitions = {episode.partition for episode in train_episodes}
    if len(partitions) != 1 or not partitions.issubset({"train", "query", "holdout"}):
        raise ValueError("T1 trace cache must contain exactly one declared split")
    adapter_sha = content_digest(dict(base_adapter_checkpoint))
    entries = []
    initial_states = []
    for episode in train_episodes:
        adapter = TSKV8Adapter.from_native_checkpoint(copy.deepcopy(dict(base_adapter_checkpoint)))
        adapter.reset_dynamics(episode_id=episode.episode_id)
        initial_states.append((episode.episode_id, adapter.cognitive_snapshot()))
        perception_weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in adapter.perception.state_dict().items()
        }
        for turn_index, turn in enumerate(episode.turns):
            trace = adapter.ingest_input(turn.frame, learn=False)
            cognitive_state = adapter.cognitive_snapshot()
            entries.append(
                JointT1TraceCacheEntryV1.create(
                    episode_id=episode.episode_id,
                    family_id=episode.family_id,
                    input_id=turn.frame.input_id,
                    turn_index=turn_index,
                    input_sha256=_sha256(bytes(turn.frame.payload)),
                    trace=trace,
                    cognitive_state=cognitive_state,
                )
            )
        if any(
            not torch.equal(value, perception_weights[name].detach().cpu())
            for name, value in adapter.perception.state_dict().items()
        ):
            raise ValueError("frozen InputTrace construction changed TSK perception parameters")
    payload = {
        "format": JOINT_T1_TRACE_CACHE_FORMAT,
        "version": 1,
        "adapter_checkpoint_sha256": adapter_sha,
        "course_sha256": str(joint_t1_course_contract()["course_sha256"]),
        "train_split_sha256": joint_t1_split_digest(train_episodes),
        "episode_initial_states": [
            {"episode_id": episode_id, "cognitive_state": state.to_payload()}
            for episode_id, state in initial_states
        ],
        "entries": [item.to_payload() for item in entries],
    }
    cache = JointT1TraceCacheV1(
        adapter_checkpoint_sha256=adapter_sha,
        course_sha256=payload["course_sha256"],
        train_split_sha256=payload["train_split_sha256"],
        episode_initial_states=tuple(initial_states),
        entries=tuple(entries),
        cache_sha256=content_digest(payload),
    )
    cache.validate_for(
        base_adapter_checkpoint=base_adapter_checkpoint,
        train_episodes=train_episodes,
    )
    return cache


def verify_joint_t1_trace_cache_replay(
    cache: JointT1TraceCacheV1,
    *,
    base_adapter_checkpoint: Mapping[str, Any],
    train_episodes: Sequence[JointT1EpisodeV1],
    episode_ids: Sequence[str],
) -> dict[str, Any]:
    """Replay selected full episodes on fresh Adapters and compare every cached field."""

    cache.validate_for(
        base_adapter_checkpoint=base_adapter_checkpoint,
        train_episodes=train_episodes,
    )
    episode_map = {item.episode_id: item for item in train_episodes}
    compared_turns = compared_input_bytes = 0
    for episode_id in episode_ids:
        episode = episode_map.get(episode_id)
        if episode is None:
            raise ValueError("trace replay audit can select only train episodes")
        adapter = TSKV8Adapter.from_native_checkpoint(copy.deepcopy(dict(base_adapter_checkpoint)))
        adapter.reset_dynamics(episode_id=episode.episode_id)
        if content_digest(adapter.cognitive_snapshot().to_payload()) != content_digest(
            cache.get_initial_state(episode.episode_id).to_payload()
        ):
            raise ValueError("trace cache episode start differs from a fresh Adapter reset")
        for turn_index, turn in enumerate(episode.turns):
            trace = adapter.ingest_input(turn.frame, learn=False)
            cognitive_state = adapter.cognitive_snapshot()
            cached = cache.get(episode.episode_id, turn_index)
            if content_digest(trace.to_payload()) != content_digest(cached.trace.to_payload()):
                raise ValueError("replayed InputTrace differs from its immutable cache row")
            if content_digest(cognitive_state.to_payload()) != content_digest(
                cached.cognitive_state.to_payload()
            ):
                raise ValueError("replayed CognitiveState differs from its immutable cache row")
            compared_turns += 1
            compared_input_bytes += len(bytes(turn.frame.payload))
    return {
        "episodes": len(episode_ids),
        "turns": compared_turns,
        "input_bytes": compared_input_bytes,
        "all_trace_and_state_fields_identical": True,
    }
