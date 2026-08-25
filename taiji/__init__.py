"""Taiji: a native persistent predictive-computing architecture.

This top-level package is independent of the legacy NeuroPlex Transformer
runtime.  PyTorch is used only as a tensor execution engine.
"""

from .adapter import TSKV8Adapter
from .config import CapacityPolicy, TaijiConfig
from .contracts import (
    CONTRACT_FORMAT,
    CONTRACT_VERSION,
    ActionIntent,
    CognitiveState,
    DevelopmentState,
    Goal,
    GoalState,
    HomeostaticState,
    LearningState,
    NativeCheckpoint,
    Observation,
    Outcome,
    PerceptEvent,
    PlanCandidate,
    PlanState,
    SelfState,
    WorkspaceState,
    WorldState,
)
from .contracts import MemoryState as NativeMemoryState
from .environment import EnvironmentOutcome, TaijiEnvironment
from .fabric import TaijiFabric
from .memory import EpisodicField, EpisodicReplay, EpisodicWrite
from .model import Taiji
from .organs import ByteMotor, ByteSensor, SparseReceptorBank
from .sparse import SparseSynapses
from .state import (
    MemoryRecall,
    MemoryState,
    PendingAction,
    PendingExperience,
    RegionState,
    TaijiConsolidation,
    TaijiDecision,
    TaijiOutcome,
    TaijiState,
    TaijiStep,
)

__all__ = [
    "ByteMotor",
    "ByteSensor",
    "ActionIntent",
    "CapacityPolicy",
    "CognitiveState",
    "CONTRACT_FORMAT",
    "CONTRACT_VERSION",
    "DevelopmentState",
    "EnvironmentOutcome",
    "EpisodicField",
    "EpisodicReplay",
    "EpisodicWrite",
    "MemoryRecall",
    "MemoryState",
    "NativeCheckpoint",
    "NativeMemoryState",
    "Observation",
    "Outcome",
    "PerceptEvent",
    "PlanCandidate",
    "PlanState",
    "PendingAction",
    "PendingExperience",
    "RegionState",
    "SparseReceptorBank",
    "SparseSynapses",
    "Taiji",
    "TaijiConfig",
    "TaijiConsolidation",
    "TaijiDecision",
    "TaijiEnvironment",
    "TaijiFabric",
    "TaijiOutcome",
    "TaijiState",
    "TaijiStep",
    "TSKV8Adapter",
    "Goal",
    "GoalState",
    "HomeostaticState",
    "LearningState",
    "SelfState",
    "WorkspaceState",
    "WorldState",
]
