"""Seed: an AGI-directed model built on the Taiji predictive substrate."""

from .config import LanguageProviderConfig, SeedConfig
from .datasets import (
    NativeDatasetError,
    NativeDatasetReport,
    inspect_native_dataset,
    iter_native_documents,
)
from .environments import TopicWorld, play
from .judge import SeedJudge
from .model import Seed
from .sleep import SeedSleepScheduler

__all__ = [
    "Seed",
    "LanguageProviderConfig",
    "SeedConfig",
    "NativeDatasetError",
    "NativeDatasetReport",
    "inspect_native_dataset",
    "iter_native_documents",
    "SeedJudge",
    "SeedSleepScheduler",
    "TopicWorld",
    "play",
]
