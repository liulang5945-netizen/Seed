"""R030 activity fixture: recurring external arrivals with a declared oracle bound.

R029's activity fixture placed one message in the queue, so ``interact`` was legal
in at most one of 32 evaluation rounds and the structural optimum was idle. This
module adds a host-side arrival schedule so the reward-bearing activity stays
reachable, plus an analytic upper bound the fixture must be able to reach before
any learning reading is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taiji.activity_learning import COMPUTE_COST, INTERACT_REWARD
from taiji.native_activity import ActivityEnvironment

INTERVAL_TICKS = 1
MESSAGE_COUNT = 32
MESSAGES = tuple(
    f"第{index:02d}号材料：蓝色杯子在窗台。".encode() for index in range(MESSAGE_COUNT)
)


@dataclass
class ArrivalSource:
    """Deliver message k once the host clock reaches tick k * interval.

    ``per_round=True`` instead tops the queue up by one message on every call, so
    the reward-bearing activity stays legal without a wait step. That variant
    removes the two-step wait->arrival->interact chain from the credit path.
    """

    messages: tuple[bytes, ...] = MESSAGES
    interval: int = INTERVAL_TICKS
    per_round: bool = False
    delivered: int = field(default=0)

    def __call__(self, env: ActivityEnvironment) -> int:
        if self.per_round:
            if self.delivered >= len(self.messages):
                return 0
            env.message_queue.append(self.messages[self.delivered])
            self.delivered += 1
            return 1
        enqueued = 0
        while self.delivered < len(self.messages) and self.delivered * self.interval <= env.now:
            env.message_queue.append(self.messages[self.delivered])
            self.delivered += 1
            enqueued += 1
        return enqueued

    def reset(self) -> None:
        self.delivered = 0


def patches(message: bytes) -> int:
    return (len(message) + 7) // 8


def legal_interactions(rounds: int, interval: int = INTERVAL_TICKS) -> int:
    """Rounds in which a message is in hand: one interact, then one wait per tick."""
    if interval < 1:
        raise ValueError("arrival interval must be at least one tick")
    interactions, tick, delivered = 0, 0, 0
    for _ in range(rounds):
        if delivered * interval <= tick and delivered < MESSAGE_COUNT:
            delivered += 1
            interactions += 1
        else:
            tick += 1
    return interactions


def oracle_cumulative_reward(rounds: int, interval: int = INTERVAL_TICKS) -> dict:
    """Analytic bound: every legal interact earns the reward minus its patch cost."""
    interactions = legal_interactions(rounds, interval)
    per_interaction = INTERACT_REWARD - COMPUTE_COST * patches(MESSAGES[0])
    return {
        "rounds": rounds,
        "interval_ticks": interval,
        "legal_interactions": interactions,
        "per_interaction_reward_floor": per_interaction,
        "oracle_reward": interactions * per_interaction,
        "idle_reward": 0.0,
    }


def reachable(rounds: int, floor: float, interval: int = INTERVAL_TICKS) -> bool:
    return oracle_cumulative_reward(rounds, interval)["oracle_reward"] >= floor
