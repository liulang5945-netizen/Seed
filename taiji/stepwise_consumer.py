"""Opt-in C2 selection and controls over saved cortical trajectories.

Selection receives vectors only. Source metadata, prompt text and task labels
are deliberately outside this module's scoring interface.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

MODES = frozenset(
    {
        "no-replay",
        "zero-replay",
        "model-top1",
        "oracle-correct",
        "wrong-episode",
        "shuffle-correct",
        "pooled-correct",
    }
)


def vector_digest(vectors: torch.Tensor) -> str:
    return hashlib.sha256(vectors.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class StepwiseConsumerRequest:
    mode: str
    candidate_episode_ids: tuple[str, ...] = ()
    control_episode_id: str | None = None
    shuffle_seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown stepwise consumer mode: {self.mode}")
        if self.mode == "no-replay":
            if self.candidate_episode_ids or self.control_episode_id is not None:
                raise ValueError("no-replay must not receive episodes")
            return
        if len(self.candidate_episode_ids) != 2 or len(set(self.candidate_episode_ids)) != 2:
            raise ValueError("C2 calibration requires exactly two distinct scoped episodes")
        controlled = self.mode not in {"model-top1", "zero-replay"}
        if controlled and self.control_episode_id not in self.candidate_episode_ids:
            raise ValueError("control episode must belong to the scoped candidate pool")
        if not controlled and self.control_episode_id is not None:
            raise ValueError("model selection cannot receive an oracle episode")


@torch.no_grad()
def select_and_prepare(
    request: StepwiseConsumerRequest,
    query: torch.Tensor,
    candidates: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Score normalized means; replay the original ordered vectors, padded to N."""
    if request.mode == "no-replay":
        raise ValueError("no-replay bypasses selection and fabric steps")
    if query.ndim != 1 or not torch.isfinite(query).all().item():
        raise ValueError("query cue must be a finite vector")
    # 行结构是异构键值（str/float/str），显式声明为 dict[str, Any]：
    # mypy 需要知道 score 可取负、episode_id 可作 candidates 的键。
    ranking: list[dict[str, Any]] = []
    for episode_id in request.candidate_episode_ids:
        vectors = candidates[episode_id]
        if (
            vectors.ndim != 2
            or vectors.shape[1] != query.numel()
            or not 1 <= vectors.shape[0] <= 256
            or vectors.dtype != torch.float32
            or not torch.isfinite(vectors).all().item()
        ):
            raise ValueError(f"missing or invalid saved trajectory: {episode_id}")
        score = float(F.cosine_similarity(query.cpu(), vectors.mean(dim=0), dim=0).item())
        ranking.append(
            {"episode_id": episode_id, "score": score, "vectors_sha256": vector_digest(vectors)}
        )
    ranking.sort(key=lambda row: (-row["score"], row["vectors_sha256"]))
    if (
        ranking[0]["score"] == ranking[1]["score"]
        and ranking[0]["vectors_sha256"] == ranking[1]["vectors_sha256"]
    ):
        raise ValueError("indistinguishable candidates cannot be ordered without metadata")
    picked = ranking[0]["episode_id"]
    selected = request.control_episode_id or picked
    original = candidates[selected]
    steps = max(int(candidates[key].shape[0]) for key in request.candidate_episode_ids)
    if request.mode == "zero-replay":
        replay = torch.zeros((steps, query.numel()), dtype=torch.float32)
    else:
        replay = original.clone()
        if request.mode == "shuffle-correct" and len(replay) >= 2:
            generator = torch.Generator().manual_seed(request.shuffle_seed)
            order = torch.randperm(len(replay), generator=generator)
            if torch.equal(order, torch.arange(len(replay))):
                order = order.roll(1)
            replay = replay[order]
        elif request.mode == "pooled-correct":
            replay = replay.mean(dim=0, keepdim=True).expand_as(replay).clone()
        replay = F.pad(replay, (0, 0, 0, steps - len(replay)))
    return replay, {
        "mode": request.mode,
        "ranking": ranking,
        "picked_episode_id": picked,
        "replayed_episode_id": None if request.mode == "zero-replay" else selected,
        "score_margin": ranking[0]["score"] - ranking[1]["score"],
        "replay_steps": steps,
        "original_steps": int(original.shape[0]),
        "shuffle_eligible": len(original) >= 2,
        "query_sha256": vector_digest(query),
        "replay_sha256": vector_digest(replay),
    }
