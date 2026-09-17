"""Audit H3.6 response-plan target geometries without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from taiji import LanguageEpisode, LanguageEpisodeCorpus, Taiji, TaijiConfig

WIDTH = 32
SALT = b"r2-h3-6-plan-geometry-v1\x00"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260916)
    return parser.parse_args()


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = float(torch.linalg.vector_norm(vector))
    if norm <= 1e-12:
        raise RuntimeError("target geometry produced a zero vector")
    return vector / norm


def _signed_hash(token: bytes, *, width: int = WIDTH) -> torch.Tensor:
    digest = hashlib.sha256(SALT + token).digest()
    return torch.tensor(
        [1.0 if digest[index % len(digest)] & 1 else -1.0 for index in range(width)],
        dtype=torch.float32,
    )


def _signed_span(episode: LanguageEpisode) -> torch.Tensor:
    characters = list(episode.response)
    count = min(8, max(1, len(characters)))
    vector = torch.zeros(WIDTH)
    for index in range(count):
        start = (index * len(characters)) // count
        end = ((index + 1) * len(characters)) // count
        vector += _signed_hash("".join(characters[start:end]).encode("utf-8"))
    return _normalize(vector)


def _compositional_ngram(episode: LanguageEpisode) -> torch.Tensor:
    text = episode.response
    vector = torch.zeros(WIDTH)
    total_weight = 0.0
    for size, weight in ((1, 1.0), (2, 1.5), (3, 2.0)):
        for index in range(max(0, len(text) - size + 1)):
            token = text[index : index + size].encode("utf-8")
            vector += weight * _signed_hash(bytes([size]) + token)
            total_weight += weight
    if total_weight == 0.0:
        vector += _signed_hash(text.encode("utf-8"))
    return _normalize(vector)


def _native_teacher_raw(model: Taiji, episode: LanguageEpisode) -> torch.Tensor:
    model.reset_dynamics(episode_id=f"h3.6:{episode.episode_id}")
    contexts = []
    model.observe(
        model.config.boundary_symbol,
        learn=False,
        readout="predictive",
        use_memory=False,
        use_identity=False,
    )
    for symbol in episode.response.encode("utf-8"):
        model.observe(
            symbol,
            learn=False,
            readout="predictive",
            use_memory=False,
            use_identity=False,
        )
        contexts.append(model.snapshot().motor_context.detach().cpu())
    return torch.stack(contexts).mean(dim=0)


def _train_whitened_native(
    corpus: LanguageEpisodeCorpus,
    raw: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    train = torch.stack([raw[item.episode_id] for item in corpus.for_split("train")])
    mean = train.mean(dim=0)
    centered = train - mean
    covariance = centered.T @ centered / max(1, train.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    keep = min(WIDTH, max(1, int((eigenvalues > 1e-6).sum().item())))
    basis = eigenvectors[:, :keep]
    scale = torch.sqrt(eigenvalues[:keep].clamp_min(1e-6))
    result = {}
    for episode in corpus.episodes:
        coordinates = ((raw[episode.episode_id] - mean) @ basis) / scale
        padded = torch.zeros(WIDTH)
        padded[:keep] = coordinates
        result[episode.episode_id] = _normalize(padded)
    return result


def _metrics(
    corpus: LanguageEpisodeCorpus,
    vectors: dict[str, torch.Tensor],
) -> dict[str, object]:
    episodes = list(corpus.episodes)
    within: list[float] = []
    between: list[float] = []
    pair_records = []
    for left_index, left in enumerate(episodes):
        for right in episodes[left_index + 1 :]:
            cosine = float(torch.dot(vectors[left.episode_id], vectors[right.episode_id]))
            same_policy = left.unknown_policy == right.unknown_policy
            (within if same_policy else between).append(cosine)
            pair_records.append(
                {
                    "left": left.episode_id,
                    "right": right.episode_id,
                    "same_policy": same_policy,
                    "cosine": cosine,
                }
            )

    train = list(corpus.for_split("train"))
    transfer = []
    for split in ("dev", "final"):
        for episode in corpus.for_split(split):
            ranked = sorted(
                (
                    (
                        float(torch.dot(vectors[episode.episode_id], vectors[item.episode_id])),
                        item,
                    )
                    for item in train
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            nearest_score, nearest = ranked[0]
            transfer.append(
                {
                    "episode_id": episode.episode_id,
                    "split": split,
                    "policy": episode.unknown_policy,
                    "nearest_train": nearest.episode_id,
                    "nearest_policy": nearest.unknown_policy,
                    "nearest_cosine": nearest_score,
                    "policy_match": nearest.unknown_policy == episode.unknown_policy,
                }
            )
    within_mean = sum(within) / len(within)
    between_mean = sum(between) / len(between)
    return {
        "within_policy_cosine_mean": within_mean,
        "between_policy_cosine_mean": between_mean,
        "policy_geometry_margin": within_mean - between_mean,
        "cross_split_nearest_train_policy_accuracy": sum(
            int(item["policy_match"]) for item in transfer
        )
        / len(transfer),
        "minimum_pair_distance": min(1.0 - item["cosine"] for item in pair_records),
        "transfer_records": transfer,
    }


def main() -> int:
    args = _parse_args()
    corpus = LanguageEpisodeCorpus.from_jsonl([args.dataset])
    config = TaijiConfig.capacity_profile(300_000, seed=args.seed)
    model = Taiji(config, episode_id="h3.6-geometry-audit")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 6103)
    projection = torch.randn(WIDTH, config.motor_context_dim, generator=generator) / math.sqrt(
        config.motor_context_dim
    )

    geometries: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    raw_native: dict[str, torch.Tensor] = {}
    for episode in corpus.episodes:
        signed = _signed_span(episode)
        compositional = _compositional_ngram(episode)
        native_raw = _native_teacher_raw(model, episode)
        raw_native[episode.episode_id] = native_raw
        native = _normalize(projection @ native_raw)
        geometries["signed_hash_span"][episode.episode_id] = signed
        geometries["compositional_char_ngram"][episode.episode_id] = compositional
        geometries["native_response_state"][episode.episode_id] = native
        geometries["hybrid_native_compositional"][episode.episode_id] = _normalize(
            native + compositional
        )

    whitened = _train_whitened_native(corpus, raw_native)
    geometries["train_whitened_native_response_state"] = whitened
    for episode in corpus.episodes:
        geometries["hybrid_whitened_compositional"][episode.episode_id] = _normalize(
            whitened[episode.episode_id]
            + geometries["compositional_char_ngram"][episode.episode_id]
        )

    results = {name: _metrics(corpus, vectors) for name, vectors in geometries.items()}
    ranked = sorted(
        results,
        key=lambda name: (
            results[name]["cross_split_nearest_train_policy_accuracy"],
            results[name]["policy_geometry_margin"],
            results[name]["minimum_pair_distance"],
        ),
        reverse=True,
    )
    report = {
        "format": "taiji-r2-h3-6-plan-target-geometry-audit-v1",
        "status": "completed",
        "training_performed": False,
        "runtime_oracle_used": False,
        "labels_used_for_read_only_evaluation_only": True,
        "dataset": corpus.manifest(),
        "seed": args.seed,
        "width": WIDTH,
        "geometries": results,
        "ranking": ranked,
        "recommended_geometry": ranked[0],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "ranking": ranked,
                "summary": {
                    name: {
                        "transfer_accuracy": results[name][
                            "cross_split_nearest_train_policy_accuracy"
                        ],
                        "geometry_margin": results[name]["policy_geometry_margin"],
                        "minimum_pair_distance": results[name]["minimum_pair_distance"],
                    }
                    for name in ranked
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
