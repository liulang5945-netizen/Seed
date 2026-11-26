"""P5.1c probe: does contrastive (pairwise-ranking) training make the
semantic internalization organ content-addressed?

Validation-only numeric probe.  P5.1b showed the value function trained on a
single family generalizes to paraphrases of ANY family (style/skeleton
generalization, not content addressing).  The learner already ships a
pairwise-ranking update (``pairwise_margin`` + ``ranking_pairs``); this
probe trains one organ contrastively (family-A examples preferred over
family-B examples) and measures family discrimination on paraphrase
queries, mirrored by a family-B contrastive organ.  Nothing here is a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from scripts.training.eval_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    FAMILY_A_HOLDOUT,
    FAMILY_A_RETENTION,
)
from scripts.training.probe_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    FEATURE_DIM,
    PLACEBO_WORKFLOWS,
    W1_WORKFLOWS,
    W2_QUERIES,
    W2B_QUERIES,
    SemanticArtifactKnowledgeEncoder,
    _admitted,
)
from seed_platform.evolution_adapters import SkillArtifactAdapter  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

PAIRWISE_MARGIN = 0.5


def _skill_fixture(
    source_id: str,
    scope_id: str,
    *,
    partition: str,
    name: str,
    description: str,
    steps: tuple[str, ...],
    target: str,
    quality: float,
):
    projection = SkillArtifactAdapter().project(
        {
            "skill_id": source_id,
            "version": "1",
            "publisher": "seed",
            "scope_id": scope_id,
            "name": name,
            "description": description,
            "instructions": [{"action_kind": capability, "target": target} for capability in steps],
            "capabilities": list(steps),
            "constraints": ["read_only"],
        },
        partition=partition,
    )
    events = tuple(
        projection.project_event(
            {
                "event_id": f"{source_id}:step:{index}",
                "event_kind": "invoke",
                "status": "success",
                "success": quality > 0.0,
                "capability_id": capability,
                "episode_id": f"{source_id}:episode",
                "tick": index,
                "result": {"ok": quality > 0.0},
                "reward_components": {"quality": quality},
            },
            parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
            partition=partition,
        )
        for index, capability in enumerate(steps, start=1)
    )
    return _admitted(projection, source_id), events


def _corpus(workflows, *, prefix: str, partition: str, quality: float):
    artifacts: list = []
    events: list = []
    for index, workflow in enumerate(workflows):
        artifact_batch, event_batch = _skill_fixture(
            f"{prefix}.{index}",
            f"{prefix}.{index}.scope",
            partition=partition,
            name=workflow["name"],
            description=workflow["description"],
            steps=tuple(workflow["steps"]),
            target=workflow["target"],
            quality=quality,
        )
        artifacts.extend(artifact_batch)
        events.extend(event_batch)
    return tuple(artifacts), tuple(events)


def _contrastive_trial(embedder: DocumentEmbedder, corpus, holdout, retention):
    trainer = ArtifactInternalizationTrainer(
        feature_dim=FEATURE_DIM,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    trainer.encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)
    trainer.semantic.pairwise_margin = PAIRWISE_MARGIN
    train_examples = trainer._examples(corpus[0], corpus[1])
    preferred = [item for item in train_examples if item.target_reward > 0.5]
    other = [item for item in train_examples if item.target_reward <= 0.5]
    ranking_pairs = [
        (preferred_item, other_item) for preferred_item in preferred for other_item in other
    ]
    dataset_digest = content_digest(
        {
            "train_artifacts": [item.artifact_digest for item in corpus[0]],
            "holdout_artifacts": [item.artifact_digest for item in holdout[0]],
            "retention_artifacts": [item.artifact_digest for item in retention[0]],
            "train_experiences": [item.experience_digest for item in corpus[1]],
            "holdout_experiences": [item.experience_digest for item in holdout[1]],
            "retention_experiences": [item.experience_digest for item in retention[1]],
            "encoder": trainer.encoder.checkpoint(),
        }
    )
    trial = InternalizedFeatureLearner.from_checkpoint(trainer.semantic.checkpoint())
    trial.pairwise_margin = PAIRWISE_MARGIN
    trial_report = trial.consolidate(
        train_examples,
        holdout_examples=trainer._examples(holdout[0], holdout[1]),
        retention_examples=trainer._examples(retention[0], retention[1]),
        replay_digest=dataset_digest,
        passes=12,
        ranking_pairs=ranking_pairs,
    )
    return trial, trial_report


def _query_values(trial, embedder: DocumentEmbedder, texts) -> dict:
    values = [
        trial.score(
            _value_query_example(
                embedder.embed([text])[0],
            )
        )
        for text in texts
    ]
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6),
    }


def _value_query_example(feature):
    from taiji.internalization import GroundedFeatureExample

    return GroundedFeatureExample(
        example_id="internal-query",
        evidence_id="internal-query",
        outcome_id="internal-query",
        affordance_id="internal-query",
        action_kind="internal-query",
        grounding=feature,
        capability_snapshot_digest="0" * 64,
        parent_checkpoint_id="0" * 64,
        feature_payload_digest=content_digest(feature),
        reward_terms=(("query", 0.0),),
        provenance=(("organ", "semantic"),),
    )


def run_probe() -> dict:
    embedder = DocumentEmbedder()
    sourced_corpus = _corpus(
        W1_WORKFLOWS, prefix="skill.p51c.sourced", partition="train", quality=1.0
    )
    placebo_corpus = _corpus(
        PLACEBO_WORKFLOWS,
        prefix="skill.p51c.placebo",
        partition="train",
        quality=0.0,
    )
    holdout = _corpus(
        FAMILY_A_HOLDOUT, prefix="skill.p51c.holdout", partition="holdout", quality=1.0
    )
    retention = _corpus(
        FAMILY_A_RETENTION,
        prefix="skill.p51c.retention",
        partition="retention",
        quality=1.0,
    )

    sourced_trial, sourced_report = _contrastive_trial(embedder, sourced_corpus, holdout, retention)
    placebo_trial, placebo_report = _contrastive_trial(embedder, placebo_corpus, holdout, retention)

    sourced_a = _query_values(sourced_trial, embedder, W2_QUERIES)
    sourced_b = _query_values(sourced_trial, embedder, W2B_QUERIES)
    placebo_a = _query_values(placebo_trial, embedder, W2_QUERIES)
    placebo_b = _query_values(placebo_trial, embedder, W2B_QUERIES)

    return {
        "probe": "p5.1c-contrastive-content-discrimination",
        "validation_only": True,
        "pairwise_margin": PAIRWISE_MARGIN,
        "ranking_updates": {
            "sourced": sourced_report.ranking_updates,
            "placebo": placebo_report.ranking_updates,
        },
        "trials_passed": {
            "sourced": bool(sourced_report.passed),
            "placebo": bool(placebo_report.passed),
        },
        "sourced": {
            "a_para": sourced_a,
            "b_para": sourced_b,
            "discrimination_a_minus_b": round(sourced_a["mean"] - sourced_b["mean"], 6),
        },
        "placebo": {
            "a_para": placebo_a,
            "b_para": placebo_b,
            "discrimination_b_minus_a": round(placebo_b["mean"] - placebo_a["mean"], 6),
        },
        "notes": "probe only; informs frozen margins, sets no gate",
    }


def main() -> int:
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
