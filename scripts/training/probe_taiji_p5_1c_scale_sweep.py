"""P5.1c scale sweep probe: family discrimination vs corpus size.

Validation-only.  P5.1c's 3-text probe showed (a) plain fitting already
separates the training families (ranking pairs never fire at margin 0.5)
and (b) paraphrase discrimination is weak (0.111) - the binding constraint
is corpus scale/diversity, not the training signal.  This probe sweeps the
per-family corpus size (5/10/20 texts) with generated phrasing variation
and margin 1.2 (forcing ranking updates beyond the fit separation), and
reports discrimination growth.  Nothing here is a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.probe_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    FEATURE_DIM,
    SemanticArtifactKnowledgeEncoder,
    _surface_tokens,
)
from scripts.training.probe_taiji_p5_1c_contrastive_discrimination import (  # noqa: E402
    _admitted,
    _value_query_example,
)
from seed_platform.evolution_adapters import SkillArtifactAdapter  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

A_SUBJECTS = (
    "workspace file",
    "workspace document",
    "stored file",
    "project entry",
    "archived record",
    "audit item",
    "review target",
    "inspection object",
    "checked file",
    "read-only asset",
)
A_VERBS = (
    "Inspect",
    "Review",
    "Examine",
    "Audit",
    "Survey",
    "Check",
    "Look over",
    "Go through",
    "Assess",
    "Study",
)
B_SUBJECTS = (
    "remote index",
    "upstream catalogue",
    "distant mirror",
    "network cache",
    "outer shards",
    "matched slices",
    "refreshed snapshots",
    "warm copies",
    "far entries",
    "deltas feed",
)
B_VERBS = (
    "Search",
    "Sweep",
    "Probe",
    "Scan",
    "Comb",
    "Fetch",
    "Pull",
    "Collect",
    "Query",
    "Harvest",
)

W2A_QUERIES = (
    "Bring up a saved item, view what is inside, then check whatever the pass yields.",
    "Look over an archived record, view the record contents, and check the trail afterwards.",
    "Examine an entry from the project folder: bring it up, view it, check the outcome.",
)
W2B_QUERIES = (
    "Probe the far catalogue for matches, inspect the deltas, then pull warm copies.",
    "Harvest the distant shards: query them, inspect the slices, collect the refreshed ones.",
    "Comb the outer mirror, query the matched entries, and collect refreshed snapshots.",
)


def _family_corpus(family, *, prefix: str, size: int, partition: str, quality: float):
    subjects, verbs = family
    artifacts: list = []
    events: list = []
    for index in range(size):
        subject = subjects[index % len(subjects)]
        verb = verbs[index % len(verbs)]
        description = (
            f"{verb} the {subject} in bounded steps: bring the {subject} up, "
            f"view its contents, and check the outcome."
            if family is A_FAMILY
            else (
                f"{verb} the {subject} for matches, inspect the matched slices, "
                f"then pull the stored copies."
            )
        )
        steps = (
            ("editor.open", "editor.read", "editor.inspect")
            if family is A_FAMILY
            else ("network.search", "index.scan", "cache.fetch")
        )
        projection = SkillArtifactAdapter().project(
            {
                "skill_id": f"{prefix}.{index}",
                "version": "1",
                "publisher": "seed",
                "scope_id": f"{prefix}.{index}.scope",
                "name": f"{verb} {subject}",
                "description": description,
                "instructions": [
                    {"action_kind": capability, "target": subject} for capability in steps
                ],
                "capabilities": list(steps),
                "constraints": ["read_only"],
            },
            partition=partition,
        )
        events_for_skill = tuple(
            projection.project_event(
                {
                    "event_id": f"{prefix}.{index}:step:{tick}",
                    "event_kind": "invoke",
                    "status": "success" if quality > 0.0 else "failure",
                    "success": quality > 0.0,
                    "capability_id": capability,
                    "episode_id": f"{prefix}.{index}:episode",
                    "tick": tick,
                    "result": {"ok": quality > 0.0},
                    "reward_components": {"quality": quality},
                },
                parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
                partition=partition,
            )
            for tick, capability in enumerate(steps, start=1)
        )
        artifacts.extend(_admitted(projection, f"{prefix}.{index}"))
        events.extend(events_for_skill)
    return tuple(artifacts), tuple(events)


A_FAMILY = (A_SUBJECTS, A_VERBS)
B_FAMILY = (B_SUBJECTS, B_VERBS)


def _query_values(trial, embedder: DocumentEmbedder, texts):
    from scripts.training.eval_taiji_p5_1b_semantic_paraphrase_transfer import (
        _value_query_example,
    )

    values = [trial.score(_value_query_example(embedder.embed([text])[0])) for text in texts]
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6),
    }


def _contrastive_trial(embedder, sourced, placebo, *, margin: float):
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
    trainer.semantic.pairwise_margin = margin
    combined = (sourced[0] + placebo[0], sourced[1] + placebo[1])
    train_examples = trainer._examples(combined[0], combined[1])
    preferred = [item for item in train_examples if item.target_reward > 0.5]
    other = [item for item in train_examples if item.target_reward <= 0.5]
    ranking_pairs = [
        (preferred_item, other_item) for preferred_item in preferred for other_item in other
    ]
    holdout = _family_corpus(A_FAMILY, prefix="p51c.hold", size=1, partition="holdout", quality=1.0)
    retention = _family_corpus(
        A_FAMILY, prefix="p51c.ret", size=1, partition="retention", quality=1.0
    )
    trial = InternalizedFeatureLearner.from_checkpoint(trainer.semantic.checkpoint())
    trial.pairwise_margin = margin
    report = trial.consolidate(
        train_examples,
        holdout_examples=trainer._examples(holdout[0], holdout[1]),
        retention_examples=trainer._examples(retention[0], retention[1]),
        replay_digest="probe-scale",
        passes=12,
        ranking_pairs=ranking_pairs,
    )
    return trial, report, len(ranking_pairs)


def run_probe() -> dict:
    embedder = DocumentEmbedder()
    sweep = []
    holdout_queries = list(W2A_QUERIES) + list(W2B_QUERIES)
    for size in (5, 10, 20):
        sourced = _family_corpus(
            A_FAMILY, prefix=f"p51c.a{size}", size=size, partition="train", quality=1.0
        )
        placebo = _family_corpus(
            B_FAMILY, prefix=f"p51c.b{size}", size=size, partition="train", quality=0.0
        )
        trial, report, pair_count = _contrastive_trial(embedder, sourced, placebo, margin=0.5)
        a_para = _query_values(trial, embedder, W2A_QUERIES)
        b_para = _query_values(trial, embedder, W2B_QUERIES)
        holdout_values = [
            trial.score(_value_query_example(embedder.embed([text])[0])) for text in holdout_queries
        ]
        sweep.append(
            {
                "corpus_size_per_family": size,
                "ranking_pairs": pair_count,
                "ranking_updates": report.ranking_updates,
                "train_loss_after": round(report.train_loss_after, 6),
                "a_para_mean": a_para["mean"],
                "b_para_mean": b_para["mean"],
                "discrimination_a_minus_b": round(a_para["mean"] - b_para["mean"], 6),
                "holdout_mean": round(sum(holdout_values) / len(holdout_values), 6),
            }
        )
    return {
        "probe": "p5.1c-scale-sweep",
        "validation_only": True,
        "margin": 0.5,
        "w2a_w2b_shared_content_words": sorted(
            set().union(*(_surface_tokens(t) for t in W2A_QUERIES))
            & set().union(*(_surface_tokens(t) for t in W2B_QUERIES))
        ),
        "sweep": sweep,
        "notes": "probe only; informs the frozen P5.1c design, sets no gate",
    }


def main() -> int:
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
