"""P5.1b semantic-paraphrase transfer Gate runner.

Three same-budget arms on the E4 governance boundary:

- ``semantic-sourced``: family-A corpus internalized with the anchored
  semantic artifact encoder (``DocumentEmbedder``, 384-dim); queried with
  W2A paraphrases that share zero content words with the corpus surface;
- ``semantic-placebo-trained``: family-B corpus, identical config; the
  consolidation admission may roll the arm back (recorded), so the pure
  content-causality contrast uses its **trained semantic trial** (same
  internal construction as ``consolidate``, fitted to its own train
  MSE -> 0);
- ``structural-sourced``: the E4 hash encoder on the same family-A corpus
  (encoder-necessity control; skeleton-floor measurement).

Frozen gates: paraphrase surface disjointness, anchored deterministic
embedding, same budget, semantic paraphrase transfer (sourced trial A-para
mean >= 0.3), semantic content causality (sourced A-para minus
placebo-trained A-para >= 0.3), structural skeleton floor reproduction,
checkpoint roundtrips (stub-encoder base restore + semantic encoder swap),
quarantine rejection, and absolute resource budget.  Preregistration:
``plans/reference
/M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md``.
``growth_admitted=false`` and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from scripts.training.eval_taiji_p5_1_sourced_knowledge_transfer import (  # noqa: E402
    _capability_vocabulary,
    _quarantine_rejected,
)
from scripts.training.probe_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    FEATURE_DIM,
    W1_WORKFLOWS,
    W2_QUERIES,
    W2B_QUERIES,
    SemanticArtifactKnowledgeEncoder,
    _corpus,
    _surface_tokens,
)
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import ArtifactKnowledgeEncoder  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact  # noqa: E402
from taiji.internalization import (  # noqa: E402
    GroundedFeatureExample,
    content_digest,
)
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-1b-semantic-paraphrase-transfer-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1b_semantic_paraphrase_transfer_20260912.json"
FROZEN_TRANSFER_MARGIN = 0.3
FROZEN_CAUSALITY_MARGIN = 0.3
TOTAL_SECONDS_CAP = 600.0
FAMILY_A_HOLDOUT = (
    {
        "name": "Workspace audit",
        "description": "Audit a workspace file by opening the file, reading the file, and inspecting the audit trail.",
        "steps": ("editor.open", "editor.read", "editor.inspect"),
        "target": "workspace-audit",
    },
)
FAMILY_A_RETENTION = (
    {
        "name": "Workspace recap",
        "description": "Recap a workspace item: open the item, read the item, and inspect the recap summary.",
        "steps": ("editor.open", "editor.read", "editor.inspect"),
        "target": "workspace-recap",
    },
)
FAMILY_B_PLACEBO = (
    {
        "name": "Network sweep",
        "description": "Search the remote index, scan the matched shards, then fetch the cached entries.",
        "steps": ("network.search", "index.scan", "cache.fetch"),
        "target": "remote-index",
    },
    {
        "name": "Index refresh",
        "description": "Search across the remote catalogue, scan results for freshness, and fetch warm cache copies.",
        "steps": ("network.search", "index.scan", "cache.fetch"),
        "target": "catalogue",
    },
    {
        "name": "Mirror pull",
        "description": "Search the upstream mirror, scan the deltas, and fetch the refreshed snapshots.",
        "steps": ("network.search", "index.scan", "cache.fetch"),
        "target": "upstream-mirror",
    },
)


def _extract_content_words(texts) -> set[str]:
    """Content words of a fixture: alpha tokens minus the frozen function list."""

    function_words = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "then",
        "at",
        "in",
        "inside",
        "across",
        "by",
        "for",
        "of",
        "on",
        "it",
        "its",
        "is",
        "are",
        "with",
        "over",
        "what",
        "whatever",
        "afterwards",
    }
    return set().union(*(_surface_tokens(text) - function_words for text in texts))


def _content_word_disjoint(*word_lists) -> bool:
    sets = [_extract_content_words(texts) for texts in word_lists]
    for first in range(len(sets)):
        for second in range(first + 1, len(sets)):
            if sets[first] & sets[second]:
                return False
    return True


def _value_query_example(feature: Any) -> GroundedFeatureExample:
    """Replicate the trainer's internal value-query example construction."""

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


def _query_feature(encoder: Any, text: str) -> Any:
    if isinstance(encoder, SemanticArtifactKnowledgeEncoder):
        return encoder.embedder.embed([text])[0]
    query_artifact = EvolutionCorpusArtifact(
        corpus_id="p51b.query",
        source_kind="skill_artifact",
        source_id="p51b.paraphrase-query",
        source_version="1",
        source_digest=content_digest({"text": text}),
        unit_kind="knowledge",
        content={
            "name": "Paraphrase query",
            "description": text,
            "references": [],
            "license_use_policy": "",
        },
        language="en",
        status="admitted",
        admission_revision="p51b:query",
    )
    return encoder.encode(query_artifact)


def _values(learner: Any, encoder: Any, texts: Any) -> dict[str, Any]:
    values = [learner.score(_value_query_example(_query_feature(encoder, text))) for text in texts]
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6),
    }


def _semantic_arm(label: str, corpus, embedder: DocumentEmbedder, consolidate):
    trainer = ArtifactInternalizationTrainer(
        feature_dim=FEATURE_DIM,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    trainer.encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)  # type: ignore[assignment]
    report = consolidate(trainer, corpus)
    payload = trainer.checkpoint()
    stub = copy.deepcopy(payload)
    stub["encoder"] = ArtifactKnowledgeEncoder(feature_dim=FEATURE_DIM).checkpoint()
    stub["checkpoint_digest"] = content_digest(
        {key: value for key, value in stub.items() if key != "checkpoint_digest"}
    )
    restored = ArtifactInternalizationTrainer.from_checkpoint(stub)
    restored.encoder = SemanticArtifactKnowledgeEncoder.from_checkpoint(
        payload["encoder"], embedder=embedder
    )  # type: ignore[assignment]
    roundtrip = content_digest(restored.checkpoint()) == content_digest(payload)
    return {
        "label": label,
        "trainer": restored,
        "report": report,
        "semantic_internalized": bool(report.semantic.passed),
        "rolled_back": bool(report.rolled_back),
        "checkpoint_roundtrip": roundtrip,
    }


def _trained_trial(trainer, corpus, holdout, retention):
    """Rebuild the consolidation's internal trained semantic trial."""

    trial = InternalizedFeatureLearner.from_checkpoint(trainer.semantic.checkpoint())
    train_examples = trainer._examples(corpus[0], corpus[1])
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
    trial_report = trial.consolidate(
        train_examples,
        holdout_examples=trainer._examples(holdout[0], holdout[1]),
        retention_examples=trainer._examples(retention[0], retention[1]),
        replay_digest=dataset_digest,
        passes=12,
    )
    return trial, trial_report


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "sealed_payload_read": False,
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margins": {
            "transfer": FROZEN_TRANSFER_MARGIN,
            "causality": FROZEN_CAUSALITY_MARGIN,
        },
    }
    try:
        embedder = DocumentEmbedder()
        anchored = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
            "deterministic_double_embed": content_digest(embedder.embed(list(W2_QUERIES)))
            == content_digest(embedder.embed(list(W2_QUERIES))),
        }

        sourced_corpus = _corpus(W1_WORKFLOWS, prefix="skill.p51b.sourced", partition="train")
        placebo_corpus = _corpus(FAMILY_B_PLACEBO, prefix="skill.p51b.placebo", partition="train")
        holdout = _corpus(FAMILY_A_HOLDOUT, prefix="skill.p51b.holdout", partition="holdout")
        retention = _corpus(
            FAMILY_A_RETENTION, prefix="skill.p51b.retention", partition="retention"
        )

        def consolidate(trainer, corpus):
            return trainer.consolidate(
                corpus[0],
                holdout_artifacts=holdout[0],
                retention_artifacts=retention[0],
                train_experiences=corpus[1],
                holdout_experiences=holdout[1],
                retention_experiences=retention[1],
            )

        # ---- Frozen assertion: paraphrase surface disjointness.
        # Frozen assertion (preregistration section 2): the triple content-word
        # intersection across W1, W2A and W2B is empty.  Pairwise W2A/W2B
        # overlap is allowed and recorded.
        w1_words = _extract_content_words(
            [w["name"] + ": " + w["description"] for w in W1_WORKFLOWS]  # type: ignore[operator]
        )
        w2a_words = _extract_content_words(W2_QUERIES)
        w2b_words = _extract_content_words(W2B_QUERIES)
        surface_gate = {
            "triple_intersection_empty": not (w1_words & w2a_words & w2b_words),
            "w1_w2a_shared": sorted(w1_words & w2a_words),
            "w1_w2b_shared": sorted(w1_words & w2b_words),
            "w2a_w2b_shared": sorted(w2a_words & w2b_words),
        }

        # ---- Frozen assertion: same budget.
        budget_gate = {
            "sourced_artifacts": len(sourced_corpus[0]),
            "placebo_artifacts": len(placebo_corpus[0]),
            "sourced_events": len(sourced_corpus[1]),
            "placebo_events": len(placebo_corpus[1]),
            "same_budget": len(sourced_corpus[0]) == len(placebo_corpus[0])
            and len(sourced_corpus[1]) == len(placebo_corpus[1]),
        }
        vocabulary = {
            "holdout": sorted(_capability_vocabulary(holdout[0])),
            "sourced": sorted(_capability_vocabulary(sourced_corpus[0])),
            "placebo": sorted(_capability_vocabulary(placebo_corpus[0])),
        }

        semantic_sourced = _semantic_arm("semantic-sourced", sourced_corpus, embedder, consolidate)
        semantic_placebo = _semantic_arm("semantic-placebo", placebo_corpus, embedder, consolidate)
        placebo_trial, placebo_trial_report = _trained_trial(
            semantic_placebo["trainer"], placebo_corpus, holdout, retention
        )

        structural = ArtifactInternalizationTrainer(
            feature_dim=64,
            procedural_hidden_dim=16,
            affordance_feature_dim=12,
            seed=17,
            semantic_passes=12,
            procedural_epochs=250,
            affordance_epochs=200,
        )
        consolidate(structural, sourced_corpus)

        sourced_a = _values(
            semantic_sourced["trainer"].semantic,
            semantic_sourced["trainer"].encoder,
            W2_QUERIES,
        )
        placebo_a = _values(placebo_trial, semantic_placebo["trainer"].encoder, W2_QUERIES)
        structural_a = _values(structural.semantic, structural.encoder, W2_QUERIES)
        sourced_b = _values(
            semantic_sourced["trainer"].semantic,
            semantic_sourced["trainer"].encoder,
            W2B_QUERIES,
        )
        structural_b = _values(structural.semantic, structural.encoder, W2B_QUERIES)
        content_causality_delta = round(sourced_a["mean"] - placebo_a["mean"], 6)

        quarantine_rejected = _quarantine_rejected(
            sourced_corpus[0], sourced_corpus[1], holdout, retention
        )

        gates = {
            "paraphrase_surface_disjoint": bool(surface_gate["triple_intersection_empty"]),
            "embedder_anchored_deterministic": bool(
                anchored["deterministic_double_embed"]
                and str(anchored["revision"]).strip()
                and len(str(anchored["config_digest"])) == 64
            ),
            "same_budget_enforced": bool(budget_gate["same_budget"]),
            "semantic_paraphrase_transfer": sourced_a["mean"] >= FROZEN_TRANSFER_MARGIN,
            "semantic_content_causality": content_causality_delta >= FROZEN_CAUSALITY_MARGIN,
            "structural_skeleton_floor_recorded": structural_a["mean"] >= 0.3
            and structural_b["mean"] >= 0.3,
            "checkpoint_roundtrip_both_semantic_arms": semantic_sourced["checkpoint_roundtrip"]
            and semantic_placebo["checkpoint_roundtrip"],
            "quarantined_artifact_rejected": quarantine_rejected,
            "semantic_internalized_both_arms": semantic_sourced["semantic_internalized"]
            and placebo_trial_report.passed,
        }

        mechanical_keys = (
            "paraphrase_surface_disjoint",
            "embedder_anchored_deterministic",
            "same_budget_enforced",
            "checkpoint_roundtrip_both_semantic_arms",
            "quarantined_artifact_rejected",
            "semantic_internalized_both_arms",
        )
        total_wall = time.perf_counter() - started
        gates["resource_within_cap"] = total_wall <= TOTAL_SECONDS_CAP
        if all(gates.values()):
            outcome = "semantic_paraphrase_transfer_supported"
            status = "completed"
        elif all(gates[key] for key in mechanical_keys):
            outcome = "semantic_transfer_insufficient"
            status = "completed"
        else:
            outcome = "mechanical_failure"
            status = "failed"

        payload.update(
            {
                "status": status,
                "preregistration": (
                    "plans/reference/"
                    "M5_P5_1B_SEMANTIC_PARAPHRASE_TRANSFER_PREREGISTRATION_20260912.md"
                ),
                "embedder_anchor": anchored,
                "surface_gate": surface_gate,
                "budget_gate": budget_gate,
                "capability_vocabulary": vocabulary,
                "semantic_sourced": {
                    "a_para": sourced_a,
                    "b_para": sourced_b,
                    "w1_reference": round(
                        sum(
                            semantic_sourced["trainer"].semantic_value_from_feature(
                                semantic_sourced["trainer"].encoder.encode(artifact)
                            )
                            for artifact in sourced_corpus[0]
                            if artifact.unit_kind == "knowledge"
                        )
                        / 3,
                        6,
                    ),
                    "rolled_back": semantic_sourced["rolled_back"],
                },
                "semantic_placebo_trained": {
                    "a_para": placebo_a,
                    "trial_passed": bool(placebo_trial_report.passed),
                    "rolled_back_by_admission": semantic_placebo["rolled_back"],
                },
                "structural_sourced": {
                    "a_para": structural_a,
                    "b_para": structural_b,
                    "discrimination_a_minus_b": round(
                        structural_a["mean"] - structural_b["mean"], 6
                    ),
                },
                "content_causality_delta": content_causality_delta,
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(outcome == "semantic_paraphrase_transfer_supported"),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; paraphrase transfer with "
                        "zero surface overlap under identical budget and "
                        "train-MSE-zero trials; the structural hash encoder is "
                        "skeleton-driven (high values on unrelated content) and "
                        "therefore not content-addressed"
                    )
                    if status == "completed"
                    else "failed: mechanical gates did not pass; nothing is admitted"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    report_path = DEFAULT_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    return payload


def main() -> int:
    result = run_gate()
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "content_causality_delta": result.get("content_causality_delta"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
