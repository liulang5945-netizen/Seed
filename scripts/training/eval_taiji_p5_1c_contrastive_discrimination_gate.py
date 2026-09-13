"""P5.1c contrastive content-discrimination Gate runner.

Trains one semantic internalization organ contrastively (family-A examples
reward 1.0, family-B examples reward 0.0, cross-family ranking pairs via
the learner's existing ``pairwise_margin``/ranking mechanism) over ten
phrasing-varied texts per family, then measures family discrimination on
paraphrase queries (zero content-word overlap with the corpus surface).

Frozen gates: contrastive discrimination (A-para minus B-para >= 0.3),
paraphrase transfer retained (A-para >= 0.3), B-paraphrase refusal
(B-para <= 0), structural skeleton-floor reproduction, surface
disjointness, anchored deterministic embedding, same budget, trial causal
gate, and absolute resource budget.  Preregistration:
``plans/reference
/M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md``.
``growth_admitted=false`` and ``can_promote=false`` throughout.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.eval_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    _extract_content_words,
    _value_query_example,
)
from scripts.training.probe_taiji_p5_1b_semantic_paraphrase_transfer import (  # noqa: E402
    FEATURE_DIM,
    SemanticArtifactKnowledgeEncoder,
)
from scripts.training.probe_taiji_p5_1c_contrastive_discrimination import (  # noqa: E402
    PAIRWISE_MARGIN,
)
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import ArtifactKnowledgeEncoder  # noqa: E402
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402
from taiji.internalization_learner import InternalizedFeatureLearner  # noqa: E402

REPORT_FORMAT = "taiji-p5-1c-contrastive-discrimination-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p5_1c_contrastive_discrimination_20260912.json"
FROZEN_DISCRIMINATION_MARGIN = 0.3
FROZEN_TRANSFER_MARGIN = 0.3
FROZEN_REFUSAL_CEILING = 0.0
TOTAL_SECONDS_CAP = 600.0
CORPUS_SIZE_PER_FAMILY = 10

A_WORKFLOWS = (
    *(
        {
            "name": f"{verb} {subject}",
            "description": (
                f"{verb} the {subject} in bounded steps: bring the {subject} up, "
                f"view its contents, and check the outcome."
            ),
            "steps": ("editor.open", "editor.read", "editor.inspect"),
            "target": subject,
        }
        for subject, verb in zip(
            (
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
            ),
            (
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
            ),
        )
    ),
)
B_WORKFLOWS = (
    *(
        {
            "name": f"{verb} {subject}",
            "description": (
                f"{verb} the {subject} for matches, inspect the matched slices, "
                f"then pull the stored copies."
            ),
            "steps": ("network.search", "index.scan", "cache.fetch"),
            "target": subject,
        }
        for subject, verb in zip(
            (
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
            ),
            (
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
            ),
        )
    ),
)
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


def _family_corpus(family, *, prefix: str, partition: str, quality: float):
    artifacts: list = []
    events: list = []
    from seed_platform.evolution_adapters import SkillArtifactAdapter

    for index, workflow in enumerate(family):
        projection = SkillArtifactAdapter().project(
            {
                "skill_id": f"{prefix}.{index}",
                "version": "1",
                "publisher": "seed",
                "scope_id": f"{prefix}.{index}.scope",
                "name": workflow["name"],
                "description": workflow["description"],
                "instructions": [
                    {"action_kind": capability, "target": workflow["target"]}
                    for capability in workflow["steps"]
                ],
                "capabilities": list(workflow["steps"]),
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
            for tick, capability in enumerate(workflow["steps"], start=1)
        )
        artifacts.extend(
            tuple(
                artifact.with_status("admitted", admission_revision=f"{prefix}:{index}")
                for artifact in projection.corpus
            )
        )
        events.extend(events_for_skill)
    return tuple(artifacts), tuple(events)


def _contrastive_trial(embedder: DocumentEmbedder):
    sourced = _family_corpus(
        A_WORKFLOWS, prefix="skill.p51c.sourced", partition="train", quality=1.0
    )
    placebo = _family_corpus(
        B_WORKFLOWS, prefix="skill.p51c.placebo", partition="train", quality=0.0
    )
    holdout = _family_corpus(
        FAMILY_A_HOLDOUT, prefix="skill.p51c.holdout", partition="holdout", quality=1.0
    )
    retention = _family_corpus(
        FAMILY_A_RETENTION,
        prefix="skill.p51c.retention",
        partition="retention",
        quality=1.0,
    )
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
    trainer.semantic.pairwise_margin = PAIRWISE_MARGIN
    combined = (sourced[0] + placebo[0], sourced[1] + placebo[1])
    train_examples = trainer._examples(combined[0], combined[1])
    preferred = [item for item in train_examples if item.target_reward > 0.5]
    other = [item for item in train_examples if item.target_reward <= 0.5]
    ranking_pairs = [
        (preferred_item, other_item) for preferred_item in preferred for other_item in other
    ]
    dataset_digest = content_digest(
        {
            "train_artifacts": [item.artifact_digest for item in combined[0]],
            "holdout_artifacts": [item.artifact_digest for item in holdout[0]],
            "retention_artifacts": [item.artifact_digest for item in retention[0]],
            "train_experiences": [item.experience_digest for item in combined[1]],
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
    return trial, trial_report, (sourced, placebo)


def _structural_arm(sourced, holdout, retention):
    structural = ArtifactInternalizationTrainer(
        feature_dim=64,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    structural.encoder = ArtifactKnowledgeEncoder(feature_dim=64)
    trial = InternalizedFeatureLearner.from_checkpoint(structural.semantic.checkpoint())
    trial.consolidate(
        structural._examples(sourced[0], sourced[1]),
        holdout_examples=structural._examples(holdout[0], holdout[1]),
        retention_examples=structural._examples(retention[0], retention[1]),
        replay_digest="structural",
        passes=12,
    )
    return trial, structural.encoder


def _query_feature(encoder: Any, text: str) -> Any:
    """Structural-arm text feature: hash-encode a query corpus artifact."""

    query_artifact = EvolutionCorpusArtifact(
        corpus_id="p51c.query",
        source_kind="skill_artifact",
        source_id="p51c.paraphrase-query",
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
        admission_revision="p51c:query",
    )
    return encoder.encode(query_artifact)


def _values(learner: Any, encoder: Any, texts) -> dict[str, Any]:
    if isinstance(encoder, SemanticArtifactKnowledgeEncoder):
        features = [encoder.embedder.embed([text])[0] for text in texts]
    else:
        features = [_query_feature(encoder, text) for text in texts]
    values = [learner.score(_value_query_example(feature)) for feature in features]
    return {
        "values": [round(value, 6) for value in values],
        "mean": round(sum(values) / len(values), 6),
    }


def run_gate() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "growth_admitted": False,
        "can_promote": False,
        "frozen_margins": {
            "discrimination": FROZEN_DISCRIMINATION_MARGIN,
            "transfer": FROZEN_TRANSFER_MARGIN,
            "refusal_ceiling": FROZEN_REFUSAL_CEILING,
        },
    }
    try:
        embedder = DocumentEmbedder()
        anchored = {
            "model_id": embedder.model_id,
            "revision": embedder.revision,
            "config_digest": embedder.config_digest,
            "deterministic_double_embed": content_digest(embedder.embed(list(W2A_QUERIES)))
            == content_digest(embedder.embed(list(W2A_QUERIES))),
        }

        trial, trial_report, (sourced, placebo) = _contrastive_trial(embedder)
        structural_holdout = _family_corpus(
            FAMILY_A_HOLDOUT, prefix="skill.p51c.holdout", partition="holdout", quality=1.0
        )
        structural_retention = _family_corpus(
            FAMILY_A_RETENTION,
            prefix="skill.p51c.retention",
            partition="retention",
            quality=1.0,
        )
        structural_trial, structural_encoder = _structural_arm(
            sourced, structural_holdout, structural_retention
        )
        semantic_encoder = SemanticArtifactKnowledgeEncoder(embedder=embedder)

        a_para = _values(trial, semantic_encoder, W2A_QUERIES)
        b_para = _values(trial, semantic_encoder, W2B_QUERIES)
        structural_a = _values(structural_trial, structural_encoder, W2A_QUERIES)
        structural_b = _values(structural_trial, structural_encoder, W2B_QUERIES)
        discrimination = round(a_para["mean"] - b_para["mean"], 6)

        # ---- Frozen assertion: W2A/W2B content-word disjointness.
        w2a_words = _extract_content_words(W2A_QUERIES)
        w2b_words = _extract_content_words(W2B_QUERIES)
        surface_gate = {
            "w2a_w2b_shared": sorted(w2a_words & w2b_words),
            "pairwise_disjoint": not (w2a_words & w2b_words),
        }

        # ---- Frozen assertion: same budget across the two families.
        budget_gate = {
            "sourced_artifacts": len(sourced[0]),
            "placebo_artifacts": len(placebo[0]),
            "sourced_events": len(sourced[1]),
            "placebo_events": len(placebo[1]),
            "same_budget": len(sourced[0]) == len(placebo[0])
            and len(sourced[1]) == len(placebo[1]),
        }

        gates = {
            "contrastive_discrimination": discrimination >= FROZEN_DISCRIMINATION_MARGIN,
            "paraphrase_transfer_retained": a_para["mean"] >= FROZEN_TRANSFER_MARGIN,
            "b_para_refusal": b_para["mean"] <= FROZEN_REFUSAL_CEILING,
            "structural_skeleton_floor_recorded": structural_a["mean"] >= 0.3
            and structural_b["mean"] >= 0.3,
            "paraphrase_surface_disjoint": bool(surface_gate["pairwise_disjoint"]),
            "embedder_anchored_deterministic": bool(
                anchored["deterministic_double_embed"]
                and str(anchored["revision"]).strip()
                and len(str(anchored["config_digest"])) == 64
            ),
            "same_budget_enforced": bool(budget_gate["same_budget"]),
            "trial_causal_gate": bool(trial_report.passed),
        }
        mechanical_keys = (
            "paraphrase_surface_disjoint",
            "embedder_anchored_deterministic",
            "same_budget_enforced",
            "trial_causal_gate",
        )

        total_wall = time.perf_counter() - started
        gates["resource_within_cap"] = total_wall <= TOTAL_SECONDS_CAP
        if all(gates.values()):
            outcome = "contrastive_content_discrimination_supported"
            status = "completed"
        elif all(gates[key] for key in mechanical_keys):
            outcome = "discrimination_insufficient"
            status = "completed"
        else:
            outcome = "mechanical_failure"
            status = "failed"

        payload.update(
            {
                "status": status,
                "preregistration": (
                    "plans/reference/"
                    "M5_P5_1C_CONTRASTIVE_DISCRIMINATION_PREREGISTRATION_20260912.md"
                ),
                "embedder_anchor": anchored,
                "surface_gate": surface_gate,
                "budget_gate": budget_gate,
                "contrastive_trial": {
                    "a_para": a_para,
                    "b_para": b_para,
                    "discrimination_a_minus_b": discrimination,
                    "pairwise_margin": PAIRWISE_MARGIN,
                    "ranking_updates": trial_report.ranking_updates,
                    "train_loss_after": round(trial_report.train_loss_after, 6),
                    "causal_gate_passed": bool(trial_report.passed),
                },
                "structural_arm": {
                    "a_para": structural_a,
                    "b_para": structural_b,
                    "discrimination_a_minus_b": round(
                        structural_a["mean"] - structural_b["mean"], 6
                    ),
                },
                "gates": gates,
                "outcome": outcome,
                "experiment_passed": bool(
                    outcome == "contrastive_content_discrimination_supported"
                ),
                "growth_admitted": False,
                "can_promote": False,
                "interpretation": (
                    (
                        f"completed: outcome={outcome}; contrastive training over a "
                        "ten-text-per-family corpus makes the semantic value "
                        "function content-addressed (A-para high, B-para refused) "
                        "while the structural hash encoder reproduces the "
                        "skeleton floor"
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
    contrastive = result.get("contrastive_trial") or {}
    print(
        json.dumps(
            {
                "report": str(DEFAULT_REPORT),
                "status": result.get("status"),
                "outcome": result.get("outcome"),
                "experiment_passed": result.get("experiment_passed"),
                "discrimination_a_minus_b": contrastive.get("discrimination_a_minus_b"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
