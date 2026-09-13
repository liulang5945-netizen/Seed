"""P5.1 scale-up probe: does semantic-embedding internalization transfer
across paraphrase?

Validation-only numeric probe ahead of the P5.1-scale preregistration.  The
P5.1 token-overlap gate proved content transfer when corpus and tasks share
surface vocabulary; this probe tests the stronger claim with the anchored
S3 document embedder:

- ``semantic-sourced``: W1-worded family-A corpus internalized with a
  semantic artifact encoder; queried with W2 paraphrases (zero content-word
  overlap with W1);
- ``semantic-placebo``: same budget, semantically unrelated family-B corpus;
- ``structural-sourced``: the E4 hash encoder on the same W1 corpus
  (encoder-necessity control - hash features cannot cross paraphrase).

Metric: ``semantic_value_from_feature`` on the paraphrase queries.  Nothing
here is a gate.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.verify_taiji_e4_artifact_internalization import (  # noqa: E402
    _admitted,
)
from seed_platform.evolution_adapters import SkillArtifactAdapter  # noqa: E402
from taiji import ArtifactInternalizationTrainer  # noqa: E402
from taiji.artifact_internalization import (  # noqa: E402
    ArtifactKnowledgeEncoder,
    SemanticArtifactKnowledgeEncoder,
)
from taiji.document_embedding import DocumentEmbedder  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

FEATURE_DIM = 384
# W1: corpus wording (train partition).
W1_WORKFLOWS = (
    {
        "name": "Workspace inspection",
        "description": "Inspect a workspace file in bounded steps: open the workspace file, then read the workspace file, then inspect the findings.",
        "steps": ("editor.open", "editor.read", "editor.inspect"),
        "target": "workspace-file",
    },
    {
        "name": "Workspace review",
        "description": "Review a workspace document by opening it, reading its content, and inspecting the result.",
        "steps": ("editor.open", "editor.read", "editor.inspect"),
        "target": "workspace-doc",
    },
    {
        "name": "File survey",
        "description": "Survey a stored file: open it first, read the stored file next, and inspect the survey output at the end.",
        "steps": ("editor.open", "editor.read", "editor.inspect"),
        "target": "stored-file",
    },
)
# W2: paraphrase queries (zero content-word overlap with W1 wording).
W2_QUERIES = (
    "Examine an entry inside the project folder across limited operations: bring the entry up, view its contents, then check what the pass yields.",
    "Look over a saved item by bringing it up, viewing what is inside, and checking whatever surfaces afterwards.",
    "Go through an archived record: bring it up, view the record contents, and check whatever the pass surfaces.",
)
# Family-B paraphrase queries (same skeleton, unrelated semantics).
W2B_QUERIES = (
    "Sweep the distant catalogue for matches, inspect the shards, then pull the stored copies across limited operations.",
    "Probe the far catalogue, inspect the deltas, and pull refreshed snapshots in bounded passes.",
    "Comb the outer catalogue, inspect the matched slices, then pull warm copies at the end.",
)
# Family B placebo wording (semantically unrelated).
PLACEBO_WORKFLOWS = (
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

_CONTENT_WORDS = {
    "workspace",
    "file",
    "inspect",
    "open",
    "read",
    "bounded",
    "steps",
    "review",
    "document",
    "opening",
    "reading",
    "content",
    "inspecting",
    "result",
    "survey",
    "stored",
    "findings",
    "survey",
}


def _surface_tokens(text: str) -> set[str]:
    return {
        token
        for token in text.lower().replace(".", " ").replace(":", " ").split()
        if token.isalpha()
    }


def _skill_fixture(
    source_id: str,
    scope_id: str,
    *,
    partition: str,
    name: str,
    description: str,
    steps: tuple[str, ...],
    target: str,
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
                "success": True,
                "capability_id": capability,
                "episode_id": f"{source_id}:episode",
                "tick": index,
                "result": {"ok": True},
                "reward_components": {"quality": 1.0},
            },
            parent_checkpoint_digest="a" * 64 if partition == "train" else "b" * 64,
            partition=partition,
        )
        for index, capability in enumerate(steps, start=1)
    )
    return _admitted(projection, source_id), events


def _corpus(workflows, *, prefix: str, partition: str):
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
        )
        artifacts.extend(artifact_batch)
        events.extend(event_batch)
    return tuple(artifacts), tuple(events)


# SemanticArtifactKnowledgeEncoder lives in taiji.artifact_internalization
# (P5.1d promotion); imported above.  The probe-local copy was removed.


def _semantic_arm(label: str, corpus, embedder: DocumentEmbedder, consolidate) -> dict:
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
    report = consolidate(trainer, corpus)
    payload = trainer.checkpoint()
    stub = copy.deepcopy(payload)
    stub["encoder"] = ArtifactKnowledgeEncoder(feature_dim=FEATURE_DIM).checkpoint()
    stub["checkpoint_digest"] = content_digest(
        {key: value for key, value in stub.items() if key != "checkpoint_digest"}
    )
    restored = ArtifactInternalizationTrainer.from_checkpoint(stub)
    restored.encoder = SemanticArtifactKnowledgeEncoder.from_checkpoint(payload["encoder"])
    return {
        "label": label,
        "trainer": restored,
        "semantic_internalized": bool(report.semantic.passed),
        "checkpoint_roundtrip": content_digest(restored.checkpoint())
        == content_digest(trainer.checkpoint()),
    }


def run_probe() -> dict:
    embedder = DocumentEmbedder()
    determinism = content_digest(embedder.embed(W2_QUERIES)) == content_digest(
        embedder.embed(list(W2_QUERIES))
    )
    w1_tokens = set()
    w2_tokens = set()
    for workflow in W1_WORKFLOWS:
        w1_tokens |= _surface_tokens(workflow["name"] + " " + workflow["description"])
    for query in W2_QUERIES:
        w2_tokens |= _surface_tokens(query)
    surface = {
        "w1_content_words": len(w1_tokens & _CONTENT_WORDS),
        "w2_content_words": len(w2_tokens & _CONTENT_WORDS),
        "w1_w2_shared_content_words": sorted(w1_tokens & w2_tokens & _CONTENT_WORDS),
    }

    sourced_corpus = _corpus(W1_WORKFLOWS, prefix="skill.p51b.sourced", partition="train")
    placebo_corpus = _corpus(PLACEBO_WORKFLOWS, prefix="skill.p51b.placebo", partition="train")
    holdout_corpus = _corpus(
        (
            {
                "name": "Workspace audit",
                "description": "Audit a workspace file by opening the file, reading the file, and inspecting the audit trail.",
                "steps": ("editor.open", "editor.read", "editor.inspect"),
                "target": "workspace-audit",
            },
        ),
        prefix="skill.p51b.holdout",
        partition="holdout",
    )
    retention_corpus = _corpus(
        (
            {
                "name": "Workspace recap",
                "description": "Recap a workspace item: open the item, read the item, and inspect the recap summary.",
                "steps": ("editor.open", "editor.read", "editor.inspect"),
                "target": "workspace-recap",
            },
        ),
        prefix="skill.p51b.retention",
        partition="retention",
    )

    def _consolidate(trainer, corpus):
        return trainer.consolidate(
            corpus[0],
            holdout_artifacts=holdout_corpus[0],
            retention_artifacts=retention_corpus[0],
            train_experiences=corpus[1],
            holdout_experiences=holdout_corpus[1],
            retention_experiences=retention_corpus[1],
        )

    semantic_sourced = _semantic_arm("semantic-sourced", sourced_corpus, embedder, _consolidate)
    semantic_placebo = _semantic_arm("semantic-placebo", placebo_corpus, embedder, _consolidate)

    structural = ArtifactInternalizationTrainer(
        feature_dim=64,
        procedural_hidden_dim=16,
        affordance_feature_dim=12,
        seed=17,
        semantic_passes=12,
        procedural_epochs=250,
        affordance_epochs=200,
    )
    _consolidate(structural, sourced_corpus)

    def _w2_query_feature(trainer, text):
        """Text-only paraphrase query: semantic arms embed the raw text."""

        if isinstance(trainer.encoder, SemanticArtifactKnowledgeEncoder):
            return trainer.encoder.embedder.embed([text])[0]
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
        return trainer.encoder.encode(query_artifact)

    def _w1_reference(trainer, corpus):
        values = [
            trainer.semantic_value_from_feature(trainer.encoder.encode(artifact))
            for artifact in corpus[0]
            if artifact.unit_kind == "knowledge"
        ]
        return round(sum(values) / len(values), 6)

    w1_reference = _w1_reference(semantic_sourced["trainer"], sourced_corpus)
    placebo_w1_reference = _w1_reference(semantic_placebo["trainer"], placebo_corpus)

    def _values(trainer, queries):
        values = [
            trainer.semantic_value_from_feature(_w2_query_feature(trainer, query))
            for query in queries
        ]
        return {
            "values": [round(value, 6) for value in values],
            "mean": round(sum(values) / len(values), 6),
        }

    def _discrimination(trainer):
        a = _values(trainer, W2_QUERIES)["mean"]
        b = _values(trainer, W2B_QUERIES)["mean"]
        return {"a_para_mean": a, "b_para_mean": b, "a_minus_b": round(a - b, 6)}

    results = {
        "deterministic_embedding": bool(determinism),
        "surface_disjointness": surface,
        "semantic_sourced": {
            "a_para": _values(semantic_sourced["trainer"], W2_QUERIES),
            "b_para": _values(semantic_sourced["trainer"], W2B_QUERIES),
            "discrimination": _discrimination(semantic_sourced["trainer"]),
        },
        "semantic_placebo": {
            "a_para": _values(semantic_placebo["trainer"], W2_QUERIES),
            "b_para": _values(semantic_placebo["trainer"], W2B_QUERIES),
            "discrimination": _discrimination(semantic_placebo["trainer"]),
        },
        "structural_sourced": {
            "a_para": _values(structural, W2_QUERIES),
            "b_para": _values(structural, W2B_QUERIES),
            "discrimination": _discrimination(structural),
        },
        "w1_same_wording_reference_mean": w1_reference,
        "placebo_w1_reference_mean": placebo_w1_reference,
        "roundtrips": {
            "semantic_sourced": semantic_sourced["checkpoint_roundtrip"],
            "semantic_placebo": semantic_placebo["checkpoint_roundtrip"],
        },
        "same_budget": len(sourced_corpus[0]) == len(placebo_corpus[0])
        and len(sourced_corpus[1]) == len(placebo_corpus[1]),
        "semantic_internalized": {
            "sourced": semantic_sourced["semantic_internalized"],
            "placebo": semantic_placebo["semantic_internalized"],
        },
        "notes": "probe only; informs frozen margins, sets no gate",
    }
    return results


def main() -> int:
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
