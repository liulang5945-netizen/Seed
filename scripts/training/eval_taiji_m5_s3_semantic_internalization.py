"""M5.S3 canary: semantic (embedding) internalization of sourced documents.

Upgrades the M5.S2 deterministic text-statistic features to preregistered
multilingual sentence embeddings (``plans/reference
/M5_S3_SEMANTIC_INTERNALIZATION_PREREGISTRATION_20260909.md``).  Same
domain-level held-out source design: train on Knowledge + IF records, hold
out the entirely unseen Chinese-general source domain.  Gate semantics are
identical to S2 plus a reported (non-gating) semantic probe: cosine
distances of same-domain text pairs versus cross-domain pairs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from instruments.document_embedding import DocumentEmbedder  # noqa: E402
from scripts.training.eval_taiji_m5_s2_document_internalization import (  # noqa: E402
    HOLDOUT_COUNT,
    HOLDOUT_DOMAIN,
    RETENTION_COUNT,
    TRAIN_DOMAINS,
    TRAIN_PER_DOMAIN,
    _load_domain_records,
)
from taiji import (  # noqa: E402
    GroundedOutcomeEvidence,
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationLedger,
    InternalizedFeatureLearner,
    Outcome,
    WorldAffordance,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-s3-semantic-internalization-v1"
PROBE_PAIRS_PER_GROUP = 200


def _evidence(
    domain: str,
    record_digest: str,
    embedding: torch.Tensor,
) -> GroundedOutcomeEvidence:
    short = record_digest[:16]
    affordance = WorldAffordance(
        affordance_id=f"affordance:ultradata:{short}",
        action_kind="document-experience",
        actor_id="ultradata-corpus",
        target_id=f"document:{domain}:{short}",
        features=embedding,
        feature_provenance="document-grounding",
        grounding_lineage=(
            f"document:ultradata/{domain}/{record_digest}",
            f"document:sha256:{record_digest}",
        ),
    )
    return GroundedOutcomeEvidence(
        evidence_id=f"evidence:ultradata:{short}",
        outcome_id=f"outcome:ultradata:{short}",
        outcome=Outcome(
            intent_id=f"intent:ultradata:{short}",
            reward=1.0,
            success=True,
            tick=1,
        ),
        affordance=affordance,
        capability_snapshot_digest="capability-sha256:m5s3",
        parent_checkpoint_id="checkpoint:m5s3-parent",
        owner_id="taiji:document-experience",
        reward_terms={"document_ingested": 1.0},
        world_digest=f"document-sha256:{record_digest}",
    )


def _semantic_probe(
    embeddings: dict[str, torch.Tensor],
    domain_records: dict[str, list[tuple[str, str]]],
) -> dict[str, float]:
    """Report-only probe: same-domain pair distance vs cross-domain pair."""
    domains = list(domain_records)
    same: list[float] = []
    cross: list[float] = []
    for domain in domains:
        digests = [digest for digest, _ in domain_records[domain][: PROBE_PAIRS_PER_GROUP * 2]]
        vectors = [embeddings[digest] for digest in digests]
        for index in range(0, min(len(vectors), PROBE_PAIRS_PER_GROUP * 2) - 1, 2):
            same.append(
                1.0
                - float(
                    torch.nn.functional.cosine_similarity(vectors[index], vectors[index + 1], dim=0)
                )
            )
    first, second = domains[0], domains[-1]
    first_vectors = [
        embeddings[digest] for digest, _ in domain_records[first][:PROBE_PAIRS_PER_GROUP]
    ]
    second_vectors = [
        embeddings[digest] for digest, _ in domain_records[second][:PROBE_PAIRS_PER_GROUP]
    ]
    for index in range(PROBE_PAIRS_PER_GROUP):
        cross.append(
            1.0
            - float(
                torch.nn.functional.cosine_similarity(
                    first_vectors[index], second_vectors[index], dim=0
                )
            )
        )
    return {
        "same_domain_mean_cosine_distance": sum(same) / max(1, len(same)),
        "cross_domain_mean_cosine_distance": sum(cross) / max(1, len(cross)),
        "pairs_same": len(same),
        "pairs_cross": len(cross),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "ultradata"
        / "derived"
        / "ultradata_sft_nothink_simplezh.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m5_s3_semantic_internalization_20260909.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    by_domain = _load_domain_records(args.corpus)

    train_records: list[tuple[str, str, str]] = []
    for domain in TRAIN_DOMAINS:
        for record_digest, text in by_domain[domain][:TRAIN_PER_DOMAIN]:
            train_records.append((domain, record_digest, text))
    holdout_records = [
        (HOLDOUT_DOMAIN, digest, text) for digest, text in by_domain[HOLDOUT_DOMAIN][:HOLDOUT_COUNT]
    ]
    retention_records = train_records[:RETENTION_COUNT]
    train_domains = {d for d, _, _ in train_records}
    domain_disjoint = HOLDOUT_DOMAIN not in train_domains

    embedder = DocumentEmbedder()
    embed_started = time.perf_counter()
    unique_texts: dict[str, str] = {}
    for _, digest, text in [*train_records, *holdout_records, *retention_records]:
        unique_texts.setdefault(digest, text)
    text_list = [unique_texts[digest] for digest in unique_texts]
    digest_order = list(unique_texts)
    matrix = embedder.embed(text_list)
    embeddings = {digest: matrix[index] for index, digest in enumerate(digest_order)}
    embed_seconds = time.perf_counter() - embed_started

    converter = InternalizationConverter(seed=17, replay_budget=8000)
    ledger = InternalizationLedger(converter=converter)
    train_examples = []
    rejected = 0
    for domain, record_digest, text in train_records:
        result = ledger.ingest(_evidence(domain, record_digest, embeddings[record_digest]))
        if result.example is None:
            rejected += 1
        else:
            train_examples.append(result.example)
    holdout_examples = []
    for domain, record_digest, text in holdout_records:
        result = converter.convert(_evidence(domain, record_digest, embeddings[record_digest]))
        if result.example is not None:
            holdout_examples.append(result.example)
    retention_examples = []
    for domain, record_digest, text in retention_records:
        result = converter.convert(_evidence(domain, record_digest, embeddings[record_digest]))
        if result.example is not None:
            retention_examples.append(result.example)
    if not train_examples or not holdout_examples or not retention_examples:
        raise RuntimeError("M5.S3 conversion unexpectedly rejected document evidence")

    learner = InternalizedFeatureLearner(feature_dim=embedder.dimension, learning_rate=0.5)
    report = learner.consolidate(
        tuple(train_examples),
        holdout_examples=tuple(holdout_examples),
        retention_examples=tuple(retention_examples),
        replay_digest=ledger.replay_digest,
        passes=8,
    )
    checkpoint = learner.checkpoint()
    restored = InternalizedFeatureLearner.from_checkpoint(checkpoint)
    checkpoint_roundtrip = content_digest(restored.checkpoint()) == content_digest(checkpoint)

    example_id = train_examples[0].example_id
    ledger.advance_status(example_id, "shadow")
    gate = InternalizationCausalGate(
        external_sufficiency=report.holdout_loss_after < report.holdout_loss_before,
        internalization_necessity=report.holdout_internalized_lesion_loss
        > report.holdout_loss_after,
        grounding_necessity=report.holdout_grounding_lesion_loss > report.holdout_loss_after,
        checkpoint_recoverable=checkpoint_roundtrip,
        old_task_retention=report.retention_loss_after <= report.retention_loss_before + 0.05,
    )
    lifecycle = ledger.advance_status(example_id, "internalized", causal_gate=gate)

    semantic_probe = _semantic_probe(embeddings, by_domain)
    checks = {
        "domain_disjoint_holdout": bool(domain_disjoint),
        "no_records_rejected": rejected == 0,
        "checkpoint_roundtrip": bool(checkpoint_roundtrip),
        "gate_passed": bool(gate.passed),
        "embedder_anchored": bool(embedder.revision != "unresolved"),
        "semantic_probe_reported": bool(semantic_probe["pairs_same"] > 0),
    }
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "corpus": str(args.corpus.as_posix()),
        "embedder": embedder.to_payload(),
        "data": {
            "train_domains": list(TRAIN_DOMAINS),
            "train_records": len(train_examples),
            "holdout_domain": HOLDOUT_DOMAIN,
            "holdout_records": len(holdout_examples),
            "retention_records": len(retention_examples),
            "domain_disjoint": domain_disjoint,
            "rejected": rejected,
        },
        "metrics": report.to_payload(),
        "semantic_probe": semantic_probe,
        "lifecycle": {
            "example_id": example_id,
            "status": lifecycle.status,
            "events": list(lifecycle.events),
        },
        "checks": checks,
        "technical_gate_all_passed": technical_gate_all_passed,
        "gate_criterion": (
            "identical to M5.S2 (external sufficiency, internalization "
            "necessity, grounding necessity, checkpoint, retention) with "
            "384-dim preregistered semantic embeddings; the semantic probe "
            "is report-only by preregistration"
        ),
        "boundary": "M5.S3 document internalization canary only; no provider execution, structural growth, or AGI claim",
        "resources": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "embed_seconds": embed_seconds,
            "embedded_texts": len(text_list),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report),
                "technical_gate_all_passed": technical_gate_all_passed,
                "failed_checks": [k for k, v in checks.items() if not v],
                "metrics": {
                    k: round(float(v), 6)
                    for k, v in report.to_payload().items()
                    if isinstance(v, (int, float))
                },
                "semantic_probe": {k: round(v, 4) for k, v in semantic_probe.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
