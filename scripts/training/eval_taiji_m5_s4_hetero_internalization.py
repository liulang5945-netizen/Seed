"""M5.S4 canary: restore holdout heterogeneity for the semantic internalization.

S3's unseen-source holdout (Chinese-general) was only a mild topical variant
of the training domains, so generalizing to it was trivial and the grounding
lesion collapsed to machine precision.  This canary keeps S3's exact
spec (384-dim embeddings, 6000 Knowledge+IF train, 2000 holdout, 2000
retention, S1 causal gate) and changes ONE variable: the holdout becomes a
genuinely heterogeneous source (Math + Code_Agent), far from Chinese QA in
embedding space.  A semantic-distance probe reports the train-vs-holdout
separation so the heterogeneity premise is verified rather than assumed.

If the grounding lesion recovers, S3's gate measured the right thing on too-
easy data; if it stays collapsed, the semantic embedding genuinely subsumes
the grounding signal and a hardened gate has a causal basis.
"""

from __future__ import annotations

import argparse
import hashlib
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
    RETENTION_COUNT,
    TRAIN_DOMAINS,
    TRAIN_PER_DOMAIN,
    _load_domain_records,
)
from scripts.training.eval_taiji_m5_s3_semantic_internalization import (  # noqa: E402
    _evidence,
)
from taiji import (  # noqa: E402
    InternalizationCausalGate,
    InternalizationConverter,
    InternalizationLedger,
    InternalizedFeatureLearner,
    content_digest,
)

REPORT_FORMAT = "taiji-m5-s4-hetero-internalization-v1"
HETERO_MANIFEST = PROJECT_ROOT / "reports" / "taiji_m5_s4_hetero_holdout_manifest_20260909.json"
HETERO_HOLDOUT_COUNT = 2000


def _load_hetero_records(corpus: Path) -> list[tuple[str, str, str]]:
    manifest = json.loads(HETERO_MANIFEST.read_text(encoding="utf-8"))
    expected = str(manifest["output_sha256"])
    lines = corpus.read_text(encoding="utf-8").splitlines()
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    if digest != expected:
        raise ValueError("hetero corpus digest does not match the S4 manifest")
    rows: list[tuple[str, str, str]] = []
    for line in lines:
        record = json.loads(line)
        text = str(record["text"])
        domain = str(record["domain"])
        rows.append((domain, content_digest(text), text))
    return rows


def _centroid_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    ca = torch.nn.functional.normalize(a.mean(dim=0), p=2, dim=0)
    cb = torch.nn.functional.normalize(b.mean(dim=0), p=2, dim=0)
    return float(1.0 - torch.dot(ca, cb))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-corpus",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "ultradata"
        / "derived"
        / "ultradata_sft_nothink_simplezh.jsonl",
    )
    parser.add_argument(
        "--holdout-corpus",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "ultradata"
        / "derived"
        / "ultradata_hetero_holdout_simplezh.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_m5_s4_hetero_internalization_20260909.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    by_domain = _load_domain_records(args.train_corpus)
    train_records: list[tuple[str, str, str]] = []
    for domain in TRAIN_DOMAINS:
        for record_digest, text in by_domain[domain][:TRAIN_PER_DOMAIN]:
            train_records.append((domain, record_digest, text))
    retention_records = train_records[:RETENTION_COUNT]

    hetero_all = _load_hetero_records(args.holdout_corpus)
    holdout_records = hetero_all[:HETERO_HOLDOUT_COUNT]
    holdout_domains = sorted({d for d, _, _ in holdout_records})
    train_domains = {d for d, _, _ in train_records}
    domain_disjoint = all(d not in train_domains for d in holdout_domains)

    embedder = DocumentEmbedder()
    embed_started = time.perf_counter()
    unique_texts: dict[str, str] = {}
    for _, digest, text in [*train_records, *holdout_records, *retention_records]:
        unique_texts.setdefault(digest, text)
    digest_order = list(unique_texts)
    matrix = embedder.embed([unique_texts[d] for d in digest_order])
    embeddings = {digest: matrix[index] for index, digest in enumerate(digest_order)}
    embed_seconds = time.perf_counter() - embed_started

    # Semantic-distance probe: verify the heterogeneity premise.
    train_vecs = torch.stack([embeddings[d] for _, d, _ in train_records])
    math_vecs = torch.stack([embeddings[d] for dom, d, _ in holdout_records if dom == "Math"])
    code_vecs = torch.stack([embeddings[d] for dom, d, _ in holdout_records if dom == "Code_Agent"])
    probe = {
        "train_vs_math_centroid_distance": _centroid_distance(train_vecs, math_vecs),
        "train_vs_code_centroid_distance": _centroid_distance(train_vecs, code_vecs),
        "math_vs_code_centroid_distance": _centroid_distance(math_vecs, code_vecs),
        "heterogeneous": bool(
            _centroid_distance(train_vecs, math_vecs) > 0.1
            and _centroid_distance(train_vecs, code_vecs) > 0.1
        ),
    }

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
        raise RuntimeError("M5.S4 conversion unexpectedly rejected document evidence")

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

    checks = {
        "domain_disjoint_holdout": bool(domain_disjoint),
        "holdout_heterogeneous": bool(probe["heterogeneous"]),
        "no_records_rejected": rejected == 0,
        "checkpoint_roundtrip": bool(checkpoint_roundtrip),
        "gate_passed": bool(gate.passed),
        "embedder_anchored": bool(embedder.revision != "unresolved"),
    }
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "train_corpus": str(args.train_corpus.as_posix()),
        "holdout_corpus": str(args.holdout_corpus.as_posix()),
        "embedder": embedder.to_payload(),
        "single_variable": "holdout domain heterogeneity (S3 Chinese-general -> S4 Math+Code_Agent); all else identical to S3",
        "data": {
            "train_domains": list(TRAIN_DOMAINS),
            "train_records": len(train_examples),
            "holdout_domains": holdout_domains,
            "holdout_records": len(holdout_examples),
            "retention_records": len(retention_examples),
            "domain_disjoint": domain_disjoint,
            "rejected": rejected,
        },
        "semantic_probe": probe,
        "metrics": report.to_payload(),
        "lifecycle": {
            "example_id": example_id,
            "status": lifecycle.status,
            "events": list(lifecycle.events),
        },
        "checks": checks,
        "technical_gate_all_passed": technical_gate_all_passed,
        "s3_comparison": {
            "s3_grounding_lesion_margin": 4e-06,
            "s4_grounding_lesion_margin": float(
                report.holdout_grounding_lesion_loss - report.holdout_loss_after
            ),
            "note": (
                "if the margin recovers materially above S3, the gate measured "
                "the right thing on too-easy data; if it stays near zero, the "
                "semantic embedding subsumes grounding and a hardened gate has "
                "a causal basis"
            ),
        },
        "boundary": "M5.S4 document internalization canary only; no provider execution, structural growth, or AGI claim",
        "resources": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "embed_seconds": embed_seconds,
            "embedded_texts": len(unique_texts),
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
                "semantic_probe": {
                    k: round(v, 4) if isinstance(v, float) else v for k, v in probe.items()
                },
                "grounding_lesion_margin": float(
                    report.holdout_grounding_lesion_loss - report.holdout_loss_after
                ),
                "holdout_loss_after": report.holdout_loss_after,
                "internalized_lesion": report.holdout_internalized_lesion_loss,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
