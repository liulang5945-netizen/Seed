"""M5.S2 canary: internalize real sourced document experiences.

First M5 periphery step (per the unfreeze order in
``03_CURRENT_EXECUTION.md`` §6): documents become sourced experiences.  The
UltraData conversion corpus (real open-source records, SHA-256 anchored,
document-grounding provenance) is converted into grounded evidence with a
deterministic feature extractor, then pushed through the S1 native
internalization pipeline with a **domain-level held-out source**: training
uses Knowledge + IF records only, and the holdout is an entirely unseen
source domain (Chinese-general).

Gate semantics mirror S1 (external sufficiency, internalization necessity,
grounding necessity, checkpoint recoverability, retention) plus a
domain-disjointness assertion.  Reward is a placeholder constant in this
canary - real outcomes require the MCP/execution loop and stay out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

REPORT_FORMAT = "taiji-m5-s2-document-internalization-v1"
FEATURE_SET_VERSION = "m5s2-text-stats-v1"
FEATURE_DIM = 12
HOLDOUT_DOMAIN = "Chinese-general"
TRAIN_DOMAINS = ("Knowledge", "IF")
TRAIN_PER_DOMAIN = 3000
HOLDOUT_COUNT = 2000
RETENTION_COUNT = 2000


def extract_features(text: str) -> tuple[float, ...]:
    """Deterministic versioned text statistics (no learned components)."""
    encoded = text.encode("utf-8")
    n = max(1, len(encoded))
    questions = text.count("？") + text.count("?")
    newlines = text.count("\n")
    unique = len(set(text)) / max(1, len(text))
    digits = sum(ch.isdigit() for ch in text) / max(1, len(text))
    cjk = sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in text) / max(1, len(text))
    ascii_ratio = sum(ord(ch) < 128 for ch in text) / max(1, len(text))
    answer_mark = 1.0 if "答：" in text else 0.0
    question_mark = 1.0 if text.startswith("问：") else 0.0
    punctuation = sum(ch in "，。、；：？！）】》" for ch in text) / max(1, len(text))
    sentences = max(1, newlines + punctuation)
    mean_sentence = len(text) / sentences
    return (
        math.log2(1 + n) / 16.0,
        questions / max(1, len(text)),
        newlines / max(1, len(text)),
        unique,
        digits,
        cjk,
        ascii_ratio,
        answer_mark,
        question_mark,
        punctuation,
        math.log2(1 + mean_sentence) / 12.0,
        1.0,
    )


def _load_domain_records(corpus: Path) -> dict[str, list[tuple[str, str]]]:
    """Load (record_digest, text) pairs grouped by domain.

    Domain assignment replays the deterministic round-robin recorded in the
    M4.R12 conversion manifest, so the split is anchored to the manifest
    rather than trusting file order.
    """
    manifest_path = (
        PROJECT_ROOT / "reports" / "taiji_m4r12_ultradata_conversion_manifest_20260909.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_digest = str(manifest["output_sha256"])
    lines = corpus.read_text(encoding="utf-8").splitlines()
    # The M4.R12 manifest records a raw sha256 over the newline-joined
    # lines, not a canonical content digest; match that exact convention.
    corpus_digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    if corpus_digest != expected_digest:
        raise ValueError("corpus digest does not match the M4.R12 manifest")

    counts: dict[str, int] = dict(manifest["records_per_domain"])
    domains: list[str] = list(manifest["domains"])
    order: list[str] = []
    cursors = {d: 0 for d in domains}
    while len(order) < int(manifest["total_records"]):
        for domain in domains:
            if cursors[domain] < counts[domain]:
                order.append(domain)
                cursors[domain] += 1

    by_domain: dict[str, list[tuple[str, str]]] = {d: [] for d in domains}
    for domain, line in zip(order, lines, strict=True):
        record = json.loads(line)
        text = str(record["text"])
        by_domain[domain].append((content_digest(text), text))
    return by_domain


def _evidence(domain: str, record_digest: str, text: str) -> GroundedOutcomeEvidence:
    features = torch.tensor(extract_features(text), dtype=torch.float32)
    short = record_digest[:16]
    affordance = WorldAffordance(
        affordance_id=f"affordance:ultradata:{short}",
        action_kind="document-experience",
        actor_id="ultradata-corpus",
        target_id=f"document:{domain}:{short}",
        features=features,
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
        capability_snapshot_digest="capability-sha256:m5s2",
        parent_checkpoint_id="checkpoint:m5s2-parent",
        owner_id="taiji:document-experience",
        reward_terms={"document_ingested": 1.0},
        world_digest=f"document-sha256:{record_digest}",
    )


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
        default=PROJECT_ROOT / "reports" / "taiji_m5_s2_document_internalization_20260909.json",
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

    converter = InternalizationConverter(seed=17, replay_budget=8000)
    ledger = InternalizationLedger(converter=converter)
    train_examples = []
    rejected = 0
    for domain, record_digest, text in train_records:
        result = ledger.ingest(_evidence(domain, record_digest, text))
        if result.example is None:
            rejected += 1
        else:
            train_examples.append(result.example)
    holdout_examples = []
    for domain, record_digest, text in holdout_records:
        result = converter.convert(_evidence(domain, record_digest, text))
        if result.example is not None:
            holdout_examples.append(result.example)
    retention_examples = []
    for domain, record_digest, text in retention_records:
        result = converter.convert(_evidence(domain, record_digest, text))
        if result.example is not None:
            retention_examples.append(result.example)
    if not train_examples or not holdout_examples or not retention_examples:
        raise RuntimeError("M5.S2 conversion unexpectedly rejected document evidence")

    learner = InternalizedFeatureLearner(feature_dim=FEATURE_DIM, learning_rate=0.5)
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
        "no_records_rejected": rejected == 0,
        "checkpoint_roundtrip": bool(checkpoint_roundtrip),
        "gate_passed": bool(gate.passed),
        "feature_set_versioned": bool(FEATURE_SET_VERSION),
    }
    technical_gate_all_passed = all(bool(v) for v in checks.values())

    payload = {
        "format": REPORT_FORMAT,
        "version": 1,
        "generated_at_epoch": int(time.time()),
        "status": "passed" if technical_gate_all_passed else "failed",
        "can_promote": False,
        "corpus": str(args.corpus.as_posix()),
        "feature_set": {
            "version": FEATURE_SET_VERSION,
            "dim": FEATURE_DIM,
            "deterministic": True,
        },
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
        "lifecycle": {
            "example_id": example_id,
            "status": lifecycle.status,
            "events": list(lifecycle.events),
        },
        "checks": checks,
        "technical_gate_all_passed": technical_gate_all_passed,
        "gate_criterion": (
            "native consolidation improves an unseen-source-domain holdout "
            "while feature, grounding, retention, and checkpoint controls "
            "remain causal; reward is a placeholder constant pending the "
            "MCP/execution loop"
        ),
        "boundary": "M5.S2 document internalization canary only; no provider execution, structural growth, or AGI claim",
        "resources": {"total_elapsed_seconds": time.perf_counter() - started},
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if technical_gate_all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
