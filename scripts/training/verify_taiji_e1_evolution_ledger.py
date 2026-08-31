"""E1 Gate: verify the unified Taiji evolution corpus and experience ledger."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_ledger import (  # noqa: E402
    EvolutionExperienceLedger,
    redact_sensitive_payload,
    workbench_outcome_to_experience,
)
from seed_platform.workbench import WorkbenchOutcome  # noqa: E402
from taiji.evolution_experience import EvolutionCorpusArtifact  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402


def _corpus() -> EvolutionCorpusArtifact:
    return EvolutionCorpusArtifact(
        corpus_id="fixture:skill.filesystem.read",
        source_kind="skill_artifact",
        source_id="skill.filesystem.read",
        source_version="1",
        source_digest=content_digest({"source": "skill.filesystem.read", "version": "1"}),
        unit_kind="procedure",
        content={"title": "filesystem read", "steps": ["validate", "read"]},
    )


def run_gate() -> dict[str, object]:
    ledger = EvolutionExperienceLedger()
    corpus = _corpus()
    ledger.add_corpus(corpus)
    admitted = ledger.admit_corpus(corpus.artifact_digest, admission_revision="e1-admission-1")

    outcome = WorkbenchOutcome(
        request_id="e1-request",
        intent_id="e1-intent",
        call_id="e1-call",
        capability_id="seed.workbench.read",
        snapshot_id="e1-snapshot",
        status="success",
        success=True,
        result={"path": "README.md", "text": "fixture output"},
        tick=1,
        mcp_registry_snapshot_id="a" * 64,
    )
    experience = workbench_outcome_to_experience(
        outcome,
        parent_checkpoint_digest="b" * 64,
        partition="train",
    )
    appended = ledger.append(experience).experience
    checkpoint = ledger.checkpoint()
    restored = EvolutionExperienceLedger.from_checkpoint(checkpoint)
    restored_tail_digest = restored.tail_event_digest
    continued = workbench_outcome_to_experience(
        WorkbenchOutcome(
            request_id="e1-request-2",
            intent_id="e1-intent",
            call_id="e1-call-2",
            capability_id="seed.workbench.read",
            snapshot_id="e1-snapshot",
            status="success",
            success=True,
            result={"path": "README.md", "text": "fixture output 2"},
            tick=2,
            mcp_registry_snapshot_id="a" * 64,
        ),
        parent_checkpoint_digest="b" * 64,
        partition="holdout",
    )
    continued = restored.append(continued).experience

    redacted, redaction_flags = redact_sensitive_payload({"api_key": "secret"})
    tampered = deepcopy(checkpoint)
    tampered["experiences"][0]["success"] = False
    try:
        EvolutionExperienceLedger.from_checkpoint(tampered)
    except ValueError as exc:
        tamper_rejected = "checkpoint digest mismatch" in str(exc)
    else:  # pragma: no cover - the gate must fail if tampering is accepted
        tamper_rejected = False

    train_corpus, train_experiences = ledger.training_view()
    checks = {
        "corpus_admitted": admitted.status == "admitted",
        "workbench_result_digest_only": "fixture output" not in appended.to_payload().__repr__(),
        "source_and_result_digests": bool(appended.source_digest and appended.result_digest),
        "redaction_required": redacted == {"api_key": "<redacted>"} and redaction_flags == ("api_key",),
        "checkpoint_restored": restored_tail_digest == checkpoint["tail_event_digest"],
        "checkpoint_continued": continued.event_sequence == 2
        and continued.previous_event_digest == appended.event_digest,
        "tamper_rejected": tamper_rejected,
        "training_view_partitioned": len(train_corpus) == 1 and len(train_experiences) == 1,
    }
    passed = all(checks.values())
    return {
        "gate": "taiji-e1-evolution-ledger",
        "status": "passed" if passed else "failed",
        "checks": checks,
        "ledger_revision": ledger.revision,
        "checkpoint_digest": checkpoint["checkpoint_digest"],
        "corpus_digest": corpus.artifact_digest,
        "experience_digest": appended.experience_digest,
        "tail_event_digest": ledger.tail_event_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "taiji_w7_e1_evolution_ledger_20260901.json",
    )
    args = parser.parse_args()
    result = run_gate()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
