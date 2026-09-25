"""C6 Z1c probe: what does widening the closed corpus ``source_kind`` allowlist cost?

Zero product change and zero training.  PLAN-M6-01 decision 1 asks whether a
workbench capability may enter the evolution corpus as a NEW ``source_kind``, as
the existing ``verified_domain_material``, or not at all.  The proposal priced
that at "two closed allowlists".  This probe measures the real surface: which
code sites gate on the kind, what happens when an OLDER build re-reads a ledger
that contains a kind it does not know, and whether a contract-version bump is
what actually protects that read.  It creates no product type and writes nothing
outside its own report file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed_platform.evolution_adapters import ArtifactCorpusProjection  # noqa: E402
from seed_platform.evolution_ledger import EvolutionExperienceLedger  # noqa: E402
from seed_platform.source_registry import DeclarativeSourceRegistry  # noqa: E402
from taiji.evolution_experience import (  # noqa: E402
    EVOLUTION_CONTRACT_VERSION,
    EVOLUTION_EXPERIENCE_SOURCE_KINDS,
    EvolutionCorpusArtifact,
)
from taiji.internalization import content_digest  # noqa: E402

CANDIDATE_KIND = "workbench_artifact"
REUSED_KIND = "verified_domain_material"

# Sites that gate on the kind, gathered by pattern rather than by hand so the
# list cannot silently drift from the source.
SITE_PATTERNS = {
    "corpus_allowlist": r"EVOLUTION_CORPUS_SOURCE_KINDS|unsupported evolution corpus",
    "experience_allowlist": r"EVOLUTION_EXPERIENCE_SOURCE_KINDS|unsupported evolution "
    r"experience",
    "event_allowlist": r"_EVENT_SOURCE_KINDS",
    "internalization_allowlist": r"_ADMITTED_SOURCE_KINDS",
    "registry_naming_rule": r"_artifact\"\s+for item in corpus|artifact kind mismatch",
    "kind_dispatch": r"source_kind == \"",
}
SCANNED = (
    "taiji/evolution_experience.py",
    "taiji/artifact_internalization.py",
    "taiji/evolution_credit.py",
    "seed_platform/evolution_adapters.py",
    "seed_platform/source_registry.py",
)


def _artifact(source_kind: str, **extra: Any) -> EvolutionCorpusArtifact:
    return EvolutionCorpusArtifact(
        corpus_id=f"{source_kind}:seed.workbench.list",
        source_kind=source_kind,
        source_id="seed.workbench.list",
        source_version="1",
        source_digest=content_digest({"capability_id": "seed.workbench.list"}),
        unit_kind="knowledge",
        content={"capability_id": "seed.workbench.list"},
        partition="train",
        **extra,
    )


def _raises(call: Any) -> tuple[bool, str]:
    try:
        call()
    except Exception as error:  # noqa: BLE001 - the probe reports whatever fires
        return True, f"{type(error).__name__}: {error}"
    return False, ""


def _re_sign(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute the checkpoint digest so the kind check is what we observe."""
    body = {key: value for key, value in payload.items() if key != "checkpoint_digest"}
    payload["checkpoint_digest"] = content_digest(body)
    return payload


def _unknown_kind_replay() -> dict[str, Any]:
    payload = _ledger_payload(REUSED_KIND)
    payload["corpus"][0]["source_kind"] = CANDIDATE_KIND
    ok, message = _raises(lambda: EvolutionExperienceLedger.from_checkpoint(_re_sign(payload)))
    return {"injected": CANDIDATE_KIND, "rejected": ok, "error": message}


def _version_bump_replay() -> dict[str, Any]:
    payload = _ledger_payload(REUSED_KIND)
    payload["corpus"][0]["version"] = EVOLUTION_CONTRACT_VERSION + 1
    ok, message = _raises(lambda: EvolutionExperienceLedger.from_checkpoint(_re_sign(payload)))
    constructed_ok, constructed = _raises(
        lambda: _artifact(REUSED_KIND, version=EVOLUTION_CONTRACT_VERSION + 1)
    )
    return {
        "injected_version": EVOLUTION_CONTRACT_VERSION + 1,
        "replay_rejected": ok,
        "replay_error": message,
        "construct_rejected": constructed_ok,
        "construct_error": constructed,
    }


def _ledger_payload(source_kind: str) -> dict[str, Any]:
    ledger = EvolutionExperienceLedger()
    ledger.add_corpus(_artifact(source_kind))
    return ledger.checkpoint()


def _reuse_round_trip() -> dict[str, Any]:
    payload = _ledger_payload(REUSED_KIND)
    ok, message = _raises(lambda: EvolutionExperienceLedger.from_checkpoint(payload))
    return {"round_trips": not ok, "error": message}


class _StubWorkbenchAdapter:
    """Stand-in for the C6 adapter: emits registry corpus units of a chosen kind.

    ``DeclarativeSourceRegistry`` itself never checks the corpus kind on the way
    in -- only ``from_checkpoint`` does -- so this measures which half of the
    lifecycle each option fails in.
    """

    source_kind = "workbench"

    def __init__(self, corpus_kind: str) -> None:
        self.corpus_kind = corpus_kind

    def project(self, artifact: Any, *, partition: str = "train") -> Any:
        return ArtifactCorpusProjection(
            source_kind="workbench",
            source_id="seed-workbench-contract-v1",
            source_version="1",
            source_digest=content_digest({"capability_count": 1}),
            scope_id="",
            publisher="",
            corpus=(_artifact(self.corpus_kind),),
        )


def _registry_lifecycle(corpus_kind: str) -> dict[str, Any]:
    """Where does a workbench registry with this corpus kind break: register or replay?"""
    adapter = _StubWorkbenchAdapter(corpus_kind)
    registry = DeclarativeSourceRegistry(adapter)
    registered, register_error = _raises(lambda: registry.register({}))
    if registered:
        return {"fails_at": "register", "error": register_error, "replayed": False}
    payload = registry.checkpoint()
    replayed, replay_error = _raises(
        lambda: DeclarativeSourceRegistry.from_checkpoint(
            payload, adapter=_StubWorkbenchAdapter(corpus_kind)
        )
    )
    return {
        "fails_at": "replay" if replayed else "nothing",
        "error": replay_error,
        "registered": True,
        "replayed": not replayed,
    }


def _sites() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {name: [] for name in SITE_PATTERNS}
    for rel in SCANNED:
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            for name, pattern in SITE_PATTERNS.items():
                if re.search(pattern, line):
                    found[name].append(f"{rel}:{number}: {line.strip()}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    construct_ok, construct_error = _raises(lambda: _artifact(CANDIDATE_KIND))
    reading = {
        "probe": "taiji_c6_kind_widening_impact",
        "establishes": "which code sites gate on the corpus source_kind, and how an "
        "existing ledger replays when it carries a kind or a contract version its "
        "reader does not know; it establishes no admission and writes no product data",
        "outside_repo_writes": True,
        "contract": {
            "EVOLUTION_CONTRACT_VERSION": EVOLUTION_CONTRACT_VERSION,
            "corpus_allowlist": sorted(str(k) for k in _corpus_allowlist()),
            "experience_allowlist": list(EVOLUTION_EXPERIENCE_SOURCE_KINDS),
        },
        "E1_construct_candidate_kind": {
            "kind": CANDIDATE_KIND,
            "rejected": construct_ok,
            "error": construct_error,
        },
        "E2_replay_unknown_kind": _unknown_kind_replay(),
        "E3_replay_version_bump": _version_bump_replay(),
        "E4_reuse_round_trip": _reuse_round_trip(),
        "E5_registry_lifecycle_new_kind": _registry_lifecycle(CANDIDATE_KIND),
        "E6_registry_lifecycle_reused_kind": _registry_lifecycle(REUSED_KIND),
        "E7_sites": _sites(),
    }
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(reading, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name in (
        "E1_construct_candidate_kind",
        "E2_replay_unknown_kind",
        "E3_replay_version_bump",
        "E4_reuse_round_trip",
        "E5_registry_lifecycle_new_kind",
        "E6_registry_lifecycle_reused_kind",
    ):
        print(f"{name}: {json.dumps(reading[name], ensure_ascii=False)}")
    print("site counts: " + ", ".join(f"{k}={len(v)}" for k, v in reading["E7_sites"].items()))
    return 0


def _corpus_allowlist() -> tuple[str, ...]:
    """Read the closed corpus set back out of the guard it enforces."""
    kinds = []
    for kind in (
        "skill_artifact",
        "mcp_artifact",
        "client_plugin_artifact",
        REUSED_KIND,
        CANDIDATE_KIND,
        "totally_unknown_kind",
    ):
        rejected, _ = _raises(lambda: _artifact(kind))
        if not rejected:
            kinds.append(kind)
    return tuple(kinds)


if __name__ == "__main__":
    raise SystemExit(main())
