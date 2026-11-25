"""Evidence-credibility ledger with correction tracking (`taiji-evidence-ledger-v1`).

Zero-training facility for the "依据证据来源及其可信程度核验和纠正认识" clause. It keeps an
append-only evidence log, aggregates per-source credibility into a posterior per claim, and
records explicit corrections when a credible source contradicts an accepted claim. Nothing is
overwritten silently: every decision and correction is provenance-logged and digestable.

Credibility priors are declared constants (a control/engineering prototype); a learned
credibility model is a later training line.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

SCHEMA = "taiji-evidence-ledger-v1"
CRED_PRIOR: dict[str, float] = {
    "owner": 0.90,
    "manual": 0.80,
    "sensor": 0.60,
    "forum": 0.30,
    "rumor": 0.15,
}
ACCEPT_THRESHOLD = 0.60


@dataclass
class Evidence:
    id: int
    claim_id: str
    polarity: int
    source: str
    credibility: float
    clock: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "polarity": self.polarity,
            "source": self.source,
            "credibility": round(self.credibility, 4),
            "clock": self.clock,
        }


@dataclass
class ClaimState:
    claim_id: str
    status: str = "pending"
    polarity: int = 0
    posterior: float = 0.0
    last_accepted_polarity: int = 0
    corrections: list[dict[str, Any]] = field(default_factory=list)


class CredibilityLedger:
    def __init__(self) -> None:
        self.evidence: list[Evidence] = []
        self.claims: dict[str, ClaimState] = {}
        self.clock = 0

    # ------------------------------------------------------------------ add
    def add(
        self,
        claim_id: str,
        polarity: int,
        source: str,
        *,
        credibility: float | None = None,
    ) -> dict[str, Any]:
        if polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        if not claim_id:
            raise ValueError("claim_id must be non-empty")
        cred = float(credibility if credibility is not None else CRED_PRIOR.get(source, 0.10))
        ev = Evidence(len(self.evidence), claim_id, polarity, source, cred, self.clock)
        self.evidence.append(ev)
        self.clock += 1

        plus = sum(
            e.credibility for e in self.evidence if e.claim_id == claim_id and e.polarity == 1
        )
        minus = sum(
            e.credibility for e in self.evidence if e.claim_id == claim_id and e.polarity == -1
        )
        total = plus + minus
        posterior_plus = plus / total if total > 0 else 0.0
        posterior_minus = minus / total if total > 0 else 0.0
        winner = 1 if plus >= minus else -1
        posterior = posterior_plus if winner == 1 else posterior_minus
        state = self.claims.setdefault(claim_id, ClaimState(claim_id))
        corrected = False
        state.posterior = round(posterior, 4)
        state.polarity = winner
        state.status = "accepted" if (total > 0 and posterior >= ACCEPT_THRESHOLD) else "pending"
        if state.status == "accepted" and state.last_accepted_polarity not in (0, winner):
            state.corrections.append(
                {
                    "at_clock": self.clock - 1,
                    "from_polarity": state.last_accepted_polarity,
                    "to_polarity": winner,
                    "trigger_source": source,
                    "trigger_credibility": round(cred, 4),
                    "posterior": round(posterior, 4),
                }
            )
            corrected = True
        if state.status == "accepted":
            state.last_accepted_polarity = winner
        return {
            "evidence": ev.as_dict(),
            "claim_id": claim_id,
            "status": state.status,
            "polarity": state.polarity,
            "posterior": state.posterior,
            "corrected": corrected,
        }

    # ----------------------------------------------------------------- views
    def get(self, claim_id: str) -> ClaimState:
        if claim_id not in self.claims:
            raise KeyError(f"unknown claim {claim_id!r}")
        return self.claims[claim_id]

    def corrections(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, state in self.claims.items():
            for c in state.corrections:
                out.append({"claim_id": cid, **c})
        return out

    def digest(self) -> str:
        payload = {
            "schema": SCHEMA,
            "threshold": ACCEPT_THRESHOLD,
            "evidence": [e.as_dict() for e in self.evidence],
            "claims": {
                cid: {
                    "status": s.status,
                    "polarity": s.polarity,
                    "posterior": s.posterior,
                    "corrections": s.corrections,
                }
                for cid, s in sorted(self.claims.items())
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
