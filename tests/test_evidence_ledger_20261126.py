"""Tests for the evidence-credibility ledger (zero training)."""

from __future__ import annotations

import sys

sys.path.insert(0, r"E:\Seed")
import pytest

from taiji.evidence_ledger import ACCEPT_THRESHOLD, CRED_PRIOR, CredibilityLedger


def test_high_credibility_accepts() -> None:
    led = CredibilityLedger()
    r = led.add("door_locked", +1, "owner")
    assert r["status"] == "accepted"
    assert led.get("door_locked").polarity == 1


def test_low_credibility_does_not_flip() -> None:
    led = CredibilityLedger()
    led.add("door_locked", +1, "owner")  # 0.90 accept
    r = led.add("door_locked", -1, "rumor")  # 0.15 oppose
    state = led.get("door_locked")
    assert state.polarity == 1
    assert state.status == "accepted"
    assert not r["corrected"]


def test_high_credibility_contradiction_corrects() -> None:
    led = CredibilityLedger()
    led.add("door_locked", +1, "forum")  # 0.30 -> pending
    led.add("door_locked", +1, "sensor")  # 0.60 + 0.30 -> accepted
    led.add("door_locked", -1, "owner")  # 0.90 vs 0.90 -> posterior 0.5 pending, not accepted
    state = led.get("door_locked")
    assert state.status == "pending"
    r2 = led.add("door_locked", -1, "owner")  # second owner evidence flips
    assert r2["corrected"] is True
    assert led.get("door_locked").polarity == -1
    corr = led.corrections()
    assert len(corr) == 1 and corr[0]["claim_id"] == "door_locked"


def test_tie_stays_pending() -> None:
    led = CredibilityLedger()
    led.add("valve_open", +1, "sensor")
    led.add("valve_open", -1, "sensor")
    state = led.get("valve_open")
    assert state.status == "pending"
    assert abs(state.posterior - 0.5) < 1e-9


def test_provenance_and_determinism() -> None:
    def build() -> CredibilityLedger:
        led = CredibilityLedger()
        led.add("c1", +1, "manual")
        led.add("c1", -1, "forum")
        led.add("c2", -1, "sensor")
        return led

    a, b = build(), build()
    assert a.digest() == b.digest()
    srcs = [e.source for e in a.evidence]
    assert srcs == ["manual", "forum", "sensor"]
    assert a.clock == 3


def test_invalid_inputs() -> None:
    led = CredibilityLedger()
    with pytest.raises(ValueError):
        led.add("c", 0, "owner")
    with pytest.raises(ValueError):
        led.add("", 1, "owner")
    with pytest.raises(KeyError):
        led.get("missing")
    assert CRED_PRIOR["owner"] > CRED_PRIOR["rumor"]
    assert 0.0 < ACCEPT_THRESHOLD < 1.0
