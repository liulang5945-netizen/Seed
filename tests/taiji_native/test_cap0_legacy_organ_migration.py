"""Live guard for the M2-2i identity-organ legacy migration (CAP-0 loader brief §5).

Why this file exists: until the migration landed, **no test anywhere** asserted what the organ
branch of ``Taiji.restore`` does.  The refusal was pinned only by contract tests that read committed
reports, which means widening that branch could not turn anything red -- the exact false-green the
brief warned about.  These tests call ``restore`` itself, in both directions: the legacy case must
load, and every refusal that still applies must keep firing.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from taiji import Taiji, TaijiConfig

REPO = Path(__file__).resolve().parents[2]
SEED_BETA = REPO / "checkpoints" / "seed_beta.pt"

#: The organ was introduced after these formats; they legitimately carry no organ payload.
LEGACY_FORMAT = "taiji-native-v8"
CURRENT_FORMAT = "taiji-native-v10"
ORGAN_MISSING = "enabled identity organ checkpoint payload is missing"


def _model() -> Taiji:
    return Taiji(TaijiConfig())


def _payload_with_organ() -> dict:
    model = _model()
    payload = model.checkpoint()
    assert isinstance(payload.get("identity_organ"), dict), "the fixture must carry a real organ"
    assert payload["identity_organ"].get("lineage"), "and a lineage the check can verify"
    return payload


def test_legacy_payload_without_organ_loads_and_keeps_a_fresh_organ() -> None:
    source = _payload_with_organ()
    legacy = copy.deepcopy(source)
    legacy["format"] = LEGACY_FORMAT
    legacy.pop("identity_organ")

    restored = _model()
    before = restored.identity_organ
    restored.restore(legacy)  # must not raise

    assert restored.identity_organ is not None, "the organ stays enabled"
    assert (
        restored.identity_organ is before
    ), "the instance is the model's own, not a fabricated one"


def test_current_format_without_organ_still_raises() -> None:
    """The migration is scoped to absence-in-legacy: a v10 file that lost its organ is still a bug."""

    payload = _payload_with_organ()
    assert payload["format"] == CURRENT_FORMAT
    payload.pop("identity_organ")
    with pytest.raises(ValueError, match=ORGAN_MISSING):
        _model().restore(payload)


def test_a_present_but_malformed_organ_payload_still_raises() -> None:
    payload = _payload_with_organ()
    payload["identity_organ"] = "not-a-mapping"
    with pytest.raises(ValueError, match=ORGAN_MISSING):
        _model().restore(payload)


def test_wrong_parent_lineage_still_fails_closed() -> None:
    """Absence was relaxed; a payload claiming the wrong parent is a different thing entirely."""

    payload = _payload_with_organ()
    payload["identity_organ"]["lineage"]["parent_checkpoint_digest"] = "0" * 64
    with pytest.raises(ValueError, match="lineage does not match Taiji core"):
        _model().restore(payload)


def test_a_lineage_from_an_identically_configured_core_is_accepted_by_design() -> None:
    """Documents what the check does *not* cover, so nobody reads it as coverage.

    First written as "grafting another model's organ must be rejected", this test failed -- because
    two ``Taiji`` instances built from the same default config are byte-deterministic, so their
    cores hash to the same digest and the graft is genuinely valid, not foreign state.  The
    protection that matters is the digest comparison itself (previous test), not the object identity.
    What this therefore does **not** prove: that an organ trained on a different core is refused --
    it is refused only insofar as that core hashes differently.
    """

    host = _payload_with_organ()
    host["identity_organ"] = _payload_with_organ()["identity_organ"]
    restored = _model()
    restored.restore(host)
    assert restored.identity_organ is not None


@pytest.mark.skipif(not SEED_BETA.exists(), reason="trained artifact is not in the repository")
def test_the_real_trained_checkpoint_loads_through_the_product_entry() -> None:
    """Exit criterion 1 of the brief, on the artifact that motivated it."""

    import torch

    envelope = torch.load(SEED_BETA, map_location="cpu", weights_only=False)
    substrate = envelope["substrate"]
    assert substrate["format"] == LEGACY_FORMAT
    assert "identity_organ" not in substrate, "the fixture is the v8 file this migration is about"

    model = Taiji(TaijiConfig.from_dict(dict(substrate["config"])))
    model.restore(substrate)
    assert model.tick == 16_000_000
    assert model.identity_organ is not None
