"""Offline intervention identity, dimension guards and fixed denominator."""

import pytest
import torch

from scripts.training import audit_taiji_n1_cross_read as audit


def test_intervention_native_mirror_and_identity():
    torch.set_num_threads(1)
    model = audit.source.ResidualQuestionEvidenceCandidate()
    views = audit.prior.checked_views()
    before = audit.paired.fingerprint(model.bundle())
    for i in range(4):
        for view in ("original", "rename"):
            item = views[view][i]
            native = audit.measure(model, item, f"native-{view}-{i}")
            mirror = audit.measure(model, item, f"mirror-{view}-{i}", mirror_native=True)
            assert torch.equal(native["logits"], mirror["logits"])
            for condition in ("address", "values", "both"):
                replay = audit.measure(
                    model, item, f"replay-{view}-{i}-{condition}", native, condition
                )
                assert torch.equal(native["logits"], replay["logits"])
    assert before == audit.paired.fingerprint(model.bundle())
    with pytest.raises(ValueError, match="train"):
        audit.measure(model, dict(views["original"][0], split="dev"), "bad")


def test_position_and_dimension_replay_guards():
    model = audit.source.ResidualQuestionEvidenceCandidate()
    item = audit.prior.checked_views()["original"][0]
    reference = audit.measure(model, item, "reference")
    with pytest.raises(ValueError, match="position-matched"):
        audit.measure(model, item, "bad-position", dict(reference, positions=()), "both")
    with pytest.raises(ValueError, match="probability"):
        audit.measure(
            model, item, "bad-address", dict(reference, addresses=[torch.ones(1)]), "address"
        )
    with pytest.raises(ValueError, match="dimensions"):
        audit.measure(model, item, "bad-values", dict(reference, values=torch.ones(1)), "values")
    assert "step" not in model.__dict__
    assert not model._observations


def test_full_unique_denominator():
    keys = [
        (model, view, item["id"], condition)
        for model in ("41", "43")
        for view in ("original", "pair")
        for item in audit.prior.checked_views()[view]
        for condition in ("reference", *audit.CONDITIONS)
    ]
    audit.validate_keys(keys)
    with pytest.raises(ValueError, match="2880"):
        audit.validate_keys(keys[:-1])
    with pytest.raises(ValueError, match="2880"):
        audit.validate_keys(keys[:-1] + [keys[0]])
