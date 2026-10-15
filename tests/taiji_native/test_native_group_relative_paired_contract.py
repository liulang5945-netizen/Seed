"""All paired controls retain the original arm when applying the shared update rule."""

import pytest

from scripts.training import audit_taiji_n1_group_relative_paired as audit
from tests.taiji_native._native_relation_fixtures import paired_sources

runner = audit.runner


@pytest.fixture(scope="module")
def sources():
    audit.torch.set_num_threads(1)
    ready, source = paired_sources()
    return ready, source


@pytest.mark.parametrize("seed", audit.SEEDS)
@pytest.mark.parametrize("arm", audit.ARMS)
def test_unbounded_two_step_wrapper_matches_same_arm_bitwise(sources, seed, arm):
    ready, source = sources
    item = audit.SYNTHETIC
    second = runner.core.renaming.second_question(item)
    model, optimizer = runner.fresh(source, seed, ready)
    expected = [
        runner.train_step(model, optimizer, item, f"neutral69-{i}", arm, second) for i in range(2)
    ]
    fingerprint = runner.paired.fingerprint(model.bundle(optimizer))
    model, optimizer = runner.fresh(source, seed, ready)
    for i in range(2):
        loss, measurements = audit.update.train_step(
            model, optimizer, item, f"neutral69-{i}", arm, second, relative_limit=1e9, audit=True
        )
        assert loss == expected[i]
        assert not any(m["clipped"] for m in measurements.values())
    assert runner.paired.fingerprint(model.bundle(optimizer)) == fingerprint


def test_cut_credit_zero_and_strict_same_arm_next_step_recovery(sources, tmp_path):
    ready, source = sources
    model, optimizer = runner.fresh(source, 13, ready)
    rng = audit.torch.get_rng_state().clone()
    saved = audit.save(model, optimizer, tmp_path / "cut.pt", 13, "cut")
    assert audit.torch.equal(rng, audit.torch.get_rng_state())
    expected = saved["expected_signature"]
    assert expected["writer_unchanged"] and expected["writer_gradient_l2"] == 0
    assert expected["relation_head_changed"]
    assert audit.restore(tmp_path / "cut.pt")["signature"] == expected
    for changed in (
        dict(saved, update_policy=None),
        dict(saved, arm="bad"),
        dict(saved, completed_updates=2),
    ):
        with pytest.raises(ValueError, match="marker|phase|identity"):
            audit.load(changed)
