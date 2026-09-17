"""Characterize frozen H3.7 limitations; these are not capability tests."""

import torch

from taiji.response_plan_target import (
    FactorizedResponsePlanTargetEncoder,
    _response_chunks,
)


def test_h37_character_chunks_do_not_match_fixed_byte_phase():
    chunks = _response_chunks("中" * 24, slots=4, phase_stride=16)
    boundaries = [sum(map(len, chunks[:index])) for index in range(1, 4)]
    assert boundaries == [15, 30, 45]
    assert boundaries != [16, 32, 48]
    assert b"".join(chunks) == ("中" * 24).encode("utf-8")


def test_h37_positive_scalar_fit_cancels_under_slot_normalization():
    def encoder(scale):
        return FactorizedResponsePlanTargetEncoder(
            slots=4,
            slot_width=12,
            phase_stride=16,
            corpus_digest="a" * 64,
            parent_checkpoint_digest="b" * 64,
            fit_episode_ids=["synthetic-train"],
            slot_scale=torch.tensor(scale),
        )

    first = encoder([1.0, 1.0, 1.0, 1.0])
    second = encoder([0.01, 3.0, 100.0, 0.5])
    torch.testing.assert_close(
        first.encode_response("中" * 24),
        second.encode_response("中" * 24),
        atol=1e-7,
        rtol=1e-6,
    )
    assert first.target_digest != second.target_digest
