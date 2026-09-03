from __future__ import annotations

from copy import deepcopy

from scripts.training.eval_taiji_m1_62_learning_data_contract import (
    _curriculum,
    _write_train,
)
from scripts.training.eval_taiji_m1_63_identity_organ_promotion import (
    _promotion_config,
    _punished_curriculum,
    _reward_contrast,
)
from scripts.training.eval_taiji_m1_identity_organ_canary import (
    _cue_pattern,
    _query,
    run_canary,
)
from scripts.training.train_taiji_memory import _memory_config
from taiji import Taiji, TaijiConfig
from taiji.identity_organ import (
    IDENTITY_ORGAN_BOUND_PROVENANCE,
    IDENTITY_ORGAN_UNBOUND_PROVENANCE,
)
from taiji.internalization import content_digest


def _enabled_config(seed: int = 11) -> TaijiConfig:
    values = _memory_config(seed).to_dict()
    values.update(
        {
            "identity_organ_enabled": True,
            "identity_organ_capacity": 8,
        }
    )
    return TaijiConfig.from_dict(values)


def _disabled_config(config: TaijiConfig) -> TaijiConfig:
    return TaijiConfig.from_dict({**config.to_dict(), "identity_organ_enabled": False})


def _write(model: Taiji, cue: int, action: int, outcome: int) -> None:
    model.reset_dynamics(episode_id=f"m1-63-write-{int(cue)}")
    model.observe(model.config.boundary_symbol, learn=False, learn_motor=False)
    model.observe(int(cue), learn=False, learn_motor=False)
    model.act((int(action),), sample=False)
    model.settle_action(1.0, learn=False, learn_memory=True)
    model.observe(int(outcome), learn=False, learn_motor=False)


def test_identity_organ_write_is_reward_modulated_not_reward_blind() -> None:
    # The organ is on the default path, so every settled action reaches it,
    # including punished ones.  A reward-blind write binds a punished action as
    # strongly as a rewarded one, which is what pinned the binary-cue task in
    # test_active_environment at chance.  This locks the mechanism rather than
    # the threshold: a punished write must move the action head in the opposite
    # direction from a rewarded write on the same cue and action.
    config = TaijiConfig(seed=311)
    rewarded = Taiji(config, episode_id="m1-63-reward-sign")
    punished = Taiji(config, episode_id="m1-63-reward-sign")

    pattern = _cue_pattern(rewarded, 65)
    baseline = float(
        rewarded.identity_organ.recall(pattern).action_probabilities[48].item()
    )

    rewarded.identity_organ.learn(pattern, 48, outcome_symbol=43, reward=1.0)
    punished.identity_organ.learn(pattern, 48, outcome_symbol=43, reward=-1.0)

    after_reward = float(
        rewarded.identity_organ.recall(pattern).action_probabilities[48].item()
    )
    after_punishment = float(
        punished.identity_organ.recall(pattern).action_probabilities[48].item()
    )

    assert after_reward > baseline
    assert after_punishment < baseline
    # Both trials are real episodes, so both must own a cue slot: only the
    # value heads are reward-gated, never cue identity itself.
    assert rewarded.identity_organ.bank.occupied_count == 1
    assert punished.identity_organ.bank.occupied_count == 1
    assert rewarded.identity_organ.punished_write_count == 0
    assert punished.identity_organ.punished_write_count == 1


def test_identity_organ_default_reward_reproduces_unmodulated_write() -> None:
    # The write baseline defaults to 0.0 so a reward of 1.0 gives a modulation
    # of exactly 1.0.  That keeps every pre-existing checkpoint and every M1-63
    # evaluator record bit-comparable, so the promotion evidence is not silently
    # invalidated by adding reward gating.
    config = TaijiConfig(seed=311)
    implicit = Taiji(config, episode_id="m1-63-reward-default")
    explicit = Taiji(config, episode_id="m1-63-reward-default")

    assert config.identity_organ_write_baseline == 0.0

    pattern = _cue_pattern(implicit, 66)
    implicit.identity_organ.learn(pattern, 49, outcome_symbol=45)
    explicit.identity_organ.learn(pattern, 49, outcome_symbol=45, reward=1.0)

    assert content_digest(
        implicit.identity_organ.to_payload(parent_checkpoint_digest="x")
    ) == content_digest(
        explicit.identity_organ.to_payload(parent_checkpoint_digest="x")
    )


def test_identity_organ_can_be_explicitly_disabled_for_ablation() -> None:
    # M1-63 promoted the organ onto the default path, so "off" is no longer the
    # default.  The disabled path still has to hold: every A/B control and every
    # lesion arm in this repo rebuilds its control from an explicitly disabled
    # configuration instead of stripping a payload out of a checkpoint.
    model = Taiji(_disabled_config(_memory_config(11)))
    step = model.observe(65, learn=False, learn_motor=False)

    assert model.identity_organ is None
    assert step.identity_recall is None
    assert "identity_organ" not in model.checkpoint()


def test_default_config_carries_the_identity_organ_as_a_first_class_organ() -> None:
    config = TaijiConfig(seed=311)
    model = Taiji(config, episode_id="m1-63-default")

    assert config.identity_organ_enabled is True
    assert model.identity_organ is not None
    assert model.identity_organ.capacity == config.identity_organ_capacity
    assert "identity_organ" in model.checkpoint()
    assert model.parameter_count() == config.planned_active_parameter_count


def test_identity_organ_budget_is_planned_before_allocation() -> None:
    # The promotion is only honest if the parameter increment is a planned line
    # item rather than an incidental side effect of a flag, so the planned delta
    # between the two configurations must equal the organ's own parameter count.
    config = TaijiConfig(seed=311)
    disabled = _disabled_config(config)
    model = Taiji(config)

    planned_delta = (
        config.planned_active_parameter_count - disabled.planned_active_parameter_count
    )

    assert planned_delta == model.identity_organ.parameter_count
    assert model.parameter_count() == config.planned_active_parameter_count
    assert Taiji(disabled).parameter_count() == disabled.planned_active_parameter_count


def test_default_identity_organ_is_trainable_checkpointable_and_lesionable() -> None:
    config = TaijiConfig(seed=311)
    model = Taiji(config, episode_id="m1-63-default-organ")

    unbound = _query(model, 65)
    assert unbound.identity_recall.used is False
    assert unbound.identity_recall.provenance == IDENTITY_ORGAN_UNBOUND_PROVENANCE

    _write(model, 65, 48, 43)
    _write(model, 66, 49, 44)
    assert model.identity_organ.write_count == 2

    bound = _query(model, 66)
    assert bound.identity_recall.used is True
    assert bound.identity_recall.provenance == IDENTITY_ORGAN_BOUND_PROVENANCE
    assert int(bound.identity_recall.action_probabilities.argmax()) == 49
    assert int(bound.identity_recall.outcome_probabilities.argmax()) == 44

    checkpoint = deepcopy(model.checkpoint())
    restored = Taiji.from_checkpoint(checkpoint)
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)

    recovered = _query(restored, 66)
    assert recovered.identity_recall.used is True
    assert int(recovered.identity_recall.action_probabilities.argmax()) == 49

    restored.identity_organ.lesion()
    lesioned = _query(restored, 66)
    assert lesioned.identity_recall.used is False
    assert lesioned.identity_recall.provenance == IDENTITY_ORGAN_UNBOUND_PROVENANCE


def test_identity_organ_checkpoint_lineage_and_roundtrip_are_exact() -> None:
    model = Taiji(_enabled_config())
    checkpoint = deepcopy(model.checkpoint())
    restored = Taiji.from_checkpoint(checkpoint)

    assert checkpoint["identity_organ"]["format"] == "taiji-native-identity-organ-v2"
    assert checkpoint["identity_organ"]["lineage"]["organ_id"] == "cue-identity-route"
    assert restored.identity_organ is not None
    assert content_digest(restored.checkpoint()) == content_digest(checkpoint)


def test_evaluator_write_path_threads_example_reward_to_the_organ() -> None:
    # The organ-level canary above proves ``learn(reward=-1.0)`` anti-binds, but
    # that was never the defect.  The defect was three layers up: the reward sat
    # on MemoryLearningExample and was dropped by _episode -> _write_episode, so
    # no evaluator could construct a punished write at all and the M1-63 gate
    # went 15/15 green while blind.  This test asserts the *transport*, through
    # the real evaluator write path, not a re-implementation of it.
    rewarded_course = _curriculum("stable_key", deterministic=True)
    punished_course = _punished_curriculum(rewarded_course, "punished_key")

    rewarded = Taiji(
        _promotion_config(11, enabled=True, capacity=None), episode_id="reward-transport"
    )
    punished = Taiji(
        _promotion_config(11, enabled=True, capacity=None), episode_id="reward-transport"
    )

    written = _write_train(rewarded, rewarded_course)
    assert _write_train(punished, punished_course) == written

    # The punishment arrived: every train write was modulated negatively.  If
    # reward were dropped anywhere on the path this count would be zero.
    assert punished.identity_organ.punished_write_count == written
    assert rewarded.identity_organ.punished_write_count == 0

    # Cue identity is not reward-gated, so the punished run must own exactly the
    # same slots.  Without this, "did not bind" is indistinguishable from
    # "refused to write", and the gate would be vacuous in the other direction.
    assert punished.identity_organ.write_count == written
    assert punished.identity_organ.skipped_write_count == 0
    assert (
        punished.identity_organ.bank.occupied_count
        == rewarded.identity_organ.bank.occupied_count
    )

    # And the two runs must actually differ, or reward is decorative.
    assert content_digest(
        punished.identity_organ.to_payload(parent_checkpoint_digest="x")
    ) != content_digest(
        rewarded.identity_organ.to_payload(parent_checkpoint_digest="x")
    )


def test_punished_course_is_derived_so_only_reward_can_explain_the_difference() -> None:
    # The punished course is derived from the rewarded one via dataclasses.replace
    # rather than hand-authored, so "these differ only in reward" is structural
    # instead of eyeballed.  This locks that claim mechanically: any future edit
    # that lets a second field drift turns the experiment into a confound and
    # must fail here rather than silently producing a wrong causal conclusion.
    rewarded_course = _curriculum("stable_key", deterministic=True)
    punished_course = _punished_curriculum(rewarded_course, "punished_key")
    contrast = _reward_contrast(rewarded_course, punished_course)

    assert contrast["train_differing_fields"] == ["reward"]
    assert contrast["differs_only_by_reward"] is True
    assert contrast["rewarded_train_all_at_pin"] is True
    assert contrast["punished_train_all_at_pin"] is True
    assert contrast["query_partitions_reward_unchanged"] is True
    assert "reward" in contrast["graded_fields_compared"]

    # Derivation must not smuggle in a shared id: the curriculum contract
    # enforces uniqueness per course, and a collision across courses would make
    # the two runs share episode provenance.
    rewarded_ids = {example.example_id for example in rewarded_course.train}
    punished_ids = {example.example_id for example in punished_course.train}
    assert rewarded_ids.isdisjoint(punished_ids)
    assert punished_course.digest != rewarded_course.digest


def test_m1_identity_organ_canary_passes_one_seed() -> None:
    result = run_canary(seeds=(11,))
    assert result["canary_passed"] is True
    record = result["records"]["identity_organ"][0]
    assert record["provenance"]["final_action_owner"] == "ByteMotor"
    assert record["no_change"]["organ_digest_unchanged"] is True
    assert record["checkpoint"]["fresh_process_checkpoint_digest_matches"] is True
