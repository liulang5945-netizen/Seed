"""The subset scheduler must preserve native learning and checkpoint state."""

import torch

from scripts.training import run_taiji_n1_plan168_isolated_train_fit as fit
from taiji import (
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    Taiji,
)


def make_trainer():
    torch.set_num_threads(1)
    episode = LanguageEpisode(
        episode_id="train-one",
        family_id="one",
        task_family="synthetic",
        split="train",
        user_input="问候",
        response="你好，测试。",
    )
    corpus = LanguageEpisodeCorpus((episode,), require_eval_splits=False)
    trainer = LanguageAlignmentTrainer(
        Taiji(),
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=8,
            response_plan_variant="factorized_v1",
            response_plan_slots=4,
            response_plan_phase_stride=4,
        ),
    )
    return trainer, corpus, episode


def test_subset_episode_matches_native_training_parameters_and_counters():
    trainer, corpus, episode = make_trainer()
    parent = trainer.checkpoint()
    parent_digest = fit.diagnostic.content_digest(parent)
    child = LanguageAlignmentTrainer.from_checkpoint(parent, corpus)
    trainer.train(epochs=1, max_episodes=1)
    fit.train_one(child, episode, 1)
    child._finish_developmental_epoch(1)
    assert child.global_step == trainer.global_step == len(episode.target_bytes)
    assert child.episode_count == trainer.episode_count == 1
    assert fit.diagnostic.content_digest(child.model.checkpoint()) == fit.diagnostic.content_digest(
        trainer.model.checkpoint()
    )
    assert fit.diagnostic.content_digest(parent) == parent_digest


def test_first_step_signature_is_repeatable_and_readonly():
    trainer, _, episode = make_trainer()
    before = trainer.checkpoint()["checkpoint_digest"]
    assert fit.signature(trainer, episode) == fit.signature(trainer, episode)
    assert trainer.checkpoint()["checkpoint_digest"] == before


def test_parameter_audit_does_not_count_dynamic_plan_state_as_learning():
    before = {
        "response_plan_readout": {"planner_weight": torch.zeros(1), "plan_state": torch.zeros(1)}
    }
    after = {
        "response_plan_readout": {"planner_weight": torch.zeros(1), "plan_state": torch.ones(1)}
    }
    assert fit.parameter_changes(before, after)["response_plan_readout"]["changed_tensors"] == 0
    after["response_plan_readout"]["planner_weight"] += 1
    assert fit.parameter_changes(before, after)["response_plan_readout"]["changed_tensors"] == 1
