from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from taiji import (
    LANGUAGE_ALIGNMENT_CREDIT_EVALUATION,
    LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION,
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    Taiji,
    TaijiConfig,
    checkpoint_roundtrip_preflight,
    paired_checkpoint_diagnostic,
)


def _corpus(tmp_path: Path) -> LanguageEpisodeCorpus:
    episodes = (
        LanguageEpisode(
            episode_id="train-1",
            family_id="train-family-1",
            task_family="fact",
            split="train",
            context="北京是中国的首都。",
            user_input="中国的首都是哪里？",
            response="北京。",
            required_terms=("北京",),
        ),
        LanguageEpisode(
            episode_id="train-2",
            family_id="train-family-2",
            task_family="instruction",
            split="train",
            context="",
            user_input="请简短回答：你好。",
            response="你好。",
            required_terms=("你好",),
        ),
        LanguageEpisode(
            episode_id="dev-1",
            family_id="dev-family-1",
            task_family="fact",
            split="dev",
            context="水在标准大气压下约一百度沸腾。",
            user_input="水的沸点约是多少？",
            response="约一百度。",
            required_terms=("一百度",),
        ),
        LanguageEpisode(
            episode_id="final-1",
            family_id="final-family-1",
            task_family="unknown",
            split="final",
            context="",
            user_input="请回答一个没有资料支持的问题。",
            response="我不知道。",
            unknown_policy="say_unknown",
            required_terms=("不知道",),
        ),
    )
    source = tmp_path / "episodes.jsonl"
    source.write_text(
        "\n".join(json.dumps(item.to_payload(), ensure_ascii=False) for item in episodes) + "\n",
        encoding="utf-8",
    )
    return LanguageEpisodeCorpus.from_jsonl([source])


def test_language_episode_serialization_has_explicit_response_boundary(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    episode = corpus.for_split("train")[0]

    assert "<|user|>" in episode.prefix_text
    assert "<|policy|>" in episode.prefix_text
    assert "<|policy|>\nanswer\n" in episode.prefix_text
    assert episode.prefix_text.endswith("<|assistant|>\n")
    assert episode.target_text.endswith("<|end|>\n")
    assert episode.full_bytes == episode.prompt_bytes + episode.target_bytes
    assert corpus.sample_counts == {"train": 2, "dev": 1, "final": 1, "retention": 0}
    assert corpus.manifest()["digest"] == corpus.digest


def test_language_episode_family_cannot_cross_splits() -> None:
    train = LanguageEpisode(
        episode_id="train-1",
        family_id="same-family",
        task_family="fact",
        split="train",
        user_input="问题一",
        response="回答一",
    )
    dev = LanguageEpisode(
        episode_id="dev-1",
        family_id="same-family",
        task_family="fact",
        split="dev",
        user_input="问题二",
        response="回答二",
    )

    with pytest.raises(ValueError, match="crosses splits"):
        LanguageEpisodeCorpus((train, dev))


def test_utf8_constraint_enforces_first_continuation_bounds() -> None:
    assert 0x80 not in LanguageAlignmentTrainer._utf8_allowed(2, 0xE0)
    assert 0xA0 in LanguageAlignmentTrainer._utf8_allowed(2, 0xE0)
    assert 0xA0 not in LanguageAlignmentTrainer._utf8_allowed(2, 0xED)
    assert 0x9F in LanguageAlignmentTrainer._utf8_allowed(2, 0xED)
    assert 0x8F not in LanguageAlignmentTrainer._utf8_allowed(3, 0xF0)
    assert 0x90 in LanguageAlignmentTrainer._utf8_allowed(3, 0xF0)
    assert 0x8F in LanguageAlignmentTrainer._utf8_allowed(3, 0xF4)
    assert 0x90 not in LanguageAlignmentTrainer._utf8_allowed(3, 0xF4)


def test_sequence_criteria_keep_required_forbidden_and_unknown_rules_separate() -> None:
    episode = LanguageEpisode(
        episode_id="criteria-1",
        family_id="criteria-family",
        task_family="fact",
        split="dev",
        user_input="问题",
        response="答案",
        unknown_policy="answer",
        required_terms=("答案",),
        forbidden_terms=("错误",),
    )

    passing = LanguageAlignmentTrainer._sequence_evaluation(
        episode,
        "答案",
        valid_utf8=True,
        no_replacement=True,
        boundary_present=True,
        stop_reason="end_marker",
    )
    failing = LanguageAlignmentTrainer._sequence_evaluation(
        episode,
        "答案，错误",
        valid_utf8=True,
        no_replacement=True,
        boundary_present=True,
        stop_reason="end_marker",
    )
    unknown_episode = LanguageEpisode(
        episode_id="criteria-unknown-1",
        family_id="criteria-unknown-family",
        task_family="unknown",
        split="final",
        user_input="未知问题",
        response="不知道",
        unknown_policy="say_unknown",
        unknown_markers=("不知道",),
    )
    unknown = LanguageAlignmentTrainer._sequence_evaluation(
        unknown_episode,
        "不知道",
        valid_utf8=True,
        no_replacement=True,
        boundary_present=True,
        stop_reason="end_marker",
    )

    assert passing["semantic_criteria_pass"] is True
    assert passing["sequence_criterion_pass"] is True
    assert passing["sequence_evaluation"] == LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION
    assert passing["generation_stop_reason"] == "end_marker"
    assert failing["forbidden_terms_hit"] == ["错误"]
    assert failing["semantic_criteria_pass"] is False
    assert unknown["required_terms"] == []
    assert unknown["unknown_policy_satisfied"] is True
    assert unknown["semantic_criteria_pass"] is True


def test_fast_slow_developmental_mode_replays_and_preserves_paired_readout() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(episode_id="r2-fast-slow"),
        corpus,
        config=LanguageAlignmentConfig(developmental_mode="fast_slow"),
    )
    baseline = trainer.checkpoint()
    training = trainer.train(epochs=1, max_episodes=1)

    assert training["developmental_mode"] == "fast_slow"
    assert training["developmental_history"][0]["replay"]["events"] > 0
    assert training["developmental_history"][0]["status"]["learning_mode"] == "fast_slow"
    assert training["developmental_history"][0]["status"]["replay_count"] == 0

    paired = paired_checkpoint_diagnostic(trainer, baseline)
    assert paired["checkpoint_read_only"] is True
    assert paired["native_mode_only"] is True
    assert paired["child_prompt_sensitivity_rate"] == 1.0

    restored = LanguageAlignmentTrainer.from_checkpoint(trainer.checkpoint(), corpus)
    assert restored.config.developmental_mode == "fast_slow"
    assert restored.model.developmental_f1_enabled is True


def test_condition_route_diagnostic_is_read_only_and_not_label_routed() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(Taiji(episode_id="r2-route"), corpus)
    before = trainer.checkpoint()["checkpoint_digest"]

    result = trainer.condition_route_diagnostic("train")

    assert result["status"] == "completed"
    assert result["episodes"] == 2
    assert result["pair_count"] == 1
    assert result["checkpoint_read_only"] is True
    assert result["native_mode_only"] is True
    assert trainer.checkpoint()["checkpoint_digest"] == before


def test_sequence_decode_diagnostic_keeps_native_owner_and_restore_boundary() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(Taiji(episode_id="r2-sequence-route"), corpus)
    before = trainer.checkpoint()["checkpoint_digest"]

    result = trainer.sequence_decode_diagnostic(
        "train",
        beam_width=2,
        top_k=4,
        max_generation_bytes=8,
    )

    assert result["status"] == "completed"
    assert result["readout_owner"] == "predictive_readout"
    assert result["decoder"] == "native-constrained-greedy-vs-beam-v1"
    assert result["checkpoint_read_only"] is True
    assert result["recovery_repeatable"] is True
    assert result["native_mode_only"] is True
    assert result["episodes"] == 2
    assert trainer.checkpoint()["checkpoint_digest"] == before


def test_response_start_readout_isolated_and_checkpointable() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(episode_id="r2-response-start"),
        corpus,
        config=LanguageAlignmentConfig(
            response_start_readout=True,
            learn_fabric=False,
            learn_predictive_context=False,
            learn_predictive_readout=False,
        ),
    )
    before = trainer.model.readout_registry_status()
    before_checkpoint = trainer.checkpoint()
    trainer.train(epochs=1, max_episodes=1)
    after = trainer.model.readout_registry_status()

    assert before["response_start"] is not None
    assert after["response_start"] is not None
    assert (
        after["response_start"]["readout_digest"]
        != before["response_start"]["readout_digest"]
    )
    assert (
        after["protected"]["readout_digest"]
        == before["protected"]["readout_digest"]
    )
    assert trainer.checkpoint()["model"][Taiji.RESPONSE_START_READOUT_KEY]
    margin = trainer.response_start_margin_diagnostic("train")
    assert margin["readout_owner"] == "predictive_readout.response_start"
    assert margin["checkpoint_read_only"] is True
    assert margin["recovery_repeatable"] is True
    assert margin["legal_start_count"] > 0
    restored = LanguageAlignmentTrainer.from_checkpoint(trainer.checkpoint(), corpus)
    assert restored.config.response_start_readout is True
    assert restored.model.response_start_readout_enabled is True
    assert restored.checkpoint()["checkpoint_digest"] == trainer.checkpoint()["checkpoint_digest"]

    trainer.model.clear_response_start_readout()
    assert trainer.model.response_start_readout_enabled is False
    assert "response_start_readout" not in trainer.model.checkpoint()
    assert before_checkpoint["model"].get(Taiji.RESPONSE_START_READOUT_KEY) is not None


def test_response_phase_readout_isolated_and_checkpointable() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(episode_id="r2-response-phase"),
        corpus,
        config=LanguageAlignmentConfig(
            response_phase_readout=True,
            learn_fabric=False,
            learn_predictive_context=False,
        ),
    )
    before = trainer.model.readout_registry_status()
    before_checkpoint = trainer.checkpoint()
    trainer.train(epochs=1, max_episodes=1)
    after = trainer.model.readout_registry_status()

    assert before["response_phase"] is not None
    assert after["response_phase"] is not None
    assert (
        after["response_phase"]["readout_digest"]
        != before["response_phase"]["readout_digest"]
    )
    assert after["protected"]["readout_digest"] == before["protected"]["readout_digest"]
    assert trainer.checkpoint()["model"][Taiji.RESPONSE_PHASE_READOUT_KEY]
    margin = trainer.response_phase_margin_diagnostic("train")
    assert margin["readout_owner"] == "predictive_readout.response_phase"
    assert margin["checkpoint_read_only"] is True
    assert margin["recovery_repeatable"] is True
    assert margin["legal_start_count"] > 0

    restored = LanguageAlignmentTrainer.from_checkpoint(trainer.checkpoint(), corpus)
    assert restored.config.response_phase_readout is True
    assert restored.model.response_phase_readout_enabled is True
    assert restored.checkpoint()["checkpoint_digest"] == trainer.checkpoint()["checkpoint_digest"]

    trainer.model.clear_response_phase_readout()
    assert trainer.model.response_phase_readout_enabled is False
    assert Taiji.RESPONSE_PHASE_READOUT_KEY not in trainer.model.checkpoint()
    assert before_checkpoint["model"].get(Taiji.RESPONSE_PHASE_READOUT_KEY) is not None


def test_response_start_and_phase_candidates_are_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LanguageAlignmentConfig(
            response_start_readout=True,
            response_phase_readout=True,
        )


def test_response_plan_candidate_isolated_checkpointable_and_ablated() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(
            config=TaijiConfig.capacity_profile(
                300_000,
                additional_predictive_readouts=1,
                response_plan_width=32,
            ),
            episode_id="r2-response-plan",
        ),
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=32,
            learn_fabric=False,
            learn_predictive_context=False,
        ),
    )
    protected_before = trainer.model.readout_registry_status()["protected"]["readout_digest"]
    candidate_before = trainer.model.response_plan_readout_digest
    parent_parameters = trainer.model.parameter_count()

    trainer.train(epochs=1, max_episodes=1)

    assert (
        trainer.model.readout_registry_status()["protected"]["readout_digest"]
        == protected_before
    )
    assert trainer.model.response_plan_readout_digest != candidate_before
    assert trainer.model.response_plan_readout.plan_state is not None
    conditioned = trainer.model.response_plan_probabilities()
    ablated = trainer.model.response_plan_probabilities(ablate_plan=True)
    assert conditioned.shape == ablated.shape
    assert not torch.equal(conditioned, ablated)
    assert trainer.model.parameter_count() == parent_parameters
    assert trainer.model.parameter_count() <= 300_000

    checkpoint = trainer.checkpoint()
    restored = LanguageAlignmentTrainer.from_checkpoint(checkpoint, corpus)
    assert restored.config.response_plan_readout is True
    assert restored.model.response_plan_readout_enabled is True
    assert restored.checkpoint()["checkpoint_digest"] == checkpoint["checkpoint_digest"]

    restored.model.clear_response_plan_readout()
    assert restored.model.response_plan_readout_enabled is False
    assert Taiji.RESPONSE_PLAN_READOUT_KEY not in restored.model.checkpoint()


def test_response_plan_is_mutually_exclusive_with_existing_candidates() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LanguageAlignmentConfig(
            response_phase_readout=True,
            response_plan_readout=True,
        )


def test_generalization_diagnostic_is_read_only_and_reports_transfer_surface() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(episode_id="r2-generalization"),
        corpus,
        config=LanguageAlignmentConfig(response_start_readout=True),
    )
    before = trainer.checkpoint()["checkpoint_digest"]

    result = trainer.generalization_diagnostic(("train", "dev", "final"))

    assert result["status"] == "completed"
    assert result["checkpoint_read_only"] is True
    assert result["recovery_repeatable"] is True
    assert result["native_mode_only"] is True
    assert result["external_provider"] is False
    assert result["profiles"]["train"]["target_first_byte_seen_in_train_rate"] == 1.0
    assert result["profiles"]["dev"]["target_first_byte_seen_in_train_rate"] == 0.0
    assert result["profiles"]["final"]["unknown_policy_seen_in_train_rate"] == 0.0
    assert trainer.checkpoint()["checkpoint_digest"] == before


def test_conditional_credit_diagnostic_is_positionwise_and_read_only() -> None:
    corpus = LanguageEpisodeCorpus.from_jsonl(
        [Path("tests/fixtures/r2_language_alignment_smoke.jsonl")]
    )
    trainer = LanguageAlignmentTrainer(
        Taiji(episode_id="r2-credit-profile"),
        corpus,
        config=LanguageAlignmentConfig(
            response_phase_readout=True,
            max_generation_bytes=8,
        ),
    )
    before = trainer.checkpoint()["checkpoint_digest"]

    result = trainer.conditional_credit_diagnostic(max_generation_bytes=4)

    assert result["format"] == LANGUAGE_ALIGNMENT_CREDIT_EVALUATION
    assert result["checkpoint_read_only"] is True
    assert result["recovery_repeatable"] is True
    assert result["native_mode_only"] is True
    assert result["external_provider"] is False
    assert result["result_credit_applied"] is False
    assert result["task_family_and_unknown_policy_forwarded"] is False
    assert result["profiles"]["train"]["by_role"]
    assert result["profiles"]["train"]["records"][0]["positions"]
    assert trainer.checkpoint()["checkpoint_digest"] == before


def test_language_alignment_training_and_checkpoint_preflight(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    trainer = LanguageAlignmentTrainer(Taiji(episode_id="r2-test"), corpus)

    report = checkpoint_roundtrip_preflight(
        trainer,
        episode=corpus.for_split("train")[0],
        directory=tmp_path / "preflight",
    )

    assert report["status"] == "passed"
    assert report["zero_step_roundtrip"] is True
    assert report["one_response_update"]["episodes"] == 1
    assert report["caller_model_unchanged"] is True
    assert report["atomic_save"] is True

    training = trainer.train(epochs=1)
    assert training["episodes"] == 2
    assert trainer.global_step > 0

    evaluation = trainer.evaluate("dev")
    assert evaluation["checkpoint_read_only"] is True
    assert evaluation["native_mode_only"] is True
    assert evaluation["episodes"] == 1
    assert "generated_text" in evaluation["records"][0]
    assert evaluation["utf8_valid_rate"] == 1.0
    assert evaluation["no_replacement_rate"] == 1.0
    assert evaluation["semantic_criteria_evaluated"] == 1
    assert evaluation["required_term_coverage"] == 0.0
    assert evaluation["semantic_criteria_pass_rate"] == 0.0
    assert "generated_bytes_hex" in evaluation["records"][0]
    assert evaluation["records"][0]["sequence_evaluation"] == (
        LANGUAGE_ALIGNMENT_SEQUENCE_EVALUATION
    )
    assert evaluation["records"][0]["generation_stop_reason"] == "max_generation_bytes"
    assert evaluation["unique_generated_texts"] == 1
    assert evaluation["generated_text_collision_rate"] == 0.0

    checkpoint_path = tmp_path / "r2-aligned.pt"
    trainer.save(checkpoint_path)
    restored_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = LanguageAlignmentTrainer.from_checkpoint(restored_payload, corpus)
    assert restored.global_step == trainer.global_step
    assert restored.corpus_digest == corpus.digest
    assert restored.evaluate("dev")["checkpoint_read_only"] is True
    assert restored.evaluate("final")["utf8_valid_rate"] == 1.0
