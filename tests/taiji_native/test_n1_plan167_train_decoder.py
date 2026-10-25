"""Train-only selection, decoder parity and label-informed state isolation."""

from types import SimpleNamespace

import pytest
import torch

from scripts.training import diagnose_taiji_n1_plan167_train_decoder as diagnosis
from scripts.training import summarize_taiji_n1_plan167_train_decoder as summary
from taiji import (
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisode,
    LanguageEpisodeCorpus,
    Taiji,
)
from taiji.language_alignment import content_digest


def episode(index: int, *, split: str = "train") -> LanguageEpisode:
    return LanguageEpisode(
        episode_id=f"e{index}",
        family_id=f"f{index}",
        task_family="synthetic_diagnostic",
        split=split,
        user_input=f"请解释事项{index}",
        response="这是仅用于合成测试的中文回答。" * 3,
    )


def test_selection_requests_only_train_and_keeps_original_order() -> None:
    episodes = [episode(i) for i in range(6)]
    seen = []

    def split(name):
        seen.append(name)
        assert name == "train"
        return episodes

    assert [
        item.episode_id for item in diagnosis.select_train(SimpleNamespace(for_split=split))
    ] == [f"e{i}" for i in range(4)]
    assert seen == ["train"]


def test_insufficient_selection_fails_instead_of_reducing_denominator() -> None:
    with pytest.raises(ValueError, match="fewer than four"):
        diagnosis.select_train(SimpleNamespace(for_split=lambda _: [episode(0)]))


@pytest.fixture
def trainer(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(diagnosis, "MAX_BYTES", 24)
    corpus = LanguageEpisodeCorpus((episode(0),), require_eval_splits=False)
    return LanguageAlignmentTrainer(
        Taiji(),
        corpus,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=8,
            response_plan_variant="factorized_v1",
            response_plan_slots=4,
            response_plan_phase_stride=4,
            use_memory=False,
        ),
    )


def test_instrumented_rollout_matches_the_real_configured_decoder(trainer) -> None:
    ep = episode(0)
    snapshot = trainer.model.checkpoint()
    actual = trainer._constrained_generate(ep, max_generation_bytes=24)
    diagnosis.restore(trainer, snapshot)
    observed = diagnosis.rollout(trainer, ep, "configured")
    assert observed == actual
    diagnosis.restore(trainer, snapshot)
    assert content_digest(trainer.model.checkpoint()) == content_digest(snapshot)


def test_oracle_changes_only_disposable_state_and_never_calls_learning(
    trainer, monkeypatch
) -> None:
    snapshot = trainer.model.checkpoint()

    def forbidden(*_args, **_kwargs):
        pytest.fail("oracle diagnostics must not perform any planner weight update")

    monkeypatch.setattr(trainer.model, "learn_response_plan_target", forbidden)
    result, distributions = diagnosis.teacher_forced(trainer, episode(0), "oracle_plan")
    assert result["positions"] == 24
    assert distributions.shape[0] == 24
    diagnosis.restore(trainer, snapshot)
    assert content_digest(trainer.model.checkpoint()) == content_digest(snapshot)


def test_tf_metrics_and_js_keep_distinct_probability_surfaces() -> None:
    p = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    assert diagnosis.js_bits(p, p) == pytest.approx(0)
    assert diagnosis.js_bits(p, p.flip(-1)) > 0.1
    assert diagnosis.prefix_match(b"abc", b"abx") == pytest.approx(2 / 3)
    rows = [
        {"target": 1, "raw_top1": 0, "masked_top1": 1, "nll_raw_bits": 3.0, "nll_masked_bits": 1.0}
    ]
    result = diagnosis.summarize_tf(rows)
    assert result["top1_raw"] == 0
    assert result["top1_masked"] == 1


def test_character_metric_requires_all_bytes_and_excludes_incomplete_character() -> None:
    raw = "中文".encode()
    rows = [{"target": byte, "raw_top1": byte} for byte in raw]
    assert summary.cjk_accuracy(rows) == (2, 2)
    rows[4]["raw_top1"] = 0
    assert summary.cjk_accuracy(rows) == (1, 2)
    assert summary.cjk_accuracy(rows[:5]) == (1, 1)
