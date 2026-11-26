"""Unit tests for open-vocabulary natural Chinese continuous task & pure-UTF-8 subject."""

from __future__ import annotations

import pytest
import torch

from taiji.open_chinese_subject import (
    OpenChineseCognitiveSubject,
    build_turn_memory_tensors,
    collate_memory_batch,
    encode_target_blocks,
)
from taiji.open_chinese_task_env import (
    BROAD_TRAIN_CJK,
    HELDOUT_DISTRACTORS,
    HELDOUT_PARAPHRASE_PATTERNS,
    INTENTS,
    OOV_ENTITIES,
    OOV_PLACES,
    TRAIN_DISTRACTORS,
    TRAIN_ENTITIES,
    TRAIN_PATTERNS,
    TRAIN_PLACES,
    OpenChineseTaskEnv,
    eval_open_policy,
    oracle_policy,
    random_policy,
)
from taiji.session_store import SessionStore


def test_disjoint_oov_pools_and_patterns() -> None:
    tr_ent_chars = set("".join(TRAIN_ENTITIES))
    oov_ent_chars = set("".join(OOV_ENTITIES))
    assert tr_ent_chars.isdisjoint(oov_ent_chars)

    tr_place_chars = set("".join(TRAIN_PLACES))
    oov_place_chars = set("".join(OOV_PLACES))
    assert tr_place_chars.isdisjoint(oov_place_chars)

    assert len(BROAD_TRAIN_CJK) >= 200
    assert set(BROAD_TRAIN_CJK).isdisjoint(oov_ent_chars | oov_place_chars)

    assert set(TRAIN_DISTRACTORS).isdisjoint(set(HELDOUT_DISTRACTORS))
    assert set(TRAIN_PATTERNS) == set(HELDOUT_PARAPHRASE_PATTERNS)
    for k in TRAIN_PATTERNS:
        assert set(TRAIN_PATTERNS[k]).isdisjoint(set(HELDOUT_PARAPHRASE_PATTERNS[k]))


def test_open_lexicon_guard_rejects_heldout_splits() -> None:
    with pytest.raises(ValueError):
        OpenChineseTaskEnv(1, split="heldout_paraphrase", open_lexicon_prob=0.5)
    with pytest.raises(ValueError):
        OpenChineseTaskEnv(1, split="heldout_oov_entities", open_lexicon_prob=0.5)
    env = OpenChineseTaskEnv(77, split="train_seen", open_lexicon_prob=1.0)
    obs = env.reset()
    assert set(obs) == {"text", "utf8_bytes", "step", "turns"}
    oov_chars = set("".join(OOV_ENTITIES)) | set("".join(OOV_PLACES))
    assert set("".join(env.entities)).isdisjoint(oov_chars)
    assert set("".join(env.places)).isdisjoint(oov_chars)


def test_zero_leak_and_obs_isolation_across_splits() -> None:
    for idx, split in enumerate(("train_seen", "heldout_paraphrase", "heldout_oov_entities")):
        seeds = list(range(100 + idx * 50, 150 + idx * 50))
        for seed in seeds:
            env = OpenChineseTaskEnv(seed, split=split)
            obs = env.reset()
            assert set(obs) == {"text", "utf8_bytes", "step", "turns"}
            while True:
                meta = env.current_meta
                if meta["is_query"]:
                    assert env.intended_answer not in env.current_text()
                obs, _, done = env.step(env.intended_intent, env.intended_answer)
                if done:
                    break


def test_oracle_and_random_calibration_plus_determinism() -> None:
    for idx, split in enumerate(("train_seen", "heldout_paraphrase", "heldout_oov_entities")):
        seeds = list(range(300 + idx * 30, 330 + idx * 30))
        ora = eval_open_policy(oracle_policy, seeds, split=split)
        assert ora["all_turn_success"] == 1.0
        assert ora["episode_success"] == 1.0
        rnd = eval_open_policy(
            random_policy, seeds, split=split, gen=torch.Generator().manual_seed(idx)
        )
        assert rnd["query_turn_success"] <= 0.05

    env1 = OpenChineseTaskEnv(999, split="heldout_oov_entities")
    env2 = OpenChineseTaskEnv(999, split="heldout_oov_entities")
    env1.reset()
    env2.reset()
    assert env1.turns_data == env2.turns_data

    # Ensure coref_recent is placed at Turn 4 immediately after turns 0-3 (no Turn-4 entity conflict).
    recent_markers = ("最近", "最后")
    for split in ("train_seen", "heldout_paraphrase", "heldout_oov_entities"):
        for s in range(500, 550):
            e = OpenChineseTaskEnv(s, split=split)
            e.reset()
            t5 = e.turns_data[5]
            if t5["is_coref"] and any(m in t5["text"] for m in recent_markers):
                raise AssertionError(f"coref_recent appeared at Turn 5 on {split} seed={s}")


def test_subject_forward_backward_utf8_validity_and_store_roundtrip(tmp_path) -> None:
    env = OpenChineseTaskEnv(42, split="heldout_oov_entities")
    obs = env.reset()
    history = [list(obs["utf8_bytes"])]
    samples = []
    while True:
        mb, mf = build_turn_memory_tensors(history, use_history=True)
        tb = encode_target_blocks(env.intended_answer)
        samples.append(
            {
                "mem_blocks": mb,
                "mem_feats": mf,
                "target_intent": env.intended_intent,
                "target_blocks": tb,
            }
        )
        obs_next, _, done = env.step(env.intended_intent, env.intended_answer)
        if done:
            break
        history.append(list(obs_next["utf8_bytes"]))

    batch = collate_memory_batch(samples)
    assert batch[1].shape[-1] == 13
    for use_history, use_copy in ((True, True), (False, True), (True, False)):
        subj = OpenChineseCognitiveSubject(123, use_history=use_history, use_copy=use_copy)
        subj.zero_grad(set_to_none=True)
        loss, iloss, bloss = subj.forward_loss(*batch)
        assert torch.isfinite(loss)
        assert torch.isfinite(iloss)
        assert torch.isfinite(bloss)
        loss.backward()

    subj_main = OpenChineseCognitiveSubject(777, use_history=True, use_copy=True)
    for t in range(len(history)):
        intent_id, text, valid = subj_main.predict_turn(history[: t + 1])
        assert 0 <= intent_id < len(INTENTS)
        assert valid is True
        text.encode("utf-8")

    store = SessionStore(tmp_path / "store")
    subj_main.save_to_session_store(store, "open_zh_test", session_id=1)
    reloaded = OpenChineseCognitiveSubject.load_from_session_store(store, "open_zh_test")
    for a, b in zip(subj_main.parameters(), reloaded.parameters(), strict=True):
        assert torch.equal(a, b)
    for t in range(len(history)):
        assert subj_main.predict_turn(history[: t + 1]) == reloaded.predict_turn(history[: t + 1])
