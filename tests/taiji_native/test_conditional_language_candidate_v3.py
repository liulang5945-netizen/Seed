"""V3 boundaries and checkpoint contracts stay separate from V2."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from taiji.conditional_language_v2 import (
    ConditionalLanguageCandidateV2,
    ConditionalLanguageTrainerV2,
)
from taiji.conditional_language_v3 import (
    ConditionalLanguageCandidateV3,
    ConditionalLanguageTrainerV3,
    TaskCoreConfigV3,
)
from taiji.task_core_v2 import TaskCoreConfigV2

ROOT = Path(__file__).resolve().parents[2]


def small_config() -> TaskCoreConfigV3:
    return TaskCoreConfigV3(
        width=8,
        embedding_width=4,
        key_width=4,
        slots=4,
        chunk_bytes=4,
        max_observation_bytes=1024,
        max_input_evidence_bytes=1024,
        max_output_position=520,
        max_generation_bytes=520,
    )


def test_input_and_output_position_ranges_are_independent():
    model = ConditionalLanguageCandidateV3(small_config())
    state = model.new_state(scope_id="test", session_id="s", native_state_ref="n")
    state = model.step_task(
        b"a" * 1024,
        state,
        scope_id="test",
        session_id="s",
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 1024},
    )
    assert len(state.byte_memory) == 1024
    assert model.evidence_cache(state).keys.shape[0] == 1024
    for _ in range(512):
        state = model.observe_response(state, ord("a"), torch.zeros(8))
    assert model.attention_weights(state).shape == (1024,)
    for _ in range(7):
        state = model.observe_response(state, ord("a"), torch.zeros(8))
    assert model.attention_weights(state).shape == (1024,)
    state = model.observe_response(state, ord("a"), torch.zeros(8))
    with pytest.raises(ValueError, match="output-position"):
        model.attention_weights(state)


def test_v2_and_v3_checkpoint_formats_reject_each_other():
    model3 = ConditionalLanguageCandidateV3(small_config())
    trainer3 = ConditionalLanguageTrainerV3(model3)
    state3 = model3.new_state(scope_id="test", session_id="s", native_state_ref="n")
    payload3 = trainer3.checkpoint(state3, cursor={"epoch": 0})
    with pytest.raises(ValueError, match="unsupported"):
        ConditionalLanguageTrainerV2.from_checkpoint(payload3)

    model2 = ConditionalLanguageCandidateV2(
        TaskCoreConfigV2(width=8, embedding_width=4, key_width=4, slots=4, chunk_bytes=4)
    )
    trainer2 = ConditionalLanguageTrainerV2(model2)
    state2 = model2.new_state(scope_id="test", session_id="s", native_state_ref="n")
    with pytest.raises(ValueError, match="unsupported"):
        ConditionalLanguageTrainerV3.from_checkpoint(
            trainer2.checkpoint(state2, cursor={"epoch": 0})
        )


def test_v2_parent_migration_copies_shared_weights_and_optimizer_by_name():
    config2 = TaskCoreConfigV2(
        width=8,
        embedding_width=4,
        key_width=4,
        slots=4,
        chunk_bytes=4,
        max_observation_bytes=64,
        max_evidence_bytes=16,
        seed=169,
    )
    parent_model = ConditionalLanguageCandidateV2(config2, bundle_id="parent")
    parent_trainer = ConditionalLanguageTrainerV2(parent_model)
    parent_trainer.update([(b"prompt", b"ok")])
    parent_state = parent_model.new_state(scope_id="test", session_id="s", native_state_ref="n")
    parent_payload = parent_trainer.checkpoint(parent_state, cursor={"epoch": 1})
    migrated = ConditionalLanguageTrainerV3.from_v2_checkpoint(
        parent_payload,
        config=TaskCoreConfigV3(
            width=8,
            embedding_width=4,
            key_width=4,
            slots=4,
            chunk_bytes=4,
            max_observation_bytes=1024,
            max_input_evidence_bytes=1024,
            max_output_position=520,
            max_generation_bytes=520,
            seed=169,
        ),
        parent_checkpoint_sha256="a" * 64,
    )
    parent_parameters = dict(parent_model.named_parameters())
    migrated_parameters = dict(migrated.model.named_parameters())
    for name, value in parent_parameters.items():
        target_names = (
            ("input_position.weight", "output_position.weight")
            if name == "position.weight"
            else (name,)
        )
        for target_name in target_names:
            target = migrated_parameters[target_name]
            if name == "position.weight":
                assert torch.equal(target[: value.shape[0]], value)
            else:
                assert torch.equal(target, value)
    assert migrated.lineage["position_rows_copied"] == 16
    assert migrated.lineage["parent_checkpoint_digest"] == parent_payload["checkpoint_digest"]
    old_position_id = parent_trainer.optimizer.state_dict()["param_groups"][0]["params"][
        list(parent_parameters).index("position.weight")
    ]
    new_state = migrated.optimizer.state_dict()["state"]
    new_ids = migrated.optimizer.state_dict()["param_groups"][0]["params"]
    new_position_id = new_ids[list(migrated_parameters).index("input_position.weight")]
    old_moment = parent_trainer.optimizer.state_dict()["state"][old_position_id]["exp_avg"]
    new_moment = new_state[new_position_id]["exp_avg"]
    assert torch.equal(new_moment[: old_moment.shape[0]], old_moment)
    assert torch.count_nonzero(new_moment[old_moment.shape[0] :]) == 0


def test_v3_step_update_and_fresh_process_checkpoint_restore(tmp_path):
    model = ConditionalLanguageCandidateV3(small_config())
    trainer = ConditionalLanguageTrainerV3(model)
    update = trainer.update([(b"<|user|>x<|assistant|>", b"ok")])
    assert update["step"] == 1
    assert update["target_byte_exposures"] == 2
    state = model.new_state(scope_id="test", session_id="s", native_state_ref="n")
    state = model.step_task(
        b"<|user|>x<|assistant|>",
        state,
        scope_id="test",
        session_id="s",
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 1024},
    )
    path = tmp_path / "v3.pt"
    trainer.save(path, state, cursor={"epoch": 1, "sample": 2})
    prompt, target = b"<|user|>x<|assistant|>", b"ok"
    control, _, _ = ConditionalLanguageTrainerV3.from_checkpoint(
        trainer.checkpoint(state, cursor={"epoch": 1, "sample": 2})
    )
    control.update([(prompt, target)])
    control_state = control.model.new_state(scope_id="test", session_id="s", native_state_ref="n")
    control_state = control.model.step_task(
        prompt,
        control_state,
        scope_id="test",
        session_id="s",
        source_id="prompt",
        resource_snapshot={"remaining_steps": 1, "max_observation_bytes": 1024},
    )
    expected_digest = control.checkpoint(control_state, cursor={"epoch": 1, "sample": 2})[
        "checkpoint_digest"
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import torch,sys; "
        "from taiji.conditional_language_v3 import ConditionalLanguageTrainerV3; "
        "p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "t,s,c=ConditionalLanguageTrainerV3.from_checkpoint(p); "
        "assert t.steps==1 and t.target_byte_exposures==2 and c['epoch']==1; "
        "assert s.to_payload()['format']=='taiji-task-state-v3'; "
        "t.update([(b'<|user|>x<|assistant|>',b'ok')]); "
        "s=t.model.new_state(scope_id='test',session_id='s',native_state_ref='n'); "
        "s=t.model.step_task(b'<|user|>x<|assistant|>',s,scope_id='test',session_id='s',"
        "source_id='prompt',resource_snapshot={'remaining_steps':1,'max_observation_bytes':1024}); "
        "print(t.checkpoint(s,cursor={'epoch':1,'sample':2})['checkpoint_digest'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == expected_digest


def test_streamed_per_example_gradients_match_one_full_batch_update():
    examples = [
        (b"<|user|>x<|assistant|>", b"ok"),
        (b"<|user|>y<|assistant|>", b"yes"),
    ]
    streamed = ConditionalLanguageTrainerV3(ConditionalLanguageCandidateV3(small_config()))
    reference = ConditionalLanguageTrainerV3(ConditionalLanguageCandidateV3(small_config()))
    streamed.update(examples)

    reference.optimizer.zero_grad(set_to_none=True)
    total_bytes = sum(len(target) for _, target in examples)
    losses = []
    for prompt, target in examples:
        logits = reference.model.teacher_forced(prompt, target)
        losses.append(
            F.cross_entropy(
                logits,
                torch.tensor(list(target), dtype=torch.long),
                reduction="sum",
            )
        )
    (torch.stack(losses).sum() / total_bytes).backward()
    torch.nn.utils.clip_grad_norm_(reference.model.parameters(), 1.0, error_if_nonfinite=True)
    reference.optimizer.step()
    reference.model.parameter_revision += 1
    for left, right in zip(streamed.model.parameters(), reference.model.parameters(), strict=True):
        assert torch.allclose(left, right, atol=1e-7, rtol=1e-6)
