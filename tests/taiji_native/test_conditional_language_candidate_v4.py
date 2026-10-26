"""V4 consumes workspace, preserves parent lineage, and restores exact continuation."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from scripts.training.audit_taiji_n1_plan176_learning_paths import (
    credit_and_state_paths,
    interventions,
    prompt_state,
)
from scripts.training.build_taiji_n1_plan176_foundation_corpus import (
    as_language_episode,
    build_rows,
    validate_rows,
)
from taiji.conditional_language_v3 import (
    ConditionalLanguageCandidateV3,
    ConditionalLanguageTrainerV3,
    TaskCoreConfigV3,
)
from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4


def parent_payload():
    config = TaskCoreConfigV3(
        width=8,
        embedding_width=4,
        key_width=4,
        slots=4,
        chunk_bytes=8,
        max_observation_bytes=1024,
        max_input_evidence_bytes=1024,
        max_output_position=128,
        max_generation_bytes=128,
    )
    trainer = ConditionalLanguageTrainerV3(ConditionalLanguageCandidateV3(config))
    trainer.update([(b"A=1;A?", b"1")])
    state = trainer.model.new_state(scope_id="test", session_id="s", native_state_ref="test:parent")
    return trainer.checkpoint(state, cursor={"step": 1})


def test_migration_preserves_shared_weights_adam_counters_and_rng():
    payload = parent_payload()
    old, _, _ = ConditionalLanguageTrainerV3.from_checkpoint(payload)
    new, state, cursor = ConditionalLanguageTrainerV4.from_v3_checkpoint(payload)
    assert cursor == {"step": 1} and state.bundle_id == new.model.bundle_id
    assert new.steps == old.steps and new.target_byte_exposures == old.target_byte_exposures
    old_params, new_params = dict(old.model.named_parameters()), dict(new.model.named_parameters())
    for name, parameter in old_params.items():
        assert torch.equal(parameter, new_params[name])
        old_adam, new_adam = old.optimizer.state[parameter], new.optimizer.state[new_params[name]]
        for key in old_adam:
            assert torch.equal(old_adam[key], new_adam[key])
    assert torch.equal(old.model.generator.get_state(), new.model.generator.get_state())
    assert not new.optimizer.state[new_params["workspace_projection.weight"]]


def test_workspace_information_and_task_credit_paths_are_both_nonzero():
    trainer, _, _ = ConditionalLanguageTrainerV4.from_v3_checkpoint(parent_payload())
    row = next(r for r in build_rows() if r["metadata"]["query_kind"] == "primary_color")
    paths = credit_and_state_paths(trainer.model, as_language_episode(row))
    assert paths["state_perturbation_max_logit_delta"]["workspace"] > 0
    assert paths["task_loss_gradient_norms"]["state.workspace"] > 0
    assert paths["task_loss_gradient_norms"]["workspace_projection.weight"] > 0
    assert paths["task_loss_gradient_norms"]["workspace_gate"] > 0


def test_cache_rejects_changed_workspace_and_ablated_workspace_has_no_residual():
    source = parent_payload()
    parent, _, _ = ConditionalLanguageTrainerV3.from_checkpoint(source)
    trainer, _, _ = ConditionalLanguageTrainerV4.from_v3_checkpoint(source)
    model = trainer.model
    state = prompt_state(model, b"A=1;A?")
    cache = model.evidence_cache(state)
    changed = replace(state, workspace=state.workspace + 1)
    with pytest.raises(ValueError, match="cache"):
        model.next_logits(changed, cache=cache)
    empty = replace(state, workspace=state.workspace[:0], evidence_refs=())
    assert torch.equal(
        model.next_logits(empty)[0],
        parent.model.next_logits(prompt_state(parent.model, b"A=1;A?"))[0],
    )
    with pytest.raises(ValueError, match="reference|span"):
        wrong = replace(state, evidence_refs=((0, 0, 9999),) * len(state.workspace))
        model.evidence_cache(wrong)


def test_checkpoint_versions_are_explicit_and_continuation_is_exact(tmp_path):
    v3 = parent_payload()
    trainer, state, cursor = ConditionalLanguageTrainerV4.from_v3_checkpoint(v3)
    with pytest.raises(ValueError, match="explicit"):
        ConditionalLanguageTrainerV4.from_checkpoint(v3)
    payload = trainer.checkpoint(state, cursor=cursor)
    with pytest.raises(ValueError, match="unsupported"):
        ConditionalLanguageTrainerV3.from_checkpoint(payload)
    restored, restored_state, restored_cursor = ConditionalLanguageTrainerV4.from_checkpoint(
        payload
    )
    assert (
        restored.checkpoint(restored_state, cursor=restored_cursor)["checkpoint_digest"]
        == payload["checkpoint_digest"]
    )
    batch = [(b"A=2;A?", b"2")]
    trainer.update(batch)
    restored.update(batch)
    # Continuing either live copy must not change the frozen input snapshot.
    frozen, frozen_state, frozen_cursor = ConditionalLanguageTrainerV4.from_checkpoint(payload)
    assert (
        frozen.checkpoint(frozen_state, cursor=frozen_cursor)["checkpoint_digest"]
        == payload["checkpoint_digest"]
    )
    state1 = trainer.model.new_state(scope_id="t", session_id="s", native_state_ref="n")
    state2 = restored.model.new_state(scope_id="t", session_id="s", native_state_ref="n")
    assert (
        trainer.checkpoint(state1, cursor={})["checkpoint_digest"]
        == restored.checkpoint(state2, cursor={})["checkpoint_digest"]
    )
    checkpoint = tmp_path / "v4.pt"
    torch.save(payload, checkpoint)
    code = (
        "import sys,torch; from taiji.conditional_language_v4 import ConditionalLanguageTrainerV4; "
        "p=torch.load(sys.argv[1],weights_only=False); "
        "t,s,c=ConditionalLanguageTrainerV4.from_checkpoint(p); "
        "assert t.checkpoint(s,cursor=c)['checkpoint_digest']==p['checkpoint_digest']"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    subprocess.run([sys.executable, "-c", code, str(checkpoint)], env=env, check=True)


@pytest.mark.parametrize(
    "kind", ["primary_color", "distractor_color", "location_update", "owner_correction"]
)
def test_audit_related_and_irrelevant_interventions_preserve_the_oracle_contract(kind):
    row = next(
        r for r in build_rows() if r["split"] == "train" and r["metadata"]["query_kind"] == kind
    )
    original = as_language_episode(row)
    related, irrelevant = interventions(row)
    validate_rows([related])
    validate_rows([irrelevant])
    assert related["response"] != row["response"]
    assert irrelevant["response"] == row["response"]
    assert as_language_episode(related).history == original.history
    assert as_language_episode(irrelevant).history == original.history
