"""Explicit PLAN48 migration: preserve shared reader weights and Adam moments."""

from copy import deepcopy

import torch

from scripts.training import run_taiji_n1_reader_transfer as source
from taiji.native_local_question_evidence_candidate import LocalQuestionEvidenceCandidate

NEW_PARAMETERS = {
    "local_projection.weight",
    "local_salience.weight",
    "local_salience.bias",
    "local_fusion.weight",
    "local_fusion.bias",
    "local_log_temperature",
}


def migrate(payload):
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed41 final reader required")
    old = source.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    old_optimizer = torch.optim.Adam(old.parameters(), lr=0.001)
    old_optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if not old_optimizer.state:
        raise ValueError("trained shared Adam moments required")
    torch.manual_seed(11)
    model = LocalQuestionEvidenceCandidate(old.config)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if set(missing) != NEW_PARAMETERS or unexpected:
        raise ValueError("migration must add exactly the frozen local binding parameters")
    model._observations = deepcopy(old._observations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for key, value in old_optimizer.param_groups[0].items():
        if key != "params":
            optimizer.param_groups[0][key] = deepcopy(value)
    new_parameters = dict(model.named_parameters())
    for name, parameter in old.named_parameters():
        if parameter not in old_optimizer.state:
            raise ValueError("all original parameter moments required")
        optimizer.state[new_parameters[name]] = deepcopy(old_optimizer.state[parameter])
        if not torch.equal(parameter, new_parameters[name]):
            raise ValueError("shared reader parameter changed")
        if source.paired.fingerprint(
            optimizer.state[new_parameters[name]]
        ) != source.paired.fingerprint(old_optimizer.state[parameter]):
            raise ValueError("shared Adam moment changed")
    if any(new_parameters[name] in optimizer.state for name in NEW_PARAMETERS):
        raise ValueError("new heads must start without trained moments")
    return model, optimizer
