"""Explicit PLAN51 migration from frozen49, preserving every trained Adam moment."""

import json
from copy import deepcopy
from pathlib import Path

import torch

from scripts.training import run_taiji_n1_local_question as source
from taiji.native_contextual_question_evidence_candidate import ContextualQuestionEvidenceCandidate

SOURCE_SHA = "7347b34b7a987fc1014aaac91e2168f9532e1dba4e607f5b50d6650f69b93af3"
NEW_PARAMETERS = {
    "context_gate",
    "context_question.weight_ih_l0",
    "context_question.weight_hh_l0",
    "context_question.bias_ih_l0",
    "context_question.bias_hh_l0",
}


def source_payload():
    if source.paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen49 source report changed")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete frozen49 final required, not a selected midpoint")
    (entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    path = Path(entry["path"])
    if source.paired.sha(path) != entry["sha256"]:
        raise ValueError("frozen49 source checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed49 completed final protocol required")
    return record, payload, path


def migrate(payload):
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed49 final required for contextual migration")
    old = source.LocalQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    old_optimizer = torch.optim.Adam(old.parameters(), lr=0.001)
    old_optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if len(old_optimizer.state) != len(list(old.parameters())):
        raise ValueError("every old parameter must retain trained moments")
    torch.manual_seed(11)
    model = ContextualQuestionEvidenceCandidate(old.config)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if set(missing) != NEW_PARAMETERS or unexpected:
        raise ValueError("migration must add exactly the contextual GRU and gate")
    model._observations = deepcopy(old._observations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for key, value in old_optimizer.param_groups[0].items():
        if key != "params":
            optimizer.param_groups[0][key] = deepcopy(value)
    parameters = dict(model.named_parameters())
    for name, parameter in old.named_parameters():
        expected = 6144 if name in source.transfer.NEW_PARAMETERS else 12288
        if int(old_optimizer.state[parameter]["step"]) != expected:
            raise ValueError("reader/local source Adam steps differ")
        optimizer.state[parameters[name]] = deepcopy(old_optimizer.state[parameter])
        if not torch.equal(parameter, parameters[name]) or source.paired.fingerprint(
            optimizer.state[parameters[name]]
        ) != source.paired.fingerprint(old_optimizer.state[parameter]):
            raise ValueError("old parameter or Adam moment changed during migration")
    if any(parameters[name] in optimizer.state for name in NEW_PARAMETERS):
        raise ValueError("new contextual parameters must have no trained moment")
    return model, optimizer
