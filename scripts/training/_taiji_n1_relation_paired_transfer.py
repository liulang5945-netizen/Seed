"""Explicit PLAN62 migration from frozen52, preserving every trained Adam moment."""

import json
from copy import deepcopy
from pathlib import Path

import torch

from scripts.training import run_taiji_n1_contextual_question as source
from taiji.native_local_relation_evidence_candidate import LocalRelationEvidenceCandidate

SOURCE_SHA = "cb26c88e8dab98819a3dbcd48348f7737230872f5444039d460af77e7d00809f"
NEW_PARAMETERS = {
    "relation_sequence.weight_ih_l0",
    "relation_sequence.weight_hh_l0",
    "relation_sequence.bias_ih_l0",
    "relation_sequence.bias_hh_l0",
    "relation_projection.weight",
    "relation_projection.bias",
}


def source_payload():
    if source.paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen52 source report changed")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete frozen52 final required, not a selected midpoint")
    (entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    path = Path(entry["path"])
    if source.paired.sha(path) != entry["sha256"]:
        raise ValueError("frozen52 source checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed52 completed final protocol required")
    return record, payload, path


def migrate(payload, seed):
    if seed not in (11, 12, 13):
        raise ValueError("frozen relation seeds only")
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed52 final required for contextual migration")
    old = source.ContextualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
    old_optimizer = torch.optim.Adam(old.parameters(), lr=0.001)
    old_optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if len(old_optimizer.state) != len(list(old.parameters())):
        raise ValueError("every old parameter must retain trained moments")
    torch.manual_seed(seed)
    model = LocalRelationEvidenceCandidate(old.config)
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    if set(missing) != NEW_PARAMETERS or unexpected:
        raise ValueError("migration must add exactly the relation GRU and zero key/value head")
    model._observations = deepcopy(old._observations)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for key, value in old_optimizer.param_groups[0].items():
        if key != "params":
            optimizer.param_groups[0][key] = deepcopy(value)
    parameters = dict(model.named_parameters())
    for name, parameter in old.named_parameters():
        expected = {"reader": 18432, "local": 12288, "context": 6144}[source.parameter_group(name)]
        if int(old_optimizer.state[parameter]["step"]) != expected:
            raise ValueError("reader/local source Adam steps differ")
        optimizer.state[parameters[name]] = deepcopy(old_optimizer.state[parameter])
        if not torch.equal(parameter, parameters[name]) or source.paired.fingerprint(
            optimizer.state[parameters[name]]
        ) != source.paired.fingerprint(old_optimizer.state[parameter]):
            raise ValueError("old parameter or Adam moment changed during migration")
    if any(parameters[name] in optimizer.state for name in NEW_PARAMETERS):
        raise ValueError("new relation parameters must have no trained moment")
    return model, optimizer
