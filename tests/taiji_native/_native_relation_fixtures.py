"""Synthetic mechanical states only; never trained evidence or experiment inputs."""

import torch

from scripts.training import _taiji_n1_contextual_question_transfer as contextual
from scripts.training import _taiji_n1_relation_paired_transfer as relation
from taiji.native_evidence_candidate import Observation


def _payload(source, model, steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(steps(name))),
            "exp_avg": torch.full_like(parameter, 1e-5),
            "exp_avg_sq": torch.full_like(parameter, 1e-4),
        }
    model.observe("zero-probe", Observation("text", "alpha uses quartz.", 1, "fixture"))
    return dict(protocol=source.PROTOCOL, completed_updates=6144, candidate=model.bundle(optimizer))


def local_parent():
    """Explicit synthetic local parent with the real migration step layout."""
    with torch.random.fork_rng():
        torch.manual_seed(11)
        source = contextual.source
        model = source.LocalQuestionEvidenceCandidate()
        return _payload(
            source, model, lambda name: 6144 if name in source.transfer.NEW_PARAMETERS else 12288
        )


def contextual_parent():
    """Synthetic contextual parent; nonzero gate exercises the learned branch."""
    with torch.random.fork_rng():
        torch.manual_seed(11)
        source = relation.source
        model = source.ContextualQuestionEvidenceCandidate()
        with torch.no_grad():
            model.context_gate.fill_(0.1)
        return _payload(
            source,
            model,
            lambda name: {"reader": 18432, "local": 12288, "context": 6144}[
                source.parameter_group(name)
            ],
        )


def paired_sources():
    """Expected pairing is derived from deterministic real migration, without IO."""
    payload = contextual_parent()
    fingerprints = {}
    with torch.random.fork_rng():
        for seed in (11, 12, 13):
            model, optimizer = relation.migrate(payload, seed)
            fingerprints[str(seed)] = [
                relation.source.paired.fingerprint(model.bundle(optimizer))
            ] * 3
    return {"initial_fingerprints": fingerprints}, payload


def relation_zero():
    with torch.random.fork_rng():
        model, optimizer = relation.migrate(contextual_parent(), 13)
        return model.bundle(optimizer)
