"""PLAN81 train-only same-entity transition quartet, one unchanged bounded Adam."""

import math
from pathlib import Path

from scripts.training import diagnose_taiji_n1_fresh_transition_factorial as factorial

train = factorial.train
paired = train.paired
torch = train.torch
POLICY = dict(
    schema="n1-same-entity-transition-factorial-learning-v1",
    transition="mean-four-condition-QA-plus-0.25-own-observation-fidelity",
    other_tasks="unchanged-paired-question-objective",
    optimizer_proposals_per_update=1,
    update_policy=train.update.POLICY,
)


def identity():
    return dict(policy=POLICY, factorial=factorial.identity(), runner=paired.sha(Path(__file__)))


def train_step(model, optimizer, item, session, arm, second=None, *, audit=False):
    if item["split"] != "train" or arm not in train.engineering.ARMS:
        raise ValueError("only train and explicit paired arm accepted")
    if item["kind"] != "transition":
        return train.update.train_step(model, optimizer, item, session, arm, second, audit=audit)
    if second is not None:
        raise ValueError("transition uses four observations, not a second question")
    entries = factorial.variants(item)
    sessions = [f"{session}/{e['condition']}" for e in entries]
    if any(s in model._observations for s in sessions):
        raise ValueError("quartet training sessions must be unique")
    optimizer.zero_grad(set_to_none=True)
    totals, qas, fidelities = [], [], []
    for entry, key in zip(entries, sessions, strict=True):
        candidate = entry["item"]
        paired.observe(model, paired.project(candidate), key)
        losses = train.runner.core.paired_question_loss(
            model, key, candidate["question"].encode(), candidate["answer"].encode(), arm=arm
        )
        if not torch.isfinite(losses.total):
            raise ValueError("nonfinite quartet loss")
        (losses.total / 4).backward()
        totals.append(float(losses.total.detach()))
        qas.append(float(losses.qa.detach()))
        fidelities.append(float(losses.observation.detach()))
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite quartet gradient")
    _, measurements = train.update.bounded.bounded_step(
        model, optimizer, optimizer.step, audit=audit
    )
    return (
        dict(
            total=math.fsum(totals) / 4,
            qa=math.fsum(qas) / 4,
            observation=math.fsum(fidelities) / 4,
            qa_sequences=4,
        ),
        measurements,
    )
