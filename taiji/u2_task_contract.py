"""Environment-side request binding, feedback rights and full-denominator scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    evidence_id: str
    entity: str
    value: str
    version: int
    source: str


@dataclass(frozen=True)
class Request:
    request_id: str
    goal_version: int
    world_version: int
    raw: bytes
    entity: str
    temporal: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    goal_raw: bytes
    goal_version: int
    world_version: int
    request: Request
    facts: tuple[Fact, ...]
    model_inputs: tuple[bytes, ...]
    target: bytes


def request_text(entity: str, temporal: str) -> bytes:
    templates = {
        "current": "{entity}现在在哪里？",
        "previous": "{entity}上一次在哪里？",
        "unknown": "{entity}的位置时间未明确。",
    }
    if temporal not in templates:
        raise ValueError("invalid temporal intent")
    return templates[temporal].format(entity=entity).encode()


def oracle_answer(request: Request, facts: tuple[Fact, ...]) -> bytes:
    if request.temporal not in {"current", "previous", "unknown"}:
        raise ValueError("invalid temporal intent")
    selected = [
        fact
        for fact in facts
        if fact.entity == request.entity and fact.version <= request.world_version
    ]
    versions = sorted({fact.version for fact in selected})
    if (
        request.temporal == "unknown"
        or not versions
        or (request.temporal == "previous" and len(versions) < 2)
    ):
        return "未知".encode()
    version = versions[-1] if request.temporal == "current" else versions[-2]
    values = {fact.value for fact in selected if fact.version == version}
    if len(values) != 1:
        raise ValueError("fixture_invalid: conflicting facts at one version")
    return next(iter(values)).encode()


def validate_case(case: Case) -> None:
    request = case.request
    if not request.request_id or not request.raw or not case.goal_raw:
        raise ValueError("missing request or goal")
    if request.goal_version != case.goal_version or request.world_version != case.world_version:
        raise ValueError("request/goal/world identity mismatch")
    if request.raw not in case.model_inputs:
        raise ValueError("request is absent from declared model inputs")
    if request.raw != request_text(request.entity, request.temporal):
        raise ValueError("microfixture request text/entity/time mismatch")
    facts = {fact.evidence_id: fact for fact in case.facts}
    if len(facts) != len(case.facts) or any(
        fact.version < 0 or fact.version > case.world_version for fact in case.facts
    ):
        raise ValueError("invalid fact identity/version")
    if any(identity not in facts for identity in request.evidence_ids):
        raise ValueError("missing evidence")
    legal = tuple(facts[identity] for identity in request.evidence_ids)
    if any(fact.source != "sensor" for fact in legal):
        raise ValueError("unauthorized evidence source")
    if case.target != oracle_answer(request, legal):
        raise ValueError("target/entity/time binding mismatch")


def validate_distinguishability(cases: tuple[Case, ...]) -> None:
    answers: dict[tuple[bytes, ...], bytes] = {}
    for case in cases:
        validate_case(case)
        previous = answers.setdefault(case.model_inputs, case.target)
        if previous != case.target:
            raise ValueError("fixture_invalid: identical legal inputs require different answers")


def allow_feedback(
    kind: str,
    *,
    split: str,
    phase: str,
    prediction_submitted: bool,
    round_index: int,
    release_round: int,
) -> None:
    if kind not in {"observation", "train_label", "online_correction", "help_reply", "score_only"}:
        raise ValueError("unregistered feedback right")
    if round_index < release_round:
        raise ValueError("feedback is not released")
    phases = {
        "observation": {"perception", "online", "development"},
        "train_label": {"development"},
        "online_correction": {"online"},
        "help_reply": {"online"},
        "score_only": {"scoring"},
    }
    if phase not in phases[kind] or split not in {"train", "fit", "check", "final"}:
        raise ValueError("feedback channel/split mismatch")
    if kind == "score_only" and phase != "scoring":
        raise ValueError("scoring oracle cannot enter cognition")
    if kind == "train_label" and (split not in {"train", "fit"} or phase != "development"):
        raise ValueError("training label outside development rights")
    if kind in {"online_correction", "help_reply"} and not prediction_submitted:
        raise ValueError("feedback arrived before prediction submission")


def score_all(
    cases: tuple[Case, ...], responses: dict[str, tuple[bytes, bool] | None]
) -> dict[str, int]:
    identifiers = [case.request.request_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate scoring request")
    correct = 0
    for case in cases:
        validate_case(case)
        response = responses.get(case.request.request_id)
        if response is None:
            continue
        content, eos = response
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        correct += int(eos and content == case.target)
    extras = len(set(responses) - set(identifiers))
    return {"correct": correct, "denominator": len(cases) + extras, "extra_responses": extras}
