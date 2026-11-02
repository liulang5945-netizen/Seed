"""Deterministic natural-Chinese task fixtures for the R029/R030 engineering path.

These fixtures expose only raw bytes, source identity, version and observation
time to the model. Categories and expected answers are scoring metadata and must
not be passed to runtime. The generator is fixed and reads no corpus split.
"""

from __future__ import annotations

from dataclasses import dataclass

from taiji.living_core import RawSource
from taiji.state_migration import CareerJournal, career_step_specs

NU = (0.5, 2.0, 0.0, 0.8, 8.0, -0.4, 0.0, 0.3)
KNOWN = (True, True, False, True, True, True, False, True)
GOAL = "根据已见材料回答。".encode()
ABSTAIN_UNKNOWN = "未知。".encode()
ABSTAIN_CONFLICT = "来源冲突。".encode()
SHORTCUTS = ("latest_source", "earliest_source", "abstain")


@dataclass(frozen=True)
class TaskFixture:
    fixture_id: str
    category: str
    journal: CareerJournal
    expected_target: bytes
    answer_denominator: int
    shortcut_label: str


def _journal(subject: str, question: str, target: str, records: tuple[RawSource, ...]):
    return CareerJournal(
        subject_id=subject,
        prompt=question.encode(),
        target=target.encode(),
        goal=GOAL,
        records=records,
        raw_nu=NU,
        known_nu=KNOWN,
        now=10,
    )


def current_correction_records() -> tuple[RawSource, ...]:
    return (
        RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=1),
        RawSource("s2", "更正：蓝色杯子在窗台。".encode(), version=2, observed_at=5),
    )


def previous_records() -> tuple[RawSource, ...]:
    return current_correction_records()


def unknown_records() -> tuple[RawSource, ...]:
    return (
        RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=1),
        RawSource("s2", "红色印章在左盒。".encode(), version=1, observed_at=2),
    )


def alias_records() -> tuple[RawSource, ...]:
    return (
        RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=1),
        RawSource("s2", "后来蓝色杯子在窗台。".encode(), version=2, observed_at=5),
    )


def conflict_records() -> tuple[RawSource, ...]:
    return (
        RawSource("s1", "蓝色杯子在木桌上。".encode(), version=1, observed_at=5),
        RawSource("s2", "蓝色杯子在抽屉里。".encode(), version=1, observed_at=5),
    )


def build_fixtures() -> tuple[TaskFixture, ...]:
    fixtures = tuple(
        TaskFixture(
            fixture_id=fixture_id,
            category=category,
            journal=journal,
            expected_target=journal.target,
            answer_denominator=len(journal.target) + 1,
            shortcut_label=shortcut_label,
        )
        for fixture_id, category, journal, shortcut_label in (
            (
                "current-correction",
                "current",
                _journal(
                    "current-correction",
                    "蓝色杯子现在在哪里？",
                    "蓝色杯子在窗台。",
                    current_correction_records(),
                ),
                "latest_source",
            ),
            (
                "previous-before-correction",
                "previous",
                _journal(
                    "previous-before-correction",
                    "更正前蓝色杯子在哪里？",
                    "蓝色杯子在木桌上。",
                    previous_records(),
                ),
                "earliest_source",
            ),
            (
                "unknown-missing",
                "unknown",
                _journal(
                    "unknown-missing",
                    "绿色钥匙现在在哪里？",
                    "未知。",
                    unknown_records(),
                ),
                "abstain",
            ),
            (
                "alias-current",
                "alias",
                _journal(
                    "alias-current",
                    "蓝杯现在在哪里？",
                    "蓝色杯子在窗台。",
                    alias_records(),
                ),
                "latest_source",
            ),
            (
                "source-conflict",
                "conflict",
                _journal(
                    "source-conflict",
                    "蓝色杯子现在在哪里？",
                    "来源冲突。",
                    conflict_records(),
                ),
                "abstain",
            ),
        )
    )
    self_check(fixtures)
    return fixtures


def answer_rounds(fixture: TaskFixture) -> int:
    return sum(1 for spec in career_step_specs(fixture.journal) if spec["kind"] == "round")


def shortcut_answer(fixture: TaskFixture, strategy: str) -> bytes:
    records = fixture.journal.records
    if strategy == "latest_source":
        return records[-1].raw
    if strategy == "earliest_source":
        return records[0].raw
    if strategy == "abstain":
        return ABSTAIN_UNKNOWN if fixture.category == "unknown" else ABSTAIN_CONFLICT
    raise ValueError(f"unknown shortcut strategy {strategy}")


def answer_correct(fixture: TaskFixture, answer: bytes) -> bool:
    return fixture.expected_target in answer


def with_records(fixture: TaskFixture, records: tuple[RawSource, ...]) -> TaskFixture:
    journal = CareerJournal(
        subject_id=fixture.journal.subject_id,
        prompt=fixture.journal.prompt,
        target=fixture.journal.target,
        goal=fixture.journal.goal,
        records=records,
        raw_nu=fixture.journal.raw_nu,
        known_nu=fixture.journal.known_nu,
        mode=fixture.journal.mode,
        now=fixture.journal.now,
        allowed_write=fixture.journal.allowed_write,
    )
    return TaskFixture(
        fixture_id=fixture.fixture_id,
        category=fixture.category,
        journal=journal,
        expected_target=fixture.expected_target,
        answer_denominator=fixture.answer_denominator,
        shortcut_label=fixture.shortcut_label,
    )


def shortcut_probe(fixtures: tuple[TaskFixture, ...]) -> dict:
    """Score position/abstention shortcuts; none may reach full marks."""
    scores = {}
    for strategy in SHORTCUTS:
        per_fixture = {}
        for fixture in fixtures:
            per_fixture[fixture.fixture_id] = answer_correct(
                fixture, shortcut_answer(fixture, strategy)
            )
        scores[strategy] = {
            "correct": sum(per_fixture.values()),
            "total": len(fixtures),
            "per_fixture": per_fixture,
        }
    reversed_scores = {}
    for strategy in SHORTCUTS:
        correct = 0
        for fixture in fixtures:
            permuted = with_records(fixture, tuple(reversed(fixture.journal.records)))
            correct += answer_correct(permuted, shortcut_answer(permuted, strategy))
        reversed_scores[strategy] = correct
    return {"as_given": scores, "record_order_reversed": reversed_scores}


def self_check(fixtures: tuple[TaskFixture, ...]) -> None:
    ids = [item.fixture_id for item in fixtures]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate fixture ids")
    categories = {item.category for item in fixtures}
    required = {"current", "previous", "unknown", "alias", "conflict"}
    if not required <= categories:
        raise ValueError(f"missing categories: {sorted(required - categories)}")
    for item in fixtures:
        if not item.expected_target:
            raise ValueError(item.fixture_id)
        item.expected_target.decode("utf-8")
        records = item.journal.records
        if len(records) < 2:
            raise ValueError(item.fixture_id)
        for record in records:
            if not record.source_id or not record.raw or record.version < 1:
                raise ValueError(item.fixture_id)
            record.raw.decode("utf-8")
        if item.answer_denominator != answer_rounds(item):
            raise ValueError(f"{item.fixture_id} denominator is not the answer-round count")
        if item.answer_denominator != len(item.journal.target) + 1:
            raise ValueError(item.fixture_id)
        if item.shortcut_label not in SHORTCUTS:
            raise ValueError(item.fixture_id)
    probe = shortcut_probe(fixtures)
    for strategy, row in probe["as_given"].items():
        if row["correct"] == 0:
            raise ValueError(f"shortcut {strategy} scores zero: the ruler has no dynamic range")
        if row["correct"] == row["total"]:
            raise ValueError(f"shortcut {strategy} reaches full marks")
    if probe["record_order_reversed"] == {
        strategy: row["correct"] for strategy, row in probe["as_given"].items()
    }:
        raise ValueError("fixture set is insensitive to record order")
    if any(item.category == "gold" for item in fixtures):
        raise ValueError("category must not leak into runtime payloads")
