"""Contract guard for 10.11: the glossary travels in records, not in the prompt."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from taiji.native_goal_attributes import OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS  # noqa: E402
from taiji.native_goal_glossary import glossary_records, journal_with_records  # noqa: E402


def test_records_cover_every_entity():
    records = glossary_records()
    assert len(records) == len(OBJECT_DESCRIPTIONS) + len(PLACE_DESCRIPTIONS)
    text = "".join(source.raw.decode() for source in records)
    for entity in {*OBJECT_DESCRIPTIONS, *PLACE_DESCRIPTIONS}:
        assert entity in text
    assert len({source.source_id for source in records}) == len(records)


def test_reverse_swaps_the_mapping_order():
    forward = [source.raw for source in glossary_records()]
    backward = [source.raw for source in glossary_records(reverse=True)]
    assert forward != backward
    assert set(forward) == set(backward)


def test_journal_keeps_the_prompt_entity_free():
    goal = "让圆口的器皿到靠窗的台面。"
    journal = journal_with_records(goal, "把蓝色杯子放到窗台", "t", glossary_records())
    assert journal.prompt == goal.encode()
    assert len(journal.records) == len(glossary_records())
    plain = journal_with_records(goal, "把蓝色杯子放到窗台", "t", ())
    assert plain.records == ()
