"""Glossary delivered as memory records, not as part of the goal prompt.

Putting the glossary inside the prompt would put the entity names back into the text
the model reads, and 10.8/10.9 showed that single-hop overlap then wins again. So the
glossary travels in ``CareerJournal.records`` while the prompt stays entity-free:

    prompt  = "让圆口的器皿到靠窗的台面。"     (no entity names)
    records = ("圆口的器皿是蓝色杯子。", ...)   (the only bridge)

``journal_with_records`` reuses the scorer's own constructor and swaps the records in,
so the no-record path stays byte-identical.
"""

from __future__ import annotations

from dataclasses import replace

from taiji.living_core import RawSource
from taiji.native_goal_attributes import OBJECT_DESCRIPTIONS, PLACE_DESCRIPTIONS
from taiji.subject_action_scorer import action_journal


def glossary_records(reverse: bool = False) -> tuple[RawSource, ...]:
    object_items = list(OBJECT_DESCRIPTIONS.items())
    place_items = list(PLACE_DESCRIPTIONS.items())
    if reverse:
        object_items = list(reversed(object_items))
        place_items = list(reversed(place_items))
    lines = [f"{text}是{entity}。" for entity, text in object_items]
    lines += [f"{text}是{entity}。" for entity, text in place_items]
    return tuple(
        RawSource(source_id=f"glossary-{index}", raw=line.encode())
        for index, line in enumerate(lines)
    )


def journal_with_records(goal_text: str, sentence: str, subject_id: str, records):
    journal = action_journal(goal_text, sentence, subject_id)
    return replace(journal, records=tuple(records))
