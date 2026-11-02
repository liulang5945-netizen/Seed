"""A fixed-answer Chinese task suite for 11.3, with the rubric written next to it.

Five items are scorable by exact match. Two items (clarify, unknown) have no scorable
answer because the current entry has no model-driven clarif/abstain channel; they are
recorded but kept out of the denominator, and that exclusion is reported.

The prompts are meant to be asked in order in one session: the follow-up and the
correction only make sense after the material item.
"""

from __future__ import annotations

STRIP = " \t\r\n。！？!?，,"


def normalize(text: str) -> str:
    return text.strip().strip(STRIP)


TASKS = (
    {
        "id": "qa-1",
        "category": "问答",
        "prompt": "中国的首都是哪里？答：",
        "expected": "北京",
        "scored": True,
    },
    {
        "id": "material-1",
        "category": "材料任务",
        "prompt": "材料：钥匙在抽屉里。问：钥匙在哪里？答：",
        "expected": "抽屉",
        "scored": True,
    },
    {
        "id": "followup-1",
        "category": "追问",
        "prompt": "刚才的钥匙在哪里？答：",
        "expected": "抽屉",
        "scored": True,
    },
    {
        "id": "correct-1",
        "category": "更正",
        "prompt": "更正：钥匙在桌上。问：钥匙在哪里？答：",
        "expected": "桌上",
        "scored": True,
    },
    {
        "id": "instruct-1",
        "category": "指令",
        "prompt": "请把「今天下雨」改成否定句。答：",
        "expected": "今天不下雨",
        "scored": True,
    },
    {
        "id": "clarify-1",
        "category": "澄清",
        "prompt": "把那个东西放到那里。",
        "expected": None,
        "scored": False,
        "note": "correct handling is a model-driven clarify; the entry has no such channel",
    },
    {
        "id": "unknown-1",
        "category": "未知",
        "prompt": "火星上第一座城市叫什么？答：",
        "expected": None,
        "scored": False,
        "note": "correct handling is abstain; the entry has no such channel",
    },
)
