"""N178 的固定新面（face2）：只用**训练/评测里都没出现过**的查询措辞，量"能否抽出实体"。

为什么只做问句：N178 的验收尺是"查询轮解析覆盖率"，它只需要（文本，应抽出的实体）这一对；
重造整条课程（含事实轮、答案目标、标注区间）既容易出错，也会把"新面"变成"新课程"。

面一旦登记就冻结：模板、实体、顺序、摘要全部写进件里；训练前登记、训练后只读一次。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .internalization import content_digest

ROOT = Path(__file__).resolve().parents[1]
FACE2_FORMAT = "taiji-n178-face2-query-coverage-v1"

# 每一条都刻意避开课程里已有的问法（"…的颜色是什么？"／"…之前是什么颜色？"）。
ORDINARY_TEMPLATES = (
    "请问，{entity}是什么颜色的？",
    "{entity}的颜色，能告诉我吗？",
    "我想知道一下{entity}的颜色。",
    "帮我查查{entity}是什么色。",
)
OLD_TEMPLATES = (
    "{entity}原来是什么颜色？",
    "之前那个{entity}是什么色的？",
    "{entity}以前是什么色儿？",
    "那个{entity}原先是什么颜色？",
)
ENTITIES = ("钥匙", "杯子", "书", "盒子", "球", "手机", "笔", "眼镜")
COVERED = ("钥匙", "杯子", "书", "盒子")


@dataclass(frozen=True)
class Face2Item:
    item_id: str
    template_class: str
    entity: str
    text: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "template_class": self.template_class,
            "entity": self.entity,
            "text": self.text,
        }


def face2_items() -> tuple[Face2Item, ...]:
    items = []
    for template_class, templates in (
        ("ordinary", ORDINARY_TEMPLATES),
        ("old", OLD_TEMPLATES),
    ):
        for t_index, template in enumerate(templates):
            for e_index, entity in enumerate(ENTITIES):
                items.append(
                    Face2Item(
                        item_id=f"face2:{template_class}:{t_index}:{e_index}",
                        template_class=template_class,
                        entity=entity,
                        text=template.format(entity=entity),
                    )
                )
    return tuple(items)


def face2_contract() -> dict[str, Any]:
    items = face2_items()
    return {
        "format": FACE2_FORMAT,
        "date": "2026-10-30",
        "templates": {
            "ordinary": list(ORDINARY_TEMPLATES),
            "old": list(OLD_TEMPLATES),
        },
        "entities": list(ENTITIES),
        "covered_entities": list(COVERED),
        "items": [item.to_payload() for item in items],
        "item_count": len(items),
        "face_sha256": content_digest([item.to_payload() for item in items]),
    }


def leakage_check(existing_texts: set[str]) -> dict[str, Any]:
    """新面与既有题面不得有同一条文本；模板骨架也不得作为子串出现。"""

    items = face2_items()
    duplicated = [item.item_id for item in items if item.text in existing_texts]
    skeleton_hits = {}
    for name, templates in (("ordinary", ORDINARY_TEMPLATES), ("old", OLD_TEMPLATES)):
        hits = 0
        for template in templates:
            skeleton = template.replace("{entity}", "")
            hits += sum(1 for text in existing_texts if skeleton in text)
        skeleton_hits[name] = hits
    return {
        "duplicate_texts": duplicated,
        "skeleton_hits_in_existing": skeleton_hits,
        "clean": not duplicated and not any(skeleton_hits.values()),
    }
