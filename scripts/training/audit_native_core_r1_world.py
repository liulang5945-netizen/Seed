"""Read-only feasibility audit for the R1 shared-content world.

Only the old E1 train split is opened for byte statistics. The proposed world
is a new 12-entity, 3-attribute, 4-value, 3-version synthetic task; no data is
generated, no model is called and no old sealed/query/holdout text is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_ATTRIBUTE,
    E1_COLORS,
    E1_ENTITIES,
    build_episode,
    e1_families,
)

REPORT = ROOT / "reports/taiji_native_core_r1_world_audit_20261030.json"
COURSE = ROOT / "taiji/joint_state_consumption_course.py"

PROPOSED_ENTITIES = (
    "雨伞",
    "背包",
    "水杯",
    "耳机",
    "遥控器",
    "台灯",
    "剪刀",
    "尺子",
    "闹钟",
    "钱包",
    "帽子",
    "围巾",
)
PROPOSED_ATTRIBUTES = {
    "颜色": ("红色", "蓝色", "绿色", "黄色"),
    "位置": ("桌上", "抽屉", "柜子", "包里"),
    "归属": ("我的", "你的", "他的", "她的"),
}
VERSIONS = 3
SLOTS = 36
VALUE_POSITION_GRID = 36 * 4 * 3
SLOT_VERSION_RECORDS = SLOTS * 3
POSSIBLE_DIFFERENT_VALUE_HISTORIES = SLOTS * 4 * 3
SAMPLE_HISTORIES = 360
TRAIN_HISTORIES = 216
QUERY_HISTORIES = 72
HOLDOUT_HISTORIES = 72
EPISODES_PER_HISTORY = 4
TURNS_PER_EPISODE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _old_train_bytes() -> dict:
    family_count = episode_count = turn_count = 0
    prompt_bytes = target_bytes = 0
    for family in e1_families()["train"]:
        family_count += 1
        for variant in range(4):
            episode = build_episode(family, variant)
            episode_count += 1
            for turn in episode.turns:
                turn_count += 1
                prompt_bytes += len(bytes(turn.frame.payload))
                target_bytes += len(bytes(turn.answer_target.target_bytes))
    return {
        "partition": "train_only",
        "families": family_count,
        "episodes": episode_count,
        "turns": turn_count,
        "prompt_bytes": prompt_bytes,
        "target_bytes": target_bytes,
        "prompt_bytes_per_turn": prompt_bytes / turn_count,
        "target_bytes_per_turn": target_bytes / turn_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R1 world read-only feasibility audit")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    args.report = args.report.resolve()
    if args.report.exists():
        raise ValueError("R1 world audit report is write-once")
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    old = _old_train_bytes()
    episodes = SAMPLE_HISTORIES * EPISODES_PER_HISTORY
    turns = episodes * TURNS_PER_EPISODE
    low_bytes = int(turns * (old["prompt_bytes_per_turn"] + old["target_bytes_per_turn"]) * 1.0)
    high_bytes = int(turns * (old["prompt_bytes_per_turn"] + old["target_bytes_per_turn"]) * 1.4)
    report = {
        "schema": "taiji-native-core-r1-world-audit-v1",
        "status": "read_only_feasibility_audit",
        "world": {
            "entities": len(PROPOSED_ENTITIES),
            "attributes": len(PROPOSED_ATTRIBUTES),
            "values_per_attribute": 4,
            "versions": VERSIONS,
            "slots": SLOTS,
            "slot_version_records": SLOT_VERSION_RECORDS,
            "value_position_grid": VALUE_POSITION_GRID,
            "possible_different_value_histories": POSSIBLE_DIFFERENT_VALUE_HISTORIES,
            "possible_histories_note": (
                "different-value two-version histories; not the same object as the "
                "432 value-position grid"
            ),
            "sample_histories": SAMPLE_HISTORIES,
            "episodes_per_history": EPISODES_PER_HISTORY,
            "episodes": episodes,
            "turns_per_episode": TURNS_PER_EPISODE,
            "turns": turns,
            "note": "432 is the value-position grid, not sample/episode count",
        },
        "split_proposal": {
            "train_histories": TRAIN_HISTORIES,
            "query_histories": QUERY_HISTORIES,
            "holdout_histories": HOLDOUT_HISTORIES,
            "split_policy": (
                "split whole history chains; keep counterfactual pairs in the same split"
            ),
            "note": (
                "360-history sample is a subset of 432 possible different-value "
                "histories; exact coverage/hash split is frozen in R2/R4"
            ),
        },
        "vocabulary": {
            "entities": list(PROPOSED_ENTITIES),
            "attributes": {
                attribute: list(values) for attribute, values in PROPOSED_ATTRIBUTES.items()
            },
            "attribute_templates": 8 * len(PROPOSED_ATTRIBUTES) * 2,
            "delay_templates": 2,
        },
        "old_e1_train": old,
        "byte_estimate": {
            "method": "old E1 train prompt+target bytes per turn × new turns × 1.0–1.4",
            "low_total_bytes": low_bytes,
            "high_total_bytes": high_bytes,
            "note": "estimate only; no new data generated",
        },
        "old_face_overlap": {
            "new_entities": list(PROPOSED_ENTITIES),
            "old_entities": list(E1_ENTITIES),
            "exact_entity_overlap": sorted(set(PROPOSED_ENTITIES) & set(E1_ENTITIES)),
            "old_attribute_reused": E1_ATTRIBUTE,
            "old_colours_reused_as_atoms": sorted(
                set(sum((list(values) for values in PROPOSED_ATTRIBUTES.values()), []))
                & set(E1_COLORS)
            ),
            "old_query_holdout_text_read": 0,
            "note": "new entity set; old exact facts/queries are not reused",
        },
        "cost_evidence": {
            "old_f0b_cpu_per_update": 4.68,
            "old_f1_projection_cpu": 121500,
            "note": "not a budget; the new core needs its own R2/R4 calibration",
        },
        "risks": [
            "Chinese templates for location/owner need naturalness review",
            "4 episodes/history and 216/72/72 split need a coverage solver",
            "byte prompt length may exceed the chosen context window",
            "new core cost is not inferred from the old F0b unit cost",
        ],
        "recommendation": (
            "world vocabulary/templates and the 360-history sample plan are feasible for "
            "a first finite task; freeze exact split/coverage and cost in R2/R4"
        ),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): _sha256(Path(__file__)),
            str(COURSE.relative_to(ROOT)): _sha256(COURSE),
        },
        "model_calls": 0,
        "optimizer_updates": 0,
        "sealed_text_read": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_histories": SAMPLE_HISTORIES,
                "episodes": episodes,
                "turns": turns,
                "old_entity_overlap": report["old_face_overlap"]["exact_entity_overlap"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
