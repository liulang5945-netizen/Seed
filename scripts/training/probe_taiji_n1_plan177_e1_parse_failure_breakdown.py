"""N177 解析缺口分解（训练侧只读）：没解析出来的查询轮，缺的是实体词表还是表面形式？

对每条 train episode 的查询轮，记下 producer 给出的 role span：
- 有 entity ⇒ 解析成功；
- 没 entity ⇒ 看它属于"已覆盖实体"（钥匙/杯子/书/盒子）还是"未覆盖实体"（球/手机/笔/眼镜），
  以及缺的是哪几个 role。两类成因指向完全不同的下一步：
  词表缺口靠补数据/扩实体，表面形式缺口靠让模型见过更多说法。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.evaluate_taiji_n1_plan177_f_f1_holdout import (
    _replay,
    load_trainer,
)  # noqa: E402
from taiji.joint_state_consumption_course import build_train_episodes, e1_families  # noqa: E402

REPORT = ROOT / "reports/taiji_n1_plan177_e1_parse_failure_breakdown_20261029.json"
COVERED = ("钥匙", "杯子", "书", "盒子")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 解析缺口分解（训练侧）")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("breakdown report is write-once")

    entity_by_family = {family.family_id: family.entity for family in e1_families()["train"]}
    trainer = load_trainer()
    rows: list[dict[str, Any]] = []
    for episode in build_train_episodes():
        entity = entity_by_family[episode.family_id]
        for index, row in enumerate(_replay(trainer, episode)):
            if row["act"] != "query":
                continue
            roles = {role: value for role, value in row["spans"].items() if value}
            rows.append(
                {
                    "episode": episode.episode_id,
                    "turn": index,
                    "family_entity": entity,
                    "entity_covered": entity in COVERED,
                    "parsed_entity": bool(roles.get("entity")),
                    "roles_present": sorted(roles),
                }
            )

    by_group = collections.Counter()
    for row in rows:
        key = ("covered" if row["entity_covered"] else "uncovered", row["parsed_entity"])
        by_group[key] += 1
    missing_roles = collections.Counter(
        role
        for row in rows
        if not row["parsed_entity"]
        for role in ("entity", "attribute", "value")
        if role not in row["roles_present"]
    )
    report = {
        "format": "taiji-n177-e1-parse-failure-breakdown-v1",
        "date": "2026-10-29",
        "scope": "train side only: query turns of all 128 train episodes",
        "query_turns": len(rows),
        "parsed": sum(row["parsed_entity"] for row in rows),
        "unparsed": sum(not row["parsed_entity"] for row in rows),
        "by_group": {
            "covered_parsed": by_group[("covered", True)],
            "covered_unparsed": by_group[("covered", False)],
            "uncovered_parsed": by_group[("uncovered", True)],
            "uncovered_unparsed": by_group[("uncovered", False)],
        },
        "missing_roles_when_unparsed": dict(missing_roles),
        "rows": rows,
    }
    report["reading"] = "未覆盖实体全部失败 ⇒ 词表缺口为主；已覆盖实体也大量失败 ⇒ 表面形式缺口为主"
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
