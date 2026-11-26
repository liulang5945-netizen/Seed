r"""债表"有效状态"普查仪：找出**头部状态格已过期、行尾日期戳才是现状**的那些行。

为什么需要它（2026-10-17 实测）：156 行里有 72 行的状态格里带 `〔…〕` 就地更正（最多一行 8 枚），
而按列切格去读"状态"的人**结构上看不见**同一行行尾追加的那句——我就这样把已结清的 DEBT-G125
当成"仅剩的未修格"写进了对外文档（08 ㊵-1041 收回）。本件把这件事变成一次可复算的门。

三条自我约束（都是本仓的老账，写在行为里而不是口号里）：
* **不假装读懂**：状态词表逐枚公开在件里（`vocabulary`），所以它能抓什么、抓不到什么都是可查的；
  表外的行一律落进 `unclassified`，**不许静默算过**。
* **两个计数并列**：`strict_rows`（`| DEBT-Gnn |` 形状，与债表列形门同一把尺）与
  `prefix_lines`（任何以 `| DEBT-G` 开头的行，含 `| DEBT-G48 状态更新（…） |` 这种追加行）；
  差值逐枚点名，避免"两台仪器各说一个数"。
* **无默认写靶**：不给 `--out-report` 只印 stdout，绝不落到 `reports/` 里任何在库件（DEBT-G35／G50 那一族）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "plans/active/roadmap/05_TECH_DEBT_REGISTER.md"

STRICT_ROW = re.compile(r"^\| (DEBT-G\d+) \|")
PREFIX_ROW = re.compile(r"^\| (DEBT-G\d+)")
BRACKET = re.compile(r"〔(.*?)〕", re.S)

#: 状态词表（公开）。CLOSED 只收"这行已经不用再做"的措辞；OPEN 只收"还欠动作"的措辞。
VOCABULARY = {
    "closed": (
        "结清",
        "已结清",
        "已修",
        "已落地",
        "已做完",
        "全部做完",
        "已实施",
        "已交付",
        "已兑现",
        "已结",
        "已清账",
        "本行结清",
        "已收口",
        "已归零",
    ),
    "open": (
        "只登记",
        "未修",
        "未实现",
        "未做",
        "未开工",
        "未启动",
        "未执行",
        "未落地",
        "未兑现",
        "未结清",
        "未登记",
        "待裁",
        "待批",
        "待 owner",
        "等 owner",
    ),
}

#: 状态词前面若挂着否定字，就不算"已做完"——`未结清` 里含着 `结清`，
#: 不加这一道守卫会把"还欠着"的行读成"已结清"，正是本件要抓的那类错的镜像。
NEGATIONS = ("未", "没", "不")


def classify(region: str) -> tuple:
    """按"区域里最后出现的一个状态词"定状态；两者都没出现就返回 unknown（不自作主张）。"""
    best = None
    for state in ("closed", "open"):
        for token in VOCABULARY[state]:
            position = region.rfind(token)
            if position < 0:
                continue
            if state == "closed" and position > 0 and region[position - 1] in NEGATIONS:
                continue
            if best is None or position > best[0]:
                best = (position, state, token)
    if best is None:
        return ("unknown", None)
    return (best[1], best[2])


def split_status_cell(row: str) -> str:
    cells = row.rstrip().split(" | ")
    return cells[-1] if len(cells) > 1 else ""


def analyse(row: str) -> dict:
    status = split_status_cell(row)
    first_stamp = status.find(chr(0x3014))
    head = status if first_stamp < 0 else status[:first_stamp]
    stamps = BRACKET.findall(row)
    #: 状态取"最后一个**带状态词**的戳"，不是字面上最后那枚——本仓常在结清之后又追加
    #: 与状态无关的注（DEBT-G125 就是：㊵-755"本行结清"之后还有一枚自述适用面缺陷的戳），
    #: 按字面最后一枚会把"已结清"读成 unknown，等于放过我要抓的那一行。
    deciding = None
    for stamp in stamps:
        state, token = classify(stamp)
        if state != "unknown":
            deciding = (state, token, stamp)
    head_state, head_token = classify(head)
    return {
        "head_state": head_state,
        "head_token": head_token,
        "stamp_count": len(stamps),
        "stamp_state": deciding[0] if deciding else ("none" if not stamps else "unknown"),
        "stamp_token": deciding[1] if deciding else None,
        "head_text": head.strip()[:90],
        "last_stamp_text": (stamps[-1] if stamps else "").strip()[:160],
        "deciding_stamp_text": (deciding[2] if deciding else "").strip()[:160],
        # 我栽的那一形：头部说"还欠"，最新带状态的戳说"已做完"。
        "stale_head": head_state == "open" and bool(deciding) and deciding[0] == "closed",
        # 反向那一形：头部说"已做完"而最新带状态的戳又把它打开（更少见但更要命）。
        "reopened_by_stamp": head_state == "closed" and bool(deciding) and deciding[0] == "open",
        "unclassified_stamp": bool(stamps) and deciding is None,
    }


def measure(text: str) -> dict:
    rows = [line for line in text.split("\n") if PREFIX_ROW.match(line)]
    strict_ids = [STRICT_ROW.match(line).group(1) for line in rows if STRICT_ROW.match(line)]
    prefix_ids = [PREFIX_ROW.match(line).group(1) for line in rows]
    #: 差值不靠集合差点名（追加行的编号与严行相同，集合差恒为空），而是点名"编号出现多次"的行。
    duplicates = sorted(
        identifier
        for identifier in set(prefix_ids)
        if prefix_ids.count(identifier) > strict_ids.count(identifier)
    )
    per_row = {}
    for line in rows:
        identifier = PREFIX_ROW.match(line).group(1)
        reading = analyse(line)
        # 同一编号可有两行（`| DEBT-G48 状态更新… |` 那种追加行）；逐行都算，再按编号合并布尔。
        previous = per_row.get(identifier)
        if previous is None:
            per_row[identifier] = reading
            continue
        for key in ("stale_head", "reopened_by_stamp", "unclassified_stamp"):
            previous[key] = previous[key] or reading[key]
        if reading["last_stamp_text"]:
            previous["extra_stamps"] = previous.get("extra_stamps", []) + [
                reading["last_stamp_text"]
            ]
    return {
        "counts": {
            "prefix_lines": len(prefix_ids),
            "strict_rows": len(strict_ids),
            "extra_lines_with_a_repeated_id": duplicates,
            "distinct_ids": len(per_row),
            "rows_with_stamps": sum(1 for value in per_row.values() if value["stamp_count"]),
            "max_stamps_in_one_row": max(
                (value["stamp_count"] for value in per_row.values()), default=0
            ),
        },
        "stale_head": sorted(key for key, value in per_row.items() if value["stale_head"]),
        "reopened_by_stamp": sorted(
            key for key, value in per_row.items() if value["reopened_by_stamp"]
        ),
        "unclassified_stamp": sorted(
            key for key, value in per_row.items() if value["unclassified_stamp"]
        ),
        "per_row": per_row,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--top", type=int, default=12)
    arguments = parser.parse_args(argv)
    text = arguments.table.read_text(encoding="utf-8")
    result = measure(text)
    counts = result["counts"]
    # 空普查＝仪器自己失效：表读到了却一行没分类，必须响亮失败。
    assert counts["prefix_lines"] > 0, "the register parsed to zero rows"
    assert (
        counts["rows_with_stamps"] > 0
    ), "no in-place correction found anywhere: check the bracket codepoint"
    payload = {
        "format": "taiji-debt-table-effective-status-census-v1",
        "table": str(arguments.table),
        "table_sha256": hashlib.sha256(arguments.table.read_bytes()).hexdigest(),
        "purpose": "找出头部状态格已过期、行尾日期戳才是现状的行（DEBT-G154 的仪器化；成因＝㊵-1041 我误引用已结清的 G125）",
        "vocabulary": VOCABULARY,
        "reading_rule": "状态＝该区域里最后出现的一个状态词；表外措辞记 unknown 并进 unclassified_stamp，不自作主张",
        "cannot_catch": [
            "用表外措辞写的更正（会落进 unclassified_stamp，不静默）",
            "同一行内没有日期戳的口头性过期（本件不判）",
            "跨行引用（某行说'见 Gnn 已结清'而 Gnn 其实没结）",
        ],
        **result,
    }
    if arguments.out_report is None:
        print(
            json.dumps(
                {
                    k: payload[k]
                    for k in ("counts", "stale_head", "reopened_by_stamp", "unclassified_stamp")
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.out_report.exists():
        print(f"REFUSE: report target already exists: {arguments.out_report}")
        return 2
    arguments.out_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.out_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "counts": counts,
                "stale_head": payload["stale_head"][: arguments.top],
                "reopened_by_stamp": payload["reopened_by_stamp"][: arguments.top],
                "unclassified_stamp_count": len(payload["unclassified_stamp"]),
                "out": str(arguments.out_report),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
