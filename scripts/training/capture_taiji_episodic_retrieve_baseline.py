"""DEBT-G106 修法① 的**改前基线**：把 `EpisodicMemoryStore.retrieve` 今天的输出封成件。

为什么要在动产品码之前封这一份
------------------------------
owner 2026-10-11 批的形状是「加 `episode_id` 参数、**默认保持现状**」。要证明"默认没变"，
不能靠改完之后自己说"看起来一样"，也不能拿改后的实现当对照（那是循环论证）。
所以本件在改动之前把**同一组输入下的命中序列**逐位封下来：
判定 `J-N6b-1` 在实施格要求实施后用同一组输入复跑，命中序列与本件**逐位相同**；
不相等 ⇒ 默认位被改了 ⇒ 那一句「默认保持现状」当场作废，要回到 owner 重新裁。

尺子先自证
----------
* **确定性**：同一进程内跑两遍，两遍必须相同（不相同 ⇒ 这把尺不能用于逐位比较，本件 rc=2）。
* **能红**：夹具里刻意造一条「与 ep-A 某条记录同形、但属于 ep-B」的记录，
  于是"跨会话命中"是**在场的事实**（基线读出来就是这个形状）；
  若哪天默认位真的按会话过滤了，`cross_episode_hits_present_in_baseline` 会翻成 false ⇒ 这一列本身能为假。

零算力：不加载检查点、不训练；只在内存里造 6 条记录。落件行尾一律 LF。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.contracts import EpisodicMemoryRecord  # noqa: E402
from taiji.episodic_memory import EpisodicMemoryStore  # noqa: E402

BASELINE_FORMAT = "taiji-episodic-retrieve-baseline-v1"

CUE_TABLE = {
    "m0": (1.0, 0.0, 0.0, 0.0),
    "m1": (0.9, 0.1, 0.0, 0.0),
    "m2": (0.0, 0.0, 1.0, 0.0),
    "m3": (1.0, 0.0, 0.0, 0.0),  # 与 m0 同 cue、不同会话 ⇒ 基线必须看见跨会话命中
    "m4": (0.0, 1.0, 0.0, 0.0),
    "m5": (0.5, 0.5, 0.0, 0.0),
}
EPISODE_OF = {"m0": "ep-A", "m1": "ep-A", "m2": "ep-A", "m3": "ep-B", "m4": "ep-B", "m5": "ep-B"}
QUERY_TABLE = {
    "q_near_m0": (1.0, 0.0, 0.0, 0.0),
    "q_mid": (0.7, 0.3, 0.0, 0.0),
    "q_axis_z": (0.0, 0.0, 1.0, 0.0),
    "q_axis_y": (0.0, 1.0, 0.0, 0.0),
}
RETRIEVE_LIMIT = 3


def _record(memory_id: str, tick: int) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=EPISODE_OF[memory_id],
        tick=tick,
        cue=torch.tensor(CUE_TABLE[memory_id], dtype=torch.float32),
        provenance="experienced",
    )


def _build_store() -> EpisodicMemoryStore:
    store = EpisodicMemoryStore(capacity=8, cue_dim=4)
    for tick, memory_id in enumerate(sorted(CUE_TABLE), start=1):
        store.write(_record(memory_id, tick))
    return store


def _one_pass() -> dict[str, object]:
    store = _build_store()
    hits: dict[str, list[list[str]]] = {}
    for query, values in QUERY_TABLE.items():
        cue = torch.tensor(values, dtype=torch.float32)
        hits[query] = [
            [hit.record.memory_id, repr(float(hit.score))]
            for hit in store.retrieve(cue, limit=RETRIEVE_LIMIT)
        ]
    episodes = sorted({record.episode_id for record in store.records})
    cross = any(
        len(
            {
                EPISODE_OF[hit.record.memory_id]
                for hit in store.retrieve(
                    torch.tensor(QUERY_TABLE[query], dtype=torch.float32), limit=RETRIEVE_LIMIT
                )
            }
        )
        > 1
        for query in QUERY_TABLE
    )
    return {
        "store_count": store.count,
        "store_episodes": episodes,
        "retrieve_limit": RETRIEVE_LIMIT,
        "hits": hits,
        "cross_episode_hits_present_in_baseline": cross,
    }


def build_baseline() -> dict[str, object]:
    first = _one_pass()
    second = _one_pass()
    if first["hits"] != second["hits"]:
        raise ValueError(
            "同一组输入两遍命中序列不同 ⇒ 这把尺不能用于逐位比较，基线件不予出版"
            f"（第一遍 {first['hits']}／第二遍 {second['hits']}）"
        )
    canonical = json.dumps(first["hits"], ensure_ascii=False, sort_keys=True)
    return {
        "format": BASELINE_FORMAT,
        "purpose": "DEBT-G106 修法① 的 J-N6b-1 兼容锚：实施后同一组输入的命中序列必须与本件逐位相同",
        "captured_before_change": True,
        "capture_head": _head_commit(),
        "input_face": {
            "capacity": 8,
            "cue_dim": 4,
            "records": {memory_id: EPISODE_OF[memory_id] for memory_id in sorted(CUE_TABLE)},
            "queries": QUERY_TABLE,
        },
        "store_count": first["store_count"],
        "store_episodes": first["store_episodes"],
        "retrieve_limit": first["retrieve_limit"],
        "hits": first["hits"],
        "hits_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "determinism_passes_equal": True,
        "cross_episode_hits_present_in_baseline": first["cross_episode_hits_present_in_baseline"],
        "limits": [
            "scores 用 repr(float) 出版，逐位比较按字符串比，不做四舍五入",
            "本件只覆盖 `retrieve(cue, *, limit)` 这一个入口；`records`／`expire`／`checkpoint` 不在锚定面内",
            "夹具是合成的（6 条记录、4 维线索），不是盘上任何一枚检查点的真实情节场；"
            "它锚的是**函数行为**，不是某一次跑的读数",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _head_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-report",
        default="reports/taiji_episodic_retrieve_baseline_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        baseline = build_baseline()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing baseline: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "store_count": baseline["store_count"],
                "query_count": len(baseline["hits"]),
                "hits_digest": baseline["hits_digest"],
                "cross_episode_hits_present_in_baseline": baseline[
                    "cross_episode_hits_present_in_baseline"
                ],
                "out": out.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
