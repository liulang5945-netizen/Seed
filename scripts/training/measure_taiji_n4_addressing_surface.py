"""N4 寻址面仪器（PLAN-N4-01 §2/§5 的落地件；只读、零训练、不改权重、不动产品码）。

为什么要有这台仪器（缺陷本体，DEBT-G58）：规划把 N4 的痛点写成"34.6% 空库/寻址失败"，
而 `EpisodicMemoryStore.retrieve`（`taiji/episodic_memory.py:60-89`）对非空库**无条件返回前 `limit` 条**
⇒ "零命中率"恒等于"空库率"，"寻址失败"作为独立状态根本产生不出来。
所以母量按 PLAN-N4-01 §2 冻的那样只有一条：**取回的是不是该取回的那条**（`wrong_top1_rate`）。

三条硬规矩：
* 不重抄生成链——排序与打分全走产品 `EpisodicMemoryStore`，本件只数不算第二遍余弦；
* 能为 false 两面都测——"零干扰"夹具必须判出 `ruler_usable=false`（尺没动态范围），
  故意造干扰的夹具必须判出 `ruler_usable=true`；
* 产品默认链的挂载态**由本件自述为 `harness`**，不许被读侧当成产品读数（§5 G-N4-3）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from taiji.contracts import EpisodicMemoryRecord  # noqa: E402
from taiji.episodic_memory import EpisodicMemoryStore  # noqa: E402

REPORT_FORMAT = "taiji-n4-addressing-surface-v1"
#: adapter 层的第四态在本件不可观测：它需要 `_recovery_memory_is_readable`（`taiji/adapter.py:10838`）
#: 与被撤销/未被选中的恢复名册，而那是适配器层的事。这里响亮写"不可观测"，
#: 免得读侧把"这个键不存在"读成"没有被过滤的命中"。
NOT_OBSERVABLE = "not_observable_here"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"{label} 文件不存在：{path}")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{label} 第 {number} 行不是对象")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} 件为空：{path}")
    return rows


def _cue(row: dict[str, Any], label: str, number: int) -> torch.Tensor:
    values = row.get("cue")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} 第 {number} 行缺非空的数值 `cue` 列表")
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def _block_means(flags: list[int], blocks: int = 5) -> list[float]:
    """按查询顺序切 `blocks` 段取段均值（PLAN-N3-02 §0 的自取形状，不引外部阈值）。"""

    size = max(1, len(flags) // blocks)
    means: list[float] = []
    for start in range(0, len(flags), size):
        chunk = flags[start : start + size]
        if chunk:
            means.append(sum(chunk) / len(chunk))
    return means


#: PLAN-N4-03 冻结的 S4 稀疏编码：随机投影 32→256（seed 入档）＋ top-16 WTA。
#: 投影矩阵按首次使用时的 cue_dim 现数并缓存（同进程内逐位同）；cue_dim 必须 =32
#: （PLAN-N4-03 §1 冻结的 d_in），其他维度响亮拒绝而不是静默换尺。
SPARSE_SEED = 20260822
SPARSE_DIM = 256
SPARSE_K = 16
_SPARSE_PROJECTION: torch.Tensor | None = None


def _sparse_projection(cue_dim: int) -> torch.Tensor:
    global _SPARSE_PROJECTION
    if cue_dim != 32:
        raise ValueError(
            f"PLAN-N4-03 冻结的 S4 编码 d_in=32，收到 cue_dim={cue_dim} ⇒ 拒绝（换尺须另批）"
        )
    if _SPARSE_PROJECTION is None:
        generator = torch.Generator().manual_seed(SPARSE_SEED)
        _SPARSE_PROJECTION = torch.randn(32, SPARSE_DIM, generator=generator)
    return _SPARSE_PROJECTION


def _maybe_encode(cue: torch.Tensor, encoding: str) -> torch.Tensor:
    if encoding == "dense":
        return cue
    if encoding == "sparse":
        return _sparse_encode(cue)
    raise ValueError(f"cue_encoding must be dense or sparse, got {encoding!r}")


def _sparse_encode(cue: torch.Tensor) -> torch.Tensor:
    projected = cue.to(torch.float32) @ _sparse_projection(int(cue.numel()))
    values, indices = torch.topk(projected.abs(), SPARSE_K)
    sparse = torch.zeros_like(projected)
    for pos, idx in enumerate(indices):
        sparse[idx] = torch.sign(projected[idx]) * values[pos]
    return sparse


def measure(
    materials: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    capacity: int,
    limit: int,
    cue_encoding: str = "dense",
) -> dict[str, Any]:
    if cue_encoding not in ("dense", "sparse"):
        raise ValueError(f"cue_encoding must be dense or sparse, got {cue_encoding!r}")
    store = EpisodicMemoryStore(capacity=capacity)
    for number, row in enumerate(materials, start=1):
        memory_id = row.get("memory_id")
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError(f"materials 第 {number} 行缺 `memory_id`")
        store.write(
            EpisodicMemoryRecord(
                memory_id=memory_id,
                episode_id=str(row.get("episode_id", f"episode-{number}")),
                tick=int(row.get("tick", number)),
                cue=_maybe_encode(_cue(row, "materials", number), cue_encoding),
                action_intent=None,
                outcome=None,
                world_transition=None,
                prediction_error=0.0,
                provenance=str(row.get("provenance", "n4-01-fixture")),
                event_ids=(),
                assembly_ids=(),
                object_ids=(),
                relation_ids=(),
            )
        )

    roster = {record.memory_id for record in store.records}
    #: 三态可数（`store_absent` 由本件自述恒 0），第四态 `all_hits_gated` 在 store 层**不可观测**，
    #: 所以下面的等式只cover可数的三态——不许把"没这个键"读成"没有被过滤的命中"。
    state_counts = {"store_empty": 0, "emitted": 0}
    wrong_flags: list[int] = []
    rank_flags: list[int] = []
    top1_scores: list[float] = []
    outside_top_limit = 0
    unpaired = 0

    for number, row in enumerate(queries, start=1):
        expected = row.get("expected_memory_id")
        cue = _maybe_encode(_cue(row, "queries", number), cue_encoding)
        if store.count == 0:
            state_counts["store_empty"] += 1
            continue
        hits = store.retrieve(cue, limit=limit)
        if not hits:
            #: 非空库走到这里＝产品语义被改动（现行实现只在 `limit<=0` 或空库时给 `()`）⇒ 响亮失败。
            raise ValueError(f"queries 第 {number} 行在**非空库**上拿到零命中，与本件的前提冲突")
        state_counts["emitted"] += 1
        top1_scores.append(float(hits[0].score))
        if not isinstance(expected, str) or expected not in roster:
            unpaired += 1
            continue
        ranks = [hit.record.memory_id for hit in hits]
        #: 真值可能落在 top-`limit` **之外**（`limit=1` 时最常见）。这一格不许让 `.index()` 抛错，
        #: 也不许把它当成"排名第 1 之外"和"根本没取回"混成一件事——它单独计数。
        if expected in ranks:
            rank_flags.append(ranks.index(expected) + 1)
            wrong_flags.append(int(ranks[0] != expected))
        else:
            outside_top_limit += 1
            wrong_flags.append(1)

    paired = len(wrong_flags)
    wrong_rate = (sum(wrong_flags) / paired) if paired else None
    p1_rate = (sum(1 for r in rank_flags if r == 1) / paired) if paired else None
    means = _block_means(wrong_flags)
    noise_band = max((abs(b - a) for a, b in zip(means, means[1:], strict=False)), default=None)
    #: 0/1 序列上"五等分相邻段最大跳幅"的**下界**是 `1/(n//5)`：n=8 ⇒ 带不可能小于 1.0。
    #: 这个数必须自己出版（DEBT-G59），否则"判据无法成立"要等面跑完才发现。
    size = max(1, paired // 5)
    noise_band_floor = (1.0 / size) if paired else None

    return {
        "format": REPORT_FORMAT,
        "store_count": int(store.count),
        "capacity": int(capacity),
        "limit": int(limit),
        "cue_encoding": cue_encoding,
        "cue_dim": None if store.cue_dim is None else int(store.cue_dim),
        "total_queries": len(queries),
        "paired_queries": paired,
        "unpaired_queries": unpaired,
        "state_counts": {**state_counts, "store_absent": 0},
        "state_accounting_note": (
            "store_absent 恒 0＝本件总是自建并挂载库；产品默认链的挂载态见 mount_layer 与 "
            "product_default_mounted，两者不许互相代答"
        ),
        "adapter_gated_state": NOT_OBSERVABLE,
        "wrong_top1_rate": wrong_rate,
        "true_rank_p1_rate": p1_rate,
        "true_rank_median": (statistics.median(rank_flags) if rank_flags else None),
        "true_rank_outside_top_limit": int(outside_top_limit),
        "true_rank_note": "median 只在真值进入前 limit 位时取；越界条数单列，不许折进中位数",
        "top1_score_mean": (sum(top1_scores) / len(top1_scores)) if top1_scores else None,
        "block_means": means,
        "block_size": int(size),
        "noise_band_floor": noise_band_floor,
        "noise_band_adjacent_block_max": noise_band,
        "ruler_usable": bool(
            paired > 0 and wrong_rate is not None and 0.0 < float(wrong_rate) < 1.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="N4 寻址面仪器（PLAN-N4-01：母量 wrong_top1_rate，只读零训练）"
    )
    parser.add_argument("--materials", required=True, help="名册 JSONL：memory_id + cue")
    parser.add_argument("--queries", required=True, help="查询 JSONL：cue + expected_memory_id")
    parser.add_argument("--capacity", type=int, default=1024, help="库容量（必须为正）")
    parser.add_argument("--limit", type=int, default=3, help="每次取回条数（必须为正）")
    parser.add_argument("--out-report", required=True, help="读数件落盘路径")
    parser.add_argument(
        "--cue-encoding",
        choices=("dense", "sparse"),
        default="dense",
        help="PLAN-N4-03：cue 编码（dense=基线逐位不变；sparse=WTA(RP(cue)) 稀疏编码）",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()

    def resolve(raw: str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    #: 响亮拒绝全部走 rc=2：缺件／非正容量／非正 limit／零配对查询——四支都有测试钉着。
    for label, raw in (("materials", args.materials), ("queries", args.queries)):
        if not resolve(raw).is_file():
            print(f"REJECT missing_file {label}")
            return 2
    if int(args.capacity) <= 0:
        print("REJECT capacity_must_be_positive")
        return 2
    if int(args.limit) <= 0:
        print("REJECT limit_must_be_positive")
        return 2

    materials_path = resolve(args.materials)
    queries_path = resolve(args.queries)
    try:
        payload = measure(
            _read_rows(materials_path, "materials"),
            _read_rows(queries_path, "queries"),
            capacity=int(args.capacity),
            limit=int(args.limit),
            cue_encoding=args.cue_encoding,
        )
    except ValueError as error:
        #: 包括 cue 维度不符、缺 memory_id、非空库拿到零命中（前提被改动）。
        print(f"REJECT {type(error).__name__}")
        payload = {"format": REPORT_FORMAT, "rejection": str(error)}
        target = resolve(args.out_report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        return 2

    payload.update(
        {
            "mount_layer": "harness",
            "mounted_by": "scripts/training/measure_taiji_n4_addressing_surface.py",
            "product_default_mounted": False,
            "debt": "DEBT-G58",
            "materials_path": str(materials_path),
            "materials_sha256": _sha256(materials_path),
            "queries_path": str(queries_path),
            "queries_sha256": _sha256(queries_path),
            "rc": 0 if payload["paired_queries"] else 2,
        }
    )
    if not payload["paired_queries"]:
        #: 零配对＝母量分母为 0，出版率会是 0/0；这里既不判级也不给率。
        payload["verdict"] = "no_paired_queries"
    #: Windows 控制台是 GBK：正文含中文，直接把中文打到 stdout 会在判完之后抛 UnicodeEncodeError，
    #: 把 rc 伪装成崩溃 ⇒ 只打 ASCII 摘要行，全量读数进 --out-report。
    print(
        "WRONG_TOP1_RATE",
        payload["wrong_top1_rate"],
        "PAIRED",
        payload["paired_queries"],
        "RULER_USABLE",
        payload["ruler_usable"],
        "RC",
        payload["rc"],
    )
    target = resolve(args.out_report)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return int(payload["rc"])


if __name__ == "__main__":
    raise SystemExit(main())
