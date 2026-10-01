"""A2.5 前置诊断（零训练）：多事件库里"挑哪条告知"到底失败在哪一层。

**为什么要先测**：A2.3b 对齐臂的 `misaddressed_episodes＝38.3%` 与语料干扰轮比例（40%）重合
（`SPEC-A-17` §7.2/§7.4）——几乎等于"只要有干扰轮就挑错"。事件选择这一步今天是
**写死的余弦 top-1**（`ToldContentStore.best_match`），没有任何可学参数。开案前必须先分清
三种完全不同的病因，否则会把学习器装在一个换键就能解决的问题上：

1. **键形态坏了**：区 1/2 的分量把区 0 的结构稀释掉（R2 归因链的两条既有结论支持这个方向：
   "槽结构只在区 0"与"退化实为跨区稀释"）⇒ 该修键，不该加参数；
2. **键本身不可分**：同模型先后两条告知的皮质态在任意切法下都分不开 ⇒ 才需要可学的事件选择器
   （现成件见 `SPEC-A-17` §7.3：`ContentSelector` 同形）；
3. **测法/时点问题**：入库 cue 取自"问：轮读完"，而查询 cue 取自"答："之后——
   两者本来就不是同一分布 ⇒ 该修的是**查询时点**，前两条都谈不上。

**判读线（本文件冻结，先于跑）**：
* 取三种键形态各算"正确告知事件排 top-1"的比例：**full**＝现状（activity+trace 全拼接）、
  **region0**＝只取区 0 的 activity+trace、**oracle**＝直接用告知文本与提问的字符重叠（键的上限参照）。
* 若 `full < 0.6` 且 `region0 ≥ 0.8` ⇒ 病因 1（键形态），下一案修键；
* 若 `full < 0.6` 且 `region0 < 0.6` 而 oracle 高 ⇒ 病因 2（需要学习器）；
* 若 `full ≥ 0.8` ⇒ 病因 3：静态比较挑得对，说明线上失败来自**逐步 cue 漂移**（生成期每步 cue 都在变），
  下一案改测"逐答案步 cue"而不是本件的单点 cue。

**纪律**：全程 `learn=False`、`checkpoints/` 只读且跑前后 sha256 复核；写靶只有 `reports/` 一份新件
（存在即加时间戳，不覆写）；不训练任何参数；报告里的文本细节读文件，不打印到 stdout。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
#: 扩展集（104 题，实体与评价集/训练表双不相交）——§13 说选择的价格是 **+17/48**，
#: 但那个数是在"替它挑对"的上界上量的；本件要回答的是"免训练的键能挑对多少"。
SURFACE_REPORT = PROJECT_ROOT / "reports/taiji_r2_copy_surface_extension_20260925.json"
SURFACE_MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
MAX_REPLY_BYTES = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extension_selection_items(arm: int = 0) -> list[tuple[str, dict[str, Any]]]:
    """扩展集里**库里确有 ≥2 条告知、且挂了电路仍未命中**的题——选择侧的真分母。

    只有一条告知的题（`given_then_ask`）不进来：那种库没有可挑的第二个事件，
    把它们算进分母就是把"无需选择"混进"选择失败"（§9 的总分就是这么被摊平的）。
    """
    payload = json.loads(SURFACE_MANIFEST.read_text(encoding="utf-8"))
    items = {str(item["id"]): item for item in payload["dimensions"]["X"]["items"]}
    data = json.loads(SURFACE_REPORT.read_text(encoding="utf-8"))
    misses = {str(row["id"]) for row in data["treated_arms"][arm]["rows"] if not bool(row["hit"])}
    return [
        (item_id, items[item_id])
        for item_id in sorted(misses)
        if item_id in items and len(items[item_id]["turns"]) >= 3
    ]


def _slice_regions(cue: torch.Tensor, region_sizes: tuple[int, ...]) -> torch.Tensor:
    """从 `cortical_context` 的拼接布局里取区 0 的 activity+trace 两段。

    布局＝`cat(activity_0..activity_n, trace_0..trace_n)`（`taiji/fabric.py:558-566`）。
    """
    total = sum(region_sizes)
    if int(cue.numel()) != 2 * total:
        raise ValueError(f"cue length {int(cue.numel())} != 2*sum(region_sizes)={2 * total}")
    size0 = int(region_sizes[0])
    return torch.cat([cue[:size0], cue[total : total + size0]], dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left = torch.nn.functional.normalize(a.detach().cpu().float(), dim=0)
    right = torch.nn.functional.normalize(b.detach().cpu().float(), dim=0)
    return float(torch.dot(left, right))


def _rank_of_correct(
    query: torch.Tensor, events: list[tuple[int, bytes, torch.Tensor]]
) -> dict[str, Any]:
    """返回余弦 top-1 挑中的事件、它与次优的 margin（本件只用"挑中正确那条"判分）。"""
    scored = sorted(
        ((_cosine(query, cue), event_id, content) for event_id, content, cue in events),
        key=lambda item: item[0],
        reverse=True,
    )
    best_score, best_id, best_content = scored[0]
    second = scored[1][0] if len(scored) > 1 else None
    return {
        "picked_id": best_id,
        "picked_content": best_content.decode("utf-8", errors="replace"),
        "margin": None if second is None else round(best_score - second, 6),
        "top_score": round(best_score, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--items", type=int, default=16, help="取 D 维多轮题的前 N 题")
    parser.add_argument(
        "--source",
        choices=("cap-dim", "extension"),
        default="cap-dim",
        help="cap-dim＝§9 那 16 题；extension＝扩展集里库里≥2条告知且挂电路仍未命中的题",
    )
    parser.add_argument(
        "--arm", type=int, default=0, help="extension 档取哪一臂的未命中清单（0＝seed-A）"
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from score_taiji_r2_copy_strict_cap import copyable_tokens, load_items

    from api.seed_runtime import SeedRuntime, record_told_history

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    circuit = substrate.copy_circuit
    assert circuit is not None
    region_sizes = tuple(int(size) for size in substrate.config.region_sizes)

    items = load_items()
    if args.source == "extension":
        multi_turn = extension_selection_items(args.arm)
    else:
        multi_turn = [
            (item_id, item)
            for item_id, item in sorted(items.items())
            if item["dimension"] == "D"
            and len(item.get("turns") or []) >= 2
            and copyable_tokens(item)
        ][: args.items]

    rows: list[dict[str, Any]] = []
    for item_id, item in multi_turn:
        turns = [str(turn) for turn in item["turns"]]
        ask = turns[-1]
        told = turns[:-1]
        tokens = list(copyable_tokens(item))
        bearers = [turn for turn in told if any(token in turn for token in tokens)]
        if args.source == "extension" and (not bearers or bearers[0] != told[0]):
            #: 本件的"正确事件"＝第一条含可复制答案词的告知。扩展集的形状是
            #: tell→distractor→ask，所以它就是 `told[0]`——**断言而不是假设**：
            #: 题面形状一变，"挑对/挑错"的分母会悄悄错掉。
            raise AssertionError(f"{item_id}: 含答案词的告知不是第一条，选择分母失效")
        history: list[tuple[str, str]] = []
        for index, turn in enumerate(told):
            record_told_history(
                substrate, circuit, history, episode_id=f"a25probe:{item_id}:{index}"
            )
            prompt = SeedRuntime._serialize(turn, history)
            raw = substrate.generate(
                prompt.encode("utf-8"), MAX_REPLY_BYTES, stop_at_boundary=True, sample=False
            )
            reply = raw.decode("utf-8", errors="replace")
            for marker in ("\n问：", "问："):
                cut = reply.find(marker)
                if cut >= 0:
                    reply = reply[:cut]
            history.append((turn, reply.strip()))
        record_told_history(substrate, circuit, history, episode_id=f"a25probe:{item_id}:final")
        prompt = SeedRuntime._serialize(ask, history)
        #: 与产品一致：`generate_input` 默认 `reset=True` ⇒ 重新喂整段，取"答："之后的皮质态。
        substrate.reset_dynamics(episode_id="generation")
        substrate.observe(
            int(substrate.config.boundary_symbol),
            learn=False,
            readout="predictive",
            use_memory=False,
        )
        probs = None
        for symbol in prompt.encode("utf-8"):
            probs = substrate.observe(
                int(symbol), learn=False, readout="predictive", use_memory=False
            )
        assert probs is not None
        query_cue = substrate.cortical_cue().detach().cpu().clone()
        events = [
            (event.event_id, event.content, event.cue.detach().cpu().clone())
            for event in circuit.store.events()
        ]
        correct_id = next(
            (eid for eid, content, _ in events if content == told[0].encode("utf-8")), None
        )
        full = _rank_of_correct(query_cue, events)
        region0 = _rank_of_correct(
            _slice_regions(query_cue, region_sizes),
            [(eid, content, _slice_regions(cue, region_sizes)) for eid, content, cue in events],
        )
        #: oracle 参照：不用神经 cue，直接用"提问与告知文本的字符重叠"给事件排序。
        overlap = [
            (
                len(set(content.decode("utf-8", errors="replace")) & set(ask))
                / max(len(set(ask)), 1),
                eid,
                content,
            )
            for eid, content, _ in events
        ]
        overlap.sort(reverse=True)
        rows.append(
            {
                "id": item_id,
                "source": args.source,
                "kind": item.get("kind"),
                "n_events": len(events),
                "told": told,
                "ask": ask,
                "correct_event_id": correct_id,
                "full_key": {**full, "picked_correct": full["picked_id"] == correct_id},
                "region0_key": {
                    **region0,
                    "picked_correct": region0["picked_id"] == correct_id,
                },
                "oracle_key": {
                    "picked_id": overlap[0][1],
                    "picked_correct": overlap[0][1] == correct_id,
                    "top_overlap": round(overlap[0][0], 4),
                },
            }
        )

    def _rate(key: str) -> float:
        return round(sum(1 for row in rows if row[key]["picked_correct"]) / max(len(rows), 1), 4)

    def _median_margin(key: str) -> float | None:
        values = [row[key]["margin"] for row in rows if row[key]["margin"] is not None]
        return None if not values else round(statistics.median(values), 6)

    def _by_stratum() -> dict[str, dict[str, float]]:
        """§13 预登记的判读线：**只按 kind 分层看，不看总分**。
        §9 那次三岔口一支没命中，就是因为总分把"字面重叠能救的族"和"间接指代"混在了一起。
        """
        strata: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            strata.setdefault(str(row.get("kind") or "unknown"), []).append(row)
        return {
            kind: {
                "n": len(group),
                "full": round(
                    sum(1 for r in group if r["full_key"]["picked_correct"]) / len(group), 4
                ),
                "region0": round(
                    sum(1 for r in group if r["region0_key"]["picked_correct"]) / len(group), 4
                ),
                "overlap": round(
                    sum(1 for r in group if r["oracle_key"]["picked_correct"]) / len(group), 4
                ),
            }
            for kind, group in sorted(strata.items())
        }

    verdict = {
        "full_key_top1_rate": _rate("full_key"),
        "region0_key_top1_rate": _rate("region0_key"),
        "oracle_key_top1_rate": _rate("oracle_key"),
        "by_kind": _by_stratum(),
        "margin_median_full": _median_margin("full_key"),
        "margin_median_region0": _median_margin("region0_key"),
        "per_item_margins_full": [row["full_key"]["margin"] for row in rows],
        "per_item_margins_region0": [row["region0_key"]["margin"] for row in rows],
    }
    report = {
        "format": "taiji-r2-a25-event-cue-separability-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §7.3",
        "reading_rule": (
            "full<0.6 且 region0>=0.8 ⇒ 病因1（键形态，先修键不加参数）；"
            "full<0.6 且 region0<0.6 而 oracle 高 ⇒ 病因2（需要可学的事件选择器）；"
            "full>=0.8 ⇒ 病因3（本件单点 cue 挑得对，线上失败来自生成期逐步 cue 漂移，"
            "下一案改测逐答案步 cue）"
        ),
        "checkpoint": args.checkpoint,
        "source": args.source,
        "arm": args.arm if args.source == "extension" else None,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "items": len(rows),
        "summary": verdict,
        "rows": rows,
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_r2_a25_event_cue_separability_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(rows),
                "source": args.source,
                "full": verdict["full_key_top1_rate"],
                "region0": verdict["region0_key_top1_rate"],
                "oracle": verdict["oracle_key_top1_rate"],
                "base_unchanged": report["base_sha256_unchanged"],
                "by_kind": verdict["by_kind"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
