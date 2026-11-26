"""真 oracle 天花板：按**题面标签**替模型选对告知，再走同一条产品链答题。

为什么要有这一件（`SPEC-A-22` §20）：本案一直引用的"天花板 43/39"其实出自
`price_taiji_r2_a25_overlap_selector.py:90` 的 `min(events, key=event_id)` ——
那是**"永远取最早入库那条"**，只在"答案告知必为第一"的题集上才等价于"替它选对"。
v3 把落位随机化之后，那条规则不再是 oracle（它会在一半的题上故意选错）。
⇒ v3 上 乙/丙 的 17–21 分到底是接近上界还是远低于上界，**没有可信参照**，本件就是补这个参照。

链路纪律：**不重抄生成链**。生成走产品那支 `_answer_raw`
（`score_taiji_r2_copy_circuit_chat_cap` 里的同一个函数：`_serialize` 铺文本 →
`record_told_history` 入库 → `generate_input(reset=True)` → `_TURN_MARKERS` 折字），
只在 `ToldContentStore.best_match` 上打一层**按题面标签选事件**的补丁。

自检（缺一条就不出件）：
1. 每题**恰好一条**告知含答案词——否则该题记入 `ambiguous` 并跳过（歧义标签不配当上界）；
2. 补丁被走到的次数 `used` 必须等于答题次数（上一版天花板档因漏装计数器被仪器自己拒签，
   见 `SPEC-A-17` §15 的过程记录）；
3. 基座 `checkpoints/*.pt` 跑前后 sha256 逐位相同。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
#: 与两台计分器同一条链、同一个折字口径 ⇒ 只有"选事件"这一维被换掉。
MAX_ANSWER_BYTES = 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", required=True, help="要抬上界的那份电路 payload")
    parser.add_argument("--manifest", default=None, help="默认 v3（位置随机）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw
    from score_taiji_r2_copy_surface_extension import load_items

    from api.seed_runtime import SeedRuntime
    from taiji.copy_circuit import ToldContentStore

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = MANIFEST if not args.manifest else Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = load_items(manifest)
    if args.limit:
        items = items[: args.limit]

    state: dict[str, Any] = {"tokens": [], "used": 0, "answered": 0, "ambiguous": 0, "no_match": 0}
    original = ToldContentStore.best_match

    def _oracle(self: ToldContentStore, cue: Any) -> Any:
        events = self.events()
        if not events:
            return None
        matches = [
            event
            for event in events
            if any(
                token in bytes(event.content).decode("utf-8", "ignore") for token in state["tokens"]
            )
        ]
        if len(matches) > 1:
            state["ambiguous"] += 1
            return None
        if not matches:
            state["no_match"] += 1
            return None
        state["used"] += 1
        return matches[0]

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    ToldContentStore.best_match = _oracle  # type: ignore[method-assign]
    rows = []
    try:
        for item in items:
            tokens = list(item["expected_contains"])
            told = [turn for turn in item["turns"][:-1]]
            matches = [turn for turn in told if any(token in turn for token in tokens)]
            if len(matches) != 1:
                #: 标签有歧义 ⇒ 这一题不存在"上界"可言，整题跳过并计数（不许硬选一条凑数）。
                state["ambiguous"] += 1
                continue
            history: list[tuple[str, str]] = []
            turns = list(item["turns"])
            answer = ""
            state["answered"] += 1
            for index, turn in enumerate(turns):
                state["tokens"] = tokens
                answer = _answer_raw(runtime, turn, history)
                if index + 1 < len(turns):
                    history.append((turn, answer))
            rows.append(
                {
                    "id": item["id"],
                    "kind": item.get("kind"),
                    "answer_tell_position": item.get("answer_tell_position"),
                    "hit": int(any(token in answer for token in tokens)),
                }
            )
    finally:
        ToldContentStore.best_match = original  # type: ignore[method-assign]

    hits = sum(row["hit"] for row in rows)
    report = {
        "format": "taiji-r2-a25-true-oracle-ceiling-v1",
        "prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md §20",
        "manifest": manifest.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "items": len(items),
        "answered": state["answered"],
        "strict_hits": hits,
        "hit_rate": round(hits / max(state["answered"], 1), 4),
        "probe": {
            "selections_used": state["used"],
            "ambiguous_items": state["ambiguous"],
            "no_matching_event": state["no_match"],
        },
        "by_position": {
            side: {
                "items": len([r for r in rows if r["answer_tell_position"] == index]),
                "hits": sum(r["hit"] for r in rows if r["answer_tell_position"] == index),
            }
            for index, side in ((0, "answer_first"), (1, "answer_second"))
        },
        "instrument_guard": {
            #: 补丁没被走到＝这份"上界"是在另一条链上量的，整件作废。
            "patch_effective": bool(state["used"] >= state["answered"]),
            "reason": (
                "selections_used 必须 ≥ answered：每题至少一次选事件"
                "（发射侧每个字节都会问一次库）"
            ),
        },
        "rows": rows,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "answered": state["answered"],
                "strict_hits": hits,
                "probe": report["probe"],
                "guard_ok": report["instrument_guard"]["patch_effective"],
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),  # 今天第三次犯同一个错：`--out-*` 指到仓库外时不许假设 relative_to 可用
            },
            ensure_ascii=False,
        )
    )
    return (
        0
        if report["instrument_guard"]["patch_effective"] and report["base_sha256_unchanged"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
