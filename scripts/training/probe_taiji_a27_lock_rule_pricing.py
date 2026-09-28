"""A-27 下一刀·零训练定价：**锁（事件选择）**换成"免训练可算特征"能拿多少。

为什么是这一刀（三条战线扫完的结论）
------------------------------------
* 战线二已证：产品在"答案落在**最新**那条告知"的题上只有 1/52（seed-B 0/52），
  而把选择**强制**到正确那条就有 **20/52（17/52）** ⇒ **+19/52 挂在这一层**；
  机理直读：自然臂**46/52 的答案是用旧那条的内容拼出来的**。
* 锁的规则现状（`CopyCircuit.selection`）：`scores = cue_cosine + head`，而 head
  **零初始化** ⇒ 未训时**只用皮质 cue 余弦**一项。它选中旧那条，不是因为旧那条"更相关"，
  而是因为**提问那一刻的 cue 恰好像旧那条**。
* 所以问题变成一句可零训练回答的话：**换一组"不需要训练就能算"的特征来打分，
  锁能不能选对？**

本支怎么算（**一次经过、所有规则同时打分**）
--------------------------------------------
`selection()` 本来就会把 5 个特征算成矩阵（`SPEC-A-22` §6 冻结表：
`cue_cosine` / `content_cosine` / `byte_overlap` / `recency` / `length_log`）。
本支在**不改产品码**的前提下包一层：从同一个特征矩阵上，把每条候选规则的 argmax 都记一遍
⇒ **一遍链路的成本，拿到全部规则的选中率**（不需要为每条规则各跑一遍）。

候选规则（都是"免训练"，权重是人给的、不是学出来的）
----------------------------------------------------
| 规则 | 权重（按上面 5 个特征的顺序） | 动机 |
|---|---|---|
| `cue_only` | (1,0,0,0,0) | **＝现存生产规则**（自检：它的选中必须与产品 `picked` 逐个相同） |
| `byte_overlap` | (0,0,1,0,0) | §15 那把免训练键（"与提问共享字符"） |
| `cue_plus_overlap` | (1,0,1,0,0) | cue 与字面重叠各半 |
| `overlap_plus_content` | (0,1,1,0,0) | 内容侧两项（不含 cue） |
| `recency_only` | (0,0,0,1,0) | **对照**：A2.5 学出来的头实际装的就是它（位置先验） |

判据（跑之前写死）
------------------
* **自检**：`cue_only` 的逐题选中必须与产品 `picked` **完全一致**，否则本支作废
  （说明我对特征顺序或平手裁决的理解有偏差）。
* **主读数**：各规则的**选中正确率**（分"答案在最早/最新"两半报）。
  以 `cue_only` 为基线；**任何规则若把"最新"那一半的选中率从 ~0 抬起来，就是可摘的增量**。
* **换算**：命中 ≈ 选中率 × 条件发射率；条件发射率由战线二实测给出
  （`oracle_chosen`：正确选中 47/52 里命中 20 ⇒ **≈0.43**）。**这是换算，不是实测**，
  故本支同时提供 `--apply-rule` 去**真跑**一遍那一臂（把锁的分数整个换成该规则）。
* 守卫：基座 sha256 不变；**产品源码指纹不变**（只在探针里包实例）；每规则分母 > 0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
#: 特征顺序＝`CopyCircuit.SELECTOR_FEATURE_NAMES`（本支不重抄，只按那个顺序给权重）。
RULES: dict[str, tuple[float, float, float, float, float]] = {
    "cue_only": (1.0, 0.0, 0.0, 0.0, 0.0),
    "byte_overlap": (0.0, 0.0, 1.0, 0.0, 0.0),
    "cue_plus_overlap": (1.0, 0.0, 1.0, 0.0, 0.0),
    "overlap_plus_content": (0.0, 1.0, 1.0, 0.0, 0.0),
    "recency_only": (0.0, 0.0, 0.0, 1.0, 0.0),
}
#: 战线二实测的条件发射率（`oracle_chosen`：正确选中 47/52 中命中 20）。
CONDITIONAL_EMISSION_RATE = 20.0 / 47.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for folder in ("taiji", "api"):
        for path in sorted((PROJECT_ROOT / folder).rglob("*.py")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--apply-rule",
        default=None,
        choices=sorted(RULES),
        help="把锁的分数整个换成该规则**真跑一遍**（量命中），而不是只算选中率",
    )
    parser.add_argument(
        "--query-scope",
        choices=("whole", "question"),
        default="whole",
        help="喂给 `lock_selection` 的 `query_bytes`：`whole`＝现状（产品传的是**整段序列化文本**）；"
        "`question`＝只传**提问那一轮**（特征 2「与提问共享字符」的原意）",
    )
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw
    from score_taiji_r2_copy_surface_extension import load_items

    from api.seed_runtime import SeedRuntime
    from taiji.copy_circuit import CopyCircuit

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    source_before = _source_fingerprint()
    items = load_items(MANIFEST)
    if args.limit:
        items = items[: args.limit]

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    circuit = runtime.model.substrate.copy_circuit
    if circuit is None:
        raise SystemExit("copy circuit is not mounted")

    labels: dict[str, list[str]] = {"tokens": []}
    #: 只记**答题轮**那一次锁：告知轮的锁里标注事件还没入库，算进来会把选中率拉偏
    #: （2026-09-28 同型缺陷已在本轮登记过一次）。
    scope: dict[str, bool] = {"record": False}
    #: 当前提问那一轮的字节（`--query-scope question` 用它替换产品传进去的整段文本）。
    ask: dict[str, bytes] = {"question": b""}
    tallies: dict[str, dict[str, int]] = {
        rule: {"locks": 0, "on_labelled": 0} for rule in RULES
    }
    disagreements = {"cue_only_vs_production": 0}
    rows: list[dict[str, Any]] = []
    originals = {"selection": CopyCircuit.selection, "lock": CopyCircuit.lock_selection}
    weights = {
        rule: torch.tensor(vector, dtype=torch.float32) for rule, vector in RULES.items()
    }

    def selection(self: Any, **kwargs: Any) -> Any:
        state = originals["selection"](self, **kwargs)
        if state is None:
            return state
        events = self.store.events()
        matrix = state["features"]
        picked_by_rule: dict[str, int] = {}
        for rule, vector in weights.items():
            scores = matrix @ vector
            picked = int(torch.argmax(scores).item())
            picked_by_rule[rule] = picked
            if scope["record"]:
                tallies[rule]["locks"] += 1
        labelled = [
            index
            for index, event in enumerate(events)
            if any(token in bytes(event.content).decode("utf-8", "ignore") for token in labels["tokens"])
        ]
        if scope["record"]:
            for rule, picked in picked_by_rule.items():
                tallies[rule]["on_labelled"] += int(
                    len(labelled) == 1 and picked == labelled[0]
                )
            if picked_by_rule["cue_only"] != int(state["picked"]):
                disagreements["cue_only_vs_production"] += 1
            rows.append(
                {
                    "id": labels.get("item_id"),
                    "answer_tell_position": labels.get("position"),
                    "events": len(events),
                    "labelled_index": labelled[0] if len(labelled) == 1 else None,
                    "production_picked": int(state["picked"]),
                    "picked_by_rule": picked_by_rule,
                }
            )
        if args.apply_rule is not None:
            override = picked_by_rule[args.apply_rule]
            return {**state, "picked": override, "event": events[override]}
        return state

    def lock_selection(self: Any, *, cue: Any, f1_context: Any, query_bytes: bytes = b"") -> Any:
        #: 口径开关：产品把**整段序列化文本**当 query（`model.py:3088-3092`），
        #: 于是特征 2「与提问共享字符」实际算的是"与整段对话共享字符"——对每条告知都接近满分、
        #: **没有区分度**（这正是本轮定价实测到 `byte_overlap` 两半各 12–13/52 近随机的头号嫌疑）。
        #: `question` 档只传提问那一轮，用来**定价**"若口径修对，内容侧特征能不能用"。
        if args.query_scope == "question" and ask["question"]:
            query_bytes = ask["question"]
        return originals["lock"](self, cue=cue, f1_context=f1_context, query_bytes=query_bytes)

    CopyCircuit.selection = selection
    CopyCircuit.lock_selection = lock_selection
    hits = 0
    answered = 0
    try:
        for item in items:
            tokens = [str(token) for token in item["expected_contains"]]
            labels["tokens"] = tokens
            labels["item_id"] = item["id"]
            labels["position"] = item.get("answer_tell_position")
            turns = [str(turn) for turn in item["turns"]]
            ask["question"] = turns[-1].encode("utf-8")
            history: list[tuple[str, str]] = []
            answer = ""
            for index, turn in enumerate(turns):
                scope["record"] = index == len(turns) - 1
                answer = _answer_raw(runtime, turn, history)
                if index + 1 < len(turns):
                    history.append((turn, answer))
            answered += 1
            hits += int(any(token in answer for token in tokens))
    finally:
        CopyCircuit.selection = originals["selection"]
        CopyCircuit.lock_selection = originals["lock"]

    def _split(rule: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for side, position in (("answer_first", 0), ("answer_second", 1)):
            subset = [row for row in rows if row["answer_tell_position"] == position]
            correct = sum(
                1
                for row in subset
                if row["labelled_index"] is not None
                and row["picked_by_rule"][rule] == row["labelled_index"]
            )
            out[side] = {"locks": len(subset), "pick_correct": correct}
        return out

    report: dict[str, Any] = {
        "format": "taiji-a27-lock-rule-pricing-v1",
        "prereg": "plans/reference/PLAN-A-27_three-gaps-decision-brief_20260928.md（三线扫完后的下一刀）",
        "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(MANIFEST),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _sha256(PROJECT_ROOT / args.circuit),
        "items": len(items),
        "answered": answered,
        "apply_rule": args.apply_rule,
        "query_scope": args.query_scope,
        "generated_hits": hits if args.apply_rule is not None else None,
        "rules": {
            rule: {
                "weights": list(RULES[rule]),
                **tallies[rule],
                "pick_rate": round(
                    tallies[rule]["on_labelled"] / max(tallies[rule]["locks"], 1), 4
                ),
                "by_position": _split(rule),
            }
            for rule in RULES
        },
        "conversion_note": {
            "conditional_emission_rate": round(CONDITIONAL_EMISSION_RATE, 4),
            "expected_hits_if_applied_latest_half": round(
                CONDITIONAL_EMISSION_RATE * tallies["byte_overlap"]["locks"], 2
            ),
            "caveat": "换算≠实测：真命中要看 `--apply-rule` 那一跑",
        },
        "rows": rows,
        "instrument_guard": {
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
            "product_source_untouched": _source_fingerprint() == source_before,
            "cue_only_matches_production_pick": disagreements["cue_only_vs_production"] == 0,
            "every_rule_has_denominator": all(
                tallies[rule]["locks"] > 0 for rule in RULES
            ),
        },
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for rule in RULES:
        entry = report["rules"][rule]
        print(
            f"{rule:24s} 选中率 {entry['pick_rate']:6.4f}  "
            f"（最早 {entry['by_position']['answer_first']['pick_correct']}"
            f"/{entry['by_position']['answer_first']['locks']}，"
            f"最新 {entry['by_position']['answer_second']['pick_correct']}"
            f"/{entry['by_position']['answer_second']['locks']}）"
        )
    print(json.dumps({"guard": report["instrument_guard"], "hits": report["generated_hits"], "out": out.name}, ensure_ascii=False))
    return 0 if all(report["instrument_guard"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())