"""A-27 战线二：发射侧三维诊断 ＋ 两条对照（零训练、只读、不改产品链）。

为什么要这一支（`PLAN-A-27` §2，owner 2026-09-28 签"乙档"）
--------------------------------------------------------
`SPEC-A-22` §21 记了一个症状：v3（落位随机 52/52）里 **答案是最新那条告知**的 52 题命中
0–1/52，而答案是最早那条的 52 题有 19–20/52 ⇒"复制通路只在目标是最早入库那条时工作"。
它据此把瓶颈从"选择"挪到"发射"，并点名下一刀：在真 oracle 之下跑逐步轨迹，量三件事——
①copy 质量是否确实落在被选中的那条上；②发出来的字节是**哪一条**告知的内容；
③目标字节在候选排序里的名次。

**先于这三件事，本支必须先回答一个更靠前的问题**（读码时发现的，必须先验后信）：
那份"真 oracle"（`price_taiji_r2_a25_true_oracle_ceiling.py`）把补丁打在
`ToldContentStore.best_match` 上；而产品生成链在"提问喂完"那一刻会
`circuit.lock_selection(...)` 把选择**整轮锁死**（`taiji/model.py:3084-3092`），此后
发射与训练寻址的**唯一入口**是 `CopyCircuit._chosen_event`，它在有锁时
**直接返回锁定的那条、根本不问 `best_match`**（`taiji/copy_circuit.py:412-423`）。
⇒ 若锁定生效，那份"替模型选对"的补丁在**发射路径上不被问**，栏杆上的
`selections_used ≥ answered` 也拦不住（那个计数器数的是**喂入期**的调用）。
本支因此加了 `lock_bypass` 自检：**在生成阶段**分别数"走了锁定路径"与
"问了 `best_match`"的次数；它们决定 §21 的读数该保留还是回炉。

四臂（同一题集、同一条产品链、只换"取事件"这一维）
--------------------------------------------------
| 臂 | 取事件时怎么选 | 用途 |
|---|---|---|
| `natural` | 原样（生产：有锁用锁） | 复现 §20 的自然成绩，作对照底 |
| `oracle_bestmatch` | 补丁打在 `best_match`（＝§21 那份仪器） | 自检：它到底有没有生效 |
| `oracle_chosen` | 补丁打在 `_chosen_event`（**发射的唯一入口**） | 真正的"替它选对" |
| `drop_old` | `_chosen_event` 里**排除最早那条**（库里只剩新那条） | 判"竞争侧 vs 键侧" |

三维读数（逐题）
----------------
① `chosen_is_labelled`：生成阶段取到的每条事件是否都是含答案词的那条；
② `emitted_owner`：发出的答案内容**属于哪一条**告知（`labelled`／`other`／`none`）；
③ `target_rank`：生成第一步时，答案首字节在 copy 证据向量里的名次（降序，1＝就是它）。

判据（跑之前写死，跑完不改）
--------------------------
* **先判自检**：若 `oracle_bestmatch` 臂在生成阶段的 `best_match` 调用为 **0**，
  则 §21 的"真 oracle"在发射上是空转 ⇒ 它的 20/1 不能读成"选择已不是瓶颈"，
  该结论**回炉**（本支不改 §21 的历史读数，只登记为"需要重算"）。
* **主判据**：`drop_old` 臂在"答案最新"那 52 题上的命中数
  —— **≥15/52 ⇒ 竞争侧**（锁/线索系统性指向旧那条，旧内容压住新内容）；
  **≤5/52 ⇒ 键侧**（新内容进不了发射，与竞争无关）。
* **校正参照**：`oracle_chosen` 的命中数就是"选择完美"的真实上界（替代 §21 那个）。
* 守卫：基座 sha256 跑前后逐位相同；`instrument_guard` 里三条分母都必须非零
  （每臂答题数、生成阶段取事件次数、copy 证据非零步数）。
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
MAX_ANSWER_BYTES = 64
ARMS = ("natural", "oracle_bestmatch", "oracle_chosen", "drop_old")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(content: bytes) -> str:
    return bytes(content).decode("utf-8", "ignore")


def run_arm(runtime: Any, items: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    """一臂：逐题走产品链（`_answer_raw`），只在"取事件"那一维上打补丁。"""

    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw
    from taiji.copy_circuit import CopyCircuit, ToldContentStore

    circuit = runtime.model.substrate.copy_circuit
    if circuit is None:
        raise SystemExit("copy circuit is not mounted")

    labels: dict[str, list[str]] = {"tokens": []}
    state: dict[str, Any] = {
        "phase": "feed",
        "lock_path": 0,
        "best_match_calls": 0,
        "best_match_calls_in_generation": 0,
        "chosen_calls_in_generation": 0,
        "chosen_ids_in_generation": [],
        "chosen_texts_in_generation": [],
        "evidence_nonzero_steps": 0,
        "first_step_evidence": None,
        "used_oracle_label": 0,
    }
    originals = {
        "chosen": CopyCircuit._chosen_event,
        "evidence": CopyCircuit.evidence,
        "best_match": ToldContentStore.best_match,
        "generate_input": runtime.model.generate_input,
    }

    def _labelled_event(store: Any) -> Any:
        matches = [
            event
            for event in store.events()
            if any(token in _text(event.content) for token in labels["tokens"])
        ]
        return matches[0] if len(matches) == 1 else None

    def chosen(self: Any, cue: Any) -> Any:
        locked = self.locked_event_id is not None
        event = originals["chosen"](self, cue)
        if arm == "oracle_chosen":
            labelled = _labelled_event(self.store)
            if labelled is not None:
                state["used_oracle_label"] += 1
                event = labelled
        elif arm == "drop_old":
            events = self.store.events()
            if len(events) > 1:
                oldest = min(int(item.event_id) for item in events)
                newer = [item for item in events if int(item.event_id) != oldest]
                #: 库里"只剩新那条"：锁或余弦若指向被摘掉的那条，就改取剩下的最新一条。
                if event is None or int(event.event_id) == oldest:
                    event = newer[-1]
        if state["phase"] == "generation":
            state["chosen_calls_in_generation"] += 1
            state["lock_path"] += int(locked)
            state["chosen_texts_in_generation"].append(
                "" if event is None else _text(event.content)
            )
            state["chosen_ids_in_generation"].append(
                None if event is None else int(event.event_id)
            )
        return event

    def evidence(self: Any, **kwargs: Any) -> Any:
        result = originals["evidence"](self, **kwargs)
        if state["phase"] == "generation":
            magnitude = float(result.abs().sum())
            if magnitude > 0.0:
                state["evidence_nonzero_steps"] += 1
                if state["first_step_evidence"] is None:
                    state["first_step_evidence"] = result.detach().cpu().clone()
        return result

    def best_match(self: Any, cue: Any) -> Any:
        state["best_match_calls"] += 1
        if state["phase"] == "generation":
            state["best_match_calls_in_generation"] += 1
        if arm == "oracle_bestmatch":
            labelled = _labelled_event(self)
            if labelled is not None:
                state["used_oracle_label"] += 1
                return labelled
        return originals["best_match"](self, cue)

    def generate_input(*args: Any, **kwargs: Any) -> Any:
        state["phase"] = "generation"
        state["first_step_evidence"] = None
        try:
            return originals["generate_input"](*args, **kwargs)
        finally:
            state["phase"] = "feed"

    CopyCircuit._chosen_event = chosen
    CopyCircuit.evidence = evidence
    ToldContentStore.best_match = best_match
    runtime.model.generate_input = generate_input

    rows: list[dict[str, Any]] = []
    try:
        for item in items:
            tokens = [str(token) for token in item["expected_contains"]]
            labels["tokens"] = tokens
            turns = [str(turn) for turn in item["turns"]]
            told = turns[:-1]
            history: list[tuple[str, str]] = []
            labelled_index = next(
                (
                    index
                    for index, turn in enumerate(told)
                    if any(token in turn for token in tokens)
                ),
                None,
            )
            state["chosen_ids_in_generation"] = []
            state["chosen_texts_in_generation"] = []
            state["first_step_evidence"] = None
            answer = ""
            for index, turn in enumerate(turns):
                answer = _answer_raw(runtime, turn, history)
                if index + 1 < len(turns):
                    history.append((turn, answer))
            hit = int(any(token in answer for token in tokens))
            labelled_text = "" if labelled_index is None else told[labelled_index]
            other_text = "".join(
                turn for index, turn in enumerate(told) if index != labelled_index
            )
            first_byte = tokens[0].encode("utf-8")[0] if tokens else None
            rank = None
            if state["first_step_evidence"] is not None and first_byte is not None:
                vector = state["first_step_evidence"]
                order = torch.argsort(vector, descending=True)
                position = (order == int(first_byte)).nonzero()
                rank = int(position[0].item()) + 1 if position.numel() else None
            chosen_ids = [value for value in state["chosen_ids_in_generation"] if value is not None]
            chosen_texts = state["chosen_texts_in_generation"]
            rows.append(
                {
                    "id": item["id"],
                    "kind": item.get("kind"),
                    "answer_tell_position": item.get("answer_tell_position"),
                    "hit": hit,
                    "answer_head": answer[:24],
                    "emitted_owner": (
                        "none"
                        if not answer.strip()
                        else (
                            "labelled"
                            if any(token in answer for token in tokens)
                            else (
                                "other"
                                if other_text and answer[:3] and answer[:3] in other_text
                                else "unattributed"
                            )
                        )
                    ),
                    "chosen_is_labelled": (
                        None
                        if labelled_index is None
                        else bool(chosen_texts)
                        and all(
                            any(token in text for token in tokens) for text in chosen_texts
                        )
                    ),
                    "chosen_event_ids": chosen_ids[:8],
                    "target_rank": rank,
                    "copy_evidence_nonzero_steps": state["evidence_nonzero_steps"],
                }
            )
    finally:
        CopyCircuit._chosen_event = originals["chosen"]
        CopyCircuit.evidence = originals["evidence"]
        ToldContentStore.best_match = originals["best_match"]
        runtime.model.generate_input = originals["generate_input"]
    return rows, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", required=True)
    parser.add_argument(
        "--position",
        choices=("latest", "earliest", "all"),
        default="latest",
        help="只跑答案落在哪条告知上（latest＝`answer_tell_position == 1`）",
    )
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from score_taiji_r2_copy_surface_extension import load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    circuit = PROJECT_ROOT / args.circuit
    sha_before = _sha256(checkpoint)
    items = load_items(MANIFEST)
    if args.position != "all":
        wanted = 1 if args.position == "latest" else 0
        items = [item for item in items if item.get("answer_tell_position") == wanted]
    if args.limit:
        items = items[: args.limit]
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    report: dict[str, Any] = {
        "format": "taiji-r2-a27-emission-side-v1",
        "prereg": "plans/reference/PLAN-A-27_three-gaps-decision-brief_20260928.md §2（战线二 乙档）",
        "manifest": MANIFEST.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(MANIFEST),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _sha256(circuit),
        "position_filter": args.position,
        "items": len(items),
        "arms": {},
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for arm in arms:
        runtime = SeedRuntime.load(checkpoint)
        runtime.enable_copy_circuit(circuit)
        rows, state = run_arm(runtime, items, arm)
        hits = sum(row["hit"] for row in rows)
        ranks = [row["target_rank"] for row in rows if row["target_rank"] is not None]
        owners: dict[str, int] = {}
        for row in rows:
            owners[row["emitted_owner"]] = owners.get(row["emitted_owner"], 0) + 1
        report["arms"][arm] = {
            "answered": len(rows),
            "hits": hits,
            "hit_rate": round(hits / max(len(rows), 1), 4),
            "emitted_owner": owners,
            "chosen_is_labelled_true": sum(
                1 for row in rows if row["chosen_is_labelled"] is True
            ),
            "target_rank_median": (sorted(ranks)[len(ranks) // 2] if ranks else None),
            "target_rank_top1": sum(1 for value in ranks if value == 1),
            "chosen_calls_in_generation": state["chosen_calls_in_generation"],
            "lock_path_calls_in_generation": state["lock_path"],
            "best_match_calls_in_generation": state["best_match_calls_in_generation"],
            "best_match_calls_total": state["best_match_calls"],
            "copy_evidence_nonzero_steps": state["evidence_nonzero_steps"],
            "oracle_label_used": state["used_oracle_label"],
            "rows": rows,
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "hits": f"{hits}/{len(rows)}",
                    "owners": owners,
                    "chosen_calls_gen": state["chosen_calls_in_generation"],
                    "lock_path_gen": state["lock_path"],
                    "best_match_calls_gen": state["best_match_calls_in_generation"],
                },
                ensure_ascii=False,
            )
        )

    report["instrument_guard"] = {
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "every_arm_answered": all(
            payload["answered"] == len(items) for payload in report["arms"].values()
        ),
        "generation_path_observed": all(
            payload["chosen_calls_in_generation"] > 0 for payload in report["arms"].values()
        ),
    }
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"guard": report["instrument_guard"], "out": out.name}, ensure_ascii=False))
    return 0 if all(report["instrument_guard"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())