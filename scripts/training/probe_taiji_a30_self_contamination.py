"""PLAN-A-30 / DEBT-G10：产品把自己写坏的答复喂回训练面，会不会**在会话内自我强化**。

`SeedRuntime.chat()` 缺省 `learn=True`，收尾执行
`self.model.learn_bytes((text + answer).encode("utf-8"), include_boundary=True)`
（`api/seed_runtime.py:446-452`），而 `answer` 就是**本轮表层答复**；下一轮同一次 `chat()`
又会把上一条答复原样序列化进 prompt 的 `答：` 段。
⇒ 一条同字拖写的答复至少有**两条**回到模型面前的通路：基底的训练面权重、下一轮的 prompt 文本。

本件要测的是这条通路有没有**可见后果**，方法是五臂配对（各臂自己一个新载入的 runtime，
**不共享内存态** ⇒ 配对按题号与轮号成立，与 P3b 那条"重放按检查点血缘算"的规矩一致）：

* `learn_false_a` / `learn_false_b`：每轮只生成、不回写（＝本件其它读数一直用的那条面）；
  两条同装配 ⇒ 一次跑内的**噪声地板**，二者逐条相同才有一根可参考的零线；
* `learn_true`：产品缺省面（每轮答复回写训练面），且下一轮 prompt 里带着上一条答复；
* `learn_true_scrubbed` / `learn_false_scrubbed`：把历史里上一条答复换成固定占位＝**断开 prompt 回路**，
  一个照做回写、一个不回写。

四格凑成一个 2×2（回写开关 × prompt 回路开关），于是有**四条单变量**对照：
`writeback_loop_on ＝ true − false_a`、`writeback_loop_off ＝ true_scrubbed − false_scrubbed`、
`loop_wb_on ＝ true − true_scrubbed`、`loop_wb_off ＝ false_a − false_scrubbed`。
四臂那版缺 `learn_false_scrubbed`，只拿到对角线 `true_scrubbed − false_a`（同时动了两个变量），
那不算效应量——1 题冒烟（n=1）暴露的正是这个设计缺口（§2t 第 3 条的隔离臂由此补全）。
**入库侧的实情要说明**：`_record_told_history` 只把历史的**用户轮**
写进剪贴板证据库（`api/seed_runtime.py:93` 的 `store.record(user.encode(...))`），答复并不入库
⇒ 本件分离的两条通路是"训练面权重"与"下一轮 prompt 文本"，**不是**"证据库"。

各臂的 `history` 用**本臂自己**的答复累积（这样每一臂都是自洽的多轮会话）。
配对比第 2、3 轮上的三件：最长同字连写、成句（带语料 n 元模型的 `well_formed`）、严格命中。
若 `learn_true` 臂在后轮更差 ⇒ 自强化成立并给出幅度；若各臂同分布 ⇒ 这条通路**当前无害**，
`DEBT-G10` 降级为"码上存在、实测不放大"，不许继续按风险吓人。

仪器自带的生效证据（不是推理）：按实例级数 `learn_bytes` 的**调用次数**——
`learn=True` 臂必须等于文本条数、`learn=False` 臂必须为 0，两列一起才叫对照
（第一版想用 `model.tick` 增量，那是错的：生成本身就推进 tick，两臂都会 >0）。
基座只读：`base_sha256_unchanged` 为假则整件作废。

用法：
    python scripts/training/probe_taiji_a30_self_contamination.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt \
        --limit 32 --arm-order treated_first
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
#: 五臂凑成一个 2×2（回写开关 × 历史里是否带着上一条自己的答复）＋一条同装配零线：
#:
#: |            | 历史带自己的答复（回路开） | 历史换占位（回路断） |
#: |---|---|---|
#: | **不回写** | `learn_false_a`（＝产品把 learn 关掉） | `learn_false_scrubbed` |
#: | **回写**   | `learn_true`（＝产品缺省面）           | `learn_true_scrubbed` |
#:
#: 少了右下那格就只能拿到对角线（`scrubbed − false_a` 同时动了两个变量），
#: 拿不到"回写单独值多少"——这是 1 题冒烟暴露出来的设计缺口，不是读数。
ARMS = (
    "learn_false_a",
    "learn_false_b",
    "learn_true",
    "learn_true_scrubbed",
    "learn_false_scrubbed",
)
#: 装配顺序也是一个变量：汇报里所有臂在**同一进程**里先后载入，
#: 若首次载入与随后载入的得出不同（torch 线程池冷热度等），那"被测臂总是最后一个"就是混淆。
#: 两条控制臂只证明第 1、2 次载入相同，证不了第 3 次——故把顺序也做成可交换的档。
#: `treated_first` 把 `learn_true` 挪到**第 1 个载入**，与 §2q 那份三臂件同位 ⇒ 可逐位对照。
ARM_ORDERS = {
    "controls_first": (
        "learn_false_a",
        "learn_false_b",
        "learn_true",
        "learn_true_scrubbed",
        "learn_false_scrubbed",
    ),
    "treated_first": (
        "learn_true",
        "learn_true_scrubbed",
        "learn_false_scrubbed",
        "learn_false_a",
        "learn_false_b",
    ),
}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(value) -> str | None:
    """回路 payload 的摘要：件里只记路径不够（`PLAN-A-24` rev22 那条复现性债）。"""

    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return _sha256(path)


def _answer_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _longest_same_char_run(text: str) -> int:
    best = run = 1
    for a, b in zip(text, text[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if text else 0


def _contrast(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """一格减另一格＝一个**单变量**对照（b 是零线）。四件指标一起给，不挑边的读。"""

    return {
        "formed": a["formed"] - b["formed"],
        "hits": a["hits"] - b["hits"],
        "mean_run": round(a["mean_longest_run"] - b["mean_longest_run"], 3),
        "run_ge_20": a["texts_with_run_ge_20"] - b["texts_with_run_ge_20"],
    }


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str | None,
    *,
    arm_name: str,
    learn: bool,
    scrub_history: bool = False,
    penalty: float,
    max_bytes: int,
) -> dict[str, Any]:
    """一臂＝一个新载入的受检基底 ＋ 该产品缺省位（learn 只有这一处差别）。"""

    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    if circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit)
    ngram = build_ngram_model()

    #: "回写被走到"的证据不能拿 `tick` 增量当量——生成本身也推进 tick，两臂都会 >0（第一版就是这么写的）。
    #: 所以按实例级包一层 `learn_bytes` 数**调用次数**：`learn=True` 臂必须等于文本条数、
    #: `learn=False` 臂必须为 0，两列一起才叫对照（仪器先证明自己被执行到）。
    learn_calls: list[int] = []
    original_learn_bytes = runtime.model.learn_bytes

    def counting_learn_bytes(*a: Any, **k: Any) -> Any:
        learn_calls.append(1)
        return original_learn_bytes(*a, **k)

    runtime.model.learn_bytes = counting_learn_bytes  # type: ignore[method-assign]

    rows: list[dict[str, Any]] = []
    tick_before = int(runtime.model.tick)
    for item in items:
        history: list[tuple[str, str]] = []
        for index, turn in enumerate([str(t) for t in item["turns"]]):
            #: `scrub_history=True` 是把**prompt 通道**切断的一臂：历史里 user 轮照留、
            #: 模型自己上一轮的答复换成固定占位（`learn=True` 的权重回写照常发生）。
            #: 于是 `learn_true` 与 `learn_true_scrubbed` 之差＝prompt 通道，
            #: `learn_true_scrubbed` 与 `learn_false_a` 之差＝权重回写（§2t 第 3 条的隔离臂）。
            arm_history = [(user, "。") for user, _ in history] if scrub_history else history
            answer = runtime.chat(
                turn,
                history=arm_history,
                learn=learn,
                max_length=max_bytes,
                repetition_penalty=penalty,
            )
            rows.append(
                {
                    "id": item["id"],
                    "turn": index,
                    "hit": any(str(tok) in answer for tok in item["expected_contains"]),
                    "well_formed": bool(well_formed(answer, ngram)),
                    "longest_run": _longest_same_char_run(answer),
                    "chars": len(answer),
                    "answer_sha": _answer_sha(answer),
                }
            )
            history.append((turn, answer))
    per_turn: dict[str, Any] = {}
    for turn_index in sorted({row["turn"] for row in rows}):
        subset = [row for row in rows if row["turn"] == turn_index]
        per_turn[f"turn_{turn_index}"] = {
            "texts": len(subset),
            "formed": sum(1 for row in subset if row["well_formed"]),
            "hits": sum(1 for row in subset if row["hit"]),
            "mean_longest_run": round(sum(row["longest_run"] for row in subset) / len(subset), 3),
            "max_longest_run": max(row["longest_run"] for row in subset),
            "texts_with_run_ge_20": sum(1 for row in subset if row["longest_run"] >= 20),
        }
    return {
        "arm": arm_name,
        "learn": learn,
        "learn_bytes_calls": len(learn_calls),
        "learn_applied_ticks_delta": int(runtime.model.tick) - tick_before,
        "items": len(items),
        "texts": len(rows),
        "per_turn": per_turn,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=None)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--max-bytes", type=int, default=256)
    parser.add_argument(
        "--arm-order",
        choices=tuple(ARM_ORDERS),
        default="controls_first",
        help="装配载入顺序；换一档即是位置对照（汇报里落 arm_order）",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"]["X"]["items"][
        : args.limit
    ]

    if set(ARMS) != set(ARM_ORDERS["controls_first"]) or set(ARM_ORDERS["controls_first"]) != set(
        ARM_ORDERS["treated_first"]
    ):
        raise SystemExit("ARMS 与两条 arm_order 不一致 ⇒ 2×2 会缺一格，而件里看不出来")
    arms = [
        run_arm(
            items,
            checkpoint,
            args.circuit,
            arm_name=arm,
            learn=arm.startswith("learn_true"),
            scrub_history=arm.endswith("_scrubbed"),
            penalty=args.repetition_penalty,
            max_bytes=args.max_bytes,
        )
        for arm in ARM_ORDERS[args.arm_order]
    ]
    by_name = {arm["arm"]: arm for arm in arms}
    #: 2×2 的四格＝回写开关 × 历史里是否带着上一条自己的答复。四格齐了才有**单变量**对照；
    #: 四臂那版只拿到对角线 `scrubbed − false_a`（同时动了两个变量），那不算效应量。
    CELLS = {
        "wb_off_loop_on": "learn_false_a",
        "wb_on_loop_on": "learn_true",
        "wb_on_loop_off": "learn_true_scrubbed",
        "wb_off_loop_off": "learn_false_scrubbed",
    }
    deltas: dict[str, Any] = {}
    for key in by_name["learn_false_a"]["per_turn"]:
        cell = {name: by_name[arm]["per_turn"][key] for name, arm in CELLS.items()}
        deltas[key] = {
            "writeback_loop_on": _contrast(cell["wb_on_loop_on"], cell["wb_off_loop_on"]),
            "writeback_loop_off": _contrast(cell["wb_on_loop_off"], cell["wb_off_loop_off"]),
            "loop_wb_on": _contrast(cell["wb_on_loop_on"], cell["wb_on_loop_off"]),
            "loop_wb_off": _contrast(cell["wb_off_loop_on"], cell["wb_off_loop_off"]),
            "cells": cell,
        }
    report = {
        "format": "taiji-a30-self-contamination-v2",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §2v（判读线先于数写下）＋DEBT-G10",
        "question": "产品缺省 learn=True 把本轮答复回写训练面，会不会让后续轮更退化",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _payload_sha(args.circuit),
        "repetition_penalty": args.repetition_penalty,
        "max_bytes": args.max_bytes,
        "arm_order": args.arm_order,
        "manifest": manifest.name,
        "items": len(items),
        "pairing": "同题号同轮号配对；各臂用自载入的 runtime（不共享内存态），各自 history 由本臂答复累积",
        "arms": [{k: v for k, v in arm.items() if k != "rows"} for arm in arms],
        "per_turn_2x2": deltas,
        "instrument_guard": {
            "learn_arms_ran_learn_bytes_once_per_text": all(
                by_name[a]["learn_bytes_calls"] == by_name[a]["texts"]
                for a in ("learn_true", "learn_true_scrubbed")
            ),
            "learn_false_arms_ran_learn_bytes_zero": all(
                by_name[a]["learn_bytes_calls"] == 0
                for a in ("learn_false_a", "learn_false_b", "learn_false_scrubbed")
            ),
            #: 同一次跑内的**噪声地板**：两条同装配 `learn=False` 臂必须逐条相同，
            #: 否则"learn=True 造成的差"里没有一根可参考的零线，本件读数一律作废。
            "control_arms_bitwise_identical": [
                row["answer_sha"] for row in by_name["learn_false_a"]["rows"]
            ]
            == [row["answer_sha"] for row in by_name["learn_false_b"]["rows"]],
            "all_arms_same_text_count": len({arm["texts"] for arm in arms}) == 1,
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        },
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_a30_self_contamination_{args.limit}item_20260928.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "items": len(items),
                "arms": [
                    {
                        "arm": arm["arm"],
                        "ticks_delta": arm["learn_applied_ticks_delta"],
                        "per_turn": arm["per_turn"],
                    }
                    for arm in arms
                ],
                "per_turn_2x2": {
                    key: {k: v for k, v in cell.items() if k != "cells"}
                    for key, cell in deltas.items()
                },
                "control_a_vs_b": [
                    {
                        "turn": row["turn"],
                        "a": row["answer_sha"],
                        "b": by_name["learn_false_b"]["rows"][n]["answer_sha"],
                    }
                    for n, row in enumerate(by_name["learn_false_a"]["rows"])
                    if row["answer_sha"] != by_name["learn_false_b"]["rows"][n]["answer_sha"]
                ][:6],
                "guard": report["instrument_guard"],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
