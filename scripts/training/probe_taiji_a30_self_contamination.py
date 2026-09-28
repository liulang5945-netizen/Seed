"""PLAN-A-30 / DEBT-G10：产品把自己写坏的答复喂回训练面，会不会**在会话内自我强化**。

`SeedRuntime.chat()` 缺省 `learn=True`，收尾执行
`self.model.learn_bytes((text + answer).encode("utf-8"), include_boundary=True)`
（`api/seed_runtime.py:446-452`），而 `answer` 就是**本轮表层答复**；同一支 `chat()`
还会把上一轮答复当历史告知写进剪贴板（`_record_told_history`）。
⇒ 一条同字拖写的答复同时污染**两处**：基底的训练面与复制回路的证据库。

本件要测的是这条通路有没有**可见后果**，方法是两臂配对（各臂自己一个新载入的 runtime，
**不共享内存态** ⇒ 配对按题号与轮号成立，与 P3b 那条"重放按检查点血缘算"的规矩一致）：

* `learn=False` 臂：每轮只生成，不回写（＝本件其它读数一直用的那条面）；
* `learn=True` 臂：产品缺省面（每轮答复回写训练面）。

两臂各自的 `history` 用**本臂自己**的答复累积（这样每一臂都是自洽的多轮会话）。
配对比第 2、3 轮上的三件：最长同字连写、成句（带语料 n 元模型的 `well_formed`）、严格命中。
若 `learn=True` 臂在后轮更差 ⇒ 自强化成立并给出幅度；若两臂同分布 ⇒ 这条通路**当前无害**，
`DEBT-G10` 降级为"码上存在、实测不放大"，不许继续按风险吓人。

仪器自带的生效证据（不是推理）：按实例级数 `learn_bytes` 的**调用次数**——
`learn=True` 臂必须等于文本条数、`learn=False` 臂必须为 0，两列一起才叫对照
（第一版想用 `model.tick` 增量，那是错的：生成本身就推进 tick，两臂都会 >0）。
基座只读：`base_sha256_unchanged` 为假则整件作废。

用法：
    python scripts/training/probe_taiji_a30_self_contamination.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --limit 12
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
ARMS = ("learn_false_a", "learn_false_b", "learn_true")
#: 装配顺序也是一个变量：汇报里所有臂在**同一进程**里先后载入，
#: 若首次载入与随后载入的得出不同（torch 线程池冷热度等），那"被测臂总是最后一个"就是混淆。
#: 两条控制臂只证明第 1、2 次载入相同，证不了第 3 次——故把顺序也做成可交换的档。
ARM_ORDERS = {
    "controls_first": ("learn_false_a", "learn_false_b", "learn_true"),
    "treated_first": ("learn_true", "learn_false_a", "learn_false_b"),
}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer_sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _longest_same_char_run(text: str) -> int:
    best = run = 1
    for a, b in zip(text, text[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if text else 0


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str | None,
    *,
    arm_name: str,
    learn: bool,
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
            answer = runtime.chat(
                turn,
                history=history,
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

    arms = [
        run_arm(
            items,
            checkpoint,
            args.circuit,
            arm_name=arm,
            learn=(arm == "learn_true"),
            penalty=args.repetition_penalty,
            max_bytes=args.max_bytes,
        )
        for arm in ARM_ORDERS[args.arm_order]
    ]
    by_name = {arm["arm"]: arm for arm in arms}
    deltas: dict[str, Any] = {}
    for key in by_name["learn_false_a"]["per_turn"]:
        off = by_name["learn_false_a"]["per_turn"][key]
        on = by_name["learn_true"]["per_turn"][key]
        deltas[key] = {
            "formed_on_minus_off": on["formed"] - off["formed"],
            "hits_on_minus_off": on["hits"] - off["hits"],
            "mean_run_on_minus_off": round(on["mean_longest_run"] - off["mean_longest_run"], 3),
            "run_ge_20_on_minus_off": on["texts_with_run_ge_20"] - off["texts_with_run_ge_20"],
        }
    report = {
        "format": "taiji-a30-self-contamination-v1",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §2h 末条＋DEBT-G10",
        "question": "产品缺省 learn=True 把本轮答复回写训练面，会不会让后续轮更退化",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "repetition_penalty": args.repetition_penalty,
        "max_bytes": args.max_bytes,
        "arm_order": args.arm_order,
        "manifest": manifest.name,
        "items": len(items),
        "pairing": "同题号同轮号配对；两臂各用自载入的 runtime（不共享内存态），各自 history 由本臂答复累积",
        "arms": [{k: v for k, v in arm.items() if k != "rows"} for arm in arms],
        "per_turn_on_minus_off": deltas,
        "instrument_guard": {
            "learn_true_ran_learn_bytes_once_per_text": by_name["learn_true"]["learn_bytes_calls"]
            == by_name["learn_true"]["texts"],
            "learn_false_ran_learn_bytes_zero": by_name["learn_false_a"]["learn_bytes_calls"] == 0
            and by_name["learn_false_b"]["learn_bytes_calls"] == 0,
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
                "on_minus_off": deltas,
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
