"""PLAN-A-30 §2f 第 3 条的定位：为什么仍有一条答复是 256 字同字拖写（停不下来）。

`repetition_penalty=2.0` 在表层链上把成句从 12/72 抬到 71/72，但**最坏那条没消**：
seed-A 两段题面的 `max_longest_run` 都是 256（一条答复整段是同一个字）。
本探针要回答的是二选一，两者修法完全不同：

  (A) **掩码饿死**：那一步的合法后继集只剩 1 个候选 ⇒ 病在"合法集 × 证据集中"的交集，解码侧再动一手；
  (B) **模型不肯停**：合法集里候选不少、边界符概率也没被压成零，但每一步它就是比续字节低
      ⇒ 病在**停止信号本身**（没学会"这句说完了"），属训练目标/规模那一层，不是解码参数能治的。

做法（**不改产品源码**，实例级包装；**不重抄生成链**——跟的是产品 `chat()` 真实走过的步）：
把 `substrate.observe` 包一层，记下"这一步被喂进哪个符号"、"这一步返回的概率向量"，
以及"**这一步是从产品源码的哪一行被调的**"（`sys._getframe(1)` 的 `co_name` + `f_lineno`）。

段起点不靠猜。前两版的教训：
  v1 按"1 次边界符 + prompt 字节数"硬切 ⇒ 第 3 步就断（一轮里 `observe` 的调用比这多）；
  v2 改成"拿答复字节在偏移窗口里找吻合段"⇒ 对齐上了，但**自检仍全红**。
v2 全红的第一嫌疑就在仪器自己漏的一步：产品解码环是**先重复惩罚、后 UTF-8 掩码**
（`taiji/model.py` 里 `probabilities / (1.0 + repetition_penalty * counts)` 在 `masked_fill` 之前），
v2 只重放了掩码 ⇒ 拿"旧口径的 argmax"去对"新口径实际发出的字节"，每步都像错。
本版把惩罚按同一口径补进重放；**若自检仍红，则原因不在这条**，本件读数作废、另找。

现在按调用点切段：从 `Taiji.generate` 的源码行号自动推出**生成环的行号区间**
（不写死数字，源码移动它跟着移动），落在区间内的记录就是生成步，喂进符号的序列就是答复字节；
区间外、同属 `generate` 的那些就是边界符与提问的喂入。
逐步验「按产品口径重放（惩罚 → 掩码 → 取最大）后的最优符号 == 本步实际被喂的符号」；
不相等 ⇒ 该步计入 `argmax_mismatch_steps`，任一项红 ⇒ `all_items_reconstructed=False`，
本件读数一律作废（不拿来下结论），并把**前 5 条失败步**原样落进件里好一次跑看清。

用法：
    python scripts/training/probe_taiji_a30_stop_failure.py \
        --circuit output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt --limit 24
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
TIE_TOLERANCE = 1e-9  # 与产品同为 float32，取等号按"并列即算命中"处理


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


def _longest_same_char_run(text: str) -> int:
    best = run = 1
    for a, b in zip(text, text[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if text else 0


def generation_loop_span(generate: Any) -> tuple[int, int]:
    """从产品源码**自己**推出"生成环"的行号区间，避免把行号写死进仪器。"""

    import inspect

    lines, start = inspect.getsourcelines(generate)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("for ") and "range(length)" in stripped and stripped.endswith(":"):
            indent = len(line) - len(line.lstrip())
            end = index
            for following in range(index + 1, len(lines)):
                body = lines[following]
                if body.strip() and (len(body) - len(body.lstrip())) > indent:
                    end = following
            return start + index, start + end
    raise RuntimeError("generation loop (`for _ in range(length)`) not found in Taiji.generate")


def replay_step(
    probabilities: Any,
    byte: int,
    boundary: int,
    state: tuple[int, int],
    recent: list[int],
    penalty: float,
    window: int,
) -> dict[str, Any]:
    """按**产品解码环的口径**重放一步：先重复惩罚，后 UTF-8 掩码，再取最大。

    产码顺序照 `taiji/model.py::Taiji.generate` 的环体：惩罚只看已发字节的最后 `window` 个，
    掩码合法集 = `utf8_allowed(remaining, lead) | {boundary}`。
    """

    import torch

    from taiji.utf8_state import utf8_allowed

    probabilities = probabilities.detach().cpu().float()
    boundary_before_penalty = float(probabilities[boundary])
    if penalty > 0.0 and recent:
        counts = torch.zeros_like(probabilities)
        for symbol in recent[-window:]:
            counts[int(symbol)] += 1.0
        probabilities = probabilities / (1.0 + penalty * counts)
    legal = sorted(set(utf8_allowed(state[0], state[1])) | {boundary})
    vector = [float(value) for value in probabilities]
    best = max(vector[index] for index in legal)
    legal_ranked = sorted(legal, key=lambda index: -vector[index])
    return {
        "emitted_is_argmax": bool(vector[byte] >= best - TIE_TOLERANCE),
        "p_emitted": round(vector[byte], 6),
        "p_boundary": round(vector[boundary], 6),
        "p_boundary_before_penalty": round(boundary_before_penalty, 6),
        "boundary_rank_in_legal": legal_ranked.index(boundary) + 1,
        "legal_candidates": len(legal) - 1,
        "ratio_best_over_boundary": (
            round(best / vector[boundary], 3) if vector[boundary] > 0 else None
        ),
        "boundary_is_argmax": bool(vector[boundary] >= best - TIE_TOLERANCE),
        "boundary_probability_is_zero": vector[boundary] == 0.0,
    }


def char_membership_of_run(text: str) -> list[bool]:
    """逐**字节**标出"这个字节所属的汉字处在一段同字连写里"。

    不能按字节直接比相等：一个汉字三字节，`我我` 的字节流是 `E6 88 91 E6 88 91`，
    逐字节比较会把"同一个字连着写"读成"不重复"，而把"同一字的中途续字节"读成"重复"。
    """

    chars = list(text)
    repeated = [
        bool(
            (index > 0 and chars[index] == chars[index - 1])
            or (index + 1 < len(chars) and chars[index] == chars[index + 1])
        )
        for index in range(len(chars))
    ]
    membership: list[bool] = []
    for index, char in enumerate(chars):
        membership.extend([repeated[index]] * len(char.encode("utf-8")))
    return membership


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit",
        default=None,
        help="不给＝不挂回路，即**今日出厂那面**（丁-2 的判据要用它，不能只在挂回路面上说事）",
    )
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument(
        "--ids",
        default=None,
        help="只跑题面里这些 id（逗号分隔）——长预算档用它点住最坏那几条，不必整批重跑",
    )
    parser.add_argument("--penalty", type=float, default=2.0)
    parser.add_argument("--penalty-window", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    all_items = json.loads(manifest.read_text(encoding="utf-8"))["dimensions"]["X"]["items"]
    if args.ids:
        wanted = [part.strip() for part in args.ids.split(",") if part.strip()]
        known = {item["id"] for item in all_items}
        missing = [name for name in wanted if name not in known]
        if missing:
            parser.error(f"ids not in manifest: {missing}")
        items = [item for item in all_items if item["id"] in wanted]
    else:
        items = all_items[: args.limit]

    from api.seed_runtime import _TURN_MARKERS, SeedRuntime
    from taiji.utf8_state import advance_utf8, trim_partial_tail

    runtime = SeedRuntime.load(checkpoint)
    if args.circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    substrate = runtime.model.substrate
    boundary = int(substrate.config.boundary_symbol)
    loop_first, loop_last = generation_loop_span(type(substrate).generate)

    records: list[dict[str, Any]] = []
    original_observe = substrate.observe

    def observing(symbol: Any, **kwargs: Any) -> Any:
        frame = sys._getframe(1)
        step = original_observe(symbol, **kwargs)
        in_loop = frame.f_code.co_name == "generate" and loop_first <= frame.f_lineno <= loop_last
        records.append(
            {
                "symbol": int(symbol),
                "probabilities": step.probabilities.detach().cpu().clone(),
                "caller": f"{frame.f_code.co_name}:{frame.f_lineno}",
                "in_generation_loop": in_loop,
            }
        )
        return step

    substrate.observe = observing  # type: ignore[method-assign]

    per_item: list[dict[str, Any]] = []
    offenders: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    caller_totals: dict[str, int] = {}
    for item in items:
        history: list[tuple[str, str]] = []
        item_rows: list[dict[str, Any]] = []
        surface_checks: list[dict[str, Any]] = []
        worst_run = 0
        for index, turn in enumerate([str(t) for t in item["turns"]]):
            records.clear()
            answer = runtime.chat(
                turn,
                history=history,
                learn=False,
                max_length=args.max_length,
                repetition_penalty=args.penalty,
            )
            loop_records = [record for record in records if record["in_generation_loop"]]
            pre_records = [
                record
                for record in records
                if record["caller"].startswith("generate:") and not record["in_generation_loop"]
            ]
            for record in records:
                caller_totals[record["caller"]] = caller_totals.get(record["caller"], 0) + 1
            fed = [int(record["symbol"]) for record in loop_records]
            raw = trim_partial_tail(bytes(fed)).decode("utf-8", errors="replace")
            #: 产品出口其实有**两种**收口：边界的 `stop_at_boundary`（模型自己说完了），
            #: 以及生成串里再次出现 `问：` 时的**事后截断**（`chat()` 里那段 marker 切割）。
            #: 两手都必须分开记，因为"预算决定长度"这句话很容易被说过头——
            #: 实测这条面上两者都不出现（72 次生成：边界 0 次、`问：` 0 次 ⇒ 72/72 吃满预算）。
            #: （默认语料每篇只有一个 `问：` 与一个 `答：` ⇒ 两种停止编码在训练里的**频次其实相等**，
            #: 谁先出现是分布问题，不能靠推理定。）
            marker_at = min(
                (index for marker in _TURN_MARKERS if (index := raw.find(marker)) >= 0),
                default=None,
            )
            raw = (raw[:marker_at] if marker_at is not None else raw).strip()
            surface_checks.append(
                {
                    "fed_bytes": len(fed),
                    "surface_is_replayed_raw": bool(raw == answer),
                    "turn_marker_fired": marker_at is not None,
                    "cut_by_marker_bytes": marker_at,
                    "answer_head": answer[:24],
                }
            )
            if not pre_records or not fed:
                surface_checks[-1]["replay_skipped"] = (
                    "no-pre-generation-observe" if not pre_records else "no-generated-bytes"
                )
                continue
            state = (0, 0)
            run_membership = char_membership_of_run(bytes(fed).decode("utf-8", errors="replace"))
            for position, record in enumerate(loop_records):
                basis = (
                    pre_records[-1]["probabilities"]
                    if position == 0
                    else loop_records[position - 1]["probabilities"]
                )
                byte = int(record["symbol"])
                row = replay_step(
                    basis,
                    byte,
                    boundary,
                    state,
                    fed[:position],
                    args.penalty,
                    args.penalty_window,
                )
                row["step"] = position
                row["in_run"] = bool(position < len(run_membership) and run_membership[position])
                if not row["emitted_is_argmax"]:
                    failure_examples.append({"id": item["id"], **row})
                item_rows.append(row)
                state = advance_utf8(state[0], state[1], byte)
            worst_run = max(worst_run, _longest_same_char_run(answer))
            if index + 1 < len(item["turns"]):
                history.append((turn, answer))
        legal_sizes = [row["legal_candidates"] for row in item_rows]
        ranks = [row["boundary_rank_in_legal"] for row in item_rows]
        ratios = [
            row["ratio_best_over_boundary"] for row in item_rows if row["ratio_best_over_boundary"]
        ]
        summary = {
            "id": item["id"],
            "answer_run": worst_run,
            "steps": len(item_rows),
            "generations": len(surface_checks),
            "generations_cut_by_turn_marker": sum(
                1 for check in surface_checks if check["turn_marker_fired"]
            ),
            "argmax_mismatch_steps": sum(1 for row in item_rows if not row["emitted_is_argmax"]),
            "surface_matches_replayed_raw": all(
                check["surface_is_replayed_raw"] for check in surface_checks
            ),
            "surface_checks": surface_checks,
            "steps_with_single_legal_candidate": sum(1 for size in legal_sizes if size <= 1),
            "steps_with_zero_boundary_probability": sum(
                1 for row in item_rows if row["boundary_probability_is_zero"]
            ),
            "steps_boundary_is_argmax": sum(1 for row in item_rows if row["boundary_is_argmax"]),
            "median_boundary_rank_in_legal": sorted(ranks)[len(ranks) // 2] if ranks else None,
            "median_ratio_best_over_boundary": (
                sorted(ratios)[len(ratios) // 2] if ratios else None
            ),
            "run_steps": sum(1 for row in item_rows if row["in_run"]),
        }
        per_item.append(summary)
        if worst_run >= 20 and item_rows:
            run_rows = [row for row in item_rows if row["in_run"]]
            run_ratios = [
                row["ratio_best_over_boundary"]
                for row in run_rows
                if row["ratio_best_over_boundary"]
            ]
            offenders.append(
                {
                    **summary,
                    "run_legal_hist": {
                        str(size): sum(1 for row in run_rows if row["legal_candidates"] == size)
                        for size in sorted({row["legal_candidates"] for row in run_rows})
                    },
                    "run_boundary_rank_hist": {
                        str(rank): sum(
                            1 for row in run_rows if row["boundary_rank_in_legal"] == rank
                        )
                        for rank in sorted({row["boundary_rank_in_legal"] for row in run_rows})
                    },
                    "run_ratio_min": min(run_ratios) if run_ratios else None,
                    "run_ratio_median": (
                        sorted(run_ratios)[len(run_ratios) // 2] if run_ratios else None
                    ),
                    "run_rows_sample": run_rows[:12],
                }
            )

    substrate.observe = original_observe  # type: ignore[method-assign]
    report = {
        "format": "taiji-a30-stop-failure-v3",
        "prereg": "plans/reference/PLAN-A-30_surface_repetition_localization_20260928.md §2f 第 3 条",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": _payload_sha(args.circuit),
        "repetition_penalty": args.penalty,
        "repetition_window": args.penalty_window,
        "max_length": args.max_length,
        "generation_loop_lines": [loop_first, loop_last],
        "items": len(items),
        "instrument_guard": {
            "observe_calls_recorded": bool(records),
            "observe_calls_by_caller": dict(
                sorted(caller_totals.items(), key=lambda pair: -pair[1])[:12]
            ),
            "all_items_reconstructed": not any(row["argmax_mismatch_steps"] for row in per_item),
            "items_failing_reconstruction": [
                row["id"] for row in per_item if row["argmax_mismatch_steps"]
            ],
            "total_argmax_mismatch_steps": sum(row["argmax_mismatch_steps"] for row in per_item),
            "all_surfaces_are_replayed_raw": all(
                row["surface_matches_replayed_raw"] for row in per_item
            ),
            "items_whose_surface_is_not_model_bytes": [
                row["id"] for row in per_item if not row["surface_matches_replayed_raw"]
            ],
            "total_steps": sum(row["steps"] for row in per_item),
            "generations": sum(row["generations"] for row in per_item),
            "generations_cut_by_turn_marker": sum(
                row["generations_cut_by_turn_marker"] for row in per_item
            ),
            "generations_eating_full_budget": sum(
                row["generations"] - row["generations_cut_by_turn_marker"] for row in per_item
            ),
            "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        },
        "failure_examples": failure_examples[:5],
        "offender_count": len(offenders),
        "offenders": offenders,
        "per_item": per_item,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_a30_stop_failure_20260928_ml{args.max_length}.json"
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
                "items": report["items"],
                "guard": report["instrument_guard"],
                "failure_examples": report["failure_examples"],
                "offender_count": report["offender_count"],
                "offenders": [
                    {
                        "id": offender["id"],
                        "run": offender["answer_run"],
                        "steps": offender["steps"],
                        "run_steps": offender["run_steps"],
                        "starved": offender["steps_with_single_legal_candidate"],
                        "boundary_zero_steps": offender["steps_with_zero_boundary_probability"],
                        "boundary_argmax_steps": offender["steps_boundary_is_argmax"],
                        "run_legal_hist": offender["run_legal_hist"],
                        "run_boundary_rank_hist": offender["run_boundary_rank_hist"],
                        "run_ratio_min": offender["run_ratio_min"],
                        "run_ratio_median": offender["run_ratio_median"],
                    }
                    for offender in offenders
                ],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
