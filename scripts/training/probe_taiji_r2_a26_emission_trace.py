"""A2.6 诊断（零训练）：内容就在库里、选择也替它做对了，**为什么还是发不出来**。

前件：`SPEC-A-17` §10——把"挑哪条记忆"替模型做对只得 9/16，且 **7 题在两臂下都失败**，
所以瓶颈已移到"告知之后那一侧"。本件不猜原因，直接把每题的生成过程摊开成分诊类别。

**做法**：对失败题按 oracle 历史（只留含答案词的那条告知）铺到「答：」之后，
然后**让模型自由贪心生成**（不 teacher-forcing，否则看不到偏离），逐步记录：

| 字段 | 含义 |
|---|---|
| `argmax_byte` | 模型这一步真正会发的字节（贪心） |
| `target_byte` | 答案串第 k 个字节（越过串尾则 None） |
| `copy_top_byte` | copy 分布自己票选出的字节（`weights.argmax()` 对应的内容字节） |
| `copy_mass_on_target` | copy 分布落在目标字节上的质量 |
| `gate_value` | 发射门当前开度（带符号） |
| `p_with_copy` / `p_vocab_only` | 目标字节在"含 copy 证据"与"只词汇 logits"两种读法下的概率 |
| `forced_open_would_hit` | 反事实：把 gate 覆写成全开（20）时，目标字节是否成为 argmax |

**分诊规则（先冻结，跑完不改）**——每题给一个主类，统计 7 题落在哪类：
**有序决策树**（先匹配者胜；顺序就是优先级，便于逐条复核）：
1. `no_trace` 没取到轨迹；2. `emitted` 每步都命中 ⇒ 这题本不该在失败清单里
   （同时看 `contradicts_failure_list`：为真说明本探针与 ceiling/CAP 构造不一致，结论回炉）；
3. `gate_closed`＝`max|gate| < 1` **且**存在某步"强行全开即命中" ⇒ 门没开（学习制度/优势信号侧）；
4. `address_miss`＝没有任何一步 `copy_top_byte == target_byte` **且** `copy_mass_on_target` 峰值 <0.2
   ⇒ copy 分布自己指不到目标（键/寻址侧；含"门没开但全开也没用"这一子类）；
5. 其余判 `emission_loses` ⇒ copy 有质量而最终 argmax 仍不是它（发射混合被词汇 logits 压住）。
三类修法互不相同（改训练制度 / 改键与寻址 / 改发射混合），所以先分诊再动手。

规则的含义：三类**修法完全不同**（改训练制度 / 改键与寻址 / 改发射混合），
所以现在最贵的是"猜"，最便宜的是先分诊。

**链路同一性自检（先于跑）**：用与产品一致的读法
`readout.probabilities(ctx, episodic_evidence=base + gate·copy_dist)`，
并在第一步与 `observe()` 真正给出的末位概率**逐位对比**，不一致就拒跑——
本仓多次实测读数依链路而定，诊断也不许例外。

纪律：零训练（`learn=False`）、`checkpoints/` 只读且跑前后 sha256 复核、只写 `reports/` 一份新件；
失败题清单从 §10 的 ceiling 读数里**机器读取**（不手抄题号），读不到就直接退出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CEILING_REPORT = PROJECT_ROOT / "reports/taiji_r2_a25_selection_ceiling_20260925.json"
#: 反事实"全开"的幅度，与 §4.1 结构存在性判据用的是同一个数（不新调参）。
FORCED_GATE = 20.0
MAX_STEPS = 12
#: `|gate|` 小于此值算"门基本没开"（训练里 gate 饱和到几十，零初始化是 0）。
GATE_EPS = 1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failing_item_ids(ceiling_report: Path = CEILING_REPORT) -> list[str]:
    """从 §10 天花板读数里机器读取"两臂都失败"的题号。"""
    data = json.loads(ceiling_report.read_text(encoding="utf-8"))
    asis = {row["id"]: bool(row["hit_strict"]) for row in data["arms"]["asis_with_circuit"]["rows"]}
    oracle = {
        row["id"]: bool(row["hit_strict"]) for row in data["arms"]["oracle_with_circuit"]["rows"]
    }
    return sorted(item_id for item_id in asis if not asis[item_id] and not oracle.get(item_id))


def _oracle_history(substrate: Any, circuit: Any, turns: list[str], tokens: list[str]) -> list[Any]:
    """铺出 oracle 历史：只保留**含可复制答案词**的那条告知，并取模型对它的自答。

    与 ceiling 探针同一构造（"取第一条含答案词的告知"），不另起一套——两处构造不一致的话，
    分诊就是在错的样本上做的。
    """
    from api.seed_runtime import SeedRuntime, record_told_history

    prior = [index for index, turn in enumerate(turns[:-1]) if any(t in turn for t in tokens)]
    if not prior:
        return []
    keep = prior[0]
    history: list[tuple[str, str]] = []
    record_told_history(substrate, circuit, history, episode_id="a26:told")
    prompt = SeedRuntime._serialize(turns[keep], history)
    reply = _reply_clean(
        substrate.generate(prompt.encode("utf-8"), 64, stop_at_boundary=True, sample=False)
    )
    history.append((turns[keep], reply))
    return history


def _reply_clean(raw: bytes) -> str:
    from api.seed_runtime import _TURN_MARKERS

    text = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        cut = text.find(marker)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


def trace_item(
    substrate: Any, circuit: Any, *, turns: list[str], tokens: list[str], told: str
) -> dict[str, Any]:
    """对一题做逐步轨迹，并给出分诊类别。"""
    from api.seed_runtime import SeedRuntime, record_told_history

    config = substrate.config
    answer = next((token for token in tokens if token in told), "")
    if not answer:
        return {"id": None, "class": "no_copyable_answer"}
    answer_bytes = answer.encode("utf-8")

    #: 入库只走一条路径：`_oracle_history` 内部按线上顺序（先入库再生成自答），
    #: 这里不再额外 record 一次，否则"轨迹诊断用的样本"与 ceiling/CAP 用的样本不同形。
    history = _oracle_history(substrate, circuit, turns, tokens)
    prompt = SeedRuntime._serialize(turns[-1], history)
    record_told_history(substrate, circuit, history, episode_id="a26:final")
    prompt_bytes = prompt.encode("utf-8")

    substrate.reset_dynamics(episode_id="generation")
    substrate.observe(
        int(config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    step = None
    for symbol in prompt_bytes:
        step = substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
    assert step is not None
    product_probs = step.probabilities.detach().cpu().clone()

    events = circuit.store.events()
    #: 库里应当正好一条告知（oracle 历史）；不是就说明入库构造与 ceiling/CAP 不同形。
    store_contents = [event.content.decode("utf-8", errors="replace") for event in events]
    rows: list[dict[str, Any]] = []
    chain_ok = True
    emitted_bytes = bytearray()
    prev_byte = int(prompt_bytes[-1])
    for k in range(min(MAX_STEPS, len(answer_bytes))):
        state = substrate._state
        cue = substrate.fabric.cortical_context(state.regions).detach().cpu().clone()
        ctx = state.motor_context
        snap = circuit.addressing(cue=cue, f1_context=ctx, prev_byte=prev_byte)
        if snap is None:
            break
        base_evidence = float(
            config.consolidation_read_gain
        ) * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
        readout = substrate.predictive_readout
        gate = float(snap["gate_value"])
        with_copy = readout.probabilities(
            ctx, episodic_evidence=base_evidence + gate * snap["copy_distribution"]
        )
        vocab_only = readout.probabilities(ctx, episodic_evidence=base_evidence)
        copy_top_byte = int(snap["codes"][int(snap["scores"].argmax())])
        target_byte = int(answer_bytes[k])
        if k == 0:
            #: 链路同一性：手算的发射读法必须与 `observe()` 给的末位概率**逐位**一致，
            #: 否则整份轨迹都是在另一条链上取的，分诊结论作废（返回码 2 拒绝出件）。
            chain_ok = bool(torch.equal(with_copy, product_probs))
        forced = readout.probabilities(
            ctx, episodic_evidence=base_evidence + FORCED_GATE * snap["copy_distribution"]
        )
        argmax_byte = int(with_copy.argmax())
        emitted = argmax_byte == target_byte
        emitted_bytes.append(argmax_byte)
        rows.append(
            {
                "step": k,
                "argmax_byte": argmax_byte,
                "target_byte": target_byte,
                "copy_top_byte": copy_top_byte,
                "copy_mass_on_target": round(float(snap["copy_distribution"][target_byte]), 6),
                "gate_value": round(gate, 4),
                "p_with_copy": round(float(with_copy[target_byte]), 8),
                "p_vocab_only": round(float(vocab_only[target_byte]), 8),
                #: 反事实：把门强行全开（与 §4.1 结构存在性判据同一个幅度），目标字节会不会成为 argmax。
                "forced_open_would_hit": bool(int(forced.argmax()) == target_byte),
                "emitted": emitted,
            }
        )
        #: **不在命中处停下**：失败题的轨迹要看的是"整串为什么没出来"，
        #: 命中一步就 break 会把后面的偏离步全部藏掉（上一版就是这么写的）。
        prev_byte = argmax_byte
        substrate.observe(argmax_byte, learn=False, readout="predictive", use_memory=False)
    reconstructed = emitted_bytes.decode("utf-8", errors="replace")
    classification = _classify(rows)
    return {
        "told": told,
        "answer": answer,
        "store_contents": store_contents,
        "reconstructed_answer": reconstructed[:80],
        #: 自检：若重构串真的含答案词，这题就不该出现在失败清单里——出现即说明
        #: 本探针与 ceiling/CAP 两处构造不一致，结论必须回炉而不是接着分诊。
        "contradicts_failure_list": answer in reconstructed,
        "chain_identical": chain_ok,
        "steps": len(rows),
        "class": classification,
        "trace": rows,
    }


def _classify(rows: list[dict[str, Any]]) -> str:
    """按文件头冻结的**有序**决策树分诊（顺序即优先级，便于逐条复核）。

    四类之外的两种非诊断结局也显式命名，不混进四类里：`no_trace`（没铺开轨迹）、
    `emitted`（每步都命中——那这题根本不该在失败清单里，见 `contradicts_failure_list`）。
    """
    if not rows:
        return "no_trace"
    if all(bool(row["emitted"]) for row in rows):
        return "emitted"
    gates = [abs(float(row["gate_value"])) for row in rows]
    forced_hits = [bool(row["forced_open_would_hit"]) for row in rows]
    aimed = [int(row["copy_top_byte"]) == int(row["target_byte"]) for row in rows]
    best_mass = max(float(row["copy_mass_on_target"]) for row in rows)
    if max(gates) < GATE_EPS and any(forced_hits):
        return "gate_closed"  # 门基本没开，但强行全开就命中 ⇒ 问题在开关的学习
    if not any(aimed) and best_mass < 0.2:
        return "address_miss"  # copy 分布自己就指不到目标字节（含"门没开且全开也没用"）
    return "emission_loses"  # copy 指对了/有质量，但最终 argmax 仍被词汇 logits 压住


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit", default="output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    ids = failing_item_ids()
    if not ids:
        print(json.dumps({"error": "ceiling 读数里没找到两臂都失败的题"}))
        return 1

    from score_taiji_r2_copy_strict_cap import copyable_tokens, load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    items = load_items()
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    payload = torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    substrate.copy_circuit.load_payload(payload)

    per_item: dict[str, Any] = {}
    for item_id in ids:
        item = items[item_id]
        turns = [str(turn) for turn in item["turns"]]
        tokens = list(copyable_tokens(item))
        told = next((turn for turn in turns[:-1] if any(t in turn for t in tokens)), "")
        if not told:
            per_item[item_id] = {"class": "no_copyable_answer"}
            continue
        per_item[item_id] = trace_item(
            substrate, substrate.copy_circuit, turns=turns, tokens=tokens, told=told
        )
        per_item[item_id]["id"] = item_id

    counts: dict[str, int] = {}
    for record in per_item.values():
        counts[str(record.get("class"))] = counts.get(str(record.get("class")), 0) + 1
    traced = [record for record in per_item.values() if record.get("trace")]
    #: 链同一性只对"真取了轨迹的题"负责——`no_copyable_answer` 那类没有发射步骤可比，
    #: 把它们算进分母会让探针自己假失败。
    chain_all = bool(traced) and all(bool(record.get("chain_identical")) for record in traced)
    report = {
        "format": "taiji-r2-a26-emission-trace-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §10",
        "failing_items": ids,
        "chain_identical_all": chain_all,
        "class_counts": counts,
        "gate_abs_median": statistics.median(
            [
                abs(float(row["gate_value"]))
                for record in per_item.values()
                for row in record.get("trace", [])
            ]
            or [0.0]
        ),
        "copy_mass_on_target_median": statistics.median(
            [
                float(row["copy_mass_on_target"])
                for record in per_item.values()
                for row in record.get("trace", [])
                if row["copy_mass_on_target"] is not None
            ]
            or [0.0]
        ),
        "per_item": per_item,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_r2_a26_emission_trace_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(ids),
                "classes": counts,
                "chain_identical_all": chain_all,
                "base_unchanged": report["base_sha256_unchanged"],
                "out": out.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    )
    return 0 if chain_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
