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

**历史形态 `--history`**：`oracle`（默认，本件首跑用的就是它）只留含答案词的那条告知，
等于替模型把干扰清干净；`scored` 逐轮走产品原语复原**当初被记分的那份历史**（含模型自己的
脏自答），并额外用 `_answer_raw` 完整重发一遍最后轮来核对"这次未命中是否复现"。
扩展集首跑后加了这一档：oracle 档里 3 轮题的 `address_miss` 明显扎堆，而真实记分链上命中
几乎只发生在 2 轮题——两种形态给出的第一限制因素不是同一个，所以占比必须说清在哪条链上取的。
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
#: 扩展集三臂读数（104 题、实体与评价集与训练表双不相交）——同一支探针在更宽更陌生的样本上
#: 再数一遍成因占比；§12 的结论是 n=6 定不了比例，这一批就是去补那个比例的。
SURFACE_REPORT = PROJECT_ROOT / "reports/taiji_r2_copy_surface_extension_20260925.json"
SURFACE_MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
#: 反事实"全开"的幅度，与 §4.1 结构存在性判据用的是同一个数（不新调参）。
FORCED_GATE = 20.0
MAX_STEPS = 12
#: `|gate|` 小于此值算"门基本没开"（训练里 gate 饱和到几十，零初始化是 0）。
GATE_EPS = 1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failing_item_ids_extension(surface_report: Path = SURFACE_REPORT, arm: int = 0) -> list[str]:
    """从扩展集三臂读数里机器读取"挂了电路仍未命中"的题号（默认 seed-A 那臂）。"""
    data = json.loads(surface_report.read_text(encoding="utf-8"))
    return sorted(
        str(row["id"]) for row in data["treated_arms"][arm]["rows"] if not bool(row["hit"])
    )


def extension_items() -> dict[str, Any]:
    """扩展集题面（与 CAP 评价集同构：`turns` + `expected_contains`）。"""
    payload = json.loads(SURFACE_MANIFEST.read_text(encoding="utf-8"))
    return {
        str(item["id"]): {**item, "dimension": "X"} for item in payload["dimensions"]["X"]["items"]
    }


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


def _scored_history(runtime: Any, turns: list[str]) -> list[tuple[str, str]]:
    """按**记分链**复原真实历史：逐轮调用产品侧那支 `_answer_raw`，历史滚动累积。

    与 `score_taiji_r2_copy_surface_extension.run_arm` 用的是同一个函数、同一个 64 字节上限，
    所以最终喂进去的 prompt 字节与当初判"未命中"时逐位相同；oracle 历史不相同
    （它只留含答案词的那条告知，且库里恒为 1 条）。分诊要落在**被记分的那条链**上。
    """
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    history: list[tuple[str, str]] = []
    for turn in turns[:-1]:
        history.append((turn, _answer_raw(runtime, turn, history)))
    return history


def _product_reply(runtime: Any, substrate: Any, prompt_bytes: bytes) -> str:
    """产品解码原语，但**不重新入库**——用来在"摘掉干扰的库"上重发同一份提示词。

    与 `_answer_raw` 除入库一步外逐字相同（同一个 `generate_input`、同一个 64 字节上限、
    同一套折字），所以两串相等是这条对照成立的先决条件，冒烟里对着核。
    """
    from score_taiji_r2_copy_circuit_chat_cap import MAX_ANSWER_BYTES

    from taiji import InputFrame

    frame = InputFrame(
        input_id=f"a26-abl:{substrate.tick}",
        modality="text",
        payload=prompt_bytes,
        source="r2.a26.ablation",
        timestamp=substrate.tick,
        provenance="external",
        confidence=1.0,
    )
    raw = runtime.model.generate_input(frame, MAX_ANSWER_BYTES, stop_at_boundary=True, sample=False)
    return _reply_clean(raw)


def _keep_only_target_events(circuit: Any, told_bytes: bytes) -> int:
    """摘掉库里的干扰事件，**沿用原 cue**（不重喂前缀——cue 一变就不是同一条链了）。

    返回保留的事件数。0 表示那条告知根本没入库，此题的消融读数无意义，调用方要响。
    """
    keep = [event for event in circuit.store.events() if bytes(event.content) == told_bytes]
    circuit.store.clear()
    for event in keep:
        circuit.store.record(bytes(event.content), event.cue)
    return len(keep)


def _reply_clean(raw: bytes) -> str:
    from api.seed_runtime import _TURN_MARKERS

    text = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        cut = text.find(marker)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


def trace_item(
    substrate: Any,
    circuit: Any,
    *,
    turns: list[str],
    tokens: list[str],
    told: str,
    history_mode: str = "oracle",
    store_mode: str = "all",
    runtime: Any = None,
    feature_sink: list | None = None,
) -> dict[str, Any]:
    """对一题做逐步轨迹，并给出分诊类别。

    `history_mode`：`oracle`＝只留含答案词的那条告知（与 §10 天花板探针同形）；
    `scored`＝逐轮用产品原语复原**当初被记分的那份历史**（自答是模型自己生成的脏串也算在内）。
    两者不是一回事：oracle 把干扰替模型清掉了，scored 才是"未命中清单"真正发生时的现场。

    `store_mode`：`all`＝按历史如实入库；`target`＝**只留含答案的那条告知**（其余事件用原 cue
    重放，不改写）。`scored`＋`target` 是给"事件选择"定价的那一刀：提示词与记分链逐位相同，
    只有库里的干扰被摘掉——oracle 档把这两件事一起改了，所以它的读数不能当选择的价格。
    """
    from api.seed_runtime import SeedRuntime, record_told_history

    config = substrate.config
    answer = next((token for token in tokens if token in told), "")
    if not answer:
        return {"id": None, "class": "no_copyable_answer"}
    answer_bytes = answer.encode("utf-8")

    #: 入库只走一条路径：两个历史构造函数内部都按线上顺序（先入库再生成自答），
    #: 这里不再额外 record 一次自答，否则"轨迹诊断用的样本"与 ceiling/CAP 用的样本不同形。
    if history_mode == "scored":
        if runtime is None:
            raise ValueError("history_mode='scored' 需要 runtime（_answer_raw 走产品原语）")
        history = _scored_history(runtime, turns)
    else:
        history = _oracle_history(substrate, circuit, turns, tokens)
    prompt = SeedRuntime._serialize(turns[-1], history)
    record_told_history(substrate, circuit, history, episode_id="a26:final")
    prompt_bytes = prompt.encode("utf-8")
    kept: int | None = None
    if store_mode == "target":
        kept = _keep_only_target_events(circuit, told.encode("utf-8"))
        if kept == 0:
            #: 目标告知没在库里——继续走只会得到一条空轨迹，把"消融没做成"读成"选择没问题"。
            return {
                "told": told,
                "answer": answer,
                "history_mode": history_mode,
                "store_mode": store_mode,
                "store_contents": [
                    event.content.decode("utf-8", errors="replace")
                    for event in circuit.store.events()
                ],
                "class": "ablation_no_target_event",
                "chain_identical": False,
                "steps": 0,
                "trace": [],
            }

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
    #: 库里有几条告知由历史构造决定：oracle 档恒 1 条，scored 档是真实的那几条
    #: （扩展集 3 轮题的中间轮本身就是一条**无关告知**）。这里只如实记下内容，不断言条数。
    store_contents = [event.content.decode("utf-8", errors="replace") for event in events]
    told_bytes = told.encode("utf-8")
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
        if feature_sink is not None:
            #: 特征只在这一档被取（别的档不付这个代价）。**关键是取在这条轨迹上**：
            #: 轨迹按"模型实际会发的字节"前进（`argmax_byte`），不是按"copy 自己会发什么"——
            #: 后者是另一条链，在它上面算出来的"上限"描述的是一条从没发生过的路径。
            codes_list = [int(code) for code in snap["codes"].tolist()]
            feature_sink.append(
                {
                    "step": k,
                    "f1": [float(value) for value in ctx.tolist()],
                    "codes": codes_list,
                    "targets": [
                        index for index, code in enumerate(codes_list) if code == target_byte
                    ],
                    "prev_byte": int(prev_byte),
                    "copy_aimed": bool(copy_top_byte == target_byte),
                    "emitted": bool(emitted),
                    "event_is_target": bool(bytes(snap["event"].content) == told_bytes),
                }
            )
        emitted_bytes.append(argmax_byte)
        rows.append(
            {
                "step": k,
                "argmax_byte": argmax_byte,
                "target_byte": target_byte,
                "copy_top_byte": copy_top_byte,
                "copy_mass_on_target": round(float(snap["copy_distribution"][target_byte]), 6),
                #: A2.8-2：**目标字节在 copy 分布里排第几**。只有 top-1 的话，
                #: "被挤到第 2 名"（打分器分辨率问题）与"根本排不进去"（信息不在里面）
                #: 会读成同一个 `address_miss`，而这两者的修法完全不同。
                "target_rank_in_copy": int(
                    (
                        snap["copy_distribution"] > float(snap["copy_distribution"][target_byte])
                    ).sum()
                    + 1
                ),
                "copy_top5": _top_bytes(snap["copy_distribution"]),
                "gate_value": round(gate, 4),
                "p_with_copy": round(float(with_copy[target_byte]), 8),
                "p_vocab_only": round(float(vocab_only[target_byte]), 8),
                #: 反事实：把门强行全开（与 §4.1 结构存在性判据同一个幅度），目标字节会不会成为 argmax。
                "forced_open_would_hit": bool(int(forced.argmax()) == target_byte),
                #: 事件归因：`best_match` 这一步挑中的是不是那条含答案的告知。scored 档下
                #: "指错字节"常常其实是"挑错了事件"——键侧与选择侧修法不同，不记就分不开。
                "on_target_event": bytes(snap["event"].content) == told_bytes,
                "emitted": emitted,
            }
        )
        #: **不在命中处停下**：失败题的轨迹要看的是"整串为什么没出来"，
        #: 命中一步就 break 会把后面的偏离步全部藏掉（上一版就是这么写的）。
        prev_byte = argmax_byte
        substrate.observe(argmax_byte, learn=False, readout="predictive", use_memory=False)
    reconstructed = emitted_bytes.decode("utf-8", errors="replace")
    classification = _classify(rows)
    replay: str | None = None
    if history_mode == "scored":
        #: 现场复现自检：把最后一步换成产品那支 `_answer_raw` 完整生成一次（64 字节、折字），
        #: 若这样仍不命中，才说明本探针看到的失败**就是**记分件看到的那次失败。
        from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

        replay = _answer_raw(runtime, turns[-1], history)
    record = {
        "told": told,
        "answer": answer,
        "history_mode": history_mode,
        "store_mode": store_mode,
        "target_events_kept": kept,
        "store_contents": store_contents,
        "steps_on_wrong_event": sum(1 for row in rows if not row["on_target_event"]),
        "reconstructed_answer": reconstructed[:80],
        #: 自检：若重构串真的含答案词，这题就不该出现在失败清单里——出现即说明
        #: 本探针与 ceiling/CAP 两处构造不一致，结论必须回炉而不是接着分诊。
        "contradicts_failure_list": answer in reconstructed,
        "chain_identical": chain_ok,
        "steps": len(rows),
        "class": classification,
        "trace": rows,
    }
    if replay is not None:
        record["scored_replay_answer"] = replay[:80]
        record["reproduces_recorded_miss"] = answer not in replay
        if store_mode == "target":
            #: 先证"不入库的那支解码"与 `_answer_raw` 同链（不摘干扰时两串必须相等），
            #: 否则下面的消融读数是在另一条链上取的。
            same_chain = _product_reply(runtime, substrate, prompt_bytes) == replay
            _keep_only_target_events(circuit, told_bytes)
            ablated = _product_reply(runtime, substrate, prompt_bytes)
            record["ablation_chain_matches_scored"] = same_chain
            record["ablated_replay_answer"] = ablated[:80]
            #: 这一条就是"事件选择"的价格：提示词与库内 cue 都和记分链相同，只把干扰告知摘掉，
            #: 看产品解码能不能把答案发出来。`oracle` 档同时改了提示词，定不出这个价。
            record["ablation_hits"] = answer in ablated
    return record


def _top_bytes(distribution: Any, k: int = 5) -> list[list[Any]]:
    """copy 分布里质量最高的 k 个字节，返回 `[[字节, 质量], ...]`（按质量降序）。

    `torch.topk` 返回的是 **(值, 下标)** 两个张量。第一版写成 `zip(*topk(...))`，
    于是把"值"当成了字节、"下标"当成了质量——所有赢家字节都被记成 `0x00`
    （值域 [0,1] 取整恒为 0）。名次那一列是独立算的没受影响，但这一列必须重跑。
    **教训：成对返回的 API 不要靠位置解包。**
    """
    values, indices = torch.topk(distribution, k=int(k))
    return [[int(byte), round(float(mass), 6)] for byte, mass in zip(indices.tolist(), values.tolist())]


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
    if max(gates) < GATE_EPS and any(forced_hits):
        return "gate_closed"  # 门基本没开，但强行全开就命中 ⇒ 问题在开关的学习
    #: `emission_loses` 必须有**正面证据**：某一步 copy 指对了目标字节，最终 argmax 却不是它。
    #: 上一版把这条写成了兜底分支（"不是纯指错就归发射被压"），于是 3 道只有 1–2 步指对的题
    #: 被标成 `emission_loses`——而它们的指对步全都 argmax==target（p_copy=1.0 且真发出了），
    #: 等于**没有任何"被词汇压住"的证据**。分诊标签把结论带偏过一次，这里按证据收紧。
    suppressed = [row["copy_top_byte"] == row["target_byte"] and not row["emitted"] for row in rows]
    if any(suppressed):
        return "emission_loses"  # copy 指对了却没发出 ⇒ 发射混合被词汇 logits 压住
    if not any(aimed):
        return "address_miss"  # 一步都没指对过（含"门没开且全开也没用"这一子类）
    #: 有指对步、却从没出现"指对而不发出" ⇒ 症结是**走不稳**（逐字节续接时掉出目标串），
    #: 不是发射侧被压。这一类此前被兜底分支错归，是本轮读数里最值得记的一次自我纠正。
    return "continuation_slips"


def _bias_variant_run(circuit: Any, bias: float | None, body):
    """把 `copy_induce_bias` 临时覆写成一个诊断值，跑完**必原值复原**。

    这不是训练，是读数：两个电路的这个参数都正好等于 `max_weight_norm=2.5`
    （＝钳位），所以"前驱约束不够强"到底是机制不行还是幅度被钳住，现在还没人被分开看过。
    极大值（如 10000）在 softmax 里就等价于硬掩码：只有"前驱＝刚发出的字节"的位置留有质量；
    若一步都没有匹配位置，全体同加 0，退回原分数——不会出现死路。
    """
    import torch

    if bias is None:
        #: asis 档＝不覆写。
        return body()
    parameter = circuit._parameters["copy_induce_bias"]
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.fill_(float(bias))
        return body()
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit", default="output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--source",
        choices=("ceiling", "extension"),
        default="ceiling",
        help="ceiling＝§10 那 6 道（选择做对仍失败）；extension＝扩展集里挂电路仍未命中的题",
    )
    parser.add_argument(
        "--history",
        choices=("oracle", "scored"),
        default="oracle",
        help="oracle＝替模型清掉干扰（与 §10 同形）；scored＝复原当初被记分的那份历史",
    )
    parser.add_argument(
        "--store",
        choices=("all", "target"),
        default="all",
        help="all＝按历史如实入库；target＝库里只留含答案的那条告知（配 scored 即选择的价格）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用；0＝全跑）")
    parser.add_argument(
        "--bias-sweep",
        default="asis",
        help="逗号分隔的 copy_induce_bias 诊断覆写档（asis＝不覆写；数值如 100,10000）",
    )
    args = parser.parse_args()
    if args.store == "target" and args.history != "scored":
        #: oracle 档本来就只有目标那一条，"摘干扰"是空操作——跑出来的消融数会把空操作读成结论。
        print(json.dumps({"error": "--store target 只在 --history scored 下有意义"}))
        return 1

    if args.source == "extension":
        ids = failing_item_ids_extension()
    else:
        ids = failing_item_ids()
    if not ids:
        print(
            json.dumps({"error": f"{args.source} 读数里没找到失败题（读数变了就先查仪器，别硬跑）"})
        )
        return 1

    if args.limit > 0:
        ids = ids[: args.limit]

    from score_taiji_r2_copy_strict_cap import copyable_tokens, load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    #: 题面与题号同源取——扩展集的 id 在 CAP 评价集里不存在，混用会静默漏题。
    items = extension_items() if args.source == "extension" else load_items()
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    payload = torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    substrate.copy_circuit.load_payload(payload)

    raw_biases = [part.strip() for part in args.bias_sweep.split(",") if part.strip()]
    #: `asis` ＝不覆写（用电路里训练后的原值，本次两电路都是钳位值 2.5）；其余是浮点覆写值。
    biases: list[float | None] = [None if token == "asis" else float(token) for token in raw_biases]
    bias_labels = list(zip(raw_biases, biases))
    per_variant: dict[str, dict[str, Any]] = {}
    for label, bias in bias_labels:
        per_item = {}
        for item_id in ids:
            item = items[item_id]
            turns = [str(turn) for turn in item["turns"]]
            tokens = list(copyable_tokens(item))
            told = next((turn for turn in turns[:-1] if any(t in turn for t in tokens)), "")
            if not told:
                per_item[item_id] = {"class": "no_copyable_answer"}
                continue

            def _trace(turns=turns, tokens=tokens, told=told):
                record = trace_item(
                    substrate,
                    substrate.copy_circuit,
                    turns=turns,
                    tokens=tokens,
                    told=told,
                    history_mode=args.history,
                    store_mode=args.store,
                    runtime=runtime,
                )
                record["bias"] = bias
                record["bias_label"] = label
                return record

            per_item[item_id] = _bias_variant_run(substrate.copy_circuit, bias, _trace)
            per_item[item_id]["id"] = item_id
            restored = float(substrate.copy_circuit._parameters["copy_induce_bias"].flatten()[0])
            if restored != float(payload["parameters"]["copy_induce_bias"].flatten()[0]):
                #: 诊断覆写没复原＝下一档是在上一档的参数上跑的，整份读数作废。
                raise AssertionError(f"{item_id}: copy_induce_bias 未复原（{restored}）")
        per_variant[label] = per_item
    #: 顶层字段沿用**第一档**（默认 asis），使本件与已入库的 scored 读数直接可比。
    per_item = per_variant[raw_biases[0]]

    counts: dict[str, int] = {}
    for record in per_item.values():
        counts[str(record.get("class"))] = counts.get(str(record.get("class")), 0) + 1
    traced = [record for record in per_item.values() if record.get("trace")]
    #: 链同一性只对"真取了轨迹的题"负责——`no_copyable_answer` 那类没有发射步骤可比，
    #: 把它们算进分母会让探针自己假失败。
    chain_all = bool(traced) and all(bool(record.get("chain_identical")) for record in traced)
    replay_flags = [
        record["reproduces_recorded_miss"]
        for record in per_item.values()
        if "reproduces_recorded_miss" in record
    ]
    #: 消融档的先决条件：不摘干扰时 `_product_reply` 必须与 `_answer_raw` 逐字相同。
    #: 一题都没证到要写 `null` 而不是 `true`（`all(空集)` 是真空真，会把"没验"读成"验过了"）。
    checked = [
        bool(record.get("ablation_chain_matches_scored"))
        for record in per_item.values()
        if "ablation_chain_matches_scored" in record
    ]
    ablation_chain_all: bool | None = sum(checked) == len(checked) if checked else None
    report = {
        "format": "taiji-r2-a26-emission-trace-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §10/§12",
        "source": args.source,
        "history_mode": args.history,
        "store_mode": args.store,
        "ablation_hits": sum(
            1 for record in per_item.values() if record.get("ablation_hits") is True
        ),
        "ablation_chain_matches_scored_all": ablation_chain_all,
        "failing_items": ids,
        "chain_identical_all": chain_all,
        "class_counts": counts,
        #: 每档只改 `copy_induce_bias`（前驱约束强度），其余同链同底同题——
        #: "指对步占比"就是这一刀要看的那个数（键侧在**正确事件内**指对目标字节的比例）。
        "bias_variants": {
            label: {
                "steps": sum(len(r.get("trace", [])) for r in records.values()),
                "aim_correct": sum(
                    1
                    for r in records.values()
                    for row in r.get("trace", [])
                    if row["copy_top_byte"] == row["target_byte"]
                ),
                "steps_emitted": sum(
                    1 for r in records.values() for row in r.get("trace", []) if row["emitted"]
                ),
                "items_fully_emitted": sum(
                    1 for r in records.values() if r.get("class") == "emitted"
                ),
                "class_counts": {
                    str(c): sum(1 for r in records.values() if r.get("class") == str(c))
                    for c in {str(r.get("class")) for r in records.values()}
                },
            }
            for label, records in per_variant.items()
        },
        #: 逐步事件归因：多少步的 `best_match` 落在**别的**告知上（选择侧），多少步落在目标事件内
        #: 却指错位置（键侧）。两者是两笔不同的账，`address_miss` 一个标签混着它们。
        "steps_total": sum(len(record.get("trace", [])) for record in per_item.values()),
        "steps_on_wrong_event": sum(
            int(record.get("steps_on_wrong_event", 0)) for record in per_item.values()
        ),
        #: scored 模式的现场复现率：复现不出"题号里那次未命中"的题，不能拿来定成因占比。
        "reproduces_recorded_miss": {
            "true": sum(1 for flag in replay_flags if flag),
            "false": sum(1 for flag in replay_flags if not flag),
        },
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
                "history_mode": args.history,
                "store_mode": args.store,
                "ablation_hits": report["ablation_hits"],
                "classes": counts,
                "bias_variants": report["bias_variants"],
                "reproduces_recorded_miss": report["reproduces_recorded_miss"],
                "chain_identical_all": chain_all,
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            }
        )
    )
    return 0 if chain_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
