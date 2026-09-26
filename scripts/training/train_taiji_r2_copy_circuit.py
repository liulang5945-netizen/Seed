"""A2.3 召回条件发射训练：让模型自己学会「何时抄、抄哪条」。

预注册：``plans/reference/M5_R2_A2_3_PREREG_20260925.md``（学习规则 §2、语料 §3、
预算与止损 §4 全部照件执行；本文件里可被触发的拒绝路径不是注释）。

**A2.3b 追加（``--protocol chat``）**：训练 episode 改用**产品 chat 序列化格式**——
「问：/答：」壳、多事件入库、历史里混入模型自己上一轮的答复。A2.4 实测这三处分布差
把裸格式 5/36 那档能力压到 chat 的 3/36（判决件 ``taiji_r2_copy_circuit_chat_cap_20260925.json``）。
预注册 ``plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md``。
两臂**共用同一段学习规则实现**（``_train_answer``）⇒ 唯一变量是喂入协议，不是规则。

阶段：
* ``--stage smoke``＝小模型 (64,48) 全新随机权重，预算 ≤20 分钟，判据＝寻址 top-1 ≥50%
  且 mean|gate| > 0.05；
* ``--stage judge``＝base_16M（seed_beta.pt，只读，跑后 sha256 复核），预算 ≤90 分钟。

纪律：S0 checkpoint 存取自检先行（训练前必查）；写靶只在 ``--out-dir``；stop 文件优雅
停机；判决读数落 ``reports/taiji_r2_copy_circuit_training[_smoke]_<date>.json``，不覆写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

PREREG = "plans/reference/M5_R2_A2_3_PREREG_20260925.md"
PREREG_BY_PROTOCOL = {
    "bare": PREREG,
    "chat": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md",
}
#: 对齐臂里"模型自己上一轮的答复"的生成上限——与 A2.4 计分器 `MAX_ANSWER_BYTES` 同值，
#: 换协议不换生成口径（两臂/两仪器读的是同一个产品形态）。
MAX_REPLY_BYTES = 64

# 训练实体表：与 CAP D/E 评测实体（阿岩/杭州/17/Seed）严格不相交（预注册 §3 冻结）。
NAMES = ("阿蒙", "晓雨", "子昂", "青禾", "若谷", "拾一", "云舟", "浣溪")
CITIES = ("成都", "厦门", "敦煌", "丽江", "曲阜", "平遥", "荔波", "额济纳")
NUMBERS = ("42", "7", "2025", "13", "96", "518", "2048")
PROJECTS = ("星舟", "春潮", "山丘", "拾光", "浮梁", "长庚")
DISTRACTORS = ("水沸点是多少？", "3 乘 4 等于几？", "天空为什么是蓝的？")


def make_episode(rng: random.Random) -> tuple[list[str], str]:
    """一条「告知 →（干扰?）提问」episode 与答案关键词（预注册 §3 模板族）。"""
    base = rng.randrange(4)
    key = (rng.choice(NAMES), rng.choice(CITIES), rng.choice(NUMBERS), rng.choice(PROJECTS))[base]
    tells = (f"我叫{key}。", f"我住在{key}。", f"我最喜欢的数字是{key}。", f"我的项目叫{key}。")
    asks = ("我的名字是什么？", "我住哪？", "那个数字是多少？", "我的项目叫什么？")
    if rng.random() < 0.4:
        return [tells[base], rng.choice(DISTRACTORS), asks[base]], key
    return [tells[base], asks[base]], key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feed(substrate: Any, data: bytes) -> Any:
    """按产品 `generate(reset=False)` 的喂法铺一段字节，返回末位概率。

    产品侧喂法＝`model.py:2953-2969`（边界符 + 逐字节 `observe(learn=False,
    readout="predictive", use_memory=False)`；`use_identity` 在 `use_memory=False` 下恒不生效）。
    两臂共用这一个函数 ⇒ "怎么喂"这一维在臂间没有第二套实现。
    """
    step = substrate.observe(
        int(substrate.config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    for symbol in data:
        step = substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
    return step.probabilities


def _reply_clean(raw: bytes) -> str:
    """把基底原始答复折成产品 chat 用的字符串（与 `chat()` 的截断口径一致）。"""
    from api.seed_runtime import _TURN_MARKERS

    answer = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return answer.strip()


def _train_answer(
    substrate: Any,
    circuit: Any,
    config: Any,
    *,
    tell_bytes: bytes,
    answer: str,
    prev_byte: int,
    lr_address: float,
    lr_gate: float,
    win: dict[str, Any],
    lr_embed: float = 0.0,
) -> None:
    """预注册 §2 的学习规则本体——**两协议臂共用同一实现**。

    A2.3b 的唯一变量是"怎么喂"，不是"怎么学"：把规则收进这一个函数，
    格式对齐臂与冻结裸格式臂的读数差才能全部归因给协议。

    `win` 是窗口累加器（hits/steps/gate/adv/wrong_event）。**rev4 标签纪律在此加码**：
    多事件库里若寻址抢到了别条告知（内容里没有这个答案），该 episode **不更新**并计数——
    事件选择是未训练的余弦 top-1，不是本通道可学参数，喂错标签会把寻址又带偏一次。
    """
    answer_bytes = answer.encode("utf-8")
    span = tell_bytes.find(answer_bytes)
    if span < 0:
        return
    for k, byte in enumerate(answer_bytes):
        state = substrate._state
        cue = substrate.fabric.cortical_context(state.regions)
        ctx = state.motor_context
        snap = circuit.addressing(
            cue=cue.detach().cpu().clone(), f1_context=ctx, prev_byte=prev_byte
        )
        if snap is None:
            return
        if bytes(snap["event"].content).find(answer_bytes) < 0:
            win["wrong_event"] += 1
            return
        top1 = int(snap["scores"].argmax())
        want = span + k
        win["hits"] += int(top1 == want)
        win["steps"] += 1
        base_evidence = float(
            config.consolidation_read_gain
        ) * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
        readout = substrate.predictive_readout
        # rev2（预注册 §2）：反事实固定为"全开"——gate 零初始化下"当前开度"基线恒零锁死。
        p_full = readout.probabilities(
            ctx,
            episodic_evidence=base_evidence + snap["copy_distribution"],
        )
        p_without = readout.probabilities(ctx, episodic_evidence=base_evidence)
        advantage = math.log(max(float(p_full[byte]), 1e-12)) - math.log(
            max(float(p_without[byte]), 1e-12)
        )
        advantage = max(-2.0, min(2.0, advantage))
        circuit.learn(
            snap,
            f1_context=ctx,
            target_position=want,
            advantage=advantage,
            lr_address=lr_address,
            lr_gate=lr_gate,
        )
        win["gate"].append(
            abs(
                circuit.addressing(
                    cue=cue.detach().cpu().clone(), f1_context=ctx, prev_byte=prev_byte
                )["gate_value"]
            )
        )
        win["adv"].append(advantage)
        substrate.observe(int(byte), learn=False, readout="predictive", use_memory=False)
        prev_byte = int(byte)


def _train_selection(
    substrate: Any,
    circuit: Any,
    feed: EpisodeFeed,
    *,
    answer: str,
    lr_selector: float,
    win: dict[str, Any],
) -> None:
    """A2.5 §2/§8：提问条件化选择头的更新，并把这条告知**锁到本轮结束**。

    锁定点与产品 `Taiji.generate` 逐字一致（提问喂完那一刻的皮质 cue＋运动语境＋prompt 全文），
    所以训练看到的选择就是发射用的那一条——`_train_answer` 里每步的 `addressing()` 因此不再
    逐步换告知（§1.2 的漂移是 32.2% 的来源）。

    标签纪律：标签＝库里**逐字含答案词**的那条告知（题面派生，非模型自答），且必须恰好一条；
    0 条或 ≥2 条一律跳过并计数——把歧义写进目标＝§13 那种分布差的复发。
    已经挑对时不更新（advantage=0），梯度只留给真错例。
    """
    state = circuit.lock_selection(
        cue=substrate.cortical_cue(),
        f1_context=substrate._state.motor_context,
        query_bytes=feed.prompt_bytes,
    )
    win["sel_episodes"] += 1
    events = circuit.store.events()
    if len(events) > 1:
        win["sel_multi_event"] += 1
    if state is None:
        win["sel_unlabelable"] += 1
        return
    answer_bytes = answer.encode("utf-8")
    labelled = [event for event in events if answer_bytes in bytes(event.content)]
    if len(labelled) != 1:
        win["sel_unlabelable"] += 1
        return
    if bytes(labelled[0].content) != feed.tell_bytes:
        #: §5 的"标签来自题面"断言：题面母本必须就是库里那条；不等＝入库与标签分家了。
        raise ValueError(
            "selector label diverged from the episode tell: "
            f"{labelled[0].content!r} vs {feed.tell_bytes!r}"
        )
    target = int(labelled[0].event_id)
    picked = int(state["event"].event_id)
    win["sel_picked_correct"] += int(picked == target)
    if len(events) > 1:
        win["sel_picked_correct_multi"] += int(picked == target)
    if picked == target:
        return
    if lr_selector <= 0.0:
        #: 屏幕档：只锁、只测，不学——用来量"未训时这条链上多事件挑对率"的对照。
        #: （`lr_selector=0` 不能整段跳过，否则连锁都不上，测的就不是同一条链了。）
        return
    win["sel_updates"] += 1
    circuit.learn_selection(state, target_event_id=target, advantage=1.0, lr_selector=lr_selector)


def _fresh_window() -> dict[str, Any]:
    """滚动窗口累加器——**一处定义**。初始化与重置两处各写一遍键名，漏一个就是一半读数在骗人。"""
    return {
        "hits": 0,
        "steps": 0,
        "gate": [],
        "adv": [],
        "wrong_event": 0,
        "sel_episodes": 0,
        "sel_multi_event": 0,
        "sel_picked_correct": 0,
        "sel_picked_correct_multi": 0,
        "sel_updates": 0,
        "sel_unlabelable": 0,
    }


@dataclass(frozen=True)
class EpisodeFeed:
    """一条 episode 铺到「答案第一个字节之前」的状态摘要（两臂同形）。

    `prompt_probs`＝训练侧喂完 prompt 的末位概率；守卫测试拿它与产品
    `generate(prompt, 1)` 的实际出字对比，钉「训练喂的那条链＝产品生成那条链」。
    """

    prev_byte: int
    tell_bytes: bytes
    prompt_bytes: bytes
    history: tuple[tuple[str, str], ...]
    prompt_probs: Any


def _run_bare_episode(substrate: Any, circuit: Any, turns: list[str]) -> EpisodeFeed:
    """A2.3 冻结体制（rev4）：裸文本告知→裸文本提问，库已由调用方清空（单事件）。"""
    tell_bytes = turns[0].encode("utf-8")
    _feed(substrate, tell_bytes)
    circuit.store.record(
        tell_bytes,
        substrate.fabric.cortical_context(substrate._state.regions).detach().cpu().clone(),
    )
    ask_bytes = "".join(turns[1:]).encode("utf-8")
    probs = _feed(substrate, ask_bytes)
    return EpisodeFeed(
        prev_byte=int(ask_bytes[-1]),
        tell_bytes=tell_bytes,
        prompt_bytes=ask_bytes,
        history=(),
        prompt_probs=probs,
    )


def _run_chat_episode(
    substrate: Any, circuit: Any, turns: list[str], *, episode_id: str
) -> EpisodeFeed:
    """A2.3b 对齐体制：整条 episode 走产品 chat 协议（`_serialize` ＋ `record_told_history`）。

    与 A2.4 实测的三重分布差逐条对应：① 「问：/答：」壳（改变入库 cue 与查询特征）；
    ② 历史里混入**模型自己上一轮的答复**（先贪心生成再入历史）；③ 多事件库（提问轮不入库）。
    """
    from api.seed_runtime import SeedRuntime, record_told_history

    history: list[tuple[str, str]] = []
    for index, turn in enumerate(turns[:-1]):
        #: 入库在前、生成在后——与 `chat()` 的接线顺序一致（本轮答复不入库）。
        record_told_history(substrate, circuit, history, episode_id=f"{episode_id}:told{index}")
        prompt = SeedRuntime._serialize(turn, history)
        reply = _reply_clean(
            substrate.generate(
                prompt.encode("utf-8"), MAX_REPLY_BYTES, stop_at_boundary=True, sample=False
            )
        )
        history.append((turn, reply))
    record_told_history(
        substrate, circuit, history, episode_id=f"{episode_id}:told{len(turns) - 1}"
    )
    prompt = SeedRuntime._serialize(turns[-1], history)
    prompt_bytes = prompt.encode("utf-8")
    #: 产品侧 `generate_input` 默认 `reset=True`（`model.py:2951`）——入库预喂留下的动力学态
    #: 不进生成；训练臂必须同样在喂 prompt 前 reset，否则学到的是另一条链。
    substrate.reset_dynamics(episode_id="generation")
    probs = _feed(substrate, prompt_bytes)
    return EpisodeFeed(
        prev_byte=int(prompt_bytes[-1]),
        #: 标签母本＝真正那条告知的**入库字节**（不是提问轮文本）。
        tell_bytes=turns[0].encode("utf-8"),
        prompt_bytes=prompt_bytes,
        history=tuple(history),
        prompt_probs=probs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "judge"), required=True)
    parser.add_argument(
        "--protocol",
        choices=("bare", "chat"),
        default="bare",
        help="bare＝A2.3 冻结裸文本臂；chat＝A2.3b 产品协议对齐臂（唯一变量＝怎么喂）",
    )
    parser.add_argument(
        "--max-minutes", type=float, default=None, help="默认按预注册：smoke 20 / judge 90"
    )
    parser.add_argument(
        "--episodes", type=int, default=100_000, help="episode 硬上限（预算双保险）"
    )
    parser.add_argument("--lr-address", type=float, default=0.05)
    parser.add_argument("--lr-gate", type=float, default=0.02)
    #: A2.5 §2：选择头与 `lr_address` 同档。`bare` 臂**强制为 0**——它的库每 episode 就一条
    #: 告知（无可学之物），且 A2.3 冻结臂的参数 digest 不许被顺带改动。
    parser.add_argument("--lr-selector", type=float, default=0.05)
    #: SPEC-A-23（丁 臂）：把 `content_embed` 交给训练。默认 **0**＝现状（固定随机基、永不更新），
    #: 所以这个参数存在本身不改变任何既有跑法——`learn()` 里那条分支一次都不会走。
    parser.add_argument("--lr-embed", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260925)
    #: 电路**初始化**种子（与 `--seed` 分开）。`--seed` 只喂语料取样流，
    #: 而"门的第二次独立取数"要的是两个独立电路 ⇒ 必须能单独换投影初始化（SPEC-A-17 §8）。
    parser.add_argument("--circuit-seed", type=int, default=None, help="None＝沿用基座 config.seed")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="默认按协议分靶：bare＝output/taiji_r2_copy_circuit（冻结件所在），"
        "chat＝output/taiji_r2_copy_circuit_chat（不覆写 A2.3/A2.4 已落地电路）",
    )
    parser.add_argument("--base-checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="读数落盘目录；自检/回归跑指到仓库外，别往判决件目录里堆临时件",
    )
    parser.add_argument("--resume", default=None, help="circuit payload .pt 续跑")
    parser.add_argument("--stop-file", default=None)
    args = parser.parse_args()

    prereg = PREREG_BY_PROTOCOL[args.protocol]
    out_dir_name = (
        args.out_dir
        if args.out_dir is not None
        else (
            "output/taiji_r2_copy_circuit"
            if args.protocol == "bare"
            else "output/taiji_r2_copy_circuit_chat"
        )
    )
    max_minutes = (
        args.max_minutes
        if args.max_minutes is not None
        else (20.0 if args.stage == "smoke" else 90.0)
    )
    out_dir = PROJECT_ROOT / out_dir_name / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = PROJECT_ROOT / args.stop_file if args.stop_file else out_dir / "STOP"
    base_path = PROJECT_ROOT / args.base_checkpoint
    base_sha = _sha256(base_path) if args.stage == "judge" and base_path.exists() else None

    from taiji import Taiji, TaijiConfig

    if args.stage == "smoke":
        substrate = Taiji(
            TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=args.seed)
        )
    else:
        from api.seed_runtime import SeedRuntime

        substrate = SeedRuntime.load(base_path).model.substrate

    # ---- S0：checkpoint 存取自检（训练前必查，预注册 §4）----
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4, init_seed=args.circuit_seed)
    circuit = substrate.copy_circuit
    assert circuit is not None
    circuit.store.record(b"\xe6\xb5\x8b", torch.ones(substrate.config.cortical_context_dim))
    probe_digest = hashlib.sha256(
        json.dumps(
            {k: v.tolist() for k, v in circuit.parameters().items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    probe_payload = circuit.to_payload()
    twin = Taiji(substrate.config)
    twin.mount_copy_circuit(max_events=4, init_seed=args.circuit_seed)
    twin.copy_circuit.load_payload(probe_payload)
    twin_digest = hashlib.sha256(
        json.dumps(
            {k: v.tolist() for k, v in twin.copy_circuit.parameters().items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    if probe_digest != twin_digest:
        raise RuntimeError("S0 self-check failed: circuit checkpoint round-trip digest mismatch")
    circuit.store.clear()  # 自检事件不进训练

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        circuit.load_payload(torch.load(resume_path, weights_only=False)["copy_circuit"])
        print(json.dumps({"resumed": str(resume_path)}), flush=True)

    rng = random.Random(args.seed)
    config = substrate.config
    lr_selector = 0.0 if args.protocol == "bare" else float(args.lr_selector)
    started = time.monotonic()
    done = 0
    win: dict[str, Any] = _fresh_window()
    misaddressed_episodes = 0
    #: 全程累加（不按 50 窗）：窗内取差值累到总量，重置窗口时不会漏掉尾巴那一截。
    selector_updates_total = 0
    selector_unlabelable_total = 0
    history: list[dict[str, Any]] = []
    hit_rates: list[float] = []
    gate_means: list[float] = []
    last_checkpoint_at = 0

    def checkpoint(tag: str) -> None:
        payload = {
            "copy_circuit": circuit.to_payload(),
            "stage": args.stage,
            "protocol": args.protocol,
            "episodes_done": done,
            "prereg": prereg,
        }
        torch.save(payload, out_dir / f"circuit-{tag}.pt")

    while done < args.episodes:
        if (time.monotonic() - started) / 60.0 >= max_minutes:
            break
        if stop_file.exists():
            break
        turns, answer = make_episode(rng)
        substrate.reset_dynamics(episode_id=f"a23-{args.stage}-{done}")
        if args.protocol == "chat":
            # A2.3b：序列化与入库全走产品原语；store 由 `record_told_history` 自清（多事件体制）。
            feed = _run_chat_episode(
                substrate, circuit, turns, episode_id=f"a23b-{args.stage}-{done}"
            )
        else:
            # 单事件训练体制（A2.3 rev4）：每 episode 清库，best_match 必中本条。
            circuit.store.clear()
            feed = _run_bare_episode(substrate, circuit, turns)
        if args.protocol == "chat":
            #: 必须排在 `_train_answer` **之前**：这个函数同时负责锁事件。锁了之后
            #: 每步寻址快照用的才是发射真正用的那条告知（否则选择头与寻址各学各的链）。
            #: 条件按**协议**而不是按 `lr_selector`——`--lr-selector 0` 是"锁而不学"的屏幕档，
            #: 用它当对照时链路必须与训练档完全同形。
            updates_before = win["sel_updates"]
            unlabelable_before = win["sel_unlabelable"]
            _train_selection(
                substrate,
                circuit,
                feed,
                answer=answer,
                lr_selector=lr_selector,
                win=win,
            )
            selector_updates_total += win["sel_updates"] - updates_before
            selector_unlabelable_total += win["sel_unlabelable"] - unlabelable_before
        wrong_before = win["wrong_event"]
        _train_answer(
            substrate,
            circuit,
            config,
            tell_bytes=feed.tell_bytes,
            answer=answer,
            prev_byte=feed.prev_byte,
            lr_address=args.lr_address,
            lr_gate=args.lr_gate,
            win=win,
            lr_embed=args.lr_embed,
        )
        misaddressed_episodes += int(win["wrong_event"] > wrong_before)
        done += 1
        if done % 50 == 0:
            hit_rate = win["hits"] / max(win["steps"], 1)
            gate_mean = sum(win["gate"]) / max(len(win["gate"]), 1)
            history.append(
                {
                    "episodes": done,
                    "minutes": round((time.monotonic() - started) / 60.0, 2),
                    "address_top1_recent50": round(hit_rate, 4),
                    "gate_abs_mean_recent50": round(gate_mean, 4),
                    "advantage_mean_recent50": round(sum(win["adv"]) / max(len(win["adv"]), 1), 4),
                    "misaddressed_episodes_recent50": win["wrong_event"],
                    #: 选择侧**必须分开报**：多数 episode 库里只有一条告知，挑中是必然的；
                    #: 混在一起报会把"无选择可做"摊进"选择成功"（§9 摊平总分那个老坑）。
                    "selection_pick_recent50": round(
                        win["sel_picked_correct"] / max(win["sel_episodes"], 1), 4
                    ),
                    "selection_pick_multi_event_recent50": round(
                        win["sel_picked_correct_multi"] / max(win["sel_multi_event"], 1), 4
                    ),
                    "selection_multi_event_recent50": win["sel_multi_event"],
                    "selection_updates_recent50": win["sel_updates"],
                    "selection_unlabelable_recent50": win["sel_unlabelable"],
                    "selection_lock_dropped": circuit.selection_lock_dropped,
                }
            )
            hit_rates.append(hit_rate)
            gate_means.append(gate_mean)
            print(json.dumps(history[-1]), flush=True)
            win = _fresh_window()
        if done - last_checkpoint_at >= args.checkpoint_every:
            checkpoint(f"ep{done}")
            last_checkpoint_at = done

    checkpoint("final")
    minutes = (time.monotonic() - started) / 60.0
    verdict_pass = bool(hit_rates and hit_rates[-1] >= 0.5 and gate_means and gate_means[-1] > 0.05)
    report = {
        "format": "taiji-r2-copy-circuit-training-v2",
        "prereg": prereg,
        "stage": args.stage,
        "protocol": args.protocol,
        "max_reply_bytes": MAX_REPLY_BYTES if args.protocol == "chat" else None,
        "episodes_done": done,
        "misaddressed_episodes": misaddressed_episodes,
        "minutes": round(minutes, 2),
        "budget_minutes": max_minutes,
        "stopped_by": (
            "budget"
            if minutes >= max_minutes * 0.98
            else (
                "stop-file"
                if stop_file.exists()
                else "episode-cap" if done >= args.episodes else "completed-loop"
            )
        ),
        "lr_address": args.lr_address,
        "lr_gate": args.lr_gate,
        #: A2.5 §2/§8 的选择头账。`§3` 没给选择器单独设 smoke 判据（只有止损时长），
        #: 所以这里**不新增通过/失败线**，只把"未训时的第一窗"与"最后一窗"成对报出——
        #: 抬升与否是一眼可读的事实，判改善仍按 §3（表层严格命中、两电路同向、差 ≥3）。
        "selector": {
            "lr_selector": lr_selector,
            "protocol": args.protocol,
            "lock_dropped": int(circuit.selection_lock_dropped),
            "head_abs_max": round(
                max(
                    float(circuit.parameters()["selector_weight"].abs().max()),
                    float(circuit.parameters()["selector_bias"].abs().max()),
                ),
                6,
            ),
            "multi_event_rate_first_window": (
                history[0].get("selection_multi_event_recent50") if history else None
            ),
            "pick_correct_first_window": (
                history[0].get("selection_pick_multi_event_recent50") if history else None
            ),
            "pick_correct_last_window": (
                history[-1].get("selection_pick_multi_event_recent50") if history else None
            ),
            "updates_total": selector_updates_total,
            "unlabelable_total": selector_unlabelable_total,
        },
        #: SPEC-A-23 丁 臂的账。`content_embed` 一旦可训，它**同时**改动寻址键、`pooled`
        #: 与选择头特征 2 ⇒ 任何读数都不许单因归给"表征"（本件的 §4 限定），
        #: 这里只留一个可核对的位移量：Frobenius 范数与"动过多少行"。
        "representation": {
            "lr_embed": float(args.lr_embed),
            "content_embed_frobenius_norm": round(
                float(circuit.parameters()["content_embed"].norm()), 4
            ),
            "content_embed_rows_touched": int(
                (circuit.parameters()["content_embed"].abs().amax(dim=1) > 0.0).sum()
            ),
        },
        "seed": args.seed,
        "circuit_seed": args.circuit_seed,
        "base_checkpoint": str(base_path.relative_to(PROJECT_ROOT)) if base_sha else None,
        "history": history,
        "smoke_criteria": {
            "address_top1_ge_0.5": bool(hit_rates and hit_rates[-1] >= 0.5),
            "gate_abs_mean_gt_0.05": bool(gate_means and gate_means[-1] > 0.05),
            "pass": verdict_pass,
        },
    }
    if base_sha is not None:
        report["base_sha256_unchanged"] = _sha256(base_path) == base_sha
    stem = (
        f"taiji_r2_copy_circuit_{'chat_training' if args.protocol == 'chat' else 'training'}"
        f"_{'smoke_' if args.stage == 'smoke' else ''}20260925"
    )
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = PROJECT_ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{stem}.json"
    if out.exists():
        out = out.with_name(out.stem + f"-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "criteria_pass": verdict_pass}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
