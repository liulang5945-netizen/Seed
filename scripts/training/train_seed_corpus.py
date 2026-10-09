"""Train Seed on the raw-byte stream of the simple_zh corpus.

阶段 1 原生数据管线：复用 ``data/simple_zh/`` 既有语料（当前 canonical 为
dialogue_extended_clean），以 raw-byte 流喂入 ``Seed.observe``；
会话边界用 ``boundary_symbol``，对话结构沿用语料里的文本标记（问：/答：），
全程不引入 tokenizer。训练循环为分片流式多 epoch + 周期 ``checkpoint()``
落盘（seed-native-v1 信封），进度曲线逐条写入 ``reports/``。

用法（默认小预算冒烟）::

    python scripts/training/train_seed_corpus.py --smoke

正式训练（放大画像、限量符号数、周期落盘）::

    python scripts/training/train_seed_corpus.py \
        --parameter-budget 500000 --device auto \
        --epochs 1 --max-symbols 400000 \
        --checkpoint checkpoints/seed_corpus.pt \
        --progress reports/seed_corpus_progress.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, SeedConfig, iter_native_documents  # noqa: E402
from seed.persistence import (  # noqa: E402
    atomic_save,
    attach_metadata,
    corpus_fingerprint,
)
from taiji import AdaptiveResidualGrowthPolicy, CapacityPolicy, TaijiConfig  # noqa: E402
from taiji.contracts import EpisodicMemoryRecord  # noqa: E402
from taiji.episodic_memory import EpisodicMemoryStore  # noqa: E402

DEFAULT_CORPUS = (
    # 2026-08-23 数据整理：canonical 对话语料仅此一文件（123090 条，
    # 已吸收 alpaca-zh/shared_core 内容）；旧声明中的 alpaca_zh_sft_clean /
    # class_a_chinese 已删除。
    "data/simple_zh/dialogue_extended_clean.jsonl",
)

# Fixed unseen probe: window statistics over the moving stream measure content
# difficulty, not model quality, so every progress entry also scores this
# constant byte string.  A monotone training run must drive its surprise down.
HOLDOUT_PROBE = (
    "水的沸点在标准大气压下是一百摄氏度。"
    "问：你好。\n答：你好，很高兴见到你。"
    "请解释一下牛顿第二定律和它的日常应用。"
).encode()

#: PLAN-N3-05（DEBT-G49 修法①）：**加新列、旧常量与旧列一字不动**。
#: 上面那枚旧探针被量出与训练语料有 5/35 个 24 字节窗口原样重合（hit_rate=0.142857，
#: `reports/taiji_n3a_data_face_scale10calib_20261008.json`）⇒ 它的名字"holdout"名不副实；
#: 但删掉它就没有人能复算历史上所有引用 `holdout_surprise` 的结论，所以留在原地。
#: 这枚新候选由 `verify_taiji_n3_05_probe.py` 按冻结的验收式选出来（同机两趟 0 命中、
#: 形状同类、字节数在 64..512 档内且是过验者里最接近旧探针的那一枚；验证件
#: `reports/taiji_n3_05_probe_verification_20261008.json`）。**文本是为零重合而新写的，不是从任何语料摘的**。
#: 两条列**不可跨列比大小**：新探针 271 B 对旧 163 B，长度不同⇒`mean_surprise` 的水平天然不同档，
#: 只能各自看同列随训练的变化方向。
HOLDOUT_PROBE_V2 = (
    "潮汐发电站把月球引力引起的潮位差转换成电能，一年里两次大潮的水量差别最明显。"
    "问：请把这句话缩短一点。\n答：好的：月亮拉扯海水，电站借这股力发电。"
    "请说明为什么铜适合做导线而玻璃不适合。"
).encode()


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a requested training device without silently ignoring CUDA."""

    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this PyTorch build or machine has no available CUDA device"
        )
    return device


def load_capacity_policy(path: Path | str) -> CapacityPolicy:
    """Load an explicit structural search policy from JSON."""

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("capacity policy must be a JSON object")
    return CapacityPolicy.from_dict(payload)


class DocumentStreamCounters:
    """PLAN-N3-10 §2 的三轴自述载体：唯一篇数 `U`、到达篇次、平均重复率 `R`。

    计数器由**取数侧自己数**（不是配置值回显）——本件的全部纪律是"重复率不许当隐藏变量"，
    照抄 `K` 就等于把要报的那一轴又变回人抄的数。
    """

    def __init__(self) -> None:
        self.unique_documents = 0
        self.document_visits = 0
        #: PLAN-N3-02 §4quater 的"序列长度"一列：由**发符处现数**，一篇一条（含边界符）。
        #: 存整列而不只存 min/max，是因为中位数与均值要能对回同一批数——只报摘要就再也验不了。
        self.document_symbol_counts: list[int] = []

    def record_document(self, symbol_count: int) -> None:
        self.document_symbol_counts.append(int(symbol_count))

    def sequence_length_stats(self) -> dict[str, float | int] | None:
        """§8.7 的"序列长度"＝**本轮进入过的每一篇**的符号数分布；一篇都没进就出版 `None`。

        取法冻结在 PLAN-N3-02 §4quater 那一行："由 `iter_corpus_symbols` 在边界符处累加得到，
        不许用 `max_symbols` 参数值冒充实测分布"。计数点在**进篇那一刻**（长度是这篇自身的
        字节数，与消费到哪一步无关），所以被符号预算切断的最后一篇按**它自身的长度**计入
        ⇒ `documents_counted` 与 `document_visits` 同数；"吃到第几符号"另有 `ticks_at_exit`。
        空档不许印 0（DEBT-G63 那条同族纪律）。
        """

        counts = self.document_symbol_counts
        if not counts:
            return None
        return {
            "documents_counted": len(counts),
            "min": min(counts),
            "median": float(statistics.median(counts)),
            "max": max(counts),
            "mean": round(sum(counts) / len(counts), 6),
        }

    def as_dict(self) -> dict[str, float | int]:
        mean = (
            round(self.document_visits / self.unique_documents, 6) if self.unique_documents else 0.0
        )
        return {
            "unique_documents": int(self.unique_documents),
            "document_visits": int(self.document_visits),
            "mean_revisits": mean,
        }


def _window_readouts(
    window_correct: int, window_surprise: float, window_ticks: int
) -> tuple[float | None, float | None]:
    """DEBT-G63：窗口计数器的除零护栏——**零窗口的行不许出版数值**。

    旧写法 `x / max(1, window_ticks)` 在收尾行（窗口刚被上一次 flush 归零）会印出
    `online_accuracy=0.0`／`mean_surprise=0.0`，与"这一档精度为零"完全同形；
    实测甲臂 250,000 符号那档的 `progress_exit.json` 就是这样（真数只有 holdout 两列）。
    """

    if int(window_ticks) <= 0:
        return None, None
    return float(window_correct) / float(window_ticks), float(window_surprise) / float(window_ticks)


def iter_corpus_symbols(
    paths: Sequence[Path | str],
    *,
    boundary: int = TaijiConfig().boundary_symbol,
    end_boundary_after_newline: bool = False,
    max_unique_documents: int | None = None,
    counters: DocumentStreamCounters | None = None,
) -> Iterator[int]:
    """Stream every corpus document as boundary-separated raw UTF-8 bytes.

    Every jsonl row contributes one session: one ``boundary_symbol`` followed by
    the document's UTF-8 bytes.  The dialogue structure already lives in the
    text (问：/答： markers), so no tokenizer and no structural re-encoding is
    needed -- the model sees exactly the bytes a reader would see.

    ``end_boundary_after_newline``（A30 §2aa 目标编码对齐，PLAN-A-30 §2z/§2aa）：
    每篇正文之后再补一个换行 ``0x0A``，让"本篇结束"的边界符号落在**模型已经会预测
    换行的那个位置之后**（§2z：真结束位第一位 66/120=55% 是 ``\\n``）。§2aa/决策级
    两档（n=30：26/30；出厂基座 n=300：263/300）证明这一落点让边界符在结束位上
    从"从不胜出"变为绝大多数胜出。**函数级默认 False**（archive 仪器与旧配方复现
    逐位不变）；主线配方在 CLI 层默认开（2026-09-29 owner 条件授权，§7-3）。
    """

    if max_unique_documents is None:
        #: 默认支＝今天的形状，一字不动；只多挂了计数器（只读地数篇次）。
        for text in iter_native_documents(paths):
            encoded = text.encode("utf-8")
            if counters is not None:
                counters.unique_documents += 1
                counters.document_visits += 1
                counters.record_document(
                    1 + len(encoded) + (1 if end_boundary_after_newline else 0)
                )
            yield boundary
            yield from encoded
            if end_boundary_after_newline:
                yield 0x0A
        return

    #: 形状 B（PLAN-N3-10 §2）：只取前 K 篇成池，之后**循环重用**——这是个无界生成器，
    #: 停手由调用方的符号预算支负责（:655/:677 那两处），所以 `document_visits` 一定 ≥ `K`。
    pool: list[str] = []
    for text in iter_native_documents(paths):
        pool.append(text)
        if len(pool) >= int(max_unique_documents):
            break
    if not pool:
        raise ValueError("给了 --max-unique-documents 但一篇文档都没取到 ⇒ 语料空，响亮拒绝")
    if counters is not None:
        counters.unique_documents = len(pool)
    while True:
        for text in pool:
            encoded = text.encode("utf-8")
            if counters is not None:
                counters.document_visits += 1
                counters.record_document(
                    1 + len(encoded) + (1 if end_boundary_after_newline else 0)
                )
            yield boundary
            yield from encoded
            if end_boundary_after_newline:
                yield 0x0A


def iter_answer_chunks(
    paths: Sequence[Path | str],
    *,
    max_answer_chars: int = 0,
    self_answers: Mapping[str, str] | None = None,
) -> Iterator[bytes]:
    """A30 §2bf：**每答一块**喂法（分块短答形状进主线的正题）。

    与 `iter_corpus_symbols`（连续流）并列的一条喂法：把每篇文档拆成 (问句, 答案)，
    产出 ``问：{q}\\n答：{a}\\n`` 的**单块**；调用方按 `learn_bytes(..., include_start_boundary=True,
    include_end_boundary=True, reset=True)` 逐块喂 ⇒ 每块一轮 episode、起沿/收沿各一个边界符，
    且**收沿边界落在换行之后**（块自身以 ``\\n`` 结尾，`organs._edge_split` 把边界追加在其后）
    ——与 `end_boundary_after_newline` 的落点性质相同（字节级实证见 PLAN-A-30 §2be）。

    ``max_answer_chars > 0`` 时按**字符**截断答案（与 on-policy 仪器的 `sized` 臂同口径：
    ``corpus_answer[:target]``）；截断只动答案，不动问句。拆不开 ``\\n答：`` 的行走响亮失败——
    不许静默少喂或把整行当答案。

    ``self_answers``（A30 §2bg：自写形状档）给了就用表里的答案替换语料答案；表由
    `build_taiji_a30_self_answers.py` 生成（问句→模型自答）。**查不到问句时响亮失败**——
    语料与缓存必须同源同序，缺一条就是配对坏了，不许拿别的答案顶上。

    为什么要有这一条：正式档四臂里 `sized`（语料短答）L2 真自停 47/72、`self` 24/72，
    而同预算的连续流主线配方（a31 满额）只有 3/72 ⇒ "分块短答形状能不能搬进主线配方"
    是 owner 2026-09-30 裁定①要求训一档验证的问题；§2bg 又证 L1 挂回路格随**答案作者**分
    （自写过、语料侧不过）⇒ 自写档是重出基座的候选。
    """

    marker = "\n答："
    for text in iter_native_documents(paths):
        question, sep, answer = text.partition(marker)
        if not sep:
            raise RuntimeError(f"语料行里没有 {marker!r} 这个接缝，拆不出答案：{text[:24]!r}")
        question = question.removeprefix("问：").strip()
        if self_answers is not None:
            if question not in self_answers:
                raise RuntimeError(
                    f"自写答案表里没有这个问句（下表与语料不同源）：{question[:24]!r}"
                )
            answer = self_answers[question]
        else:
            answer = answer.strip()
        if max_answer_chars > 0:
            answer = answer[:max_answer_chars]
        yield f"问：{question}\n答：{answer}\n".encode()


def patch_envelope_config_flags(envelope: dict[str, Any], flags: dict[str, bool]) -> int:
    """把实验开关写进受载档里**每一处** taiji config 副本（PLAN-A-26 的热启动需要它）。

    为什么必须改档而不能只改模型 config：`Taiji.restore` 会拿档里的 config 与架构 config 逐键比，
    不等就抛 `checkpoint configuration does not match architecture`——**那是对的**，
    它挡住"悄悄换配方还用别人的权重"。热启动时我们**故意**翻这几个实验键，
    所以要把档里同名的副本一起翻过来让那次比较通过；**只翻点名的键**、绝不整份替换
    （整份替换会把这道理序关掉）。返回改到的副本数，便于冒烟核对。

    副本数取决于信封版本：`seed-native-v1`（含 `seed_beta.pt` 这种 P1 之前的旧档）只有
    `config.taiji` 与 `substrate.config` **两处**；v10 信封还多一处 `taiji.kernel.config`。

    **已知边界**：v10 档若带身份器官，改 config 会让"身份器官血缘"校验 (`identity organ
    checkpoint lineage does not match Taiji core`) 不通过——血缘把器官绑在它当时的核心上，
    改配方等于换核心，这是**故意**的。本函数不替它重发血缘；PLAN-A-26 的热启动目标是
    `seed_beta`（v8/v1、不带身份器官），不受此限。
    """

    patched = 0

    def _touch(node: Any) -> None:
        nonlocal patched
        if isinstance(node, dict):
            for key, value in flags.items():
                node[key] = bool(value)
            patched += 1

    _touch(envelope.setdefault("config", {}).setdefault("taiji", {}))
    substrate = envelope.get("substrate")
    if isinstance(substrate, dict):
        _touch(substrate.setdefault("config", {}))
    native = envelope.get("taiji")
    if isinstance(native, dict):
        kernel = native.get("kernel")
        if isinstance(kernel, dict):
            _touch(kernel.setdefault("config", {}))
    return patched


def exit_record_path(progress_path: Path | str) -> Path:
    """退出记账件的位置：与进度日志同目录、同词干，**按臂分开**（多台训练共用目录时不许互抢）。

    单独成一个函数是因为有两个读者（`run_training` 写它、`main` 打印它）与测试——
    路径规则只能住一处（一副档只住一处）。
    """

    progress_path = Path(progress_path)
    return progress_path.with_name(f"{progress_path.stem}_exit.json")


def _prune_history(history_dir: Path, cap: int | None) -> int:
    """DEBT-G40 的上限策略：超过 `cap` 就薄中间，**首尾必留**，返回删除数。

    默认 `cap=None` ⇒ 一字不动（现行"每次落盘都留一份"的保号语义逐位不变）。
    """

    if cap is None:
        return 0
    snaps = sorted(history_dir.glob("checkpoint_*.pt"))
    if len(snaps) <= cap:
        return 0
    keep = {0, len(snaps) - 1}
    stride = (len(snaps) - 1) / (cap - 1)
    keep.update(round(step * stride) for step in range(cap))
    doomed = [path for index, path in enumerate(snaps) if index not in keep]
    for path in doomed:
        path.unlink()
    return len(doomed)


def _file_sha256(path: Path) -> str | None:
    """DEBT-G39：退出记账要能自述"我写下的是哪些字节"，而不只是"我往哪个路径写过"。"""

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_training(
    *,
    corpus_paths: Sequence[Path | str],
    config: SeedConfig,
    epochs: int,
    checkpoint_path: Path | str,
    progress_path: Path | str,
    checkpoint_every: int,
    progress_every: int,
    max_symbols: int | None = None,
    resume_checkpoint: Path | str | None = None,
    resume_config_overrides: dict[str, bool] | None = None,
    readout: str = "action",
    device: str | torch.device = "cpu",
    keep_history: Path | str | None = None,
    keep_history_max: int | None = None,
    end_boundary_after_newline: bool = False,
    answer_chunking: str = "stream",
    answer_max_chars: int = 0,
    answer_source: str = "corpus",
    self_answers_path: Path | str | None = None,
    pressure_record: Path | str | None = None,
    developmental_fast_slow: bool = False,
    developmental_bridge_gate: float | None = None,
    growth_minimum_pressure: float | None = None,
    episodic_mount: bool = False,
    n5_shadow: bool = False,
    max_unique_documents: int | None = None,
) -> dict[str, float]:
    """Stream the corpus through ``Seed.observe`` with periodic persistence.

    ``resume_config_overrides`` 只在热启动（``resume_checkpoint`` 非空）时生效：把点名实验键
    写进受载档里的每一处 config 副本，让 `Seed/Taiji.restore` 的"档配比架构"守卫放行。
    为 None 时热启动行为与从前逐位相同（缺键按默认值补齐，不会静默改配方）。

    ``readout`` 选 `"action"`（默认，训 F4/运动解码器）或 `"predictive"`（训 F1 预测读出＋
    私有时间语境，`learn_motor=False`——`observe` 明令预测读出不得训运动器）。**A 支线的
    所有已证部件（位置输入、复制电路、UTF-8 证据门）都挂在 F1 预测读出的链上**，所以在
    `"action"` 档跑 `--readout-position` 是**静默空转**；`main` 对此响亮报错。
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    #: PLAN-N3-10 §2 形状 B 的两条响亮拒绝：`K<=0` 不是"不限"而是错用（不限＝不给这个参数）；
    #: 封顶篇数会造出**无界**的循环流，所以必须同时给符号预算，否则这支生成器永不停手。
    if max_unique_documents is not None and int(max_unique_documents) <= 0:
        raise ValueError("max_unique_documents must be positive (omit it for an uncapped stream)")
    if max_unique_documents is not None and max_symbols is None:
        raise ValueError(
            "max_unique_documents cycles a fixed pool, so it needs max_symbols "
            "to bound the stream"
        )
    stream_counters = DocumentStreamCounters()
    if max_unique_documents is not None and str(answer_chunking) != "stream":
        #: PLAN-N3-10 §1 明写：封顶只作用于连续流那一支取篇处，`per-answer` 的 `iter_answer_chunks`
        #: 不动 ⇒ 两轴一起给会是"以为封了顶其实没封"，只能响亮拒绝。
        raise ValueError(
            f"--max-unique-documents 只作用于连续流（answer_chunking={answer_chunking!r} 走另一支取数）"
        )
    if checkpoint_every <= 0 or progress_every <= 0:
        raise ValueError("checkpoint/progress intervals must be positive")
    if readout not in {"action", "predictive"}:
        raise ValueError("readout must be 'action' or 'predictive'")

    checkpoint_path = Path(checkpoint_path)
    progress_path = Path(progress_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    if keep_history is not None:
        keep_history = Path(keep_history)
        keep_history.mkdir(parents=True, exist_ok=True)
        if keep_history_max is not None and keep_history_max < 2:
            raise ValueError("keep_history_max must be >= 2 — 上限至少留得住「首」与「尾」各一枚")
    elif keep_history_max is not None:
        raise ValueError(
            "keep_history_max 需要 keep_history 同时在用——没有快照可删时设上限是个空承诺"
        )
    #: DEBT-G40：删了几枚快照也要生产者自己说，不能只留"目录里现在有几枚"。
    history_pruned = 0

    model = Seed(config, device=resolve_device(device), episode_id="seed-corpus")
    tick_offset = 0
    if resume_checkpoint is not None:
        envelope = torch.load(resume_checkpoint, weights_only=False)
        if resume_config_overrides:
            patched = patch_envelope_config_flags(envelope, resume_config_overrides)
            # 任何 Seed 信封都至少有**两处** config 副本：`config.taiji`（Seed 层比对的）与
            # `substrate.config`（Taiji 层比对的）。v10 信封还多一处 `taiji.kernel.config`。
            # 少于两处说明信封结构不认识，宁可响亮失败也不要静默少改一处——
            # 少改一处就会在 restore 里撞"档配比架构"，那时很难判断是哪一处。
            if patched < 2:
                raise RuntimeError(
                    "resume envelope is missing expected config copies: "
                    f"patched={patched} (want >=2)"
                )
        model.restore(envelope)
    #: 换读出链必须开新情节：`observe` 明令"情节活跃时不许换读出"（
    #: `readout changed inside an active dynamics episode; reset before switching`），
    #: 而 `seed_beta` 的最后一步是 `action` 档。`reset_dynamics` 只清活动、保留全部学习到的
    #: 突触（与 `train_taiji_langfloor.py` 的起手一致）。**同链续训不 reset** ⇒ 既有 `action`
    #: 档的续训行为逐位不变。
    #: `reset_dynamics` 清的是**情节局部的** `model.tick`（`_development_ticks` 另留累计值），
    #: 所以记下偏移，让进度与保号存档名继续读作"从基底那一步起"（同 langfloor 自持
    #: `absolute_tick` 的做法）。
    if model.snapshot().readout_kind != readout:
        tick_offset = int(model.tick)
        model.reset_dynamics(episode_id="seed-corpus")
    boundary = config.taiji.boundary_symbol
    fingerprint = corpus_fingerprint(corpus_paths)
    #: 绝对刻度（见上）：换读出链时它是"基底 tick + 本次已走步数"，否则等于 `model.tick`。
    ticks = tick_offset + int(model.tick)
    base_ticks = ticks

    def _persist() -> None:
        # 2026-08-23 M0：原子落盘 + 信封元数据，崩溃不产生半写文件。
        if episodic_store is not None and doc_pending:
            cue = (doc_pending + [0] * 32)[:32]
            episodic_store.write(
                EpisodicMemoryRecord(
                    memory_id=f"doc-{episodic_store.count:06d}",
                    episode_id=f"n4-material-{episodic_store.count:06d}",
                    tick=ticks,
                    cue=torch.tensor([b / 255.0 for b in cue], dtype=torch.float32),
                )
            )
            doc_pending.clear()
        envelope = attach_metadata(
            model.checkpoint(),
            tick=ticks,
            corpus_fingerprint=fingerprint,
            extra={
                "trainer": "train_seed_corpus",
                # A30 §2aa：喂入形状（结束边界是否落在换行之后）随档登记——
                # 配方可从档里查出来，不靠外部记录。
                "end_boundary_after_newline": bool(end_boundary_after_newline),
                # A30 §2bf：分块喂法（每答一块、可选短答截断）同样随档登记。
                "answer_chunking": str(answer_chunking),
                "answer_max_chars": int(answer_max_chars),
                "answer_source": str(answer_source),
                "self_answers_path": (
                    str(self_answers_path) if self_answers_path is not None else None
                ),
            },
        )
        atomic_save(envelope, checkpoint_path)
        if adaptive_shadow is not None:
            envelope["n5_shadow"] = {
                "candidate_id": adaptive_shadow.candidate_id,
                "bridge_id": adaptive_shadow.bridge_id,
                "gate": adaptive_shadow.gate,
                "unit_count": adaptive_shadow.unit_count,
            }
        if episodic_store is not None:
            envelope["episodic_memory"] = episodic_store.checkpoint()
        if episodic_store is not None and episodic_store.count:
            base = Path(checkpoint_path)
            records = episodic_store.records
            materials = [
                {"memory_id": record.memory_id, "cue": record.cue.tolist()} for record in records
            ]
            stride = max(1, len(materials) // 100)
            picked = materials[::stride][:100]
            queries = []
            for offset, row in enumerate(picked):
                donor = materials[(offset + 1) % len(materials)]
                cue = list(row["cue"])
                if cue:
                    cue[-1] = donor["cue"][-1]
                queries.append({"expected_memory_id": row["memory_id"], "cue": cue})
            newline = chr(10)
            (base.parent / (base.stem + "_episodic_materials.jsonl")).write_text(
                newline.join(json.dumps(row, ensure_ascii=False) for row in materials) + newline,
                encoding="utf-8",
                newline=newline,
            )
            (base.parent / (base.stem + "_episodic_queries.jsonl")).write_text(
                newline.join(json.dumps(row, ensure_ascii=False) for row in queries) + newline,
                encoding="utf-8",
                newline=newline,
            )
        # 2026-09-23：**保号存档**。此前 `--checkpoint-every` 反复覆盖同一个文件，
        # 于是"训练中途某个 tick 的状态"直接消失——后来才想到要量的指标（例如槽可分离性）
        # 连事后补算都做不到，只剩首尾两个端点。这里每次落盘**额外**写一份带 tick 的快照；
        # 主路径 `checkpoint_path` 的行为一字不变（兼容既有工具与流程）。
        if keep_history is not None:
            atomic_save(envelope, keep_history / f"checkpoint_{ticks:012d}.pt")
            nonlocal history_pruned
            history_pruned += _prune_history(keep_history, keep_history_max)

    started = time.perf_counter()
    window_ticks = 0
    window_correct = 0
    window_surprise = 0.0

    def _flush(final: bool, exit_reason: str | None = None) -> None:
        if window_ticks <= 0 and not final:
            return
        #: **DEBT-G63**：窗口为零时 `0/max(1,0)` 会把"这一行没有窗口样本"出版成
        #: `online_accuracy=0.0`／`mean_surprise=0.0`，看起来完全像能力读数（甲臂收尾行实测如此）。
        #: 现在窗口为零一律出版 `None`；周期性行不受影响（它们本来就只在 `window_ticks>0` 时写）。
        accuracy, surprise = _window_readouts(window_correct, window_surprise, window_ticks)
        entry = {
            "epoch": epoch,
            "ticks": ticks,
            "window_ticks": window_ticks,
            "online_accuracy": accuracy,
            "mean_surprise": surprise,
            "holdout_surprise": model.score_bytes(HOLDOUT_PROBE)["mean_surprise"],
            #: PLAN-N3-05：与训练语料零窗口重合的第二把尺（旧列取法与值一字不动，见上面的常量注释）。
            "holdout_surprise_v2": model.score_bytes(HOLDOUT_PROBE_V2)["mean_surprise"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        if exit_reason is not None:
            #: DEBT-G14：退出原因与预算达成与否**必须落盘**。此前档里只有 `progress.jsonl` 的
            #: 一条正常记录当收尾（`train_seed_corpus` 的预算支是 `ticks >= base_ticks + max_symbols`），
            #: 于是"这枚件吃满预算没有"只能反向推——实测代价是一枚只吃到 37.5% 预算的件
            #: 被当成"在跑"引用了一整轮（PLAN-A-30 §2ai）。这些键**只出现在收尾那一行**，
            #: 周期性行一字不动（守卫 `test_g14_trainer_exit_accounting` 钉这一条）。
            entry["exit_reason"] = exit_reason
            entry["base_ticks"] = int(base_ticks)
            entry["budget_max_symbols"] = int(max_symbols) if max_symbols is not None else None
            entry["ticks_at_exit"] = int(ticks)
            entry["reached_budget"] = bool(
                max_symbols is not None and ticks >= base_ticks + int(max_symbols)
            )
            #: PLAN-N3-10 §2：三轴自述只挂**收尾那一行与独立 exit 件**（周期性行键集一字不动，
            #: 与 G14 那三条同形）。`mean_revisits` 由这里现数，不许由命令行反推 `K`。
            entry.update(stream_counters.as_dict())
            #: PLAN-N3-02 §8.7 的"序列长度"一列（同一台仪器、同一纪律：现数、只挂收尾行、
            #: 一篇都没喂完时出版 `None` 而不是 0）。判读器 `adjudicate_taiji_n3a_scaling_probe.py:36`
            #: 按 `("exit","sequence_length")` 取它，取不到就把整档锁在 `ran_not_measured`（㊵-543）。
            entry["sequence_length"] = stream_counters.sequence_length_stats()
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        #: 压强面收尾自述：把"这张面实际写了多少观测、收尾时走到第几 tick"钉在同一件里。
        #: 理由不是洁癖——第一版实测：周期性 `_flush` 里的 holdout 探针会让压强支在此后不再产出
        #: 观测（进度行照涨），于是分布只覆盖最前面一小段而**看不出来**（2026-10-08）。
        if final and pressure_record is not None:
            #: 收尾那一次 learning_mode 读数与面首的 mode_readings **合起来**才是在场性证据：
            #: 模式不入档、也可能被任何一次 restore 静默打回 read_only，所以两端都要现读（PLAN-N3-04 §3）。
            with pressure_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        {
                            "kind": "tail",
                            "ticks_at_close": int(ticks),
                            "records_written": int(pressure_lines),
                            "learning_mode_at_close": substrate.developmental_f1_learning_mode,
                            "bridge_gate_at_close": (
                                substrate.adaptive_residual_bridge.gate
                                if substrate.adaptive_residual_bridge is not None
                                else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if exit_reason is not None:
            #: 计量键与进度收尾那一行**同源同值**（同一份 `entry`），所以两者不可能互相矛盾；
            #: 独立件另外多带 `checkpoint_path`／`checkpoint_sha256`／`corpus_fingerprint` 三条**只属于文件**的自述，
            #: 它们不进进度行（`test_periodic_lines_keep_their_old_shape` 钉着那一行的键集）。
            #: `checkpoint_sha256` 之所以能等于终件字节，靠的是三处退出点都先 `_persist()` 再 `_flush(final=True)`。
            exit_record_path(progress_path).write_text(
                json.dumps(
                    {
                        **entry,
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_sha256": _file_sha256(checkpoint_path),
                        "corpus_fingerprint": fingerprint,
                        #: DEBT-G40：数量与字节由生产者现数（不是配置值回显）——没有这两条，
                        #: "这轮保号存档留了多少"只能人事后 du，而无自述的量一定会被估错。
                        "history_files": (
                            len(list(keep_history.glob("checkpoint_*.pt")))
                            if keep_history is not None
                            else None
                        ),
                        "history_bytes": (
                            sum(p.stat().st_size for p in keep_history.glob("checkpoint_*.pt"))
                            if keep_history is not None
                            else None
                        ),
                        "history_pruned": (history_pruned if keep_history is not None else None),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
                newline="\n",
            )

    #: A-4／PLAN-A-26：读出模式决定**哪条链**在学。`action`＝既有行为逐位不变；
    #: `predictive` 把下一字节误差送到 F1 专用读出（＋私有时间语境）——`observe` 明令此时
    #: 不得同时训运动器，所以 `learn_motor=False`。A 支线的已证部件都在这条链上。
    observe_kwargs: dict[str, object] = {"learn": True, "readout": readout}
    episodic_store = None
    doc_pending: list[int] = []
    adaptive_shadow = None
    if episodic_mount:
        episodic_store = EpisodicMemoryStore(capacity=1024)
        model.substrate.attach_episodic_memory(episodic_store)
    if readout == "predictive":
        observe_kwargs["learn_motor"] = False

    #: PLAN-N3-01 步骤一（乙线的只读压强面）：默认关 ⇒ 上面两支一字不动。开着时按产品自己的    #: 显式 API 挂 bridge(`gate=0.0`＝挂上但不放行)＋R4 pressure trigger，并把每次观测原样落进
    #: JSONL——生产者**不在这里重写**（`Taiji._record_adaptive_residual_growth_pressure` 在
    #: `taiji/model.py:1069-1106` 已经在算五信号），本处只在 `trigger.observe` 外面套一层记录。
    #: 两条前提不满足就**响亮拒绝**而不是静默跑出零读数：压强记录的调用点在 `observe` 的
    #: predictive 分支里（:2151-2160），而 `answer_chunking` 非 `stream` 时本函数走
    #: `learn_bytes`（下方那支），根本不经过 `observe`。
    if pressure_record is not None:
        if readout != "predictive":
            raise RuntimeError("压强读数只存在于 predictive 读出链上 ⇒ 需要 --readout predictive")
        if str(answer_chunking) != "stream":
            raise RuntimeError(
                f"answer_chunking={answer_chunking!r} 走 learn_bytes 不经过 observe ⇒ 没有压强读数"
            )
        substrate = model.substrate
        #: PLAN-N3-04 §5bis 的两个默认关旗标。`developmental_bridge_gate` 为 None 时**连 setter 都不调**——
        #: G-N3b-1 要的是"不开旗标逐位不变"，少调一次就少一次可动的地方。
        bridge_gate_at_mount = (
            0.0 if developmental_bridge_gate is None else float(developmental_bridge_gate)
        )
        bridge_info = substrate.enable_adaptive_residual_bridge(gate=bridge_gate_at_mount)
        if developmental_bridge_gate is None:
            bridge_gate_actual = bridge_info["gate"]
        else:
            bridge_gate_actual = substrate.set_adaptive_residual_bridge_gate(bridge_gate_at_mount)
            if float(bridge_gate_actual) != bridge_gate_at_mount:
                raise RuntimeError(
                    f"bridge gate 挂载后读回 {bridge_gate_actual!r} 不等于请求值 "
                    f"{bridge_gate_at_mount!r} ⇒ 面作废（放行档的 gate 必须由实际生效值定义）"
                )
        #: owner 2026-10-08 第六次弹窗裁"τ＝按链分别给值"⇒ 落地形状＝训练器给挂载传 policy
        #: （`taiji/model.py:917-943` 的 `enable_adaptive_residual_growth` 本来就收 `policy`），
        #: **产品默认常量 0.70 一字不动**：缺旗标时连 `policy` 参数都不给（＝逐位不变）；给值时用
        #: `dataclasses.replace` **只替 `minimum_pressure` 一枚**，其余五道阈与 `ema_rate` 仍取产品默认，
        #: 挂载后必须读回复核——τ 是判据的母量，请求值不等于生效值时整张面作废而不是照请求值出版。
        if growth_minimum_pressure is None:
            growth_info = substrate.enable_adaptive_residual_growth()
        else:
            growth_info = substrate.enable_adaptive_residual_growth(
                policy=replace(
                    AdaptiveResidualGrowthPolicy(),
                    minimum_pressure=float(growth_minimum_pressure),
                )
            )
        trigger = substrate.adaptive_residual_growth_trigger
        if trigger is None:
            raise RuntimeError("growth trigger 没挂上 ⇒ 压强面无效")
        if growth_minimum_pressure is not None:
            minimum_pressure_actual = float(trigger.policy.minimum_pressure)
            if minimum_pressure_actual != float(growth_minimum_pressure):
                raise RuntimeError(
                    f"minimum_pressure 挂载后读回 {minimum_pressure_actual!r} 不等于请求值 "
                    f"{float(growth_minimum_pressure)!r} ⇒ 面作废（τ 必须由实际生效值定义）"
                )
        developmental_info: dict[str, Any] | None = None
        #: `mode_readings`＝面内自述"每次施加之后读回的 learning_mode"。该模式不入档
        #: （`Taiji.restore` 恢复即回 `read_only`），所以在场性不许由命令行反推（PLAN-N3-04 §3）。
        mode_readings: list[dict[str, Any]] = []
        if developmental_fast_slow:
            developmental_info = substrate.migrate_f1_to_developmental_synapses()
            substrate.set_developmental_f1_learning_mode("fast_slow")
        mode_readings.append(
            {
                "after": "mount",
                "tick": int(ticks),
                "learning_mode": substrate.developmental_f1_learning_mode,
            }
        )
        pressure_path = Path(pressure_record)
        pressure_path.parent.mkdir(parents=True, exist_ok=True)
        pressure_lines = 0
        with pressure_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "kind": "face",
                        "format": "taiji-n3-pressure-face-v2",
                        "bridge": bridge_info,
                        "growth": growth_info,
                        #: 六道阈＋`required_pressure_steps`／`growth_resource_cost`／`ema_rate` 全部读自
                        #: 产品自己的 `trigger.policy`（PLAN-N3-08 §1 的"零抄写"：这一段里不许出现阈值字面量）。
                        "policy": {
                            "minimum_pressure": trigger.policy.minimum_pressure,
                            "minimum_residual_error": trigger.policy.minimum_residual_error,
                            "minimum_fast_slow_conflict": trigger.policy.minimum_fast_slow_conflict,
                            "minimum_activity_saturation": (
                                trigger.policy.minimum_activity_saturation
                            ),
                            "minimum_utility_gap": trigger.policy.minimum_utility_gap,
                            "minimum_resource_state": trigger.policy.minimum_resource_state,
                            "required_pressure_steps": trigger.policy.required_pressure_steps,
                            "growth_resource_cost": trigger.policy.growth_resource_cost,
                            "ema_rate": trigger.policy.ema_rate,
                        },
                        #: DEBT-G61 修法②：`decision_consecutive_pressure_steps` 的语义自述。
                        #: 该列是产品 decision 在"提议后清零"之后出版的读数（提议行恒为 0，
                        #: 可见最大值＝required_pressure_steps-1），由真 trigger 逐行复现于
                        #: `tests/taiji_native/test_n5_02_consecutive_column_semantics_contract.py`。
                        "consecutive_steps_column_semantics": "post_increment_reset_on_proposal",
                        #: 面头写在这支 trigger 被 observe 之前 ⇒ 这里读到的就是它自己的初值（不是抄的）。
                        "ema_initial": {
                            "residual_error_ema": trigger.residual_error_ema,
                            "fast_slow_conflict_ema": trigger.fast_slow_conflict_ema,
                            "activity_saturation_ema": trigger.activity_saturation_ema,
                            "utility_gap_ema": trigger.utility_gap_ema,
                            "resource_state_ema": trigger.resource_state_ema,
                            "consecutive_pressure_steps": trigger.consecutive_pressure_steps,
                        },
                        "developmental": {
                            "fast_slow_requested": bool(developmental_fast_slow),
                            "bundle": developmental_info,
                            "bridge_gate_requested": developmental_bridge_gate,
                            "bridge_gate_actual": bridge_gate_actual,
                            #: τ（`minimum_pressure`）的请求值与生效值成对出版：缺旗标时 requested 为 None、
                            #: actual 是产品默认（现读，不抄常量），给值时两者必须相等（上面已核）。
                            "minimum_pressure_requested": growth_minimum_pressure,
                            "minimum_pressure_actual": float(trigger.policy.minimum_pressure),
                            "mode_readings": mode_readings,
                        },
                        "readout": readout,
                        "seed": int(config.taiji.seed),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        original_trigger_observe = trigger.observe

        def _record_pressure(
            pressure: object, **trigger_kwargs: object
        ) -> object:  # 签名跟随 trigger.observe(pressure, *, structural_budget)
            decision = original_trigger_observe(pressure, **trigger_kwargs)
            line = dict(pressure.to_payload())  # type: ignore[attr-defined]
            line["kind"] = "pressure"
            line["decision_should_propose"] = bool(getattr(decision, "should_propose", False))
            #: PLAN-N3-08 §1：面行补记**产品 decision 自带**的那十个读数（全部 `getattr` 读回，零重算）。
            #: 多加的一个 `decision_digest` 是预注册 §1 清单之外的实现期增项（逐行完整性锚点，已在 ㊵-508 自报）。
            for _key in (
                "pressure",
                "residual_error_ema",
                "fast_slow_conflict_ema",
                "activity_saturation_ema",
                "utility_gap_ema",
                "resource_state_ema",
            ):
                line[f"decision_{_key}"] = float(getattr(decision, _key))
            line["decision_consecutive_pressure_steps"] = int(decision.consecutive_pressure_steps)
            line["decision_structural_budget"] = int(decision.structural_budget)
            line["decision_resource_cost"] = int(decision.resource_cost)
            line["decision_reasons"] = list(decision.reasons)
            line["decision_digest"] = str(decision.decision_digest)
            nonlocal pressure_lines
            pressure_lines += 1
            with pressure_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
            return decision

        trigger.observe = _record_pressure  # type: ignore[method-assign]

    for epoch in range(epochs):  # noqa: B007 — epoch 被 _flush 闭包引用（进度日志）
        if answer_chunking == "per-answer":
            # A30 §2bf：分块喂法。每块一轮 episode（起沿+收沿边界），喂法经产品门面
            # `learn_bytes`（DEBT-G13 已开的口）。窗口统计用 learn_bytes 返回的聚合值，
            # 与连续流路径的"逐符号 prior_prediction"口径不同——进度日志两档都是窗口均值，
            # 不跨档比数值（件里用 answer_chunking 点名面）。
            #: 节奏用**阈值式**（`ticks - last_* >= every`）而不是连续流那支的 `ticks % every == 0`：
            #: 分块档每步跳 len(chunk)+2（约百字节级），取模几乎永不落在 0 上 ⇒
            #: 第一版实测进度与检查点都不落盘（进程在算、磁盘静默）。这是本档第一个自伤，
            #: 守卫 `test_per_answer_progress_and_checkpoint_cadence` 钉住。
            last_progress = base_ticks
            last_checkpoint = base_ticks
            self_answers: Mapping[str, str] | None = None
            if answer_source == "self":
                if not self_answers_path:
                    raise RuntimeError("answer_source=self needs self_answers_path")
                table_path = Path(self_answers_path)
                self_answers = {}
                for line in table_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self_answers[str(row["question"])] = str(row["answer"])
                if not self_answers:
                    raise RuntimeError(f"self answer table is empty: {table_path}")
            for chunk in iter_answer_chunks(
                corpus_paths, max_answer_chars=answer_max_chars, self_answers=self_answers
            ):
                result = model.learn_bytes(
                    chunk,
                    epochs=1,
                    include_boundary=False,
                    include_start_boundary=True,
                    include_end_boundary=True,
                    reset=True,
                )
                observations = int(result.get("observations", 0))
                ticks += len(chunk) + 2  # 起沿/收沿各一个边界符
                window_ticks += max(1, observations)
                window_correct += float(result.get("online_accuracy", 0.0)) * max(1, observations)
                window_surprise += float(result.get("mean_surprise", 0.0)) * max(1, observations)
                if ticks - last_progress >= progress_every:
                    _flush(final=False)
                    window_ticks = 0
                    window_correct = 0
                    window_surprise = 0.0
                    last_progress = ticks
                if ticks - last_checkpoint >= checkpoint_every:
                    _persist()
                    last_checkpoint = ticks
                if max_symbols is not None and ticks >= base_ticks + max_symbols:
                    #: DEBT-G39：先落盘再写退出记账，否则记账自述的是**上一次**保存的字节。
                    _persist()
                    _flush(final=True, exit_reason="max_symbols_reached")
                    return _summary(model, ticks)
            continue
        for symbol in iter_corpus_symbols(
            corpus_paths,
            boundary=boundary,
            end_boundary_after_newline=end_boundary_after_newline,
            max_unique_documents=max_unique_documents,
            counters=stream_counters,
        ):
            step = model.observe(symbol, **observe_kwargs)
            ticks += 1
            if episodic_store is not None:
                if symbol == boundary and doc_pending:
                    cue = (doc_pending + [0] * 32)[:32]
                    episodic_store.write(
                        EpisodicMemoryRecord(
                            memory_id=f"doc-{episodic_store.count:06d}",
                            episode_id=f"n4-material-{episodic_store.count:06d}",
                            tick=ticks,
                            cue=torch.tensor([b / 255.0 for b in cue], dtype=torch.float32),
                        )
                    )
                    doc_pending.clear()
                elif symbol != boundary and len(doc_pending) < 32:
                    doc_pending.append(symbol)
            if n5_shadow and adaptive_shadow is None:
                decision = model.adaptive_residual_growth_decision
                if decision is not None and decision.should_propose:
                    model.propose_adaptive_residual_growth_candidate()
                    adaptive_shadow = model.materialize_adaptive_residual_shadow()
            if step.prior_prediction is not None:
                window_ticks += 1
                window_correct += int(step.prior_prediction == symbol)
                window_surprise += float(step.surprise)
            if ticks % progress_every == 0:
                _flush(final=False)
                window_ticks = 0
                window_correct = 0
                window_surprise = 0.0
            if ticks % checkpoint_every == 0:
                _persist()
            if max_symbols is not None and ticks >= base_ticks + max_symbols:
                _persist()  # DEBT-G39：先落盘，退出记账才自述得成终件字节
                _flush(final=True, exit_reason="max_symbols_reached")
                return _summary(model, ticks)
    _persist()  # DEBT-G39：同上，耗尽支也先落盘
    _flush(final=True, exit_reason="corpus_exhausted")
    return _summary(model, ticks)


def _summary(model: Seed, ticks: int) -> dict[str, float]:
    return {
        "ticks": float(ticks),
        "parameters": float(model.parameter_count()),
    }


def apply_experiment_flags(
    config: SeedConfig,
    *,
    receptors_factored: bool = False,
    predictive_context_region0_only: bool = False,
    readout_position: bool = False,
) -> SeedConfig:
    """把各处**默认关**的实验开关一次写进 config（A-4：把 A 支线已证的部件接到主训练线）。

    抽成纯函数是为了可被单测钉住：**全 False 时必须逐键等于入参**（"默认关 ⇒ 行为不变"），
    单个 True 只许翻自己那一键。守卫见 `tests/taiji_native/test_a4_mainline_flags.py`。
    """

    taiji = config.taiji
    if receptors_factored:
        taiji = replace(taiji, receptors_factored=True)
    if predictive_context_region0_only:
        taiji = replace(taiji, predictive_context_region0_only=True)
    if readout_position:
        taiji = replace(taiji, readout_utf8_position_input=True)
    return config if taiji is config.taiji else replace(config, taiji=taiji)


def default_output_paths(*, smoke: bool, project_root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    """`--checkpoint`／`--progress` 的缺省值：**冒烟既不碰产品件，也不碰 `reports/`**。

    来历（2026-09-28 实测事故）：`--smoke` 只改预算、不改输出路径 ⇒ 它的缺省
    `--checkpoint` 仍是 `checkpoints/seed_corpus.pt`（`PROTECTED_OUTPUTS` 之一）⇒
    一次"快速端到端"把产品件覆盖成了 5000-tick 的冒烟模型
    （靠 `dist/Seed/_internal/checkpoints/` 里的打包副本按 sha256 `c8025db44c65…` 复原）。
    正式跑缺省写法一字不变；只有 `--smoke` 改走 `output/`。

    **那条修法当时只堵了一半，本句是补上的另一半（DEBT-G50，2026-10-08）**：progress 原来仍留在
    `reports/`，而收尾那枚 `*_exit.json` 由它派生 ⇒ 任何一次不带 `--progress` 的冒烟都会覆写
    **受版本控制的** `reports/seed_corpus_smoke_progress_exit.json`（本轮两支烟测实测踩中，
    件内 `online_accuracy` 从 0.2436… 被改成 0.0034…，`git checkout HEAD --` 才救回）。
    旧 docstring 写的"冒烟绝不落到产品件上"**在 progress 这一半上当时并不成立**——
    只读到"机制存在"就把修法规格估低，是同族缺陷。⇒ 现在两个冒烟缺省都在 `output/`。
    """

    if smoke:
        return (
            project_root / "output" / "seed_corpus_smoke.pt",
            project_root / "output" / "seed_corpus_smoke_progress.jsonl",
        )
    return (
        project_root / "checkpoints" / "seed_corpus.pt",
        project_root / "reports" / "seed_corpus_progress.jsonl",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        nargs="+",
        default=[str(PROJECT_ROOT / name) for name in DEFAULT_CORPUS],
    )
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument(
        "--parameter-budget",
        type=int,
        default=None,
        help="自动规划不超过该数量的可学习参数；设置后替代 --scale",
    )
    parser.add_argument(
        "--capacity-policy",
        default=None,
        help="容量策略 JSON；可改变区域深度、比例与 fan-in 密度，需配合 --parameter-budget",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="训练设备：auto、cpu、cuda 或 cuda:N",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-symbols", type=int, default=200_000)
    #: PLAN-N3-10 §2 形状 B：缺省＝不限（今天的形状逐位不变），给了值＝前 K 篇成池后循环重用。
    parser.add_argument("--max-unique-documents", type=int, default=None)
    parser.add_argument("--episodic-mount", action="store_true")
    parser.add_argument("--n5-shadow", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="落盘路径；缺省见 `default_output_paths`：正式跑＝`checkpoints/seed_corpus.pt`，"
        "`--smoke`＝`output/seed_corpus_smoke.pt`（**冒烟绝不碰产品件**）。"
        "2026-09-28 起：正式跑不显式给 `--checkpoint` 会被**响亮拒绝**（防产品件被意外覆盖）。",
    )
    parser.add_argument(
        "--i-accept-default-product-checkpoint",
        action="store_true",
        help="显式接受「正式跑缺省写靶＝产品件」这一行为（2026-09-28 加固旗标；日常应改用 --checkpoint）。",
    )
    parser.add_argument(
        "--allow-overwrite-checkpoint",
        action="store_true",
        help="显式接受「覆写一枚已存在且不是本次 --resume 源的检查点」（2026-10-03 加固旗标，"
        "`DEBT-G35`：原地覆写会让旧读数的源件字节永久消失；日常应改用新的 --checkpoint 路径）。",
    )
    parser.add_argument(
        "--progress",
        default=None,
        help="进度流路径；缺省随 `--smoke` 一起走 `output/seed_corpus_smoke_progress.jsonl`"
        "（DEBT-G50：冒烟的两个缺省都不再落 `reports/`）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="热启动：从该 Seed 信封继续训练。给了它就用**档里的**架构（忽略 "
        "--scale/--parameter-budget/--capacity-policy），再在其上应用实验开关。",
    )
    parser.add_argument(
        "--keep-checkpoints",
        choices=("on", "off"),
        default="on",
        help="保号存档：每次落盘额外写一份 checkpoint_<tick>.pt（默认开）。"
        "关掉它会退回「反复覆盖同一个文件」的旧行为——那样后来才想到的指标无法事后补算。",
    )
    parser.add_argument(
        "--checkpoint-history-dir",
        default=None,
        help="保号存档目录；缺省为 <--checkpoint>.history/",
    )
    parser.add_argument(
        "--keep-history-max",
        type=int,
        default=None,
        help="保号存档上限（DEBT-G40）：超限薄中间，首尾各一枚必留；不给＝不限，"
        "现行「每次落盘都留一份」的行为逐位不变。",
    )
    parser.add_argument(
        "--receptors-factored",
        action="store_true",
        help="R2 组合绑定实验：把 BytePredictiveContext.receptors 改成按两半分块。"
        "**默认关，现行行为与载荷逐位不变**（守卫 tests/taiji_native/"
        "test_receptor_factorization_contract.py）。",
    )
    parser.add_argument(
        "--predictive-context-region0-only",
        action="store_true",
        help="A-4：把 `predictive_context` 的输入限制在**区 0**（其余区按零掩码、宽度不变）。"
        "R2 组合绑定线实测：区 0 单独承载槽结构（16M 时 0.729/0.709），三区全拼接会被"
        "区1/2 的随机方向稀释（0.420）。**默认关，现行行为与载荷逐位不变**"
        "（守卫 tests/taiji_native/test_predictive_context_region0_mask.py）。",
    )
    parser.add_argument(
        "--readout",
        choices=("action", "predictive"),
        default="predictive",
        help="哪条读出链在学：`predictive`（**2026-09-28 owner 裁定 (a) 起的主线默认**，训 F1 预测"
        "读出＋私有时间语境——A 支线已证部件都挂在这条链上）或 `action`（旧缺省，训 F4／运动"
        "解码器；与 2026-09-27 前行为逐位相同的显式逃生口）。",
    )
    parser.add_argument(
        "--pressure-record",
        type=Path,
        default=None,
        help="PLAN-N3-01 步骤一：把每 tick 的 R4 pressure 观测落到该 JSONL（只在 --readout "
        "predictive 且 answer_chunking=stream 时可用；bridge 以 gate=0.0 挂上，不放行生长）",
    )
    parser.add_argument(
        "--developmental-fast-slow",
        dest="developmental_fast_slow",
        action="store_true",
        default=False,
        help="PLAN-N3-04 §5bis（默认关＝逐位不变）：在 bridge 挂载之后把 F1 迁移到发育突触叠加层，"
        "并把写入模式设为 fast_slow（wake 更新写进 fast_delta）。该模式**故意不入档**"
        "（`Taiji.restore` 恢复即回 read_only，`taiji/model.py:3612/3785`），所以面内必须自述"
        "每次施加之后读回的 learning_mode。仅与 `--pressure-record` 同用；不调 propose/promote。",
    )
    parser.add_argument(
        "--developmental-bridge-gate",
        dest="developmental_bridge_gate",
        type=float,
        default=None,
        help="PLAN-N3-04 §5bis（默认 None＝沿用挂载时的 gate=0.0，行为不变）：把 bridge 挂载的"
        "gate 换成此值，并在挂载后用 set_adaptive_residual_bridge_gate 复核实际生效值。"
        "gate>0 会让 bridge 参与读出（＝改了行为，不是中性改动），owner 已裁 0.25 与 1.0 两档。",
    )
    parser.add_argument(
        "--growth-min-pressure",
        dest="growth_minimum_pressure",
        type=float,
        default=None,
        help="PLAN-N4-01 之前的 N3 乙线（owner 2026-10-08 第六次弹窗裁「τ 按链分别给值」）：把生长闸的 "
        "`minimum_pressure` 换成本值，其余五道阈与 ema_rate 仍取产品默认（只动一枚）。"
        "缺省 None＝连 policy 参数都不给，行为与今天逐位相同；产品默认常量 0.70 不在本处改动。"
        "挂载后用 trigger.policy 读回复核并把 requested/actual 成对写进面头。"
        "只在 --pressure-record 的挂载分支里生效。",
    )
    parser.add_argument(
        "--readout-position",
        dest="readout_position",
        action="store_true",
        default=True,
        help="PLAN-R2-01：给 F1 读出加一条**显式的 UTF-8 字节位置输入**（4 维 one-hot，"
        "零初始化）。A 支线两臂实测：它把裸通道的字节合法性从 ~0 抬到 100%%"
        "（真非法率 99.0%%→0%%）、表层成句 0→68。**2026-09-28 owner 裁定 (a) 起为主线"
        "训练默认**；`--no-readout-position` 是显式逃生口（对照/复现旧配方用）。",
    )
    parser.add_argument(
        "--no-readout-position",
        dest="readout_position",
        action="store_false",
        help="关掉位置输入（P0 对照臂/旧配方复现用）。",
    )
    parser.add_argument(
        "--no-end-boundary-after-newline",
        dest="end_boundary_after_newline",
        action="store_false",
        help="关掉'每篇正文后补一个换行再落结束边界'（A30 §2aa 目标编码对齐——把结束目标"
        "放到模型已会预测的换行之后，决策级两档 26/30 与 263/300 对对照 0）。"
        "2026-09-29 owner 条件授权起为主线配方默认；本旗标是旧形状复现的逃生口。",
    )
    parser.add_argument(
        "--answer-chunking",
        choices=("stream", "per-answer"),
        default="stream",
        help="A30 §2bf（owner 2026-09-30 裁定①）：喂法。`stream`（默认，逐位不变）＝连续流"
        "（每篇文档一个边界符）；`per-answer`＝每答一块（问句+答案+换行，起沿/收沿各一边界、"
        "每块一轮 episode）——把正式档里买到真自停的分块短答形状搬进主线。",
    )
    parser.add_argument(
        "--answer-max-chars",
        type=int,
        default=0,
        dest="answer_max_chars",
        help="仅 `--answer-chunking per-answer` 生效：答案按**字符**截到此上限（0＝不截断）。"
        "截断只动答案不动问句；与 on-policy 仪器 `sized` 臂同口径。与 stream 档组合会响亮拒绝。",
    )
    parser.add_argument(
        "--answer-source",
        choices=("corpus", "self"),
        default="corpus",
        help="A30 §2bg：答案从哪来。`corpus`（默认，逐位不变）＝语料答案；`self`＝模型自写的"
        "短答表（`build_taiji_a30_self_answers.py` 产物，`--self-answers` 给路径）——§2bg 三线表里"
        "唯一三线全过的是自写答案形状。仅 `per-answer` 档生效，与 stream 组合会响亮拒绝。",
    )
    parser.add_argument(
        "--self-answers",
        default=None,
        dest="self_answers_path",
        help="`--answer-source self` 的表路径（jsonl：{question, answer}）。查不到问句时响亮失败"
        "（语料与缓存必须同源同序）。",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny default config and budget for a fast end-to-end run",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    #: 响亮失败而不是静默空转：位置输入挂在 F1 预测读出上，`action` 档根本不走那条链，
    #: 2026-09-28 实测两臂读数**逐位相同**才发现（详见 PLAN-A-26 §6）。
    if args.readout_position and args.readout != "predictive":
        parser.error(
            "--readout-position wires the F1 predictive readout only; with --readout action "
            "it is a silent no-op. Pass --readout predictive."
        )

    #: A30 §2bf：截断只在分块喂法里有对象；与连续流组合是静默空转 ⇒ 响亮拒绝。
    if args.answer_max_chars and args.answer_chunking != "per-answer":
        parser.error(
            "--answer-max-chars only applies to --answer-chunking per-answer; with the "
            "stream feed it is a silent no-op."
        )
    #: A30 §2bg：自写答案同样只在分块喂法里有对象；且必须给表路径。
    if args.answer_source == "self" and args.answer_chunking != "per-answer":
        parser.error(
            "--answer-source self only applies to --answer-chunking per-answer; with the "
            "stream feed it is a silent no-op."
        )
    if args.answer_source == "self" and not args.self_answers_path:
        parser.error("--answer-source self requires --self-answers <path>.")
    if args.answer_source == "corpus" and args.self_answers_path:
        parser.error("--self-answers given without --answer-source self.")

    #: PLAN-N3-04 §5bis 的响亮拒绝：两个新旗标的挂载点住在 `--pressure-record` 那一支里 ⇒ 只给旗标
    #: 不给压强面就是**静默空转**（同一类雷：`--readout-position` 在 action 档空转，注释里写着）。
    if (
        args.developmental_fast_slow
        or args.developmental_bridge_gate is not None
        or args.growth_minimum_pressure is not None
    ) and args.pressure_record is None:
        parser.error(
            "--developmental-fast-slow / --developmental-bridge-gate / --growth-min-pressure 只在 "
            "--pressure-record 的挂载分支里生效；单给旗标会静默空转 ⇒ 请同时给 --pressure-record <path>。"
        )
    if (
        args.developmental_bridge_gate is not None
        and not 0.0 <= args.developmental_bridge_gate <= 1.0
    ):
        parser.error("--developmental-bridge-gate 必须落在 0.0..1.0 之间。")
    #: τ 的取值域与 bridge gate 同一形状：越界＝响亮拒绝，而不是挂上一个全场够不到的阈
    #: （DEBT-G53 的实证就是"0.70 在 EMA 口径上算术不可达"，那种阈不该被静默接受）。
    if args.growth_minimum_pressure is not None and not 0.0 <= args.growth_minimum_pressure <= 1.0:
        parser.error("--growth-min-pressure 必须落在 0.0..1.0 之间。")

    #: 二次事故加固（2026-09-28，同日第二撞）：缺省写靶＝产品件 `checkpoints/seed_corpus.pt`，
    #: 而缺省 readout 已改 `predictive`＋位置输入 ⇒ 任何"只传一两个旗标"的调用（含测试里
    #: 的 monkeypatch argv）都会**真训并把产品件覆盖掉**。⇒ 正式跑必须显式给 `--checkpoint`
    #: （或加 `--i-accept-default-product-checkpoint` 的显式确认旗标）；`--smoke` 走 `output/` 不受影响。
    default_checkpoint, default_progress = default_output_paths(smoke=bool(args.smoke))
    if not args.smoke and not args.checkpoint and not args.i_accept_default_product_checkpoint:
        parser.error(
            "refusing to write the product checkpoint by default. Pass --checkpoint "
            "(e.g. output/<run>/checkpoint.pt) or --i-accept-default-product-checkpoint."
        )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    progress_path = Path(args.progress) if args.progress else default_progress

    #: DEBT-G35 的机器侧防口（2026-10-03）：`--checkpoint` 落在一枚**已存在**的件上、而它又**不是**本次的
    #: `--resume` 源 ⇒ 响亮拒绝。为什么不能只靠约定：`output/a31_ding3_boundary/checkpoint.pt` 被第二次跑档
    #: 原地覆写过，于是 10 份已入库读数记的 `checkpoint_sha256_before=79b1a99c…` 在盘上再也找不到对应字节，
    #: 那批结论（§2ai–§2an）永远无法复算——按 `git ls-files` 或链接检查都查不出来，因为**路径是对的**。
    #: 原地续训（`--resume X` 且 `--checkpoint X`）是合法用法，不在拒绝之列；其余覆写要么换名，
    #: 要么显式给 `--allow-overwrite-checkpoint`。
    resume_source = Path(args.resume).resolve() if args.resume else None
    if (
        checkpoint_path.exists()
        and checkpoint_path.resolve() != resume_source
        and not args.allow_overwrite_checkpoint
    ):
        parser.error(
            f"refusing to overwrite existing checkpoint {checkpoint_path}: it is not this run's "
            "--resume source, so earlier readings that anchor on it would become unreproducible. "
            "Point --checkpoint at a new path, or pass --allow-overwrite-checkpoint deliberately."
        )

    if args.smoke:
        config = SeedConfig()
        max_symbols = 5_000
    elif args.resume:
        # 热启动必须"同底"：架构直接从档里重建，**不**用 --scale/--parameter-budget 的画像——
        # `seed_beta` 的 config 不等于任何 scale 画像（守卫见 tests/taiji_native/
        # test_p3b_campaign_contract.py::test_config_must_be_rebuilt_from_the_envelope），
        # 拿画像重建会在 restore 的"档配比架构"守卫处撞墙。
        envelope = torch.load(args.resume, weights_only=False)
        if not isinstance(envelope, dict) or "config" not in envelope:
            parser.error(f"--resume {args.resume} is not a Seed envelope (no 'config' key)")
        config = SeedConfig.from_dict(dict(envelope["config"]))
        max_symbols = args.max_symbols
    else:
        if args.parameter_budget is None:
            if args.capacity_policy is not None:
                parser.error("--capacity-policy requires --parameter-budget")
            taiji_config = TaijiConfig.training_profile(scale=args.scale, seed=args.seed)
        else:
            policy = (
                load_capacity_policy(args.capacity_policy)
                if args.capacity_policy is not None
                else None
            )
            taiji_config = TaijiConfig.capacity_profile(
                args.parameter_budget,
                policy=policy,
                seed=args.seed,
            )
        config = SeedConfig(taiji=taiji_config)
        max_symbols = args.max_symbols

    history_dir = None
    if args.keep_checkpoints == "on":
        history_dir = (
            Path(args.checkpoint_history_dir)
            if args.checkpoint_history_dir
            else (Path(str(checkpoint_path) + ".history"))
        )

    #: A-4（把 A 支线已证的部件推广到主训练线）：全部开关**默认关**，关着时 config 与
    #: 载荷逐位不变；打开即写进 config（随 checkpoint 一起落盘，所以"这条读数用的是哪套配方"
    #: 永远可从档里查出来，不靠外部记录）。
    experiment_flags = {
        "receptors_factored": bool(args.receptors_factored),
        "predictive_context_region0_only": bool(args.predictive_context_region0_only),
        "readout_utf8_position_input": bool(args.readout_position),
    }
    config = apply_experiment_flags(
        config,
        receptors_factored=experiment_flags["receptors_factored"],
        predictive_context_region0_only=experiment_flags["predictive_context_region0_only"],
        readout_position=experiment_flags["readout_utf8_position_input"],
    )

    summary = run_training(
        corpus_paths=args.corpus,
        config=config,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
        progress_path=progress_path,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        max_symbols=max_symbols,
        resume_checkpoint=args.resume,
        # PLAN-A-26 热启动：让受载档里这几处 config 副本与本臂架构一致，
        # 否则 restore 的"档配比架构"守卫会拦下这次有意的配方切换。
        resume_config_overrides=experiment_flags if args.resume else None,
        readout=args.readout,
        device=args.device,
        keep_history=history_dir,
        keep_history_max=args.keep_history_max,
        end_boundary_after_newline=bool(args.end_boundary_after_newline),
        answer_chunking=str(args.answer_chunking),
        answer_max_chars=int(args.answer_max_chars),
        answer_source=str(args.answer_source),
        self_answers_path=args.self_answers_path,
        pressure_record=args.pressure_record,
        developmental_fast_slow=bool(args.developmental_fast_slow),
        developmental_bridge_gate=args.developmental_bridge_gate,
        growth_minimum_pressure=args.growth_minimum_pressure,
        max_unique_documents=args.max_unique_documents,
        episodic_mount=args.episodic_mount,
        n5_shadow=args.n5_shadow,
    )
    print(json.dumps(summary, ensure_ascii=False))
    #: DEBT-G14②：操作侧曾拿着一个 **0 字节的 `run.log`** 判断"这轮跑到哪了"——空文件比没有更误导。
    #: 收尾把退出记账**打到 stdout**（谁想留档就重定向这一段），它的键与 `*_exit.json` 同源同值。
    print(
        json.dumps(
            json.loads(exit_record_path(progress_path).read_text(encoding="utf-8")),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
