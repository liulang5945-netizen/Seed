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
import json
import sys
import time
from collections.abc import Iterator, Sequence
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
from taiji import CapacityPolicy, TaijiConfig  # noqa: E402

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


def iter_corpus_symbols(
    paths: Sequence[Path | str],
    *,
    boundary: int = TaijiConfig().boundary_symbol,
    end_boundary_after_newline: bool = False,
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

    for text in iter_native_documents(paths):
        yield boundary
        yield from text.encode("utf-8")
        if end_boundary_after_newline:
            yield 0x0A


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
    end_boundary_after_newline: bool = False,
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
        envelope = attach_metadata(
            model.checkpoint(),
            tick=ticks,
            corpus_fingerprint=fingerprint,
            extra={
                "trainer": "train_seed_corpus",
                # A30 §2aa：喂入形状（结束边界是否落在换行之后）随档登记——
                # 配方可从档里查出来，不靠外部记录。
                "end_boundary_after_newline": bool(end_boundary_after_newline),
            },
        )
        atomic_save(envelope, checkpoint_path)
        # 2026-09-23：**保号存档**。此前 `--checkpoint-every` 反复覆盖同一个文件，
        # 于是"训练中途某个 tick 的状态"直接消失——后来才想到要量的指标（例如槽可分离性）
        # 连事后补算都做不到，只剩首尾两个端点。这里每次落盘**额外**写一份带 tick 的快照；
        # 主路径 `checkpoint_path` 的行为一字不变（兼容既有工具与流程）。
        if keep_history is not None:
            atomic_save(envelope, keep_history / f"checkpoint_{ticks:012d}.pt")

    started = time.perf_counter()
    window_ticks = 0
    window_correct = 0
    window_surprise = 0.0

    def _flush(final: bool) -> None:
        if window_ticks <= 0 and not final:
            return
        entry = {
            "epoch": epoch,
            "ticks": ticks,
            "window_ticks": window_ticks,
            "online_accuracy": window_correct / max(1, window_ticks),
            "mean_surprise": window_surprise / max(1, window_ticks),
            "holdout_surprise": model.score_bytes(HOLDOUT_PROBE)["mean_surprise"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    #: A-4／PLAN-A-26：读出模式决定**哪条链**在学。`action`＝既有行为逐位不变；
    #: `predictive` 把下一字节误差送到 F1 专用读出（＋私有时间语境）——`observe` 明令此时
    #: 不得同时训运动器，所以 `learn_motor=False`。A 支线的已证部件都在这条链上。
    observe_kwargs: dict[str, object] = {"learn": True, "readout": readout}
    if readout == "predictive":
        observe_kwargs["learn_motor"] = False

    for epoch in range(epochs):  # noqa: B007 — epoch 被 _flush 闭包引用（进度日志）
        for symbol in iter_corpus_symbols(
            corpus_paths, boundary=boundary, end_boundary_after_newline=end_boundary_after_newline
        ):
            step = model.observe(symbol, **observe_kwargs)
            ticks += 1
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
                _flush(final=True)
                _persist()
                return _summary(model, ticks)
    _flush(final=True)
    _persist()
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
    """`--checkpoint`／`--progress` 的缺省值：**冒烟绝不落到产品件上**。

    来历（2026-09-28 实测事故）：`--smoke` 只改预算、不改输出路径 ⇒ 它的缺省
    `--checkpoint` 仍是 `checkpoints/seed_corpus.pt`（`PROTECTED_OUTPUTS` 之一）⇒
    一次"快速端到端"把产品件覆盖成了 5000-tick 的冒烟模型
    （靠 `dist/Seed/_internal/checkpoints/` 里的打包副本按 sha256 `c8025db44c65…` 复原）。
    正式跑缺省写法一字不变；只有 `--smoke` 改走 `output/`。
    """

    if smoke:
        return (
            project_root / "output" / "seed_corpus_smoke.pt",
            project_root / "reports" / "seed_corpus_smoke_progress.jsonl",
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
        "--progress",
        default=None,
        help="进度流路径；缺省随 `--smoke` 一起改走 `reports/seed_corpus_smoke_progress.jsonl`",
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
        "--readout-position",
        dest="readout_position",
        action="store_true",
        default=True,
        help="PLAN-R2-01：给 F1 读出加一条**显式的 UTF-8 字节位置输入**（4 维 one-hot，"
        "零初始化）。A 支线两臂实测：它把裸通道的字节合法性从 ~0 抬到 100%"
        "（真非法率 99.0%→0%）、表层成句 0→68。**2026-09-28 owner 裁定 (a) 起为主线"
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
        end_boundary_after_newline=bool(args.end_boundary_after_newline),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
