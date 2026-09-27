"""F1：模型有没有"我在被问什么"这个对象（零训练线性探针；PLAN-A-24 §5d 前置阶梯第 1 阶）。

**它回答的问题**：同一件事、换一种问法，模型"答："那一刻的状态**变不变**。
`PLAN-A-24` §5d 把 F1 判为"问句是否被当作槽位请求"，并规定：
F1 不过线时记"**没有『问题』这个对象**"，**后续机制测试不得假定它能读题**。
（这条闸门的存在理由：A2 探针已实测"提问末端状态线性读不出该查哪条"，
A 支线此前所有"检索/选择"类工作都建立在一个未验证的前提上。）

## 与 P3 的区别（别混用）
`probe_taiji_r2_readout_conditionality.py`（P3）问的是"**内容词**换了，读出器输入变不变"；
本件问的是"**同一个内容**、换问法，状态里能不能读出来"——是同一条链上的另一问。

## 刺激集（跑前冻结，随本文件一起提交，之后不得增删改）
12 个事实 × 4 种问法 ＝ 48 条题面；四种问法**都要求同一个答案（名字）**、**刻意等长**
（各 9 个汉字＋全角问号＝30 字节），问法之间的差异只在**用词**，不在长度。
题面形态 `问：我叫{token}。{phrasing}\\n答：`——事实与提问同轮 ⇒ 不涉记忆，纯"读题"。

## 特征（三把都报，避免单口径当结论）
* `f1_context`（`motor_context`，96 维）：读出器赖以预测下一字节的**直接输入**；
* `cue_full`（`fabric.cortical_context`，1152 维）：生产选择机构用的那把；
* `probs`（257 维）：`softmax(synapses·context + bias)`，即读出分布本身。

**披露**：`motor_context` 不在公开的 `TaijiStep` 上，本件从 `substrate._state` 取（私有），
与 P3/A2 同法——因为这条判据问的是"喂给读出的那个东西"，公开接口没暴露它。

## 零假设对照（仪器可用性前置；不满足 ⇒ 整件拒跑）
同一条题面在**两个全新 episode** 里各跑一遍：三把特征都必须**逐位相同**。
否则任何"非零差"都可能是噪声底，全件作废。

## 判读线（跑前冻结；2026-09-28 在**真实打分前**做过一次仪器有效性修正，理由见下）
**为什么阳性对照用"相对边际"而不是绝对 0.8**：本题面只有 48 条（12 类事实），
4 折 CV 下每折训练 36 条、12 类 ⇒ 绝对线没有统计效力（分类器可能拟合不动而误报"状态不可读"）。
诚实的对照是**相对它自己的打乱标签零假设**（这也是本仓 A2/P3 探针已有的纪律）。
* **阳性对照（缺一不出结论）**：`fact CV − fact 打乱对照 ≥ 0.30`——状态里连"是哪条事实"都读不出，
  说明该状态整体不可读，本件结论降级为"探针在该状态上无区分度"。
* **P-过**：`phrasing CV ≥ 0.60`（chance 0.25）**且** `phrasing CV − 打乱对照 ≥ 0.25`
  **且**"只用题面长度"的单特征对照 < 0.60 ⇒ **问法进了表征** ⇒ 机制测试可假定能读题。
* **P-否**：`phrasing CV − 打乱对照 ≤ 0.10` ⇒ **没有问题这个对象**（与 A2 探针互证）。
* 中间带 ⇒ `not_resolved`，分层报数不裁定（不硬凑结论）。

## 参考臂
`pos_on`（主对象，位置输入已训）／`pos_off`（同配方对照）／`checkpoints/seed_beta.pt`
（读出未训的裸基座，作"读出从未被训"的历史参照）。

纪律：零训练（探针权重不进模型）、`learn=False`、`use_memory=False`、基座 sha 复核、
判读件不覆写、两臂同协议同种子。
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

FOLDS = 4
EPOCHS = 400
LR = 0.05
WEIGHT_DECAY = 1e-3
SEED = 20260928
FACT_MARGIN = 0.30
PHRASING_HIGH_LINE = 0.60
PHRASING_HIGH_MARGIN = 0.25
PHRASING_LOW_MARGIN = 0.10
CONTROL_SLACK = 0.05
MAX_ANSWER_BYTES = 24

#: 12 个双字名（与 F0 同一批人名池，另补四个）。
TOKENS: tuple[str, ...] = (
    "天磊",
    "静怡",
    "志远",
    "小雨",
    "建华",
    "文博",
    "海燕",
    "国强",
    "明轩",
    "阿岩",
    "阿蒙",
    "秋兰",
)

#: 四种问法：都要求同一个答案（名字），且**刻意等长**（各 9 个汉字 + 全角问号 ＝ 30 字节）。
#: 等长是设计上的去混淆：首版用的四种问法长度不等（21/24/24/30 字节），冒烟时
#: "只用题面长度"这一个特征就把问法分到 0.667 ⇒ 问法可读性会被长度冒充。改成等长后，
#: 长度单特征对照退化为 chance，混淆由**设计**消掉而不再只靠对照打折。
PHRASINGS: tuple[str, ...] = (
    "你还记得我的名字吗？",
    "你能说出我的名字吗？",
    "你再讲一遍我的名字？",
    "我叫什么名字来着啊？",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(token: str, phrasing: str) -> str:
    return f"问：我叫{token}。{phrasing}\n答："


def _neutral_prompt() -> str:
    """零假设对照用的一条固定题面（不含任何标签信息）。"""
    return "问：你好。\n答："


def _observe_prompt(substrate: Any, text: str) -> None:
    """产品链喂法（与 A2.6/F0 同形）：reset → 边界符 → 逐字节 observe(learn=False)。"""
    substrate.reset_dynamics(episode_id="f1-probe")
    substrate.observe(
        int(substrate.config.boundary_symbol),
        learn=False,
        readout="predictive",
        use_memory=False,
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _features(substrate: Any) -> dict[str, torch.Tensor]:
    state = substrate._state
    cue = substrate.fabric.cortical_context(state.regions).detach().cpu().float()
    return {
        "f1_context": state.motor_context.detach().cpu().float(),
        "cue_full": cue,
        "probs": state.motor_probabilities.detach().cpu().float(),
    }


def _output_bytes(substrate: Any, text: str) -> bytes:
    """同一条链的贪心输出（L0 只作回归，不入机制结论）。"""
    return substrate.generate(
        text.encode("utf-8"),
        MAX_ANSWER_BYTES,
        stop_at_boundary=True,
        sample=False,
        use_memory=False,
    )


def collect_rows(checkpoint: Path, *, rows_limit: int | None = None) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is not None:
        raise SystemExit("copy circuit mounted; F1 requires a bare base")

    #: 零假设对照：同一条题面在两个全新 episode 里各跑一遍。
    null_runs = []
    for _ in range(2):
        _observe_prompt(substrate, _neutral_prompt())
        null_runs.append(_features(substrate))
    null_identical = all(
        bool(torch.equal(null_runs[0][name], null_runs[1][name])) for name in null_runs[0]
    )

    rows: list[dict[str, Any]] = []
    cells = [(fact, phrasing) for fact in range(len(TOKENS)) for phrasing in range(len(PHRASINGS))]
    if rows_limit is not None:
        cells = cells[:rows_limit]
    for fact, phrasing in cells:
        text = _prompt(TOKENS[fact], PHRASINGS[phrasing])
        _observe_prompt(substrate, text)
        features = _features(substrate)
        rows.append(
            {
                "fact": fact,
                "phrasing": phrasing,
                "prompt_bytes": len(text.encode("utf-8")),
                "output": _output_bytes(substrate, text).hex(),
                **{name: value.tolist() for name, value in features.items()},
            }
        )
    return {
        "rows": rows,
        "null_control": {
            "identical_across_episodes": null_identical,
            "detail": "同一条固定题面两个全新 episode 的三把特征逐位相同",
        },
    }


def _cv_accuracy(features: torch.Tensor, labels: torch.Tensor, seed: int) -> dict[str, Any]:
    """FOLDS 折 CV 的多类逻辑回归（torch 全批 Adam；样本 ≤48、维度 ≤1152，全批足够）。

    与 A2/P3 探针同一协议（同一批超参），只把二分类换成多类 softmax，以便同时读
    "事实"（12 类）与"问法"（4 类）两把标签。
    """

    classes = int(labels.max().item()) + 1
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(features.shape[0], generator=generator).tolist()
    fold_accs: list[float] = []
    for fold in range(FOLDS):
        test_idx = order[fold::FOLDS]
        train_idx = [index for index in order if index not in set(test_idx)]
        if not test_idx or not train_idx:
            continue
        x_train, y_train = features[train_idx], labels[train_idx].long()
        x_test, y_test = features[test_idx], labels[test_idx].long()
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        weight = torch.zeros(x_train.shape[1], classes, requires_grad=True)
        bias = torch.zeros(classes, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x_train @ weight + bias, y_train)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = (x_test @ weight + bias).argmax(dim=1)
            fold_accs.append(float((predictions == y_test).float().mean()))
    return {
        "fold_accuracies": [round(value, 4) for value in fold_accs],
        "mean_accuracy": round(sum(fold_accs) / max(len(fold_accs), 1), 4),
        "chance": round(1.0 / classes, 4),
    }


def evaluate(arm: dict[str, Any]) -> dict[str, Any]:
    rows = arm["rows"]
    fact_labels = torch.tensor([row["fact"] for row in rows])
    phrasing_labels = torch.tensor([row["phrasing"] for row in rows])
    lengths = torch.tensor([[float(row["prompt_bytes"])] for row in rows])

    def _shuffled(labels: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator().manual_seed(SEED + 1)
        return labels[torch.randperm(labels.shape[0], generator=generator)]

    features: dict[str, Any] = {}
    for name in ("f1_context", "cue_full", "probs"):
        tensor = torch.tensor([row[name] for row in rows])
        entry: dict[str, Any] = {}
        for label, labels in (("fact", fact_labels), ("phrasing", phrasing_labels)):
            real = _cv_accuracy(tensor, labels, seed=SEED)
            control = _cv_accuracy(tensor, _shuffled(labels), seed=SEED)
            entry[label] = {
                "real": real,
                "shuffled_label_control": control,
                "above_shuffled_margin": round(
                    real["mean_accuracy"] - control["mean_accuracy"], 4
                ),
            }
        features[name] = entry
    #: 长度单特征对照：若光凭题面字节数就能把问法分出来，phrasing CV 要打折。
    length_control = {
        "phrasing": _cv_accuracy(lengths, phrasing_labels, seed=SEED)["mean_accuracy"],
        "fact": _cv_accuracy(lengths, fact_labels, seed=SEED)["mean_accuracy"],
    }

    outputs = [bytes.fromhex(row["output"]) for row in rows]
    by_fact: dict[int, list[bytes]] = {}
    by_phrasing: dict[int, list[bytes]] = {}
    for row, output in zip(rows, outputs):
        by_fact.setdefault(row["fact"], []).append(output)
        by_phrasing.setdefault(row["phrasing"], []).append(output)

    def _pairs(values: list[bytes]) -> list[int]:
        return [
            sum(a != b for a, b in zip(values[i], values[j]))
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ]

    phrasing_byte_delta = [d for group in by_fact.values() for d in _pairs(group)]
    fact_byte_delta = [d for group in by_phrasing.values() for d in _pairs(group)]

    def _mean(values: list[int]) -> float | None:
        return round(sum(values) / len(values), 4) if values else None

    return {
        "features": features,
        "length_only_control": length_control,
        "surface_L0": {
            "note": "只作回归报数，不入机制结论（§5d：L0 不得作能力/机制判据）",
            "mean_byte_delta_within_fact_across_phrasing": _mean(phrasing_byte_delta),
            "mean_byte_delta_within_phrasing_across_fact": _mean(fact_byte_delta),
            "exact_match_rate_within_fact_across_phrasing": round(
                sum(
                    1
                    for group in by_fact.values()
                    for i in range(len(group))
                    for j in range(i + 1, len(group))
                    if group[i] == group[j]
                )
                / max(sum(1 for group in by_fact.values() for i in range(len(group)) for j in range(i + 1, len(group))), 1),
                4,
            ),
        },
    }


def _best(evaluation: dict[str, Any], label: str, key: str) -> float:
    """三把特征里取最好的一把（避免把单口径当结论；三把原样都进判读件）。"""

    values: list[float] = []
    for result in evaluation["features"].values():
        entry = result[label]
        values.append(
            float(entry["real"]["mean_accuracy"]) if key == "real" else float(entry[key])
        )
    return max(values)


def _verdict(evaluation: dict[str, Any]) -> str:
    """判读（跑前冻结）：阳性对照先过，再看问法。"""

    if _best(evaluation, "fact", "above_shuffled_margin") < FACT_MARGIN:
        return "positive_control_failed_state_unreadable"
    phrasing_cv = _best(evaluation, "phrasing", "real")
    phrasing_margin = _best(evaluation, "phrasing", "above_shuffled_margin")
    if (
        phrasing_cv >= PHRASING_HIGH_LINE
        and phrasing_margin >= PHRASING_HIGH_MARGIN
        and evaluation["length_only_control"]["phrasing"] < PHRASING_HIGH_LINE
    ):
        return "question_object_present"
    if phrasing_margin <= PHRASING_LOW_MARGIN:
        return "no_question_object"
    return "not_resolved"


def _controls_clean(evaluation: dict[str, Any]) -> bool:
    for result in evaluation["features"].values():
        for label in ("fact", "phrasing"):
            entry = result[label]
            if (
                entry["shuffled_label_control"]["mean_accuracy"]
                > entry["real"]["chance"] + CONTROL_SLACK
            ):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--label", default=None, help="判读件里记的臂名（缺省＝checkpoint 路径）")
    parser.add_argument("--out-report", required=True)
    parser.add_argument("--limit", type=int, default=None, help="smoke 用；缺省全 48 题")
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    arm = collect_rows(checkpoint, rows_limit=args.limit)
    rows = arm["rows"]
    if not rows:
        print(json.dumps({"guard_ok": False, "error": "no rows"}, ensure_ascii=False))
        return 2
    evaluations = evaluate(arm)
    verdict = _verdict(evaluations)
    report = {
        "format": "taiji-f1-question-phrasing-probe-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §5d 前置阶梯 F1",
        "checkpoint": args.checkpoint,
        "arm_label": args.label or args.checkpoint,
        "stimuli": {
            "tokens": list(TOKENS),
            "phrasings": list(PHRASINGS),
            "items": len(rows),
            "pattern": "问：我叫{token}。{phrasing}\\n答：",
        },
        "cv_protocol": {
            "folds": FOLDS,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "classifier": "multiclass logistic regression (torch full-batch Adam)",
        },
        "reading_lines": {
            "positive_control_fact_margin": FACT_MARGIN,
            "phrasing_high": PHRASING_HIGH_LINE,
            "phrasing_high_margin": PHRASING_HIGH_MARGIN,
            "phrasing_low_margin": PHRASING_LOW_MARGIN,
            "control_slack": CONTROL_SLACK,
        },
        "null_control": arm["null_control"],
        "evaluation": evaluations,
        "verdict": verdict,
        "what_would_overturn": (
            "零假设对照不在位（同题面两 episode 特征不同）⇒ 整件作废；"
            "打乱标签对照高于 chance+0.05 ⇒ 特征泄漏；"
            "长度单特征对照 ≥0.60 ⇒ 问法 CV 可被题面长度解释，P-过不成立；"
            f"阳性对照 fact CV 相对边际 <{FACT_MARGIN} ⇒ 该状态整体不可读，结论降级为'探针无区分度'；"
            "换更强探针（非线性／更多题面）读出则本条降级为'线性不可读'"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(rows),
        "null_control_identical": bool(arm["null_control"]["identical_across_episodes"]),
        "shuffled_controls_at_chance": _controls_clean(evaluations),
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "arm": report["arm_label"],
                "items": len(rows),
                "null_control_identical": report["instrument_guard"]["null_control_identical"],
                "fact_cv": {
                    name: result["fact"]["real"]["mean_accuracy"]
                    for name, result in evaluations["features"].items()
                },
                "fact_cv_margin_vs_shuffled": {
                    name: result["fact"]["above_shuffled_margin"]
                    for name, result in evaluations["features"].items()
                },
                "phrasing_cv": {
                    name: result["phrasing"]["real"]["mean_accuracy"]
                    for name, result in evaluations["features"].items()
                },
                "phrasing_cv_shuffled": {
                    name: result["phrasing"]["shuffled_label_control"]["mean_accuracy"]
                    for name, result in evaluations["features"].items()
                },
                "phrasing_cv_margin_vs_shuffled": {
                    name: result["phrasing"]["above_shuffled_margin"]
                    for name, result in evaluations["features"].items()
                },
                "length_only_control": evaluations["length_only_control"],
                "verdict": verdict,
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())