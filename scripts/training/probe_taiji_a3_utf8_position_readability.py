"""A3-probe：内部状态对"字节序列位置"的线性可读性（零训练，L2）——架构案（3）的判决前置。

**为什么先测这个**：owner 裁定"2 之后 3"。2（解码掩码产品化）已判：产品从占位句变成
合法的字汤——**掩码修"合法"不修"成句"**。3 的问题是"字节序列的结构约束该进模型哪里"。
而选设计前必须先分清两件事，否则又会在错的层加部件：

1. **信息缺**：`motor_context`（预测下一个字节的直接输入）里线性读不出"下一个字节处在
   字符内什么位置"（ASCII 后 / 2 字节引导后 / 3 字节引导后 / 续字节中 / 4 字节引导后）
   ⇒ 结构先验必须以**输入特征**进读出（模型没有原料去学这条规则）；
2. **有而不用**：线性读得出 ⇒ 信息在场、学习/接线没利用它 ⇒ 先查读出为何没吃到，
   不该先动输入。

**任务定义**：沿真实对话语料的字节流逐步 `observe(learn=False)`（与产品/训练喂法同形），
在第 k 步记 `motor_context`，标签＝**第 k+1 个观察字节**的 UTF-8 位置类：
`cont`（0x80-0xBF 续字节）/ `ascii` / `lead2`（C2-DF）/ `lead3`（E0-EF）/ `lead4`（F0-F4）/
`illegal-lead`（C0/C1/F5-FF 等，语料里应≈0，单独计数）。
5 类 softmax 线性探针，5 折 CV，标签不均衡 ⇒ **对照三件**：
①多数类基线（无信息）；②**DFA 状态 oracle**（"还差几个续字节"的显式 one-hot——这才是
位置信息的诚实上界：rem>0 处标签必为 cont，rem=0 处取决于数据分布）；
③"上一字节 one-hot"基线（**跑前初稿把它当成了上界，这是错的**：上一字节是 cont 时
既可能序列未完也可能刚完，它不决定 rem——smoke 件 `taiji_a3_position_probe_smoke_20260927.json`
按错误对照跑出过 "present"，本件按 oracle 重判，smoke 留作执行史）。

判读线（先于跑冻结，按 oracle 对照重钉）：motor_context 探针 CV ≥ **DFA-oracle** − 0.10 ⇒
**有而不用**（2 号结论）；≤ 多数类 + 0.05 ⇒ **信息缺**（1 号结论）；中间 ⇒ 分层报数不裁定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

CLASSES = ("ascii", "lead2", "lead3", "lead4", "cont")
FOLDS = 5
EPOCHS = 300
LR = 0.05
SEED = 20260927


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def byte_class(symbol: int) -> str | None:
    if symbol < 0x80:
        return "ascii"
    if symbol <= 0xBF:
        return "cont"
    if symbol <= 0xDF:
        return "lead2" if symbol >= 0xC2 else None  # C0/C1 illegal
    if symbol <= 0xEF:
        return "lead3"
    if symbol <= 0xF4:
        return "lead4"
    return None  # F5..FF illegal lead


def stream_bytes(corpus: Path, budget: int, rng: random.Random) -> list[int]:
    """从语料随机行取字节流（跨若干文档，含 boundary 节奏）。"""
    lines: list[str] = []
    with corpus.open(encoding="utf-8") as handle:
        pool = [line.strip() for line in handle if line.strip()]
    rng.shuffle(pool)
    total = 0
    for line in pool:
        try:
            text = json.loads(line)["text"]
        except Exception:
            continue
        data = text.encode("utf-8")
        lines.append(data)
        total += len(data)
        if total >= budget:
            break
    flat: list[int] = []
    for data in lines:
        flat.append(256)  # boundary
        flat.extend(data)
    return flat[:budget]


def collect(
    corpus: Path, budget: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    from api.seed_runtime import SeedRuntime

    rng = random.Random(SEED)
    stream = stream_bytes(corpus, budget, rng)
    runtime = SeedRuntime.load(PROJECT_ROOT / "checkpoints" / "seed_beta.pt")
    substrate = runtime.model.substrate
    substrate.reset_dynamics(episode_id="a3-probe")
    feats, labels, prevs, rems = [], [], [], []
    illegal_leads = 0
    from taiji.utf8_state import advance_utf8

    rem, lead = 0, 0
    for index, symbol in enumerate(stream):
        # 先观察当前字节 ⇒ 这一步的状态正是"下一个字节要由它预测"的那一步。
        substrate.observe(
            symbol, learn=False, readout="predictive", use_memory=False, use_identity=False
        )
        # 记录观察 symbol 之后的 DFA 状态（决定 nxt 是否被迫为 cont）。
        rem_after, lead_after = advance_utf8(rem, lead, symbol) if symbol < 256 else (0, 0)
        rem, lead = rem_after, lead_after
        if index + 1 >= len(stream):
            break
        nxt = stream[index + 1]
        if nxt == 256:
            continue  # boundary 是停止符，不是一个"待预测的字节类"，也不该进非法计数
        cls = byte_class(nxt)
        if cls is None:
            illegal_leads += 1
            continue
        feats.append(substrate._state.motor_context.detach().cpu().clone().float())
        labels.append(CLASSES.index(cls))
        prevs.append(int(symbol))
        rems.append(rem)
    meta = {
        "stream_bytes": len(stream),
        "samples": len(feats),
        "illegal_lead_count": illegal_leads,
        "class_distribution": {name: 0 for name in CLASSES},
    }
    label_tensor = torch.tensor(labels, dtype=torch.long)
    for index, name in enumerate(CLASSES):
        meta["class_distribution"][name] = int((label_tensor == index).sum())
    return (
        torch.stack(feats),
        label_tensor,
        torch.tensor(prevs, dtype=torch.long),
        torch.tensor(rems, dtype=torch.long),
        meta,
    )


def _subsample(*tensors: torch.Tensor, cap: int) -> list[torch.Tensor]:
    """统一步距子样：CV 比较的是三路准确率，40k 样本对 ±0.01 足够；
    257 维 one-hot 基线在全量上跑 1500 次全批前后向是真时间坑（本件判据不需要它跑满）。"""

    n = tensors[0].shape[0]
    if n <= cap:
        return list(tensors)
    stride = -(-n // cap)  # ceil
    return [t[::stride] for t in tensors]


def _cv(features: torch.Tensor, labels: torch.Tensor, classes: int) -> float:
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(features.shape[0], generator=generator).tolist()
    accs = []
    for fold in range(FOLDS):
        test_idx = order[fold::FOLDS]
        train_idx = [i for i in order if i not in set(test_idx)]
        x_train, y_train = features[train_idx], labels[train_idx]
        x_test, y_test = features[test_idx], labels[test_idx]
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        weight = torch.zeros(x_train.shape[1], classes, requires_grad=True)
        bias = torch.zeros(classes, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=LR, weight_decay=1e-4)
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x_train @ weight + bias, y_train)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            pred = (x_test @ weight + bias).argmax(dim=1)
            accs.append(float((pred == y_test).float().mean()))
    return accs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/simple_zh/dialogue_extended_clean.jsonl")
    parser.add_argument(
        "--budget", type=int, default=60_000, help="观察字节数（逐字节 observe≈430B/s）"
    )
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    base = PROJECT_ROOT / "checkpoints/seed_beta.pt"
    sha_before = _sha256(base)
    corpus = PROJECT_ROOT / args.corpus
    features, labels, prev_bytes, rem_bytes, meta = collect(corpus, args.budget)

    # 三种对照：①DFA 状态 oracle（位置信息的诚实上界——给定 rem，nxt 是否被迫为 cont 是确定的）
    # ②多数类基线（无信息）③上一字节 one-hot（跑前初稿误当上界，保留为执行史对照）。
    oracle = torch.nn.functional.one_hot(rem_bytes, num_classes=4).float()
    onehot = torch.nn.functional.one_hot(prev_bytes, num_classes=257).float()
    majority = max(meta["class_distribution"].values()) / max(meta["samples"], 1)
    # CV 前统一步距子样到 ≤40k：判定只比三路准确率，40k 足够；
    # 免除 257 维基线在全量上 1500 次全批前后向的时间坑。
    feat_s, label_s, oracle_s, onehot_s = _subsample(features, labels, oracle, onehot, cap=40_000)
    motor_accs = _cv(feat_s, label_s, len(CLASSES))
    oracle_accs = _cv(oracle_s, label_s, len(CLASSES))
    onehot_accs = _cv(onehot_s, label_s, len(CLASSES))
    motor_mean = sum(motor_accs) / len(motor_accs)
    oracle_mean = sum(oracle_accs) / len(oracle_accs)
    onehot_mean = sum(onehot_accs) / len(onehot_accs)
    cv_samples = int(feat_s.shape[0])

    if motor_mean >= oracle_mean - 0.10:
        verdict = "information_present_not_used"
    elif motor_mean <= majority + 0.05:
        verdict = "information_missing"
    else:
        verdict = "mixed_report_only"

    report = {
        "format": "taiji-a3-utf8-position-probe-v1",
        "prereg": "owner 2026-09-27 裁定『2之后3』的第 3 步判决前置；判读线见本文件 docstring",
        "corpus": args.corpus,
        "meta": meta,
        "cv_samples": cv_samples,
        "majority_baseline": round(majority, 4),
        "motor_context_cv": [round(a, 4) for a in motor_accs],
        "motor_context_mean": round(motor_mean, 4),
        "dfa_state_oracle_mean": round(oracle_mean, 4),
        "prev_byte_onehot_cv_mean": round(onehot_mean, 4),
        "gap_oracle_minus_motor": round(oracle_mean - motor_mean, 4),
        "verdict": verdict,
        "reading": {
            "information_missing": "结构先验必须以输入特征进读出（模型没有原料学这条规则）",
            "information_present_not_used": "先查读出/学习接线为何没吃到在场信息，不动输入",
        },
        "what_would_overturn": (
            "线性探针只测一阶信息；若非线性探针可读而线性不可读，本件判 1 号属过早——"
            "但设计选择（把位置喂进输入）在该情形下仍然成立，只是归因不同"
        ),
        "base_sha256_unchanged": _sha256(base) == sha_before,
    }
    top2 = sorted(meta["class_distribution"].values(), reverse=True)[:2]
    sparse = [name for name, count in meta["class_distribution"].items() if count == 0]
    report["sparse_classes"] = sparse
    report["instrument_guard"] = {
        "samples_nonzero": meta["samples"] > 10_000,
        "primary_classes_balanced": bool(top2) and min(top2) > meta["samples"] * 0.05,
        "illegal_leads_near_zero": meta["illegal_lead_count"] < meta["samples"] * 0.001,
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
                "verdict": verdict,
                "motor_cv": report["motor_context_mean"],
                "onehot_cv": report["prev_byte_onehot_cv_mean"],
                "majority": report["majority_baseline"],
                "samples": meta["samples"],
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
