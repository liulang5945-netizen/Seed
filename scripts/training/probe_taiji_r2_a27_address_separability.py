"""A2.7-2 寻址**存在性**测试（零训练）：同形式的线性映射里，存不存在"每个字都指对"的那一个。

**为什么问这个**。复制回路给每个字节配的"身份向量"`content_embed` 是**固定随机基、永不更新**
（`taiji/copy_circuit.py` 的 `learn()` 注释里写死），会动的只有外面那层线性变换
`query_state` / `query_content`。§17 已排除了"旋钮被钳住"这一支（把前驱偏置从钳位值 2.5 放到
≈硬掩码，指对率反从 14.0% 掉到 13.1%）。剩下的问题就一个：

    现在只有 14%~16% 的步指对——是**这套参数化根本做不到**（那就是表征被钉死，
    要动内容表征），还是**做得到但没学到**（那是训练制度的问题，不用动架构）？

**为什么这题可算**。位置分数是

    score(p) = (f1 @ query_state) @ (embed[code_p] @ query_content).T / scale
             = f1 @ M @ embed[code_p] / scale,   M := query_state @ query_content.T

对 `M`（96×64＝6144 个 entry）是**线性**的，而且任意 96×64 的 `M` 都能由某对
(`query_state`,`query_content`) 分解得到 ⇒ "存不存在 M"＝一组**线性不等式**是否可行：
每一步要求"含目标字节的位置"分数严格高于"不含的位置"。

**做法（两段）**：
1. 取特征：在**已入库的 scored＋target 那条链**上（库里只留含答案那条告知，所以每一步都在
   正确事件里——把"选择"这个变量彻底摘掉，只留"定位"），逐步记下 `f1`、事件字节、目标位置集合；
2. 求解：先原样评估**训练后的那个 M**（应当复现实测指对率，这是对整套特征的锚点检查），
   再用间隔最大化（hinge＋Adam）找可达上限；若找到 100% 且间隔为正的 M，那是**构造性证明**
   （"做得到，没学到"）；若停在某个平台，报平台值与按 kind 的分层，**不把它说成不可能**
   ——平台只是"这一族优化算法在这套参数化里没找到"，不是数学上的不可分证书。

**口径写死**：本件测的是"同一参数化内的上限估计"，且约束是在**实测漂移轨迹的状态点**上取的；
把 M 换成分离版后轨迹本身会变，因此它是**必要条件**检验，不是部署保证。
判读线（跑前立，见 `SPEC-A-17` §17 末段）：上限≈现状 ⇒ 表征被钉死；上限很高（≥90%）⇒ 训练制度；
中间 ⇒ 分层报数不裁定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MAX_STEPS = 12
MARGIN_EPS = 1e-3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_steps(runtime, ids, items, tokens_of):
    """在**已入库的 scored＋target 那条链**上逐步取特征。

    链路构造整个复用 §13 那支探针（`trace_item` 加一个 `feature_sink` 口）——不在这里重写一遍
    喂历史/推进轨迹的代码。第一版自己写了一份，轨迹按"copy 自己会发什么"前进，
    与"模型实际发什么"那条链不是一回事，算出来的上限描述的是从没发生过的路径（锚点数值
    10.9% vs §13 的 16.1% 就是这么来的）。
    """
    from probe_taiji_r2_a26_emission_trace import trace_item

    substrate = runtime.model.substrate
    circuit = substrate.copy_circuit
    steps: list[dict[str, Any]] = []
    for item_id in ids:
        item = items[item_id]
        turns = [str(turn) for turn in item["turns"]]
        tokens = list(tokens_of(item))
        told = next((turn for turn in turns[:-1] if any(t in turn for t in tokens)), "")
        if not told:
            continue
        sink: list[dict[str, Any]] = []
        trace_item(
            substrate,
            circuit,
            turns=turns,
            tokens=tokens,
            told=told,
            history_mode="scored",
            store_mode="target",
            runtime=runtime,
            feature_sink=sink,
        )
        for record in sink:
            if not record["event_is_target"]:
                #: target 档库里只有一条告知，理论上不该出现；出现说明消融没做成。
                raise AssertionError(f"{item_id}: 第 {record['step']} 步不在目标事件上")
            record["item"] = item_id
            record["kind"] = item.get("kind")
            steps.append(record)
    return steps


def _rebuild_history(runtime: Any, substrate: Any, circuit: Any, turns: list[str]) -> list[Any]:
    from probe_taiji_r2_a26_emission_trace import _scored_history

    return _scored_history(runtime, turns)


def batched_steps(steps: list[dict], embed: torch.Tensor) -> dict[str, torch.Tensor]:
    """把每步摊成批次：`f1`（96）、事件字节向量矩阵 `E`（长度×64）、目标位置掩码。

    不用"目标位置 > 其余位置"的单条线性不等式——那样得替每个候选目标位置展开，
    而"指对"的定义是 **argmax 落在目标位置集合里**（对多个同字节位置来说是"存在"而非"任意"，
    是析取、不是线性约束）。所以直接按原定义批量算分数：
    `scores = (f1 @ M) @ Eᵀ`，损失用 `max_T − max_O` 的间隔 hinge。
    """
    context_dim = len(steps[0]["f1"])
    longest = max(len(step["codes"]) for step in steps)
    count = len(steps)
    f1 = torch.zeros(count, context_dim, dtype=torch.float64)
    codes = torch.zeros(count, longest, dtype=torch.long)
    mask = torch.zeros(count, longest, dtype=torch.bool)
    target = torch.zeros(count, longest, dtype=torch.bool)
    for index, step in enumerate(steps):
        f1[index] = torch.tensor(step["f1"], dtype=torch.float64)
        positions = len(step["codes"])
        codes[index, :positions] = torch.tensor(step["codes"], dtype=torch.long)
        mask[index, :positions] = True
        target[index, step["targets"]] = True
    successor = torch.zeros(count, longest, dtype=torch.float64)
    for index, step in enumerate(steps):
        previous = step.get("prev_byte")
        if previous is None:
            continue
        codes_row = step["codes"]
        for position in range(1, len(codes_row)):
            if codes_row[position - 1] == int(previous):
                successor[index, position] = 1.0
    return {
        "f1": f1,
        "codes": codes,
        "mask": mask,
        "target": target,
        "embed": embed,
        "successor": successor,
    }


def scores_of(m, batch, lam=None) -> torch.Tensor:
    """`score = f1 · M · embed[c] + λ·[前驱＝刚发出的字节]`。

    λ 必须一起算：漏了它，离线重算的"训练后 M"给 12% 而活轨迹实测 21%（冒烟时撞出来的），
    两个数对不上就等于整套离线读数没有锚点。这里既用训练值做锚点检查，也把它当**可变量**
    一起求解——λ 本来就是这套架构能学的一个参数，钉死它反而会虚报"表征不够"。
    """
    query = batch["f1"] @ m.reshape(batch["f1"].shape[1], -1)
    keys = batch["embed"][batch["codes"]]  # (步, 位置, 表征维度)
    scored = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
    if lam is not None:
        scored = scored + lam * batch["successor"]
    return scored


def perceptron(batch, start_m, passes, margin, lr):
    """集合标注感知机：只在「argmax 没落进目标位置集合」的步上更新。

    为什么单独要它：hinge＋Adam 停在多少**什么都不能说明**（自由度比约束多得多，
    谈不到「证明做不到」）。感知机的价值在反向——它一旦收敛到「全对且间隔>0」，
    那就是一个**构造性证明**：这套参数化做得到，只是没学到。
    """
    mask, target, embed = batch["mask"], batch["target"], batch["embed"]
    f1 = batch["f1"]
    codes = batch["codes"]
    m = start_m.clone().reshape(f1.shape[1], -1).to(torch.float64)
    rows_index = torch.arange(target.shape[0])

    def evaluate(current):
        scored = scores_of(current.reshape(-1), batch).masked_fill(~mask, -1e18)
        picked = scored.argmax(dim=1)
        hit = target[rows_index, picked]
        best_target = scored.masked_fill(~target, -1e18).max(dim=1)
        worst_other = scored.masked_fill(target, -1e18).max(dim=1)
        gap = (best_target.values - worst_other.values).min()
        return float(hit.to(torch.float64).mean()), float(gap), picked, scored

    best_aim, _, _, _ = evaluate(m)
    for sweep in range(passes):
        aim, gap, picked, scored = evaluate(m)
        if aim >= 1.0 and gap > margin:
            return {"aim": aim, "gap": gap, "passes_used": sweep, "converged": True}
        fails = torch.nonzero(~target[rows_index, picked]).flatten()
        if fails.numel() == 0:
            return {
                "aim": aim,
                "gap": gap,
                "passes_used": sweep,
                "converged": bool(gap > margin),
            }
        good_position = scored.masked_fill(~target, -1e18).argmax(dim=1)[fails]
        good = embed[codes[fails, good_position]]
        bad = embed[codes[fails, picked[fails]]]
        update = torch.einsum("bi,bj->bij", f1[fails], good - bad).sum(dim=0)
        norm = float(update.norm()) or 1.0
        m = m + lr * update / norm / math.sqrt(float(fails.numel()))
        best_aim = max(best_aim, aim)
    aim, gap, _, _ = evaluate(m)
    return {
        "aim": max(aim, best_aim),
        "gap": gap,
        "passes_used": passes,
        "converged": bool(aim >= 1.0 and gap > margin),
    }


def solve(batch, steps, trained_m, trained_lam, epochs, lr, margin):
    """先做**锚点检查**（训练后的 M 与 λ 应当复现实测指对率），再找可达上限。

    锚点不过＝离线重算与活轨迹不是一条链，整套上限读数作废（返回里 `anchor_ok=false` 会让退出码非零）。
    """
    mask, target = batch["mask"], batch["target"]
    flat_trained = trained_m.reshape(-1).to(torch.float64)

    def aim_rate(m: torch.Tensor, lam) -> float:
        scored = scores_of(m, batch, lam).masked_fill(~mask, -1e18)
        picked = scored.argmax(dim=1)
        hit = target[torch.arange(target.shape[0]), picked]
        return float(hit.to(torch.float64).mean())

    def min_gap(m: torch.Tensor, lam) -> float:
        scored = scores_of(m, batch, lam).masked_fill(~mask, -1e18)
        best_target = scored.masked_fill(~target, -1e18).max(dim=1).values
        best_other = scored.masked_fill(target, -1e18).max(dim=1).values
        return float((best_target - best_other).min())

    anchor_aim = aim_rate(flat_trained, float(trained_lam))
    observed = sum(1 for step in steps if step["copy_aimed"]) / len(steps)
    #: 多起点：只从训练解出发会卡在它自己的盆地里（6 题屏幕上第一版就这样停在 93%）。
    seeds = {
        "trained": flat_trained.clone(),
        "zero": torch.zeros_like(flat_trained),
        "small": torch.randn(flat_trained.numel(), dtype=torch.float64) * 1e-3,
    }
    best = {"m": flat_trained.clone(), "lam": float(trained_lam), "aim": anchor_aim, "gap": 0.0}
    for seed in seeds.values():
        m = seed.clone().requires_grad_(True)
        lam = torch.tensor(float(trained_lam), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([m, lam], lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            scored = scores_of(m, batch, lam).masked_fill(~mask, -1e18)
            best_target = scored.masked_fill(~target, -1e18).max(dim=1).values
            best_other = scored.masked_fill(target, -1e18).max(dim=1).values
            #: 间隔目标随轮数抬高：先求得"全指对"，再要求分得开。
            goal = margin * min(1.0, (epoch + 1) / max(epochs / 4.0, 1.0))
            loss = torch.relu(goal - (best_target - best_other)).pow(2).mean()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10:
                continue
            with torch.no_grad():
                current = aim_rate(m, lam)
                gap = min_gap(m.detach(), lam.detach())
            if (current, gap) > (best["aim"], best["gap"]):
                best = {
                    "m": m.detach().clone(),
                    "lam": float(lam.detach()),
                    "aim": current,
                    "gap": gap,
                }
    return {
        "anchor_aim_rate": round(anchor_aim, 4),
        "observed_aim_rate_live": round(observed, 4),
        "anchor_ok": bool(abs(anchor_aim - observed) <= 0.02),
        "trained_lam": round(float(trained_lam), 4),
        "best_lam": round(float(best["lam"]), 4),
        "best_aim_rate": round(best["aim"], 4),
        "best_min_margin": round(best["gap"], 6),
        "steps": len(steps),
        "constructively_separated": bool(best["aim"] >= 1.0 and best["gap"] > MARGIN_EPS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit", default="output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--passes", type=int, default=400, help="感知机扫描轮数")
    parser.add_argument("--perceptron-margin", type=float, default=1e-3)
    parser.add_argument("--perceptron-lr", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--margin",
        default="0.1",
        help="间隔目标，逗号分隔可多档（6144 个参数 > 几百条约束，只问“可不可分”近乎空洞，"
        "有意义的是要多大的间隔才分得开）",
    )
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 题（冒烟用）")
    parser.add_argument("--features-out", default=None)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from probe_taiji_r2_a26_emission_trace import extension_items, failing_item_ids_extension
    from score_taiji_r2_copy_strict_cap import copyable_tokens

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    ids = failing_item_ids_extension()
    if args.limit > 0:
        ids = ids[: args.limit]
    items = extension_items()

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    payload = torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    substrate.copy_circuit.load_payload(payload)
    circuit = substrate.copy_circuit

    steps = collect_steps(runtime, ids, items, copyable_tokens)
    if not steps:
        print(json.dumps({"error": "没取到任何答案步"}))
        return 1
    embed = circuit._parameters["content_embed"].detach().cpu().to(torch.float64)
    trained_m = (
        circuit._parameters["query_state"].detach().cpu().to(torch.float64)
        @ circuit._parameters["query_content"].detach().cpu().to(torch.float64).T
    )
    batch = batched_steps(steps, embed)
    #: 活轨迹里分数是 `raw/scale + bias·S`，本件不除 scale ⇒ 同一个偏置在本件单位里是
    #: `bias·sqrt(width)`。第一次冒烟锚点差 6 个点（15% vs 21%）就是漏了这个换算。
    scale = math.sqrt(float(circuit.evidence_width))
    trained_bias = float(circuit._parameters["copy_induce_bias"].detach().cpu().flatten()[0])
    trained_lam = trained_bias * scale
    margins = [float(part) for part in str(args.margin).split(",") if part.strip()]
    sweep = {
        str(target): solve(batch, steps, trained_m, trained_lam, args.epochs, args.lr, target)
        for target in margins
    }
    result = sweep[max(sweep, key=lambda key: sweep[key]["best_aim_rate"])]
    perc = perceptron(
        batch, trained_m.reshape(-1), args.passes, args.perceptron_margin, args.perceptron_lr
    )
    result["perceptron"] = perc
    #: 只有正向结果算"证明了做得到"。
    result["constructively_separated"] = bool(
        result["constructively_separated"] or perc["converged"]
    )
    observed_aim = round(sum(1 for step in steps if step["copy_aimed"]) / len(steps), 4)
    if not result["anchor_ok"]:  # 各档锚点同源，查一档即够
        #: 离线重算对不上活轨迹＝特征取错了链，后面那个"上限"没有意义。
        print(
            json.dumps(
                {"error": "锚点检查未过：离线 trained M+λ 的指对率与活轨迹不一致", "check": result}
            )
        )
        return 2

    features = args.features_out or "output/a27_address_features.json"
    feature_path = PROJECT_ROOT / features
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_text(json.dumps(steps), encoding="utf-8", newline="\n")

    report = {
        "format": "taiji-r2-a27-address-separability-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §17",
        "chain": "scored + store=target（每步都在正确事件里，只测定位）",
        "circuit": args.circuit,
        "items": len(ids),
        "answer_steps": len(steps),
        "observed_aim_rate": observed_aim,
        "note_units": (
            "分数未除 scale（同除一个正数不改 argmax），故 λ 与 margin 都是 M 空间的量；"
            f"λ = 活轨迹偏置 × scale = {trained_bias} × {round(scale, 4)}"
        ),
        "margin_sweep": sweep,
        "solve": result,
        "reading_rule": (
            "**只有正向结果可证**：constructively_separated=true ⇒ 这套参数化做得到、只是没学到"
            "（训练制度问题，不动架构）。搜索停在平台 ⇒ **本件无结论**——6144 个自由度对几百条约束，"
            "平台不构成「做不到」的证据；要判「表征被钉死」需要线性规划对偶给出的不可行证书，那是下一刀。"
            "（§17 预登记时我把「上限≈现状」当成了「表征受限」的证据——判读线本身写错了，就地更正，不改口径。）"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = Path(args.out_report or "reports/taiji_r2_a27_address_separability_20260926.json")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        from datetime import datetime

        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, default=str)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
