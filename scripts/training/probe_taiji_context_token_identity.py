"""语境里能不能读出"是哪个实体"——配对状态距离检验（零训练，闭掉"token 线性可读"这条未决项）。

## 为什么需要本件
`probe_taiji_context_retention_curve.py` v2 里那一格"从 `motor_context` 线性读出是哪个 token"
被**删掉**了：当时每个 (token, 距离) 组合只有 1 个样本，8 类做 4 折 CV 恒为 0
（冒烟实测 `token_cv = cv_shuffled = 0.0`），那个样本量下该量不可用。
本件用**不需要分类器**的配对检验补上这一格——样本量由"载体变异"撑起来，判据也更硬。

## 它问什么
同一个 token、同一个距离，换一个**载体**（`我叫` / `他叫` / `你叫` / `她叫` / `同学叫` / `朋友叫`）
⇒ 状态会变（载体是干扰变量）。那么：
**"同一个 token、不同载体"之间的状态差，是否**显著小于**"不同 token、同一载体"之间的状态差？**
是 ⇒ **状态里真的带着"是哪个实体"，而且这个信息比载体变异更强**；
否（两者同量级）⇒ 状态只在反映"输入字符串不同"，并没有把实体身份组织成一个可用的维度。

## 设计与统计（跑前冻结）
* 题面 = `{carrier}{token}{suffix}`；**距离＝suffix 的字节数**（载体在 token 之前，不影响距离）。
* 三个距离：**7 / 18 / 51 字节**（近场／中距／远场）。
* 状态量＝`motor_context`（读出的直接输入），相对 L1：`||a−b||₁ / ((||a||₁+||b||₁)/2)`。
* 统计量 `ratio = mean(同 token 不同载体) / mean(不同 token 同载体)`。
  载体与 token 都在变，"实体身份比载体强" ⇒ `ratio < 1`。
* **零假设＝置换检验**：把 token 标签在 token 轴上随机重排 50 次，重算 `ratio` 得到一个零分布。
  判"实体可读"＝**实测 ratio 低于零分布 3 个标准差以下**（并报出零分布均值/标准差）。
  **为什么不设绝对阈值**：绝对线（如 0.9）在不同探测面上没有可比性；置换零假设自带标度。

## 仪器有效性（两条，都过才出研究读数）
* **确定性**：同一条题面在两个全新 episode 里 ⇒ 状态**逐位相同**。
* **载体敏感性**：换载体确实改变状态（同 token 同距离、不同载体之间的相对 L1 **> 0.01**）
  ——否则"载体是干扰变量"这个前提不成立，`ratio` 就没有意义。

纪律：零训练、`learn=False`、`use_memory=False`、基座 sha 复核、判读件不覆写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

SEED = 20260928
PERMUTATIONS = 50
SIGMA_LINE = 3.0
CARRIER_SENSITIVITY_LINE = 0.01

#: 8 个双字名（F0 同一池）。
TOKENS: tuple[str, ...] = (
    "天磊",
    "静怡",
    "志远",
    "小雨",
    "建华",
    "文博",
    "海燕",
    "国强",
)

#: 载体：都在 token 之前，所以不改变"token 末字节 → 答位"的距离。
CARRIERS: tuple[str, ...] = ("我叫", "他叫", "你叫", "她叫", "同学叫", "朋友叫")

#: 三个距离（suffix 字节数）：近场／中距／远场。
SUFFIXES: tuple[str, ...] = (
    "\n我叫",
    "。我的名字是",
    "。今天天气很好，我再说一遍，我叫",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(carrier: str, token: str, suffix: str) -> str:
    return f"{carrier}{token}{suffix}"


def _observe(substrate: Any, text: str) -> None:
    substrate.reset_dynamics(episode_id="token-identity")
    substrate.observe(
        int(substrate.config.boundary_symbol),
        learn=False,
        readout="predictive",
        use_memory=False,
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _cue(substrate: Any, text: str) -> torch.Tensor:
    _observe(substrate, text)
    return substrate._state.motor_context.detach().cpu().float()


def _rel_l1(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = 0.5 * (float(left.abs().sum()) + float(right.abs().sum()))
    return float((left - right).abs().sum()) / max(denominator, 1e-9)


def collect(checkpoint: Path) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is not None:
        raise SystemExit("copy circuit mounted; identity probe requires a bare base")

    deterministic = bool(
        torch.equal(
            _cue(substrate, _prompt(CARRIERS[0], TOKENS[0], SUFFIXES[0])),
            _cue(substrate, _prompt(CARRIERS[0], TOKENS[0], SUFFIXES[0])),
        )
    )

    per_distance: list[dict[str, Any]] = []
    for suffix in SUFFIXES:
        states = [
            [_cue(substrate, _prompt(carrier, token, suffix)) for carrier in CARRIERS]
            for token in TOKENS
        ]
        token_axis = len(TOKENS)
        carrier_axis = len(CARRIERS)

        def _same(axis: list[list[torch.Tensor]]) -> float:
            values = [
                _rel_l1(axis[index][left], axis[index][right])
                for index in range(token_axis)
                for left in range(carrier_axis)
                for right in range(left + 1, carrier_axis)
            ]
            return sum(values) / len(values)

        def _diff(axis: list[list[torch.Tensor]]) -> float:
            values = [
                _rel_l1(axis[left][carrier], axis[right][carrier])
                for left in range(token_axis)
                for right in range(left + 1, token_axis)
                for carrier in range(carrier_axis)
            ]
            return sum(values) / len(values)

        def _ratio(axis: list[list[torch.Tensor]]) -> float:
            return _same(axis) / max(_diff(axis), 1e-9)

        observed = _ratio(states)
        #: 零假设＝**打散「状态 ↔ token 分组」的对应**（把 48 个状态整体重排后重新按 6 个一组分箱）。
        #: 首版写成"只重排 token 轴"，而统计量对 token 轴的置换是**不变的**（每行仍是一个 token 的
        #: 载体集合、跨行仍是全部对）⇒ 零分布退化成单点（`null_std = 0`，冒烟实测）。整表重排才对。
        flattened = [state for row in states for state in row]
        generator = torch.Generator().manual_seed(SEED)
        null: list[float] = []
        for _ in range(PERMUTATIONS):
            order = torch.randperm(len(flattened), generator=generator).tolist()
            reshuffled = [flattened[index] for index in order]
            regrouped = [
                reshuffled[index * carrier_axis : (index + 1) * carrier_axis]
                for index in range(token_axis)
            ]
            null.append(_ratio(regrouped))
        null_mean = sum(null) / len(null)
        null_std = (sum((value - null_mean) ** 2 for value in null) / len(null)) ** 0.5
        carrier_sensitivity = _same(states)

        per_distance.append(
            {
                "distance_bytes": len(suffix.encode("utf-8")),
                "suffix": suffix,
                "observed_ratio": round(observed, 4),
                "null_mean": round(null_mean, 4),
                "null_std": round(null_std, 4),
                "sigma_below_null": round((null_mean - observed) / max(null_std, 1e-9), 2),
                "carrier_sensitivity_rel_l1": round(carrier_sensitivity, 6),
                "same_token_mean_rel_l1": round(_same(states), 6),
                "diff_token_mean_rel_l1": round(_diff(states), 6),
            }
        )
    return {
        "curve": per_distance,
        "instrument": {
            "deterministic": deterministic,
            "carrier_sensitivity_line": CARRIER_SENSITIVITY_LINE,
        },
    }


def _verdict(curve: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = [entry for entry in curve if entry["sigma_below_null"] >= SIGMA_LINE]
    best = max(curve, key=lambda entry: entry["sigma_below_null"])
    return {
        "name": "token_identity_encoded" if encoded else "not_above_carrier_variation",
        "encoded_at_distances": [entry["distance_bytes"] for entry in encoded],
        "best_sigma_below_null": best["sigma_below_null"],
        "best_ratio": best["observed_ratio"],
        "reading": (
            "状态里带着「是哪个实体」，且这个信息强于载体变异" "（可在该区间的状态上做线性读出）"
            if encoded
            else "状态只反映「输入串不同」：实体身份并不比载体变异更强"
            " ⇒ 不要把「状态变了」读成「实体可读」"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    collected = collect(checkpoint)
    curve = collected["curve"]
    if not curve:
        print(json.dumps({"guard_ok": False, "error": "no rows"}, ensure_ascii=False))
        return 2
    verdict = _verdict(curve)
    instrument = collected["instrument"]
    carrier_ok = all(
        entry["carrier_sensitivity_rel_l1"] > CARRIER_SENSITIVITY_LINE for entry in curve
    )
    #: 零分布不许退化：`null_std ≈ 0` 说明零假设没打散结构（首版就是这样），判据就不成立。
    null_live = all(entry["null_std"] > 1e-6 for entry in curve)
    instrument_ok = bool(instrument["deterministic"] and carrier_ok and null_live)
    report = {
        "format": "taiji-context-token-identity-v1",
        "prereg": "本文件 docstring（配对状态距离 + 置换零假设；判据随文件一起冻结）",
        "checkpoint": args.checkpoint,
        "arm_label": args.label or args.checkpoint,
        "stimuli": {
            "tokens": list(TOKENS),
            "carriers": list(CARRIERS),
            "suffixes": [
                {"suffix": suffix, "distance_bytes": len(suffix.encode("utf-8"))}
                for suffix in SUFFIXES
            ],
            "pattern": "{carrier}{token}{suffix}",
        },
        "statistic": {
            "ratio": "mean(同 token 不同载体) / mean(不同 token 同载体)，相对 L1",
            "null": f"token 标签置换 {PERMUTATIONS} 次的零分布",
            "sigma_line": SIGMA_LINE,
        },
        "instrument": {**instrument, "carrier_sensitivity_ok": carrier_ok, "null_live": null_live},
        "curve": curve,
        "verdict": verdict,
        "dropped_or_replaced": (
            "取代 retention 曲线 v2 里被删掉的那格'从 motor_context 线性读出是哪个 token（8 类 CV）'"
            "——那格每类仅 1 样本、4 折 CV 恒 0，不可用；本件改用配对距离+置换零假设，"
            "样本量由载体变异撑起，且不需要分类器"
        ),
        "what_would_overturn": (
            "确定性或载体敏感性不过 ⇒ 仪器无效、整件作废；"
            "标量（线性）不可读不等于非线性不可读 —— 本件只否证'线性/一阶可读'这一档；"
            "换载体集合若改变量级，说明读数依赖载体选择（本件只用一套载体集，如实披露）"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(curve),
        "instrument_valid": instrument_ok,
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "arm": report["arm_label"],
                "instrument": report["instrument"],
                "verdict": verdict,
                "curve": curve,
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
