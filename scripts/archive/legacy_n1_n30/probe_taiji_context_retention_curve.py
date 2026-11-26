"""语境保持曲线 v2：一个刚出现过的双字名，隔多远还"活着"；出口对它是"有反应"还是"会复述"。

> v1（同日）作废，原因＝**距离口径画错**：它把距离定义成填隙长度，而题面骨架 `。我的名字是`
> 本身还隔 18 字节 ⇒ 标着 `d=0` 的那一点实为 24 字节，**近场一个点都没有**。
> v2 把距离改成**由程序按 suffix 字节数算**（骨架与提示词都算进距离），并把
> 「仪器有效性」与「研究读数」**分开**：前者不过则整件作废，后者无论结果如何都如实报
> ——不再出现"正对照一挂就什么都报不出"。

## 它回答的问题（A 支线"告知→复述"的第一跳）

F0 的 T3（名字距答位 18 字节、即时复述）是 **0/32**。本件把距离做成扫描，并对每个距离**同时**量三件事，
因为"有反应"与"会复述"是两回事：

* **`mean_log_ratio`（会复述吗）**：答位那一刻，**名字自己的首字节**的概率，相对"换成另一个名字"
  抬高多少（对数比）。≥ ln2 记该距离**活着**。
* **`exit_kl_between_tokens`（出口对 token 有反应吗）**：同一距离上、不同 token 的出口分布之间的 KL。
  它若明显 >0，说明出口**确实随 token 变**——那么 `mean_log_ratio≈0` 的含义就升级为
  "**有反应、但不朝自己的字节去**"（＝没有复述通路），而不是"出口是常数"。
* **`cue_rel_l1_between_tokens`（状态随 token 变吗）**：同一距离上、不同 token 的 `motor_context`
  之间的相对 L1。同上，作为"状态确实随 token 变"的对照。

**已删掉的量（如实记，免得以后有人问为什么没有）**：v2 初稿曾想用"从 `motor_context` 线性读出
这是哪个 token"（8 类）来分出"状态里有没有"这一格，但**每类只有 1 个样本、4 折 CV 恒为 0**
——那个样本量下该量不可用（冒烟实测 `token_cv = cv_shuffled = 0.0`）。
"token 能否被线性读出"因此**仍是未决项**，需要另一个有幂次的设计，不在本件范围。

## 距离口径
题面 = `我叫{token}{suffix}`，`distance_bytes = len(suffix.encode())`；答案位就在题面之后，
所以这个字节数就是"token 末字节 → 答位"的**真距离**。suffix 从 7 到 66 字节，都以"我叫"收尾（复述提示）。

## 仪器有效性（两条，都过才出研究读数）
* **确定性**：同一条题面在两个全新 episode 里 ⇒ 分布**逐位相同**。
* **输入敏感性**：两条互不相关的题面 ⇒ 分布 **KL > 0.01**（证明本件读的 `motor_probabilities`
  确实随输入变，不是常数；线上取 0.01 是"明显不同的输入"的下界，不是照实测值凑的）。

纪律：零训练、`learn=False`、`use_memory=False`、贪心、基座 sha 复核、判读件不覆写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
ALIVE_LINE = math.log(2.0)
KL_SENSITIVITY_LINE = 0.01
EXIT_RESPONDS_KL_LINE = 0.01
MAX_ANSWER_BYTES = 24

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

#: 复述提示的 suffix，由短到长（真距离＝其字节数）。都以"我叫"/"我的名字是"收尾。
SUFFIXES: tuple[str, ...] = (
    "\n我叫",
    "。我叫",
    "。我的名字是",
    "，我再说一遍，我叫",
    "。我再说一遍我的名字，我叫",
    "。今天天气很好，我再说一遍，我叫",
    "。昨天夜里下了很大的雨，我再说一遍，我叫",
)

#: 输入敏感性对照：两条互不相关的题面。
SENSITIVITY_PROMPTS: tuple[str, ...] = ("今天天气不错。", "昨天夜里下了很大的雨。")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(token: str, suffix: str) -> str:
    return f"我叫{token}{suffix}"


def _observe_prompt(substrate: Any, text: str) -> None:
    substrate.reset_dynamics(episode_id="retention-v2")
    substrate.observe(
        int(substrate.config.boundary_symbol),
        learn=False,
        readout="predictive",
        use_memory=False,
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _exit_distribution(substrate: Any, text: str) -> torch.Tensor:
    _observe_prompt(substrate, text)
    return substrate._state.motor_probabilities.detach().cpu().float()


def _cue(substrate: Any, text: str) -> torch.Tensor:
    _observe_prompt(substrate, text)
    return substrate._state.motor_context.detach().cpu().float()


def _greedy(substrate: Any, text: str) -> bytes:
    return substrate.generate(
        text.encode("utf-8"),
        MAX_ANSWER_BYTES,
        stop_at_boundary=True,
        sample=False,
        use_memory=False,
    )


def _kl(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.clamp_min(1e-12)
    right = right.clamp_min(1e-12)
    return float(torch.sum(left * torch.log(left / right)))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def collect(checkpoint: Path) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is not None:
        raise SystemExit("copy circuit mounted; retention probe requires a bare base")

    #: 仪器有效性 ①：确定性。
    deterministic = bool(
        torch.equal(
            _exit_distribution(substrate, _prompt(TOKENS[0], SUFFIXES[2])),
            _exit_distribution(substrate, _prompt(TOKENS[0], SUFFIXES[2])),
        )
    )
    #: 仪器有效性 ②：输入敏感性。
    sensitivity = [_exit_distribution(substrate, text) for text in SENSITIVITY_PROMPTS]
    sensitivity_kl = round(_kl(sensitivity[0], sensitivity[1]), 6)

    curve: list[dict[str, Any]] = []
    for suffix in SUFFIXES:
        distance = len(suffix.encode("utf-8"))
        exits = [_exit_distribution(substrate, _prompt(token, suffix)) for token in TOKENS]
        cues = [_cue(substrate, _prompt(token, suffix)) for token in TOKENS]
        ratios: list[float] = []
        hits: list[bool] = []
        for index, token in enumerate(TOKENS):
            control_index = (index + 1) % len(TOKENS)
            first_byte = int(token.encode("utf-8")[0])
            ratios.append(
                float(
                    torch.log(exits[index][first_byte].clamp_min(1e-12))
                    - torch.log(exits[control_index][first_byte].clamp_min(1e-12))
                )
            )
            output = _greedy(substrate, _prompt(token, suffix))
            hits.append(bool(token.encode("utf-8") in output))
        #: 出口对 token 的反应强度（同一距离、相邻 token 两两）。
        exit_kl = [_kl(exits[i], exits[(i + 1) % len(TOKENS)]) for i in range(len(TOKENS))]
        #: 状态随 token 的变化幅度（相对 L1）。
        cue_rel = [
            float((cues[i] - cues[(i + 1) % len(TOKENS)]).abs().sum())
            / max(float(cues[i].abs().sum()), 1e-9)
            for i in range(len(TOKENS))
        ]
        curve.append(
            {
                "suffix": suffix,
                "distance_bytes": distance,
                "items": len(TOKENS),
                "mean_log_ratio": _mean(ratios),
                "min_log_ratio": round(min(ratios), 6),
                "max_log_ratio": round(max(ratios), 6),
                "alive_rate": round(
                    sum(1 for value in ratios if value >= ALIVE_LINE) / len(ratios), 4
                ),
                "l0_hit_rate": round(sum(1 for value in hits if value) / len(hits), 4),
                "exit_kl_between_tokens": _mean(exit_kl),
                "exit_responds_to_token": bool(_mean(exit_kl) > EXIT_RESPONDS_KL_LINE),
                "cue_rel_l1_between_tokens": _mean(cue_rel),
            }
        )
    return {
        "curve": curve,
        "instrument": {
            "deterministic": deterministic,
            "sensitivity_kl": sensitivity_kl,
            "sensitivity_line": KL_SENSITIVITY_LINE,
        },
    }


def _verdict(curve: list[dict[str, Any]]) -> dict[str, Any]:
    alive = [entry["distance_bytes"] for entry in curve if entry["mean_log_ratio"] >= ALIVE_LINE]
    responds = [entry for entry in curve if entry["exit_responds_to_token"]]
    if alive:
        name = "copy_path_alive"
    elif responds:
        name = "exit_responds_but_no_copy_path"
    else:
        name = "exit_insensitive_to_token"
    return {
        "name": name,
        "farthest_alive_distance_bytes": max(alive) if alive else None,
        "responds_at_distances": [entry["distance_bytes"] for entry in responds],
        "best_mean_log_ratio": round(max(entry["mean_log_ratio"] for entry in curve), 6),
        "reading": (
            "出口会把语境里的名字顶到自己的字节上（有复述通路）"
            if name == "copy_path_alive"
            else (
                "出口**对 token 有反应**（不同名字给出不同分布），但**不朝自己的字节去** ⇒ "
                "没有复述通路；配上 F0 的 T3=0/32 与 A2 电路才有命中的事实 ⇒ "
                "复述靠的是电路那条旁路，不是裸读出"
                if name == "exit_responds_but_no_copy_path"
                else "出口连 token 都不敏感 ⇒ 先怀疑语境的产出方"
            )
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
    instrument_ok = bool(
        instrument["deterministic"] and instrument["sensitivity_kl"] > KL_SENSITIVITY_LINE
    )
    report = {
        "format": "taiji-context-retention-curve-v2",
        "prereg": "本文件 docstring（距离口径、仪器有效性两条、研究读数按实测报，随文件一起冻结）",
        "checkpoint": args.checkpoint,
        "arm_label": args.label or args.checkpoint,
        "stimuli": {
            "tokens": list(TOKENS),
            "pattern": "我叫{token}{suffix}；distance_bytes = len(suffix.encode())",
            "suffixes": [
                {"suffix": suffix, "distance_bytes": len(suffix.encode("utf-8"))}
                for suffix in SUFFIXES
            ],
            "control": "同一距离上把名字换成另一个 token，量同一个首字节的对数比",
        },
        "instrument": instrument,
        "curve": curve,
        "verdict": verdict,
        "dropped_measurements": {
            "token_linear_decodability": (
                "v2 初稿的'从 motor_context 线性读出是哪个 token（8 类）'已删：每类仅 1 个样本、"
                "4 折 CV 恒 0（冒烟实测 token_cv = cv_shuffled = 0.0）⇒ 该样本量下不可用；"
                "'token 能否被线性读出'因此仍是未决项，需要另一个有幂次的设计"
            )
        },
        "what_would_overturn": (
            "确定性或输入敏感性不过 ⇒ 仪器无效、整件作废；"
            "换更强探针（非线性／有幂次）读出 token ⇒ 结论要重写；"
            "本件只量裸读出这条链（无 copy 电路、无记忆、贪心），与 C3 的情节场保持曲线不是一个东西"
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
                "instrument": instrument,
                "verdict": verdict,
                "curve": [
                    {
                        "d": entry["distance_bytes"],
                        "mean_log_ratio": entry["mean_log_ratio"],
                        "alive": entry["alive_rate"],
                        "l0_hit": entry["l0_hit_rate"],
                        "exit_kl": entry["exit_kl_between_tokens"],
                        "responds": entry["exit_responds_to_token"],
                        "cue_rel": entry["cue_rel_l1_between_tokens"],
                    }
                    for entry in curve
                ],
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
