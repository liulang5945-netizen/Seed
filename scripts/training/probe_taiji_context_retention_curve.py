"""语境保持曲线：一个刚出现过的双字名，隔多远之后模型还会把它"顶到出口"上（零训练探针）。

**它回答的问题**：A 支线要交付的"逐字复述"，第一步是**信息得先在语境里活着**。
F0 的 T3（`我的名字是{名}。我的名字是`，名与答位相距 6 字节）已是 **0/32**；
本件把"距离"做成**扫描**，并同时给两个层级读数：

* **L1 抬升**（判机制用）：在"答："那一刻，名字首字节的概率**相对一个不在语境里的名字**
  抬高多少（对数比）。正对照＝距离 0 必须明显为正；零假设＝配对换了名字。
* **L0 命中**（只作回归）：贪心输出里有没有那个名字（§5d：L0 不得作能力/机制判据）。

**距离口径**：填隙用「的」重复 k 次，距离 d = 3k 字节（k 见 DISTANCE_FILLERS）。
距离 0 ＝ 名字紧贴答位（这是**正对照**，曲线若连这里都抬不起来，说明问题不在"保持多久"
而在"根本进不到出口"）。

**判读（跑前冻结）**
* **正对照**：`d=0` 的平均 L1 抬升 **≥ ln(2) ≈ 0.6931**（即该字节的几率至少翻倍）。
  不满足 ⇒ **整件降级为"探针无区分度"**，不报曲线结论。
* **保持**：把"抬升 ≥ ln(2)"记为该距离**活着**；报出**最后一个活着的距离**（保持半径）。
* 如实报：L0 命中率随距离的数（预期很低——F0 已示 0/32），**不用它下机制结论**。

**与既有件的关系**：`C3` 那份保持曲线量的是**情节场**（8 条绑定、恒 1.0、无饱和）；
本件量的是**读出的直接输入**（递归痕迹，96 维）——两者不是一个东西，别互相引用。

纪律：零训练、`learn=False`、`use_memory=False`、贪心、基座 sha 复核、判读件不覆写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: 距离扫描：填隙「的」的重复次数 k，距离 d = 3k 字节。
DISTANCE_FILLERS: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32)
FILLER = "的"
POSITIVE_CONTROL_LINE = math.log(2.0)
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt(token: str, fillers: int) -> str:
    return f"我的名字是{token}。{FILLER * fillers}我的名字是"


def _observe_prompt(substrate: Any, text: str) -> None:
    substrate.reset_dynamics(episode_id="retention-probe")
    substrate.observe(
        int(substrate.config.boundary_symbol),
        learn=False,
        readout="predictive",
        use_memory=False,
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _answer_distribution(substrate: Any, text: str) -> torch.Tensor:
    """喂完题面后，"答："那一刻的下一字节分布（＝`state.motor_probabilities`）。"""

    _observe_prompt(substrate, text)
    return substrate._state.motor_probabilities.detach().cpu().float()


def _greedy(substrate: Any, text: str) -> bytes:
    return substrate.generate(
        text.encode("utf-8"),
        MAX_ANSWER_BYTES,
        stop_at_boundary=True,
        sample=False,
        use_memory=False,
    )


def collect_rows(checkpoint: Path) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is not None:
        raise SystemExit("copy circuit mounted; retention probe requires a bare base")

    #: 零假设对照前置：同题面两 episode 的分布必须逐位相同。
    repeated = [
        _answer_distribution(substrate, _prompt(TOKENS[0], 2)) for _ in range(2)
    ]
    null_identical = bool(torch.equal(repeated[0], repeated[1]))

    rows: list[dict[str, Any]] = []
    for fillers in DISTANCE_FILLERS:
        for index, token in enumerate(TOKENS):
            control_token = TOKENS[(index + 1) % len(TOKENS)]
            first_byte = int(token.encode("utf-8")[0])
            own = _answer_distribution(substrate, _prompt(token, fillers))
            other = _answer_distribution(substrate, _prompt(control_token, fillers))
            output = _greedy(substrate, _prompt(token, fillers))
            rows.append(
                {
                    "distance_bytes": 3 * fillers,
                    "token": token,
                    "control_token": control_token,
                    "first_byte": first_byte,
                    "log_prob_own": float(torch.log(own[first_byte].clamp_min(1e-12))),
                    "log_prob_control": float(torch.log(other[first_byte].clamp_min(1e-12))),
                    "hit": bool(token.encode("utf-8") in output),
                    "output": output.hex(),
                }
            )
    return {"rows": rows, "null_control": {"identical_across_episodes": null_identical}}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_distance: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_distance.setdefault(row["distance_bytes"], []).append(row)
    curve: list[dict[str, Any]] = []
    for distance in sorted(by_distance):
        group = by_distance[distance]
        deltas = [row["log_prob_own"] - row["log_prob_control"] for row in group]
        curve.append(
            {
                "distance_bytes": distance,
                "items": len(group),
                "mean_log_ratio": round(sum(deltas) / len(deltas), 4),
                "min_log_ratio": round(min(deltas), 4),
                "alive_rate": round(
                    sum(1 for value in deltas if value >= POSITIVE_CONTROL_LINE) / len(deltas), 4
                ),
                "l0_hit_rate": round(sum(1 for row in group if row["hit"]) / len(group), 4),
            }
        )
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    collected = collect_rows(checkpoint)
    rows = collected["rows"]
    if not rows:
        print(json.dumps({"guard_ok": False, "error": "no rows"}, ensure_ascii=False))
        return 2
    curve = summarize(rows)
    control = next(entry for entry in curve if entry["distance_bytes"] == 0)
    positive_control_ok = bool(control["mean_log_ratio"] >= POSITIVE_CONTROL_LINE)
    alive = [entry["distance_bytes"] for entry in curve if entry["mean_log_ratio"] >= POSITIVE_CONTROL_LINE]
    verdict = (
        "retention_curve_measured"
        if positive_control_ok
        else "void_positive_control_failed_probe_not_discriminating"
    )
    report = {
        "format": "taiji-context-retention-curve-v1",
        "prereg": "本文件 docstring（判据与距离口径随文件一起冻结）",
        "checkpoint": args.checkpoint,
        "arm_label": args.label or args.checkpoint,
        "stimuli": {
            "tokens": list(TOKENS),
            "pattern": "我的名字是{token}。{的×k}我的名字是",
            "distances_bytes": [3 * k for k in DISTANCE_FILLERS],
            "control": "同一距离上，把名字换成另一个 token，量同一个首字节的对数比",
        },
        "positive_control_line": POSITIVE_CONTROL_LINE,
        "null_control": collected["null_control"],
        "curve": curve,
        "verdict": verdict,
        "farthest_alive_distance_bytes": max(alive) if alive else None,
        "what_would_overturn": (
            "零假设对照不在位 ⇒ 整件作废；正对照（d=0 抬升 ≥ ln2）不过 ⇒ 不报曲线结论；"
            "换非贪心/经语言器官通道另计；"
            "本件只量'读出的直接输入'这条链，与 C3 的情节场保持曲线不是一个东西"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(rows),
        "null_control_identical": bool(collected["null_control"]["identical_across_episodes"]),
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values()) and positive_control_ok
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "arm": report["arm_label"],
                "positive_control_ok": positive_control_ok,
                "verdict": verdict,
                "curve": curve,
                "farthest_alive": report["farthest_alive_distance_bytes"],
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