"""C4 增益面那把尺有没有分辨率——只读两枚已封存的 N2B 件，出版 `usable_endpoints`（PLAN-N-02 §2 的 P-尺要求判增益前先量尺）。

owner 于 2026-10-18 弹窗裁 C4 选包＝①（N2 权重写入范围），发射单写在 PLAN-N-02 §6ter。
本件是那张单子的**第 4 条前置**：判读器 `adjudicate_taiji_n02_c4_net_gain.py` 的
`--gain-face` 要吃一枚带 `usable_endpoints` 的件，而今天全仓只有 N1 那条腿的仪器出版过这一列
（`verify_taiji_n1_v6_range_check.py`／`eval_taiji_n1_hard_neighbor_v7..v9.py`），
N2B 增益面**没有**——所以不补这台仪，B 档一跑就只能在"尺未证"上停住，白花 45–60 分钟。

它只回答两个问题，都不涉及能力：
* **来源通道有没有范围**：`by_source` 里几条通道真有投影记录（`usable_endpoints`＝非零通道数）；
  只有 1 条活着 ⇒ "来源与保持"这一侧跨通道不可归因。
* **双臂的分母与两列 Δ**：`material_{treated,control}_{whole,window}` 的 `items` 与
  `j_n2b_prime_delta_quality`／`delta_accuracy` 逐列原样出版，**不重新计算、不相减跨件的数**。

`models_loaded=0`、`model_queries=0`、`parameter_updates=0`、`writes_any_face=false`、
`is_capability_claim=false`；写靶一次性（`--out-report` 必填，存在即拒）。
母档是否漂移**不在本件判**：本件只出版封存件自述的 `mother_digest`／`mother_tick`，
今天的值由调用方现取（2026-10-18 实测：同 tick 273、摘要不同 ⇒ 见 08 ㊵-1115）。
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
POWERUP = ROOT / "reports/taiji_n2b_powerup_20261008_n2c1.json"
GUARD = ROOT / "reports/taiji_n2b_guard_default_20261008_n2c1.json"
FORMAT = "taiji-n02-gain-ruler-range-v1"
RANGE_KEY = "usable_endpoints"
CHANNEL_FIELDS = ("material_treated_whole", "material_control_whole")
DELTAS = ("j_n2b_prime_delta_quality", "j_n2b_prime_delta_accuracy")

#: 判级词写死在这里，读数只能落进其一；不在表内即拒（不许自造判级词）。
VERDICTS = {
    0: "no_channel_live",
    1: "single_channel_only",
    2: "two_channels_live",
}


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不落盘。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def load(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"sealed artifact missing: {path}")
    entry = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(entry, dict), f"{path.name} is not a JSON object")
    return entry


def measure(powerup: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    require(
        powerup.get("stage") == "done", f"powerup must be stage=done, got {powerup.get('stage')!r}"
    )
    require(powerup.get("fail_closed_pass") is True, "powerup fail_closed_pass must be true")
    require(
        powerup.get("g_n2_5_mother_roundtrip_identical") is True, "mother roundtrip not identical"
    )
    #: 对照臂必须真的没改权重，否则"双臂"是假的（digest 相等是它自己的守卫）。
    require(guard.get("guard_pass") is True, "guard arm did not pass")
    require(
        guard.get("digest_before") == guard.get("digest_after"),
        "guard arm changed weights: control is not a control",
    )
    channels = powerup.get("by_source")
    require(isinstance(channels, dict) and bool(channels), "powerup lacks by_source")
    counts: dict[str, int] = {}
    for name, value in channels.items():
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"by_source.{name} is not a non-negative int",
        )
        counts[str(name)] = int(value)
    live = sum(1 for value in counts.values() if value > 0)
    require(live in VERDICTS, f"live channel count {live} has no pre-declared verdict word")
    for key in DELTAS:
        require(isinstance(powerup.get(key), (int, float)), f"powerup lacks numeric {key}")
    denominators = {}
    for field in CHANNEL_FIELDS:
        block = powerup.get(field)
        require(
            isinstance(block, dict) and isinstance(block.get("items"), (int, float)),
            f"{field}.items missing",
        )
        denominators[field] = float(block["items"])
    require(
        denominators[CHANNEL_FIELDS[0]] == denominators[CHANNEL_FIELDS[1]],
        "treated and control denominators differ: numerator/denominator not the same set",
    )
    return {
        "format": FORMAT,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "powerup": {"path": str(POWERUP), "stage": powerup.get("stage")},
            "guard": {"path": str(GUARD), "guard_pass": guard.get("guard_pass")},
        },
        RANGE_KEY: live,
        "range_verdict": VERDICTS[live],
        "channels": dict(sorted(counts.items())),
        "denominator_items_per_arm": denominators,
        "deltas": {key: powerup[key] for key in DELTAS},
        "sealed_mother": {
            "digest": powerup.get("mother_digest"),
            "tick": powerup.get("mother_tick"),
        },
        "historical_delta_transferable": False,
        "transfer_blocked_because": (
            "本件只出版封存件自述的母档摘要；今天 `DEFAULT_CHECKPOINT` 用同一支 `_digest()` 现取为"
            " `3d812715bd76…`、`tick=273`（08 ㊵-1115）⇒ 同 tick 不同权重，"
            "跨母档相减不合法，Δ 只能在同一次运行的双臂之间读。"
        ),
        "reading_limit": (
            "本件只判『增益面这把尺有没有分辨率』，不判净增益、不判能力、不替 §3 的四条合取改口径；"
            "`usable_endpoints` 是给判读器 `--gain-face` 的输入列，取值定义＝非零来源通道数（存在性判据，无阈值）。"
        ),
        "models_loaded": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "writes_any_face": False,
        "is_capability_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C4 增益面的尺：只读两枚 N2B 封存件")
    parser.add_argument("--powerup-report", type=Path, default=POWERUP)
    parser.add_argument("--guard-report", type=Path, default=GUARD)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    out = arguments.out_report
    if not out.is_absolute():
        out = ROOT / out
    try:
        report = measure(load(arguments.powerup_report), load(arguments.guard_report))
        require(out.parent.is_dir(), f"write target directory missing: {out.parent}")
        require(not out.exists() or arguments.overwrite, f"report already exists: {out}")
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "usable_endpoints": report[RANGE_KEY],
                "range_verdict": report["range_verdict"],
                "deltas": report["deltas"],
                "out": str(out),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
