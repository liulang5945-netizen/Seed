"""PLAN-N5-12 批次 B：VISION §10 唤醒行 J-N5l-2／J-N5l-4 的双臂取数件（零训练、不更新权重）。

测什么
------
`HomeostaticController.select_mode`（`taiji/homeostasis.py:83-91`）是纯阈值比较、无滞回。
VISION §10 给它配的三条验收里，"固定调度对照"与"噪声不得反复唤醒"今天是**形式**而不是散文：

* **J-N5l-2**＝同一条干净流与它的确定性噪声副本，自适应臂的**模式翻转次数**必须相等，
  才许说"噪声不引起反复唤醒"；
* **J-N5l-4**＝对照必须跑在**同一条真实轨迹**上（同流、同步、只换臂），两臂的模式序列必须**可分**；
  逐位相同 ⇒ `arm_not_actually_two`（双臂要先证真的是两臂），两臂都零翻转 ⇒ 尺不可用。

不新造阈值
----------
本件**没有任何判据数值阈**：噪声幅值取"该流自身相邻步差分的最大值"（流的内生尺度），
"尺有没有动态范围"沿用 `adjudicate_taiji_n5_shadow_gate.py:299` 的 `ruler_usable` 形状
（要求档位读数 **> 0** 才算有用，不是新数），判定只用"两档相等／两档相同"。
流里的步数与各档取值是**夹具参数**（决定形状），不参与任何结论。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from taiji.homeostasis import HomeostaticController, HomeostaticState  # noqa: E402

REPORT_FORMAT = "taiji-n5-12-wake-schedule-arms-v1"
PREREG = "plans/reference/PLAN-N5-12_autonomous_wake_constraints_prereg_20261011.md §3"
MODES = ("wake", "sleep", "play")
STEPS = 24  # 夹具：轨迹长度，不是判据阈


def clean_stream() -> list[tuple[float, float, float, float]]:
    """确定性输入流：误差／新颖性／资源成本同向爬升，后半段叠加负奖励。"""

    stream: list[tuple[float, float, float, float]] = []
    for step in range(STEPS):
        ramp = step / (STEPS - 1)
        reward = 0.0 if step < STEPS // 2 else -1.0
        stream.append((ramp, ramp, reward, ramp))
    return stream


def noise_units(stream: list[tuple[float, float, float, float]]) -> tuple[float, ...]:
    """噪声幅值＝**每一路信号自己的**相邻步差分值。

    第一版把四路展平后取相邻差，于是跨路跳变（`reward` 从 0.0 跳到 −1.0）被当成"流的内生尺度"，
    量出的单位是 2.0——比任何一路都粗，噪声把信号淹掉，那样的"噪声不反复唤醒"判据没有意义。
    逐路取之后每一路只按自己的粒度扰动。
    """

    columns = list(zip(*stream, strict=True))
    units: list[float] = []
    for column in columns:
        diffs = [abs(column[index] - column[index - 1]) for index in range(1, len(column))]
        units.append(max(diffs) if diffs else 0.0)
    return tuple(units)


def raw_diff_scales(stream: list[tuple[float, float, float, float]]) -> set[float]:
    """梯的真身：逐路全部**非零**相邻步差分（浮点精确相等去重，不引容差）。

    集合为空 ⇒ 流恒定，`J-N5l-6` 必须先判 `stream_degenerate` 并响亮退出——
    常值流下扰动会被每一路自己的取值范围夹死，副本＝原流，任何"通过"都是空过。
    """

    return {
        abs(column[index] - column[index - 1])
        for column in zip(*stream, strict=True)
        for index in range(1, len(column))
        if abs(column[index] - column[index - 1]) > 0.0
    }


def scale_ladder(stream: list[tuple[float, float, float, float]]) -> tuple[float, ...]:
    """`S = 非零步尺集 ∪ {已冻那一档}`，升序；成员与取值全部随读数件出版。"""

    return tuple(sorted(raw_diff_scales(stream) | {frozen_amplitude(stream)}))


def worst_case_verdicts(clean_flips: int, rows: list[dict[str, float]]) -> str:
    """J-N5l-6：只要有一档 `f(s) != f_clean` 就判不成立；全等才判通过。"""

    differs = [row for row in rows if row["flips"] != clean_flips]
    if not differs:
        return "noise_does_not_re_wake"
    worst = max(differs, key=lambda row: abs(row["flips"] - clean_flips))
    return f"noise_changes_scheduling:{clean_flips}->{int(worst['flips'])}"


def frozen_amplitude(stream: list[tuple[float, float, float, float]]) -> float:
    """已冻 J-N5l-2 原文的"幅值取该流自身的最大值"——判据只用这一档。"""

    return max(value for event in stream for value in event)


def perturb(
    stream: list[tuple[float, float, float, float]], *, seed: int, amplitude: float
) -> list[tuple[float, float, float, float]]:
    """加噪副本。回落范围＝**每一路自己在流里的取值范围**，不是硬写的 [0, 1]。

    第一版回落到 [0, 1]：`reward` 那一列的 −1.0 被改写成 0.0，于是"噪声副本"与干净流之间
    多了一条**系统性**差异（负奖励被抹平），J-N5l-2 比的就不是噪声。当时读到的 13→15 里有
    多少来自这条改写，无法归因 ⇒ 那个读数作废、按修好的取法重算（作废过程入台账 ㊵-686）。
    现在 amplitude=0 就是逐位恒等，这一支由契约测钉住。
    """

    bounds = [(min(column), max(column)) for column in zip(*stream, strict=True)]
    rng = random.Random(seed)
    perturbed: list[tuple[float, float, float, float]] = []
    for event in stream:
        row = []
        for value, (low, high) in zip(event, bounds, strict=True):
            jittered = value + rng.uniform(-amplitude, amplitude)
            row.append(max(low, min(high, jittered)))
        perturbed.append((row[0], row[1], row[2], row[3]))
    return perturbed


def _count_flips(modes: list[str]) -> int:
    return sum(1 for index in range(1, len(modes)) if modes[index] != modes[index - 1])


def run_arm(
    controller: HomeostaticController, stream: list[tuple[float, float, float, float]], *, arm: str
):
    """在一条流上走一遍；`adaptive` 每步先 `select_mode`，`fixed_wake` 永远按 wake（＝产品默认）。"""

    state = HomeostaticState(tick=0)
    modes: list[str] = []
    for prediction_error, novelty, reward, resource_cost in stream:
        mode = controller.select_mode(state) if arm == "adaptive" else "wake"
        modes.append(mode)
        state = controller.update(
            state,
            prediction_error=prediction_error,
            novelty=novelty,
            reward=reward,
            resource_cost=resource_cost,
            mode=mode,
        )
    return modes, state


def measure(*, seed: int = 7) -> dict[str, object]:
    if STEPS < 2:
        raise ValueError("轨迹太短，翻转会恒为 0（这不是尺不可用，是夹具退化）")
    controller = HomeostaticController()
    stream = clean_stream()
    units = noise_units(stream)
    amplitude = frozen_amplitude(stream)
    noise = perturb(stream, seed=seed, amplitude=amplitude)

    adaptive_clean, _ = run_arm(controller, stream, arm="adaptive")
    adaptive_noisy, _ = run_arm(controller, noise, arm="adaptive")
    fixed_clean, fixed_end = run_arm(controller, stream, arm="fixed_wake")
    adaptive_end = run_arm(controller, stream, arm="adaptive")[1]

    flips_adaptive_clean = _count_flips(adaptive_clean)
    flips_adaptive_noisy = _count_flips(adaptive_noisy)
    flips_fixed = _count_flips(fixed_clean)

    # 尺的可用性只看自适应臂：`fixed_wake` 臂按定义永远待在 wake（翻转恒 0），
    # 拿它的翻转数当动态范围会把「对照臂本应恒定」误读成「尺没有动态范围」。
    ruler_usable = flips_adaptive_clean > 0 and flips_adaptive_noisy > 0 and amplitude > 0.0
    if ruler_usable:
        j_n5l_2 = (
            "noise_does_not_re_wake"
            if flips_adaptive_noisy == flips_adaptive_clean
            else f"noise_changes_scheduling:{flips_adaptive_clean}->{flips_adaptive_noisy}"
        )
    else:
        j_n5l_2 = "ruler_unusable"
    # J-N5l-4 单独判，且不被可用性挡在前面：两臂逐位相同就是「其实只有一臂」——
    # 那正是需要被抓出来的形状（守卫必须能为假）。
    j_n5l_4 = "arm_not_actually_two" if adaptive_clean == fixed_clean else "arms_separable"
    if not raw_diff_scales(stream):
        raise ValueError(
            "stream_degenerate：这条流的逐路相邻步差分全为零，扰动副本＝原流，"
            "任何『噪声不反复唤醒』的读法都是空过（PLAN-N5-13 §2）"
        )
    ladder: list[dict[str, float]] = []
    for scale in scale_ladder(stream):
        ladder_modes, _ = run_arm(
            controller, perturb(stream, seed=seed, amplitude=scale), arm="adaptive"
        )
        ladder.append({"amplitude": scale, "flips": _count_flips(ladder_modes)})
    j_n5l_6 = worst_case_verdicts(flips_adaptive_clean, ladder)
    return {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "fixture": {
            "steps": STEPS,
            "seed": seed,
            "frozen_amplitude_from_stream_max": amplitude,
            "noise_units_from_stream": list(units),
            "noise_unit_fields": ["prediction_error", "novelty", "reward", "resource_cost"],
        },
        "noise_scale_ladder": {
            "used_for_verdict": "J-N5l-6（最坏情形）；J-N5l-2 仍只取已冻那一档 frozen_amplitude",
            "dedupe_rule": "浮点精确相等去重，不引容差——引了就是一枚新阈值（DEBT-G90 的形状）",
            "rows": ladder,
        },
        "modes": {
            "adaptive_clean": adaptive_clean,
            "adaptive_noisy": adaptive_noisy,
            "fixed_wake_clean": fixed_clean,
        },
        "flips": {
            "adaptive_clean": flips_adaptive_clean,
            "adaptive_noisy": flips_adaptive_noisy,
            "fixed_wake_clean": flips_fixed,
        },
        "end_state": {
            "adaptive": adaptive_end.__dict__,
            "fixed_wake": fixed_end.__dict__,
            "final_drives_adaptive": controller.drive(adaptive_end).__dict__,
            "final_drives_fixed_wake": controller.drive(fixed_end).__dict__,
        },
        "verdicts": {
            "ruler_usable": bool(ruler_usable),
            "j_n5l_2_noise_rewake": j_n5l_2,
            "j_n5l_6_noise_rewake_worst_case": j_n5l_6,
            "j_n5l_4_fixed_schedule_control": j_n5l_4,
        },
        "caveats": [
            "两臂的差别只发生在被选中的 mode 上；产品默认链今天等价于 fixed_wake 臂"
            '（`taiji/adapter.py:10921`／`:11660` 写死 mode="wake"）⇒ 过了可分性也只说明'
            "『自适应与默认不同』，不说明『调度有效』",
            "fixed 臂的模式翻转数**按定义为 0**（它永远 wake），所以 J-N5l-4 的『序列可分』是定义性的；"
            "有信息量的那一半在两臂的驱力终态差（`end_state`）：自适应臂会走 sleep／play 的恢复支，"
            "固定臂不会",
            "本件的流是合成夹具，不是真实回合日志；它测的是选择器的行为形状，不是能力",
            "取法自曝：第一版把四路展平取相邻差当噪声幅值，跨路跳变把幅值抬到 2.0（超出信号可表示范围，那已不是噪声而是饱和），当时的读数是 13->14；修回已冻原文那一档后重算为 13->13。"
            "⇒ 这条判据的结论对噪声幅度敏感，缺陷已入册（PLAN-N5-12 §9）：不改判据口径，只把两档一起披露",
            "J-N5l-1（选择是否参与真实回合）由 walked-ledger 出值＝`scheduler_not_in_loop`，不在本件重测",
        ],
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="噪声副本的确定性种子（夹具参数）")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_n5_12_wake_schedule_arms_first_values_20261011.json",
    )
    parser.add_argument("--overwrite", action="store_true", help="显式允许覆写已存在的报告件")
    args = parser.parse_args()
    report = measure(seed=args.seed)
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({"verdicts": report["verdicts"], "flips": report["flips"]}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
