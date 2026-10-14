"""`PLAN-N5-23` 的地板仪：把「同配置再实现一次」的散布做成**逐指标**读数（J-N5v-2／J-N5v-3）。

存在的理由：批次 C 的幅度句被 `J-N5t-4`／`J-N5u-4` 挡着，条件写的是「需同配置重复臂」——
但「有了重复臂」不等于「有了地板」：地板只能是**两两差的实测值**，不能是我心里估的一个数。
所以这台仪器只读四份 `run_report.json`（批次 C 治疗／对照 ＋ 重复治疗／重复对照）与（可选）
两份 CAP 配对件，逐指标出版地板、观察差，以及该差能不能离开地板。

三条能红的资格判据（缺一条整件不判）：

* `all_four_arms_at_budget`——四枚件都 `status=completed` 且 `symbols_consumed == symbols_budget`；
  半路停下来的臂不配给散布定价。
* `same_configuration_within_role`——同角色的两枚件必须同基座 sha、同预算、同 `skip_symbols`、
  同 `fresh`、同 `generated_feed.every_symbols`；否则那是第四臂，不是重复。
* `floors_formed`——每个进判据的指标都真的算出了地板。
* `runner_commit_equal_across_campaigns`——`--parity-batchc`／`--parity-repeat` 两枚并排件现读的
  `runner_commit_per_arm` 必须逐角色相等。缺这一条会把 `J-N5v-1` 的红漏成 `holds`：㊵-833 实测两臂跑器
  同为 `fe39dfadef6e…`、批次 C 那枚是 `9299352d439b…`，而仪器照样给了资格齐——跑器版差正是 §2 里
  「任一项不同就不是重复」的那一项。

地板为 `0.0` 的出口是 `no_spread_unjudged` 而**不是**「散布很小」：两次逐位相同更常见的含义是
某处根本没被走到（恒等式读数不是观察）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 契约册按文件路径加载本件、也被起成真子进程，脚本自己那层目录先上 `sys.path`，
#: 统一落盘助手才解析得到（同㊵-790／802 那条例外约定）。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

#: v2 的原因：资格面多了 `runner_commit` 一条，件里多出 `runner_commits` 与那枚判据键。
#: v1 那枚已入库读数（`reports/taiji_n5_23_repeat_arm_noise_floor_20261014.json`）**不覆写**——
#: 按 `PLAN-N5-23` §6 的规矩，它只能连着并排仪的 `runner_commit_per_arm` 一起引。
REPORT_FORMAT = "taiji-n5-23-repeat-arm-noise-floor-v2"
#: 进判据的训练面指标（单位＝件里那一列的原值，不做二次归一）。
METRICS = ("cumulative_mean_surprise", "cumulative_online_accuracy")
READING_LIMIT = (
    "本件只说「同配置再实现一次的散布有多大」与「批次 C 那枚观察差能不能离开那个散布」。"
    "它不给能力分数、不给机制、不给因果：幅度句被支持时也只到「离开/未离开散布」，"
    "要把差写成主张还得点名是哪两枚件、哪张面、什么预算。"
    "`--cap-pairing-*` 指到的配对件若 `verdict` 非 `holds` 或 `pairing_valid` 非真，本件在算地板之前"
    "就拒收（rc=2、不落半件）⇒ 件里的 CAP 维地板只在两趟 CAP 配对资格齐时才存在。"
)


def _load_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run report missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "last_progress" not in payload:
        raise ValueError(f"not a run_report (no `last_progress`): {path}")
    return payload


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    progress = payload["last_progress"]
    feed = payload.get("generated_feed") or {}
    return {
        "status": payload.get("status"),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "symbols_budget": payload.get("symbols_budget", progress.get("symbols_budget")),
        "skip_symbols": payload.get("skip_symbols"),
        "fresh": payload.get("fresh"),
        "symbols_consumed": progress.get("symbols_consumed"),
        "feed_every_symbols": feed.get("every_symbols"),
    }


def _metric(payload: dict[str, Any], name: str) -> float | None:
    value = payload["last_progress"].get(name)
    return None if value is None else float(value)


def _cap_deltas(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "dimensions" not in payload:
        raise ValueError(f"not a CAP pairing report (no `dimensions`): {path}")
    #: 件里那句「两趟各自须 `pairing_valid=true`」原先只是文字——㊵-797 实测一枚
    #: `verdict=rejected`／`pairing_valid=false` 的配对件会被照读进来并出版 CAP 地板。
    #: 现在把它做成前置校验：不合格的配对不作数，而不是让 `cap` 带着无效配对成形。
    if payload.get("verdict") != "holds" or payload.get("pairing_valid") is not True:
        verdict = payload.get("verdict")
        valid = payload.get("pairing_valid")
        raise ValueError(
            f"CAP pairing report not qualified (verdict={verdict!r}, "
            f"pairing_valid={valid!r}): {path}"
        )
    return {
        str(row["dimension"]): int(row["delta"])
        for row in payload["dimensions"]
        if row.get("enters_verdict")
    }


def _floors(
    treated_c: dict[str, Any],
    control_c: dict[str, Any],
    treated_r: dict[str, Any],
    control_r: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        values = {
            key: _metric(payload, name)
            for key, payload in (
                ("batchc_treated", treated_c),
                ("batchc_control", control_c),
                ("repeat_treated", treated_r),
                ("repeat_control", control_r),
            )
        }
        if any(value is None for value in values.values()):
            out[name] = {"values": values, "verdict": "metric_absent"}
            continue
        spread_treated = abs(values["repeat_treated"] - values["batchc_treated"])
        spread_control = abs(values["repeat_control"] - values["batchc_control"])
        floor = max(spread_treated, spread_control)
        observed = abs(values["batchc_treated"] - values["batchc_control"])
        if floor == 0.0:
            verdict = "no_spread_unjudged"
        elif observed > floor:
            verdict = "magnitude_supported"
        else:
            verdict = "magnitude_blocked"
        out[name] = {
            "values": values,
            "spread_treated_role": spread_treated,
            "spread_control_role": spread_control,
            "floor": floor,
            "observed_difference_batchc": observed,
            "verdict": verdict,
            "unit": "件里那一列的原值；地板＝同角色两两差绝对值取大",
        }
    return out


def _runner_pair(batchc: Path, repeat: Path) -> dict[str, dict[str, str]]:
    """Take the per-arm runner commits from two arm-parity reports.

    A parity report whose own verdict is not ``holds`` is refused before any floor is priced:
    a failed parity means the two arms are not one comparison at all.
    """

    out: dict[str, dict[str, str]] = {}
    for label, path in (("batchc", batchc), ("repeat", repeat)):
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.is_file():
            raise FileNotFoundError(f"parity report missing: {resolved}")
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if payload.get("verdict") != "holds":
            raise ValueError(f"parity report {resolved.name} verdict is not holds")
        commits = payload.get("runner_commit_per_arm")
        if not isinstance(commits, dict) or not {"treated", "control"} <= set(commits):
            raise ValueError(f"parity report {resolved.name} lacks runner_commit_per_arm")
        if any(str(value).startswith("UNRESOLVED") for value in commits.values()):
            raise ValueError(f"parity report {resolved.name} runner commit unresolved")
        out[label] = {key: str(value) for key, value in commits.items()}
    return out


def build(
    treated_c: dict[str, Any],
    control_c: dict[str, Any],
    treated_r: dict[str, Any],
    control_r: dict[str, Any],
    cap_batchc: dict[str, int] | None,
    cap_repeat: dict[str, int] | None,
    runner: dict[str, dict[str, str]],
) -> dict[str, Any]:
    arms = {
        "batchc_treated": _config(treated_c),
        "batchc_control": _config(control_c),
        "repeat_treated": _config(treated_r),
        "repeat_control": _config(control_r),
    }
    floors = _floors(treated_c, control_c, treated_r, control_r)

    def _same(left: dict[str, Any], right: dict[str, Any]) -> bool:
        keys = (
            "base_checkpoint_sha256",
            "symbols_budget",
            "skip_symbols",
            "fresh",
            "feed_every_symbols",
        )
        return all(left[key] == right[key] for key in keys)

    at_budget = all(
        row["status"] == "completed" and row["symbols_consumed"] == row["symbols_budget"]
        for row in arms.values()
    )
    same_config = _same(arms["batchc_treated"], arms["repeat_treated"]) and _same(
        arms["batchc_control"], arms["repeat_control"]
    )
    formed = all(row.get("floor") is not None for row in floors.values())
    cap: dict[str, Any] | None = None
    if cap_batchc is not None and cap_repeat is not None:
        dimensions = sorted(set(cap_batchc) & set(cap_repeat))
        cap = {
            "dimensions": dimensions,
            "per_dimension": {
                key: {
                    "batchc_delta": cap_batchc[key],
                    "repeat_delta": cap_repeat[key],
                    "floor": abs(cap_repeat[key] - cap_batchc[key]),
                }
                for key in dimensions
            },
            "note": "CAP 的地板＝同一维度两次 `delta` 之差的绝对值；两趟各自须 `pairing_valid=true`。",
        }
    return {
        "arms": arms,
        "floors": floors,
        "cap": cap,
        "verdicts": {
            "all_four_arms_at_budget": at_budget,
            "same_configuration_within_role": same_config,
            "floors_formed": formed,
            "runner_commit_equal_across_campaigns": (
                runner["batchc"]["treated"] == runner["repeat"]["treated"]
                and runner["batchc"]["control"] == runner["repeat"]["control"]
            ),
            "any_metric_leaves_its_floor": formed
            and any(row.get("verdict") == "magnitude_supported" for row in floors.values()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="批次 C 与重复臂的逐指标噪声地板（J-N5v-2／J-N5v-3）"
    )
    parser.add_argument("--batchc-treated", type=Path, required=True)
    parser.add_argument("--batchc-control", type=Path, required=True)
    parser.add_argument("--repeat-treated", type=Path, required=True)
    parser.add_argument("--repeat-control", type=Path, required=True)
    parser.add_argument("--parity-batchc", type=Path, required=True)
    parser.add_argument("--parity-repeat", type=Path, required=True)
    parser.add_argument("--cap-pairing-batchc", type=Path, default=None)
    parser.add_argument("--cap-pairing-repeat", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    out_path = args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report
    if out_path.exists() and not args.overwrite:
        print(
            json.dumps(
                {"format": REPORT_FORMAT, "status": "refuse_overwrite", "path": str(out_path)}
            )
        )
        return 2

    try:
        runner = _runner_pair(args.parity_batchc, args.parity_repeat)
        payload = build(
            _load_run(args.batchc_treated),
            _load_run(args.batchc_control),
            _load_run(args.repeat_treated),
            _load_run(args.repeat_control),
            _cap_deltas(args.cap_pairing_batchc),
            _cap_deltas(args.cap_pairing_repeat),
            runner,
        )
    except (FileNotFoundError, ValueError) as error:
        print(json.dumps({"format": REPORT_FORMAT, "status": "refuse_input", "detail": str(error)}))
        return 2

    payload["format"] = REPORT_FORMAT
    payload["prereg"] = "plans/reference/PLAN-N5-23_repeat_arm_noise_floor_prereg_20261013.md §2"
    payload["reading_limit"] = READING_LIMIT
    payload["runner_commits"] = runner
    payload["inputs_as_given"] = [
        str(args.batchc_treated),
        str(args.batchc_control),
        str(args.repeat_treated),
        str(args.repeat_control),
        str(args.parity_batchc),
        str(args.parity_repeat),
        str(args.cap_pairing_batchc) if args.cap_pairing_batchc else None,
        str(args.cap_pairing_repeat) if args.cap_pairing_repeat else None,
    ]
    #: `failing_verdicts` 只列**资格**判据；`any_metric_leaves_its_floor` 是结果，不是资格，
    #: 把它混进失败列表会让「资格齐但读数为否」看着像「这档没跑成」。
    qualification_keys = (
        "all_four_arms_at_budget",
        "same_configuration_within_role",
        "floors_formed",
        "runner_commit_equal_across_campaigns",
    )
    payload["failing_verdicts"] = [
        key for key in qualification_keys if not payload["verdicts"][key]
    ]
    qualified = not payload["failing_verdicts"]
    payload["verdict"] = "holds" if qualified else "rejected"
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    report_io.write_text_report(out_path, text, newline="\n", overwrite=args.overwrite)
    print(text)
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
