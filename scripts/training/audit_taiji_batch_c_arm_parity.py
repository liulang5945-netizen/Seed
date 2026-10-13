"""批次 C 两臂的并排取证仪（`DEBT-G127` 修法①；判据冻在 `PLAN-N5-20` §2 的 J-N5r-1／2／4）。

这块要答的只有一句话：**治疗臂与对照臂之间，除了 `generated_feed_every` 这一枚旗标，还有没有第二处不同**。
在这台仪器出数之前，任何"喂入导致了 X"的句子都不许说——冻件 §2 的原文是「两臂还差 corpus／seed／base sha／
`--fresh` 中任一项 ⇒ 差异不可归因，两臂同 0 或同非 0 都不算结论」（J-N5r-4 的"为假的情形"那半句）。

三张面分开走，各自只干自己的事：
* **配置面**（`CONFIG_FACE`）＝两臂必须**逐字段相等**的东西：基座与它的 sha、语料路径与指纹、预算、
  `skip_symbols`、`observe_kwargs`、`expected_write_surface`、`device`、`fresh`、跑器与合同件名。
  这一面里出现任何一处不等 ⇒ 本批两臂结论当场收回（rc=1），而不是把差异解释成无害。
* **旗标面**（`FLAG_FACE`）＝旗标值本身与由它**派生**出来的三枚计数；这里出的是
  J-N5r-1（对照臂关旗标即无副作用：`feeds`／`generated_write_count`／`generated_row_weight_max_abs`
  三枚全为 0）与 J-N5r-2（治疗臂长出权重才算成：写计数>0 **且** 行内绝对值最大>0）。
* **结果面**（`OUTCOME_FACE`）＝每次跑各自的东西（墙钟、速率、摘要、会话号、落盘路径……）。
  这一面**只并排出版，绝不判等**——它必然不同，判它等就是把仪器改成永远为假的那一支。

只读、零算力、不动 `taiji/`，也不重跑任何一臂；输入是两臂各自的 `run_report.json`＋`campaign_report.json`。
缺件一律响亮失败（rc=2）并把缺的那条路径点名：**"文件不在"不许被读成"差为空"**（DEBT-G58 同族）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: 与 `eval_taiji_cap0_baseline.py` 及三台普查仪同一条落盘约定（㊵-790／795／796／801）：
#: 脚本自己那层目录先上 `sys.path`，统一助手才在"被契约册按路径加载"和"起成真子进程"两种入口下都解析得到。
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _taiji_report_io as report_io  # noqa: E402

REPORT_FORMAT = "taiji-batch-c-arm-parity-v1"
PREREG = "plans/reference/PLAN-N5-20_batch_c_run_spec_and_pricing_prereg_20261012.md §2"
DEBT = "DEBT-G127"
RUNNER = "scripts/training/train_taiji_r2_readout_retrain.py"

#: 配置面：两臂必须逐字段相等。路径是点分的，缺任何一枚都算 missing_paths（不是"值为空"）。
CONFIG_FACE = (
    "trainer",
    "contract",
    "base_checkpoint",
    "base_checkpoint_sha256",
    "corpus.paths",
    "corpus.fingerprint",
    "symbols_budget",
    "device",
    "fresh",
    "observe_kwargs",
    "expected_write_surface",
    "campaign.skip_symbols",
    "skip_symbols",
    "campaign.base_checkpoint_sha256",
    "campaign.symbols_budget",
)
#: 旗标面：旗标值本身＋由它派生的计数。等与不等都在这里判，但判的是预注册的形状，不是相等。
FLAG_FACE = (
    "generated_feed.every_symbols",
    "generated_feed.feeds",
    "generated_feed.generated_write_count",
    "generated_feed.generated_row_weight_max_abs",
    "generated_feed.generated_reward_sum",
)
#: 结果面：并排出版、不判等（每次都不同的东西）。
OUTCOME_FACE = (
    "status",
    "arm_description",
    "realised_write_surface",
    "write_surface_ok",
    "symbols_consumed",
    "absolute_tick",
    "session.session_id",
    "session.wall_seconds",
    "session.seconds_per_symbol",
    "session_readout.online_accuracy",
    "session_readout.mean_surprise",
    "session_readout.scored_symbols",
    "generated_feed.measured_symbols_per_second",
    "generated_feed.eta_hours_at_requested_budget",
    "digests_before.motor",
    "digests_after.motor",
    "digests_before.predictive_readout",
    "digests_after.predictive_readout",
    "checkpoint_sha256",
    "campaign.wall_seconds",
    "campaign.status",
)

_MISSING = "<<absent>>"


def _dig(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _load_arm(arm_dir: Path) -> dict[str, Any]:
    """读一臂的两份自述；任何一份不在场都响亮失败，并把点名的路径放进异常文本。"""

    run_path = arm_dir / "run_report.json"
    campaign_path = arm_dir.parent / "campaign_report.json"
    missing = [str(path) for path in (run_path, campaign_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("arm report file(s) missing: " + ", ".join(missing))
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload["campaign"] = json.loads(campaign_path.read_text(encoding="utf-8"))
    return payload


def _side_by_side(
    treated: dict[str, Any], control: dict[str, Any], paths: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = []
    for dotted in paths:
        left = _dig(treated, dotted)
        right = _dig(control, dotted)
        rows.append(
            {
                "path": dotted,
                "treated": left,
                "control": right,
                "equal": left == right,
                "present": left is not _MISSING and right is not _MISSING,
            }
        )
    return rows


def _number(value: Any, cast) -> float | int | None:
    """读数缺位或形状不对 ⇒ 返回 None，让那一支判成假，而不是让整台仪器炸出栈。"""

    try:
        return cast(value)
    except (TypeError, ValueError):
        return None


def _flag_verdicts(treated: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    """J-N5r-1／J-N5r-2 的机检形：每支都能为假，且不为真时不许换成次要指标。"""

    feed_t = treated.get("generated_feed", {})
    feed_c = control.get("generated_feed", {})
    budget = _number(_dig(treated, "symbols_budget"), int)
    every = _number(feed_t.get("every_symbols"), int)
    feeds = _number(feed_t.get("feeds"), int)
    return {
        "j_n5r_1_control_flag_off": _number(feed_c.get("every_symbols"), int) == 0,
        "j_n5r_1_control_no_side_effects": (
            _number(feed_c.get("feeds"), int) == 0
            and _number(feed_c.get("generated_write_count"), int) == 0
            and _number(feed_c.get("generated_row_weight_max_abs"), float) == 0.0
        ),
        "j_n5r_2_treated_write_count_positive": bool(
            (_number(feed_t.get("generated_write_count"), int) or 0) > 0
        ),
        "j_n5r_2_treated_row_weight_positive": bool(
            (_number(feed_t.get("generated_row_weight_max_abs"), float) or 0.0) > 0.0
        ),
        "treated_feeds_match_budget_arithmetic": (
            budget is not None
            and every is not None
            and feeds is not None
            and every > 0
            and feeds == budget // every
        ),
        "reward_sum_zero_on_both_arms": (
            _number(feed_t.get("generated_reward_sum"), float) == 0.0
            and _number(feed_c.get("generated_reward_sum"), float) == 0.0
        ),
        "motor_digest_unchanged_in_both_arms": _dig(treated, "digests_before.motor")
        == _dig(treated, "digests_after.motor")
        and _dig(control, "digests_before.motor") == _dig(control, "digests_after.motor"),
    }


#: 跑器自述里「零奖励这件事的依据」那两枚键是 ㊵-756 才加进跑器的，而批次 C 两臂都载自更早那版
#: （按起跑时刻回查＝同一个更早 commit），所以**两臂件里都没有它们**。
#: 于是 `generated_reward_sum=0.0` 在批次 C 的产物里是一枚**没有依据自述的字面量**：
#: 它不参与归因判定（两臂同缺＝跑器版本面差，不是某一臂偷偷写了东西），但必须点名出版，
#: 否则下一批会把「件里印了 0.0」读成「件里实测过奖励为零」（DEBT-G121 同族）。
DISCLOSURE_PATHS = (
    "generated_feed.generated_reward_sum_basis",
    "generated_feed.entry_has_reward_parameter",
)


def _disclosure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "paths": list(DISCLOSURE_PATHS),
        "present_on_both_arms": [row["path"] for row in rows if row["present"]],
        "absent_on_both_arms": [row["path"] for row in rows if not row["present"]],
        "one_sided_absence": [
            row["path"]
            for row in rows
            if (row["treated"] is _MISSING) != (row["control"] is _MISSING)
        ],
        "note": (
            "这一组不参与 verdict：两臂同缺是跑器版本面差。它记的是「件里没自述」这件事本身——"
            "引用批次 C 的零奖励时，那句依据要由引用者补，不许说成「长跑件实测奖励为 0」。"
        ),
    }


def _runner_commit_at(arm_payload: dict[str, Any], runner: str) -> str:
    """这一臂实际载进内存的那版跑器是哪个 commit——按**起跑时刻**回查，不按 HEAD 现读。

    两臂可以同参数、同语料、同基座，却跑在不同版的跑器码上：㊵-756 就把跑器改了一次，而那次落在
    对照臂起跑之后。只比"今天盘上那份"会把这一支整个漏掉（[[attribution-must-name-the-file-holding-the-bytes]]
    同族；`gate-green-must-prove-it-scanned-the-diff` 的反向教训）。
    """

    session_id = str(_dig(arm_payload, "session.session_id"))
    started = session_id.split("A-", 1)[-1] if session_id.startswith("A-") else ""
    proc = subprocess.run(
        ["git", "rev-list", "-1", f"--before={started}", "HEAD", "--", runner],
        capture_output=True,
        cwd=PROJECT_ROOT,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return f"UNRESOLVED(session_id={session_id!r},rc={proc.returncode})"
    return proc.stdout.strip()


def compare(treated: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    config_rows = _side_by_side(treated, control, CONFIG_FACE)
    flag_rows = _side_by_side(treated, control, FLAG_FACE)
    outcome_rows = _side_by_side(treated, control, OUTCOME_FACE)
    disclosure_rows = _side_by_side(treated, control, DISCLOSURE_PATHS)
    face_rows = config_rows + flag_rows + outcome_rows
    #: 只把"单边缺席"当归因问题；两臂同缺＝跑器版本面差，走 disclosure 通道单独点名。
    one_sided = [
        row["path"]
        for row in face_rows
        if (row["treated"] is _MISSING) != (row["control"] is _MISSING)
    ]
    differing_config = [row["path"] for row in config_rows if not row["equal"]]
    flag_differing = [row["path"] for row in flag_rows if not row["equal"]]
    runner_commits = {
        "treated": _runner_commit_at(treated, RUNNER),
        "control": _runner_commit_at(control, RUNNER),
    }
    verdicts = {
        "config_face_identical": not differing_config,
        "no_paths_absent_from_a_single_arm": not one_sided,
        "no_disclosure_path_is_one_sided": not _disclosure(disclosure_rows)["one_sided_absence"],
        "flag_face_differs": "generated_feed.every_symbols" in flag_differing,
        "same_runner_commit_for_both_arms": (
            runner_commits["treated"] == runner_commits["control"]
            and not runner_commits["treated"].startswith("UNRESOLVED")
        ),
        **_flag_verdicts(treated, control),
    }
    return {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "debt": DEBT,
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "faces": {
            "config": config_rows,
            "flag": flag_rows,
            "outcome": outcome_rows,
        },
        "differing_config_keys": differing_config,
        "differing_flag_keys": flag_differing,
        "runner_commit_per_arm": runner_commits,
        "paths_absent_from_one_arm": one_sided,
        "disclosure": _disclosure(disclosure_rows),
        "verdicts": verdicts,
        "verdict": "holds" if all(verdicts.values()) else "not_holds",
        "failing_verdicts": sorted(name for name, ok in verdicts.items() if not ok),
        "reading_limit": (
            "这台仪器证的是**可归因性**（两臂除旗标外无第二处不同）与关旗标那一臂的无副作用，"
            "不证任何能力涨点。结果面（`status`／墙钟／摘要／`online_accuracy`／`mean_surprise`……）"
            "只并排出版、绝不判等；把它判成等就是造一支永远为假的尺。"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="批次 C 两臂并排取证：配置面判等、旗标面按 J-N5r-1/2 判级、结果面只出版（DEBT-G127 修法①）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--treated-dir",
        default=str(PROJECT_ROOT / "output" / "n5_20_batchC_treated" / "A"),
        help="治疗臂的 A 目录（含 run_report.json）",
    )
    parser.add_argument(
        "--control-dir",
        default=str(PROJECT_ROOT / "output" / "n5_20_batchC_control" / "A"),
        help="对照臂的 A 目录（含 run_report.json）",
    )
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    #: DEBT-G81 修法①：这台仪器的用法屏带中文与 ①，win32 控制台按 cp936 写得出中文但写不出 `⇒`；
    #: 守卫按仓里既有形状走（有 reconfigure 才调），摘要一律走 sys.stdout.buffer 的 ASCII 通道。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    try:
        treated = _load_arm(Path(args.treated_dir).resolve())
        control = _load_arm(Path(args.control_dir).resolve())
    except FileNotFoundError as error:
        print(f"REJECT arm reports missing: {error}", file=sys.stderr)
        return 2

    report = compare(treated, control)
    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)
    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    report_io.write_json_report(out, report, newline="\n", overwrite=args.overwrite)

    summary = {
        "verdict": report["verdict"],
        "failing": report["failing_verdicts"],
        "differing_config_keys": report["differing_config_keys"],
        "differing_flag_keys": report["differing_flag_keys"],
        "paths_absent_from_one_arm": report["paths_absent_from_one_arm"],
        "disclosure_absent_on_both": report["disclosure"]["absent_on_both_arms"],
        "runner_commits": report["runner_commit_per_arm"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
