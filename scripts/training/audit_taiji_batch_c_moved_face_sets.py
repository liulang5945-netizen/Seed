"""批次 C 的**动面集合差**仪（`PLAN-N5-21` §6.1 第 3 条点名的载体；㊵-771 入库，出数等对照臂落盘）。

它要答的是一句 `J-N5t-2` 答不了的：把两趟**已在册**读数（基座↔治疗臂、基座↔对照臂）各自出版
`pairs[].differing_slots_names` 拿来做**集合差**——「只在治疗臂相对基座动过的那些名字」才是
喂入那条通路的候选面。比较器本身没有"面集"概念（它一次只看一对信封），所以这里只做集合运算，
**不重抄比较链**（[[dont-reimplement-the-chain]]）。

三件事必须先成立，否则件里的数没意义（都做成能红的 verdict，不是装饰）：
* `reference_is_the_same_bytes`＝两趟的**基座侧**必须是同一枚文件（`files[0].file_sha256` 相等）。
  参照系不同源的两个"差异名集"相减，得出的东西看着像定位、实际是两台面之差（[[judge-a-training-change-on-its-own-face]]）。
* `both_reports_measured`＝两趟都得是 `status=measured`；任一趟是响亮拒绝（名集不齐／镜像分叉）
  就不许进集合差——那是在拿一份不存在的读数算差。
* `localization_signal_present`＝差集非空。**注意方向**：差集为空**不许**出版成"喂入没动任何东西"，
  只出版成 `no_localization_signal`（一台只会说"有"的仪器，把"没读到"说成"没有"是它最容易犯的错，
  见 [[probe-output-must-be-verified-present]]、[[absence-prove-by-enumeration-not-guessed-name]]）。

面名归组＝**去掉存放根之后的第一段**（`substrate.memory.provenance_readout.edge_weight` → `memory`），
两个存放根（`substrate.*` 与 `taiji.kernel.*`）在同一面上会并成一枚，所以件里同时出版原始名数与归组后的面数。
只读、零算力；缺件一律 rc=2 并点名路径。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_FORMAT = "taiji-batch-c-moved-face-sets-v2"
PREREG = (
    "plans/reference/PLAN-N5-21_batch_c_attribution_prereg_20261013.md \u00a76.1 \u7b2c 3 \u6761"
)
NAME_CAP = 20
MIRROR_ROOTS = ("substrate.", "taiji.kernel.", "taiji.components.")

READING_LIMIT = (
    "本件的输入是两趟比较器读数里的**差异名集**，不是权重本身：它说得出"
    "「哪些名字只相对于基座在治疗臂里动过」，说不出幅度、也说不出因果。"
    "名字集合差里出现的每一项都仍可能是普通观察的副产物（喂入那一步多走一次 observe），"
    "所以定位句之后还要一次单变量复核；而差集为空的正确出口是 `no_localization_signal`，"
    "不是「喂入无效」。参照系那枚 sha 只证两趟用同一份基座字节，不证跑器版本（版本归并排仪）。"
)


def _load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison report missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "pairs" not in payload:
        raise ValueError(f"not a comparator report (no `pairs`): {path}")
    return payload


def _names(payload: dict[str, Any]) -> set[str]:
    """取第一趟 pair 的差异名集；多 pair 的件要点名是哪一趟，不许自己混。"""

    return set(payload["pairs"][0]["differing_slots_names"])


def _strip_root(name: str) -> str:
    for root in MIRROR_ROOTS:
        if name.startswith(root):
            return name[len(root) :]
    return name


def _face(name: str) -> str:
    """面名＝**去掉存放根之后**的第一段（`substrate.memory.association.edge_weight` → `memory`）。

    取原始点分的第二段是错的：`taiji.kernel.*` 那一根比 `substrate.*` 多一段，同一枚面会被
    数成 `memory` 与 `kernel` 两枚（本仓第一版就这么错过一次，被契约测抓住）。
    """

    stripped = _strip_root(name)
    parts = stripped.split(".")
    return parts[0] if parts and parts[0] else name


def compare(treated: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    treated_names = _names(treated)
    control_names = _names(control)
    only_treated = sorted(treated_names - control_names)

    #: 同一枚权重住在两个根里；按去根后的形状归并，才不把"一面"数成"两面"。
    def _dedup(names: set[str]) -> set[str]:
        return {_strip_root(name) for name in names}

    distinct_only_treated = sorted(_dedup(treated_names) - _dedup(control_names))
    treated_faces = Counter(_face(name) for name in treated_names)
    control_faces = Counter(_face(name) for name in control_names)
    only_treated_faces = Counter(_face(name) for name in only_treated)
    verdicts = {
        "reference_is_the_same_bytes": treated["files"][0]["file_sha256"]
        == control["files"][0]["file_sha256"],
        "both_reports_measured": treated["status"] == "measured"
        and control["status"] == "measured",
        #: 两趟必须是**同一版比较器**产出的：v3 起"名集不齐按交集比"，拿 v2 的拒绝面与 v3 的
        #: 交集面做集合差＝比两台面（v2 那份根本没出未声明面的缺席）。
        "same_instrument_version": treated.get("format") == control.get("format"),
        "localization_signal_present": bool(distinct_only_treated),
    }
    return {
        "inputs": {
            "treated_reference_sha256": treated["files"][0]["file_sha256"],
            "control_reference_sha256": control["files"][0]["file_sha256"],
            "treated_pair_label": treated["pairs"][0]["label"],
            "control_pair_label": control["pairs"][0]["label"],
            "treated_report_format": treated.get("format"),
            "control_report_format": control.get("format"),
        },
        "differing_slots": {"treated": len(treated_names), "control": len(control_names)},
        "sets": {
            "only_in_treated": len(distinct_only_treated),
            "only_in_control": len(_dedup(control_names) - _dedup(treated_names)),
            "in_both": len(_dedup(treated_names) & _dedup(control_names)),
        },
        "only_in_treated_names": distinct_only_treated[:NAME_CAP],
        "only_in_treated_names_raw": only_treated[:NAME_CAP],
        "faces": {
            "treated": dict(sorted(treated_faces.items())),
            "control": dict(sorted(control_faces.items())),
            "only_in_treated": dict(sorted(only_treated_faces.items())),
        },
        "verdicts": verdicts,
        "failing_verdicts": sorted(key for key, ok in verdicts.items() if not ok),
        "verdict": "holds" if all(verdicts.values()) else "rejected",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批次 C 两趟 base↔臂读数的动面集合差")
    parser.add_argument("--treated-report", required=True, help="base↔治疗臂的比较件")
    parser.add_argument("--control-report", required=True, help="base↔对照臂的比较件")
    parser.add_argument(
        "--out-report", required=True, help="落盘路径（件已存在则拒覆盖 rc=2，除非 --overwrite）"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    paths = []
    for raw in (args.treated_report, args.control_report):
        path = Path(raw)
        paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    try:
        treated = _load_report(paths[0])
        control = _load_report(paths[1])
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"REJECT {error}", file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "prereg": PREREG,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_as_given": [str(args.treated_report), str(args.control_report)],
        "reading_limit": READING_LIMIT,
        **compare(treated, control),
    }
    report["wall_clock_ms"] = round((time.perf_counter() - started) * 1000, 3)

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists() and not args.overwrite:
        print(f"refusing to overwrite an existing report: {out}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "verdict": report["verdict"],
        "failing_verdicts": report["failing_verdicts"],
        "differing_slots": report["differing_slots"],
        "sets": report["sets"],
        "only_in_treated_names": report["only_in_treated_names"],
        "out": out.as_posix(),
    }
    sys.stdout.buffer.write(json.dumps(summary, ensure_ascii=True).encode("ascii") + b"\n")
    return 0 if report["verdict"] == "holds" else 1


if __name__ == "__main__":
    raise SystemExit(main())
