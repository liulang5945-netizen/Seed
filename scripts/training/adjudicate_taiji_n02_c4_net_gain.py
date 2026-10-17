"""PLAN-N-02 §3／§4 的只读判读器：C4「净增益」四条合取的判级仪器。

设计约束（都是这条线付过代价买来的，写在代码里而不是写在注释里）：
* **不重抄测量链**：保持侧与来源侧的读数直接吃已封存的配对件（`taiji-n2-04-retention-pair-v1`，
  它自带 `per_column` 与 `G_N2c_3`／`G_N2c_4`），本件只做**跨件可比性**与**合取判级**。
* **判级只从本函数出**（P-门，DEBT-G153）：判读件与台账若要引用 C4 的结论，必须引用本件出版的值。
* **线值三件一起写**（P-分母，DEBT-G152）：每条判据都随件出版分子集合、分母与聚合层级。
* **先测尺再测能力**（P-尺）：增益面若没有逐端点动态范围列，整件停在 `range_unverified`，
  不许进入合取——这一条在今天的件面上**必然为真**，所以本件今天的合法产物就是那张"缺哪一列"的清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RETENTION_FORMAT = "taiji-n2-04-retention-pair-v1"
GAIN_QUALITY_KEY = "j_n2b_prime_delta_quality"
GAIN_ACCURACY_KEY = "j_n2b_prime_delta_accuracy"
FLOOR_QUALITY_KEY = "noise_floor_delta_quality_abs"
RANGE_KEY = "usable_endpoints"
COLUMNS_EXPECTED = 7

STATUS_BY_CRITERIA = {
    "range_unverified": "G-N-02-4：增益面缺动态范围列，禁止进入合取",
    "separation_unverified": "G-N-02-1／J-4：来源与保持的字节级交集未取到数",
    "not_judged": "G-N-02-3：增益量没有同量纲的噪声地板，两侧都不得代答",
    "net_gain_candidate": "J-1∧J-2∧J-3∧J-4 同枚成立",
    "cost_persists": "J-2 假：保持集有列跌幅 ≥1",
    "no_gain": "J-1 假：Δ ≤ 0",
    "not_resolved": "J-1 不可判：0 < Δ ≤ 地板",
}


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"input artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pair(path: Path) -> dict:
    """保持／来源那一侧：吃已封存的配对件，并核验它登记的四面字节没被换过（G-N-02-2）。"""
    pair = read_json(path)
    require(
        pair.get("format") == RETENTION_FORMAT,
        f"retention pair must be {RETENTION_FORMAT!r}, got {pair.get('format')!r}",
    )
    require(isinstance(pair.get("per_column"), dict), "the pair carries no per_column table")
    require(
        pair.get("prereg") == "PLAN-N2-04 §2/§3/§4",
        f"the pair is not bound to PLAN-N2-04: {pair.get('prereg')!r}",
    )
    rebind = {}
    for key, value in pair.items():
        if not key.endswith("_path") or not isinstance(value, str):
            continue
        face = Path(value)
        digest_key = key[: -len("_path")] + "_sha256"
        recorded = pair.get(digest_key)
        require(face.is_file(), f"{digest_key}: recorded face is gone: {face}")
        actual = sha(face)
        require(
            recorded == actual,
            f"{digest_key}: bytes changed under the pair ({recorded} recorded, {actual} actual)",
        )
        rebind[digest_key] = actual
    return pair


def load_gain(path: Path) -> dict:
    """新能力那一侧：只取判据点名的两枚字段，缺任一即不判。"""
    gain = read_json(path)
    for key in (GAIN_QUALITY_KEY, GAIN_ACCURACY_KEY):
        require(key in gain, f"gain artifact lacks {key!r}: {path.name}")
    return gain


def floor_quality(path: Path | None) -> float | None:
    """噪声地板必须与增益量同量纲；不同量纲的极差一律不算地板（G-N-02-3）。"""
    if path is None:
        return None
    reading = read_json(path)
    for source in (reading, reading.get("criteria", {}), reading.get("guards", {})):
        if isinstance(source, dict) and FLOOR_QUALITY_KEY in source:
            return float(source[FLOOR_QUALITY_KEY])
    return None


def range_column(path: Path | None) -> dict:
    """P-尺：增益面必须自带逐端点动态范围列；今天的件里没有，就照实出版 absent。"""
    if path is None:
        return {"state": "absent", "checked_face": None, "key": RANGE_KEY}
    reading = read_json(path)
    found = reading.get(RANGE_KEY) or reading.get("adjudication", {}).get(
        "endpoints_with_triple_errors"
    )
    if found is None:
        return {"state": "absent", "checked_face": path.name, "key": RANGE_KEY}
    return {"state": "present", "checked_face": path.name, "key": RANGE_KEY, "value": found}


def retention_judgement(pair: dict) -> dict:
    per_column = pair["per_column"]
    dropped = sorted(name for name, cell in per_column.items() if cell["delta"] < 0)
    return {
        "molecule": "逐列 after - before（不是合计，不是均值）",
        "denominator": len(per_column),
        "columns_compared": pair.get("criteria", {}).get("columns_compared"),
        "columns_expected": COLUMNS_EXPECTED,
        "dropped_columns": dropped,
        "passed": not dropped,
        "verdict": "retention_holds" if not dropped else "cost_persists",
    }


def separation_judgement(pair: dict) -> dict:
    """成功词汇按真件来：`status == "measured"` 且 `result.windows_found_in_corpus == 0` 才算过。

    我第一版把这条件写成了 `status == "ok"`，而封存件的成功状态是 `"measured"` ⇒ 守卫会惩罚它自己要测的
    那种行为。改这一格时同步加了两条负对照（`measured` 但 overlap>0／状态非 `measured`）。
    """
    guard = pair.get("guards", {}).get("G_N2c_4_disjointness", {})
    result = guard.get("result", {}) if isinstance(guard.get("result"), dict) else {}
    overlap = guard.get(
        "windows_found_in_corpus", result.get("windows_found_in_corpus", guard.get("overlap"))
    )
    measured = guard.get("status") == "measured"
    return {
        "molecule": "巩固材料与保持集的字节级交集（探针窗口命中训练语料的条数）",
        "denominator": "保持集探针窗口全集",
        "overlap": overlap,
        "windows": guard.get("retention_windows"),
        "skipped_short_rows": guard.get("skipped_short_rows"),
        "state": guard.get("status"),
        "passed": bool(measured) and overlap == 0,
    }


def comparability_judgement(pair: dict) -> dict:
    guards = pair.get("guards", {})
    cap0 = guards.get("G_N2c_3_cap0_same_source", {})
    replay = guards.get("G_N2c_3_replay_same_source", {})
    conflicts = list(cap0.get("conflicts", [])) + list(replay.get("conflicts", []))
    return {
        "molecule": "两侧是否同题集同偏移（本件只复核，不重算）",
        "denominator": "cap0 与 replay 两组守卫",
        "conflicts": conflicts,
        "passed": (not conflicts and cap0.get("status") == "ok" and replay.get("status") == "ok"),
        "rechecked_face_digests": [key for key in pair if key.endswith("_sha256")],
    }


def gain_judgement(gain: dict, floor: float | None) -> dict:
    delta_quality = float(gain[GAIN_QUALITY_KEY])
    delta_accuracy = float(gain[GAIN_ACCURACY_KEY])
    block = {
        "molecule": "治疗臂 - 对照臂 的新能力读数（Δquality 同窗口径）",
        "denominator": "与 §1① 那块面同一次运行的窗口均值，非题数",
        "delta_quality": delta_quality,
        "delta_accuracy": delta_accuracy,
        "noise_floor": floor,
    }
    if floor is None:
        block.update(
            {
                "passed": None,
                "verdict": "not_judged",
                "reason": f"地板缺失或与增益不同量纲：判读需 {FLOOR_QUALITY_KEY}",
            }
        )
        return block
    block["floor_source"] = "同代码同面的重复读数极差"
    if delta_quality <= 0:
        block.update({"passed": False, "verdict": "no_gain"})
    elif delta_quality <= floor:
        block.update({"passed": None, "verdict": "not_resolved", "reason": "0 < Δ ≤ 地板"})
    else:
        block.update({"passed": True, "verdict": "gain_holds"})
    return block


def adjudicate(pair: dict, gain: dict, floor: float | None, ruler: dict) -> dict:
    """四条合取；任一非真即按出口枚举返回，不自选、不折中。"""
    retention = retention_judgement(pair)
    separation = separation_judgement(pair)
    comparability = comparability_judgement(pair)
    gain_block = gain_judgement(gain, floor)
    if ruler["state"] != "present":
        status = "range_unverified"
    elif not separation["passed"]:
        status = "separation_unverified"
    elif gain_block["verdict"] == "not_judged":
        status = "not_judged"
    elif (
        gain_block["passed"]
        and retention["passed"]
        and comparability["passed"]
        and separation["passed"]
    ):
        status = "net_gain_candidate"
    elif not retention["passed"]:
        status = "cost_persists"
    elif gain_block["verdict"] == "no_gain":
        status = "no_gain"
    else:
        status = "not_resolved"
    return {
        "status": status,
        "exit": status,
        "criteria": {
            "J_N02_1_gain": gain_block,
            "J_N02_2_retention": retention,
            "J_N02_3_comparability": comparability,
            "J_N02_4_separation": separation,
        },
        "ruler_precondition": ruler,
        "conjunction_note": "四条同枚为真才谈『净增益机制成立』，这也是 C5 的硬前置",
        "status_meanings": STATUS_BY_CRITERIA,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retention-pair", type=Path, required=True)
    parser.add_argument("--gain", type=Path, required=True)
    parser.add_argument("--noise-floor", type=Path, default=None)
    parser.add_argument("--gain-face", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        require(
            not arguments.out_report.exists(),
            f"report target already exists (write-once): {arguments.out_report}",
        )
        pair = load_pair(arguments.retention_pair)
        gain = load_gain(arguments.gain)
        floor = floor_quality(arguments.noise_floor)
        ruler = range_column(arguments.gain_face)
        verdict = adjudicate(pair, gain, floor, ruler)
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2
    payload = {
        "format": "taiji-n02-01-c4-net-gain-v1",
        "prereg": "PLAN-N-02 §3/§4",
        "instrument": Path(__file__).name,
        "reads": {
            "retention_pair": str(arguments.retention_pair),
            "retention_pair_sha256": sha(arguments.retention_pair),
            "gain": str(arguments.gain),
            "gain_sha256": sha(arguments.gain),
            "noise_floor": None if arguments.noise_floor is None else str(arguments.noise_floor),
            "gain_face": None if arguments.gain_face is None else str(arguments.gain_face),
        },
        "models_loaded": 0,
        "parameter_updates": 0,
        "writes_any_face": False,
        "is_capability_claim": False,
        **verdict,
    }
    arguments.out_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.out_report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps({"status": payload["status"], "out": str(arguments.out_report)}, sort_keys=True)
    )
    return (
        0
        if payload["status"] in {"net_gain_candidate", "cost_persists", "no_gain", "not_resolved"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
