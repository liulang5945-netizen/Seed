"""把"同一材料基线上的两次独立重复"出版成**可复现性件**，而不是噪声地板件。

为什么单独立一台（2026-10-18 实测逼出来的）：

* `emit_taiji_n02_noise_floor_from_pair.py` 在"两趟末段权重摘要逐位相同"时**拒绝**出版地板，
  那条守卫是对的：极差 0 说的是"这条链是确定的"，不是"这条统计量没有噪声"。
* 但拒绝之后那对读数仍然要出版，否则我就花了两趟机时却只留下一条 REFUSE。本件负责把它出版成
  **"确定性证据"**：哪些列逐位相同、哪些列确实不同（pass_id／时刻／墙钟／存档字节），
  并据此明写 `floor_informative_for_j_n02_1=false` 与"下一张该跑的面"。
* 历史两趟的极差由本件**现算**（不手抄），并同时出版"那两趟没有代码身份、也没有材料指纹"这一列，
  所以它是材料维抖动的**参考值**，不是可代答 J-1 的地板（DEBT-G168）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPEAT_FORMAT = "taiji-n02-same-material-repeat-v1"

#: 判"确定"要看的列：末段权重摘要＋治疗窗／整件读数／增益量。任一动了就不是零极差。
DIGEST_KEYS = ("mother_digest", "treated_digest", "candidate_digest_after_load")
READOUT_KEYS = (
    "material_treated_window",
    "material_control_window",
    "material_treated_whole",
    "material_control_whole",
    "surprise_modulation",
)
GAIN_KEYS = ("j_n2b_prime_delta_quality", "j_n2b_prime_delta_accuracy", "whole_text_delta_quality")
#: 判"两趟真的是两趟"要看的列：这几列必须**不同**，否则本件测的是一次跑的两份拷贝。
MUST_DIFFER_KEYS = ("pass_id", "started_utc", "seconds")


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def _floor_module():
    """复用地板仪的准入函数，不在新仪器里重抄一遍校验链。"""

    path = ROOT / "scripts" / "training" / "emit_taiji_n02_noise_floor_from_pair.py"
    spec = importlib.util.spec_from_file_location("n02_noise_floor_reused", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise Refuse(f"report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_payload(run_a_path: Path, run_b_path: Path, historical: list[Path]) -> dict:
    floor = _floor_module()
    try:
        run_a, run_b = floor.load_run(run_a_path), floor.load_run(run_b_path)
        identity = floor.pair_identity(run_a, run_b)
    except floor.Refuse as exc:
        # 复用仪的拒绝要转成本仪的拒绝类型，否则 CLI 收到的是 traceback（rc=1）而不是 rc=2。
        raise Refuse(str(exc)) from exc

    equal = {
        key: run_a.get(key) == run_b.get(key) for key in DIGEST_KEYS + READOUT_KEYS + GAIN_KEYS
    }
    differed = {key: run_a.get(key) != run_b.get(key) for key in MUST_DIFFER_KEYS}
    if not all(differed.values()):
        raise Refuse(
            "the two reports are not two runs: "
            + ", ".join(sorted(key for key, value in differed.items() if not value))
            + " agree verbatim"
        )
    deterministic = all(equal[key] for key in DIGEST_KEYS + GAIN_KEYS)

    delta_a = float(run_a["j_n2b_prime_delta_quality"])
    delta_b = float(run_b["j_n2b_prime_delta_quality"])
    spread = abs(delta_a - delta_b)

    archived = []
    for key in ("mother_saved_path", "candidate_saved_path", "rollback_saved_path"):
        left, right = run_a.get(key), run_b.get(key)
        if left and right and (ROOT / left).is_file() and (ROOT / right).is_file():
            archived.append(
                {"key": key, "bytes_identical": _sha(ROOT / left) == _sha(ROOT / right)}
            )

    history: list[dict[str, Any]] = []
    for path in historical:
        reading = read_json(path)
        history.append(
            {
                "report": path.name,
                "delta_quality": reading.get("j_n2b_prime_delta_quality"),
                "code_identified": isinstance(reading.get("code_identity"), dict),
                "material_fingerprinted": isinstance(reading.get("material_fingerprint"), dict),
                "by_source": reading.get("by_source"),
            }
        )
    historical_values = [
        item["delta_quality"] for item in history if isinstance(item["delta_quality"], (int, float))
    ]
    historical_spread = (
        abs(max(historical_values) - min(historical_values))
        if len(historical_values) >= 2
        else None
    )

    payload = {
        "format": REPEAT_FORMAT,
        "prereg": "plans/reference/PLAN-N-02_c4_net_gain_package_prereg_20261017.md",
        "pair": [floor.display_name(run_a_path), floor.display_name(run_b_path)],
        "material_snapshot": "hand-authored baseline with projected=[] (the pile was re-projected, not new)",
        "identity": identity,
        "seconds": {"run_a": run_a.get("seconds"), "run_b": run_b.get("seconds")},
        "pass_ids": {"run_a": run_a.get("pass_id"), "run_b": run_b.get("pass_id")},
        "started_utc": {"run_a": run_a.get("started_utc"), "run_b": run_b.get("started_utc")},
        "proves_two_runs": differed,
        "identical_columns": equal,
        "chain_is_deterministic_on_this_material": deterministic,
        "same_material_repeat_floor_delta_quality_abs": spread,
        "floor_informative_for_j_n02_1": bool(not deterministic),
        "degenerate_reason": (
            "同材料、同起始权重、同剂量下这条链给出逐位相同的权重摘要与读数 ⇒ 极差 0 证的是确定性，"
            "不是'抖动很小'。J-N-02-1 要的地板必须来自**材料维**重复。"
            if deterministic
            else None
        ),
        "required_next_face": "same code identity + same mother weight + two different material baselines",
        "archived_bytes": archived,
        "archive_note": (
            "存档 `.pt` 文件字节可以不同（信封里有 pass_id 与时刻）；判定确定性只看张量摘要列，"
            "文件字节差不能反证'两趟不同'。"
        ),
        "historical_reference": {
            "reports": history,
            "delta_quality_spread": historical_spread,
            "admissible_as_j_n02_1_floor": False,
            "why_not": "这两趟既无代码身份也无材料指纹，极差里混着版本差与材料差，无法归因（DEBT-G168）",
        },
        "guards": {
            "pair_is_two_runs": True,
            "pair_identity_equal": True,
            "pair_stage_done_and_error_free": True,
        },
    }
    return payload


def refuse_overwrite(path: Path) -> None:
    if path.is_file():
        raise Refuse(
            f"refusing to overwrite a published artifact: {path}; choose a new --out-report"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--historical", type=Path, action="append", default=[])
    parser.add_argument("--out-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        refuse_overwrite(arguments.out_report)
        payload = build_payload(arguments.run_a, arguments.run_b, list(arguments.historical))
    except Refuse as exc:
        print(f"REFUSE: {exc}")
        return 2
    arguments.out_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.out_report.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"chain_is_deterministic_on_this_material={payload['chain_is_deterministic_on_this_material']} "
        f"floor_informative_for_j_n02_1={payload['floor_informative_for_j_n02_1']} "
        f"historical_delta_quality_spread={payload['historical_reference']['delta_quality_spread']!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
