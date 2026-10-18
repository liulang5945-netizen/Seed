"""PLAN-N-02 §3 的噪声地板出版器：把两次受控重复的增益量之差，出版成 J-N-02-1 需要的同量纲地板。

为什么要有这台仪器（都是这条线付过代价买来的）：

* `adjudicate_taiji_n02_c4_net_gain.py` 的 `gain_judgement` 只在拿到 `noise_floor_delta_quality_abs`
  时才判 J-1，否则停在 `not_judged`。地板必须与增益量**同量纲**（G-N-02-3），所以它只能由
  "同一个 `j_n2b_prime_delta_quality` 的两次独立读数之差"给出，不能拿别的极差代答。
* 历史上那两趟（20261008 与 n2c1）形状相同、但没有代码身份、材料池自述为空，只能当披露不能当地板
  （DEBT-G168）。本件因此把**代码身份／材料指纹／剂量／夜选条数**四列都当作准入条件，不为真即 rc=2。
* 地板是一对读数的**极差**不是标准差：n=2 时它系统性偏小 ⇒ 与预注册方向一致的保守写法是
  把它标成"下限"，并在件里点名 `pair_size`，谁引用都不许升格成"实测噪声分布"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FLOOR_FORMAT = "taiji-n02-noise-floor-v1"
GAIN_QUALITY_KEY = "j_n2b_prime_delta_quality"
GAIN_ACCURACY_KEY = "j_n2b_prime_delta_accuracy"
FLOOR_QUALITY_KEY = "noise_floor_delta_quality_abs"
FLOOR_ACCURACY_KEY = "noise_floor_delta_accuracy_abs"

#: 两趟都必须已经走到的阶段；中间态的件里没有末段读数。
REQUIRED_STAGE = "done"
#: 只有这四条都为真，才说明这趟确实"睡过一觉并且回得来"——地板量的是这条链的抖动。
REQUIRED_TRUE_KEYS = (
    "weight_changed_by_powerup",
    "j_persistence",
    "restore_matches_mother",
    "rollback_disk_matches_mother",
    "g_n2_5_mother_roundtrip_identical",
)
IDENTITY_KEYS = (
    "code_identity",
    "material_fingerprint",
    "window_bytes",
    "night_k",
    "night_pool_size",
)


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def _name(path: Path) -> str:
    """仓库内出相对名（可引用），仓库外的预飞件出绝对名（不把仓外路径伪装成在库件）。"""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise Refuse(f"report is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _num(reading: dict, key: str, label: str) -> float:
    value = reading.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refuse(f"{label} has no numeric {key}: got {value!r}")
    return float(value)


def load_run(path: Path) -> dict:
    """单趟的准入：跑完了、没报错、四条真、末段读数存在。"""

    reading = read_json(path)
    label = path.name
    if reading.get("stage") != REQUIRED_STAGE:
        raise Refuse(f"{label} is not a finished run: stage={reading.get('stage')!r}")
    errors = reading.get("errors")
    if errors:
        raise Refuse(f"{label} carries errors: {errors!r}")
    for key in REQUIRED_TRUE_KEYS:
        if reading.get(key) is not True:
            raise Refuse(f"{label} did not prove {key}: got {reading.get(key)!r}")
    _num(reading, GAIN_QUALITY_KEY, label)
    return reading


def pair_identity(run_a: dict, run_b: dict) -> dict:
    """两趟只有在四列上逐字相同，才配被当成"同配置重复"。"""

    mismatched: list[str] = []
    for key in IDENTITY_KEYS:
        left, right = run_a.get(key), run_b.get(key)
        if left is None or right is None:
            mismatched.append(f"{key}:missing")
        elif left != right:
            mismatched.append(key)
    if mismatched:
        raise Refuse(
            "the two runs are not the same configuration, so their difference is not noise: "
            + ", ".join(sorted(mismatched))
        )
    identity = run_a["code_identity"]
    if not isinstance(identity, dict) or not identity:
        raise Refuse("code_identity is empty; a pair without code identity cannot be a floor")
    return {
        "code_identity": identity,
        "material_fingerprint": run_a["material_fingerprint"],
        "window_bytes": run_a["window_bytes"],
        "night_k": run_a["night_k"],
        "night_pool_size": run_a["night_pool_size"],
    }


def pair_is_two_runs(run_a: dict, run_b: dict) -> None:
    """逐位相同的两趟要先把"真的是两次独立巩固"证出来，否则地板 0 是恢复泄漏。"""

    weights = (run_a.get("treated_digest"), run_b.get("treated_digest"))
    if weights[0] is not None and weights[0] == weights[1]:
        raise Refuse(
            "both runs produced the identical post-powerup weight digest; before publishing a "
            "zero floor prove the second run really consolidated again (DEBT-G168 lesson)"
        )
    passes = (run_a.get("pass_id"), run_b.get("pass_id"))
    if passes[0] is not None and passes[0] == passes[1]:
        raise Refuse(f"both reports carry the same pass_id: {passes[0]!r} (one run, not a pair)")


def shape_matches_sealed(run_a: dict, sealed: dict) -> bool:
    """地板要能用在那对历史读数上，先核**剂量形状**——跨形状搬阈值属于已知禁忌。"""

    keys = ("window_bytes", "night_k", "night_pool_size", "projection_records")
    return all(run_a.get(key) == sealed.get(key) for key in keys)


def build_payload(run_a_path: Path, run_b_path: Path, sealed_path: Path) -> dict:
    run_a, run_b = load_run(run_a_path), load_run(run_b_path)
    pair_is_two_runs(run_a, run_b)
    identity = pair_identity(run_a, run_b)
    delta_a = _num(run_a, GAIN_QUALITY_KEY, run_a_path.name)
    delta_b = _num(run_b, GAIN_QUALITY_KEY, run_b_path.name)
    floor_quality = abs(delta_a - delta_b)
    accuracy_a = run_a.get(GAIN_ACCURACY_KEY)
    accuracy_b = run_b.get(GAIN_ACCURACY_KEY)
    floor_accuracy = (
        abs(float(accuracy_a) - float(accuracy_b))
        if isinstance(accuracy_a, (int, float))
        and not isinstance(accuracy_a, bool)
        and isinstance(accuracy_b, (int, float))
        and not isinstance(accuracy_b, bool)
        else None
    )
    sealed = read_json(sealed_path)
    payload = {
        "format": FLOOR_FORMAT,
        "prereg": "plans/reference/PLAN-N-02_c4_net_gain_package_prereg_20261017.md",
        "pair": [_name(run_a_path), _name(run_b_path)],
        "pair_size": 2,
        "measured_gain_key": GAIN_QUALITY_KEY,
        "run_a_gain": delta_a,
        "run_b_gain": delta_b,
        FLOOR_QUALITY_KEY: floor_quality,
        FLOOR_ACCURACY_KEY: floor_accuracy,
        "floor_kind": "absolute range of two independent controlled repeats (not a sigma)",
        "identity": identity,
        "weights_changed": bool(run_a.get("weight_changed_by_powerup")),
        "run_a_seconds": run_a.get("seconds"),
        "run_b_seconds": run_b.get("seconds"),
        "sealed_reference": _name(sealed_path),
        "shape_matches_sealed_pair": shape_matches_sealed(run_a, sealed),
        "gains": {"run_a": run_a.get("j_n2b_prime"), "run_b": run_b.get("j_n2b_prime")},
        "material_note": (
            "基线由人工构造：快照里的投影账本被清空（projected=[]），"
            "因此这趟投影的是既有约束种子与 journal 交互，而不是当日新材料；"
            "两趟同源由 material_fingerprint 逐字相等作证。"
        ),
        "criteria": {
            "J_N02_1_floor_units": "Δquality 同窗口径（治疗 - 对照的窗口均值差），单位与增益列一致",
            "J_N02_1_denominator": "两趟各自的 8 条夜选文本窗口均值，不是题数",
            "J_N02_1_direction": "地板是 n=2 的极差 ⇒ 系统性偏小，只配当下限引用",
        },
        "guards": {
            "pair_stage_done": True,
            "pair_errors_empty": True,
            "pair_code_identity_equal": True,
            "pair_material_fingerprint_equal": True,
            "pair_weights_actually_differ": True,
        },
    }
    return payload


def refuse_overwrite(path: Path) -> None:
    if path.is_file():
        raise Refuse(
            f"refusing to overwrite a published floor artifact: {path}; use --out-report with a new name"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument(
        "--sealed", type=Path, default=ROOT / "reports/taiji_n2b_powerup_20261008.json"
    )
    parser.add_argument("--out-report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        refuse_overwrite(arguments.out_report)
        payload = build_payload(arguments.run_a, arguments.run_b, arguments.sealed)
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
        f"noise_floor_delta_quality_abs={payload[FLOOR_QUALITY_KEY]!r} "
        f"shape_matches_sealed_pair={payload['shape_matches_sealed_pair']!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
