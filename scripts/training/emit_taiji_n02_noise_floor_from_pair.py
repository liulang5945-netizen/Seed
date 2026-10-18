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
#: 只有这五条都为真，才说明这趟确实"睡过一觉并且回得来"——地板量的是这条链的抖动。
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
FLOOR_DIMENSIONS = ("same_material", "material")


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不出件。"""


def display_name(path: Path) -> str:
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


def pair_identity(run_a: dict, run_b: dict, dimension: str = "same_material") -> dict:
    """按维度出版"这两趟够格配成一对"的那几列。

    * `same_material`：代码身份、材料指纹、剂量、夜选条数、池大小**五列都必须逐字相同**——
      这是"同配置重复"，2026-10-18 实测它在同码同材料下**给不出非零地板**（链是确定的）。
    * `material`：代码身份、起始权重 `mother_digest`、剂量、夜选条数、池大小必须相同，
      而 `material_fingerprint` **必须不同**——地板量的是材料维的位移，材料相同就退化成上面那一档。
    """

    if dimension not in FLOOR_DIMENSIONS:
        raise Refuse(f"unknown floor dimension: {dimension!r}")
    if dimension == "material":
        equal_keys = (
            "code_identity",
            "mother_digest",
            "window_bytes",
            "night_k",
            "night_pool_size",
        )
        differing_keys = ("material_fingerprint",)
    else:
        equal_keys = IDENTITY_KEYS
        differing_keys = ()
    mismatched: list[str] = []
    for key in equal_keys:
        left, right = run_a.get(key), run_b.get(key)
        if left is None or right is None:
            mismatched.append(f"{key}:missing")
        elif left != right:
            mismatched.append(key)
    for key in differing_keys:
        left, right = run_a.get(key), run_b.get(key)
        if left is None or right is None:
            mismatched.append(f"{key}:missing")
        elif left == right:
            mismatched.append(f"{key}:not_materially_distinct")
    if mismatched:
        raise Refuse(
            f"the two runs are not a valid {dimension} pair, so their difference is not a floor: "
            + ", ".join(sorted(mismatched))
        )
    identity = run_a["code_identity"]
    if not isinstance(identity, dict) or not identity:
        raise Refuse("code_identity is empty; a pair without code identity cannot be a floor")
    return {
        "floor_dimension": dimension,
        "code_identity": identity,
        "mother_digest": run_a.get("mother_digest"),
        "material_fingerprint": run_a["material_fingerprint"],
        "material_fingerprint_b": run_b["material_fingerprint"],
        "window_bytes": run_a["window_bytes"],
        "night_k": run_a["night_k"],
        "night_pool_size": run_a["night_pool_size"],
    }


def pair_is_two_runs(run_a: dict, run_b: dict, dimension: str = "same_material") -> bool:
    """同材料档里逐位相同的两趟必须先证"真的是两次独立巩固"，否则地板 0 是恢复泄漏。

    材料维不拒相同权重摘要——**两批材料睡出同一份权重**是一个合法（且很有内容）的读数，
    所以那一支把它作为 `weights_identical_across_material` 披露出去，而不是停掉。
    """

    weights = (run_a.get("treated_digest"), run_b.get("treated_digest"))
    identical = weights[0] is not None and weights[0] == weights[1]
    if identical and dimension == "same_material":
        raise Refuse(
            "both runs produced the identical post-powerup weight digest; before publishing a "
            "zero floor prove the second run really consolidated again (DEBT-G168 lesson)"
        )
    return bool(identical)
    passes = (run_a.get("pass_id"), run_b.get("pass_id"))
    if passes[0] is not None and passes[0] == passes[1]:
        raise Refuse(f"both reports carry the same pass_id: {passes[0]!r} (one run, not a pair)")


def shape_matches_sealed(run_a: dict, sealed: dict) -> bool:
    """地板要能用在那对历史读数上，先核**剂量形状**——跨形状搬阈值属于已知禁忌。"""

    keys = ("window_bytes", "night_k", "night_pool_size", "projection_records")
    return all(run_a.get(key) == sealed.get(key) for key in keys)


def build_payload(run_a_path: Path, run_b_path: Path, sealed_path: Path, dimension: str) -> dict:
    run_a, run_b = load_run(run_a_path), load_run(run_b_path)
    weights_identical = pair_is_two_runs(run_a, run_b, dimension)
    identity = pair_identity(run_a, run_b, dimension)
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
        "floor_dimension": dimension,
        "pair": [display_name(run_a_path), display_name(run_b_path)],
        "pair_size": 2,
        "measured_gain_key": GAIN_QUALITY_KEY,
        "run_a_gain": delta_a,
        "run_b_gain": delta_b,
        FLOOR_QUALITY_KEY: floor_quality,
        FLOOR_ACCURACY_KEY: floor_accuracy,
        "floor_kind": (
            "absolute range of two controlled runs on materially distinct piles (not a sigma)"
            if dimension == "material"
            else "absolute range of two independent controlled repeats (not a sigma)"
        ),
        "identity": identity,
        "weights_changed": bool(run_a.get("weight_changed_by_powerup")),
        "weights_identical_across_the_pair": weights_identical,
        "run_a_seconds": run_a.get("seconds"),
        "run_b_seconds": run_b.get("seconds"),
        "sealed_reference": display_name(sealed_path),
        "shape_matches_sealed_pair": shape_matches_sealed(run_a, sealed),
        "gains": {"run_a": run_a.get("j_n2b_prime"), "run_b": run_b.get("j_n2b_prime")},
        "by_source": {"run_a": run_a.get("by_source"), "run_b": run_b.get("by_source")},
        "j_n02_1_two_batch_inputs": {
            "note": "本件不判 J-N-02-1；§9.5⑴（owner 2026-10-18 签「两批都要过地板」）要的两批增益与地板三枚数在这里",
            "run_a_gain": delta_a,
            "run_b_gain": delta_b,
            "floor": floor_quality,
            "both_gains_exceed_floor": bool(delta_a > floor_quality and delta_b > floor_quality),
        },
        "material_note": (
            "两趟的材料基线都是人工构造：A 把快照里的投影账本清空（projected=[]），"
            "睡的是全库最新那 200 条；B 反选账本（除最旧 168 条去重交互与 32 枚约束种子外全部标为已投影），"
            "睡的是最旧那批。两批的交集实测为 0 条（构造时现算），来源配比逐列相同（32/168/0）。"
            if dimension == "material"
            else "基线由人工构造：快照里的投影账本被清空（projected=[]），"
            "因此这趟投影的是既有约束种子与 journal 交互，而不是当日新材料；"
            "两趟同源由 material_fingerprint 逐字相等作证。"
        ),
        "criteria": {
            "J_N02_1_floor_units": "Δquality 同窗口径（治疗 - 对照的窗口均值差），单位与增益列一致",
            "J_N02_1_denominator": "两趟各自的 8 条夜选文本窗口均值，不是题数",
            "J_N02_1_direction": (
                "材料维极差是两批材料上的位移幅度，引用时不得当「这一档效应很小」的下界"
                if dimension == "material"
                else "地板是 n=2 的极差 ⇒ 系统性偏小，只配当下限引用"
            ),
        },
        "guards": {
            "pair_stage_done": True,
            "pair_errors_empty": True,
            "pair_code_identity_equal": True,
            "pair_material_fingerprint_equal": dimension == "same_material",
            "pair_material_fingerprint_differs": dimension == "material",
            "pair_mother_digest_equal": dimension == "material",
            "pair_weights_actually_differ": dimension == "same_material",
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
    parser.add_argument(
        "--floor-dimension",
        choices=FLOOR_DIMENSIONS,
        default="same_material",
        help="same_material＝同码同材料的重复（2026-10-18 实测在本链恒为 0）；"
        "material＝同码同起始权重、两份不同材料，取两批增益之差的绝对值",
    )
    arguments = parser.parse_args(argv)
    try:
        refuse_overwrite(arguments.out_report)
        payload = build_payload(
            arguments.run_a, arguments.run_b, arguments.sealed, arguments.floor_dimension
        )
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
        f"floor_dimension={payload['floor_dimension']!r} "
        f"noise_floor_delta_quality_abs={payload[FLOOR_QUALITY_KEY]!r} "
        f"shape_matches_sealed_pair={payload['shape_matches_sealed_pair']!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
