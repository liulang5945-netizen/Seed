"""N5 影子通电验收判读器（PLAN-N5-02 的 J-N5b-1／J-N5b-2／J-N5b-3 前半，机械执行、不手算）。

为什么要有这台仪器：㊵-597 冻的判据里，「在场性」「学习是否发生」「同批配对能否成立」三支可以纯机械判，
而本仓已两次栽在「缺列被读成满足」与「手对齐两个均值」（㊵-590 的 `get()` 假 null、`J-N3a′` 的手算末段）。
所以这三支由脚本出数，人只抄它的输出。

**本器不实现 J-N5b-4 的收益比较**：母量口径（`online_accuracy` 末两段段均值、自取噪声带、
`wrong_top1_rate`）在 N3／N4 的专门仪器里已有冻结算法，在这里重抄一份就是第二条链
（本仓纪律：别在新仪器里重抄生成链）。本器因此把 `j_n5b_4`／`j_n5b_5`／`j_n5b_6`
一律记成 `unverified_*`／`not_adjudicable_*`，让合取条件读起来必然是「未判」而不是「成立」。

用法：

    python scripts/training/adjudicate_taiji_n5_shadow_gate.py --arm-treated <ckpt.pt> \
        [--arm-control <ckpt.pt>] [--face-treated <pressure.jsonl>] [--face-control <...>] \
        --out <json>

判据映射（全部 fail-closed）：

* **J-N5b-1 自述在场性**：`n5_shadow` 块与五枚键必须齐（用 `key in env` 断言，不用 `get`）；
  缺任一 ⇒ 该臂 `ran_not_measured`，`rc=2`。
* **J-N5b-2 学习确曾发生**：`gate > 0` 且两枚效用不全为 0 ⇒ `shadow_learned`；
  `gate > 0` 而两枚效用同为 0 ⇒ `shadow_inert`（`rc=1`）；`gate == 0` ⇒ `not_powered`（`rc=1`）。
* **J-N5b-3 同批配对**：两臂 `shadow_gate_requested` 必须不同（相同 ⇒ `pairing_invalid`）；
  给了两枚压强面时再核对四元组（`seed`／`policy.minimum_pressure`／`readout`／
  `developmental.fast_slow_requested`／`developmental.bridge_gate_actual`），
  任一不同 ⇒ `pairing_invalid` 并点名是哪一维；全同 ⇒ `pairing_valid_on_quadruple`。
  只给一臂 ⇒ `unverified_no_control_arm`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORMAT = "taiji-n5-shadow-gate-adjudication-v1"
REQUIRED_KEYS = (
    "gate",
    "shadow_gate_requested",
    "candidate_gate",
    "candidate_utility",
    "candidate_counterfactual_utility",
)
#: 四元组（实际五维）逐条从**面头**取，不从命令行取：臂间差异只能来自现场，不能来自我的转述。
QUADRUPLE = (
    ("seed", ("seed",)),
    ("tau", ("policy", "minimum_pressure")),
    ("readout", ("readout",)),
    ("fast_slow", ("developmental", "fast_slow_requested")),
    ("bridge_gate", ("developmental", "bridge_gate_actual")),
)


def _shadow_block(checkpoint: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """返回 (n5_shadow 块, 缺失键列表)。块不在场时缺失键＝全部五枚。"""
    import torch  # 局部导入：只有真要读检查点时才拉起 torch

    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    env = loaded.get("envelope", loaded) if isinstance(loaded, dict) else {}
    block = env.get("n5_shadow") if isinstance(env, dict) else None
    if not isinstance(block, dict):
        return None, list(REQUIRED_KEYS)
    return block, [key for key in REQUIRED_KEYS if key not in block]


def _dig(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def judge_arm(name: str, checkpoint: Path) -> dict[str, Any]:
    block, missing = _shadow_block(checkpoint)
    out: dict[str, Any] = {"arm": name, "checkpoint": str(checkpoint)}
    if missing:
        out["j_n5b_1"] = "ran_not_measured"
        out["missing_self_report_keys"] = missing
        out["j_n5b_2"] = "unverified_missing_face"
        return out
    out["j_n5b_1"] = "present"
    out["missing_self_report_keys"] = []
    for key in REQUIRED_KEYS:
        out[key] = float(block[key])
    out["candidate_id"] = str(block.get("candidate_id", ""))
    powered = out["gate"] > 0.0
    nonzero_utility = (
        out["candidate_utility"] != 0.0 or out["candidate_counterfactual_utility"] != 0.0
    )
    if not powered:
        out["j_n5b_2"] = "not_powered"
    elif nonzero_utility:
        out["j_n5b_2"] = "shadow_learned"
    else:
        out["j_n5b_2"] = "shadow_inert"
    return out


def _face_header(path: Path) -> dict[str, Any] | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if row.get("kind") == "face":
            return row
    return None


def judge_pairing(
    treated: dict[str, Any],
    control: dict[str, Any] | None,
    treated_face: Path | None,
    control_face: Path | None,
) -> dict[str, Any]:
    """J-N5b-3：只问「这两臂能不能比」，不问「谁更好」。"""
    requested = {
        "treated_requested": treated.get("shadow_gate_requested"),
        "control_requested": control.get("shadow_gate_requested") if control else None,
    }
    if control is None:
        return {
            **requested,
            "status": "unverified_no_control_arm",
            "j_n5b_3": "unverified",
        }
    if treated.get("shadow_gate_requested") == control.get("shadow_gate_requested"):
        return {
            **requested,
            "status": "pairing_invalid",
            "reason": "identical_requested_gate",
            "j_n5b_3": "invalid",
        }
    base: dict[str, Any] = {**requested, "quadruple": {}}
    if treated_face is None or control_face is None:
        base["status"] = "gate_differs_quadruple_unverified"
        base["j_n5b_3"] = "unverified"
        return base
    head_t, head_c = _face_header(treated_face), _face_header(control_face)
    if head_t is None or head_c is None:
        return {
            **base,
            "status": "pairing_invalid",
            "reason": "face_header_missing",
            "j_n5b_3": "invalid",
        }
    differing: list[str] = []
    for label, path in QUADRUPLE:
        value_t, value_c = _dig(head_t, path), _dig(head_c, path)
        base["quadruple"][label] = {"treated": value_t, "control": value_c}
        if value_t != value_c:
            differing.append(label)
    if differing:
        return {
            **base,
            "status": "pairing_invalid",
            "reason": "quadruple_differs",
            "differing": differing,
            "j_n5b_3": "invalid",
        }
    base["status"] = "pairing_valid_on_quadruple"
    base["j_n5b_3"] = "valid"
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N5 影子通电验收判读（PLAN-N5-02 J-N5b-1/2/3）")
    parser.add_argument("--arm-treated", required=True)
    parser.add_argument("--arm-control", default=None)
    parser.add_argument("--face-treated", default=None)
    parser.add_argument("--face-control", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    treated_path = Path(args.arm_treated)
    if not treated_path.is_file():
        print(json.dumps({"format": FORMAT, "status": "missing_face", "face": str(treated_path)}))
        return 2

    arms: dict[str, Any] = {"treated": judge_arm("treated", treated_path)}
    rc = 0
    if arms["treated"]["j_n5b_1"] == "ran_not_measured":
        rc = max(rc, 2)
    elif arms["treated"]["j_n5b_2"] != "shadow_learned":
        rc = max(rc, 1)

    control_arm: dict[str, Any] | None = None
    if args.arm_control:
        control_path = Path(args.arm_control)
        if not control_path.is_file():
            print(
                json.dumps({"format": FORMAT, "status": "missing_face", "face": str(control_path)})
            )
            return 2
        control_arm = judge_arm("control", control_path)
        arms["control"] = control_arm
        if control_arm["j_n5b_1"] == "ran_not_measured":
            rc = max(rc, 2)

    pairing = judge_pairing(
        arms["treated"],
        control_arm,
        Path(args.face_treated) if args.face_treated else None,
        Path(args.face_control) if args.face_control else None,
    )
    if pairing["j_n5b_3"] == "invalid":
        rc = max(rc, 2)

    payload = {
        "format": FORMAT,
        "prereg": "PLAN-N5-02 §2 J-N5b-1／J-N5b-2／J-N5b-3 前半",
        "arms": arms,
        "pairing": pairing,
        #: 收益比较不在本器内实现（不重抄 N3／N4 已冻的算法链）⇒ 合取必然未判。
        "j_n5b_4": "unverified_metric_lane_lives_in_n3_and_n4_instruments",
        "j_n5b_5": "unverified_retention_lane_not_run",
        "j_n5b_6": "not_adjudicable_until_2_3_4_5_are_all_measured",
        "rc": rc,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload, ensure_ascii=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
