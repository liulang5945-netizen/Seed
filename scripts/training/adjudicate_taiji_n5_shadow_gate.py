"""N5 影子通电验收判读器（PLAN-N5-02 §2 的 J-N5b-1／J-N5b-2 两支，机械执行、不手算）。

为什么要有这台仪器：㊵-597 冻的判据里，「在场性」和「学习是否发生」两支是可以纯机械判的，
而本仓已经吃过两次「手算/眼看把缺列读成满足」的亏（E 的 `get()` 假 null、
`J-N3a′` 的手对齐末段均值）。所以这两支一律由脚本出数，人只抄它的输出。

用法：

    python scripts/training/adjudicate_taiji_n5_shadow_gate.py --arm-treated <checkpoint.pt> \
        [--arm-control <checkpoint.pt>] --out <json>

判据映射（全部 fail-closed）：

* **J-N5b-1 自述在场性**：`n5_shadow` 块与五枚键必须齐（`key in env` 断言，不用 `get`）；
  缺任一 ⇒ 该臂 `ran_not_measured`，进程 `rc=2`。
* **J-N5b-2 学习确曾发生**：`gate > 0.0` **且** `candidate_utility` 与
  `candidate_counterfactual_utility` 不全为 0 ⇒ `shadow_learned`；
  若 `gate > 0` 而两枚效用同为 0 ⇒ `shadow_inert`（通电但没学到，独立否证，`rc=1`）；
  若 `gate == 0` ⇒ `not_powered`（`rc=1`，与 F 的形状同）。
* 两臂都给了才做**同批配对核对**（J-N5b-3 的前半：corpus 指纹＋gate 请求值不同），
  不成立 ⇒ `pairing_invalid`（`rc=2`）；收益比较本身不在这里判（另格）。
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


def _shadow_block(checkpoint: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """返回 (n5_shadow 块, 缺失键列表)。块不在场时缺失键＝全部五枚。"""
    import torch  # 局部导入：只有真要读检查点时才拉起 torch

    loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    env = loaded.get("envelope", loaded) if isinstance(loaded, dict) else {}
    block = env.get("n5_shadow") if isinstance(env, dict) else None
    if not isinstance(block, dict):
        return None, list(REQUIRED_KEYS)
    return block, [key for key in REQUIRED_KEYS if key not in block]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N5 影子通电验收判读（PLAN-N5-02 J-N5b-1/2）")
    parser.add_argument("--arm-treated", required=True)
    parser.add_argument("--arm-control", default=None)
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

    pairing: dict[str, Any] = {"status": "unverified_no_control_arm"}
    if args.arm_control:
        control_path = Path(args.arm_control)
        if not control_path.is_file():
            print(
                json.dumps({"format": FORMAT, "status": "missing_face", "face": str(control_path)})
            )
            return 2
        arms["control"] = judge_arm("control", control_path)
        if arms["control"]["j_n5b_1"] == "ran_not_measured":
            rc = max(rc, 2)
        same_gate = arms["treated"].get("shadow_gate_requested") is not None and arms[
            "treated"
        ].get("shadow_gate_requested") == arms["control"].get("shadow_gate_requested")
        pairing = {
            "status": "identical_requested_gate" if same_gate else "arm_gate_differs",
            "treated_requested": arms["treated"].get("shadow_gate_requested"),
            "control_requested": arms["control"].get("shadow_gate_requested"),
        }
        if same_gate:
            pairing["status"] = "pairing_invalid"
            rc = max(rc, 2)

    payload = {
        "format": FORMAT,
        "prereg": "PLAN-N5-02 §2 J-N5b-1／J-N5b-2（J-N5b-3 后半与 4/5/6 不在本器内）",
        "arms": arms,
        "pairing": pairing,
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
