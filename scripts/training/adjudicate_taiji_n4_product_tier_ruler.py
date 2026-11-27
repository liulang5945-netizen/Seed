"""产品档寻址尺判读器（PLAN-N4-06 的 J-N4f-2／J-N4f-3／J-N4f-5，机械执行、不手算）。

用法：

    python scripts/training/adjudicate_taiji_n4_product_tier_ruler.py \
        --report-small <a035.json> --report-large <a070.json> --out <json>

判据（先冻，见 `plans/reference/PLAN-N4-06_perturbation_strength_and_ruler_prereg_20261009.md`）：

* **J-N4f-1 在场性**：两份报告都要有 `wrong_top1_rate`／`noise_band_adjacent_block_max`／
  `block_means`／`faces_provenance`（含 `mount_layer`、`episodic_writes`），
  缺任一 ⇒ `ran_not_measured` 且 `rc=2`。
* **J-N4f-2 尺可用**：`band_small > 0` 且 `band_large > 0` 且
  `|w_small - w_large| >= max(band_small, band_large)`。不满足 ⇒ `ruler_unusable`，
  **两档都不许发表优劣**（`rc=1`）。
* **J-N4f-3 方向性**：尺可用时，只有 `w_large > w_small` 才判 `angle_sensitive`；
  反向或相等 ⇒ `not_angle_sensitive` ⇒ **即使数值好看也不许**宣称产品档寻址有效。
* **J-N4f-5 溯源**：两档 `faces_provenance.mount_layer` 必须都是 `product` 且
  `episodic_writes > 0`，否则判 `provenance_not_product_tier`（`rc=2`）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FORMAT = "taiji-n4-product-tier-ruler-adjudication-v1"
REQUIRED_TOP = ("wrong_top1_rate", "noise_band_adjacent_block_max", "block_means")
REQUIRED_PROVENANCE = ("mount_layer", "episodic_writes")


def _load(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_TOP if key not in report]
    prov = report.get("faces_provenance")
    if not isinstance(prov, dict):
        missing.append("faces_provenance")
        prov = {}
    else:
        missing += [f"faces_provenance.{key}" for key in REQUIRED_PROVENANCE if key not in prov]
    return (None if missing else report), missing


def adjudicate(small: Path, large: Path) -> tuple[dict[str, Any], int]:
    rep_s, miss_s = _load(small)
    rep_l, miss_l = _load(large)
    if miss_s or miss_l:
        return {
            "format": FORMAT,
            "status": "ran_not_measured",
            "missing": {"small": miss_s, "large": miss_l},
            "j_n4f_2": "unverified_missing_face",
            "j_n4f_3": "unverified_missing_face",
        }, 2
    assert rep_s is not None and rep_l is not None

    band_s = float(rep_s["noise_band_adjacent_block_max"])
    band_l = float(rep_l["noise_band_adjacent_block_max"])
    w_s = float(rep_s["wrong_top1_rate"])
    w_l = float(rep_l["wrong_top1_rate"])
    prov_s = rep_s["faces_provenance"]
    prov_l = rep_l["faces_provenance"]
    out: dict[str, Any] = {
        "format": FORMAT,
        "prereg": "PLAN-N4-06 §3 J-N4f-1/2/3/5",
        "bands": {"small": band_s, "large": band_l},
        "wrong_top1": {"small": w_s, "large": w_l},
        "provenance": {
            "small": {
                "mount_layer": prov_s.get("mount_layer"),
                "episodic_writes": prov_s.get("episodic_writes"),
            },
            "large": {
                "mount_layer": prov_l.get("mount_layer"),
                "episodic_writes": prov_l.get("episodic_writes"),
            },
        },
        "delta_large_minus_small": w_l - w_s,
        "line_required": max(band_s, band_l),
    }
    rc = 0

    product_tier = all(
        str(side.get("mount_layer")) == "product" and int(side.get("episodic_writes") or 0) > 0
        for side in (prov_s, prov_l)
    )
    if not product_tier:
        out["j_n4f_5"] = "provenance_not_product_tier"
        out["verdict"] = "not_adjudicable_provenance"
        return out, 2
    out["j_n4f_5"] = "product_tier_verified"

    usable = band_s > 0.0 and band_l > 0.0 and abs(w_l - w_s) >= max(band_s, band_l)
    out["ruler_usable"] = bool(usable)
    if not usable:
        out["j_n4f_2"] = "ruler_unusable"
        out["verdict"] = "ruler_unusable"
        return out, 1
    out["j_n4f_2"] = "ruler_usable"
    if w_l > w_s:
        out["j_n4f_3"] = "angle_sensitive"
        out["verdict"] = "angle_sensitive"
    else:
        out["j_n4f_3"] = "not_angle_sensitive"
        out["verdict"] = "not_angle_sensitive"
        #: 数值好看也不许宣称有效 ⇒ 判读器把这条禁令变成非零退出，而不是靠人记着。
        rc = 1
    return out, rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="产品档寻址尺判读（PLAN-N4-06）")
    parser.add_argument("--report-small", required=True)
    parser.add_argument("--report-large", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    for raw in (args.report_small, args.report_large):
        if not Path(raw).is_file():
            print(json.dumps({"format": FORMAT, "status": "missing_report", "path": str(raw)}))
            return 2
    payload, rc = adjudicate(Path(args.report_small), Path(args.report_large))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({**payload, "rc": rc}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({**payload, "rc": rc}, ensure_ascii=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
