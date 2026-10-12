"""N5 生长五面里的**资源面**摘要器（PLAN-N5-01 §1 行④ 的执行体；只读，不训练）。

行④ 写着"资源面：参数量/激活"。㊵-676 的完成度对照发现这一列从来没被出版过——"两臂参数量相同"
只是逐张量比较的副产物，不是设计内读数。本器补上取法，并且**在第一版就把取法错堵住**：

* 出厂件 `checkpoints/seed_beta.pt` 只存 **93 枚张量**（权重面），而训练臂的信封里存 **17,726 枚**
  （连运行时状态一起落盘）。直接对两枚文件求张量总和会得到 `+4,628,052` 这样一枚**看起来像"参数暴涨"**
  的数——它量的是"训练把运行时状态写进了文件"，不是"模型长大了"。⇒ 参数量这一半**只在同名集合上比**
  （`shared_numel_*` 那三枚数），异名部分单独作为"运行时状态"出版，两笔不许相加。
* 激活这一半不是"没有"：只在臂里出现的那批名字里就有 `*.activity`／`*.membrane`／`*.trace`／
  `*.last_input` 这类**状态张量**。本器对它们出聚合统计（枚数、元素数、非零数、最大绝对值），
  并把语义钉死成"**收尾时刻的状态快照**"，不是时间序列、不是逐 tick 的激活幅度。
  ⇒ 不许拿 `shadow_forward_hits`（调用次数＝消费面的在场证据）冒充激活。

判定 `j_n5res_1` 的取值集合每一支都能为假：`parameters_grew_on_shared_face`（真生长）／
`parameters_unchanged_runtime_materialized`（本档现读）／`no_runtime_state_present`（若哪天臂里
不落状态）。缺键不猜：臂的 `n5_shadow` 块里没有 `unit_count` ⇒ 响亮失败，不出版影子容量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "taiji-n5-resource-face-v2"
ACTIVATION_NAME_TOKENS = (
    "activity",
    "membrane",
    "trace",
    "last_input",
    "eligibility",
)
#: 名字里带 `predictive_context` 的那批张量多数是**计数器**（`usage`／`age` 实测最大值 149,998），
#: 把它们算进"激活快照"会得到一枚看起来像激活幅度、其实是步数的数。计数器单独一档出版。
COUNTER_LAST_SEGMENTS = ("usage", "age", "count", "steps", "version")
SHADOW_KEY = "n5_shadow"


def _classify(name: str) -> str:
    last = name.rsplit(".", 1)[-1]
    if last in COUNTER_LAST_SEGMENTS:
        return "counter"
    if any(token in name for token in ACTIVATION_NAME_TOKENS):
        return "activation"
    return "other"


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    if isinstance(node, torch.Tensor):
        out[prefix] = node
    elif isinstance(node, dict):
        for key, value in node.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            out.update(_flatten(value, f"{prefix}[{index}]"))
    return out


def _load_pack(path: Path) -> dict[str, Any]:
    import torch

    if not path.is_file():
        raise ValueError(f"检查点不在场：{path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    #: `n5_shadow` 装的是标量，`_flatten` 只收张量 ⇒ 影子块必须从**原始信封**上取，
    #: 从展平后的表里取会永远取不到（那就是"键不存在被读成值为 null"的老形状）。
    shadow = raw.get(SHADOW_KEY) if isinstance(raw, dict) else None
    return {
        "tensors": _flatten(raw),
        "shadow": dict(shadow) if isinstance(shadow, dict) else {},
    }


def _activation_summary(tensors: dict[str, Any], names: list[str]) -> dict[str, Any]:

    elements = 0
    nonzero = 0
    max_abs = 0.0
    for name in names:
        node = tensors[name]
        if not node.dtype.is_floating_point:
            continue
        flat = node.reshape(-1)
        elements += int(flat.numel())
        nonzero += int((flat != 0).sum().item())
        if flat.numel():
            max_abs = max(max_abs, float(flat.abs().max().item()))
    return {
        "state_tensors": len(names),
        "elements": elements,
        "nonzero_elements": nonzero,
        "max_abs_element": round(max_abs, 6),
        "meaning": "final-state snapshot of runtime/activation tensors stored in the arm envelope",
    }


def summarize(arms: dict[str, Path], base: Path) -> dict[str, Any]:
    base_pack = _load_pack(base)
    base_names = set(base_pack["tensors"])
    per_arm: dict[str, Any] = {}
    for name, path in arms.items():
        pack = _load_pack(path)
        tensors = pack["tensors"]
        arm_names = set(tensors)
        shadow = pack["shadow"]
        if "unit_count" not in shadow:
            raise ValueError(f"臂 {name} 的 {SHADOW_KEY} 块缺 unit_count ⇒ 影子容量不可读，不猜")
        shared = sorted(base_names & arm_names)
        only_arm = sorted(arm_names - base_names)
        activation_names = [key for key in only_arm if _classify(key) == "activation"]
        counter_names = [key for key in only_arm if _classify(key) == "counter"]
        base_shared = sum(int(base_pack["tensors"][key].numel()) for key in shared)
        arm_shared = sum(int(tensors[key].numel()) for key in shared)
        per_arm[name] = {
            "checkpoint": str(path),
            "base_tensor_entries": len(base_names),
            "arm_tensor_entries": len(arm_names),
            "shared_entries": len(shared),
            "shared_numel_base": base_shared,
            "shared_numel_arm": arm_shared,
            "shared_numel_delta": arm_shared - base_shared,
            "only_in_arm_entries": len(only_arm),
            "only_in_arm_numel": sum(int(tensors[key].numel()) for key in only_arm),
            "only_in_base_entries": len(base_names - arm_names),
            "shadow_unit_count": shadow.get("unit_count"),
            "shadow_gate": shadow.get("gate"),
            "activation": _activation_summary(tensors, activation_names),
            "counters": _activation_summary(tensors, counter_names),
        }
    grew = sorted(n for n, row in per_arm.items() if row["shared_numel_delta"] > 0)
    no_state = all(row["only_in_arm_entries"] == 0 for row in per_arm.values())
    verdict = (
        "parameters_grew_on_shared_face"
        if grew
        else (
            "no_runtime_state_present" if no_state else "parameters_unchanged_runtime_materialized"
        )
    )
    return {
        "format": FORMAT,
        "status": "ok",
        "prereg": "plans/reference/PLAN-N5-01_four_layer_loop_design_prereg_20261009.md (§1 行④)",
        "base_checkpoint": str(base),
        "arms": per_arm,
        "arms_with_shared_parameter_growth": grew,
        "j_n5res_1": verdict,
        "reading_limit": (
            "parameter growth is compared ONLY on tensor names shared with the factory base, "
            "because the factory file stores weights while trained arms also persist runtime "
            "state; the two numbers must never be summed. The activation half is a FINAL-STATE "
            "SNAPSHOT of tensors whose names match the activation vocabulary, not a per-tick "
            "magnitude series, and call counts (shadow_forward/learn_hits) are consumption "
            "evidence that must not be substituted for it. Counter-like tensors (`usage`, `age`) "
            "are published in a SEPARATE `counters` bucket: they live under the same "
            "`predictive_context` name prefix and, measured on the 150k arms, they dominate the "
            "maximum at 149,998 - folding them into the activation snapshot would turn a step "
            "counter into an apparent activation magnitude."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N5 resource-face summary (read-only)")
    parser.add_argument(
        "--arm", action="append", default=[], metavar="NAME=PATH", help="repeatable arm checkpoint"
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "seed_beta.pt",
        help="base checkpoint the arms resumed from",
    )
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)

    arms: dict[str, Path] = {}
    bad: list[str] = []
    for token in args.arm:
        name, sep, raw = token.partition("=")
        if not sep or not name.strip() or not raw.strip():
            bad.append(token)
            continue
        arms[name.strip()] = Path(raw.strip())
    if bad or not arms:
        print(json.dumps({"status": "bad_arm_spec", "entries": bad, "arms_given": len(arms)}))
        return 2
    try:
        payload = summarize(arms, args.base)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "summary_failed", "error": str(error)}))
        return 2
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "j_n5res_1": payload["j_n5res_1"],
                "per_arm": {
                    name: {
                        "shared_delta": row["shared_numel_delta"],
                        "only_in_arm": row["only_in_arm_entries"],
                        "unit_count": row["shadow_unit_count"],
                        "activation_tensors": row["activation"]["state_tensors"],
                        "activation_nonzero": row["activation"]["nonzero_elements"],
                        "activation_max_abs": row["activation"]["max_abs_element"],
                    }
                    for name, row in payload["arms"].items()
                },
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
