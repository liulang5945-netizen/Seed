"""双臂／三臂权重的逐张量比较器：给「`DIFFERING 20 of 220`」这类读数一个仓内可复算的出处。

来历（2026-10-10，DEBT-G92）：㊵-656 发表的那次比对是在仓外一次性跑的，仓内既无仪器也无封存件；
㊵-662 想复核时只能再写一支脚本。本件把那套比较逻辑固定下来，并解决两个**单位口径**问题：

* **槽 vs 枚**：同一枚权重在信封里住着两个根（`substrate.*` 与 `taiji.kernel.*`）⇒ "20 个差异"可能只是
  10 枚权重各被数了一次。本件同时出版槽数与**按内容去重后的枚数**，两者缺一都会让双臂设计误倍。
* **镜像一致性**：两根的副本在同臂内应当逐位相等（实测 94/94）；不相等就说明存盘路径有分叉，
  那是一条独立缺陷，不能靠"反正比出来一样"混过去。

判定权全部在读数里：`--out-report` 落 JSON，stdout 只出 ASCII（`ensure_ascii=True`），
因为 win32 控制台默认按 GBK 解码、`⇒` 与中文会把一次正常判读砸成 `UnicodeEncodeError`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAT = "taiji-n5-arm-tensor-comparison-v1"
#: 同一枚权重的两个存放根；镜像核对只走这一对，不猜别的形状。
MIRROR_PREFIXES = ("substrate.", "taiji.kernel.")
TOP_DELTA_COUNT = 6


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensors_of(obj: Any, trail: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    """递归收信封里的张量，键是点分路径。

    刻意**不**走进 list：存盘 list 里的张量若混进同一张表就会用索引当名字，跨档一改顺序就错位
    （同 [[checkpoint-entry-names-not-portable-across-save-shapes]]）。list 里的张量另由
    `_list_tensor_count` 计数并出版，不静默丢弃。
    """
    collected = {} if out is None else out
    if isinstance(obj, dict):
        for key, value in obj.items():
            _tensors_of(value, f"{trail}.{key}" if trail else str(key), collected)
    elif _is_tensor(obj):
        collected[trail] = obj
    return collected


def _list_tensor_count(obj: Any) -> int:
    total = 0
    if isinstance(obj, dict):
        for value in obj.values():
            total += _list_tensor_count(value)
    elif isinstance(obj, (list, tuple)):
        total += sum(1 for item in obj if _is_tensor(item))
        total += sum(_list_tensor_count(item) for item in obj if isinstance(item, (list, tuple, dict)))
    return total


def _is_tensor(value: Any) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


def _numpy_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _load(path: Path) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    tensors = _tensors_of(obj)
    return {
        "tensors": tensors,
        "list_tensors": _list_tensor_count(obj),
        "file_sha256": _file_digest(path),
        "bytes": path.stat().st_size,
        "torch": torch,
    }


def _mirror_consistency(side: dict[str, Any]) -> dict[str, Any]:
    tensors = side["tensors"]
    torch = side["torch"]
    source, target = MIRROR_PREFIXES
    pairs = [
        (name, target + name[len(source) :])
        for name in tensors
        if name.startswith(source) and (target + name[len(source) :]) in tensors
    ]
    disagreeing = [
        f"{left} vs {right}"
        for left, right in pairs
        if not torch.equal(tensors[left], tensors[right])
    ]
    return {
        "mirrored_slots": len(pairs),
        "one_sided_slots": sum(1 for name in tensors if name.startswith(source)) - len(pairs),
        "agree_within_arm": len(pairs) - len(disagreeing),
        "disagreeing": disagreeing,
    }


def compare_pairs(left: dict[str, Any], right: dict[str, Any], label: str) -> dict[str, Any]:
    torch = left["torch"]
    a, b = left["tensors"], right["tensors"]
    only_left = sorted(set(a) - set(b))
    only_right = sorted(set(b) - set(a))
    common = sorted(set(a) & set(b))
    differing = [name for name in common if not torch.equal(a[name], b[name])]
    groups: dict[tuple[str, str, tuple[int, ...]], list[str]] = {}
    for name in differing:
        key = (
            hashlib.sha256(_numpy_bytes(a[name])).hexdigest()[:16],
            hashlib.sha256(_numpy_bytes(b[name])).hexdigest()[:16],
            tuple(a[name].shape),
        )
        groups.setdefault(key, []).append(name)
    deltas = sorted(
        (
            {
                "slot": name,
                "max_abs_delta": float((a[name].float() - b[name].float()).abs().max()),
                "shape": list(a[name].shape),
            }
            for name in differing
        ),
        key=lambda row: -row["max_abs_delta"],
    )
    return {
        "label": label,
        "same_name_set": not only_left and not only_right,
        "only_in_left": only_left[:8],
        "only_in_right": only_right[:8],
        "compared_slots": len(common),
        "differing_slots": len(differing),
        "distinct_content_pairs": len(groups),
        #: 每组的槽数（例：10 组各 2 槽 ⇒ [2,2,…]）。用列表而不是字典，避免同尺寸组互相覆盖计数。
        "distinct_group_sizes": sorted((len(names) for names in groups.values()), reverse=True),
        "verdict": "identical" if not differing else "differing",
        "top_deltas": deltas[:TOP_DELTA_COUNT],
        "differing_slots_names": sorted(differing),
    }


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="N5 臂间权重逐张量比较（槽数与枚数一起出）")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="PT",
        help="两枚或三枚检查点；第一枚是基准，其余逐条与它比",
    )
    parser.add_argument("--out-report", type=Path, default=None)
    args = parser.parse_args(argv)

    if len(args.checkpoint) < 2:
        print(json.dumps({"format": FORMAT, "status": "refuse_need_two_checkpoints"}))
        return 2
    missing = [str(path) for path in args.checkpoint if not Path(path).is_file()]
    if missing:
        print(json.dumps({"format": FORMAT, "status": "refuse_missing_checkpoint", "missing": missing}))
        return 2

    sides = [(_load(Path(path)), str(path)) for path in args.checkpoint]
    payload: dict[str, Any] = {
        "format": FORMAT,
        "status": "measured",
        "files": [
            {"path": name, "file_sha256": side["file_sha256"], "bytes": side["bytes"]}
            for side, name in sides
        ],
        "tensor_slots": len(sides[0][0]["tensors"]),
        "list_tensor_count": sides[0][0]["list_tensors"],
        "mirror_consistency": [_mirror_consistency(side) for side, _ in sides],
        "pairs": [
            compare_pairs(sides[0][0], side, f"{args.checkpoint[0]} -> {name}")
            for side, name in sides[1:]
        ],
    }
    structural = [row for row in payload["pairs"] if not row["same_name_set"]]
    rc = 0
    if structural:
        payload["status"] = "refuse_name_sets_differ"
        rc = 2
    if any(row["disagreeing"] for row in payload["mirror_consistency"]):
        #: 镜像本臂内不一致＝存盘路径分叉，比"两臂有差"更该响亮。
        payload["status"] = "refuse_mirror_divergence"
        rc = 2
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out_report is not None:
        out_path = args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
