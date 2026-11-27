"""双臂／三臂权重的逐张量比较器：给「`DIFFERING 20 of 220`」这类读数一个仓内可复算的出处。

来历（2026-10-10，DEBT-G92）：㊵-656 发表的那次比对是在仓外一次性跑的，仓内既无仪器也无封存件；
㊵-662 想复核时只能再写一支脚本。本件把那套比较逻辑固定下来，并解决两个**单位口径**问题：

* **槽 vs 枚**：同一枚权重在信封里住着两个根（`substrate.*` 与 `taiji.kernel.*`）⇒ "20 个差异"可能只是
  10 枚权重各被数了一次。本件同时出版槽数与**按内容去重后的枚数**，两者缺一都会让双臂设计误倍。
* **镜像一致性**：两根的副本在同臂内应当逐位相等（实测 94/94）；不相等就说明存盘路径有分叉，
  那是一条独立缺陷，不能靠"反正比出来一样"混过去。

判定权全部在读数里：`--out-report` 落 JSON，stdout 只出 ASCII（`ensure_ascii=True`），
因为 win32 控制台默认按 GBK 解码、`⇒` 与中文会把一次正常判读砸成 `UnicodeEncodeError`。

2026-10-13（DEBT-G129／㊵-769）升版 v2：形状不齐的两枚此前会在算 delta 的路上抛 `RuntimeError`
（拿批次 B 之前的基座比治疗臂当场崩，rc=1 且件没落）。现在形状不齐的那批**点名出版**
（`shape_mismatch_slots`／`shape_mismatch_names`／`shape_pairs`），delta 表只覆盖
`delta_comparable_slots` 那一半，两边相加＝`differing_slots`。旧封存件是 v1，引用要点名哪一版。

2026-10-13（DEBT-G133 修法①／㊵-777）升版 v3：名集不齐此前**一律** rc=2，而批次 C 的实测形状是
"治疗臂的信封多出两枚 `memory.last_event`（对照臂与基座都从未写过那一格）"⇒ 单变量实验做对了
反而让两台下游仪器（臂↔臂自证、base↔臂集合差）都没有判决载体。现在按**跑器自己的面账本**分流：
缺席的名字若住在 `readout_retrain_spec.SURFACES` 声明过的那几面（＝spec 说它存在的面），
仍然响亮拒绝（那是结构缺失）；只住在**未声明**面里的缺席＝"某臂从未写过那一格"，
按交集比较并出版 `comparable_on_intersection_only=true` 与 `name_set_difference` 的逐面归类。
`same_name_set` 这一枚**照旧为假**——判据（`J-N5t-3` 的名集齐）不许被工具的宽容悄悄满足，改的只是拒绝面。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from readout_retrain_spec import SURFACES as DECLARED_SURFACES  # noqa: E402

FORMAT = "taiji-n5-arm-tensor-comparison-v3"
#: 同一枚权重的两个存放根；镜像核对只走这一对，不猜别的形状。
MIRROR_PREFIXES = ("substrate.", "taiji.kernel.")
TOP_DELTA_COUNT = 6
#: `PLAN-N5-21` §6（㊵-770）要的三档分列：按**名字后缀**机械分档，不靠人读。
#: 计量面按构造必然随观察次数增长（喂入那一步多走一次 observe），所以它只出版、不参与"第四类"判定。
METER_SUFFIXES = (".symbol_exposure", ".visits")
TOPOLOGY_SUFFIXES = (".pre_index",)
#: 每档点名到枚的上限；真面一次比对可能有几十枚同名后缀，全部列出会把件撑爆而读数没多信息。
BUCKET_NAME_CAP = 6


def _slot_bucket(name: str) -> str:
    if name.endswith(METER_SUFFIXES):
        return "meter"
    if name.endswith(TOPOLOGY_SUFFIXES):
        return "topology"
    return "weight"


#: 存放根；归类面名时先剥掉它们，再取第一段（`taiji.kernel.*` 比 `substrate.*` 多一段，
#: 直接取第二段会把同一枚面数成两枚——DEBT-G92 那件"槽 vs 枚"的同一根坑）。
STORAGE_ROOTS = MIRROR_PREFIXES + ("taiji.components.", "taiji.")


def _name_face(name: str) -> str:
    stripped = name
    for root in STORAGE_ROOTS:
        if stripped.startswith(root):
            stripped = stripped[len(root) :]
            break
    return stripped.split(".")[0] if stripped else name


def _name_is_declared(name: str) -> bool:
    """该张量是否住在跑器**面账本声明过**的那几面（`readout_retrain_spec.SURFACES`）。

    分流靠它，而不是靠一份手抄的名字表：spec 说这面存在，那一面的槽就该在两枚信封里都在场；
    不在场＝结构缺失，仍然要响亮拒绝。未声明面的缺席才可能是"某臂从未写过那一格"。
    """

    return bool(set(name.split(".")) & set(DECLARED_SURFACES))


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
        total += sum(
            _list_tensor_count(item) for item in obj if isinstance(item, (list, tuple, dict))
        )
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
    #: 形状不同的两枚**做不了减法**：`(4,N) - (5,N)` 抛 `RuntimeError`，而它抛在算 delta 的路上——
    #: 2026-10-13 用"批次 B 之前的基座"比治疗臂时整台仪器 rc=1 崩掉（DEBT-G129）。
    #: 所以先把形状不一致的那批挑出来点名出版，剩下的才进 delta 表：既不混成 0，也不让整台仪器崩。
    shape_mismatch = sorted(name for name in differing if a[name].shape != b[name].shape)
    uncomparable = set(shape_mismatch)
    comparable = [name for name in differing if name not in uncomparable]
    shape_pairs = sorted(
        {f"{list(a[name].shape)} vs {list(b[name].shape)}" for name in shape_mismatch}
    )
    buckets: dict[str, list[str]] = {"weight": [], "topology": [], "meter": []}
    for name in differing:
        buckets[_slot_bucket(name)].append(name)
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
            for name in comparable
        ),
        key=lambda row: -row["max_abs_delta"],
    )
    #: 名集差的归类（DEBT-G133 修法①）：缺席的名字里，住在声明面上的那批＝结构缺失（必须拒绝），
    #: 只住在未声明面的那批＝"这一侧从未写过那一格"，可以按交集比。
    absent = only_left + only_right
    declared_absent = sorted(name for name in absent if _name_is_declared(name))
    undeclared_faces: dict[str, int] = {}
    for name in absent:
        if not _name_is_declared(name):
            face = _name_face(name)
            undeclared_faces[face] = undeclared_faces.get(face, 0) + 1
    return {
        "label": label,
        "same_name_set": not only_left and not only_right,
        "only_in_left": only_left[:8],
        "only_in_right": only_right[:8],
        "name_set_difference": {
            "only_in_left_count": len(only_left),
            "only_in_right_count": len(only_right),
            "declared_surface_absences": declared_absent[:8],
            "undeclared_absence_faces": dict(sorted(undeclared_faces.items())),
            "absence_source": (
                "缺席名字的归类按 `readout_retrain_spec.SURFACES` 判定；"
                "声明面上的缺席＝结构缺失，仍走响亮拒绝。"
            ),
        },
        #: 只在"名集不齐 且 不齐全住在未声明面"时为真；名集全同时它是假（没有东西可窄）。
        "comparable_on_intersection_only": bool(absent) and not declared_absent,
        "compared_slots": len(common),
        "differing_slots": len(differing),
        #: `top_deltas` 只覆盖 `delta_comparable_slots`；剩下那批以形状对出版。
        "delta_comparable_slots": len(comparable),
        "shape_mismatch_slots": len(shape_mismatch),
        "shape_mismatch_names": shape_mismatch[:8],
        "shape_pairs": shape_pairs[:8],
        #: 三档分列（`PLAN-N5-21` §6）：`J-N5t-2` 的"第四类"只看 weight＋topology，meter 只出版。
        "differing_by_bucket": {key: len(names) for key, names in buckets.items()},
        "differing_bucket_names": {
            key: sorted(names)[:BUCKET_NAME_CAP] for key, names in buckets.items()
        },
        "meter_face_note": (
            "计量面（symbol_exposure／visits）按构造随观察次数增长：治疗臂比对照臂多"
            " `generated_feed.feeds` 次 observe，所以这一档两臂不等**不是**混杂未对齐。"
        ),
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
        print(
            json.dumps(
                {"format": FORMAT, "status": "refuse_missing_checkpoint", "missing": missing}
            )
        )
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
        #: 分流用的那枚名单**从 spec 现读**，不是手抄：改 spec 的那几面就会改这台仪器的拒绝面。
        "declared_surfaces": list(DECLARED_SURFACES),
        "declared_surfaces_source": "readout_retrain_spec.SURFACES",
        "mirror_consistency": [_mirror_consistency(side) for side, _ in sides],
        "pairs": [
            compare_pairs(sides[0][0], side, f"{args.checkpoint[0]} -> {name}")
            for side, name in sides[1:]
        ],
    }
    structural = [row for row in payload["pairs"] if not row["same_name_set"]]
    rc = 0
    if structural:
        #: 名集不齐分两种（DEBT-G133 修法①）：未声明面的缺席可以按交集比；
        #: 声明面上的缺席＝那一面少了一枚槽，仍然响亮拒绝，不许被宽容掉。
        if all(row["comparable_on_intersection_only"] for row in structural):
            payload["intersection_note"] = (
                "名集不齐的缺席全部住在跑器面账本未声明的面（`comparable_on_intersection_only=true`）"
                "⇒ 本件按**共有槽交集**出比较读数；`same_name_set` 照旧为假，"
                "`J-N5t-3` 的「名集齐」那一支在件里就是不支持的，不许被工具的宽容悄悄满足。"
            )
        else:
            payload["status"] = "refuse_name_sets_differ"
            rc = 2
    if any(row["disagreeing"] for row in payload["mirror_consistency"]):
        #: 镜像本臂内不一致＝存盘路径分叉，比"两臂有差"更该响亮。
        payload["status"] = "refuse_mirror_divergence"
        rc = 2
    text = json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if args.out_report is not None:
        out_path = (
            args.out_report if args.out_report.is_absolute() else PROJECT_ROOT / args.out_report
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
