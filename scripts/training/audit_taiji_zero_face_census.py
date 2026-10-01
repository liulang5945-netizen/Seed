"""零面普查：一个检查点里**所有**参数面，哪些范数恰好为零。

为什么要有这一支
----------------
`SPEC-A-22` §23 的记忆三层审计末尾自报了一条局限：它的 `inventory` 只枚举了
**该探针点名的几个面**（motor／predictive_readout／predictive_context／fabric 的快慢两组
decoders），所以"**还有哪些面是零**"当时说不出；原文写着"另跑一次全面普查才能说"。
本支就是那一次普查：**不点名**，递归遍历整份载荷，把所有张量按路径列全、逐个量范数。

它回答的是一个反复出现、且已被付过学费的问题类别
------------------------------------------------
本仓已多次登记"训练后仍全零"的面（"六个面训练后仍全零"），2026-09-28 又实测到
了一次同型的**静默空转**：一个默认关的开关被打开后，它挂的那条链在主入口下**根本没被
走到**，于是那个面的权重**永远是 0**——训练跑了两个 2M-tick 的臂才发现（见
`PLAN-A-26` §6.2）。零面普查给出的正是这类面的**名单**：一个面为零，只有三种解释
（① 默认关；② 开着但那条链没被走到；③ 开着也走到了、但更新量恰好归零），
普查负责把①②与③分开——**它不下结论，它给分母**。

纪律
----
* **只读**：不改任何检查点、不改任何配置、不跑训练；
* **跑前后基座 sha256 必须逐位相同**（写进 `instrument_guard`）；
* **结构性索引不算可学面**：`pre_index`／`channel`／`polarity` 这类是固定扇入的拓扑索引
  （建库时抽的边），它们"不变"是设计，不是缺陷——单独计数、不与可学面混在一起；
* 分母为 0（一个张量都没枚举到）时**不出结论**，退出码 2。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_FORMAT = "taiji-zero-face-census-v1"
VERSION = 1
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"

#: 固定扇入抽出来的拓扑索引：它们逐位不变是**设计**（见 `SparseSynapses`），单独计数。
STRUCTURAL_SUFFIXES = ("pre_index", "channel", "polarity", "unit_ids", "offsets")

#: 活动状态（不是权重）：`state.*` 是每-tick 的膜电位/痕迹/反馈，静止时为零是**正常**的，
#: 不能和"学不到的面"混在一起，否则会把"零面占比"算虚高。
STATE_PREFIXES = ("substrate.state", "state")


def _classify(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1].split("[")[0]
    if leaf in STRUCTURAL_SUFFIXES:
        return "structural"
    if path.startswith(STATE_PREFIXES):
        return "state"
    return "learnable"


#: v10 信封把**同一批权重存两遍**：`taiji.kernel.*`（原生）与 `substrate.*`（兼容载荷）
#: 是同一活体模型的两次序列化，实测逐位相同（2026-09-28：`fabric.decoders[0].edge_weight`
#: `torch.equal` 为 True）。若不去重，零面占比会被**系统性放大近一倍**，普查自己的头条数字
#: 就是假的。所以先把两份映到同一"逻辑名"，逐位相等则留一份；**不相等就响亮记冲突**。
MIRROR_PREFIXES = ("taiji.kernel.", "substrate.")


def _logical_name(path: str) -> str:
    for prefix in MIRROR_PREFIXES:
        if path.startswith(prefix):
            return "kernel." + path[len(prefix) :]
    return path


def _deduplicate_mirrors(
    tensors: list[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, Any]]:
    kept: dict[str, tuple[str, torch.Tensor]] = {}
    order: list[str] = []
    duplicates = 0
    conflicts: list[dict[str, Any]] = []
    for path, tensor in tensors:
        key = _logical_name(path)
        if key not in kept:
            kept[key] = (path, tensor)
            order.append(key)
            continue
        first_path, first = kept[key]
        if first.shape == tensor.shape and torch.equal(first, tensor):
            duplicates += 1
            continue
        conflicts.append({"logical": key, "paths": [first_path, path]})
        alias = f"{key}#conflict{len(conflicts)}"
        kept[alias] = (path, tensor)
        order.append(alias)
    return [kept[key] for key in order], {
        "mirror_duplicates": duplicates,
        "mirror_conflicts": conflicts,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def iter_tensors(node: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """不点名地递归遍历：只要张量就收，路径即"面"的名字。"""

    found: list[tuple[str, torch.Tensor]] = []
    if isinstance(node, torch.Tensor):
        found.append((prefix or "<root>", node))
    elif isinstance(node, dict):
        for key, value in node.items():
            found.extend(iter_tensors(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found.extend(iter_tensors(value, f"{prefix}[{index}]"))
    return found


def census(checkpoint: Path) -> dict[str, Any]:
    started = datetime.now(UTC)
    digest_before = _sha256(checkpoint)
    envelope = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(envelope, dict):
        raise SystemExit(f"{checkpoint} is not a checkpoint payload (not a dict)")

    tensors = iter_tensors(envelope)
    tensors, mirror = _deduplicate_mirrors(tensors)
    learnable = [(path, tensor) for path, tensor in tensors if _classify(path) == "learnable"]
    structural = [(path, tensor) for path, tensor in tensors if _classify(path) == "structural"]
    state_faces = [(path, tensor) for path, tensor in tensors if _classify(path) == "state"]

    faces: list[dict[str, Any]] = []
    for path, tensor in learnable:
        values = tensor.detach().to(torch.float64)
        finite = bool(torch.isfinite(values).all())
        norm = float(values.norm()) if finite else float("nan")
        faces.append(
            {
                "face": path,
                "params": int(tensor.numel()),
                "l2_norm": round(norm, 9) if finite else None,
                "abs_sum": round(float(values.abs().sum()), 9) if finite else None,
                "dtype": str(tensor.dtype),
                "is_zero": bool(finite and norm == 0.0),
                "finite": finite,
            }
        )

    zero_faces = [face for face in faces if face["is_zero"]]
    zero_params = sum(face["params"] for face in zero_faces)
    total_params = sum(face["params"] for face in faces)
    #: 身份器官的表是**惰性分配、从未写过**的容量（路由键仓 64 槽 × 原型宽），实测占载荷九成以上
    #: ⇒ 若把它算进"零面占比"，那个头条数字只反映"一张空表有多大"，不是健康度。两条都给。
    identity_zero = [face for face in zero_faces if "identity_organ" in face["face"]]
    identity_zero_params = sum(face["params"] for face in identity_zero)
    identity_total = sum(face["params"] for face in faces if "identity_organ" in face["face"])
    digest_after = _sha256(checkpoint)

    metadata = envelope.get("metadata") or {}
    payload = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "completed",
        "scope": (
            "不点名地遍历整份检查点载荷，逐个参数面量范数，列出**恰好为零**的面；"
            "只读、零训练、不改配置"
        ),
        "does_not_do": [
            "does not decide why a face is zero (默认关 / 链没被走到 / 更新量归零)",
            "writes no checkpoint and trains nothing",
            "does not treat structural index tensors as learnable faces",
            "does not see organs that are absent from the file: v8/v9 档（如 `seed_beta.pt`）"
            "里 F1 两个器官要载入时才由迁移生出，本普查在**文件级**看不到它们"
            "（那需要活体级普查）",
        ],
        "checkpoint": {
            "path": _relative(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": digest_before,
            "tick": int(metadata.get("tick", -1)),
            "payload_format": str(envelope.get("format", "")),
        },
        "instrument_guard": {
            "base_sha_unchanged": digest_before == digest_after,
            "learnable_faces": len(faces),
            "learnable_params": total_params,
            "denominator_nonzero": len(faces) > 0 and total_params > 0,
            "mirror_duplicates_dropped": mirror["mirror_duplicates"],
            "mirror_conflicts": mirror["mirror_conflicts"],
        },
        "summary": {
            "learnable_faces": len(faces),
            "learnable_params": total_params,
            "zero_faces": len(zero_faces),
            "zero_params": zero_params,
            "zero_share_of_learnable": (
                round(zero_params / total_params, 6) if total_params else None
            ),
            "identity_organ_faces": sum(1 for face in faces if "identity_organ" in face["face"]),
            "identity_organ_params": identity_total,
            "identity_organ_zero_params": identity_zero_params,
            "zero_params_excluding_identity_organ": zero_params - identity_zero_params,
            "learnable_params_excluding_identity_organ": total_params - identity_total,
            "zero_share_excluding_identity_organ": (
                round(
                    (zero_params - identity_zero_params) / (total_params - identity_total),
                    6,
                )
                if total_params - identity_total
                else None
            ),
            "structural_faces": len(structural),
            "structural_params": sum(int(t.numel()) for _, t in structural),
            "state_faces": len(state_faces),
            "state_params": sum(int(t.numel()) for _, t in state_faces),
            "non_finite_faces": [face["face"] for face in faces if not face["finite"]],
        },
        "zero_face_list": sorted(zero_faces, key=lambda face: -face["params"]),
        "faces": sorted(faces, key=lambda face: -face["params"]),
        "started_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="写出路径；缺省 reports/taiji_zero_face_census_<stem>_<date>.json",
    )
    parser.add_argument(
        "--print-zero",
        type=int,
        default=20,
        help="标准输出里列几个零面（默认 20）",
    )
    args = parser.parse_args(argv)

    checkpoint = (
        args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    )
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint missing: {checkpoint}")
    report = args.report
    if report is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d")
        report = PROJECT_ROOT / "reports" / f"taiji_zero_face_census_{checkpoint.stem}_{stamp}.json"
    elif not report.is_absolute():
        report = PROJECT_ROOT / report

    payload = census(checkpoint)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(report)

    summary = payload["summary"]
    guard = payload["instrument_guard"]
    print(f"checkpoint: {payload['checkpoint']['path']} (tick {payload['checkpoint']['tick']})")
    print(f"sha256: {payload['checkpoint']['sha256']}")
    print(
        "learnable: {learnable_faces} 面 / {learnable_params} 数；"
        "零面: {zero_faces} 面 / {zero_params} 数（占 {share}）".format(
            share=summary["zero_share_of_learnable"], **summary
        )
    )
    print(f"structural: {summary['structural_faces']} 面 / {summary['structural_params']} 数")
    print(f"guard: {json.dumps(guard, ensure_ascii=False)}")
    print(f"\n零面清单（前 {args.print_zero} 个，按参数数降序）：")
    for face in payload["zero_face_list"][: args.print_zero]:
        print(f"  {face['face']:60s} 数={face['params']:7d}  norm={face['l2_norm']}")
    if len(payload["zero_face_list"]) > args.print_zero:
        print(f"  ...另有 {len(payload['zero_face_list']) - args.print_zero} 个零面，见读数件")
    print(f"\n读数件: {_relative(report)}")

    if not guard["denominator_nonzero"]:
        print("分母为 0 ⇒ 不出结论", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
