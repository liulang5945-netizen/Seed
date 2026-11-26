"""F1 后继的第一步（训练侧）：训练后的 value encoder 特征对颜色是否可分。

只读 train 面：8 个颜色各过一次编码器，报特征矩阵的秩、类内/类间距离比、
最近质心留一准确率。用来回答"主路径没超出复制下界"里最先该查的那一项——
特征本身有没有可能携带值身份（不是能力，是必要条件）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_taiji_n1_plan177_f_f0b import _Arm, _rel  # noqa: E402
from taiji.joint_state_consumption_course import E1_COLORS  # noqa: E402

F1_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f1_feature_separability_20261029.json"


def separability(vectors: torch.Tensor, labels: list[int]) -> dict[str, Any]:
    centred = vectors - vectors.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centred)
    rank = int((singular > 1e-6 * singular.max()).sum())
    centroids = torch.stack(
        [vectors[[i for i, l in enumerate(labels) if l == c]].mean(0) for c in sorted(set(labels))]
    )
    within = sum(
        float((vectors[i] - centroids[labels[i]]).norm()) for i in range(len(labels))
    ) / len(labels)
    between = float(
        torch.cdist(centroids, centroids)[
            torch.triu_indices(len(centroids), len(centroids), offset=1)
        ].mean()
    )
    per_class = {label: labels.count(label) for label in set(labels)}
    measurable = all(count >= 2 for count in per_class.values())
    hits = 0
    if measurable:
        for i, label in enumerate(labels):
            own = [j for j, l in enumerate(labels) if l == label and j != i]
            hits += (vectors[i] - vectors[own[0]]).norm() < min(
                (vectors[i] - v).norm() for j, v in enumerate(vectors) if labels[j] != label
            )
    return {
        "rank": rank,
        "max_rank_after_centering": len(vectors) - 1,
        "within_radius": within,
        "between_centroid_distance": between,
        "ratio_between_over_within": between / within if within > 0 else None,
        "nearest_own_class_rate": hits / len(labels) if measurable else None,
        "distinct_vectors": len({tuple(round(float(x), 6) for x in v) for v in vectors}),
        "ruler_usable": measurable,
        "why_not_usable": (
            None
            if measurable
            else "每色只有一条样本：中心化后最大秩就是 n-1，类内距离恒 0，留一/质心类指标都无定义"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F1 后继：编码器特征可分性（训练侧）")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--arm", default="main_credit_on")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("separability report is write-once")
    state = json.loads(
        (F1_ROOT / f"{args.arm}__seed_{args.seed}" / "run_state.json").read_text(encoding="utf-8")
    )
    arm = _Arm.from_checkpoint(
        torch.load(ROOT / state["endpoint"], map_location="cpu", weights_only=False)
    )
    with torch.no_grad():
        vectors = torch.stack([arm.encoder.encode_text(colour) for colour in E1_COLORS])
    labels = list(range(len(E1_COLORS)))
    result = separability(vectors, labels)
    report = {
        "format": "taiji-n177-f1-separability-v1",
        "date": "2026-10-29",
        "scope": "train-side only: 8 colours through the trained encoder; no sealed read",
        "arm": args.arm,
        "seed": args.seed,
        "endpoint": _rel(ROOT / state["endpoint"]),
        "colours": list(E1_COLORS),
        "readings": result,
        "note": "rank < 8 means the feature cannot even represent 8 distinct values linearly",
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
