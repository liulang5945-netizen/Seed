# MIGRATED: scripts/archive/diagnostics/_diag_lateral_health.py → scripts/legacy/_diag_lateral_health.py
# 仅历史诊断脚本（崩溃检查点分析），不再被训练/产品流程引用；请勿在新代码中 import 本文件。
"""崩塌检查点的侧向竞争权重体检。"""

import sys

sys.path.insert(0, r"e:\Seed")

import torch

from seed import Seed
from scripts.legacy import CHECKPOINT_DIR  # 公共检查点目录常量（scripts/legacy/__init__.py）

model = Seed.from_checkpoint(torch.load(CHECKPOINT_DIR / "seed_corpus.pt", weights_only=False))
fabric = model.substrate.fabric
for index, lateral in enumerate(fabric.laterals):
    w = lateral.edge_weight
    print(
        f"lateral[{index}] shape={tuple(w.shape)} "
        f"mean={float(w.mean()):.6f} max={float(w.max()):.6f} "
        f"frac_nonzero={float((w > 0).float().mean()):.4f}"
    )

# 对照：刚初始化（未训练）模型的侧向权重
fresh = Seed(model.config)
for index, lateral in enumerate(fresh.substrate.fabric.laterals):
    w = lateral.edge_weight
    print(
        f"fresh lateral[{index}] mean={float(w.mean()):.6f} "
        f"max={float(w.max()):.6f} frac_nonzero={float((w > 0).float().mean()):.4f}"
    )
