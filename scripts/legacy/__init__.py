"""legacy 隔离区：仅历史诊断脚本，不再被训练/产品流程引用。

本目录收纳早期「崩溃检查点」一次性诊断脚本（如 _diag_*_health.py 系列），
它们依赖运行目录（`sys.path.insert(0, r"e:\\Seed")` + `from seed import Seed`），
不可作为包被自动测试或训练流程 import。请勿在新代码中引用本目录。

公共常量
--------
CHECKPOINT_DIR
    指向仓库根下的 ``checkpoints/`` 目录（即 ``e:\\Seed\\checkpoints``）。
    诊断脚本统一从此读取，避免各自硬编码绝对路径。

    NOTE: 审计建议原文写 ``parents[1]``，但本文件位于
    ``scripts/legacy/__init__.py``：``parents[1]`` 会得到 ``scripts/checkpoints``
    （不存在）。仓库根应为 ``parents[2]``，故此处用 ``parents[2]`` 以保证
    torch.load 仍加载到真实的 ``checkpoints/seed_corpus.pt``，不改变加载语义。

新脚本编写规范
--------------
新脚本应从项目根以 ``from scripts.training.X import`` 形式可被 import，
避免运行目录相对导入（如 ``from verify_taiji_m6_endogenous_replay import``、
``from _diag_m6_write_basis import`` 这类依赖运行目录的写法）。
"""

from pathlib import Path

# scripts/legacy/__init__.py → parents[2] = 仓库根 E:/Seed
LEGACY_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"

__all__ = ["LEGACY_DIR", "CHECKPOINT_DIR"]
