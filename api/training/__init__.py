"""
训练 API 路由子包
================
Seed 原生训练、数据集管理、训练控制、检查点续训与发布查询端点。

模块:
  - common.py   → 公共工具函数
  - control.py  → 训练控制 (暂停/恢复/停止/重置)
  - datasets.py → 数据集管理 (上传/列表/删除/预览)
  - checkpoints.py → 检查点列表 (供前端训练页断点续训选择)
  - resume.py   → 检查点续训 (SSE 流式进度，前端"恢复训练"按钮)
  - native.py   → Taiji raw-byte 原生训练 (SSE 流式进度)
  - publish.py  → 模型发布查询 & GGUF 不支持消息
  - recommend.py → 硬件检测 & 数据集质量检查
"""

from fastapi import APIRouter

router = APIRouter()

from .checkpoints import router as checkpoints_router
from .resume import router as resume_router
from .control import router as control_router
from .datasets import router as datasets_router
from .publish import router as publish_router
from .native import router as native_router
from .recommend import router as recommend_router

router.include_router(recommend_router)
router.include_router(control_router)
router.include_router(datasets_router)
router.include_router(checkpoints_router)
router.include_router(resume_router)
router.include_router(publish_router)
router.include_router(native_router)
