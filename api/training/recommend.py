"""
训练配置推荐 API
================
提供硬件自适应训练配置推荐和数据集质量检查端点
"""

import logging
import os
from typing import cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from seed.datasets import inspect_native_dataset
from taiji import TaijiConfig

logger = logging.getLogger("ApiServer.Training.Recommend")
router = APIRouter()


class RecommendRequest(BaseModel):
    dataset_size: int = 100
    preset: str = "mid"
    parameter_budget: int | None = None
    seed: int = 20260822


class DatasetCheckRequest(BaseModel):
    file_path: str = ""


def _detect_hardware() -> dict:
    """检测本地硬件，为训练提供配置参考"""
    try:
        import psutil

        cpu_count = psutil.cpu_count(logical=False) or os.cpu_count() or 4
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        cpu_count = os.cpu_count() or 4
        ram_gb = 0
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = ""
        vram_gb = 0.0
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3)
    except ImportError:
        cuda_available = False
        gpu_name = ""
        vram_gb = 0.0
    return {
        "cpu_cores": cpu_count,
        "ram_gb": round(ram_gb, 1),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
    }


_NATIVE_PRESETS = {
    "tiny": {
        "parameter_budget": 100_000,
        "epochs": 1,
        "description": "极轻量（适合 CPU 快速实验）",
    },
    "small": {
        "parameter_budget": 300_000,
        "epochs": 1,
        "description": "小容量（适合消费级 GPU）",
    },
    "mid": {
        "parameter_budget": 1_000_000,
        "epochs": 2,
        "description": "中等容量（适合单卡训练）",
    },
    "large": {
        "parameter_budget": 4_000_000,
        "epochs": 3,
        "description": "大容量（需更充足的内存）",
    },
}


def _native_capacity(budget: int, seed: int) -> dict:
    config = TaijiConfig.capacity_profile(budget, seed=seed)
    return {
        "parameter_budget": budget,
        "planned_active_parameters": config.planned_active_parameter_count,
        "region_sizes": list(config.region_sizes),
        "synapse_fan_in": config.synapse_fan_in,
        "motor_fan_in": config.motor_fan_in,
        "memory_units": config.memory_units,
        "memory_meta_dim": config.memory_meta_dim,
        "memory_time_dim": config.memory_time_dim,
        "memory_episode_dim": config.memory_episode_dim,
        "memory_fan_in": config.memory_fan_in,
        "memory_readout_fan_in": config.memory_readout_fan_in,
        "lateral_fan_in": config.lateral_fan_in,
    }


@router.post("/api/training/recommend")
async def get_training_recommendation(req: RecommendRequest):
    """获取基于本地硬件的原生训练推荐配置"""
    try:
        if req.preset not in _NATIVE_PRESETS:
            raise HTTPException(status_code=400, detail=f"未知原生训练预设: {req.preset}")
        if req.dataset_size < 0:
            raise HTTPException(status_code=400, detail="dataset_size 不能为负数")
        if req.parameter_budget is not None and req.parameter_budget <= 0:
            raise HTTPException(status_code=400, detail="parameter_budget 必须为正数")
        hw = _detect_hardware()
        selected = _NATIVE_PRESETS[req.preset]
        budget = int(req.parameter_budget or cast(int, selected["parameter_budget"]))
        return {
            "status": "success",
            "hardware": hw,
            "runtime": "seed-taiji-native",
            "selected_preset": req.preset,
            "selected": {
                **selected,
                "dataset_size": req.dataset_size,
                "device": "cuda" if hw["cuda_available"] else "cpu",
                "capacity": _native_capacity(budget, req.seed),
            },
            "presets": {
                key: {
                    **value,
                    "capacity": _native_capacity(
                        int(cast(int, value["parameter_budget"])), req.seed
                    ),
                }
                for key, value in _NATIVE_PRESETS.items()
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取训练推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/training/check_dataset")
async def check_dataset_quality(req: DatasetCheckRequest):
    """检查原生 raw-byte 训练数据集质量。"""
    if not req.file_path.strip():
        raise HTTPException(status_code=400, detail="file_path 不能为空")
    try:
        result = inspect_native_dataset(req.file_path)
        return {"status": "success", **result.to_dict()}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"数据集不存在: {req.file_path}") from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据集检查失败: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
