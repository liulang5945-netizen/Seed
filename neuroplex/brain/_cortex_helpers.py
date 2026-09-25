"""
Cortex 纯算法辅助函数（从 neuroplex/brain/cortex.py 抽离，P2 拆分第一步）

设计约束（冻结基线）：本模块只放**无 self 状态、纯函数**的算法，
确保从 Cortex 方法体迁移到此处时逐位等价、不改任何推理数学。
后续 Cortex 大拆分（路由集群 / 域推断集群 / 生成集群）以本文件为样板，
每迁一簇都要配同等数值/行为等价测试。
"""

import re

import torch

__all__ = ["detect_modality", "fuse_leader_quality", "is_degenerate_text"]


def is_degenerate_text(text: str) -> bool:
    """R9（REMEDIATION_PLAN 2026-08-14）：退化输出检测（与 verify 脚本同规则）。

    三类已知退化（欠训练 dialogue neuron 实测）：
    - 编号列表塌缩：`1.\\n` 或字面量 `1.<0x0A>`（裸 prompt 死循环残留）
    - 重复标点/字符：同字符连续 >=4
    - 纯数字长串：剔除数字/空白/运算符/标点后仍 >=2 个非汉字非字母字符
      （>=2 排除 "1 + " 这类短算术 stub——那是截断不是退化循环）
    """
    if not text:
        return False
    if re.search(r"\d+\.\s*\n", text) or "1.<0x0A>" in text:
        return True
    if re.search(r"([。！？，,.！、]{2})\1{1,}", text) or re.search(
        r"(.)\1{3,}", text.replace("……", "。。")
    ):
        return True
    stripped = re.sub(r"[0-9\s,，。.!！?？;；:：、+\-*/=×÷\"\'“”（）()<>]", "", text)
    return bool(stripped) and len(stripped) >= 2 and not re.search(r"[\u4e00-\u9fff\w]", stripped)


def detect_modality(input_data: str | torch.Tensor | dict) -> str:
    """P8: 检测输入数据的模态（从 `cortex.py` 抽离，纯函数、逐位等价）。

    路由顺序：
    1. 显式 dict {"modality": "image", "data": ...} → 直接取
    2. torch.Tensor → 根据维度推断（3D float → image/audio 连续特征）
    3. str → "text"

    Args:
        input_data: str / torch.Tensor / dict

    Returns:
        modality name ("text"/"image"/"audio"/"video")
    """
    if isinstance(input_data, dict):
        return input_data.get("modality", "text")
    if isinstance(input_data, torch.Tensor):
        # [B, L, raw_dim] float → 连续特征（图像/音频）
        if input_data.dim() == 3 and input_data.dtype != torch.long:
            # 默认归为 image，具体模态由调用方通过 dict 指定
            return "image"
        # [B, L] long → token id（文本或离散化的多模态）
        return "text"
    return "text"


def _minmax(values: dict) -> dict:
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return {k: 0.5 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def fuse_leader_quality(resonance_scores: dict, nll_quality: dict, alpha: float = 0.5) -> dict:
    """连续 leader 融合分数（共振分与 -NLL 域内 min-max 归一化后加权）。

    Args:
        resonance_scores: {nid: round1_scores}（t=0 场共振分，越大越强）
        nll_quality: {nid: -NLL}（生成质量，越大越好）
        alpha: 共振分权重（0.5 = 等权；质量权重 = 1-alpha）

    Returns:
        {nid: fused_score}（仅含两信号都有的 neuron；空 → 调用方回退）
    """
    common = [k for k in nll_quality if k in resonance_scores]
    if not common:
        return {}
    r_norm = _minmax({k: resonance_scores[k] for k in common})
    q_norm = _minmax({k: nll_quality[k] for k in common})
    return {k: alpha * r_norm[k] + (1 - alpha) * q_norm[k] for k in common}
