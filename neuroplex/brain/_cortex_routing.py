"""Cortex 路由纯函数（B-3 C-3 簇，从 `neuroplex/brain/cortex.py` 抽离，2026-09-26）。

设计约束（冻结基线）：**逐行搬移、只把 self 依赖提升为参数**，不改任何语句——
等价性由黄金向量（`reports/cortex_fingerprint_golden_20260926.json`）与活体对照测试保证。

留壳（状态/编排耦合）：`_executive_route`（调 think + EMA 状态）、`_auto_topk_route`（依赖 ensemble + 多个活体对象）、
`_instance_route_evolve`（1017 行、30 个 self 引用）⇒ 属 C-4 生成编排范畴。
"""

from __future__ import annotations

# B-3 C-3 抽离时（2026-09-26）漏掉的依赖：`torch.tensor/torch.long` 在相似度分支使用。
# ⚠️ 危害被 `except Exception` 掩盖：缺 import ⇒ NameError 被吞 ⇒ 恒走 fallback
#    （返回全部 neuron），相似度路由静默失效。由 test_cortex_routing_golden 的合成
#    embed_adapter 神经元用例钉住（黄金夹具无 embed_adapter ⇒ 该分支原本**未覆盖**）。
import torch

__all__ = ["fingerprint_route"]



def fingerprint_route(
    neurons: dict,
    shared_embedding,
    device,
    general_ids: list[int],
    top_k: int = 2,
) -> list[str]:
    """Level 2 prototype 路由：用 domain_prototype cosine 相似度选 top-k neuron。

    每个 neuron 用自己的 embed_adapter 投影 prompt，再与自己的 domain_prototype
    做 cosine。每个 neuron 用自己的视角"看"prompt，符合神经元独立性。

    Args:
        general_ids: prompt 的 general tokenizer id 列表。
        top_k: 选择的 neuron 数量（不含 general）。

    Returns:
        active neuron id 列表。
    """
    if not neurons or shared_embedding is None:
        return list(neurons.keys())

    try:
        ids_tensor = torch.tensor([general_ids], dtype=torch.long, device=device)
        prompt_emb = shared_embedding(ids_tensor)  # [1, L, 512]
        prompt_pooled = prompt_emb.mean(dim=1)  # [1, 512]
    except Exception:
        return list(neurons.keys())

    # 每个 neuron 用自己的 embed_adapter 投影 prompt，再与 prototype 比较
    # C5: 多原型模式取 max cosine（与最近原型的相似度）
    sims = {}
    for nid, neuron in neurons.items():
        try:
            if hasattr(neuron, "embed_adapter") and neuron.embed_adapter is not None:
                # 用 neuron 自己的 embed_adapter 投影到 768 维
                projected = neuron.embed_adapter(prompt_pooled)  # [1, 768]
                proj_vec = projected.squeeze(0)  # [768]
                proj_norm = proj_vec / (proj_vec.norm() + 1e-8)
                # C5: 多原型取 max cosine
                if (
                    getattr(neuron, "num_prototypes", 1) > 1
                    and neuron.domain_prototypes is not None
                ):
                    # 多原型: [K, 768] → max cosine
                    protos = neuron.domain_prototypes  # [K, 768]
                    proto_norms = protos / (protos.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = float((proj_norm.unsqueeze(0) * proto_norms).sum(dim=-1).max().item())
                else:
                    # 单原型（向后兼容）
                    proto = neuron.domain_prototype  # [768]
                    proto_norm = proto / (proto.norm() + 1e-8)
                    sim = float((proj_norm * proto_norm).sum().item())
            else:
                # fallback: 无 embed_adapter 则跳过
                continue
            sims[nid] = sim
        except Exception:
            continue

    if not sims:
        return list(neurons.keys())

    # 按相似度排序，选 top-k（排除 general，单独保证）
    sorted_nids = sorted(sims, key=sims.get, reverse=True)
    non_general = [nid for nid in sorted_nids if nid != "general"]
    selected = non_general[:top_k]

    # general 始终包含
    if "general" in neurons and "general" not in selected:
        selected.append("general")

    return selected if selected else list(neurons.keys())
