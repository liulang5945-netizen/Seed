"""N38 Open-Chinese Unified Cognitive Backbone (`taiji-n38-open-zh-backbone-v1`).

Native PyTorch causal Transformer with RMSNorm, RoPE, SwiGLU, GQA, tied token
embeddings, and a persistent cross-session memory slot bank (`n_mem_slots`)
that can be exported/reloaded bit-identically via `SessionStore.capsule`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA = "taiji-n38-open-zh-backbone-v1"


@dataclass(frozen=True)
class BackboneConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 10
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1024
    max_seq_len: int = 512
    n_mem_slots: int = 8
    rope_theta: float = 10000.0
    eps: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, BackboneConfig] = {
    "small_8m": BackboneConfig(
        vocab_size=8192,
        d_model=256,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        d_ff=768,
        max_seq_len=512,
        n_mem_slots=8,
    ),
    "mid_18m": BackboneConfig(
        vocab_size=8192,
        d_model=384,
        n_layers=10,
        n_heads=6,
        n_kv_heads=2,
        d_ff=1024,
        max_seq_len=512,
        n_mem_slots=8,
    ),
    "large_30m": BackboneConfig(
        vocab_size=8192,
        d_model=512,
        n_layers=10,
        n_heads=8,
        n_kv_heads=2,
        d_ff=1280,
        max_seq_len=512,
        n_mem_slots=8,
    ),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * c - x2 * s
    out2 = x1 * s + x2 * c
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class GQAAttention(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        n_mem: int,
    ) -> torch.Tensor:
        bsz, total_len, _ = x.shape
        q = self.q_proj(x).view(bsz, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, total_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, total_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE only to text tokens (keep persistent memory slots position-invariant)
        if total_len > n_mem:
            q_mem, q_txt = q[:, :, :n_mem, :], q[:, :, n_mem:, :]
            k_mem, k_txt = k[:, :, :n_mem, :], k[:, :, n_mem:, :]
            q_txt = _apply_rope(q_txt, cos, sin)
            k_txt = _apply_rope(k_txt, cos, sin)
            q = torch.cat((q_mem, q_txt), dim=2)
            k = torch.cat((k_mem, k_txt), dim=2)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(bsz, total_len, -1)
        return cast(torch.Tensor, self.o_proj(out))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.attn = GQAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None,
        n_mem: int,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, attn_mask, n_mem)
        x = x + self.mlp(self.norm2(x))
        return x


class TaijiOpenChineseLM(nn.Module):
    """Causal LM with persistent cross-session memory slots."""

    def __init__(self, cfg: BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Persistent memory slot gated update
        self.mem_gate = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.mem_cand = nn.Linear(cfg.d_model * 2, cfg.d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_mem_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.cfg.n_mem_slots, self.cfg.d_model, device=device, dtype=torch.float32
        )

    def _build_prefix_causal_mask(
        self, n_mem: int, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        total = n_mem + seq_len
        mask = torch.zeros((total, total), device=device, dtype=torch.bool)
        # Memory slots attend bidirectionally to each other (not to future text)
        mask[:n_mem, :n_mem] = True
        # Text tokens attend to all memory slots + causal prefix of text
        mask[n_mem:, :n_mem] = True
        mask[n_mem:, n_mem:] = torch.tril(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mem_state: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {seq_len} > max_seq_len {self.cfg.max_seq_len}")
        if mem_state is None:
            mem_state = self.init_mem_state(bsz, device=input_ids.device)

        x_txt = self.tok_emb(input_ids)
        mem_in = mem_state.to(dtype=x_txt.dtype)
        x = torch.cat((mem_in, x_txt), dim=1)

        head_dim = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _build_rope_cache(
            seq_len, head_dim, self.cfg.rope_theta, input_ids.device, x.dtype
        )
        attn_mask = self._build_prefix_causal_mask(self.cfg.n_mem_slots, seq_len, input_ids.device)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask, self.cfg.n_mem_slots)

        x = self.norm_f(x)
        mem_out = x[:, : self.cfg.n_mem_slots, :]
        txt_out = x[:, self.cfg.n_mem_slots :, :]
        logits = self.lm_head(txt_out)

        # Gated persistent memory update
        txt_summary = txt_out.mean(dim=1, keepdim=True).expand_as(mem_out)
        pair = torch.cat((mem_out, txt_summary), dim=-1)
        gate = torch.sigmoid(self.mem_gate(pair))
        cand = torch.tanh(self.mem_cand(pair))
        next_mem_state = ((1.0 - gate) * mem_in + gate * cand).to(dtype=torch.float32)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=0,
            )
        return logits, loss, next_mem_state

    def export_capsule(self, prefix: str = "open_zh") -> dict[str, torch.Tensor]:
        return {f"{prefix}.{k}": v.detach().cpu().clone() for k, v in self.state_dict().items()}

    def load_capsule(self, capsule: dict[str, torch.Tensor], prefix: str = "open_zh") -> None:
        head = f"{prefix}."
        sub = {k[len(head) :]: v for k, v in capsule.items() if k.startswith(head)}
        self.load_state_dict(sub, strict=True)
