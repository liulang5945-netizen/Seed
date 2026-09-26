"""产品基座"增长普查"：16M tick 到底把**哪些参数面**推动了、哪些一动没动。

用户的问题（2026-09-26）："这个 16M 的规模到底是增长的什么"。
先钉住口径：`metadata.tick = 16000000` 是**训练步/符号数**，不是参数量——
参数量由架构与 `config` 定死，训练不改形状（唯一会改形状的是 identity organ 的显式扩容，
它有独立的 growth history）。所以"增长了什么"必须答成三件事：

1. **哪些面动了**：与"同 config 同种子新建的初始态"逐面对比（初始态是可复现的，
   本仓有位级不变的钉子测试为证）；
2. **哪些面仍是零**：全零面＝从未通电（`SPEC-A-22` §23 已点名查到三个巩固解码头，
   本件做**全量**普查，不再点名）；
3. **规模本身有没有长**：`structural_events`／`development_ticks`／identity growth history／
   情节库计数——这些是"结构性增长"的账，与"权重被训练推动"是两回事。

纪律：只读；不写 `checkpoints/`；不改 config（`TaijiConfig` 是 frozen dataclass）；
基座 sha256 跑前后逐位相同；普查面数必须非零，否则退出码 2（仪器不许静默交空表）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(node: Any, prefix: str, out: dict[str, torch.Tensor]) -> None:
    """递归收集 payload 里所有张量，键＝点路径。不点名 ⇒ 不会因为漏写路径而少报面。"""
    if torch.is_tensor(node):
        out[prefix] = node.detach().cpu().float()
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{prefix}.{key}" if prefix else str(key), out)
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, f"{prefix}[{index}]", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    metadata = payload.get("metadata", {})

    substrate_tensors: dict[str, torch.Tensor] = {}
    _walk(payload.get("substrate", {}), "substrate", substrate_tensors)

    #: 同 config 的初始态（不喂任何数据）：这就是"16M tick 之前"的参照。
    from taiji import Taiji, TaijiConfig

    config_payload = payload["substrate"].get("config") or payload.get("config")
    config = (
        config_payload
        if isinstance(config_payload, TaijiConfig)
        else (
            TaijiConfig.from_payload(config_payload)
            if hasattr(TaijiConfig, "from_payload")
            else TaijiConfig(**config_payload)
        )
    )
    twin = Taiji(config)
    twin_tensors: dict[str, torch.Tensor] = {}
    #: 用 `checkpoint()`（`model.py:3210`）而不是 `to_payload()`——Taiji 没有后者；
    #: 检查点文件里 `substrate` 那一段就是它产出的同构字典，所以两边点路径可比。
    _walk(twin.checkpoint(), "substrate", twin_tensors)

    rows = []
    for path, tensor in sorted(substrate_tensors.items()):
        init = twin_tensors.get(path)
        norm = float(tensor.norm())
        identical_to_init = (
            None if init is None or init.shape != tensor.shape else bool(torch.equal(tensor, init))
        )
        rows.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "norm": round(norm, 6),
                "abs_max": round(float(tensor.abs().max()), 6) if tensor.numel() else 0.0,
                "all_zero": bool(tensor.numel()) and float(tensor.abs().sum()) == 0.0,
                "identical_to_init": identical_to_init,
                "delta_norm_vs_init": (
                    None
                    if init is None or init.shape != tensor.shape
                    else round(float((tensor - init).norm()), 6)
                ),
            }
        )

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    structural = {
        "tick_metadata": metadata.get("tick"),
        "saved_at_utc": metadata.get("saved_at_utc"),
        "corpus_fingerprint": metadata.get("corpus_fingerprint"),
        "profile": metadata.get("profile"),
        "state_tick": int(getattr(substrate._state, "tick", -1)),
        "development_ticks": int(getattr(substrate, "_development_ticks", -1)),
        "fabric_structural_events": int(getattr(substrate.fabric, "structural_events", -1)),
        "memory_write_count": getattr(substrate.memory, "write_count", None),
        "identity_growth_history_len": len(
            getattr(substrate, "_identity_growth_history", ()) or ()
        ),
        "parameter_count_active": int(substrate.parameter_count(active_only=True)),
        "parameter_count_all": int(substrate.parameter_count(active_only=False)),
    }

    moved = [row for row in rows if row["delta_norm_vs_init"] not in (None, 0.0)]
    report = {
        "format": "taiji-substrate-growth-census-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §5 A3",
        "checkpoint": args.checkpoint,
        "structural": structural,
        "summary": {
            "surfaces_enumerated": len(rows),
            "surfaces_all_zero": sum(1 for row in rows if row["all_zero"]),
            "surfaces_moved_vs_init": len(moved),
            "surfaces_identical_to_init": sum(
                1 for row in rows if row["identical_to_init"] is True
            ),
            "surfaces_not_in_twin": sum(1 for row in rows if row["identical_to_init"] is None),
            "total_numel": sum(row["numel"] for row in rows),
            "moved_numel": sum(row["numel"] for row in moved),
        },
        "all_zero_surfaces": [row["path"] for row in rows if row["all_zero"]],
        "top_moved": sorted(moved, key=lambda row: -row["delta_norm_vs_init"])[:20],
        "rows": rows,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "surfaces_nonzero": report["summary"]["surfaces_enumerated"] > 0,
        "twin_covers_checkpoint": report["summary"]["surfaces_not_in_twin"] == 0,
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "summary": report["summary"],
                "structural": {
                    key: structural[key]
                    for key in (
                        "tick_metadata",
                        "development_ticks",
                        "fabric_structural_events",
                        "memory_write_count",
                        "parameter_count_active",
                    )
                },
                "all_zero": report["all_zero_surfaces"][:14],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
