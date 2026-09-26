"""A2.8-3 检验（零训练）：把"同一字节所有位置求和"换成"取最大"，指对率与整句命中各变多少。

**动因（`SPEC-A-17` §22）**：答案第一个字节上，目标字节的**名次**通常很靠前（前五占 84%）却拿不到第一，
而赢家系统性地是 `E5/E6/E8`——中文最常见的首字节。发射侧把同一字节在句中**所有位置**的质量
`index_add` 相加 ⇒ 位置数会被读成"更相关"。本件只动这一处聚合（`pool_override`，默认 `"sum"` 位级不变），
其余（皮质 cue 选事件、位置打分、发射门、读出）全原样。

**两段读数的顺序不能反**：先看机制是否真的改变指点（屏幕），再花一次整跑看表层（价格）；
并且按 §20 那两条制度要求——轨迹必须是**自己走出来的**，结果指标必须是**整句**的严格命中。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

SEED_A_CIRCUIT = "output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
SURFACE_MANIFEST = "plans/manifests/r2_copy_surface_extension_v1.json"
SURFACE_BASELINE_HITS = 23  # §15 冻结读数里 seed-A 那一臂（只用于报告里标注，不参与计算）


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=SEED_A_CIRCUIT)
    parser.add_argument("--pools", default="sum,max", help="逗号分隔：sum＝现状对照，max＝取最大")
    parser.add_argument("--stage", choices=("screen", "price", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from probe_taiji_r2_a26_emission_trace import extension_items, failing_item_ids_extension
    from probe_taiji_r2_a27_address_onpolicy import trace_all
    from score_taiji_r2_copy_strict_cap import copyable_tokens

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    ids = failing_item_ids_extension()
    if args.limit > 0:
        ids = ids[: args.limit]
    items = extension_items()

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    substrate.copy_circuit.load_payload(
        torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    )
    circuit = substrate.copy_circuit

    screens: dict[str, object] = {}
    for pool in [part.strip() for part in args.pools.split(",") if part.strip()]:
        circuit.pool_override = pool
        try:
            screens[pool] = trace_all(substrate, circuit, runtime, ids, items, copyable_tokens)
        finally:
            circuit.pool_override = "sum"

    prices: dict[str, object] = {}
    if args.stage in ("price", "both"):
        from score_taiji_r2_copy_surface_extension import load_items, run_arm

        surface = load_items(PROJECT_ROOT / SURFACE_MANIFEST)
        for pool in [part.strip() for part in args.pools.split(",") if part.strip()]:
            circuit.pool_override = pool
            try:
                arm = run_arm(surface, checkpoint, args.circuit)
            finally:
                circuit.pool_override = "sum"
            prices[pool] = {
                "strict_hits": arm["strict_hits"],
                "texts": arm["texts"],
                "well_formed_texts": arm["well_formed_texts"],
            }

    report = {
        "format": "taiji-r2-a28-pool-effect-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §22",
        "changed": "只有 pool_override（字节聚合 sum→max）；其余链路原样",
        "circuit": args.circuit,
        "items": len(ids),
        "limit": args.limit,
        "frozen_surface_hits_for_reference": SURFACE_BASELINE_HITS,
        "screen_by_pool": screens,
        "price_by_pool": prices,
        "reading_rule": (
            "max 档第 0 步/整体指对率明显高于 sum ⇒ 位置数被当成质量这一假设成立；"
            "指对升而表层不升 ⇒ 印证 §20 的'点 vs 串'，问题不在聚合；"
            "两者都不动 ⇒ 求和假设被否证，回到查询不分辨那一支"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = Path(args.out_report or "reports/taiji_r2_a28_pool_effect_20260926.json")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "items": len(ids),
                "limit": args.limit,
                "screen": screens,
                "price": prices,
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["base_sha256_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
