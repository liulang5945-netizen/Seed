"""生成 N4 的**产品档**寻址面：材料只由产品自己的写入路径产生（PLAN-N4-05 J-N4e-1）。

与训练侧档（`train_seed_corpus.py --episodic-mount`，cue＝该篇文档前 32 字节 ÷255）的区别：
这里的每条记忆都是产品在一次**落定回合**里自己写的（`Taiji.settle_action` 内的
`self._episodic_memory.write(record)`，`taiji/adapter.py:11663→:11699`），
cue 来自产品认知状态（`percept.features`，缺则 `world.latent`）。
⇒ 本件**不自己往知识库里写条目**（J-N4e-1 的机检点，且必须用 AST 数调用点：
grep 会被文档里提到这个写法时假命中，本轮就踩了一次）。

查询档沿用训练侧那一条机械扰动（末位换成相邻条的末位），但两档 cue 形状不同
⇒ 按 J-N4e-4 **禁止跨档比数**，本件只出产品档自己的数。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, SeedConfig  # noqa: E402
from taiji.config import TaijiConfig  # noqa: E402

TINY = dict(
    region_sizes=(8,),
    synapse_fan_in=2,
    motor_fan_in=4,
    predictive_context_fan_in=2,
    memory_units=16,
    memory_fan_in=2,
    memory_readout_fan_in=2,
    memory_meta_dim=4,
    memory_time_dim=2,
    memory_episode_dim=2,
    lateral_fan_in=2,
    identity_organ_capacity=8,
    concept_capacity=8,
    seed=101,
    episodic_memory_default_mount=True,
)
ACTIONS = (0, 1, 2, 3)


def _rotate_to(cue: list[float], donor: list[float], theta: float) -> tuple[list[float], float]:
    """把 cue 朝 donor 方向旋转 `theta` 弧度，返回（新 cue，实达角度）。

    扰动形状按 PLAN-N4-06 §1 冻死：在 cue 的正交补上取 donor 分量（Gram–Schmidt），
    只依赖几何、不看寻址结果 ⇒ 不存在"调到某个值让数好看"的路径。
    """
    norm_cue = math.sqrt(sum(v * v for v in cue))
    if norm_cue == 0.0:
        raise ValueError("cue 全零，无法定义角度扰动")
    unit = [v / norm_cue for v in cue]
    dot = sum(d * u for d, u in zip(donor, unit))
    perp = [d - dot * u for d, u in zip(donor, unit)]
    norm_perp = math.sqrt(sum(v * v for v in perp))
    if norm_perp < 1e-9:
        raise ValueError("donor 与 cue 共线，无法构造正交方向（换 donor）")
    perp = [v / norm_perp for v in perp]
    rotated = [math.cos(theta) * u + math.sin(theta) * v for u, v in zip(unit, perp)]
    achieved = math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(unit, rotated)))))
    return rotated, achieved


def build(
    episodes: int, symbols: int, jitter_seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = Seed(SeedConfig(taiji=TaijiConfig(**TINY)))  # type: ignore[arg-type]
    store = model.substrate._episodic_memory
    if store is None:
        raise RuntimeError("默认挂载位未生效：产品没有自建 store")
    rng = random.Random(jitter_seed)
    cue_sources = set()
    for index in range(episodes):
        payload = bytes(rng.randrange(32, 256) for _ in range(symbols))
        for symbol in payload:
            model.observe(symbol, learn=True)
        model.substrate.act(ACTIONS, sample=False)
        model.substrate.settle_action(1.0 if index % 2 == 0 else -1.0, terminal=True)
        record = store.records[-1]
        cue_sources.add("percept.features" if record.cue.numel() else "empty")
    materials = [
        {"memory_id": str(record.memory_id), "cue": [float(v) for v in record.cue.tolist()]}
        for record in store.records
    ]
    meta: dict[str, Any] = {
        "format": "taiji-n4-product-tier-faces-v1",
        "mount_layer": "product",
        "product_default_mounted": True,
        "write_trigger": "settle_action",
        "cue_source": "percept.features 优先，缺则 cognitive_state.world.latent",
        "j_n4e_1_check": "AST-based; grep 会假命中本文档提法（本轮实测）",
        "store_write_call_sites_this_file": 0,
        "episodic_writes": int(store.count),
        "cue_dim": int(store.cue_dim),
        "episodes_requested": int(episodes),
        "symbols_per_episode": int(symbols),
        "jitter_seed": int(jitter_seed),
        "observed_record_shapes": sorted(int(record.cue.numel()) for record in store.records[:1]),
        "cue_sources_seen": sorted(cue_sources),
    }
    return materials, meta


def pairs(
    materials: list[dict[str, Any]], angle: float | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """`angle=None` 沿用训练侧档的末位替换（只为与历史可比）；给了角度则按 PLAN-N4-06 旋转。"""
    queries: list[dict[str, Any]] = []
    achieved: list[float] = []
    if angle is None:
        for offset, row in enumerate(materials):
            donor = materials[(offset + 1) % len(materials)]
            cue = list(row["cue"])
            if cue:
                cue[-1] = donor["cue"][-1]
            queries.append({"expected_memory_id": row["memory_id"], "cue": cue})
        note = {"perturbation": "last_component_swap", "angle_nominal_rad": None}
        return queries, {**note, "achieved_angle_rad": [], "queries": len(queries)}
    for offset, row in enumerate(materials):
        donor = materials[(offset + 1) % len(materials)]
        rotated, achieved_rad = _rotate_to(list(row["cue"]), list(donor["cue"]), float(angle))
        achieved.append(achieved_rad)
        queries.append({"expected_memory_id": row["memory_id"], "cue": rotated})
    note = {
        "perturbation": "rotate_toward_donor",
        "angle_nominal_rad": float(angle),
        "queries": len(queries),
        "achieved_angle_min": min(achieved),
        "achieved_angle_max": max(achieved),
        "achieved_angle_mean": sum(achieved) / len(achieved),
        #: J-N4f-4 的机检点之一：实达与名义偏差 >0.02 rad ⇒ 扰动实现有缺陷，不出读数。
        "achieved_within_0p02_of_nominal": bool(
            all(abs(a - float(angle)) <= 0.02 for a in achieved)
        ),
    }
    return queries, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 N4 产品档寻址面（材料由产品写入路径产生）")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--symbols", type=int, default=20)
    parser.add_argument("--jitter-seed", type=int, default=20261009)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--angle",
        type=float,
        default=None,
        help="查询扰动角度（弧度）。不给＝沿用训练侧档的末位替换；"
        "PLAN-N4-06 的两档是 0.35 与 0.70，判据先冻、不许按结果回头调。",
    )
    args = parser.parse_args(argv)

    if args.episodes <= 0 or args.symbols <= 0:
        parser.error("--episodes 与 --symbols 必须为正")
    if args.angle is not None and not 0.0 < args.angle < 1.5707963267948966:
        parser.error("--angle 必须落在 (0, pi/2) 之间")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    materials, meta = build(args.episodes, args.symbols, args.jitter_seed)
    if not materials:
        print("REJECT no_product_writes")
        return 2
    newline = "\n"
    (out / "product_materials.jsonl").write_text(
        newline.join(json.dumps(row, ensure_ascii=False) for row in materials) + newline,
        encoding="utf-8",
        newline=newline,
    )
    queries, perturbation = pairs(materials, args.angle)
    (out / "perturbation_meta.json").write_text(
        json.dumps(perturbation, ensure_ascii=False, indent=2) + newline,
        encoding="utf-8",
        newline=newline,
    )
    (out / "product_queries.jsonl").write_text(
        newline.join(json.dumps(row, ensure_ascii=False) for row in queries) + newline,
        encoding="utf-8",
        newline=newline,
    )
    (out / "faces_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + newline, encoding="utf-8", newline=newline
    )
    print(json.dumps({**meta, "materials_written": len(materials), "queries": len(queries)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
