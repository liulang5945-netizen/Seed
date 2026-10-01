"""T1/T2 第二遍审计（一次性，用完即删）。

查两件我上一遍没查的事：

**A. 训练是不是逐位确定的？** 如果同一个 seed + 同一份数据 + 同一段预算跑两遍结果不同，
那"n=1 也能归因"就不成立（消融与对照的差可能只是跑动之间的抖动）。
判据：两次构造 + 训练后的 substrate 载荷**逐节指纹完全相同**。

**B. 筛选集里 `f6` 的槽词不等长（`很累`=6B vs `很精神`=9B）会不会动结论？**
换 B 的同时改了句子长度 ⇒ 那条 `consistency_B` 混进了长度效应。
做法：把 6 族与 **去掉 f6 的 5 族**各量一遍，对比 tick0 与 16M 的读数，
看 §11 的结论（"结构初始化时就有、被训练花掉"）是否仍然成立。
确认族集是等长的（已实测），所以 T1/T2 的确认那一半不受影响。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from probe_taiji_r2_compositionality import FAMILIES  # noqa: E402
from probe_taiji_r2_separability_profile import STAGES, _cells, _cos, _median  # noqa: E402

CLEAN = [f for f in FAMILIES if f["id"] != "f6"]


def _digest(value: object) -> str:
    hasher = hashlib.sha256()

    def walk(node: object) -> None:
        if isinstance(node, torch.Tensor):
            hasher.update(str(tuple(node.shape)).encode())
            hasher.update(node.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(node, dict):
            for key in sorted(node, key=str):
                hasher.update(str(key).encode())
                walk(node[key])
        elif isinstance(node, (list, tuple)):
            hasher.update(str(len(node)).encode())
            for item in node:
                walk(item)
        else:
            hasher.update(repr(node).encode())

    walk(value)
    return hasher.hexdigest()


def measure(substrate: object, families: list[dict[str, object]], tag: str) -> dict[str, float]:
    per_stage: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for family in families:
        cells = _cells(substrate, None, family, tag)
        for stage in STAGES:
            c00, c10 = cells["00"][stage], cells["10"][stage]
            c01, c11 = cells["01"][stage], cells["11"][stage]
            per_stage[stage].append(_cos(c10 - c00, c11 - c01))
    return {stage: round(_median(values), 6) for stage, values in per_stage.items()}


def main() -> int:
    from seed import Seed, SeedConfig
    from taiji import TaijiConfig

    values = dict(
        torch.load(
            PROJECT_ROOT / "checkpoints" / "seed_beta.pt", map_location="cpu", weights_only=False
        )["config"]["taiji"]
    )
    values.pop("receptors_factored", None)
    config = SeedConfig(taiji=TaijiConfig.from_dict(values))

    # ---- A. 确定性 ----
    digests = []
    for run in (1, 2):
        model = Seed(config, episode_id=f"audit-det-{run}")
        for _, symbol in enumerate(b"abcdefgh" * 40):
            model.substrate.observe(int(symbol), readout="action", learn=True)
        digests.append({key: _digest(val) for key, val in model.substrate.checkpoint().items()})
    differing = sorted(key for key in digests[0] if digests[0][key] != digests[1][key])
    determinism = {"identical_sections": not differing, "differing_sections": differing}

    # ---- B. f6 敏感性 ----
    tick0 = Seed(config, episode_id="audit-tick0")
    trained = Seed.from_checkpoint(
        torch.load(
            PROJECT_ROOT / "checkpoints" / "seed_beta.pt", map_location="cpu", weights_only=False
        )
    )
    sensitivity = {}
    for name, families in (("six_families", FAMILIES), ("five_families_no_f6", CLEAN)):
        before = measure(tick0.architecture, families, f"a-{name}-before")
        after = measure(trained.architecture, families, f"a-{name}-after")
        sensitivity[name] = {
            "tick0": before,
            "trained_16m": after,
            "delta": {stage: round(after[stage] - before[stage], 6) for stage in STAGES},
        }

    print(json.dumps({"determinism": determinism}, ensure_ascii=False))
    for name, block in sensitivity.items():
        print(f"  --- {name} ---")
        for stage in STAGES:
            print(
                f"    {stage:22} tick0={block['tick0'][stage]:+.4f} "
                f"16M={block['trained_16m'][stage]:+.4f} delta={block['delta'][stage]:+.4f}"
            )
    out = PROJECT_ROOT / "reports" / "taiji_r2_t1t2_second_audit_20260923.json"
    out.write_text(
        json.dumps(
            {"determinism": determinism, "f6_sensitivity": sensitivity},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
