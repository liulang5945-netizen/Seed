"""记忆三层的**接通性审计**（零训练、只读、不改产品代码）。

动机（2026-09-26 用户的假设）："架构问题可能没那么大,可能是我们都没接进去。"
这句话必须拆成两件可分别测量的事,否则会变成又一次"涨了分但说不清机制":

* **接线**：默认生成链上,情节记忆到底**读不读**、**写不写**；
* **通电**：就算读了,那条通路的权重里**有没有东西**（本仓已登记过"六个面训练后仍全零"）。

三项测量,每项都自带"被走到"的计数（分母为 0 就不出结论,退出码 2）：

1. `inventory`：产品检查点里各参数面的范数（哪些面全零＝没通电）、情节场的写入计数、
   两个读增益的配置值；
2. `episodic_read_ab`：同一批 prompt,`Taiji.generate(use_memory=False)`（**产品默认**）
   vs `use_memory=True` —— 逐字节比 argmax 与概率的 L1 距离。全同 ⇒ 这条读通路在默认链上
   要么没接、要么接了也没电；
3. `consolidation_probe`：在"提问读完那一刻"的状态上,对比带／不带
   `consolidation_read_gain` 那一份证据的下一字节分布——**不改配置**（`TaijiConfig` 是
   frozen dataclass,而且改配置＝改被测对象）,喂法复用训练器已登记的 `_feed`；

纪律：**不重抄生成链**。两项 A/B 都直接调 `Taiji.generate`（产品那一条）,只动一个开关；
基座 `checkpoints/*.pt` 跑前后 sha256 必须逐位相同。
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

MAX_ANSWER_BYTES = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(substrate: Any) -> dict[str, Any]:
    zeros: list[str] = []
    norms: dict[str, float] = {}
    for name, tensor in (
        substrate.named_parameter_tensors() if hasattr(substrate, "named_parameter_tensors") else []
    ):
        norms[name] = round(float(tensor.norm()), 6)
        if float(tensor.abs().sum()) == 0.0:
            zeros.append(name)
    if not norms:
        #: 没有"带名字的参数面"这个读取面时,退回到分组张量＋已知面名,别静默报"没有零面"。
        for name in (
            "motor.synapses.edge_weight",
            "motor.bias",
            "predictive_context.recurrent.edge_weight",
            "predictive_readout.synapses.edge_weight",
            "predictive_readout.bias",
        ):
            node: Any = substrate
            for part in name.split("."):
                node = getattr(node, part)
            norms[name] = round(float(node.norm()), 6)
            if float(node.abs().sum()) == 0.0:
                zeros.append(name)
        for index, decoder in enumerate(substrate.fabric.consolidation_decoders):
            for pname, tensor in (
                decoder.named_parameters() if hasattr(decoder, "named_parameters") else []
            ):
                key = f"fabric.consolidation_decoders[{index}].{pname}"
                norms[key] = round(float(tensor.norm()), 6)
                if float(tensor.abs().sum()) == 0.0:
                    zeros.append(key)
    #: 快／慢两组读出头的权重范数必须进产物：判断"通了电没有"靠的就是这两个数,
    #: 只留在终端上＝读数不可复核。`SparseSynapses` 直接持有 `edge_weight`（没有 `.synapses`）。
    for group in ("decoders", "consolidation_decoders"):
        for index, decoder in enumerate(getattr(substrate.fabric, group, ())):
            weight = getattr(decoder, "edge_weight", None)
            if weight is None:
                continue
            key = f"fabric.{group}[{index}].edge_weight"
            norms[key] = round(float(weight.norm()), 6)
            if float(weight.abs().sum()) == 0.0:
                zeros.append(key)
    memory = substrate.memory
    return {
        "parameter_norms": norms,
        "all_zero_surfaces": zeros,
        "memory_write_count": getattr(memory, "write_count", None),
        "memory_records": getattr(memory, "count", None),
        "consolidation_read_gain": float(substrate.config.consolidation_read_gain),
        "memory_read_gain": float(substrate.config.memory_read_gain),
    }


def _compare(left: bytes, right: bytes) -> dict[str, Any]:
    n = min(len(left), len(right))
    same = sum(1 for i in range(n) if left[i] == right[i])
    return {
        "len_left": len(left),
        "len_right": len(right),
        "compared_bytes": n,
        "identical_bytes": same,
        "differing_bytes": n - same,
    }


def _run(prompts: list[str], substrate: Any, *, use_memory: bool) -> list[bytes]:
    return [
        substrate.generate(
            prompt.encode("utf-8"),
            MAX_ANSWER_BYTES,
            stop_at_boundary=True,
            sample=False,
            use_memory=use_memory,
        )
        for prompt in prompts
    ]


def _consolidation_probe(substrate: Any, prompts: list[str]) -> list[dict[str, Any]]:
    """在"提问读完、第一个答案字节还没发"那一刻,量慢通路承载多少。

    **不改任何产品状态**：`TaijiConfig` 是 frozen dataclass,把增益置零那条路走不通,
    而且改配置＝改被测对象。这里换成同一状态下的两次读数对比（带／不带
    `consolidation_read_gain` 那一份证据）——等价于把该通路摘掉,但不落到 config 上。

    喂法**复用训练器里已登记的 `_feed`**（`train_taiji_r2_copy_circuit.py`，其 docstring 写明
    与产品 `generate` 的喂法逐字对应：边界符＋逐字节 `observe(learn=False, readout="predictive",
    use_memory=False)`）⇒ 不在新仪器里重抄生成链。
    """
    from train_taiji_r2_copy_circuit import _feed

    rows: list[dict[str, Any]] = []
    gain = float(substrate.config.consolidation_read_gain)
    for prompt in prompts:
        substrate.reset_dynamics(episode_id="memory-wiring-audit")
        _feed(substrate, prompt.encode("utf-8"))
        state = substrate._state
        evidence = gain * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
        readout = substrate.predictive_readout
        context = state.motor_context
        with_path = readout.probabilities(context, episodic_evidence=evidence)
        without = readout.probabilities(context, episodic_evidence=torch.zeros_like(evidence))
        rows.append(
            {
                "evidence_norm": round(float(evidence.norm()), 6),
                "evidence_abs_max": round(float(evidence.abs().max()), 6),
                "argmax_changed": int(with_path.argmax() != without.argmax()),
                "prob_l1_distance": round(float((with_path - without).abs().sum()), 8),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--manifest",
        default="plans/manifests/r2_copy_surface_extension_v3_position_random.json",
        help="取提问文本用（只取最后一轮），不评分",
    )
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    from score_taiji_r2_copy_surface_extension import load_items

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    items = load_items(
        PROJECT_ROOT / args.manifest
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )[: args.limit]
    prompts = [item["turns"][-1] for item in items]

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    report: dict[str, Any] = {
        "format": "taiji-memory-wiring-audit-v1",
        "prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md §21",
        "checkpoint": args.checkpoint,
        "prompts": len(prompts),
        "inventory": _inventory(substrate),
    }

    #: A/B 1：情节记忆的**读**通路（产品默认 use_memory=False）。
    off = _run(prompts, substrate, use_memory=False)
    on = _run(prompts, substrate, use_memory=True)
    report["episodic_read_ab"] = {
        "pairs": len(prompts),
        "byte_level": [_compare(a, b) for a, b in zip(off, on, strict=False)],
        "identical_outputs": sum(1 for a, b in zip(off, on, strict=False) if a == b),
    }

    #: A/B 2：睡眠巩固出来的**慢通路**——不改配置,同状态两次读数对比。
    consolidation = _consolidation_probe(substrate, prompts)
    report["consolidation_probe"] = {
        "pairs": len(consolidation),
        "rows": consolidation,
        "argmax_changed_count": sum(row["argmax_changed"] for row in consolidation),
        "mean_evidence_norm": round(
            sum(row["evidence_norm"] for row in consolidation) / max(len(consolidation), 1), 6
        ),
        "max_prob_l1": max((row["prob_l1_distance"] for row in consolidation), default=0.0),
    }

    #: A/B 3：产品一轮 chat 之后,情节场有没有被写过。
    before = getattr(substrate.memory, "write_count", None)
    runtime.chat(
        prompts[0], history=[], max_length=MAX_ANSWER_BYTES, learn=True, repetition_penalty=0.0
    )
    report["write_path"] = {
        "write_count_before": before,
        "write_count_after": getattr(substrate.memory, "write_count", None),
    }

    report["instrument_guard"] = {
        "prompts_nonzero": bool(prompts),
        "ab_pairs_measured": report["episodic_read_ab"]["pairs"] > 0,
        "consolidation_pairs_measured": report["consolidation_probe"]["pairs"] > 0,
        #: 配置不许被动过：审计若改了被测对象,读数就不是产品那条链的。
        "config_untouched": float(substrate.config.consolidation_read_gain)
        == report["inventory"]["consolidation_read_gain"],
    }
    report["base_sha256_unchanged"] = _sha256(checkpoint) == sha_before

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values()) and report["base_sha256_unchanged"]
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "zero_surfaces": report["inventory"]["all_zero_surfaces"][:12],
                "memory_write_count": report["inventory"]["memory_write_count"],
                "episodic_identical": f"{report['episodic_read_ab']['identical_outputs']}/{len(prompts)}",
                "consolidation_argmax_changed": (
                    f"{report['consolidation_probe']['argmax_changed_count']}/{len(prompts)}"
                ),
                "consolidation_mean_evidence_norm": report["consolidation_probe"][
                    "mean_evidence_norm"
                ],
                "write_path": report["write_path"],
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
