"""A2.7-4 on-policy 重测（零训练）：那套"指对 95%"的寻址，在**它自己走出来的轨迹**上还准不准。

§18 量到"同一层线性变换存在把指对率做到 94.5% 的取值"；§19 把它装回发射链跑表层，
**反而掉分**（v1 23→12、v2 16→1）。两种解释都成立，本件就是把它们分开：

* **(a) 分布位移吃掉全部离线收益**：那个 95% 是在旧策略走到的状态点上量的，换解后模型发的字变了、
  后面每步看到的东西也变了 ⇒ 新解在自己轨迹上并不指对。
  ⇒ 结论是制度性的：**任何改寻址的方案（含 A2.3 重训、A2.5 学习器）都必须自带 on-policy 读数**。
* **(b) 收益还在，掉分在别处**：换位置分布顺带改了门读到的 `pooled`（机制本来就连着）。
  ⇒ 下一刀改测那道闸。

**本件自己踩过的坑，写在码里当注释**：轨迹件消费的是 `addressing()`，而生产发射消费的是
`_position_weights()`（`taiji/copy_circuit.py:262` vs `:291` 自己重算 softmax）——
**只打后者，轨迹那一跑会静默按原样走**，正好是 §18 附记那个假阴性的翻版。
所以这里两处分开发子，并加一条**锚点**：把覆写设回"训练后的 M＋训练后的偏置"，
`addressing` 的分数必须与原实现逐位吻合（容差内、argmax 必须相同），否则整套 on-policy 读数作废。

纪律：零训练（不改任何持久权重，只在进程内覆写、跑完复原）；`checkpoints/` 只读并跑前后复核 sha256；
题号从 §13 的 scored 读数机器读取；只写 `reports/` 一份新件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

FEATURES = PROJECT_ROOT / "output/a27_address_features.json"
SEED_A_CIRCUIT = "output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AddressingOverride:
    """在进程内把"该看哪个位置"换成 `f1·M·embed[c]/scale + bias·[前驱匹配]`。

    同时覆盖**两个消费点**：`addressing()`（轨迹件用）与 `_position_weights()`（生产发射用）。
    只覆盖一个就会得到一份静默未生效的读数——§18 附记已经为这个坑付过一次代价。
    """

    def __init__(self, m: torch.Tensor, bias: float) -> None:
        self.m = m.to(torch.float64)
        self.bias = float(bias)
        self.calls = 0
        self.diverged = 0
        self._originals: dict[str, Any] = {}

    def _weights(self, self_obj: Any, event: Any, f1_context: Any, prev_byte: Any, scale: float):
        codes = torch.tensor(list(event.content), device=self_obj.device, dtype=torch.long)
        embed = self_obj._parameters["content_embed"][codes].detach().cpu().to(torch.float64)
        scores = (f1_context.detach().cpu().to(torch.float64) @ self.m) @ embed.T / scale
        if prev_byte is not None:
            bonus = torch.zeros_like(scores)
            for position in range(1, len(codes)):
                if int(codes[position - 1]) == int(prev_byte):
                    bonus[position] = 1.0
            scores = scores + self.bias * bonus
        return scores.to(torch.float32)

    def install(self) -> None:
        from taiji.copy_circuit import CopyCircuit

        original_addressing = CopyCircuit.addressing
        original_weights = CopyCircuit._position_weights
        outer = self

        def patched_addressing(self, *, cue, f1_context, prev_byte=None):
            snap = original_addressing(self, cue=cue, f1_context=f1_context, prev_byte=prev_byte)
            if snap is None:
                return None
            scale = math.sqrt(float(self.evidence_width))
            scores = outer._weights(self, snap["event"], f1_context, prev_byte, scale)
            outer.calls += 1
            with torch.no_grad():
                if int(scores.argmax()) != int(snap["scores"].argmax()):
                    outer.diverged += 1
            codes = snap["codes"]
            keys = self._parameters["content_embed"][codes] @ self._parameters["query_content"]
            weights = torch.softmax(scores.double(), dim=0).to(torch.float32)
            distribution = torch.zeros(
                self.config.alphabet_size, dtype=torch.float32, device=self.device
            ).index_add(0, codes, weights)
            pooled = weights @ keys
            gate = (
                f1_context @ self._parameters["gate_state"]
                + pooled @ self._parameters["gate_content"]
                + self._parameters["gate_bias"]
            )
            return {
                **snap,
                "scores": scores,
                "copy_distribution": distribution,
                "pooled": pooled,
                "gate_value": float(gate),
            }

        def patched_weights(self, event, f1_context, prev_byte=None):
            scale = math.sqrt(float(self.evidence_width))
            scores = outer._weights(self, event, f1_context, prev_byte, scale)
            return torch.softmax(scores.double(), dim=0).to(torch.float32)

        self._originals = {
            "addressing": original_addressing,
            "weights": original_weights,
            "cls": CopyCircuit,
        }
        CopyCircuit.addressing = patched_addressing
        CopyCircuit._position_weights = patched_weights

    def uninstall(self) -> None:
        cls = self._originals["cls"]
        cls.addressing = self._originals["addressing"]
        cls._position_weights = self._originals["weights"]


def trace_all(substrate: Any, circuit: Any, runtime: Any, ids: list[str], items: dict, tokens_of):
    """跑一遍 §13 那条链（scored＋只留目标告知），返回逐步指对/发出统计。"""
    from probe_taiji_r2_a26_emission_trace import trace_item

    steps = aimed = emitted = 0
    step0 = step0_aimed = 0
    fully = 0
    for item_id in ids:
        item = items[item_id]
        turns = [str(turn) for turn in item["turns"]]
        tokens = list(tokens_of(item))
        told = next((turn for turn in turns[:-1] if any(t in turn for t in tokens)), "")
        if not told:
            continue
        sink: list[dict[str, Any]] = []
        record = trace_item(
            substrate,
            circuit,
            turns=turns,
            tokens=tokens,
            told=told,
            history_mode="scored",
            store_mode="target",
            runtime=runtime,
            feature_sink=sink,
        )
        for row in sink:
            steps += 1
            aimed += int(row["copy_aimed"])
            emitted += int(row["emitted"])
            if int(row["step"]) == 0:
                #: §22 量到 56/81 题在**答案第一个字节**就断，所以第 0 步要单列，
                #: 不能和"整串平均"混成一个数。
                step0 += 1
                step0_aimed += int(row["copy_aimed"])
        fully += int(record.get("class") == "emitted")
    if steps == 0:
        raise AssertionError("一个答案步都没取到——链路或题号读错了")
    return {
        "steps": steps,
        "aim_correct": aimed,
        "aim_rate": round(aimed / steps, 4),
        "emitted_rate": round(emitted / steps, 4),
        "step0_steps": step0,
        "step0_aim_rate": round(step0_aimed / step0, 4) if step0 else None,
        "items_fully_emitted": fully,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=SEED_A_CIRCUIT)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 题（冒烟用）")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from price_taiji_r2_a27_address_refit import refit_m
    from probe_taiji_r2_a26_emission_trace import extension_items, failing_item_ids_extension
    from score_taiji_r2_copy_strict_cap import copyable_tokens

    from api.seed_runtime import SeedRuntime

    if not FEATURES.exists():
        print(json.dumps({"error": f"缺 §18 的逐步特征件 {FEATURES.name}"}))
        return 1
    steps_offline = json.loads(FEATURES.read_text(encoding="utf-8"))
    ids = failing_item_ids_extension()
    if args.limit > 0:
        ids = ids[: args.limit]
    items = extension_items()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    payload = torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    substrate.copy_circuit.load_payload(payload)
    circuit = substrate.copy_circuit
    tokens_of = copyable_tokens

    trained_m = (
        circuit._parameters["query_state"].detach().cpu().to(torch.float64)
        @ circuit._parameters["query_content"].detach().cpu().to(torch.float64).T
    )
    trained_bias = float(circuit._parameters["copy_induce_bias"].detach().cpu().flatten()[0])

    #: **锚点**：把覆写设回"训练后的 M＋训练后的偏置"，它必须和原实现给出**同一个位置分布**
    #: （逐位接近且 argmax 相同）。这一条不过，说明我这层覆写本身写错了，
    #: 后面所有 on-policy 读数都会是在另一条实现上取的——直接退出，不出件。
    from taiji.copy_circuit import CopyCircuit, ToldEvent

    original_weights = CopyCircuit._position_weights
    anchor = AddressingOverride(trained_m, trained_bias)
    sample = [step for step in steps_offline if step["step"] == 0][:40]
    mismatch = 0
    for step in sample:
        event = ToldEvent(
            event_id=0, content=bytes(step["codes"]), cue=torch.zeros(circuit.store.cue_dim)
        )
        f1 = torch.tensor(step["f1"], dtype=torch.float32)
        before = original_weights(circuit, event, f1, step["prev_byte"])
        anchor.install()
        try:
            after = circuit._position_weights(event, f1, step["prev_byte"])
        finally:
            anchor.uninstall()
        if (
            int(before.argmax()) != int(after.argmax())
            or float((before - after).abs().max()) > 1e-4
        ):
            mismatch += 1
    if mismatch or not sample:
        print(
            json.dumps(
                {
                    "error": "锚点不过：覆写在训练取值上与原实现不一致（或没样本可比）",
                    "mismatch": mismatch,
                    "checked": len(sample),
                }
            )
        )
        return 2

    baseline = trace_all(substrate, circuit, runtime, ids, items, tokens_of)

    refit = refit_m(circuit, steps_offline, args.epochs, args.lr, args.margin)
    override = AddressingOverride(refit["m"], refit["lam"] / refit["scale"])
    override.install()
    try:
        onpolicy = trace_all(substrate, circuit, runtime, ids, items, tokens_of)
    finally:
        override.uninstall()
    onpolicy["patched_addressing_calls"] = override.calls
    onpolicy["patched_diverged_calls"] = override.diverged

    report = {
        "format": "taiji-r2-a27-address-onpolicy-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §19",
        "chain": "scored + store=target（每步都在正确事件里）",
        "circuit": args.circuit,
        "items": len(ids),
        "off_line_aim_rate": refit["refit_offline_aim"],
        "off_line_aim_note": "在旧轨迹的状态点上量的指对率（§18 那个 94%）",
        "baseline_on_own_trajectory": baseline,
        "refit_on_own_trajectory": onpolicy,
        "reading_rule": (
            "refit 在自己轨迹上的 aim_rate 塌回接近 baseline ⇒ (a) 分布位移吃掉全部离线收益，"
            "改寻址必须自带 on-policy 评测；仍显著高于 baseline ⇒ (b) 掉分在门/pooled 那道闸上，"
            "下一刀测那道闸。calls 或 diverged 为 0 ⇒ 覆写没生效，本件作废"
        ),
        "patch_effective": bool(override.calls > 0 and override.diverged > 0),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = Path(args.out_report or "reports/taiji_r2_a27_address_onpolicy_20260926.json")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["patch_effective"] or not report["base_sha256_unchanged"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
