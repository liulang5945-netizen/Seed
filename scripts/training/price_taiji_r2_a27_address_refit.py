"""§18 的下一刀（零训练）：把"能把寻址做到 95% 的那个解"**装回发射链**跑表层。

为什么要这一步。§18 证到的只是"这套参数化有能力远好于现状"（同一层线性变换换个取值，
指对率 16.1% → 94.96%），但那个数是在**现状策略自己的轨迹**上量的——换了解之后模型发的字节会变、
后面每一步看到的状态也跟着变。所以"有能力"还不等于"有增益"，中间差的正是这一测：
把解装进去，让模型自己重新答一遍，数严格真命中。

**装哪里**。只换 `_position_weights`（同一事件内"该看哪个位置"的分布），
其余（皮质 cue 选事件、发射门、读出）全部原样。口径要写清：换了位置分布，
门读到的 `pooled` 也会跟着变——这不是混淆，机制本来就是这样连着的；
本件问的是"更好的寻址接进这条链，表层会怎样"，不是"只改一个数别的都不动"。

**解怎么来**：每次运行**现算**（从 §18 存的逐步特征里重新求一遍），不读任何手传的权重文件——
带随机起点又不落盘的解，隔天就复现不出来；现算能保证"装进去的就是这一份"，
并把它的离线指对率一起写进报告。固定 torch 种子保证可重跑。

**判读线（跑之前立）**：
* 表层严格命中较基线（v1 23/104、v2 16/104）**同向上升且 ≥3 题** ⇒ 能力→增益成立，
  优先级重排：**A2.3 那套寻址训练制度**（优势信号／学习率／窗口）是第一优先，
  A2.5（可学选择器）与"改内容表征"都往后排；
* 不升或升得不足 3 题 ⇒ "指对"与"发得出"之间另有一道闸（§13 的 `emission_loses` 那一族），
  下一刀改测那道闸，**不许**回过头把 §18 那个 95% 说成"其实没用"。

纪律：零训练（不改任何已训参数的持久值，只在进程内换寻址取法）；`checkpoints/` 只读并跑前后
复核 sha256；只写 `reports/` 一份新件；基线沿用已入库读数（同链同底同题集）。
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
V1_REPORT = PROJECT_ROOT / "reports/taiji_r2_copy_surface_extension_20260925.json"
V2_REPORT = PROJECT_ROOT / "reports/taiji_r2_a25_overlap_selector_price_v2_20260926.json"
SEED_A_CIRCUIT = "output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refit_m(circuit: Any, steps: list[dict], epochs: int, lr: float, margin: float) -> dict:
    """从 §18 的逐步特征里**重新求**一个寻址映射 M（与产品同形式：分数＝f1·M·embed[c]＋λS）。"""
    from probe_taiji_r2_a27_address_separability import batched_steps, scores_of

    embed = circuit._parameters["content_embed"].detach().cpu().to(torch.float64)
    batch = batched_steps(steps, embed)
    trained = (
        circuit._parameters["query_state"].detach().cpu().to(torch.float64)
        @ circuit._parameters["query_content"].detach().cpu().to(torch.float64).T
    )
    scale = math.sqrt(float(circuit.evidence_width))
    lam = float(circuit._parameters["copy_induce_bias"].detach().cpu().flatten()[0]) * scale
    mask, target = batch["mask"], batch["target"]
    flat = trained.reshape(-1).clone()
    index = torch.arange(target.shape[0])

    def evaluate(current_m, current_lam) -> tuple[float, float]:
        with torch.no_grad():
            scored = scores_of(current_m, batch, current_lam).masked_fill(~mask, -1e18)
            hit = target[index, scored.argmax(dim=1)].to(torch.float64).mean()
            gap = (
                scored.masked_fill(~target, -1e18).max(dim=1).values
                - scored.masked_fill(target, -1e18).max(dim=1).values
            ).min()
        return float(hit), float(gap)

    trained_aim, _ = evaluate(flat, torch.tensor(lam, dtype=torch.float64))

    def run_seed(start: torch.Tensor):
        m = start.clone().requires_grad_(True)
        lam_t = torch.tensor(lam, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([m, lam_t], lr=lr)
        local_best_m, local_best = m.detach().clone(), 0.0
        for epoch in range(epochs):
            opt.zero_grad()
            scored = scores_of(m, batch, lam_t).masked_fill(~mask, -1e18)
            good = scored.masked_fill(~target, -1e18).max(dim=1).values
            bad = scored.masked_fill(target, -1e18).max(dim=1).values
            goal = margin * min(1.0, (epoch + 1) / max(epochs / 4.0, 1.0))
            torch.relu(goal - (good - bad)).pow(2).mean().backward()
            opt.step()
            if (epoch + 1) % 10:
                continue
            aim, _ = evaluate(m.detach(), lam_t.detach())
            if aim > local_best:
                local_best, local_best_m = aim, m.detach().clone()
        return local_best, local_best_m

    #: **起点必须与 §18 一致**（trained／zero／小随机三起点，各跑满 epochs）。
    #: 冒烟时只用了随机起点＋1200 轮，离线只到 50%，而 §18 的 95% 是多起点长跑的结果——
    #: 拿弱解去测表层会得到一个假阴性，看起来像"能力不等于增益"，其实只是没找到好解。
    torch.manual_seed(20260926)
    best_m, best = flat.clone(), trained_aim
    for start in (flat.clone(), torch.zeros_like(flat), torch.randn_like(flat) * 1e-3):
        aim, candidate = run_seed(start)
        if aim > best:
            best, best_m = aim, candidate
    return {
        "m": best_m.reshape(trained.shape),
        "lam": lam,
        "scale": scale,
        "trained_offline_aim": round(trained_aim, 4),
        "refit_offline_aim": round(best, 4),
        "steps": len(steps),
    }


PROBE: dict[str, int] = {"calls": 0, "diverged": 0}


def install(m: torch.Tensor, lam: torch.Tensor, scale: float) -> Any:
    """把寻址取法换成 `f1·M·embed[c] + λS`（λ 已换回本件单位＝偏置×scale）。

    **必须打类级补丁**（`CopyCircuit._position_weights`），不能打在某个实例上：
    第一版打在已加载的 circuit 对象上，而 `run_arm` 自己会 `SeedRuntime.load` 出一个新运行时
    ⇒ 补丁挂在没人消费的那个旧对象上，跑出来 104/104 答复与基线**逐字节相同**、delta 恰好 0。
    那是个假阴性，不是"能力不等于增益"的结论。
    """
    from taiji.copy_circuit import CopyCircuit

    original = CopyCircuit._position_weights
    embed_key = "content_embed"

    def patched(self, event, f1_context, prev_byte=None):
        codes = torch.tensor(list(event.content), device=self.device, dtype=torch.long)
        keys = self._parameters[embed_key][codes].detach().cpu().to(torch.float64)
        scores = (f1_context.detach().cpu().to(torch.float64) @ m) @ keys.T
        if prev_byte is not None:
            bonus = torch.zeros_like(scores)
            for position in range(1, len(codes)):
                if int(codes[position - 1]) == int(prev_byte):
                    bonus[position] = 1.0
            scores = scores + lam * bonus
        weights = torch.softmax(scores / scale, dim=0).to(dtype=torch.float32, device=self.device)
        PROBE["calls"] += 1
        with torch.no_grad():
            if int(weights.argmax()) != int(original(self, event, f1_context, prev_byte).argmax()):
                #: 与训练取法**选到不同位置**的次数——为 0 就说明这次重解等于没改。
                PROBE["diverged"] += 1
        return weights

    CopyCircuit._position_weights = patched
    return original


def uninstall(original: Any) -> None:
    from taiji.copy_circuit import CopyCircuit

    CopyCircuit._position_weights = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=SEED_A_CIRCUIT)
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument(
        "--sets",
        default="v1,v2",
        help="逗号分隔：v1＝软干扰集（基线 23/104），v2＝难干扰集（基线 16/104）",
    )
    parser.add_argument("--limit", type=int, default=0, help="每个题集只取前 N 题（冒烟用）")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from score_taiji_r2_copy_surface_extension import load_items, run_arm

    from api.seed_runtime import SeedRuntime

    if not FEATURES.exists():
        print(json.dumps({"error": f"缺 §18 的逐步特征件 {FEATURES.name}，先跑那支探针"}))
        return 1
    steps = json.loads(FEATURES.read_text(encoding="utf-8"))

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    payload = torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    substrate.copy_circuit.load_payload(payload)
    circuit = substrate.copy_circuit

    refit = refit_m(circuit, steps, args.epochs, args.lr, args.margin)
    m, lam, scale = refit["m"], torch.tensor(refit["lam"]), refit["scale"]

    baselines = {
        "v1": (
            V1_REPORT,
            lambda data: int(
                next(
                    arm["strict_hits"]
                    for arm in data["treated_arms"]
                    if str(arm["circuit"]) == args.circuit
                )
            ),
            "plans/manifests/r2_copy_surface_extension_v1.json",
        ),
        "v2": (
            V2_REPORT,
            lambda data: int(
                next(
                    arm["baseline_strict_hits"]
                    for arm in data["arms"]
                    if arm["circuit"] == args.circuit
                )
            ),
            "plans/manifests/r2_copy_surface_extension_v2.json",
        ),
    }
    arms: list[dict[str, Any]] = []
    for name in [part.strip() for part in args.sets.split(",") if part.strip()]:
        report_path, pick, manifest = baselines[name]
        data = json.loads(report_path.read_text(encoding="utf-8"))
        baseline_hits = pick(data)
        PROBE["calls"] = 0
        PROBE["diverged"] = 0
        original = install(m, lam, scale)
        try:
            items = load_items(PROJECT_ROOT / manifest)
            if args.limit > 0:
                items = items[: args.limit]
            arm = run_arm(items, checkpoint, args.circuit)
        finally:
            uninstall(original)
        arms.append(
            {
                "set": name,
                "manifest": manifest,
                "items": arm["items"],
                "patched_calls": PROBE["calls"],
                "patched_diverged_calls": PROBE["diverged"],
                "strict_hits": arm["strict_hits"],
                "items_scored": len(arm["rows"]),
                "baseline_strict_hits": baseline_hits,
                #: 冒烟（--limit）时基线是整集的数，差值无意义——照实标记，不假装可比。
                "comparable": bool(args.limit == 0),
                "delta": arm["strict_hits"] - baseline_hits,
                "baseline_report": report_path.name,
                "rows": arm["rows"],
            }
        )

    decided = [arm for arm in arms if int(arm["delta"]) >= 3]
    report = {
        "format": "taiji-r2-a27-address-refit-surface-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §18",
        "what_changed": (
            "只换 _position_weights（同一事件内看哪个位置）；选事件、发射门、读出全部原样。"
            "位置分布变了，门读到的 pooled 跟着变——机制本就连着，不是混淆"
        ),
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "offline_aim": {
            "trained_m": refit["trained_offline_aim"],
            "refit_m": refit["refit_offline_aim"],
            "steps": refit["steps"],
        },
        "arms": arms,
        "reading_rule": (
            "任一集 delta>=3 ⇒ 能力→增益成立，A2.3 寻址训练制度提到第一优先；"
            "都不足 3 ⇒ 指对与发出之间另有一道闸，下一刀测那道闸"
        ),
        #: 补丁没被走到 / 一次都没改变选择 ⇒ 这一跑是**未生效**，不许读成"没有增益"。
        "patch_effective": bool(all(int(arm["patched_calls"]) > 0 for arm in arms))
        and bool(all(int(arm["patched_diverged_calls"]) > 0 for arm in arms)),
        "judged_gain": bool(decided and all(bool(arm["comparable"]) for arm in arms)),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = Path(args.out_report or "reports/taiji_r2_a27_address_refit_surface_20260926.json")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "offline_aim": report["offline_aim"],
                "arms": [
                    {
                        "set": arm["set"],
                        "hits": arm["strict_hits"],
                        "baseline": arm["baseline_strict_hits"],
                        "delta": arm["delta"],
                        "calls": arm["patched_calls"],
                        "diverged": arm["patched_diverged_calls"],
                    }
                    for arm in arms
                ],
                "judged_gain": report["judged_gain"],
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
    if not report["base_sha256_unchanged"] or not report["patch_effective"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
