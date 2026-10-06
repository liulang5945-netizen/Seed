"""A0 重判诊断：T8 的 (c) 判定在**零向量注入**下做出，本件用**真实非零反馈**重判。零 `taiji/` 改动。

前件（本仓实测，`reports/taiji_r2_recall_split_20260925.json`）：T8 Exp-1 反馈范数全 0.0、
Exp-2 forced 与 zero 生成文本逐字相同——因为 `write` 唯一入口是 act→settle_action→
PendingExperience（`model.py:1815-1829/2245`），而 T8 的写入段用 `readout="predictive"`
走 `learn=True`，**从不触发 act/settle ⇒ write_count 恒 0 ⇒ 反馈恒零**。
「(c) 召回后解码有罪」的结论因此无效（T8 §5 边界条已自认打折，实际是全量失效）。

**本件运行前新暴露的架构事实**（首跑实测）：`model.py:1752` 禁止同一 dynamics episode 内切换
readout——唯一写路径要 `action` 读出、语言生成要 `predictive` 读出，**写回路与语言回路在
episode 级互斥**。故本件三臂均为「action 写入段 → 真 `reset_dynamics`（记忆器官持久，清的
只是动力学态）→ predictive 生成段」的跨 episode 协议——这正是「先告知→后提问」的最近合法近似。

## 三个实验（每模型：base_16M、A_16M）

* **Phase W（写入段，action 读出）**：`observe(byte, readout="action", learn=True,
  use_memory=True, use_identity=True)` → `act([byte])` → `settle_action(reward=1.0)` →
  下一 observe 消费 pending 写情景事件。断言 `write_count` 增长且捕获到**非零**
  `state.memory.cortical_feedback`（记其范数）。
* **Exp-N（native 开闸，(b)+(c) 合判）**：Phase W 后直接 `substrate.generate(ask, 24,
  reset=False, use_memory=True)`（生成器本身有 `use_memory` 透传，硬编码在协议上层）。
  同轮用只读包装记录生成期每 tick 真实进 `fabric.step` 的 `episodic_feedback` 范数。
* **Exp-F（强制非零注入，隔离 (b)）**：Phase W 后把生成期 `episodic_feedback` 强制替换为
  告知段捕获的**非零**反馈（绕过最近邻寻址）；对照臂强制零。复用 T8 的实例级包装与
  冻结 `generate`（`diag_taiji_r2_surface_decode.generate`，greedy）。

## 判读（先冻结）

* Phase W 失败（write_count 不增或反馈范数全 0）⇒ `write-failed`，本诊断无效；
* Exp-N 命中 ⇒ **协议闸门是元凶**（写入+开闸即端到端答对；架构增量降为增强项）；
* Exp-N 不中 且 Exp-F(forced) 中 且 Exp-F(zero) 不中 ⇒ **(b) 寻址是元凶**（内容通道在，真实召回匹配不上）；
* Exp-N 不中 且 Exp-F(forced) 不中 ⇒ **(c) 真有罪**（真实内容反馈也解不出 ⇒ F1 内容直读/copy 通道的架构增量必要）；
* Exp-F(forced) 与 Exp-F(zero) 文本逐字相同 ⇒ 强制无效 ⇒ `invalid`。

判决件不覆写（同 T8 纪律）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

ITEM = {"turns": ["我叫阿岩。", "我的名字是什么？"], "expected_contains": ["阿岩"]}
GEN_BYTES = 24


def _feedback(substrate: object) -> torch.Tensor:
    return substrate._state.memory.cortical_feedback.detach().clone()  # type: ignore[attr-defined]


def phase_write(
    runtime: object, substrate: object, tell_symbols: list[int]
) -> tuple[int, torch.Tensor, list[float]]:
    """告知段走 action 读出 + act/settle，制造真实情景写入。返回 (write_count, 末段非零反馈, 范数轨迹)。"""
    substrate.reset_dynamics(episode_id="a0-write")  # type: ignore[attr-defined]
    memory = substrate.memory  # type: ignore[attr-defined]
    count_before = int(memory.write_count)
    norms: list[float] = []
    last_fb = _feedback(substrate)
    step_args = {
        "learn": True,
        "readout": "action",
        "use_memory": True,
        "use_identity": True,
    }
    substrate.observe(int(substrate.config.boundary_symbol), **step_args)  # type: ignore[attr-defined]
    for symbol in tell_symbols:
        substrate.observe(int(symbol), **step_args)  # type: ignore[attr-defined]
        substrate.act([int(symbol)], sample=False)  # type: ignore[attr-defined]
        substrate.settle_action(reward=1.0)  # type: ignore[attr-defined]
        fb = _feedback(substrate)
        norms.append(round(float(fb.norm()), 4))
        if float(fb.norm()) > 0.0:
            last_fb = fb
    substrate.observe(int(substrate.config.boundary_symbol), **step_args)  # 冲掉最后一条 pending
    return int(memory.write_count) - count_before, last_fb, norms


def run_frozen_generate(
    substrate: object, ask_symbols: list[int], forced: torch.Tensor | None
) -> str:
    """T8 同款：fabric.step 强制替换 + reset 变 no-op，走冻结生成器（greedy）。"""
    from diag_taiji_r2_surface_decode import generate

    fabric = substrate.fabric  # type: ignore[attr-defined]
    original_step = fabric.step
    original_reset = substrate.reset_dynamics  # type: ignore[attr-defined]

    def stepped(sensory: object, previous: object, **kwargs: object) -> object:
        if forced is not None:
            kwargs["episodic_feedback"] = forced
        return original_step(sensory, previous, **kwargs)

    fabric.step = stepped
    substrate.reset_dynamics = lambda episode_id=None: None  # type: ignore[method-assign]
    try:
        text = generate(substrate, bytes(ask_symbols), "greedy").decode("utf-8", errors="replace")
    finally:
        fabric.step = original_step
        substrate.reset_dynamics = original_reset  # type: ignore[method-assign]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import _serialize_prompt

    from api.seed_runtime import SeedRuntime

    checks = [
        ("base_16M", PROJECT_ROOT / "checkpoints" / "seed_beta.pt"),
        ("A_16M", PROJECT_ROOT / "output/taiji_r2_readout_retrain/A/checkpoint.pt"),
    ]
    tell, ask = ITEM["turns"][0], ITEM["turns"][1]
    expected = ITEM["expected_contains"]
    rows: list[dict[str, object]] = []

    for name, path in checks:
        probe = SeedRuntime.load(path)
        tell_symbols = list(_serialize_prompt(probe, tell))
        ask_symbols = list(_serialize_prompt(probe, ask))
        hit_of = lambda text: any(token in text for token in expected)  # noqa: E731

        # ---- Phase W：写入断言（三臂各自重放，互不污染）----
        runtime_w = SeedRuntime.load(path)
        substrate_w = runtime_w.model.substrate
        written, tell_fb, w_norms = phase_write(runtime_w, substrate_w, tell_symbols)
        write_ok = written > 0 and float(tell_fb.norm()) > 0.0

        # ---- Exp-N：native 开闸（真实寻址，跨 episode：action 写入段 → 真 reset → predictive 生成段）----
        native_text, native_norms = "", []
        if write_ok:
            runtime_n = SeedRuntime.load(path)
            substrate_n = runtime_n.model.substrate
            phase_write(runtime_n, substrate_n, tell_symbols)
            substrate_n.reset_dynamics(episode_id="a0-native-gen")  # 记忆器官持久，清的是动力学态
            fabric_n = substrate_n.fabric
            original_step_n = fabric_n.step

            def recording_step(
                sensory: object,
                previous: object,
                _norms: list[float] = native_norms,
                _original=original_step_n,
                **kwargs: object,
            ) -> object:
                fb = kwargs.get("episodic_feedback")
                if isinstance(fb, torch.Tensor):
                    _norms.append(round(float(fb.norm()), 4))
                return _original(sensory, previous, **kwargs)

            fabric_n.step = recording_step
            native_text = substrate_n.generate(
                bytes(ask_symbols), GEN_BYTES, reset=False, use_memory=True
            ).decode("utf-8", errors="replace")
            fabric_n.step = original_step_n

        # ---- Exp-F：强制非零 vs 强制零（各自全新加载 + 重放 Phase W，互不污染）----
        forced_text, zero_text = "", ""
        if write_ok:
            runtime_f = SeedRuntime.load(path)
            substrate_f = runtime_f.model.substrate
            phase_write(runtime_f, substrate_f, tell_symbols)
            substrate_f.reset_dynamics(episode_id="a0-forced-gen")
            forced_text = run_frozen_generate(substrate_f, ask_symbols, tell_fb)
            runtime_z = SeedRuntime.load(path)
            substrate_z = runtime_z.model.substrate
            phase_write(runtime_z, substrate_z, tell_symbols)
            substrate_z.reset_dynamics(episode_id="a0-zero-gen")
            zero_text = run_frozen_generate(substrate_z, ask_symbols, torch.zeros_like(tell_fb))

        # ---- 判读（冻结）----
        if not write_ok:
            verdict = "write-failed（诊断无效：情景写入未发生）"
        elif forced_text == zero_text:
            verdict = "invalid（forced 与 zero 逐字相同：强制无效）"
        elif hit_of(native_text):
            verdict = "协议闸门是元凶（native 开闸端到端答对；架构增量降为增强项）"
        elif hit_of(forced_text) and not hit_of(zero_text):
            verdict = "(b) 寻址是元凶（内容通道在，真实召回匹配不上）"
        else:
            verdict = "(c) 真有罪（非零内容反馈也解不出 ⇒ F1 内容直读/copy 通道必要）"

        row = {
            "model": name,
            "phase_w": {
                "events_written": written,
                "tell_feedback_norm": round(float(tell_fb.norm()), 4),
                "feedback_norms_head": w_norms[:8],
            },
            "exp_native": {
                "hit": write_ok and hit_of(native_text),
                "text": native_text[:80],
                "recall_feedback_norms_during_generation": native_norms[:24],
            },
            "exp_forced_nonzero": {
                "hit": write_ok and hit_of(forced_text),
                "text": forced_text[:80],
            },
            "exp_forced_zero": {"hit": write_ok and hit_of(zero_text), "text": zero_text[:80]},
            "verdict": verdict,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "model": name,
                    "verdict": verdict,
                    "written": written,
                    "tell_fb_norm": row["phase_w"]["tell_feedback_norm"],
                    "native_hit": row["exp_native"]["hit"],
                    "forced_hit": row["exp_forced_nonzero"]["hit"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "format": "taiji-r2-recall-split-v2-diagnostic-v1",
        "prereg": "本件 docstring（判读冻结）；前件 reports/taiji_r2_recall_split_20260925.json",
        "item": ITEM,
        "rows": rows,
        "note": "A0 重判：T8 的 (c) 判定建立在零向量注入上；本件先经 action 路径真写入，再重判 (b)/(c)/协议闸",
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_recall_split_v2_20260925.json")
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
