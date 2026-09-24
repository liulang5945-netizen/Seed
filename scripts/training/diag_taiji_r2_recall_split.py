"""T8 分裂诊断：(b) 召回匹配 vs (c) 召回后解码 —— 零 `taiji/` 改动（实例级读/包装）。

预注册：`M5_R2_T8_MECHANISM_DESIGN_20260925.md` §1-§2。

## 三个实验

* **Exp-1（反馈存在性）**：learn=True 走完 D01 全提示（告知+提问），逐 tick 记录
  `state.memory.cortical_feedback` 的范数——记忆到底有没有在反馈、告知段与提问段有无差别。
* **Exp-2 (b 检验：召回匹配)**：告知段学完后，**提问+生成期把 `episodic_feedback` 强制替换为
  告知段最后的反馈向量**（绕过最近邻匹配）。若生成命中「阿岩」⇒ (c) 无罪、**(b) 是元凶**；
  若仍不命中 ⇒ (c) 有罪。基线：同设置但反馈强制为零。
* **实现要点**：`generate` 内部会 `reset_dynamics` ⇒ 用实例级包装把它变成 no-op，
  让「告知（learn=True）→ 提问+生成（learn=False）」在**同一个 episode** 里完成——
  这正是冻结生成器做不到、而诊断必须做的部分。生成循环本身用**冻结的 `generate`**，零分叉。

## 判读（先冻结）

* Exp-2 forced 命中数 > zero 命中数 且 forced ≥ 1 ⇒ **(b) 元凶**；
* forced 与 zero 均 0 ⇒ **(c) 元凶**（读出无"召回→字节"通路）；
* 两者都 ≥2 且相等 ⇒ 匹配与解码都不缺，问题在别处（须重审）。
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


def _feedback(substrate: Any) -> torch.Tensor:
    return substrate._state.memory.cortical_feedback.detach().clone()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="可多次；缺省=base+A")
    parser.add_argument("--gen-bytes", type=int, default=24)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import _serialize_prompt, generate

    from api.seed_runtime import SeedRuntime

    checks: list[tuple[str, Path]] = [
        ("base_16M", PROJECT_ROOT / "checkpoints" / "seed_beta.pt"),
        ("A_16M", PROJECT_ROOT / "output/taiji_r2_readout_retrain/A/checkpoint.pt"),
    ]
    for relative in args.checkpoint:
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        checks.append((path.stem, path))

    tell = ITEM["turns"][0]
    ask = ITEM["turns"][1]
    expected = ITEM["expected_contains"]

    rows: list[dict[str, object]] = []
    for name, path in checks:
        runtime = SeedRuntime.load(path)
        substrate = runtime.model.substrate

        # ---- Exp-1：反馈存在性（learn=True 全提示）----
        substrate.reset_dynamics(episode_id=f"t8-exp1-{name}")
        fb_norms: list[float] = []
        fb_vectors: list[torch.Tensor] = []
        step = substrate.observe(
            substrate.config.boundary_symbol,
            learn=True,
            readout="predictive",
            use_memory=True,
            use_identity=True,
        )
        fb_norms.append(round(float(_feedback(substrate).norm()), 4))
        fb_vectors.append(_feedback(substrate))
        for symbol in _serialize_prompt(runtime, prompt="".join(ITEM["turns"])):
            step = substrate.observe(
                int(symbol),
                learn=True,
                readout="predictive",
                use_memory=True,
                use_identity=True,
            )
            fb = _feedback(substrate)
            fb_norms.append(round(float(fb.norm()), 4))
            fb_vectors.append(fb)
        tell_fb = fb_vectors[len(_serialize_prompt(runtime, tell)) + 1 - 1]
        tell_fb = fb_vectors[len(tell) + 1]  # 告知段（含边界）结束时的反馈
        zero_fb = torch.zeros_like(tell_fb)

        # ---- Exp-2：(c) 检验——强制告知反馈，看能否解码出「阿岩」----
        def run_with_feedback(forced: torch.Tensor | None, tag: str) -> dict[str, object]:
            runtime2 = SeedRuntime.load(path)
            substrate2 = runtime2.model.substrate
            # 告知段：learn=True 写入（与 Exp-1 同）
            substrate2.reset_dynamics(episode_id=f"t8-{tag}-learn")
            for symbol in [substrate2.config.boundary_symbol] + list(
                _serialize_prompt(runtime2, tell)
            ):
                substrate2.observe(
                    int(symbol),
                    learn=True,
                    readout="predictive",
                    use_memory=True,
                    use_identity=True,
                )
            captured = _feedback(substrate2)
            if forced is not None:
                captured = forced
            # 生成期：fabric.step 的 episodic_feedback 强制替换 + reset 变 no-op（同 episode 续跑）
            fabric = substrate2.fabric
            original_step = fabric.step

            def stepped(sensory: object, previous: object, **kwargs: object) -> object:
                kwargs["episodic_feedback"] = captured
                return original_step(sensory, previous, **kwargs)

            fabric.step = stepped
            original_reset = substrate2.reset_dynamics

            def noop_reset(episode_id: str | None = None) -> None:
                return None

            substrate2.reset_dynamics = noop_reset
            text = generate(
                substrate2,
                _serialize_prompt(runtime2, ask),
                "greedy",
            ).decode("utf-8", errors="replace")
            fabric.step = original_step
            substrate2.reset_dynamics = original_reset
            hit = any(expected in text for expected in expected)
            return {"hit": hit, "text": text[:80]}

        forced_res = run_with_feedback(tell_fb, "forced")
        zero_res = run_with_feedback(zero_fb, "zero")

        verdict = (
            "(b) 召回匹配是元凶（强制反馈能解码出答案）"
            if forced_res["hit"] and not zero_res["hit"]
            else ("(c) 召回后解码有罪（强制反馈也解不出）" if not forced_res["hit"] else "两者都不缺，须重审")
        )
        rows.append(
            {
                "model": name,
                "exp1_feedback_norms_head": fb_norms[:8],
                "exp1_feedback_norm_tail_mean": round(sum(fb_norms[-8:]) / 8, 4),
                "exp2_forced": forced_res,
                "exp2_zero": zero_res,
                "verdict": verdict,
            }
        )
        print(
            json.dumps(
                {
                    "model": name,
                    "forced_hit": forced_res["hit"],
                    "forced_text": forced_res["text"][:40],
                    "zero_hit": zero_res["hit"],
                    "verdict": verdict,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "format": "taiji-r2-recall-split-diagnostic-v1",
        "prereg": "plans/reference/M5_R2_T8_MECHANISM_DESIGN_20260925.md",
        "item": ITEM,
        "rows": rows,
        "note": "零 taiji/ 改动：状态外读 + fabric.step/reset 实例级包装；forced=强制告知段反馈",
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_recall_split_20260925.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
