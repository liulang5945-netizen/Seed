"""诊断探针（**非计分**）：CAP D 题（先告知→后提问）在三种生成协议下能否命中？

背景（2026-09-24 深夜）：T6 显示四份模型 CAP D+E **全部 0/36**；M1 同款生成的**回声率=0.0000**；
而 D 维恰是「先告知、后提问」＋`expected_contains`（如「我叫阿岩。」→「我的名字是什么？」须含「阿岩」）。
且 `generate`（`diag_taiji_r2_surface_decode.py:158`）**硬编码** `use_memory=False, use_identity=False, learn=False`
⇒ 模型既不照抄、又被禁止用记忆/身份 ⇒ **0/36 可能是协议的必然，不是能力的测量**。

## 三种协议变体（同一冻结的 `generate` 循环，用**实例级包装**注入开关 ⇒ 生成逻辑零分叉）

| 变体 | prompt 字节 | 生成期 |
|---|---|---|
| `off`（现行冻结协议） | `learn=False, use_memory=False, use_identity=False` | 同左 |
| `tap_on` | 同上 | `use_memory=True, use_identity=True`（记忆是空的 ⇒ 预期无变化，作为对照） |
| `write_then_recall` | **`learn=True`**（把"告知"写进记忆/身份） | `use_memory=True, use_identity=True`（且仍在学——诚实标注） |

每个变体都用 `Seed.from_checkpoint` **重新加载同一份 checkpoint** ⇒ 变体之间零串扰
（`write_then_recall` 的扰动不泄漏给别的变体）。

**判读（先写死）**：`write_then_recall` 的命中数显著高于 `off` ⇒ **0/36 是协议造成的**
（告知从未被写入记忆/身份），下一步是"把告知写进记忆"的正式协议与预注册；
三者全 0 ⇒ 架构在**关着记忆/身份且无复制机制**的前提下确实无法回答 ⇒ 回到稳定性/写入侧。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "cap0_eval_set_v2.json"

VARIANTS = ("off", "tap_on", "write_then_recall")


def _force_observe(substrate: Any, *, learn: bool, use_memory: bool, use_identity: bool) -> Any:
    """实例级遮蔽 `observe`，强制三个开关；返回原方法以便恢复。"""

    original = substrate.observe

    def observed(symbol: int, **kwargs: object) -> object:
        kwargs["learn"] = learn
        kwargs["use_memory"] = use_memory
        kwargs["use_identity"] = use_identity
        return original(symbol, **kwargs)

    substrate.observe = observed
    return original  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="可多次；缺省=基座+A 臂")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import _serialize_prompt, generate

    from api.seed_runtime import SeedRuntime

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # 只取**固定答案**题（带 expected_contains）；该维另有 4 题是人工复核题（无此字段）⇒ 不进本探针
    all_items = [it for it in manifest["dimensions"]["D"]["items"] if "expected_contains" in it]
    items = all_items[: args.limit]

    checks: list[tuple[str, Path]] = [("base_16M", PROJECT_ROOT / "checkpoints" / "seed_beta.pt")]
    arm_a = PROJECT_ROOT / "output/taiji_r2_readout_retrain/A/checkpoint.pt"
    if arm_a.is_file():
        checks.append(("A_16M", arm_a))
    for relative in args.checkpoint:
        path = Path(relative)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        checks.append((path.stem, path))

    rows: list[dict[str, object]] = []
    for name, path in checks:
        for variant in VARIANTS:
            runtime = SeedRuntime.load(path)
            substrate = runtime.model.substrate
            if variant == "off":
                original = _force_observe(
                    substrate, learn=False, use_memory=False, use_identity=False
                )
            elif variant == "tap_on":
                original = _force_observe(
                    substrate, learn=False, use_memory=True, use_identity=True
                )
            else:  # write_then_recall
                original = _force_observe(
                    substrate, learn=True, use_memory=True, use_identity=True
                )
            hits = 0
            samples: list[str] = []
            for item in items:
                prompt = "".join(item["turns"])
                text = generate(substrate, _serialize_prompt(runtime, prompt), "greedy").decode(
                    "utf-8", errors="replace"
                )
                hit = any(expected in text for expected in item["expected_contains"])
                hits += int(hit)
                if len(samples) < 3:
                    samples.append(text[:50])
            substrate.observe = original  # 恢复
            rows.append(
                {
                    "model": name,
                    "variant": variant,
                    "expected_contains_hits": hits,
                    "n_items": len(items),
                    "samples": samples,
                }
            )
            print(
                json.dumps(
                    {"model": name, "variant": variant, "hits": f"{hits}/{len(items)}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "format": "taiji-r2-cap-d-protocol-probe-v2",
        "note": "**非计分**：协议变体未冻结、序列化是近似；只回答'协议开关是不是那个原因'",
        "manifest": str(MANIFEST),
        "rows": rows,
    }
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_cap_d_protocol_probe_20260924.json"
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
