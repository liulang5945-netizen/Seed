"""P3b 吞吐标定：测真实训练单步耗时并外推机时（**只读**：不写任何检查点）。

依据 [P3b 预注册](../../plans/reference/M5_P3B_ALIGNED_LANGUAGE_TRAINING_PREREGISTRATION_20260915.md) §2：
"预算先标定（小规模试跑测吞吐后冻结总机时）；标定前不承诺任何机时数字"。

做法：
- 加载起点检查点（`checkpoints/seed_beta.pt`，旧格式 ⇒ 需进程内放宽身份器官守卫）；
- 从**已归档的对话子集**取真实字节流，喂 `steps` 步 `observe(symbol, learn=True)`；
- 计时得出 **steps/s**，并外推"过一遍子集"与"相对 16M ticks 的倍数"。

**副作用说明**：为测**真实**训练吞吐，本脚本会在**内存中**更新参数若干步，
但**绝不写检查点、不改任何文件**；进程退出即消失。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"
DEFAULT_SUBSET = PROJECT_ROOT / "data" / "p3b_dialogue_subset.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_dialogue_subset_manifest.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_p3b_throughput_calibration_20260915.json"
BASELINE_TICKS = 16_000_000


def _byte_stream(subset: Path, *, limit: int):
    """从对话子集流式产出 UTF-8 字节（与训练脚本同口径：每行一个 session）。"""

    emitted = 0
    with subset.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                document = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            text = document.get("text")
            if not isinstance(text, str):
                continue
            for value in text.encode("utf-8"):
                yield value
                emitted += 1
                if emitted >= limit:
                    return


def calibrate(
    checkpoint: Path = DEFAULT_CHECKPOINT,
    subset: Path = DEFAULT_SUBSET,
    *,
    steps: int = 2000,
    relax_legacy_guard: bool = True,
) -> dict[str, Any]:
    guard_info: dict[str, object] = {}
    if relax_legacy_guard:
        from scripts.training.probe_taiji_cap0_legacy_load import _install_legacy_guard

        guard_info = _install_legacy_guard()

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    model = runtime.model.substrate

    started = time.perf_counter()
    consumed = 0
    for symbol in _byte_stream(subset, limit=steps):
        model.observe(symbol, learn=True)
        consumed += 1
    elapsed = time.perf_counter() - started

    steps_per_second = consumed / elapsed if elapsed > 0 else None
    manifest = {}
    if DEFAULT_MANIFEST.is_file():
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    subset_bytes = int(manifest.get("kept_bytes") or 0)

    extrapolation: dict[str, Any] = {
        "baseline_ticks": BASELINE_TICKS,
        "one_pass_symbols": subset_bytes,
        "note": "机时为外推值，非实测；标定前不承诺任何机时数字。",
    }
    if steps_per_second:
        one_pass_seconds = subset_bytes / steps_per_second
        extrapolation.update(
            {
                "one_pass_hours": round(one_pass_seconds / 3600.0, 3),
                "seconds_per_1m_symbols": round(1_000_000 / steps_per_second, 3),
                "hours_for_16m_ticks": round(BASELINE_TICKS / steps_per_second / 3600.0, 3),
                "symbols_in_1_hour": int(steps_per_second * 3600),
            }
        )

    return {
        "format": "taiji-p3b-throughput-calibration-v1",
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "subset": str(subset.relative_to(PROJECT_ROOT)),
        "subset_sha256": manifest.get("output_sha256"),
        "guard_relaxed": bool(relax_legacy_guard),
        "legacy_guard": guard_info,
        "measured_steps": consumed,
        "measured_seconds": round(elapsed, 4),
        "steps_per_second": round(steps_per_second, 3) if steps_per_second else None,
        "checkpoint_written": False,
        "extrapolation": extrapolation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P3b 吞吐标定（只读，不写检查点）")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    result = calibrate(args.checkpoint, args.subset, steps=args.steps)
    report = args.report if args.report.is_absolute() else PROJECT_ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
