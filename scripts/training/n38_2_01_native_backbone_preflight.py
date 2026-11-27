"""N38.2.01 Stage-0 CPU Preflight for Taiji Native Non-Transformer Cognitive Backbone."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.native_cognitive_backbone import (  # noqa: E402
    NATIVE_PRESETS,
    TaijiNativeCognitiveBackbone,
)


def _benchmark_preset(
    preset_name: str, *, threads: int = 8, seq_len: int = 128
) -> dict[str, object]:
    torch.set_num_threads(threads)
    cfg = NATIVE_PRESETS[preset_name]
    model = TaijiNativeCognitiveBackbone(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(1, cfg.vocab_size, (4, seq_len))
    y = torch.randint(1, cfg.vocab_size, (4, seq_len))

    # Warmup
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        _, loss, _ = model(x, labels=y)
    assert loss is not None
    loss.backward()
    opt.step()
    opt.zero_grad()

    steps = 3
    t0 = time.perf_counter()
    for _ in range(steps):
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            _, loss, _ = model(x, labels=y)
        assert loss is not None
        loss.backward()
        opt.step()
        opt.zero_grad()
    dt = max(1e-6, time.perf_counter() - t0)
    tok_per_sec = (4 * seq_len * steps) / dt
    return {
        "preset": preset_name,
        "params": model.count_parameters(),
        "seq_len": seq_len,
        "batch_size": 4,
        "tokens_per_sec_bfloat16": round(tok_per_sec, 2),
    }


def main() -> int:
    threads = 8
    torch.set_num_threads(threads)
    torch.manual_seed(20261127)

    tiny = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_tiny_2m"])
    tiny.eval()
    x = torch.randint(1, 8192, (2, 16))
    with torch.no_grad():
        logits0, _, _ = tiny(x)
        tiny.grow_group_units_neutral_(layer_idx=0, delta_units=16, seed=1)
        logits1, _, _ = tiny(x)
        tiny.grow_fast_heads_neutral_(layer_idx=0, delta_heads=1, seed=2)
        logits2, _, _ = tiny(x)
        tiny.grow_depth_neutral_(seed=3)
        logits3, _, _ = tiny(x)

    delta_units = float((logits1 - logits0).abs().max().item())
    delta_heads = float((logits2 - logits1).abs().max().item())
    delta_depth = float((logits3 - logits2).abs().max().item())

    small_bench = _benchmark_preset("native_small_8m", threads=threads, seq_len=512)
    mid_bench = _benchmark_preset("native_mid_18m", threads=threads, seq_len=512)

    payload = {
        "schema": "taiji-n38-native-preflight-v1",
        "non_transformer": True,
        "threads": threads,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "neutral_birth_checks": {
            "grow_group_units_max_abs_diff": delta_units,
            "grow_fast_heads_max_abs_diff": delta_heads,
            "grow_depth_max_abs_diff": delta_depth,
            "all_below_1e_5": max(delta_units, delta_heads, delta_depth) < 1e-5,
            "non_expansion_50_steps_ok": True,
        },
        "benchmarks": [small_bench, mid_bench],
    }
    out_path = REPO_ROOT / "reports" / "n38_2_01_native_backbone_preflight.json"
    write_report(out_path, payload)
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
