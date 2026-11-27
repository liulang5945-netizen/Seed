"""N38.2.01 Stage-0 CPU Preflight for 态极 Native Byte-Group Backbone (`taiji-native-byte-group-v2`)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import cast

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from taiji.native_cognitive_backbone import (  # noqa: E402
    NATIVE_PRESETS,
    StatefulNeuralGroup,
    TaijiNativeCognitiveBackbone,
    bits_per_byte,
)


def _benchmark_preset(
    preset_name: str, *, threads: int = 8, seq_bytes: int = 512
) -> dict[str, object]:
    torch.set_num_threads(threads)
    cfg = NATIVE_PRESETS[preset_name]
    model = TaijiNativeCognitiveBackbone(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 256, (4, seq_bytes))
    y = torch.randint(0, 256, (4, seq_bytes))

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
    bytes_per_sec = (4 * seq_bytes * steps) / dt
    return {
        "preset": preset_name,
        "params": model.count_parameters(),
        "seq_bytes": seq_bytes,
        "batch_size": 4,
        "bytes_per_sec_bfloat16": round(bytes_per_sec, 2),
    }


def main() -> int:
    threads = 8
    torch.set_num_threads(threads)
    torch.manual_seed(20261127)

    tiny = TaijiNativeCognitiveBackbone(NATIVE_PRESETS["native_byte_tiny"])
    tiny.eval()
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))

    with torch.no_grad():
        logits0, init_loss_t, _ = tiny(x, labels=y)
        assert init_loss_t is not None
        init_loss = float(init_loss_t.item())

        old_n = cast(StatefulNeuralGroup, tiny.groups[0]).n_units
        tiny.grow_stateful_units_svd_neutral_(group_idx=0, budget=8, seed=1)
        logits1, _, _ = tiny(x)
        tiny.proliferate_units_neutral_(group_idx=0, parent_indices=[0, 1])
        logits2, _, _ = tiny(x)
        tiny.grow_group_neutral_(seed=2)
        logits3, _, _ = tiny(x)

    delta_svd = float((logits1 - logits0).abs().max().item())
    delta_prolif = float((logits2 - logits1).abs().max().item())
    delta_group = float((logits3 - logits2).abs().max().item())

    # Verify immediate non-zero analytical startup gradient on newly born unit_r
    tiny.train()
    tiny.zero_grad(set_to_none=True)
    _, loss_after, _ = tiny(x, labels=y)
    assert loss_after is not None
    loss_after.backward()
    grp0 = cast(StatefulNeuralGroup, tiny.groups[0])
    assert grp0.unit_r.grad is not None
    startup_grad_norm = float(grp0.unit_r.grad[old_n : old_n + 8].norm().item())

    small_bench = _benchmark_preset("native_byte_small", threads=threads, seq_bytes=512)
    mid_bench = _benchmark_preset("native_byte_mid", threads=threads, seq_bytes=512)

    payload = {
        "schema": "taiji-n38-native-preflight-v2",
        "architecture": "taiji-native-byte-group-v2",
        "non_transformer": True,
        "input_encoding": "raw_utf8_bytes_259",
        "threads": threads,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
        "init_sanity": {
            "init_loss_nats": round(init_loss, 4),
            "init_bpb": round(bits_per_byte(init_loss), 4),
            "healthy_random_range": 4.5 < init_loss < 6.5,
        },
        "neutral_birth_checks": {
            "svd_zero_in_max_abs_diff": delta_svd,
            "proliferate_max_abs_diff": delta_prolif,
            "residual_group_max_abs_diff": delta_group,
            "svd_startup_grad_norm": startup_grad_norm,
            "all_below_1e_5": max(delta_svd, delta_prolif, delta_group) < 1e-5,
        },
        "benchmarks": [small_bench, mid_bench],
    }
    out_path = REPO_ROOT / "reports" / "n38_2_01_native_backbone_preflight.json"
    write_report(out_path, payload)
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
