"""rev2b 止损后的补充测量（不重训、不改判据）：final circuit 的字节级 vs 位置级寻址命中。

背景：smoke 判据冻结在**位置级** top-1 ≥0.5，三轮读数停在 0.35–0.51 平台。本件用同一
circuit 只读测两种口径，供所有者裁定"线定错位（多字节汉字＋相邻步 query 缓变）还是真不行"。
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from train_taiji_r2_copy_circuit import make_episode  # noqa: E402


def main() -> int:
    from taiji import Taiji, TaijiConfig

    circuit_payload = torch.load(
        PROJECT_ROOT / "output/taiji_r2_copy_circuit/smoke/circuit-final.pt", weights_only=False
    )["copy_circuit"]

    substrate = Taiji(
        TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=20260925)
    )
    substrate.mount_copy_circuit(max_events=4)
    substrate.copy_circuit.load_payload(circuit_payload)
    circuit = substrate.copy_circuit
    assert circuit is not None

    rng = random.Random(999)
    pos_hits = byte_hits = steps = 0
    for _ in range(300):
        turns, answer = make_episode(rng)
        substrate.reset_dynamics(episode_id="rev2b-measure")
        substrate.observe(
            int(substrate.config.boundary_symbol),
            learn=False,
            readout="predictive",
            use_memory=False,
        )
        for symbol in turns[0].encode("utf-8"):
            substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
        circuit.store.record(
            turns[0].encode("utf-8"),
            substrate.fabric.cortical_context(substrate._state.regions).detach().cpu().clone(),
        )
        for turn in turns[1:]:
            for symbol in turn.encode("utf-8"):
                substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
        emitted: dict[int, int] = {}
        for byte in answer.encode("utf-8"):
            state = substrate._state
            snap = circuit.addressing(
                cue=substrate.fabric.cortical_context(state.regions).detach().cpu().clone(),
                f1_context=state.motor_context,
            )
            if snap is None:
                break
            codes = snap["codes"].tolist()
            matches = [i for i, c in enumerate(codes) if c == byte]
            want = matches[min(emitted.get(byte, 0), len(matches) - 1)] if matches else -1
            top1 = int(snap["scores"].argmax())
            steps += 1
            if want >= 0:
                pos_hits += int(top1 == want)
                # 字节级：top-1 位置的字节 == 目标字节（不要求同一位置）
                byte_hits += int(codes[top1] == byte)
            emitted[byte] = emitted.get(byte, 0) + 1
            substrate.observe(int(byte), learn=False, readout="predictive", use_memory=False)
    out = {
        "format": "taiji-r2-copy-circuit-addressing-measure-v1",
        "steps": steps,
        "position_top1": round(pos_hits / max(steps, 1), 4),
        "byte_top1": round(byte_hits / max(steps, 1), 4),
        "note": "只读测量，不改 smoke 判据；供所有者裁定位置线是否错位",
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
