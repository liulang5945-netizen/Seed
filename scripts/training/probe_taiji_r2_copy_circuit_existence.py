"""A2.2 结构存在性判据（立项 §4.1，零训练）：真件 checkpoint 上钉「内容 → F1 发射」通路。

合同测试（`tests/taiji_native/test_copy_circuit_contract.py`）钉机制；本件在**产品基座
base_16M（seed_beta.pt）**上出判决读数：

* **Arm-0 位级惰性**：挂载（gate=0、store 空）前后，同一 predictive 喂入的末位概率
  **逐位相同**（digest 相等）。
* **Arm-1 oracle 发射**：`record_told_content(告知字节)` 后，诊断覆写寻址 one-hot 指向
  「阿」首字节位置、gate 幅度覆写 20 ⇒ 冻结 greedy 生成的**第一个字节 = 0xE9（阿 的
  UTF-8 首字节）**，且末位 argmax 同值。

判读（冻结）：两臂皆真 ⇒ §4.1 成立（通路存在，A2.3 有训练资格）；任一臂假 ⇒ 当场翻案并红。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

TELL = "我叫阿岩。"
ASK = "我的名字是什么？"
A_FIRST_BYTE = 0xE9
A_FIRST_POSITION = 6  # 我(3B)+叫(3B) 之后
GATE_OVERRIDE = 20.0


def _feed_predictive(substrate, symbols):
    substrate.reset_dynamics(episode_id="r2-copy-existence")
    step = substrate.observe(substrate.config.boundary_symbol, learn=False, readout="predictive")
    for symbol in symbols:
        step = substrate.observe(int(symbol), learn=False, readout="predictive")
    return step.probabilities.detach().cpu().clone()


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from diag_taiji_r2_surface_decode import _serialize_prompt, generate

    from api.seed_runtime import SeedRuntime

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Arm-0：挂载前基线读数（同一份权重、同一喂入）。
    probe = SeedRuntime.load(path)
    tell_symbols = list(_serialize_prompt(probe, TELL))
    ask_symbols = list(_serialize_prompt(probe, ASK))

    runtime_before = SeedRuntime.load(path)
    probs_before = _feed_predictive(runtime_before.model.substrate, ask_symbols)

    runtime_mounted = SeedRuntime.load(path)
    substrate = runtime_mounted.model.substrate
    substrate.mount_copy_circuit()
    probs_after_mount = _feed_predictive(substrate, ask_symbols)
    inert = bool(torch.equal(probs_before, probs_after_mount))

    # Arm-1：写入告知内容 → oracle 寻址 + gate 覆写 → 冻结 greedy 生成。
    _feed_predictive(substrate, tell_symbols)
    substrate.record_told_content(TELL.encode("utf-8"))
    circuit = substrate.copy_circuit
    assert circuit is not None
    weights = torch.zeros(len(TELL.encode("utf-8")))
    weights[A_FIRST_POSITION] = 1.0
    circuit.address_override = weights
    circuit.gate_override = GATE_OVERRIDE
    probs_oracle = _feed_predictive(substrate, ask_symbols)
    argmax_ok = int(probs_oracle.argmax()) == A_FIRST_BYTE
    first_byte = generate(substrate, bytes(ask_symbols), "greedy")[:1]
    generation_ok = first_byte == bytes([A_FIRST_BYTE])

    verdict = inert and argmax_ok and generation_ok
    payload = {
        "format": "taiji-r2-copy-circuit-existence-v1",
        "prereg": "plans/reference/M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md §4.1",
        "checkpoint": path.name,
        "arm0_bitwise_inert": {
            "pass": inert,
            "digest_before": _digest(probs_before),
            "digest_after_mount": _digest(probs_after_mount),
        },
        "arm1_oracle_emission": {
            "argmax_pass": argmax_ok,
            "generation_first_byte_pass": generation_ok,
            "target_byte": A_FIRST_BYTE,
            "observed_first_byte": int(first_byte[0]) if first_byte else None,
            "gate_override": GATE_OVERRIDE,
        },
        "verdict": (
            "§4.1 成立：内容→F1 发射通路存在（A2.3 取得训练资格）"
            if verdict
            else "§4.1 失败：通路不存在，(c) 判定翻案"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_copy_circuit_existence_20260925.json")
    )
    if out.exists():
        parser.error(f"{out} already exists; 判决件不覆写")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
