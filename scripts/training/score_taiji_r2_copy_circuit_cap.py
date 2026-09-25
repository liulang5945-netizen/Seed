"""A2.3 判读②：CAP D+E 抽查——挂载训练后 circuit 的产品基座 vs 无电路对照（同协议）。

预注册：`M5_R2_A2_3_PREREG_20260925.md` §4-S2②（判据：D+E 机器计分 > 0，对照 base=0/36 已钉）。

协议与训练逐字一致（跨 episode「告知→入库→提问→贪心 24 字节」，无中途 reset）；
唯一差别＝评测实体是冻结题集本身（阿岩/杭州/…），训练实体表与其不相交（预注册 §3）——
这是**泛化**测试，不是背题。对照臂不挂载电路（同协议、同题集）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
GEN_BYTES = 24


def _feed(substrate: Any, data: bytes) -> None:
    substrate.observe(
        int(substrate.config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    for symbol in data:
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _greedy(substrate: Any, n: int) -> bytes:
    out = bytearray()
    for _ in range(n):
        probs = substrate._state.motor_probabilities  # noqa: SLF001 - F1 分布（predictive 态）
        nxt = int(probs.argmax())
        if nxt == int(substrate.config.boundary_symbol):
            break
        out.append(nxt)
        substrate.observe(nxt, learn=False, readout="predictive", use_memory=False)
    return bytes(out)


def run_items(checkpoint: Path, circuit_payload: dict[str, Any] | None) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if circuit_payload is not None:
        substrate.mount_copy_circuit(max_events=4)
        assert substrate.copy_circuit is not None
        substrate.copy_circuit.load_payload(circuit_payload)
    items = [
        item
        for dim in ("D", "E")
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["dimensions"][dim]["items"]
        if item.get("expected_contains")
    ]
    rows = []
    for item in items:
        turns = list(item["turns"])
        tell = turns[0].encode("utf-8")
        substrate.reset_dynamics(episode_id=f"cap-{item['id']}")
        _feed(substrate, tell)
        if substrate.copy_circuit is not None:
            substrate.copy_circuit.store.clear()
            substrate.copy_circuit.store.record(
                tell,
                substrate.fabric.cortical_context(substrate._state.regions)
                .detach()
                .cpu()
                .clone(),  # noqa: SLF001
            )
        for turn in turns[1:]:
            _feed(substrate, turn.encode("utf-8"))
        answer = _greedy(substrate, GEN_BYTES)
        text = answer.decode("utf-8", errors="replace")
        hit = any(token in text for token in item["expected_contains"])
        rows.append({"id": item["id"], "hit": hit, "text": text[:60]})
    return {
        "items": len(rows),
        "correct": sum(1 for row in rows if row["hit"]),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default="output/taiji_r2_copy_circuit/judge/circuit-final.pt")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    base = Path(args.checkpoint)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    payload = None
    if args.circuit:
        cp = Path(args.circuit)
        if not cp.is_absolute():
            cp = PROJECT_ROOT / cp
        payload = torch.load(cp, weights_only=False)["copy_circuit"]

    control = run_items(base, None)
    treated = run_items(base, payload)
    report = {
        "format": "taiji-r2-copy-circuit-cap-spotcheck-v1",
        "prereg": "plans/reference/M5_R2_A2_3_PREREG_20260925.md §4-S2②",
        "checkpoint": base.name,
        "circuit": args.circuit or "none",
        "control_no_circuit": control,
        "treated_with_circuit": treated,
        "verdict": (
            "S2② 通过（D+E>0 且 > 对照）"
            if treated["correct"] > 0 and treated["correct"] > control["correct"]
            else "S2② 未通过"
        ),
    }
    print(
        json.dumps(
            {
                "control_correct": control["correct"],
                "treated_correct": treated["correct"],
                "items": control["items"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
        )
    )
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_copy_circuit_cap_spotcheck_20260925.json")
    )
    if out.exists():
        out = out.with_name(f"{out.stem}-{torch.randint(0, 999999, (1,)).item()}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
