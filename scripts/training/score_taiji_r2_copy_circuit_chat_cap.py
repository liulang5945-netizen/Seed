"""A2.4 重测：复制回路接进 chat 协议后，CAP D+E 全量＋成句率（对照臂 vs 电路臂）。

预注册判据沿用 `M5_R2_A2_3_PREREG_20260925.md` §4-S2（D+E>0）＋§4-③（成句率不塌，
本次补上正式仪器）。两臂同协议同题集，唯一差别＝是否挂载并启用剪贴板。

**口径如实声明**：计分读**基底原始字节答案**（`generate_input`），不经语言器官表层——
当前产品态 `chat_enabled=false` 下器官把答案统一替换为占位句（2026-09-23 真机记录），
经器官计分测到的是器官不是模型。除表层外本件走的全是产品原语：
`_serialize`、`_record_told_history`（A2.4 接线本体）、`generate_input(reset=True)`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
MAX_ANSWER_BYTES = 64


def _answer_raw(runtime: Any, prompt: str, history: list[tuple[str, str]]) -> str:
    from api.seed_runtime import _TURN_MARKERS
    from taiji import InputFrame

    text = runtime._serialize(prompt, history)
    substrate = runtime.model.substrate
    circuit = substrate.copy_circuit
    if circuit is not None:
        runtime._record_told_history(circuit, history)
    frame = InputFrame(
        input_id=f"a24-eval:{substrate.tick}",
        modality="text",
        payload=text.encode("utf-8"),
        source="r2.a24.eval",
        timestamp=substrate.tick,
        provenance="external",
        confidence=1.0,
    )
    raw = runtime.model.generate_input(
        frame, MAX_ANSWER_BYTES, stop_at_boundary=True, sample=False
    )
    answer = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return answer.strip()


def run_arm(
    checkpoint: Path,
    circuit_payload: str | None,
    *,
    evidence_utf8_gate: bool = False,
) -> dict[str, Any]:
    """一臂：CAP 的 D+E 计分（基底原始字节）。

    `evidence_utf8_gate`（PLAN-A-25）：只在评测期把复制回路的加性证据按 UTF-8 位置状态门控
    ——默认 False ⇒ 与冻结链逐位相同；开启走 `Taiji.set_copy_evidence_utf8_gate` 运行时覆写。
    """

    from api.seed_runtime import SeedRuntime
    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    runtime = SeedRuntime.load(checkpoint)
    if circuit_payload is not None:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit_payload)
    if evidence_utf8_gate:
        runtime.model.substrate.set_copy_evidence_utf8_gate(True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [
        {**item, "dimension": dim}
        for dim in ("D", "E")
        for item in manifest["dimensions"][dim]["items"]
        if item.get("expected_contains")
    ]
    rows = []
    for item in items:
        history: list[tuple[str, str]] = []
        turns = list(item["turns"])
        answer = ""
        for index, turn in enumerate(turns):
            answer = _answer_raw(runtime, turn, history)
            if index + 1 < len(turns):
                history.append((turn, answer))
        hit = any(token in answer for token in item["expected_contains"])
        rows.append({"id": item["id"], "dimension": item["dimension"], "hit": hit, "answer": answer[:60]})
    ngram = build_ngram_model()
    return {
        "items": len(rows),
        "correct": sum(1 for row in rows if row["hit"]),
        "well_formed_rate": round(
            sum(1 for row in rows if well_formed(row["answer"], ngram)) / len(rows), 4
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default="output/taiji_r2_copy_circuit/judge/circuit-final.pt")
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--copy-evidence-utf8-gate",
        action="store_true",
        help="PLAN-A-25：把复制回路的加性证据按 UTF-8 位置状态门控（默认关 ⇒ 与冻结链逐位相同）",
    )
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    gate = bool(args.copy_evidence_utf8_gate)
    circuit_path = PROJECT_ROOT / args.circuit if args.circuit else None
    circuit_sha256 = (
        hashlib.sha256(circuit_path.read_bytes()).hexdigest()
        if circuit_path is not None and circuit_path.is_file()
        else None
    )
    control = run_arm(checkpoint, None)
    treated = run_arm(checkpoint, args.circuit, evidence_utf8_gate=gate)
    verdict = "A2.4 重测通过（D+E>0 且成句率不塌于对照）" if (
        treated["correct"] > control["correct"] and treated["correct"] > 0
        and treated["well_formed_rate"] >= control["well_formed_rate"]
    ) else "A2.4 重测未通过（如实记录）"
    report = {
        "format": "taiji-r2-copy-circuit-chat-cap-v1",
        "prereg": "plans/reference/M5_R2_A2_3_PREREG_20260925.md §4-S2（判据沿用）",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        #: 复现性债的修法（PLAN-A-24 rev22）：光记**路径**不够——同名路径会被后来的长跑覆盖
        #: （实测：`output/taiji_r2_copy_circuit/judge/circuit-final.pt` 现在跑出 0/16，
        #: 而登记值是 2/16）。记下 payload 指纹，旧判读件才既验得了"当时用的是哪份权重"、
        #: 也不至于把覆盖后的新权重当成旧读数。
        "circuit_sha256": circuit_sha256,
        #: PLAN-A-25：门开/关必须落在件上，否则两份读数看起来像同一次实验。
        "copy_evidence_utf8_gate": gate,
        "control_no_circuit": control,
        "treated_with_circuit": treated,
        "verdict": verdict,
    }
    print(json.dumps({
        "control_correct": control["correct"], "treated_correct": treated["correct"],
        "control_well_formed": control["well_formed_rate"],
        "treated_well_formed": treated["well_formed_rate"],
        "items": control["items"], "verdict": verdict,
    }, ensure_ascii=False), flush=True)
    out = Path(args.out_report) if args.out_report else (
        PROJECT_ROOT / "reports" / "taiji_r2_copy_circuit_chat_cap_20260925.json")
    if out.exists():
        from datetime import datetime, timezone
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
