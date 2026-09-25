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
    raw = runtime.model.generate_input(frame, MAX_ANSWER_BYTES, stop_at_boundary=True, sample=False)
    answer = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return answer.strip()


def run_arm(checkpoint: Path, circuit_payload: str | None) -> dict[str, Any]:
    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    if circuit_payload is not None:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit_payload)
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
        rows.append(
            {"id": item["id"], "dimension": item["dimension"], "hit": hit, "answer": answer[:60]}
        )
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
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    control = run_arm(checkpoint, None)
    treated = run_arm(checkpoint, args.circuit)
    verdict = (
        "A2.4 重测通过（D+E>0 且成句率不塌于对照）"
        if (
            treated["correct"] > control["correct"]
            and treated["correct"] > 0
            and treated["well_formed_rate"] >= control["well_formed_rate"]
        )
        else "A2.4 重测未通过（如实记录）"
    )
    report = {
        "format": "taiji-r2-copy-circuit-chat-cap-v1",
        "prereg": "plans/reference/M5_R2_A2_3_PREREG_20260925.md §4-S2（判据沿用）",
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "control_no_circuit": control,
        "treated_with_circuit": treated,
        "verdict": verdict,
    }
    print(
        json.dumps(
            {
                "control_correct": control["correct"],
                "treated_correct": treated["correct"],
                "control_well_formed": control["well_formed_rate"],
                "treated_well_formed": treated["well_formed_rate"],
                "items": control["items"],
                "verdict": verdict,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    out = (
        Path(args.out_report)
        if args.out_report
        else (PROJECT_ROOT / "reports" / "taiji_r2_copy_circuit_chat_cap_20260925.json")
    )
    if out.exists():
        from datetime import datetime, timezone

        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
