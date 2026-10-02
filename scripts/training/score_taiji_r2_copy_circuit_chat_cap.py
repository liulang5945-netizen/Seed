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
from datetime import UTC
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/cap0_eval_set_v2.json"
MAX_ANSWER_BYTES = 64


def _answer_raw(
    runtime: Any,
    prompt: str,
    history: list[tuple[str, str]],
    *,
    utf8_strict: bool = False,
    repetition_penalty: float = 0.0,
    max_bytes: int = MAX_ANSWER_BYTES,
) -> str:
    """产品装配下取基底原始答复（SPEC-A-21 那条冻结链）。

    `utf8_strict=True`＝把**产品表层链用的同一把解码掩码**（SPEC-R2-02，`SeedRuntime.chat` 里
    `generate_input(..., utf8_strict=True)`）套到这条评测链上——默认 `False` ⇒ 冻结读数逐位不变。
    为什么要有这一档：`PLAN-A-30` 要分「是掩码放大了电路的复读，还是语言器官」，
    而只有把掩码单独加在原始字节链上，才能把"掩码"与"器官"这两手分开量（不然只有两端可比）。
    `max_bytes`＝生成预算，默认 64（冻结面逐位不变）。**它是 `PLAN-A-30` §2h 量出来的一个口径缺陷**：
    产品 `chat()` 的预算是 256，而这条评测链一直是 64 ⇒ 跨链比较"表层比原始链差"时，
    链与预算两个变量同时动了。要分开就得能把预算拨到 256（`probe_taiji_a30_surface_repetition.py`
    的 `--max-bytes` 走的就是这一手）。
    """

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
        frame,
        max_bytes,
        stop_at_boundary=True,
        sample=False,
        utf8_strict=utf8_strict,
        repetition_penalty=repetition_penalty,
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
    evidence_alpha: float = 1.0,
    max_bytes: int = MAX_ANSWER_BYTES,
    limit: int | None = None,
) -> dict[str, Any]:
    """一臂：CAP 的 D+E 计分（基底原始字节）。

    `evidence_utf8_gate`（PLAN-A-25）：只在评测期把复制回路的加性证据按 UTF-8 位置状态门控
    ——默认 False ⇒ 与冻结链逐位相同；开启走 `Taiji.set_copy_evidence_utf8_gate` 运行时覆写。
    `max_bytes`＝生成预算，默认 64（冻结面逐位不变）；`PLAN-A-30` §2h/§2i 查出预算本身就是
    一个会动读数的变量（同链同装配把 64 拨到 256，命中 3→9、成句 13→6）⇒ 用它做同装配双预算档。
    `limit`＝只取前 N 道题（D/E 混排后取前 N，两臂用同一子集 ⇒ 配对成立；默认全量）。
    """

    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    if circuit_payload is not None:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit_payload)
    if evidence_utf8_gate:
        runtime.model.substrate.set_copy_evidence_utf8_gate(True)
    calls = [0]
    if evidence_alpha != 1.0:
        # PLAN-A-30 §DEBT-G19 的剂量档：复用剂量探针那个接口级包装（不另写一份缩放）。
        from probe_taiji_a30_copy_evidence_dose import _make_scaled_evidence

        circuit = runtime.model.substrate.copy_circuit
        if circuit is None:
            raise RuntimeError("要求缩放证据但回路不在场 ⇒ 这一臂没有可缩放的通道")
        scaled, calls = _make_scaled_evidence(circuit.evidence, evidence_alpha)
        circuit.evidence = scaled
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [
        {**item, "dimension": dim}
        for dim in ("D", "E")
        for item in manifest["dimensions"][dim]["items"]
        if item.get("expected_contains")
    ]
    if limit is not None:
        items = items[:limit]
    ngram = build_ngram_model()
    rows = []
    for item in items:
        history: list[tuple[str, str]] = []
        turns = list(item["turns"])
        answer = ""
        for index, turn in enumerate(turns):
            answer = _answer_raw(runtime, turn, history, max_bytes=max_bytes)
            if index + 1 < len(turns):
                history.append((turn, answer))
        hit = any(token in answer for token in item["expected_contains"])
        #: `formed_full`＝**整条答复**过 `well_formed`（与下面那把只看 60 字符前缀的尺子不同，
        #: 见 `well_formed_scope`）。三者合起来才回答"回路买到的到底是词在场，还是一句能看的答案"
        #: ——`DEBT-G12` 要的那把联合判据。旧字段 `correct`／`well_formed_rate` 语义逐位不变。
        rows.append(
            {
                "id": item["id"],
                "dimension": item["dimension"],
                "hit": hit,
                "formed_full": bool(well_formed(answer, ngram)),
                "answer": answer[:60],
            }
        )
    return {
        "items": len(rows),
        "correct": sum(1 for row in rows if row["hit"]),
        "joint_hits": sum(1 for row in rows if row["hit"] and row["formed_full"]),
        "formed_full_texts": sum(1 for row in rows if row["formed_full"]),
        "well_formed_rate": round(
            sum(1 for row in rows if well_formed(row["answer"], ngram)) / len(rows), 4
        ),
        "rows": rows,
        #: 剂量档的"被走到"计数（`_make_scaled_evidence` 实际消费了几次）。
        "evidence_calls": calls[0],
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
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_ANSWER_BYTES,
        help="生成预算，默认 64＝冻结面；PLAN-A-30 §2h 用它跑同装配的双预算档",
    )
    parser.add_argument(
        "--copy-evidence-alpha",
        type=float,
        default=1.0,
        help="DEBT-G19 剂量档：治疗臂把回路加性证据乘 α（默认 1.0 ⇒ 与已入库两臂档逐位可比）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只取前 N 道题（D/E 混排后取前 N，两臂同一子集）",
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
    control = run_arm(checkpoint, None, max_bytes=args.max_bytes, limit=args.limit)
    treated = run_arm(
        checkpoint,
        args.circuit,
        evidence_utf8_gate=gate,
        evidence_alpha=args.copy_evidence_alpha,
        max_bytes=args.max_bytes,
        limit=args.limit,
    )
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
        #: 复现性债的修法（PLAN-A-24 rev22）：光记**路径**不够——同名路径会被后来的长跑覆盖
        #: （实测：`output/taiji_r2_copy_circuit/judge/circuit-final.pt` 现在跑出 0/16，
        #: 而登记值是 2/16）。记下 payload 指纹，旧判读件才既验得了"当时用的是哪份权重"、
        #: 也不至于把覆盖后的新权重当成旧读数。
        "circuit_sha256": circuit_sha256,
        #: PLAN-A-25：门开/关必须落在件上，否则两份读数看起来像同一次实验。
        "copy_evidence_utf8_gate": gate,
        #: DEBT-G19 剂量档：治疗臂的证据乘数（1.0 ⇒ 与已入库两臂档逐位可比）。
        "copy_evidence_alpha": args.copy_evidence_alpha,
        #: 两条口径必须落在件上，否则这份读数会被当成"整条答复、预算 256"的那类去比：
        #: ①生成预算（`PLAN-A-30` §2h 实测同一链同一装配 64→256 会让命中 3→9、成句 13→6）；
        #: ②成句率量的是 `answer[:60]` **字符前缀**，不是整条答复（与 `probe_taiji_a30_*` 的
        #:    全文口径不同 ⇒ 两个"成句"不许互换）。
        "max_bytes": args.max_bytes,
        "item_limit": args.limit,
        "well_formed_scope": "answer[:60] 字符前缀（非整条答复）",
        "control_no_circuit": control,
        "treated_with_circuit": treated,
        "dose_guard": {
            "treated_consumed_scaler": bool(treated.get("evidence_calls")),
            "control_consumed_scaler": bool(control.get("evidence_calls")),
            "evidence_calls_by_arm": {
                "control": control.get("evidence_calls"),
                "treated": treated.get("evidence_calls"),
            },
        },
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
        from datetime import datetime

        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
