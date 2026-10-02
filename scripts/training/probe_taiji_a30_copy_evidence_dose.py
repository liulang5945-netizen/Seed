"""复制回路加性证据的**剂量—响应**探针（零训练；缩放发生在仪器侧，不动产品源码）。

为什么要这一档：`taiji/model.py:2209-2228` 实读到——三条加性证据通道里 consolidated 前有
`consolidation_read_gain`(1.00)、memory 前有 `memory_read_gain`(3.00)、身份证据带
`identity_organ_evidence_gain`(16.00)，**只有复制回路是 `episodic_evidence + copy_circuit.evidence(...)`，
没有任何强度系数**。两端读数已在案（同一枚 (c) 件、同一批 300 篇：不挂回路 **186/300**、挂 seed-A **0/300**），
所以中间只补剂量档，不必再训件。

做法＝**接口级**包一层：把 `CopyCircuit.evidence` 的返回值乘 α，跑完撤掉实例属性回到类上的绑定方法。
刻意不碰实现行、不碰产品源码。

三条守卫（都设计成能为 false）：
* `unit_dose_bitwise_identical`——α=1 那档与"完全不包层"的那一趟必须**逐位相同**（否则实验连自己都没做对）；
* `wrapper_consumed_on_every_dose`＋每档调用次数——计数为 0 就说明补丁没被消费（"写了个不被走到的旋钮"是这类实验的经典假读数）；
* `checkpoint_untouched`——审计只读，落盘前后检查点 sha 必须相同。

α=0 只**报告**不硬失败：它是"算完再乘零"，与"根本没挂载"在这条审计路径上应当同值；若不同值，
正好暴露挂载还有第二条影响面，那是更值得知道的事。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import torch  # noqa: E402
from audit_taiji_a30_stop_signal_presence import audit, document_symbols  # noqa: E402


def _end_face_scalars(result: dict[str, Any]) -> dict[str, Any]:
    """只取比较用的标量列（不含调用计数），逐位比较时两边形状必须一致。"""

    end = result["faces"]["end"]
    return {
        "n": end["n"],
        "boundary_is_argmax_count": end["boundary_is_argmax_count"],
        "p_boundary_median": end["p_boundary"]["median"],
        "p_boundary_mean": end["p_boundary"]["mean"],
        "p_boundary_max": end["p_boundary"]["max"],
        "boundary_rank_median": end["boundary_rank"]["median"],
        "argmax_winners_top5": [
            (int(row["symbol"]), int(row["count"])) for row in end["argmax_winners_top5"]
        ],
        #: v1 加性：并取 `other` 面三条。理由见 `p_boundary_median_other` 的用途——它用来**独立证明**
        #: "下限确实改变了轨迹上的打分"。若接缝面不动而 other 面也不动，那说明下限根本没生效，
        #: 此时"接缝位置自己落在高相似度那一段"这条推论就不成立（这是那条推论的否证检验）。
        "other_n": result["faces"]["other"]["n"],
        "other_boundary_is_argmax_count": result["faces"]["other"]["boundary_is_argmax_count"],
        "other_p_boundary_median": result["faces"]["other"]["p_boundary"]["median"],
    }


def _make_scaled_evidence(
    original: Any,
    alpha: float,
    record: list[Any] | None = None,
    *,
    circuit: Any | None = None,
    scores: list[float] | None = None,
    floor_tau: float | None = None,
) -> tuple[Any, list[int]]:
    """接口级包装：把 `evidence(**kwargs)` 的返回值乘 α，并数它**被消费**了几次。

    传 `record` 就顺手量注入的**硬度**（这条通道是 `gate * distribution`，`distribution` 是池化进
    257 维的软权重 ⇒ 量级全在学出来的那个标量 `gate` 上）。记的是每次调用的
    `(最大值, L1 和, 非零个数)`——用来判"加性证据"实际是不是当成硬值在用。
    """

    calls = [0]

    def scaled(**kwargs: Any) -> Any:
        out_tensor = original(**kwargs)
        calls[0] += 1
        #: 相似度走**产品自己的那条算式**（`CopyCircuit._cosine` 的 docstring 明写它与 `best_match`
        #: 同一套），所以这里不另实现一份打分。但"算完再丢"与"根本不发"在**计数器副作用**上不等价
        #: （`_chosen_event` 的锁丢弃计数照旧会走），件里如实披露这一条。
        if (scores is not None or floor_tau is not None) and circuit is not None:
            event = circuit.store.best_match(kwargs["cue"])
            score = None if event is None else float(circuit._cosine(kwargs["cue"], event.cue))
            if scores is not None:
                scores.append(-1.0 if score is None else score)
            if floor_tau is not None and (score is None or score < floor_tau):
                return out_tensor * 0.0
        if record is not None:
            with torch.no_grad():
                record.append(
                    (
                        float(out_tensor.max()),
                        float(out_tensor.abs().sum()),
                        int(torch.count_nonzero(out_tensor)),
                    )
                )
        return out_tensor * alpha

    return scaled, calls


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--corpus", default="data/simple_zh/dialogue_extended_clean.jsonl")
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--append-newline", action="store_true")
    parser.add_argument(
        "--doses", default="0.25,0.50,1.0", help="逗号分隔的 α；含 1.0 时自动做逐位锚点复检"
    )
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--record-scores",
        action="store_true",
        help="记录每一步被挑中告知的余弦相似度（τ 从这个分布里取，不拍脑袋定）",
    )
    parser.add_argument(
        "--relevance-floor-tau",
        type=float,
        default=None,
        help="DEBT-G19 修法①的模拟：相似度低于 τ ⇒ 证据归零（模拟 best_match 返回 None）",
    )
    parser.add_argument(
        "--record-magnitudes",
        action="store_true",
        help="顺手量这条通道的注入硬度（`(max, L1, nnz)` 三列分位数）——用来验'它是硬值还是软加性'",
    )
    args = parser.parse_args()

    out = Path(args.out_report)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    if out.exists():
        print(f"[拒绝落盘] {out} 已存在 ⇒ 不许覆盖既有证据件，换个文件名再跑", file=sys.stderr)
        return 2

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    corpus = Path(args.corpus)
    corpus = corpus if corpus.is_absolute() else PROJECT_ROOT / corpus

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(PROJECT_ROOT / args.circuit)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:  # 没有可缩放的东西就响亮停，别静默给一排 0
        raise RuntimeError("copy_circuit 不在场 ⇒ 没有可缩放的证据通道，本档不成立")
    boundary = int(substrate.config.boundary_symbol)
    sample = document_symbols(corpus, args.docs, append_newline=args.append_newline)
    chunks = sample["chunks"]

    # 不包任何一层的那一趟：α=1 的逐位锚点。
    unpatched = _end_face_scalars(audit(runtime, chunks, boundary, mask=args.mask))

    doses: dict[str, Any] = {}
    unit_dose_identical: bool | None = None
    for alpha in [float(x) for x in args.doses.split(",") if x.strip()]:
        circuit = substrate.copy_circuit
        original_bound = circuit.evidence
        magnitudes: list[Any] | None = [] if args.record_magnitudes else None
        scores: list[float] | None = [] if args.record_scores else None
        scaled, calls = _make_scaled_evidence(
            original_bound,
            alpha,
            record=magnitudes,
            circuit=circuit,
            scores=scores,
            floor_tau=args.relevance_floor_tau,
        )
        circuit.evidence = scaled
        try:
            row = _end_face_scalars(audit(runtime, chunks, boundary, mask=args.mask))
        finally:
            del circuit.evidence  # 撤实例属性 ⇒ 下一档拿到的仍是类上的绑定方法
        assert circuit.evidence == original_bound, "补丁没撤干净 ⇒ 后面的档会叠乘"
        if alpha == 1.0:
            #: 逐位锚点**只在下限关闭时**才是"包层不扰动"的检验；下限开着时两趟**本就该不同**
            #: （接缝与 other 的差别正是那一枪的内容），所以那时把这条读成 false 是误读。
            #: 于是下限开着时报 `None`（不适用），而不是报 false。
            unit_dose_identical = (
                None if args.relevance_floor_tau is not None else (row == unpatched)
            )
            if args.relevance_floor_tau is not None:
                row["anchor_not_applicable_reason"] = "floor is active ⇒ two passes should differ"
        row["evidence_calls"] = calls[0]
        if scores:
            ordered = sorted(scores)
            row["picked_cosine"] = {
                "median": round(statistics.median(ordered), 4),
                "p10": round(ordered[int(0.10 * (len(ordered) - 1))], 4),
                "p25": round(ordered[int(0.25 * (len(ordered) - 1))], 4),
                "p75": round(ordered[int(0.75 * (len(ordered) - 1))], 4),
                "p90": round(ordered[int(0.90 * (len(ordered) - 1))], 4),
                "max": round(ordered[-1], 4),
                "share_below_0p3": round(sum(1 for x in ordered if 0 <= x < 0.3) / len(ordered), 4),
                "share_no_event": round(sum(1 for x in ordered if x < 0) / len(ordered), 4),
            }
            scores.clear()
        if magnitudes:
            row["injected_logit_max"] = {
                "median": round(statistics.median([m[0] for m in magnitudes]), 4),
                "p90": round(sorted(magnitudes)[int(0.9 * (len(magnitudes) - 1))][0], 4),
                "max": round(max(m[0] for m in magnitudes), 4),
            }
            row["injected_l1_median"] = round(statistics.median([m[1] for m in magnitudes]), 4)
            row["injected_nnz_median"] = statistics.median([m[2] for m in magnitudes])
            row["empty_evidence_share"] = round(
                sum(1 for m in magnitudes if m[2] == 0) / len(magnitudes), 4
            )
            magnitudes.clear()
        doses[str(alpha)] = row

    sha_after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "format": "taiji-a30-copy-evidence-dose-v1",
        "question": "把复制回路的加性证据按 α 缩放／按相似度下限截断，语料接缝上的停止信号能不能回来",
        "relevance_floor_tau": args.relevance_floor_tau,
        "checkpoint": args.checkpoint,
        "circuit": args.circuit,
        "circuit_sha256": hashlib.sha256((PROJECT_ROOT / args.circuit).read_bytes()).hexdigest()[
            :16
        ],
        "docs": args.docs,
        "documents_face": (
            "corpus_body_plus_appended_newline" if args.append_newline else "corpus_body"
        ),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "anchors_on_record": {
            "circuit_absent_same_checkpoint": "186/300（档 (c) 不挂回路，docs_sha256 同批）",
            "circuit_mounted_full_dose": "0/300（档 (c)＋seed-A）",
        },
        "unpatched_end_face": unpatched,
        "doses": doses,
        "instrument_guard": {
            "unit_dose_bitwise_identical": unit_dose_identical,
            "wrapper_consumed_on_every_dose": all(r["evidence_calls"] > 0 for r in doses.values()),
            "evidence_calls_by_dose": {k: v["evidence_calls"] for k, v in doses.items()},
            "checkpoint_untouched": sha_before == sha_after,
        },
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {"doses": report["doses"], "guard": report["instrument_guard"]}, ensure_ascii=False
        )[:1500]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
