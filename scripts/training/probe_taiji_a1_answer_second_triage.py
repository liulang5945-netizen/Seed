"""A1：v3 落位=最新的 52 题逐步轨迹分诊（零训练）→ 三选一归属：键侧／出口竞争／缺承诺。

预注册：`PLAN-A-24` §5 队列 A1 行（"落位=最新的 52 题逐步轨迹诊断：copy 质量落点／
发出来的是哪条的内容／目标字节名次"）。用途：按归属取其一——
键侧 ⇒ G4=B1（`--lr-embed` 两臂）；出口竞争 ⇒ G2（阈值＋竞争抑制）；缺承诺 ⇒ G3（复述承诺状态）。

**与 A2.6 探针的关系（为何不直接调 `trace_item`）**：A2.6 成形于 A2.5"整轮锁定"（2026-09-26）
之前，其逐步 `addressing` 消费的是**上一轮 generate 残留的旧锁**，与记分链（提问喂完那一刻
重新上锁，`model.py:2981-2989`）在锁定事件上不同源——V097 类"轨迹对、记分件错"的矛盾即源于此。
本件的轨迹改为**锁感知**，且发射轨迹不由手算重构，而由产品 `observe` 链的概率逐步驱动
（与 `generate` 解码环逐位同构：第 0 步＝喂入末位概率（锁前），第 1 步起＝上锁后的 observe 概率，
边界符即停——`stop_at_boundary` 语义）。寻址快照只做诊断（copy 指向／门开度／名次），
其与产品 `evidence()` 路径的一致性由两处链路同一性检查守卫（锁前 k=0、锁后 k=1 各一次）。

分诊规则（复用 A2.6 冻结的 `_classify`，先于跑冻结）＋三选一映射：
* `address_miss`（copy 分布一步都没指到目标字节）→ **键侧** ⇒ G4=B1；
* `emission_loses`（copy 指对了、argmax 仍不发）→ **出口竞争** ⇒ G2；
* `continuation_slips`（有指对且发出来过，整串仍断）→ **缺承诺** ⇒ G3；
* `gate_closed` 单列（训练制度侧；本件电路是训过的，预期 0，非 0 回炉看仪器）；
* `emitted` 计数必须为 0（非 0 ⇒ 本探针与记分件构造不一致，回炉）。

判读纪律：两电路（seed-A／seed-B）各自出占比，主归属**同向**才判，否则 `not_resolved`；
每题轨迹与重构答案入判读件。基座 sha 跑前后复核。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
#: 记分链读数（seed-A／seed-B 两臂的电路 payload 路径与逐题命中从此机器读取，不手抄）。
SURFACE_REPORT = PROJECT_ROOT / "reports/taiji_r2_a25_lock_only_surface_v3_20260926.json"
MAX_STEPS = 12
FORCED_GATE = 20.0

CLASS_TO_ATTRIBUTION = {
    "address_miss": "key_side",
    "emission_loses": "emission_competition",
    "continuation_slips": "commitment_missing",
    "gate_closed": "gate_training",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer_second_items() -> tuple[list[dict[str, Any]], list[str]]:
    """v3 的 `answer_tell_position==1` 题面 + 两电路 payload 路径（机器读取，不手抄）。"""
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = [
        dict(item)
        for item in payload["dimensions"]["X"]["items"]
        if int(item["answer_tell_position"]) == 1
    ]
    data = json.loads(SURFACE_REPORT.read_text(encoding="utf-8"))
    circuits = [str(arm["circuit"]) for arm in data["treated_arms"]]
    return items, circuits


def _trace_locked(
    runtime: Any,
    *,
    turns: list[str],
    tokens: list[str],
    told: str,
    item_id: str,
) -> dict[str, Any]:
    """锁感知的单题轨迹：发射走产品 observe 链，寻址快照只做诊断。"""
    from probe_taiji_r2_a26_emission_trace import _classify, _scored_history

    from api.seed_runtime import SeedRuntime, record_told_history

    substrate = runtime.model.substrate
    circuit = substrate.copy_circuit
    config = substrate.config
    answer = next((token for token in tokens if token in told), "")
    if not answer:
        return {"id": item_id, "class": "no_copyable_answer"}
    answer_bytes = answer.encode("utf-8")

    history = _scored_history(runtime, turns)
    prompt = SeedRuntime._serialize(turns[-1], history)
    record_told_history(substrate, circuit, history, episode_id="a1:final")
    prompt_bytes = prompt.encode("utf-8")

    #: 喂入段（产品 generate 的喂法逐字同构：reset → 边界符 → 逐字节 learn=False）。
    substrate.reset_dynamics(episode_id="generation")
    substrate.observe(
        int(config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    step = None
    for symbol in prompt_bytes:
        step = substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
    assert step is not None
    #: 第 0 步发射用的就是这份概率（产品解码环的第 0 步取喂入末位 observe，锁前）。
    feed_probs = step.probabilities.detach().cpu().clone()

    def _snap(prev_byte: int) -> dict[str, Any] | None:
        state = substrate._state
        return circuit.addressing(
            cue=substrate.fabric.cortical_context(state.regions),
            f1_context=state.motor_context,
            prev_byte=prev_byte,
        )

    def _diag_row(
        snap: dict[str, Any], k: int, target_byte: int, argmax_byte: int
    ) -> dict[str, Any]:
        distribution = snap["copy_distribution"]
        return {
            "step": k,
            "argmax_byte": argmax_byte,
            "target_byte": target_byte,
            "copy_top_byte": int(distribution.argmax()),
            "copy_mass_on_target": round(float(distribution[target_byte]), 6),
            "target_rank_in_copy": int((distribution > float(distribution[target_byte])).sum() + 1),
            "gate_value": round(float(snap["gate_value"]), 4),
            "on_target_event": bytes(snap["event"].content) == told.encode("utf-8"),
            "emitted": argmax_byte == target_byte,
        }

    #: 链路同一性（锁前）：手算读法 == 喂入末位概率 ⇒ 诊断快照与产品 evidence 路径同源。
    chain_ok = True
    pre_snap = _snap(int(prompt_bytes[-1]))
    readout = substrate.predictive_readout
    if pre_snap is not None:
        state = substrate._state
        base_evidence = float(
            config.consolidation_read_gain
        ) * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
        with_copy = readout.probabilities(
            state.motor_context,
            episodic_evidence=base_evidence
            + pre_snap["gate_value"] * pre_snap["copy_distribution"],
        )
        chain_ok = bool(torch.equal(with_copy, feed_probs))

    #: 产品 generate 的上锁位置：提问喂完之后、解码环之前（model.py:2981-2989）。
    lock_state = circuit.lock_selection(
        cue=substrate.cortical_cue(),
        f1_context=substrate._state.motor_context,
        query_bytes=prompt_bytes,
    )
    locked_event_content = (
        bytes(lock_state["event"].content).decode("utf-8", errors="replace")
        if lock_state is not None
        else None
    )

    events = [event.content.decode("utf-8", errors="replace") for event in circuit.store.events()]
    rows: list[dict[str, Any]] = []
    emitted_bytes = bytearray()
    prev_byte = int(prompt_bytes[-1])
    current_probs = feed_probs
    boundary = int(config.boundary_symbol)
    stopped_at_boundary = False
    for k in range(min(MAX_STEPS, len(answer_bytes))):
        target_byte = int(answer_bytes[k])
        argmax_byte = int(current_probs.argmax())
        if argmax_byte == boundary:
            stopped_at_boundary = True
            break
        snap = _snap(prev_byte)
        if snap is None:
            break
        row = _diag_row(snap, k, target_byte, argmax_byte)
        if k == 0:
            row["emission_source"] = "feed_probs_pre_lock"
        else:
            row["emission_source"] = "observe_probs_post_lock"
            #: 链路同一性（锁后）：诊断快照的手算读法 == 本步 observe 概率。
            state = substrate._state
            base_evidence = float(
                config.consolidation_read_gain
            ) * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
            with_copy = readout.probabilities(
                state.motor_context,
                episodic_evidence=base_evidence + snap["gate_value"] * snap["copy_distribution"],
            )
            if not torch.equal(with_copy, current_probs):
                chain_ok = False
        forced = readout.probabilities(
            substrate._state.motor_context,
            episodic_evidence=(
                float(config.consolidation_read_gain)
                * substrate.fabric.consolidated_decode(0, substrate._state.regions[0].trace)
                + FORCED_GATE * snap["copy_distribution"]
            ),
        )
        row["forced_open_would_hit"] = bool(int(forced.argmax()) == target_byte)
        rows.append(row)
        emitted_bytes.append(argmax_byte)
        prev_byte = argmax_byte
        step = substrate.observe(argmax_byte, learn=False, readout="predictive", use_memory=False)
        current_probs = step.probabilities.detach().cpu()

    reconstructed = emitted_bytes.decode("utf-8", errors="replace")
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    replay = _answer_raw(runtime, turns[-1], history)
    return {
        "id": item_id,
        "told": told,
        "answer": answer,
        "locked_event_content": locked_event_content,
        "store_contents": events,
        "chain_identical": chain_ok,
        "stopped_at_boundary": stopped_at_boundary,
        "steps": len(rows),
        "reconstructed_answer": reconstructed[:80],
        "scored_replay_answer": replay[:80],
        "reproduces_recorded_miss": answer not in replay,
        "contradicts_failure_list": answer in reconstructed,
        "class": _classify(rows),
        "trace": rows,
    }


def run_circuit_arm(
    items: list[dict[str, Any]], checkpoint: Path, circuit_path: str, limit: int | None
) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    payload = Path(circuit_path)
    runtime.enable_copy_circuit(payload if payload.is_absolute() else PROJECT_ROOT / payload)
    if runtime.model.substrate.copy_circuit is None:
        raise RuntimeError("copy circuit did not mount")

    rows = [
        _trace_locked(
            runtime,
            turns=[str(turn) for turn in item["turns"]],
            tokens=[str(token) for token in item["expected_contains"]],
            told=item["turns"][int(item["answer_tell_position"])],
            item_id=str(item["id"]),
        )
        for item in items[:limit]
    ]
    return {"circuit": circuit_path, "rows": rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    diagnostic = {key: value for key, value in counts.items() if key in CLASS_TO_ATTRIBUTION}
    primary = max(diagnostic, key=lambda key: diagnostic[key]) if diagnostic else None
    locked_to_target = sum(
        1
        for row in rows
        if row.get("locked_event_content") is not None
        and any(row["locked_event_content"] == told for told in [row.get("told")])
    )
    return {
        "class_counts": counts,
        "diagnostic_counts": diagnostic,
        "primary_class": primary,
        "attribution": CLASS_TO_ATTRIBUTION.get(primary) if primary else None,
        "lock_picked_target_event": f"{locked_to_target}/{len(rows)}",
        "contradictions": sum(1 for row in rows if row.get("contradicts_failure_list")),
        "chain_mismatches": sum(1 for row in rows if not row.get("chain_identical", True)),
        "boundary_stops": sum(1 for row in rows if row.get("stopped_at_boundary")),
        "mean_steps_on_wrong_event": round(
            sum(
                sum(1 for step in row.get("trace", []) if not step.get("on_target_event", True))
                for row in rows
            )
            / max(len(rows), 1),
            4,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--limit", type=int, default=None, help="smoke 用；缺省 52 题")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    items, circuits = _answer_second_items()
    if not items or len(circuits) != 2:
        print(
            json.dumps(
                {"guard_ok": False, "error": f"items={len(items)} circuits={len(circuits)}"},
                ensure_ascii=False,
            )
        )
        return 2

    arms = [run_circuit_arm(items, checkpoint, circuit, args.limit) for circuit in circuits]
    summaries = [summarize(arm["rows"]) for arm in arms]
    attributions = {summary["attribution"] for summary in summaries}
    verdict = (
        attributions.pop().upper()
        if len(attributions) == 1 and None not in attributions
        else "not_resolved"
    )
    report = {
        "format": "taiji-a1-answer-second-triage-v2",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §5 A1",
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "surface_report": str(SURFACE_REPORT.relative_to(PROJECT_ROOT)),
        "denominator": "v3 answer_tell_position==1 全部 52 题（记分链 seed-A 51/52、seed-B 52/52 未命中）",
        "class_to_attribution": CLASS_TO_ATTRIBUTION,
        "trace_design": (
            "锁感知：第 0 步发射=喂入末位概率(锁前)、第 1 步起=上锁后 observe 概率、边界符即停；"
            "寻址快照只做诊断，链路同一性在锁前/锁后各验一次"
        ),
        "arms": [
            {
                "circuit": arm["circuit"],
                "summary": summary,
                "rows": [
                    {
                        key: row[key]
                        for key in (
                            "id",
                            "class",
                            "chain_identical",
                            "contradicts_failure_list",
                            "locked_event_content",
                            "store_contents",
                            "reconstructed_answer",
                            "scored_replay_answer",
                            "reproduces_recorded_miss",
                            "stopped_at_boundary",
                        )
                    }
                    for row in arm["rows"]
                ],
            }
            for arm, summary in zip(arms, summaries, strict=True)
        ],
        "verdict": verdict,
        "what_would_overturn": (
            "两电路主归属不一致 ⇒ not_resolved；emitted 计数非 0 ⇒ 仪器与记分件构造不一致，回炉；"
            "chain_identical 为假非 0 ⇒ 诊断快照与产品 evidence 路径不同源，全件作废"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(items),
        "no_contradictions": all(summary["contradictions"] == 0 for summary in summaries),
        "all_chains_identical": all(summary["chain_mismatches"] == 0 for summary in summaries),
        "replays_reproduce_miss": all(
            row.get("reproduces_recorded_miss", True) for arm in arms for row in arm["rows"]
        ),
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "items": len(items),
                "summaries": [
                    {
                        key: summary[key]
                        for key in (
                            "class_counts",
                            "primary_class",
                            "attribution",
                            "lock_picked_target_event",
                        )
                    }
                    for summary in summaries
                ],
                "verdict": verdict,
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
