"""G5/A0 四格：给语言轮补情节写入口 → 开 use_memory → 量四格（写/不写 × 读/不读）。

预注册：`PLAN-A-24`（M5_R2_A_BRANCH_PLAN_REV2_20260926.md）§4 乙-1／§5 A0／§5b G5 行。
主判据＝v3 严格真命中 /104（`r2_copy_surface_extension_v3_position_random.json`）。
判据（§5 A0 原文）：「写＋读」相对「不写」抬升 ⇒ 断路＝写入口，路线乙成立且便宜；
四格全同 ⇒ 读通路对语言内容是死的（action_evidence 对语言内容无响应），乙在开案前被否。

设计决定（逐条披露）：
* **不挂复制电路**。A0 打架构层；旁路在场会把架构层贡献混进旁路读数（§1b 分层不混）。
  代价：本件读数与 v3 主判据的历史读数（挂电路 17–21）不同链，只与对照 0 同链可比。
* 写入口在**探针内**显式实现，零产品码改动（G5 进产品默认属所有者裁定，不在本件）：
  - 时机＝告知轮文本喂完、其答复生成之前（复刻 `pending_experience` 的 settle 时点语义：
    `settle_action` 存的 `cortical_context` 就是 `fabric.cortical_context(regions)`，model.py:2313）；
  - 内容＝`memory.write(cortical_context, action_symbol=49, reward=1.0, outcome_symbol=边界符)`
    ——action_symbol 取定值（语言轮没有动作，该符号不承载内容）、reward 恒 1.0
    （写入显著性手工给定；「后果」语义＝3b-6，需所有者裁定，探针不装）；
  - 手段＝直接调产品写入原语 `Taiji.memory.write`，不重抄其内部；
  - 条目间 `to_payload`/`load_payload` 往返清场（防跨条目串扰；写计数随之归零）。
* 读开关＝`Taiji.generate(use_memory=...)`（产品生成原语的显式参数，默认 False＝opt-in，
  model.py:2896）。`generate_input` 不暴露该参数，故四臂统一绕过 InputFrame 校验直接调
  `generate`——链路其余逐字相同（`_serialize` 铺文本、`_TURN_MARKERS` 折字）。
* 每臂独立 `SeedRuntime.load`（评估链全程 learn=False，权重不动 ⇒ 臂间唯一差别＝两个开关）。

判读纪律：差距按「两电路同向且 ≥3」的分辨率线执行——本件无电路，改为**同臂复跑一次**
（独立载入）确认方向稳定后才判；|gap|<3 一律 `not_resolved`（≠无差别）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
MAX_ANSWER_BYTES = 64
WRITE_ACTION_SYMBOL = 49  # ord("1")：语言轮无动作，取定值并在读数里如实披露
REWARD = 1.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_items(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = list(payload["dimensions"]["X"]["items"])
    return items if limit is None else items[:limit]


def _answer(runtime: Any, prompt: str, history: list[tuple[str, str]], *, use_memory: bool) -> str:
    """与 v3 仪器同链的答复获取（唯一差别＝use_memory 显式可调）。"""
    from api.seed_runtime import _TURN_MARKERS

    text = runtime._serialize(prompt, history)
    raw = runtime.model.substrate.generate(
        text.encode("utf-8"),
        MAX_ANSWER_BYTES,
        stop_at_boundary=True,
        sample=False,
        use_memory=use_memory,
    )
    answer = raw.decode("utf-8", errors="replace")
    for marker in _TURN_MARKERS:
        index = answer.find(marker)
        if index >= 0:
            answer = answer[:index]
    return answer.strip()


def _feed_for_write_cue(runtime: Any, text: str) -> None:
    """把"到本告知轮为止"的对话铺进皮质态（learn=False 纯感觉，不解码、不学习），
    让写入时的 cortical_context 精确落在该告知轮末尾。"""
    substrate = runtime.model.substrate
    substrate.reset_dynamics(episode_id="g5-write-cue")
    substrate.observe(
        substrate.config.boundary_symbol, learn=False, readout="predictive", use_memory=False
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def _write_told_turn(runtime: Any, episode_id: str) -> dict[str, Any]:
    substrate = runtime.model.substrate
    state = substrate._state
    cortical = substrate.fabric.cortical_context(state.regions).detach().clone()
    write = substrate.memory.write(
        cortical,
        action_symbol=WRITE_ACTION_SYMBOL,
        reward=REWARD,
        outcome_symbol=substrate.config.boundary_symbol,
        tick=int(state.tick),
        episode_id=episode_id,
        provenance="experienced",
        threshold=state.memory.threshold,
    )
    return {"strength": round(float(write.strength), 6), "write_count": substrate.memory.write_count}


def run_arm(
    items: list[dict[str, Any]], checkpoint: Path, *, write_turns: bool, use_memory: bool
) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    pristine_memory = substrate.memory.to_payload()

    hits = 0
    rows: list[dict[str, Any]] = []
    write_strengths: list[float] = []
    for item in items:
        if write_turns:
            substrate.memory.load_payload(pristine_memory)
        history: list[tuple[str, str]] = []
        turns = [str(turn) for turn in item["turns"]]
        answer = ""
        for index, turn in enumerate(turns):
            if write_turns and index + 1 < len(turns):
                _feed_for_write_cue(runtime, runtime._serialize(turn, history))
                record = _write_told_turn(runtime, f"g5:{item['id']}:{index}")
                write_strengths.append(record["strength"])
            answer = _answer(runtime, turn, history, use_memory=use_memory)
            if index + 1 < len(turns):
                history.append((turn, answer))
        tokens = [str(token) for token in item["expected_contains"]]
        hit = any(token in answer for token in tokens)
        hits += int(hit)
        rows.append({"id": item["id"], "hit": hit, "answer": answer[:60]})
    return {
        "write_turns": write_turns,
        "use_memory": use_memory,
        "items": len(rows),
        "strict_hits": hits,
        "episodes_written": len(write_strengths),
        "write_strength_mean": (
            round(sum(write_strengths) / max(len(write_strengths), 1), 6)
            if write_strengths
            else None
        ),
        "final_write_count": substrate.memory.write_count,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--limit", type=int, default=None, help="smoke 用；缺省全量 104")
    parser.add_argument("--arms", default="W0R0,W0R1,W1R0,W1R1", help="逗号分隔的四格子集")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    items = _load_items(
        Path(args.manifest) if Path(args.manifest).is_absolute() else PROJECT_ROOT / args.manifest,
        args.limit,
    )
    if not items:
        print(json.dumps({"guard_ok": False, "error": "no items loaded"}, ensure_ascii=False))
        return 2

    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]
    arm_specs = {
        "W0R0": dict(write_turns=False, use_memory=False),
        "W0R1": dict(write_turns=False, use_memory=True),
        "W1R0": dict(write_turns=True, use_memory=False),
        "W1R1": dict(write_turns=True, use_memory=True),
    }
    arms: dict[str, dict[str, Any]] = {}
    for name in wanted:
        if name not in arm_specs:
            print(json.dumps({"guard_ok": False, "error": f"unknown arm {name}"}, ensure_ascii=False))
            return 2
        arms[name] = run_arm(items, checkpoint, **arm_specs[name])

    baseline = arms.get("W0R0")
    gaps = {
        name: (
            None
            if baseline is None or name == "W0R0"
            else arm["strict_hits"] - baseline["strict_hits"]
        )
        for name, arm in arms.items()
    }
    report = {
        "format": "taiji-g5-episodic-write-four-grid-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §4 乙-1／§5 A0／§5b G5",
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "items": len(items),
        "write_entrance": {
            "action_symbol": WRITE_ACTION_SYMBOL,
            "reward": REWARD,
            "timing": "told-turn-end cortical context, before that turn's answer",
            "clear_between_items": "memory payload round-trip",
            "read_switch": "Taiji.generate(use_memory=...)",
        },
        "arms": {name: {key: value for key, value in arm.items() if key != "rows"} for name, arm in arms.items()},
        "gaps_vs_W0R0": gaps,
        "rows": {name: arm["rows"] for name, arm in arms.items()},
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(items),
        "w0r0_is_zero_or_reported": (
            baseline is None or baseline["strict_hits"] == 0 or "w0r0_nonzero" in report
        ),
        "write_arms_actually_wrote": all(
            arms[name]["episodes_written"] > 0 for name in arms if name.startswith("W1")
        ),
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "items": len(items),
                "strict_hits": {name: arm["strict_hits"] for name, arm in arms.items()},
                "episodes_written": {
                    name: arm["episodes_written"] for name, arm in arms.items()
                },
                "gaps": gaps,
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
