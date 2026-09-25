"""A2.3 召回条件发射训练：让模型自己学会「何时抄、抄哪条」。

预注册：``plans/reference/M5_R2_A2_3_PREREG_20260925.md``（学习规则 §2、语料 §3、
预算与止损 §4 全部照件执行；本文件里可被触发的拒绝路径不是注释）。

阶段：
* ``--stage smoke``＝小模型 (64,48) 全新随机权重，预算 ≤20 分钟，判据＝寻址 top-1 ≥50%
  且 mean|gate| > 0.05；
* ``--stage judge``＝base_16M（seed_beta.pt，只读，跑后 sha256 复核），预算 ≤90 分钟。

纪律：S0 checkpoint 存取自检先行（训练前必查）；写靶只在 ``--out-dir``；stop 文件优雅
停机；判决读数落 ``reports/taiji_r2_copy_circuit_training[_smoke]_<date>.json``，不覆写。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

PREREG = "plans/reference/M5_R2_A2_3_PREREG_20260925.md"

# 训练实体表：与 CAP D/E 评测实体（阿岩/杭州/17/Seed）严格不相交（预注册 §3 冻结）。
NAMES = ("阿蒙", "晓雨", "子昂", "青禾", "若谷", "拾一", "云舟", "浣溪")
CITIES = ("成都", "厦门", "敦煌", "丽江", "曲阜", "平遥", "荔波", "额济纳")
NUMBERS = ("42", "7", "2025", "13", "96", "518", "2048")
PROJECTS = ("星舟", "春潮", "山丘", "拾光", "浮梁", "长庚")
DISTRACTORS = ("水沸点是多少？", "3 乘 4 等于几？", "天空为什么是蓝的？")


def make_episode(rng: random.Random) -> tuple[list[str], str]:
    """一条「告知 →（干扰?）提问」episode 与答案关键词（预注册 §3 模板族）。"""
    base = rng.randrange(4)
    key = (rng.choice(NAMES), rng.choice(CITIES), rng.choice(NUMBERS), rng.choice(PROJECTS))[base]
    tells = (f"我叫{key}。", f"我住在{key}。", f"我最喜欢的数字是{key}。", f"我的项目叫{key}。")
    asks = ("我的名字是什么？", "我住哪？", "那个数字是多少？", "我的项目叫什么？")
    if rng.random() < 0.4:
        return [tells[base], rng.choice(DISTRACTORS), asks[base]], key
    return [tells[base], asks[base]], key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feed(substrate: Any, data: bytes) -> None:
    substrate.observe(
        int(substrate.config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    for symbol in data:
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "judge"), required=True)
    parser.add_argument(
        "--max-minutes", type=float, default=None, help="默认按预注册：smoke 20 / judge 90"
    )
    parser.add_argument(
        "--episodes", type=int, default=100_000, help="episode 硬上限（预算双保险）"
    )
    parser.add_argument("--lr-address", type=float, default=0.05)
    parser.add_argument("--lr-gate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--out-dir", default="output/taiji_r2_copy_circuit")
    parser.add_argument("--base-checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--resume", default=None, help="circuit payload .pt 续跑")
    parser.add_argument("--stop-file", default=None)
    args = parser.parse_args()

    max_minutes = (
        args.max_minutes
        if args.max_minutes is not None
        else (20.0 if args.stage == "smoke" else 90.0)
    )
    out_dir = PROJECT_ROOT / args.out_dir / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_file = PROJECT_ROOT / args.stop_file if args.stop_file else out_dir / "STOP"
    base_path = PROJECT_ROOT / args.base_checkpoint
    base_sha = _sha256(base_path) if args.stage == "judge" and base_path.exists() else None

    from taiji import Taiji, TaijiConfig

    if args.stage == "smoke":
        substrate = Taiji(
            TaijiConfig(region_sizes=(64, 48), synapse_fan_in=16, motor_fan_in=48, seed=args.seed)
        )
    else:
        from api.seed_runtime import SeedRuntime

        substrate = SeedRuntime.load(base_path).model.substrate

    # ---- S0：checkpoint 存取自检（训练前必查，预注册 §4）----
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    circuit = substrate.copy_circuit
    assert circuit is not None
    circuit.store.record(b"\xe6\xb5\x8b", torch.ones(substrate.config.cortical_context_dim))
    probe_digest = hashlib.sha256(
        json.dumps(
            {k: v.tolist() for k, v in circuit.parameters().items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    probe_payload = circuit.to_payload()
    twin = Taiji(substrate.config)
    twin.mount_copy_circuit(max_events=4)
    twin.copy_circuit.load_payload(probe_payload)
    twin_digest = hashlib.sha256(
        json.dumps(
            {k: v.tolist() for k, v in twin.copy_circuit.parameters().items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    if probe_digest != twin_digest:
        raise RuntimeError("S0 self-check failed: circuit checkpoint round-trip digest mismatch")
    circuit.store._events.clear()  # 自检事件不进训练

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        circuit.load_payload(torch.load(resume_path, weights_only=False)["copy_circuit"])
        print(json.dumps({"resumed": str(resume_path)}), flush=True)

    rng = random.Random(args.seed)
    config = substrate.config
    started = time.monotonic()
    done = 0
    window_steps = window_hits = 0
    window_gate: list[float] = []
    window_adv: list[float] = []
    history: list[dict[str, Any]] = []
    hit_rates: list[float] = []
    gate_means: list[float] = []
    last_checkpoint_at = 0

    def checkpoint(tag: str) -> None:
        payload = {
            "copy_circuit": circuit.to_payload(),
            "stage": args.stage,
            "episodes_done": done,
            "prereg": PREREG,
        }
        torch.save(payload, out_dir / f"circuit-{tag}.pt")

    while done < args.episodes:
        if (time.monotonic() - started) / 60.0 >= max_minutes:
            break
        if stop_file.exists():
            break
        turns, answer = make_episode(rng)
        substrate.reset_dynamics(episode_id=f"a23-{args.stage}-{done}")
        _feed(substrate, turns[0].encode("utf-8"))
        circuit.store.record(
            turns[0].encode("utf-8"),
            substrate.fabric.cortical_context(substrate._state.regions).detach().cpu().clone(),
        )
        _feed(substrate, "".join(turns[1:]).encode("utf-8"))
        emitted: dict[int, int] = {}
        for byte in answer.encode("utf-8"):
            state = substrate._state
            cue = substrate.fabric.cortical_context(state.regions)
            ctx = state.motor_context
            snap = circuit.addressing(cue=cue.detach().cpu().clone(), f1_context=ctx)
            if snap is None:
                break
            top1 = int(snap["scores"].argmax())
            codes = snap["codes"]
            matches = [i for i, c in enumerate(codes.tolist()) if c == byte]
            want = matches[min(emitted.get(byte, 0), len(matches) - 1)] if matches else -1
            window_hits += int(want >= 0 and top1 == want)
            window_steps += 1
            base_evidence = float(
                config.consolidation_read_gain
            ) * substrate.fabric.consolidated_decode(0, state.regions[0].trace)
            readout = substrate.predictive_readout
            # rev2（预注册 §2）：反事实固定为"全开"——gate 零初始化下"当前开度"基线恒零锁死。
            p_full = readout.probabilities(
                ctx,
                episodic_evidence=base_evidence + snap["copy_distribution"],
            )
            p_without = readout.probabilities(ctx, episodic_evidence=base_evidence)
            advantage = math.log(max(float(p_full[byte]), 1e-12)) - math.log(
                max(float(p_without[byte]), 1e-12)
            )
            advantage = max(-2.0, min(2.0, advantage))
            if want >= 0:
                circuit.learn(
                    snap,
                    f1_context=ctx,
                    target_position=want,
                    advantage=advantage,
                    lr_address=args.lr_address,
                    lr_gate=args.lr_gate,
                )
            window_gate.append(
                abs(
                    circuit.addressing(cue=cue.detach().cpu().clone(), f1_context=ctx)["gate_value"]
                )
            )
            window_adv.append(advantage)
            emitted[byte] = emitted.get(byte, 0) + 1
            substrate.observe(int(byte), learn=False, readout="predictive", use_memory=False)
        done += 1
        if done % 50 == 0:
            hit_rate = window_hits / max(window_steps, 1)
            gate_mean = sum(window_gate) / max(len(window_gate), 1)
            history.append(
                {
                    "episodes": done,
                    "minutes": round((time.monotonic() - started) / 60.0, 2),
                    "address_top1_recent50": round(hit_rate, 4),
                    "gate_abs_mean_recent50": round(gate_mean, 4),
                    "advantage_mean_recent50": round(sum(window_adv) / max(len(window_adv), 1), 4),
                }
            )
            hit_rates.append(hit_rate)
            gate_means.append(gate_mean)
            print(json.dumps(history[-1]), flush=True)
            window_steps = window_hits = 0
            window_gate, window_adv = [], []
        if done - last_checkpoint_at >= args.checkpoint_every:
            checkpoint(f"ep{done}")
            last_checkpoint_at = done

    checkpoint("final")
    minutes = (time.monotonic() - started) / 60.0
    verdict_pass = bool(hit_rates and hit_rates[-1] >= 0.5 and gate_means and gate_means[-1] > 0.05)
    report = {
        "format": "taiji-r2-copy-circuit-training-v1",
        "prereg": PREREG,
        "stage": args.stage,
        "episodes_done": done,
        "minutes": round(minutes, 2),
        "budget_minutes": max_minutes,
        "stopped_by": (
            "budget"
            if minutes >= max_minutes * 0.98
            else (
                "stop-file"
                if stop_file.exists()
                else "episode-cap" if done >= args.episodes else "completed-loop"
            )
        ),
        "lr_address": args.lr_address,
        "lr_gate": args.lr_gate,
        "seed": args.seed,
        "base_checkpoint": str(base_path.relative_to(PROJECT_ROOT)) if base_sha else None,
        "history": history,
        "smoke_criteria": {
            "address_top1_ge_0.5": bool(hit_rates and hit_rates[-1] >= 0.5),
            "gate_abs_mean_gt_0.05": bool(gate_means and gate_means[-1] > 0.05),
            "pass": verdict_pass,
        },
    }
    if base_sha is not None:
        report["base_sha256_unchanged"] = _sha256(base_path) == base_sha
    out = (
        PROJECT_ROOT
        / "reports"
        / (
            f"taiji_r2_copy_circuit_training_{'smoke_' if args.stage == 'smoke' else ''}20260925.json"
        )
    )
    if out.exists():
        out = out.with_name(out.stem + f"-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "criteria_pass": verdict_pass}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
