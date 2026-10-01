"""产品基座"增长普查"：16M tick 到底把**哪些参数面**推动了、哪些一动没动。

用户的问题（2026-09-26）："这个 16M 的规模到底是增长的什么"。
先钉住口径：`metadata.tick = 16000000` 是**训练步/符号数**，不是参数量——
参数量由架构与 `config` 定死，训练不改形状（唯一会改形状的是 identity organ 的显式扩容，
它有独立的 growth history）。所以"增长了什么"必须答成三件事：

1. **哪些面动了**：与"同 config 同种子新建的初始态"逐面对比（初始态是可复现的，
   本仓有位级不变的钉子测试为证）；
2. **哪些面仍是零**：全零面＝从未通电（`SPEC-A-22` §23 已点名查到三个巩固解码头，
   本件做**全量**普查，不再点名）；
3. **规模本身有没有长**：`structural_events`／`development_ticks`／identity growth history／
   情节库计数——这些是"结构性增长"的账，与"权重被训练推动"是两回事。

纪律：只读；不写 `checkpoints/`；不改 config（`TaijiConfig` 是 frozen dataclass）；
基座 sha256 跑前后逐位相同；普查面数必须非零，否则退出码 2（仪器不许静默交空表）。

P1 增补（2026-09-26，`PLAN-A-24` §5b 队首）：§2d 发现"存盘全零 vs 运行时非零"的口径冲突后，
本件加 `--mode {file,runtime,both}` 做**三态对表**（检查点文件／载入后运行时／同种子新建），
并对每个存盘面标注差异落在 restore 链的哪一句（`taiji/model.py` 的 `restore`）。
另有**载入后活动阶梯**（默认跑，`--skip-activity` 跳过）：载入干净态 → observe(learn=True)×3 →
产品 chat(learn=True) → act/settle_action 写入路径，逐步重读全零面的范数——
用于裁决"§2d 的非零读数"到底是载入重建（坏）还是写入/学习步之后的即时态（口径错位）。
活动阶梯只改**进程内**状态，不落盘；基座 sha 守卫覆盖文件不被触碰。
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk(node: Any, prefix: str, out: dict[str, torch.Tensor]) -> None:
    """递归收集 payload 里所有张量，键＝点路径。不点名 ⇒ 不会因为漏写路径而少报面。"""
    if torch.is_tensor(node):
        out[prefix] = node.detach().cpu().float()
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk(value, f"{prefix}.{key}" if prefix else str(key), out)
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _walk(value, f"{prefix}[{index}]", out)


#: 存盘面 → restore 链上的落点（`Taiji.restore`，`taiji/model.py:3240` 起）。
#: 只对文件里实际存在的段负责；文件缺段时该面属于"不在存盘里"，不进这张表。
_RESTORE_STEP_BY_SECTION = {
    "config": "restore 配置校验（model.py:3248，不改参数面）",
    "fabric": "model.py:3252 fabric.load_payload",
    "motor": "model.py:3253 motor.load_payload",
    "memory": "model.py:3420 memory.load_payload",
    "state": "model.py:3494 附近 TaijiState.from_payload",
    "rng_state": "model.py:3579 rng 恢复",
}


def _restore_step(path: str) -> str:
    section = path.split(".")[1] if path.count(".") else path
    return _RESTORE_STEP_BY_SECTION.get(section, "restore 链未点名（面不在已知段内）")


def _activity_ladder(runtime: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """载入后活动阶梯：裁决"§2d 的非零读数"出现在哪个生命周期时点。

    只走产品链、不重抄喂法：`observe(learn=True)` 是训练喂法（无动作 ⇒ 无
    `pending_experience`）、`chat(learn=True)` 是产品语言轮（写入门仍不开）、
    observe→act→settle_action 是产品动作路径（`pending_experience` 的唯一来源，
    写发生在 settle 后的下一次 observe，`model.py:1868-1882`）。
    每步后重读"存盘时全零"面的范数与 `write_count`。只改进程内状态，不落盘。
    """
    substrate = runtime.model.substrate
    zero_paths = sorted(row["path"] for row in rows if row["all_zero"])
    steps: list[dict[str, Any]] = []

    def snapshot(step: str, note: str) -> None:
        tensors: dict[str, torch.Tensor] = {}
        _walk(substrate.checkpoint(), "substrate", tensors)
        lit = {
            path: round(float(tensors[path].norm()), 6)
            for path in zero_paths
            if tensors.get(path) is not None and float(tensors[path].abs().sum()) > 0.0
        }
        steps.append(
            {
                "step": step,
                "note": note,
                "memory_write_count": getattr(substrate.memory, "write_count", None),
                "zero_surfaces_total": len(zero_paths),
                "zero_surfaces_still_zero": len(zero_paths) - len(lit),
                "zero_surfaces_lit_norms": lit,
            }
        )

    snapshot("after_load_clean", "载入后、任何活动步之前（三态对表的运行时态即取于此）")

    substrate.reset_dynamics(episode_id="census-ladder-observe")
    for symbol in "我叫阿蒙".encode()[:4]:
        substrate.observe(int(symbol), learn=True, readout="predictive")
    snapshot(
        "observe_learn_true_x4", "训练喂法：observe(learn=True)，无动作 ⇒ 无 pending_experience"
    )

    substrate.reset_dynamics(episode_id="census-ladder-chat")
    runtime.chat("我叫阿蒙。", history=[], max_length=32, learn=True, repetition_penalty=0.0)
    snapshot(
        "chat_learn_true", "产品语言轮 chat(learn=True)（接线审计同款：预期 write_count 不动）"
    )

    substrate.reset_dynamics(episode_id="census-ladder-write")
    substrate.observe(256, learn=False)
    substrate.act((ord("1"),), sample=False)
    substrate.settle_action(1.0, learn=True, learn_memory=True, provenance="experienced")
    substrate.observe(ord("+"), learn=False)
    snapshot(
        "act_settle_write",
        "产品动作路径：observe→act→settle_action→observe ⇒ pending_experience 落一次写",
    )

    lit_after_write = steps[-1]["zero_surfaces_lit_norms"]
    return {
        "zero_surfaces_tracked": len(zero_paths),
        "steps": steps,
        "verdict": (
            "nonzero_readings_are_post_write_state"
            if lit_after_write
            else "readouts_stay_zero_even_after_write"
        ),
        "what_would_overturn": (
            "若 after_load_clean 步即出现非零面 ⇒ 载入重建成立，P1 判 load_differs；"
            "若 act_settle_write 后仍全零 ⇒ §2d 的非零读数另有来源（须复跑其原仪器）"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--out-report", required=True)
    parser.add_argument(
        "--mode",
        choices=("file", "runtime", "both"),
        default="both",
        help="file=只查检查点文件；runtime=只查载入后运行时；both=三态对表（P1 默认）",
    )
    parser.add_argument(
        "--skip-activity",
        action="store_true",
        help="跳过载入后活动阶梯（定位 §2d 非零读数的生命周期时点）",
    )
    args = parser.parse_args()

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    metadata = payload.get("metadata", {})

    substrate_tensors: dict[str, torch.Tensor] = {}
    if args.mode in ("file", "both"):
        _walk(payload.get("substrate", {}), "substrate", substrate_tensors)

    #: 载入后、任何活动步之前的干净运行时态：用仪器自己的 `checkpoint()` 序列化
    #: （与文件同构、同点路径），不摸私有属性、不重抄序列化链。
    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    runtime_tensors: dict[str, torch.Tensor] = {}
    if args.mode in ("runtime", "both"):
        _walk(substrate.checkpoint(), "substrate", runtime_tensors)

    #: 同 config 的初始态（不喂任何数据）：这就是"16M tick 之前"的参照。
    from taiji import Taiji, TaijiConfig

    config_payload = payload["substrate"].get("config") or payload.get("config")
    config = (
        config_payload
        if isinstance(config_payload, TaijiConfig)
        else (
            TaijiConfig.from_payload(config_payload)
            if hasattr(TaijiConfig, "from_payload")
            else TaijiConfig(**config_payload)
        )
    )
    twin = Taiji(config)
    twin_tensors: dict[str, torch.Tensor] = {}
    #: 用 `checkpoint()`（`model.py:3210`）而不是 `to_payload()`——Taiji 没有后者；
    #: 检查点文件里 `substrate` 那一段就是它产出的同构字典，所以两边点路径可比。
    _walk(twin.checkpoint(), "substrate", twin_tensors)

    #: rows 锚定面集：file/both 以存盘面为分母（§2e 容量账口径不变）；
    #: runtime 单独跑时以运行时面为分母（file 列在缺失处为 None，如实留空不硬凑）。
    anchor = runtime_tensors if args.mode == "runtime" else substrate_tensors
    rows = []
    for path, tensor in sorted(anchor.items()):
        init = twin_tensors.get(path)
        file_t = substrate_tensors.get(path)
        loaded = runtime_tensors.get(path)
        norm = float(tensor.norm())
        identical_to_init = (
            None if init is None or init.shape != tensor.shape else bool(torch.equal(tensor, init))
        )
        if args.mode == "runtime":
            #: 锚＝运行时面，对照＝存盘同路径面（形状不兼容时如实记 None）。
            other = file_t
        else:
            #: 锚＝存盘面，对照＝载入后运行时同路径面。
            other = loaded
        comparable = other is not None and other.shape == tensor.shape
        identical_other = None if not comparable else bool(torch.equal(tensor, other))
        rows.append(
            {
                "path": path,
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "norm": round(norm, 6),
                "abs_max": round(float(tensor.abs().max()), 6) if tensor.numel() else 0.0,
                "all_zero": bool(tensor.numel()) and float(tensor.abs().sum()) == 0.0,
                "identical_to_init": identical_to_init,
                "delta_norm_vs_init": (
                    None
                    if init is None or init.shape != tensor.shape
                    else round(float((tensor - init).norm()), 6)
                ),
                "runtime_identical_to_file": identical_other,
                "runtime_norm": round(float(loaded.norm()), 6) if loaded is not None else None,
                "restore_step": _restore_step(path) if identical_other is False else None,
            }
        )

    structural = {
        "tick_metadata": metadata.get("tick"),
        "saved_at_utc": metadata.get("saved_at_utc"),
        "corpus_fingerprint": metadata.get("corpus_fingerprint"),
        "profile": metadata.get("profile"),
        "state_tick": int(getattr(substrate._state, "tick", -1)),
        "development_ticks": int(getattr(substrate, "_development_ticks", -1)),
        "fabric_structural_events": int(getattr(substrate.fabric, "structural_events", -1)),
        "memory_write_count": getattr(substrate.memory, "write_count", None),
        "identity_growth_history_len": len(
            getattr(substrate, "_identity_growth_history", ()) or ()
        ),
        "parameter_count_active": int(substrate.parameter_count(active_only=True)),
        "parameter_count_all": int(substrate.parameter_count(active_only=False)),
    }

    moved = [row for row in rows if row["delta_norm_vs_init"] not in (None, 0.0)]
    if args.mode == "both":
        load_diffs = [row for row in rows if row["runtime_identical_to_file"] is False]
        uncovered = [row["path"] for row in rows if row["runtime_identical_to_file"] is None]
        runtime_only = sorted(set(runtime_tensors) - set(substrate_tensors))
    else:
        load_diffs, uncovered, runtime_only = [], [], []
    report = {
        "format": "taiji-substrate-growth-census-v2",
        "prereg": (
            "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §5 A3（原 v1 口径）"
            "＋§5b P1（三态对表与活动阶梯）"
        ),
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "structural": structural,
        "summary": {
            "surfaces_enumerated": len(rows),
            "surfaces_all_zero": sum(1 for row in rows if row["all_zero"]),
            "surfaces_moved_vs_init": len(moved),
            "surfaces_identical_to_init": sum(
                1 for row in rows if row["identical_to_init"] is True
            ),
            "surfaces_not_in_twin": sum(1 for row in rows if row["identical_to_init"] is None),
            "total_numel": sum(row["numel"] for row in rows),
            "moved_numel": sum(row["numel"] for row in moved),
            "runtime_surfaces_enumerated": len(runtime_tensors),
            "surfaces_runtime_differs_from_file": len(load_diffs),
            "surfaces_missing_in_runtime": len(uncovered),
            "runtime_only_surfaces": len(runtime_only),
        },
        "p1_verdict": (
            "not_applicable_mode_without_file_or_runtime"
            if args.mode != "both"
            else ("three_state_identical" if not load_diffs and not uncovered else "load_differs")
        ),
        "p1_load_diff_rows": [
            {key: row[key] for key in ("path", "norm", "runtime_norm", "restore_step")}
            for row in load_diffs[:40]
        ],
        "p1_missing_in_runtime": uncovered[:40],
        "runtime_only_sections": sorted(
            {path.split(".")[1] for path in runtime_only if path.count(".")}
        ),
        "all_zero_surfaces": [row["path"] for row in rows if row["all_zero"]],
        "top_moved": sorted(moved, key=lambda row: -row["delta_norm_vs_init"])[:20],
        "rows": rows,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }

    if args.mode in ("runtime", "both") and not args.skip_activity:
        report["activity_ladder"] = _activity_ladder(runtime, rows)

    report["instrument_guard"] = {
        "surfaces_nonzero": report["summary"]["surfaces_enumerated"] > 0,
        "twin_covers_checkpoint": report["summary"]["surfaces_not_in_twin"] == 0,
        "base_unchanged": report["base_sha256_unchanged"],
        "runtime_covers_file": args.mode != "both" or not uncovered,
        "activity_ladder_ran": (
            args.mode not in ("runtime", "both")
            or args.skip_activity
            or "activity_ladder" in report
        ),
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
                "mode": args.mode,
                "p1_verdict": report["p1_verdict"],
                "p1_load_diffs": report["summary"]["surfaces_runtime_differs_from_file"],
                "p1_missing_in_runtime": report["summary"]["surfaces_missing_in_runtime"],
                "activity_verdict": (
                    report.get("activity_ladder", {}).get("verdict")
                    if isinstance(report.get("activity_ladder"), dict)
                    else None
                ),
                "summary": report["summary"],
                "structural": {
                    key: structural[key]
                    for key in (
                        "tick_metadata",
                        "development_ticks",
                        "fabric_structural_events",
                        "memory_write_count",
                        "parameter_count_active",
                    )
                },
                "all_zero": report["all_zero_surfaces"][:14],
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
