"""Taiji R2 受控重训语言读出：步骤 0 成本标定仪器（只读，不落任何 checkpoint）。

所属合同草案：``plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md`` §4。
那里只授权一个先行动作——"跑一次几十 tick 的试标定，测 wall/tick 与峰值内存"，
本脚本就是那件事，且只做那件事。它回答三个问题：

1. **每 tick 墙钟**。三条臂分别是——
   ``A`` 只训读出头（``readout="predictive"``，只有 ``predictive_readout`` 学）；
   ``B`` 只训运动面（``readout="action"``，只有 ``motor`` 学）；
   ``C`` 两处都冻结（``learn=False``，用来隔离"权重没变时的漂移"）。
2. **峰值工作集**（Windows ``GetProcessMemoryInfo``）。取不到就记 ``null`` 并标
   ``unavailable``，不拿别的量冒充。
3. **每条臂真的只动了它该动的那一处**——按 ``motor`` / ``predictive_readout`` 的
   载荷指纹前后差判定。这一条是"写进代码的预期值"的**可否决通路**：若 A 动了 motor、
   或 C 动了任何一处，本件判仪器不可信并**非零退出**，那批秒数不得写进预算。

它**不是**训练：不落 checkpoint、不读封存评价集、不改产品默认、不改任何已冻结阈值，
并把"跑完之后基座文件的 sha256 必须逐位不变"写成硬断言（不成立即抛错，报告不落盘）。

两条如实登记的边界（不粉饰）：

* ``learn`` 这个开关同时管着感知面的学习（``taiji/adapter.py`` 的
  ``perception.observe(..., learn=...)``），而当前架构没有单独的感知冻结开关。
  因此 A/B 两臂是"感知也学"，C 臂是"感知也不学"。A 与 B 之间的对比不受影响，
  A 与 C 之间的差里混着这一项——这正是 C 臂作为**漂移对照**而不是"严格同前向"的原因。
* 本件只测成本，**不产生任何能力读数**，也不构成对合同 §5 数值线的认可。
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import hashlib
import json
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from train_seed_corpus import iter_corpus_symbols  # noqa: E402

from seed import Seed  # noqa: E402
from taiji import content_digest  # noqa: E402

#: 硬上限：本仪器的身份是"标定"而不是"训练"。超过它就不再是几十 tick 的试标定，
#: 而被合同授权的是试标定，所以这里直接拒跑，不给"反正没人管"留口子。
CALIBRATION_TICK_CAP = 200

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "p3b_all_fresh.jsonl"
#: 这份清单里写着它自己的血缘推导（``skip_derivation`` 与 ``skip_symbols``），
#: 也就是"该副本已经吃过的前缀"是怎么算出来的——不许手抄。
DEFAULT_LINEAGE_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "p3b_all_fresh_manifest.json"

#: 三条臂的观察面开关。``write_surface`` 是**预期**写入面，跑完要核；
#: ``("motor",)`` 表示只有 motor 的指纹允许变，以此类推。
ARMS: dict[str, dict[str, Any]] = {
    "A": {
        "description": "只训读出头：运动面冻结，predictive_readout 单点写入",
        "observe_kwargs": {
            "learn": True,
            "readout": "predictive",
            "learn_motor": False,
            "learn_fabric": False,
            "learn_predictive_context": False,
            "learn_predictive_readout": True,
        },
        "write_surface": ("predictive_readout",),
    },
    "B": {
        "description": "只训运动面：原运动面路径，读出冻结",
        "observe_kwargs": {
            "learn": True,
            "readout": "action",
            "learn_motor": True,
            "learn_predictive_readout": False,
        },
        "write_surface": ("motor",),
    },
    "C": {
        "description": "双冻结漂移对照：符号流过，两处都不学",
        "observe_kwargs": {"learn": False, "readout": "action"},
        "write_surface": (),
    },
}

_SURFACES = ("motor", "predictive_readout")


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    )


def peak_working_set_bytes() -> int | None:
    """Peak working set of this process, or ``None`` when it cannot be read.

    ``None`` is a first-class result: the report marks the field ``unavailable`` rather than
    substituting another quantity that happens to be easy to obtain.
    """

    if sys.platform != "win32":
        return None
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        )
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except Exception:  # pragma: no cover - platform/API dependent
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lineage_skip(manifest_path: Path, corpus_path: Path) -> dict[str, Any]:
    """Read the lineage-derived skip point for ``corpus_path`` out of its manifest.

    The point of going through the manifest is that the skip is *derived* from checkpoint
    lineage (``skip_derivation``) rather than typed by hand.  A corpus that is not the
    manifest's own output has no such derivation, and this function says so instead of
    guessing a number.
    """

    if not manifest_path.is_file():
        raise SystemExit(f"lineage manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = str(manifest.get("output", ""))
    if Path(recorded).name != corpus_path.name:
        raise SystemExit(
            f"{corpus_path.name} is not the output recorded in {manifest_path.name} "
            f"(it says {recorded!r}); there is no lineage-derived skip for it"
        )
    skip = manifest.get("skip_symbols")
    if not isinstance(skip, int) or skip < 0:
        raise SystemExit(f"{manifest_path.name} carries no usable skip_symbols")
    try:
        recorded_path = manifest_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # A manifest outside the repo is legitimate for a fixture; do not invent a relative path.
        recorded_path = str(manifest_path)
    return {
        "manifest": recorded_path,
        "skip_symbols": skip,
        "first_emitted_row": manifest.get("first_emitted_row"),
        "replay_symbols_from_seen_region": manifest.get("replay_symbols_from_seen_region"),
        "skip_derivation": manifest.get("skip_derivation"),
    }


def take_symbols(corpus_paths: Sequence[Path], skip_symbols: int, count: int) -> list[int]:
    """The next ``count`` symbols after the already-consumed prefix."""

    stream: Iterator[int] = iter_corpus_symbols(corpus_paths)
    for _ in range(skip_symbols):
        try:
            next(stream)
        except StopIteration as exc:  # pragma: no cover - short corpus
            raise SystemExit(
                f"corpus is shorter than the lineage skip ({skip_symbols} symbols)"
            ) from exc
    taken: list[int] = []
    for _ in range(count):
        try:
            taken.append(next(stream))
        except StopIteration as exc:  # pragma: no cover - short corpus
            raise SystemExit(f"corpus ran out after {len(taken)} of {count} symbols") from exc
    return taken


def run_arm(
    *,
    checkpoint: dict[str, Any],
    name: str,
    symbols: Sequence[int],
    warmup: int,
    device: str,
) -> dict[str, Any]:
    """Run one arm over ``symbols`` and report its cost plus its realised write surface."""

    spec = ARMS[name]
    model = Seed.from_checkpoint(checkpoint, device=device)
    substrate = model.architecture
    # ``readout`` 不允许在同一个 dynamics episode 内切换；每个臂是全新实例 + 全新 episode。
    substrate.reset_dynamics(episode_id=f"readout-retrain-step0-{name}")

    before = {
        surface: content_digest(getattr(substrate, surface).to_payload()) for surface in _SURFACES
    }

    for index in range(warmup):
        substrate.observe(symbols[index], **spec["observe_kwargs"])

    per_tick: list[float] = []
    for index in range(warmup, len(symbols)):
        started = time.perf_counter()
        substrate.observe(symbols[index], **spec["observe_kwargs"])
        per_tick.append(time.perf_counter() - started)

    after = {
        surface: content_digest(getattr(substrate, surface).to_payload()) for surface in _SURFACES
    }
    changed = {surface: before[surface] != after[surface] for surface in _SURFACES}
    expected = set(spec["write_surface"])
    realised = {surface for surface, moved in changed.items() if moved}
    wall = sum(per_tick)
    return {
        "description": spec["description"],
        "observe_kwargs": spec["observe_kwargs"],
        "expected_write_surface": sorted(expected),
        "realised_write_surface": sorted(realised),
        "write_surface_ok": realised == expected,
        "timed_ticks": len(per_tick),
        "wall_seconds": round(wall, 6),
        "seconds_per_tick": round(wall / len(per_tick), 9) if per_tick else None,
        "ticks_per_hour": round(3600.0 * len(per_tick) / wall, 3) if wall > 0 else None,
        "first_tick_seconds": round(per_tick[0], 9) if per_tick else None,
        "last_tick_seconds": round(per_tick[-1], 9) if per_tick else None,
        "digests_before": before,
        "digests_after": after,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="基底文件；缺省取 api.seed_runtime.DEFAULT_CHECKPOINT（产品默认入口服务的那一份）",
    )
    parser.add_argument("--corpus", nargs="+", default=[str(DEFAULT_CORPUS)])
    parser.add_argument("--lineage-manifest", default=str(DEFAULT_LINEAGE_MANIFEST))
    parser.add_argument(
        "--skip-symbols",
        type=int,
        default=None,
        help="覆盖血缘推导出的前缀跳过量；低于清单值需加 --i-accept-replaying-seen-symbols",
    )
    parser.add_argument("--ticks", type=int, default=30, help=f"计时 tick 数（上限 {CALIBRATION_TICK_CAP}）")
    parser.add_argument("--warmup-ticks", type=int, default=5)
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-report",
        default="reports/taiji_r2_readout_retrain_step0_calibration.json",
        help="报告落盘路径；已存在即拒跑（封存件不覆写）",
    )
    parser.add_argument("--i-accept-replaying-seen-symbols", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not 1 <= args.ticks <= CALIBRATION_TICK_CAP:
        parser.error(f"--ticks must be within 1..{CALIBRATION_TICK_CAP}")
    if not 0 <= args.warmup_ticks <= CALIBRATION_TICK_CAP:
        parser.error(f"--warmup-ticks must be within 0..{CALIBRATION_TICK_CAP}")

    arms = [name.strip().upper() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")

    # 报告不得写进 checkpoints/：那是产品权重目录，本件只读。
    report_path = Path(args.out_report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    checkpoint_dir = (PROJECT_ROOT / "checkpoints").resolve()
    if report_path.resolve().is_relative_to(checkpoint_dir):
        parser.error(f"--out-report must not live under {checkpoint_dir}")
    if report_path.exists():
        parser.error(f"{report_path} already exists; this instrument never overwrites a report")

    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else _default_checkpoint(parser)
    )
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint not found: {checkpoint_path}")

    corpus_paths = [
        Path(item) if Path(item).is_absolute() else PROJECT_ROOT / item for item in args.corpus
    ]
    for path in corpus_paths:
        if not path.is_file():
            parser.error(f"corpus not found: {path}")
    lineage = load_lineage_skip(Path(args.lineage_manifest), corpus_paths[0])
    skip_symbols = lineage["skip_symbols"] if args.skip_symbols is None else args.skip_symbols
    if skip_symbols < lineage["skip_symbols"] and not args.i_accept_replaying_seen_symbols:
        parser.error(
            f"--skip-symbols {skip_symbols} sits inside the seen prefix "
            f"({lineage['skip_symbols']}); pass --i-accept-replaying-seen-symbols to accept replay"
        )

    sha_before = sha256_of(checkpoint_path)
    envelope = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    symbols = take_symbols(corpus_paths, skip_symbols, args.ticks + args.warmup_ticks)

    arms_report: dict[str, Any] = {}
    for name in arms:
        arms_report[name] = run_arm(
            checkpoint=envelope,
            name=name,
            symbols=symbols,
            warmup=args.warmup_ticks,
            device=args.device,
        )

    sha_after = sha256_of(checkpoint_path)
    peak = peak_working_set_bytes()
    report = {
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "contract": "plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md",
        "step": "step-0 cost calibration (wall/tick + peak working set)",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha_before,
            "bytes": checkpoint_path.stat().st_size,
            "envelope_tick": envelope.get("metadata", {}).get("tick")
            if isinstance(envelope.get("metadata"), dict)
            else None,
        },
        "corpus": {
            "paths": [str(path) for path in corpus_paths],
            "lineage": lineage,
            "skip_symbols": skip_symbols,
            "replay_symbols_in_window": max(0, lineage["skip_symbols"] - skip_symbols),
        },
        "device": args.device,
        "warmup_ticks": args.warmup_ticks,
        "timed_ticks": args.ticks,
        "arms": arms_report,
        "peak_working_set_bytes": peak,
        "peak_working_set_mb": None if peak is None else round(peak / (1 << 20), 3),
        "peak_working_set_status": "unavailable" if peak is None else "measured",
        "checkpoint_sha256_after": sha_after,
        "checkpoint_unchanged": sha_after == sha_before,
    }

    bad = sorted(name for name, entry in arms_report.items() if not entry["write_surface_ok"])
    if bad:
        report["status"] = "failed"
        report["failure"] = f"write surface violated for arm(s) {bad}"
    if not report["checkpoint_unchanged"]:
        report["status"] = "failed"
        report["failure"] = "the substrate file changed during a read-only calibration"

    if report["status"] == "failed":
        # 不落盘：一份把"仪器不可信"写在外面的报告容易被下游当成可用读数引用。
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: entry["seconds_per_tick"] for name, entry in arms_report.items()}))
    print(f"report: {report_path}")
    return 0


def _default_checkpoint(parser: argparse.ArgumentParser) -> Path:
    from api.seed_runtime import DEFAULT_CHECKPOINT

    if not Path(DEFAULT_CHECKPOINT).is_file():
        parser.error(f"product default checkpoint not on disk: {DEFAULT_CHECKPOINT}")
    return Path(DEFAULT_CHECKPOINT)


if __name__ == "__main__":
    raise SystemExit(main())
