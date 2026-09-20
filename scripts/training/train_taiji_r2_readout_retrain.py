"""R2 受控重训语言读出：正式三臂 runner（A 读出头 / B 运动面 / C 双冻结）。

合同草案：``plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md``。
所有者 2026-09-20 批准 §5 数值线按草案冻结、§4 预算 N ≤ 16,000,000 符号/臂。
臂定义与语料血缘切片来自 ``readout_retrain_spec``（与标定仪器共用同一份）。

本 runner 只做"训练 + 落盘"，**不做评价**：合同 §5 的 M1（未见 dev 成句率）与 M2（CAP D+E）
由各自既有仪器在训练后执行，且 §7 第 3 件（是否新增未见题面集）尚未裁决 ⇒ 这里不碰。

硬约束（都是可被触发的拒绝路径，不是注释）：

* 每臂符号数不得超过 ``APPROVED_SYMBOL_CEILING``（16,000,000）；要超必须显式
  ``--i-accept-exceeding-approved-budget``。
* 写靶一律落在 ``--out-dir`` 下的隔离目录；拒绝写进 ``checkpoints/``，
  也拒绝把写靶设成基底文件本身。
* 跑完硬核基底文件 sha256 逐位未变（本件只读基底）。
* 三臂看的是**同一段符号流**（同 ``skip_symbols`` 起、同长度），否则配对作废。
* 每条臂跑完核**写入面**（``motor`` / ``predictive_readout`` 载荷指纹前后差）是否等于
  预期；不符即把报告标 ``failed`` 并以非零码退出。
* 断点续跑：臂目录里已有 ``checkpoint.pt`` 就自动从它继续（``--fresh`` 才重开）；
  已经跑满预算的臂拒绝重跑；``--fresh`` 时旧报告先归档不删除。

用法::

    python scripts/training/train_taiji_r2_readout_retrain.py --arms A,B,C --symbols 16000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from readout_retrain_spec import (  # noqa: E402
    APPROVED_SYMBOL_CEILING,
    ARMS,
    DEFAULT_CORPUS,
    DEFAULT_LINEAGE_MANIFEST,
    SURFACES,
    TRAINER_NAME,
    iter_corpus_window,
    load_lineage_skip,
    sha256_of,
)

from seed import Seed  # noqa: E402
from seed.persistence import atomic_save, attach_metadata, corpus_fingerprint  # noqa: E402
from taiji import content_digest  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "taiji_r2_readout_retrain"
CONTRACT = "plans/reference/M5_R2_READOUT_RETRAIN_CONTRACT_DRAFT_20260920.md"


def surface_digests(substrate: Any) -> dict[str, str]:
    """Fingerprints of the two weight surfaces this experiment may touch."""

    return {surface: content_digest(getattr(substrate, surface).to_payload()) for surface in SURFACES}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _envelope_metadata(envelope: dict[str, Any]) -> dict[str, Any]:
    metadata = envelope.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def run_arm(
    *,
    arm: str,
    symbols: int,
    checkpoint_every: int,
    progress_every: int,
    arm_dir: Path,
    base_checkpoint: Path,
    base_sha256: str,
    corpus_paths: Sequence[Path],
    corpus_digest: str,
    skip_symbols: int,
    device: str,
    fresh: bool,
) -> dict[str, Any]:
    spec = ARMS[arm]
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_dir / "checkpoint.pt"
    report_path = arm_dir / "run_report.json"
    progress_path = arm_dir / "progress.jsonl"

    base_envelope = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_tick = int(_envelope_metadata(base_envelope).get("tick", 0))

    resumed_from: str | None = None
    archived_report: str | None = None
    consumed = 0
    base_sources = [str(base_checkpoint)]
    if checkpoint_path.is_file() and not fresh:
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        extra = _envelope_metadata(previous)
        if extra.get("arm") != arm:
            raise SystemExit(f"{checkpoint_path} belongs to arm {extra.get('arm')!r}, not {arm!r}")
        if extra.get("base_checkpoint_sha256") != base_sha256:
            raise SystemExit(
                f"{checkpoint_path} was trained from a different substrate "
                f"({extra.get('base_checkpoint_sha256')!r} != {base_sha256!r})"
            )
        consumed = int(extra.get("symbols_consumed", 0))
        if consumed >= symbols:
            raise SystemExit(
                f"arm {arm} already consumed {consumed} >= requested {symbols}; "
                "pass --fresh to redo it"
            )
        resumed_from = str(checkpoint_path)
        base_sources = [str(base_checkpoint), str(checkpoint_path)]
        model = Seed.from_checkpoint(previous, device=device)
    else:
        if fresh and report_path.is_file():
            archived_report = str(report_path.with_name(f"run_report.prev-{_utc_now()}.json"))
            report_path.rename(archived_report)
        model = Seed.from_checkpoint(base_envelope, device=device)

    substrate = model.architecture
    # ``readout`` 不允许在同一个 dynamics episode 内切换 ⇒ 每臂/每次续跑都开新 episode。
    substrate.reset_dynamics(episode_id=f"readout-retrain-{arm}")

    session_started = time.perf_counter()
    # 进度流是**跨进程续跑追加**的；没有会话标识就无法把两条不同进程的记录分开，
    # 按相邻行的差算速率会算出负数（本会话实测踩到过）。
    session_id = f"{arm}-{_utc_now()}"
    session_symbols = 0
    before = surface_digests(substrate)
    seen = 0
    correct = 0
    surprise = 0.0
    absolute_tick = base_tick + consumed
    last_progress: dict[str, Any] = {}

    def _persist() -> None:
        envelope = attach_metadata(
            model.checkpoint(),
            tick=absolute_tick,
            corpus_fingerprint=corpus_digest,
            extra={
                "trainer": TRAINER_NAME,
                "arm": arm,
                "arm_description": spec["description"],
                "observe_kwargs": spec["observe_kwargs"],
                "symbols_consumed": consumed,
                "symbols_budget": symbols,
                "skip_symbols": skip_symbols,
                "base_checkpoint": str(base_checkpoint),
                "base_checkpoint_sha256": base_sha256,
                "base_sources": list(base_sources),
            },
        )
        atomic_save(envelope, checkpoint_path)

    def _write_progress(final: bool) -> dict[str, Any]:
        nonlocal seen, correct, surprise
        entry = {
            "arm": arm,
            "session_id": session_id,
            "written_at_utc": _utc_now(),
            "symbols_consumed": consumed,
            "symbols_budget": symbols,
            "absolute_tick": absolute_tick,
            "session_symbols": session_symbols,
            "online_accuracy": correct / max(1, seen),
            "mean_surprise": surprise / max(1, seen),
            "elapsed_seconds": round(time.perf_counter() - session_started, 3),
            "final": final,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        seen = 0
        correct = 0
        surprise = 0.0
        return entry

    # 三臂看的是同一段符号流：起点固定为血缘跳过量，长度固定为预算。
    stream = iter_corpus_window(corpus_paths, skip_symbols)
    for _ in range(consumed):
        next(stream)

    for _ in range(consumed, symbols):
        symbol = next(stream)
        step = substrate.observe(symbol, **spec["observe_kwargs"])
        consumed += 1
        session_symbols += 1
        absolute_tick += 1
        prior = getattr(step, "prior_prediction", None)
        if prior is not None:
            seen += 1
            correct += int(prior == symbol)
            surprise += float(step.surprise)
        if consumed % progress_every == 0:
            last_progress = _write_progress(final=False)
        if consumed % checkpoint_every == 0:
            _persist()

    last_progress = _write_progress(final=True)
    _persist()

    after = surface_digests(substrate)
    realised = {surface for surface in SURFACES if before[surface] != after[surface]}
    expected = set(spec["write_surface"])
    wall = time.perf_counter() - session_started
    report = {
        "status": "completed" if realised == expected else "failed",
        "arm": arm,
        "arm_description": spec["description"],
        "observe_kwargs": spec["observe_kwargs"],
        "expected_write_surface": sorted(expected),
        "realised_write_surface": sorted(realised),
        "write_surface_ok": realised == expected,
        "written_at_utc": _utc_now(),
        "trainer": TRAINER_NAME,
        "contract": CONTRACT,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha256,
        "base_sources": list(base_sources),
        "resumed_from": resumed_from,
        "fresh": bool(fresh),
        "archived_previous_report": archived_report,
        "symbols_budget": symbols,
        "symbols_consumed": consumed,
        "skip_symbols": skip_symbols,
        "corpus": {"paths": [str(path) for path in corpus_paths], "fingerprint": corpus_digest},
        "device": device,
        "absolute_tick": absolute_tick,
        "session": {
            "session_id": session_id,
            "wall_seconds": round(wall, 3),
            "seconds_per_symbol": round(wall / session_symbols, 9) if session_symbols else None,
            "symbols": session_symbols,
        },
        "last_progress": last_progress,
        "digests_before": before,
        "digests_after": after,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_of(checkpoint_path),
        "report_path": str(report_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="A,B,C")
    parser.add_argument(
        "--symbols",
        type=int,
        default=APPROVED_SYMBOL_CEILING,
        help=f"每臂符号预算（批准上限 {APPROVED_SYMBOL_CEILING}）",
    )
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--corpus", nargs="+", default=[str(DEFAULT_CORPUS)])
    parser.add_argument("--lineage-manifest", default=str(DEFAULT_LINEAGE_MANIFEST))
    parser.add_argument("--skip-symbols", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="忽略已有臂 checkpoint，从基底重开")
    parser.add_argument("--i-accept-exceeding-approved-budget", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.symbols <= 0:
        parser.error("--symbols must be positive")
    if args.symbols > APPROVED_SYMBOL_CEILING and not args.i_accept_exceeding_approved_budget:
        parser.error(
            f"--symbols {args.symbols} exceeds the approved ceiling {APPROVED_SYMBOL_CEILING}; "
            "pass --i-accept-exceeding-approved-budget if the owner has raised it"
        )
    if args.checkpoint_every <= 0 or args.progress_every <= 0:
        parser.error("--checkpoint-every / --progress-every must be positive")

    arms = [name.strip().upper() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in arms if name not in ARMS]
    if unknown:
        parser.error(f"unknown arm(s) {unknown}; known: {sorted(ARMS)}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    checkpoints_dir = (PROJECT_ROOT / "checkpoints").resolve()
    if out_dir.resolve().is_relative_to(checkpoints_dir):
        parser.error(f"--out-dir must not live under {checkpoints_dir}")

    base_checkpoint = (
        Path(args.base_checkpoint) if args.base_checkpoint else _default_checkpoint(parser)
    )
    if not base_checkpoint.is_file():
        parser.error(f"base checkpoint not found: {base_checkpoint}")
    for arm in arms:
        target = (out_dir / arm / "checkpoint.pt").resolve()
        if target == base_checkpoint.resolve():
            parser.error(f"arm {arm}'s write target is the substrate file itself: {target}")

    corpus_paths = [
        Path(item) if Path(item).is_absolute() else PROJECT_ROOT / item for item in args.corpus
    ]
    for path in corpus_paths:
        if not path.is_file():
            parser.error(f"corpus not found: {path}")
    lineage = load_lineage_skip(Path(args.lineage_manifest), corpus_paths[0])
    skip_symbols = lineage["skip_symbols"] if args.skip_symbols is None else args.skip_symbols
    if skip_symbols < lineage["skip_symbols"]:
        parser.error(
            f"--skip-symbols {skip_symbols} sits inside the seen prefix "
            f"({lineage['skip_symbols']}); the arms must train on unseen data"
        )

    base_sha256 = sha256_of(base_checkpoint)
    digest = corpus_fingerprint(corpus_paths)
    started = time.perf_counter()
    reports: list[dict[str, Any]] = []
    for arm in arms:
        print(f"[{arm}] start: budget={args.symbols} skip={skip_symbols}", flush=True)
        report = run_arm(
            arm=arm,
            symbols=args.symbols,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            arm_dir=out_dir / arm,
            base_checkpoint=base_checkpoint,
            base_sha256=base_sha256,
            corpus_paths=corpus_paths,
            corpus_digest=digest,
            skip_symbols=skip_symbols,
            device=args.device,
            fresh=args.fresh,
        )
        print(
            f"[{arm}] {report['status']}: consumed={report['symbols_consumed']} "
            f"wall={report['session']['wall_seconds']}s surface={report['realised_write_surface']}",
            flush=True,
        )
        reports.append(report)

    base_unchanged = sha256_of(base_checkpoint) == base_sha256
    failed = [report["arm"] for report in reports if report["status"] != "completed"]
    summary = {
        "trainer": TRAINER_NAME,
        "contract": CONTRACT,
        "written_at_utc": _utc_now(),
        "arms": arms,
        "symbols_budget": args.symbols,
        "skip_symbols": skip_symbols,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha256,
        "base_checkpoint_unchanged": base_unchanged,
        "out_dir": str(out_dir),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "failed_arms": failed,
        "status": "completed" if (not failed and base_unchanged) else "failed",
    }
    (out_dir / "campaign_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["status"] == "completed" else 1


def _default_checkpoint(parser: argparse.ArgumentParser) -> Path:
    from api.seed_runtime import DEFAULT_CHECKPOINT

    if not Path(DEFAULT_CHECKPOINT).is_file():
        parser.error(f"product default checkpoint not on disk: {DEFAULT_CHECKPOINT}")
    return Path(DEFAULT_CHECKPOINT)


if __name__ == "__main__":
    raise SystemExit(main())
