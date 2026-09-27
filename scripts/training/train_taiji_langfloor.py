"""SPEC-R2-01：语言地板可训——UTF-8 续字节加权读出重训 runner（两臂 D0/D）。

预注册：``plans/reference/SPEC-R2-01_language_floor_utf8_weighted_readout_prereg_20260927.md``
（判据/臂定义/预算/止损先于代码冻结）。

**它修什么**：F0 判 fail——被计分的无掩码通道上，基座守不住 UTF-8 字节序列
（真汉字与断裂序列混吐，首非法字节中位位置 1–4）。retrain arm A（读出重训 16M 符号）
把"认字"提上去了（掩码成句 0.87），但无掩码仍 0/32——缺的是"续字节必须跟上调"这条硬结构。
本件在只训读出（arm A 配方，写面唯一）的基础上，把**续字节位置的读出更新加权 W 倍**
（走现成钩子 `_predictive_update_scale`，非新增参数面、非产品码改动）；
D0（W=1.0）为同流对照臂，归因"加权"这个变量本身。

纪律（沿用 `train_taiji_r2_readout_retrain` 全套）：写靶只在 ``--out-dir``；基底只读
（跑前后 sha256 复核）；断点续跑（臂目录已有 checkpoint 即继续，``--fresh`` 重开）；
stop 文件优雅停；两臂同一段符号流；判读一律事后用 F0 探针，本件不做评价。

用法::

    python scripts/training/train_taiji_langfloor.py --arm D --weight 4.0 \\
        --symbols 2000000 --out-dir output/taiji_r2_langfloor_D
    python scripts/training/train_taiji_langfloor.py --arm D --weight 4.0 --smoke \\
        --out-dir output/taiji_r2_langfloor_D
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed import Seed, iter_native_documents  # noqa: E402
from seed.persistence import atomic_save, attach_metadata, corpus_fingerprint  # noqa: E402
from taiji import TaijiConfig  # noqa: E402

TRAINER_NAME = "train_taiji_langfloor"

#: arm A 配方：写面唯一＝predictive_readout。
OBSERVE_KWARGS: dict[str, Any] = {
    "learn": True,
    "readout": "predictive",
    "learn_motor": False,
    "learn_fabric": False,
    "learn_predictive_context": False,
    "learn_predictive_readout": True,
}

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"
DEFAULT_BASE = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def iter_dialogue_symbols(paths: list[Path]) -> Iterator[int]:
    """每行一个会话：boundary + 该行 UTF-8 字节（对话结构已在文本内，问：/答：标记）。"""
    boundary = TaijiConfig().boundary_symbol
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield boundary
                # jsonl 行本身带 {"text": ...} 信封；喂信封内的 text（与 train_seed_corpus
                # 的 iter_native_documents 同口径，避免把 JSON 语法喂进模型）。
                payload = json.loads(text)
                content = payload["text"] if isinstance(payload, dict) and "text" in payload else text
                for symbol in content.encode("utf-8"):
                    yield symbol


def is_utf8_continuation(symbol: int) -> bool:
    """UTF-8 多字节序列的第 2..n 字节（0x80–0xBF）：落在这些位置＝正在写字的中间。"""
    return 0x80 <= symbol <= 0xBF


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("D0", "D"))
    parser.add_argument("--weight", type=float, required=True, help="续字节位置的读出更新倍数（D0=1.0，D=4.0）")
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE))
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--symbols", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-minutes", type=float, default=None, help="保险上限；预算以符号数为准")
    parser.add_argument("--smoke", action="store_true", help="tiny 预算快速端到端")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.smoke:
        args.symbols = min(args.symbols, 200_000)
        args.checkpoint_every = min(args.checkpoint_every, 100_000)
        args.progress_every = min(args.progress_every, 25_000)

    if args.arm == "D0" and args.weight != 1.0:
        raise SystemExit("arm D0 is the unweighted control; --weight must be 1.0")
    if args.arm == "D" and args.weight == 1.0:
        raise SystemExit("arm D is the weighted arm; --weight 1.0 would duplicate D0")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.pt"
    report_path = out_dir / "run_report.json"
    progress_path = out_dir / "progress.jsonl"
    stop_file = out_dir / "STOP"

    base_checkpoint = Path(args.base_checkpoint)
    if not base_checkpoint.is_absolute():
        base_checkpoint = PROJECT_ROOT / base_checkpoint
    base_sha = _sha256(base_checkpoint)

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = PROJECT_ROOT / corpus_path

    resumed_from: str | None = None
    consumed = 0
    if checkpoint_path.is_file() and not args.fresh:
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = previous.get("metadata", {})
        extra = meta.get("extra", meta)
        if extra.get("arm") != args.arm:
            raise SystemExit(f"{checkpoint_path} belongs to arm {extra.get('arm')!r}, not {args.arm!r}")
        if extra.get("base_checkpoint_sha256") != base_sha:
            raise SystemExit("checkpoint was trained from a different base substrate")
        if float(extra.get("weight", -1)) != float(args.weight):
            raise SystemExit("checkpoint was trained with a different --weight")
        consumed = int(extra.get("symbols_consumed", 0))
        if consumed >= args.symbols:
            print(json.dumps({"guard_ok": False, "error": f"arm already complete at {consumed}"}))
            return 2
        resumed_from = str(checkpoint_path)
        model = Seed.from_checkpoint(previous, device=args.device)
    else:
        if args.fresh and report_path.is_file():
            report_path.rename(report_path.with_name(f"run_report.prev-{_stamp_for_filename()}.json"))
        base_envelope = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        model = Seed.from_checkpoint(base_envelope, device=args.device)

    substrate = model.architecture
    substrate.reset_dynamics(episode_id=f"langfloor-{args.arm}")

    corpus_digest = corpus_fingerprint([corpus_path])
    stream = iter_dialogue_symbols([corpus_path])
    for _ in range(consumed):
        next(stream)

    def _section_digests() -> dict[str, str]:
        """checkpoint 两层深的分段摘要：写面前后对比，证明只动了 predictive_readout。"""
        core = model.checkpoint()
        digests: dict[str, str] = {}

        def _digest(name: str, payload: Any) -> None:
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            digests[name] = hashlib.sha256(buffer.getvalue()).hexdigest()

        for section, payload in core.items():
            if section in ("rng_state", "state"):
                continue
            if isinstance(payload, dict):
                for key, sub in payload.items():
                    _digest(f"{section}.{key}", sub)
            else:
                _digest(section, payload)
        return digests

    before_digests = _section_digests()

    started = time.perf_counter()
    session_id = f"{args.arm}-{_utc_now()}"
    seen = 0
    correct = 0
    surprise = 0.0
    cont_steps = 0  # 续字节加权步数（真"被走到"计数，防 B1 首跑式静默失效）
    plain_steps = 0
    cont_seen = 0
    cont_correct = 0
    absolute_tick = consumed
    stopped_by_request = False

    def _persist() -> None:
        envelope = attach_metadata(
            model.checkpoint(),
            tick=absolute_tick,
            corpus_fingerprint=corpus_digest,
            extra={
                "trainer": TRAINER_NAME,
                "arm": args.arm,
                "weight": float(args.weight),
                "observe_kwargs": OBSERVE_KWARGS,
                "symbols_consumed": consumed,
                "symbols_budget": args.symbols,
                "corpus": str(corpus_path),
                "base_checkpoint": str(base_checkpoint),
                "base_checkpoint_sha256": base_sha,
            },
        )
        atomic_save(envelope, checkpoint_path)

    def _write_progress(final: bool) -> None:
        entry = {
            "arm": args.arm,
            "session_id": session_id,
            "written_at_utc": _utc_now(),
            "symbols_consumed": consumed,
            "symbols_budget": args.symbols,
            "session_symbols": consumed,
            "online_accuracy": correct / seen if seen else None,
            "continuation_online_accuracy": cont_correct / cont_seen if cont_seen else None,
            "continuation_weighted_steps": cont_steps,
            "final": final,
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    kwargs = dict(OBSERVE_KWARGS)
    while consumed < args.symbols:
        symbol = next(stream)
        if is_utf8_continuation(symbol):
            kwargs["_predictive_update_scale"] = float(args.weight)
            cont_steps += 1
        else:
            kwargs["_predictive_update_scale"] = 1.0
            plain_steps += 1
        step = substrate.observe(symbol, **kwargs)
        consumed += 1
        absolute_tick += 1
        prior = getattr(step, "prior_prediction", None)
        if prior is not None:
            seen += 1
            correct += int(prior == symbol)
            surprise += float(step.surprise)
            if is_utf8_continuation(symbol):
                cont_seen += 1
                cont_correct += int(prior == symbol)
        if consumed % args.progress_every == 0:
            _write_progress(final=False)
        if consumed % args.checkpoint_every == 0:
            _persist()
        if (
            stop_file.exists()
            and consumed % 10_000 == 0
        ):
            stop_file.unlink()
            stopped_by_request = True
            break
        if args.max_minutes is not None and (time.perf_counter() - started) > args.max_minutes * 60:
            break

    _persist()
    _write_progress(final=True)
    after_digests = _section_digests()
    changed = sorted(name for name in after_digests if before_digests.get(name) != after_digests[name])
    #: 声明式写面守卫：读出必须变；motor/fabric/memory（快通路主干与记忆）必须不变。
    #: 其余段（cognitive_state/state/perception 等）是运行时状态流，随观察演化属预期，只报不 gate。
    write_surface_guard = {
        "predictive_readout_changed": any("predictive_readout" in name for name in changed),
        "motor_fabric_memory_unchanged": not any(
            any(part in name for part in (".motor", ".fabric", ".memory"))
            for name in changed
        ),
    }

    elapsed = round(time.perf_counter() - started, 1)
    base_unchanged = _sha256(base_checkpoint) == base_sha
    report = {
        "format": "taiji-langfloor-run-v1",
        "prereg": "plans/reference/SPEC-R2-01_language_floor_utf8_weighted_readout_prereg_20260927.md",
        "trainer": TRAINER_NAME,
        "arm": args.arm,
        "weight": float(args.weight),
        "symbols_consumed": consumed,
        "symbols_budget": args.symbols,
        "stopped_by": "stop-file" if stopped_by_request else ("time-cap" if consumed < args.symbols else "episode-cap"),
        "elapsed_seconds": elapsed,
        "resumed_from": resumed_from,
        "corpus": str(corpus_path),
        "corpus_digest": corpus_digest,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_sha,
        "write_surface": {"changed_named_tensors": changed, "expected_only": ["predictive_readout"]},
        "write_surface_guard": write_surface_guard,
        "learning_counters": {
            "continuation_weighted_steps": cont_steps,
            "plain_steps": plain_steps,
            "online_accuracy": round(correct / seen, 4) if seen else None,
            "continuation_online_accuracy": round(cont_correct / cont_seen, 4) if cont_seen else None,
        },
        "base_sha256_unchanged": base_unchanged,
        "written_at_utc": _utc_now(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guard = {
        "symbols_consumed": consumed > 0,
        "continuation_steps_ran": cont_steps > 0,
        "base_unchanged": base_unchanged,
        "write_surface_ok": write_surface_guard["predictive_readout_changed"]
        and write_surface_guard["motor_fabric_memory_unchanged"],
    }
    print(
        json.dumps(
            {
                "guard_ok": all(guard.values()),
                "arm": args.arm,
                "weight": args.weight,
                "symbols": consumed,
                "online_accuracy": report["learning_counters"]["online_accuracy"],
                "continuation_online_accuracy": report["learning_counters"]["continuation_online_accuracy"],
                "changed_surfaces": changed,
                "elapsed_s": elapsed,
            },
            ensure_ascii=False,
        )
    )
    return 0 if all(guard.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
