"""N38.2.02 Long-Horizon CPU Pretraining for 态极 Native Byte-Group Backbone.

Pure UTF-8 raw bytes (ZERO BPE, ZERO Transformer), carrying persistent subject state
(H, z, a, F) across training windows (`CORE_ALG §15.2`), reporting `val_loss` (nats/byte)
and `val_bpb` (bits-per-byte) on the exact same deduplicated Chinese corpus.

Usage:
  python scripts/training/n38_2_02_pretrain_native_zh.py --smoke
  python scripts/training/n38_2_02_pretrain_native_zh.py --preset native_byte_mid --max-hours 24 --threads 8 --dtype bfloat16 --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.native_cognitive_backbone import (  # noqa: E402
    EOS_BYTE,
    NATIVE_PRESETS,
    NativeBackboneConfig,
    NativeSubjectState,
    TaijiNativeCognitiveBackbone,
    bits_per_byte,
)

NATIVE_DIR = ROOT / "data" / "n38_native_zh"
VAL_BYTES_PATH = NATIVE_DIR / "val_bytes.bin"
STATUS_PATH = NATIVE_DIR / "native_backbone_status.json"
METRICS_PATH = NATIVE_DIR / "training_metrics.jsonl"
DEFAULT_CKPT_DIR = NATIVE_DIR / "checkpoints"

TRAIN_SOURCES: list[Path] = [
    ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl",
    ROOT / "data" / "simple_zh" / "shared_core.jsonl",
    ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl",
    ROOT / "data" / "p3b_dialogue_subset.jsonl",
    ROOT / "data" / "p3b_dialogue_fresh.jsonl",
    ROOT / "data" / "ultradata" / "derived" / "ultradata_sft_nothink_simplezh.jsonl",
]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _atomic_save_pt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _extract_text(obj: dict[str, Any]) -> str:
    if "text" in obj and isinstance(obj["text"], str):
        return obj["text"]
    if "messages" in obj and isinstance(obj["messages"], list):
        parts = [
            f"<|{m.get('role', 'user')}|>{m['content']}"
            for m in obj["messages"]
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        ]
        return "\n".join(parts)
    if "conversations" in obj and isinstance(obj["conversations"], list):
        parts = [
            m["value"]
            for m in obj["conversations"]
            if isinstance(m, dict) and isinstance(m.get("value"), str)
        ]
        return "\n".join(parts)
    return ""


def _bucket_id(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000


def _ensure_val_bytes(path: Path, max_bytes: int = 1_000_000) -> np.ndarray:
    if path.exists():
        return np.fromfile(path, dtype=np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf: list[int] = []
    for src in TRAIN_SOURCES:
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = _extract_text(obj).strip()
                if len(txt) < 16:
                    continue
                if 980 <= _bucket_id(txt) < 990:
                    buf.extend(txt.encode("utf-8"))
                    buf.append(EOS_BYTE)
                    if len(buf) >= max_bytes:
                        break
        if len(buf) >= max_bytes:
            break
    arr = np.array(buf[:max_bytes], dtype=np.uint16)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    arr.tofile(tmp)
    os.replace(tmp, path)
    return arr


def _stream_byte_chunks(
    seq_bytes: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    chunk_len = seq_bytes + 1
    buf: list[int] = []
    batch_x: list[list[int]] = []
    batch_y: list[list[int]] = []
    existing = [p for p in TRAIN_SOURCES if p.exists()]
    while True:
        for src in existing:
            with src.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    txt = _extract_text(obj).strip()
                    if len(txt) < 16 or _bucket_id(txt) >= 980:
                        continue
                    buf.extend(txt.encode("utf-8"))
                    buf.append(EOS_BYTE)
                    while len(buf) >= chunk_len:
                        chunk = buf[:chunk_len]
                        buf = buf[chunk_len:]
                        batch_x.append(chunk[:-1])
                        batch_y.append(chunk[1:])
                        if len(batch_x) == batch_size:
                            yield (
                                torch.tensor(batch_x, dtype=torch.long),
                                torch.tensor(batch_y, dtype=torch.long),
                            )
                            batch_x.clear()
                            batch_y.clear()


@torch.no_grad()
def evaluate_val(
    model: TaijiNativeCognitiveBackbone,
    val_arr: np.ndarray,
    seq_bytes: int,
    batch_size: int,
    use_bf16: bool,
    n_batches: int = 16,
) -> tuple[float, float]:
    model.eval()
    stride = seq_bytes + 1
    max_start = max(1, len(val_arr) - stride * batch_size - 1)
    losses: list[float] = []
    for b in range(n_batches):
        base = (b * stride * batch_size) % max_start
        xs = []
        ys = []
        for k in range(batch_size):
            st = base + k * stride
            seg = val_arr[st : st + stride].astype(np.int64)
            xs.append(seg[:-1])
            ys.append(seg[1:])
        xb = torch.from_numpy(np.stack(xs))
        yb = torch.from_numpy(np.stack(ys))
        if use_bf16:
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                _, loss, _ = model(xb, labels=yb, p_allowed=0.0)
        else:
            _, loss, _ = model(xb, labels=yb, p_allowed=0.0)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return round(mean_loss, 4), round(bits_per_byte(mean_loss), 4)


def _lr_for_step(
    step: int, warmup_steps: int, total_steps_est: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps_est - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    val_arr = _ensure_val_bytes(Path(args.val_bytes_path))

    cfg = NATIVE_PRESETS[args.preset]
    seq_bytes = 128 if args.smoke else cfg.max_seq_bytes
    batch_size = 2 if args.smoke else args.batch_size
    grad_accum = 1 if args.smoke else args.grad_accum
    use_bf16 = args.dtype == "bfloat16"

    ckpt_dir = Path(args.ckpt_dir)
    status_path = Path(args.status_path)
    metrics_path = Path(args.metrics_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = ckpt_dir / "latest.pt"
    opt_step = 0
    bytes_seen = 0
    best_val_loss = float("inf")
    resume_state: dict[str, Any] | None = None
    if args.resume and latest_ckpt.exists():
        resume_state = torch.load(latest_ckpt, map_location="cpu", weights_only=True)
        if isinstance(resume_state, dict) and "cfg" in resume_state:
            cfg = NativeBackboneConfig.from_dict(resume_state["cfg"])

    model = TaijiNativeCognitiveBackbone(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        model.growth_lineage = list(resume_state.get("growth_lineage", []))
        opt.load_state_dict(resume_state["opt"])
        opt_step = int(resume_state.get("opt_step", 0))
        bytes_seen = int(resume_state.get("bytes_seen", 0))
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))

    params = model.count_parameters()
    bytes_per_opt = batch_size * seq_bytes * grad_accum
    total_steps_est = max(100, int((args.max_hours * 3600.0 * 1500.0) / bytes_per_opt))

    stream = _stream_byte_chunks(seq_bytes, batch_size)
    t_start = time.monotonic()
    max_sec = args.max_hours * 3600.0
    last_train_loss = 0.0
    last_val_loss = best_val_loss if math.isfinite(best_val_loss) else 0.0
    last_val_bpb = round(bits_per_byte(last_val_loss), 4) if last_val_loss > 0 else 0.0
    last_sample = ""

    # Persistent subject state carried across windows (CORE_ALG §15.2)
    subject_state: NativeSubjectState | None = model.init_state(batch_size)

    max_opt_steps = 4 if args.smoke else 10_000_000
    eval_every = 2 if args.smoke else args.eval_every

    while opt_step < max_opt_steps:
        elapsed = time.monotonic() - t_start
        if not args.smoke and elapsed >= max_sec:
            break

        lr = _lr_for_step(opt_step, 400, total_steps_est, 3e-4, 3e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(grad_accum):
            xb, yb = next(stream)
            if use_bf16:
                with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                    _, loss, next_state = model(xb, state=subject_state, labels=yb)
            else:
                _, loss, next_state = model(xb, state=subject_state, labels=yb)
            assert loss is not None
            (loss / grad_accum).backward()
            accum_loss += float(loss.item()) / grad_accum
            subject_state = next_state.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        opt_step += 1
        bytes_seen += bytes_per_opt
        last_train_loss = round(accum_loss, 4)
        elapsed = time.monotonic() - t_start
        bps = round(bytes_seen / max(1e-3, elapsed), 1)
        progress_pct = round(min(100.0, 100.0 * elapsed / max(1.0, max_sec)), 2)

        if opt_step % eval_every == 0 or opt_step == 1:
            last_val_loss, last_val_bpb = evaluate_val(
                model, val_arr, seq_bytes, batch_size, use_bf16, n_batches=4 if args.smoke else 16
            )
            last_sample, _ = model.generate_utf8(
                "今天我们一起", max_new_bytes=16 if args.smoke else 48
            )
            ckpt_payload = {
                "model": model.state_dict(),
                "cfg": model.cfg.to_dict(),
                "growth_lineage": list(model.growth_lineage),
                "opt": opt.state_dict(),
                "opt_step": opt_step,
                "bytes_seen": bytes_seen,
                "best_val_loss": min(best_val_loss, last_val_loss),
                "preset": args.preset,
            }
            _atomic_save_pt(latest_ckpt, ckpt_payload)
            if last_val_loss < best_val_loss:
                best_val_loss = last_val_loss
                _atomic_save_pt(ckpt_dir / "best.pt", ckpt_payload)

            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(
                    json.dumps(
                        {
                            "ts_utc": datetime.now(UTC).isoformat(),
                            "opt_step": opt_step,
                            "bytes_seen": bytes_seen,
                            "bytes_per_sec": bps,
                            "elapsed_sec": round(elapsed, 1),
                            "train_loss": last_train_loss,
                            "val_loss": last_val_loss,
                            "val_bpb": last_val_bpb,
                            "lr": round(lr, 7),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if opt_step % 10 == 0 or opt_step == 1 or args.smoke:
            _atomic_write_json(
                status_path,
                {
                    "schema": "taiji-n38-native-training-status-v2",
                    "architecture": "taiji-native-byte-group-v2",
                    "input_encoding": "raw_utf8_bytes_259",
                    "non_transformer": True,
                    "status": "running",
                    "preset": args.preset,
                    "params": params,
                    "pid": os.getpid(),
                    "opt_step": opt_step,
                    "bytes_seen": bytes_seen,
                    "bytes_per_sec": bps,
                    "elapsed_sec": round(elapsed, 1),
                    "max_hours": args.max_hours,
                    "progress_pct": progress_pct,
                    "train_loss": last_train_loss,
                    "val_loss": last_val_loss,
                    "val_bpb": last_val_bpb,
                    "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else 0,
                    "lr": round(lr, 7),
                    "sample_generation": last_sample,
                    "topology": model.topology_summary(),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )

    final_payload = {
        "schema": "taiji-n38-native-training-status-v2",
        "architecture": "taiji-native-byte-group-v2",
        "input_encoding": "raw_utf8_bytes_259",
        "non_transformer": True,
        "status": "completed",
        "preset": args.preset,
        "params": params,
        "pid": os.getpid(),
        "opt_step": opt_step,
        "bytes_seen": bytes_seen,
        "val_loss": last_val_loss,
        "val_bpb": last_val_bpb,
        "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else 0,
        "sample_generation": last_sample,
        "topology": model.topology_summary(),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(status_path, final_payload)
    return final_payload


def main() -> int:
    ap = argparse.ArgumentParser(description="N38.2.02 态极 Native Byte-Group CPU Pretraining")
    ap.add_argument("--preset", default="native_byte_mid", choices=sorted(NATIVE_PRESETS))
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-bytes-path", default=str(VAL_BYTES_PATH))
    ap.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    ap.add_argument("--status-path", default=str(STATUS_PATH))
    ap.add_argument("--metrics-path", default=str(METRICS_PATH))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            args.ckpt_dir = str(tmp_p / "checkpoints")
            args.status_path = str(tmp_p / "native_backbone_status.json")
            args.metrics_path = str(tmp_p / "training_metrics.jsonl")
            out = run_training(args)
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return 0

    out = run_training(args)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
