"""N38.1.01 Long-Horizon CPU Pretraining for TaijiOpenChineseLM (`mid_18m`).

Streams deduplicated `train`-bucket Chinese docs (strictly excluding `val` and
`sealed_eval`), trains with bfloat16 autocast, AdamW + cosine schedule,
atomic checkpointing (`latest.pt` / `best.pt`), and live telemetry JSON/JSONL.

Usage:
  python scripts/training/n38_1_01_pretrain_open_zh.py --smoke
  python scripts/training/n38_1_01_pretrain_open_zh.py --preset mid_18m --max-hours 24 --threads 8 --dtype bfloat16 --resume
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
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.open_chinese_backbone import PRESETS, TaijiOpenChineseLM  # noqa: E402

DATA_DIR = ROOT / "data" / "n38_open_zh"
TOKENIZER_PATH = DATA_DIR / "tokenizer_8k.json"
VAL_BIN_PATH = DATA_DIR / "val_tokens.bin"
STATUS_PATH = DATA_DIR / "training_status.json"
METRICS_PATH = DATA_DIR / "training_metrics.jsonl"
DEFAULT_CKPT_DIR = DATA_DIR / "checkpoints"

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


def _is_train_bucket(text: str) -> bool:
    return (int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000) < 980


def _stream_token_chunks(
    tok: Tokenizer, seq_len: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    eos_id = tok.token_to_id("<|eos|>")
    chunk_len = seq_len + 1
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
                    if len(txt) < 16 or not _is_train_bucket(txt):
                        continue
                    ids = tok.encode(txt).ids
                    if not ids:
                        continue
                    buf.extend(ids)
                    buf.append(eos_id)
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
    model: TaijiOpenChineseLM,
    val_arr: np.ndarray,
    seq_len: int,
    batch_size: int,
    use_bf16: bool,
    n_batches: int = 24,
) -> tuple[float, float]:
    model.eval()
    stride = seq_len + 1
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
                _, loss, _ = model(xb, labels=yb)
        else:
            _, loss, _ = model(xb, labels=yb)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(min(20.0, mean_loss))
    return round(mean_loss, 4), round(ppl, 2)


@torch.no_grad()
def sample_chinese(
    model: TaijiOpenChineseLM, tok: Tokenizer, prompt: str = "今天我们一起", max_new: int = 48
) -> str:
    model.eval()
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new):
        x_cond = x[:, -model.cfg.max_seq_len :]
        logits, _, _ = model(x_cond)
        next_logits = logits[:, -1, :] / 0.8
        top_v, top_i = torch.topk(next_logits, k=min(40, next_logits.size(-1)))
        probs = torch.softmax(top_v, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_tok = top_i.gather(-1, idx)
        x = torch.cat((x, next_tok), dim=1)
    model.train()
    return tok.decode(x[0].tolist())


def _lr_for_step(
    step: int, warmup_steps: int, total_steps_est: int, max_lr: float, min_lr: float
) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps_est - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    val_arr = np.fromfile(VAL_BIN_PATH, dtype=np.uint16)

    cfg = PRESETS[args.preset]
    seq_len = 128 if args.smoke else cfg.max_seq_len
    batch_size = 2 if args.smoke else args.batch_size
    grad_accum = 1 if args.smoke else args.grad_accum
    use_bf16 = args.dtype == "bfloat16"

    model = TaijiOpenChineseLM(cfg)
    params = model.count_parameters()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    ckpt_dir = Path(args.ckpt_dir)
    status_path = Path(args.status_path)
    metrics_path = Path(args.metrics_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    latest_ckpt = ckpt_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        state = torch.load(latest_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        opt_step = int(state.get("opt_step", 0))
        tokens_seen = int(state.get("tokens_seen", 0))
        best_val_loss = float(state.get("best_val_loss", float("inf")))

    # Rough total opt steps estimate for cosine schedule (~4,300 tok/s => ~45,700 opt steps in 24h)
    est_tps = 4343.0 if args.preset == "mid_18m" else 6400.0
    toks_per_opt = batch_size * seq_len * grad_accum
    total_steps_est = max(100, int((args.max_hours * 3600.0 * est_tps) / toks_per_opt))

    stream = _stream_token_chunks(tok, seq_len, batch_size)
    t_start = time.monotonic()
    max_sec = args.max_hours * 3600.0
    last_train_loss = 0.0
    last_val_loss = best_val_loss if math.isfinite(best_val_loss) else 0.0
    last_val_ppl = round(math.exp(min(20.0, last_val_loss)), 2) if last_val_loss > 0 else 0.0
    last_sample = ""

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
                    _, loss, _ = model(xb, labels=yb)
            else:
                _, loss, _ = model(xb, labels=yb)
            assert loss is not None
            (loss / grad_accum).backward()
            accum_loss += float(loss.item()) / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        opt_step += 1
        tokens_seen += toks_per_opt
        last_train_loss = round(accum_loss, 4)
        elapsed = time.monotonic() - t_start
        tps = round(tokens_seen / max(1e-3, elapsed), 1)
        progress_pct = round(min(100.0, 100.0 * elapsed / max(1.0, max_sec)), 2)

        if opt_step % eval_every == 0 or opt_step == 1:
            last_val_loss, last_val_ppl = evaluate_val(
                model, val_arr, seq_len, batch_size, use_bf16, n_batches=4 if args.smoke else 24
            )
            last_sample = sample_chinese(model, tok, max_new=16 if args.smoke else 48)
            ckpt_payload = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "opt_step": opt_step,
                "tokens_seen": tokens_seen,
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
                            "tokens_seen": tokens_seen,
                            "tokens_per_sec": tps,
                            "elapsed_sec": round(elapsed, 1),
                            "train_loss": last_train_loss,
                            "val_loss": last_val_loss,
                            "val_ppl": last_val_ppl,
                            "lr": round(lr, 7),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        if opt_step % 10 == 0 or opt_step == 1 or opt_step % eval_every == 0:
            _atomic_write_json(
                status_path,
                {
                    "schema": "taiji-n38-training-status-v1",
                    "status": "running",
                    "preset": args.preset,
                    "params": params,
                    "pid": os.getpid(),
                    "opt_step": opt_step,
                    "tokens_seen": tokens_seen,
                    "tokens_per_sec": tps,
                    "elapsed_sec": round(elapsed, 1),
                    "max_hours": args.max_hours,
                    "progress_pct": progress_pct,
                    "train_loss": last_train_loss,
                    "val_loss": last_val_loss,
                    "val_ppl": last_val_ppl,
                    "best_val_loss": (
                        round(best_val_loss, 4) if math.isfinite(best_val_loss) else 0.0
                    ),
                    "lr": round(lr, 7),
                    "sample_generation": last_sample,
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )

    final_status = {
        "schema": "taiji-n38-training-status-v1",
        "status": "completed",
        "preset": args.preset,
        "params": params,
        "pid": os.getpid(),
        "opt_step": opt_step,
        "tokens_seen": tokens_seen,
        "elapsed_sec": round(time.monotonic() - t_start, 1),
        "max_hours": args.max_hours,
        "progress_pct": 100.0,
        "train_loss": last_train_loss,
        "val_loss": last_val_loss,
        "val_ppl": last_val_ppl,
        "best_val_loss": round(best_val_loss, 4) if math.isfinite(best_val_loss) else 0.0,
        "sample_generation": last_sample,
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_write_json(status_path, final_status)
    return final_status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="mid_18m", choices=tuple(PRESETS.keys()))
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--status-path", default=str(STATUS_PATH))
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            args.ckpt_dir = str(tdir / "ckpts")
            args.status_path = str(tdir / "status.json")
            args.metrics_path = str(tdir / "metrics.jsonl")
            res1 = run_training(args)
            assert math.isfinite(res1["train_loss"])
            args.resume = True
            res2 = run_training(args)
            print(json.dumps({"smoke": "ok", "step": res2["opt_step"], "loss": res2["train_loss"]}))
        return 0

    res = run_training(args)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
