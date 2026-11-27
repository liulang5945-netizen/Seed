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
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.development_learning import LearningPermissionPolicy, OutcomeRouter  # noqa: E402
from taiji.dynamic_byte_chunker import (  # noqa: E402
    BOUNDARY_ROLE,
    DynamicByteVocab,
    EntropyBoundaryChunker,
)
from taiji.native_cognitive_backbone import (  # noqa: E402
    EOS_BYTE,
    NATIVE_PRESETS,
    NativeBackboneConfig,
    NativeSubjectState,
    StatefulNeuralGroup,
    TaijiNativeCognitiveBackbone,
    bits_per_byte,
)
from taiji.native_growth_controller import (  # noqa: E402
    NativeGrowthController,
    load_growth_policy,
)

NATIVE_DIR = ROOT / "data" / "n38_native_zh"
VAL_BYTES_PATH = NATIVE_DIR / "val_bytes.bin"
STATUS_PATH = NATIVE_DIR / "native_backbone_status.json"
METRICS_PATH = NATIVE_DIR / "training_metrics.jsonl"
DEFAULT_CKPT_DIR = NATIVE_DIR / "checkpoints"
DEFAULT_POLICY_PATH = ROOT / "data" / "taiji_ops" / "growth_policy.json"

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


@torch.no_grad()
def _eval_slice_neg_loss(
    model: TaijiNativeCognitiveBackbone,
    val_arr: np.ndarray,
    seq_bytes: int,
    batch_size: int,
    use_bf16: bool,
    *,
    start_batch: int = 0,
    n_batches: int = 2,
) -> float:
    was_training = model.training
    model.eval()
    stride = seq_bytes + 1
    max_start = max(1, len(val_arr) - stride * batch_size - 1)
    total_loss = 0.0
    for b in range(start_batch, start_batch + n_batches):
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
        total_loss += float(loss.item())
    if was_training:
        model.train()
    return -(total_loss / max(1, n_batches))


@torch.no_grad()
def _compute_entropy_chunk_stats(
    model: TaijiNativeCognitiveBackbone,
    val_arr: np.ndarray,
    seq_bytes: int,
    use_bf16: bool,
    chunker: EntropyBoundaryChunker,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    seg_len = min(seq_bytes, max(8, len(val_arr) - 1))
    xb = torch.from_numpy(val_arr[:seg_len].astype(np.int64)).unsqueeze(0)
    if use_bf16:
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            logits, _, _ = model(xb, p_allowed=0.0)
    else:
        logits, _, _ = model(xb, p_allowed=0.0)
    probs = F.softmax(logits[0].float(), dim=-1).clamp(min=1e-9)
    ent = -(probs * probs.log()).sum(dim=-1)
    spans = chunker.segment([float(v) for v in ent.tolist()])
    lengths = [end - start for start, end in spans]
    if was_training:
        model.train()
    return {
        "boundary_role": BOUNDARY_ROLE,
        "n_bytes": seg_len,
        "n_chunks": len(spans),
        "mean_chunk_len": round(sum(lengths) / max(1, len(lengths)), 3),
        "boundary_ratio": round(len(spans) / max(1, seg_len), 4),
    }


def _pack_capsule_meta(model: TaijiNativeCognitiveBackbone) -> dict[str, Any]:
    return {
        "n_residual": len(model.residual_groups),
        "unit_ids": [list(cast(StatefulNeuralGroup, g).unit_ids) for g in model.groups],
        "unit_branches": [list(cast(StatefulNeuralGroup, g).unit_branches) for g in model.groups],
        "retired_unit_ids": [
            list(cast(StatefulNeuralGroup, g).retired_unit_ids) for g in model.groups
        ],
        "dynamic_vocab": model.encoder.dynamic_vocab.to_dict(),
    }


def _apply_capsule_meta(model: TaijiNativeCognitiveBackbone, meta: dict[str, Any]) -> None:
    dv = meta.get("dynamic_vocab")
    if isinstance(dv, dict):
        model.encoder.dynamic_vocab = DynamicByteVocab.from_dict(dv)
    saved_ids = meta.get("unit_ids")
    if isinstance(saved_ids, list):
        for g_m, ids in zip(model.groups, saved_ids, strict=True):
            cast(StatefulNeuralGroup, g_m).unit_ids = list(ids)
    saved_branches = meta.get("unit_branches")
    if isinstance(saved_branches, list):
        for g_m, brs in zip(model.groups, saved_branches, strict=True):
            cast(StatefulNeuralGroup, g_m).unit_branches = list(brs)
    saved_retired = meta.get("retired_unit_ids")
    if isinstance(saved_retired, list):
        for g_m, rids in zip(model.groups, saved_retired, strict=True):
            cast(StatefulNeuralGroup, g_m).retired_unit_ids = list(rids)


def _restore_subject_state(
    model: TaijiNativeCognitiveBackbone,
    raw_state: Any,
    batch_size: int,
) -> NativeSubjectState:
    fresh = model.init_state(batch_size)
    if not isinstance(raw_state, dict):
        return fresh
    try:
        restored = NativeSubjectState.from_dict(raw_state)
        if (
            len(restored.h_states) != len(fresh.h_states)
            or restored.h_states[0].shape[0] != batch_size
            or len(restored.unit_z) != len(fresh.unit_z)
            or any(a.shape != b.shape for a, b in zip(restored.unit_z, fresh.unit_z, strict=True))
            or len(restored.residual_h) != len(fresh.residual_h)
        ):
            return fresh
        return restored.detach()
    except Exception:
        return fresh


def _sync_state_after_growth(
    prev: NativeSubjectState | None,
    model: TaijiNativeCognitiveBackbone,
    batch_size: int,
) -> NativeSubjectState:
    fresh = model.init_state(batch_size)
    if prev is None:
        return fresh
    for i, h in enumerate(prev.h_states):
        if i < len(fresh.h_states) and h.shape == fresh.h_states[i].shape:
            fresh.h_states[i].copy_(h.to(dtype=fresh.h_states[i].dtype))
    for i, f in enumerate(prev.fast_memories):
        if i < len(fresh.fast_memories) and f.shape == fresh.fast_memories[i].shape:
            fresh.fast_memories[i].copy_(f.to(dtype=fresh.fast_memories[i].dtype))
    for i, (oz, oa) in enumerate(zip(prev.unit_z, prev.unit_a, strict=False)):
        if i < len(fresh.unit_z) and oz.shape[0] == batch_size:
            keep = min(int(oz.shape[1]), int(fresh.unit_z[i].shape[1]))
            fresh.unit_z[i][:, :keep].copy_(oz[:, :keep].to(dtype=fresh.unit_z[i].dtype))
            fresh.unit_a[i][:, :keep].copy_(oa[:, :keep].to(dtype=fresh.unit_a[i].dtype))
    return fresh.detach()


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
    if resume_state is not None:
        capsule_meta = resume_state.get("capsule_meta")
        if isinstance(capsule_meta, dict):
            for _ in range(int(capsule_meta.get("n_residual", 0))):
                model.grow_group_neutral_()
            _apply_capsule_meta(model, capsule_meta)
        macro_w = resume_state.get("model", {}).get("encoder.macro_delta")
        if isinstance(macro_w, torch.Tensor) and macro_w.shape != model.encoder.macro_delta.shape:
            model.encoder.macro_delta = torch.nn.Parameter(torch.zeros_like(macro_w))
        model.load_state_dict(resume_state["model"])
        model.growth_lineage = list(resume_state.get("growth_lineage", []))
        opt_step = int(resume_state.get("opt_step", 0))
        bytes_seen = int(resume_state.get("bytes_seen", 0))
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        subject_state: NativeSubjectState | None = _restore_subject_state(
            model, resume_state.get("subject_state"), batch_size
        )
    else:
        subject_state = model.init_state(batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    if resume_state is not None and "opt" in resume_state:
        opt.load_state_dict(resume_state["opt"])

    params = model.count_parameters()
    bytes_per_opt = batch_size * seq_bytes * grad_accum
    total_steps_est = max(100, int((args.max_hours * 3600.0 * 1500.0) / bytes_per_opt))

    policy_path = Path(args.policy_path)
    growth_controller = NativeGrowthController(load_growth_policy(policy_path))
    chunker = EntropyBoundaryChunker(entropy_threshold=2.0, min_chunk=2, max_chunk=16)
    outcome_router = OutcomeRouter(
        LearningPermissionPolicy(
            allow_slow_theta=True,
            max_retention_drop=growth_controller.policy.max_retention_drop,
            min_gain=growth_controller.policy.min_gain,
        )
    )

    stream = _stream_byte_chunks(seq_bytes, batch_size)
    t_start = time.monotonic()
    max_sec = args.max_hours * 3600.0
    last_train_loss = 0.0
    last_val_loss = best_val_loss if math.isfinite(best_val_loss) else 0.0
    last_val_bpb = round(bits_per_byte(last_val_loss), 4) if last_val_loss > 0 else 0.0
    last_sample = ""
    last_chunk_stats: dict[str, Any] = {
        "boundary_role": BOUNDARY_ROLE,
        "n_bytes": 0,
        "n_chunks": 0,
        "mean_chunk_len": 0.0,
        "boundary_ratio": 0.0,
    }
    last_xb: torch.Tensor | None = None
    last_yb: torch.Tensor | None = None

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
            last_xb, last_yb = xb, yb
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
            last_chunk_stats = _compute_entropy_chunk_stats(
                model, val_arr, seq_bytes, use_bf16, chunker
            )
            growth_controller.refresh_policy(policy_path)
            if (
                not args.smoke
                and opt_step >= args.min_growth_step
                and growth_controller.policy.auto_grow_enabled
                and last_xb is not None
                and last_yb is not None
            ):
                proposal = growth_controller.propose_action(model, last_val_loss, step=opt_step)
                if proposal.get("action", "no_grow") != "no_grow":
                    old_handles = model.snapshot_param_handles()
                    adapt_xb, adapt_yb = last_xb, last_yb

                    def _shadow_adapt(
                        cand: TaijiNativeCognitiveBackbone,
                        *,
                        lr_val: float = lr,
                        x_b: torch.Tensor = adapt_xb,
                        y_b: torch.Tensor = adapt_yb,
                    ) -> None:
                        s_opt = torch.optim.AdamW(cand.parameters(), lr=lr_val, weight_decay=0.0)
                        for _ in range(3):
                            s_opt.zero_grad(set_to_none=True)
                            _, s_loss, _ = cand(x_b, labels=y_b)
                            assert s_loss is not None
                            s_loss.backward()
                            cand.apply_selective_plasticity_grad_mask(freeze_mature_slow=True)
                            s_opt.step()

                    grown_model, g_receipt = growth_controller.run_shadow_growth_cycle(
                        model,
                        proposal,
                        adapt_fn=_shadow_adapt,
                        eval_old_fn=lambda m: _eval_slice_neg_loss(
                            m, val_arr, seq_bytes, batch_size, use_bf16, start_batch=0, n_batches=2
                        ),
                        eval_new_fn=lambda m: _eval_slice_neg_loss(
                            m, val_arr, seq_bytes, batch_size, use_bf16, start_batch=2, n_batches=2
                        ),
                        sample_input=adapt_xb[:1, : min(64, adapt_xb.shape[1])],
                        outcome_router=outcome_router,
                        seed=opt_step,
                    )
                    if g_receipt.get("accepted"):
                        model = grown_model
                        model.migrate_adamw_state_(opt, old_handles)
                        subject_state = _sync_state_after_growth(subject_state, model, batch_size)
                        params = model.count_parameters()

            last_sample, _ = model.generate_utf8(
                "今天我们一起", max_new_bytes=16 if args.smoke else 48
            )
            ckpt_payload = {
                "model": model.state_dict(),
                "cfg": model.cfg.to_dict(),
                "growth_lineage": list(model.growth_lineage),
                "capsule_meta": _pack_capsule_meta(model),
                "subject_state": subject_state.to_dict() if subject_state is not None else None,
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
                            "params": params,
                            "entropy_chunk_stats": last_chunk_stats,
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
                    "subject_state_persisted": True,
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
                    "entropy_chunk_stats": last_chunk_stats,
                    "growth_controller_receipts_count": len(growth_controller.history),
                    "topology": model.topology_summary(),
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )

    final_payload = {
        "schema": "taiji-n38-native-training-status-v2",
        "architecture": "taiji-native-byte-group-v2",
        "input_encoding": "raw_utf8_bytes_259",
        "non_transformer": True,
        "subject_state_persisted": True,
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
        "entropy_chunk_stats": last_chunk_stats,
        "growth_controller_receipts_count": len(growth_controller.history),
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
    ap.add_argument("--policy-path", default=str(DEFAULT_POLICY_PATH))
    ap.add_argument("--min-growth-step", type=int, default=1000)
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
