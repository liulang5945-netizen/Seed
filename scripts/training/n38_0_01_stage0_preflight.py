"""N38.0.01 Stage 0 Preflight: Data Audit, 8K Chinese BPE Tokenizer & CPU Throughput Benchmark.

Zero confirmation seed reads, zero check/final reads.
Outputs:
  data/n38_open_zh/tokenizer_8k.json
  data/n38_open_zh/val_tokens.bin
  data/n38_open_zh/split_manifest.json
  reports/n38_0_01_stage0_preflight.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.n38_baseline_transformer import PRESETS, TaijiOpenChineseLM  # noqa: E402

DATA_OUT = ROOT / "data" / "n38_open_zh"
REPORT_OUT = ROOT / "reports" / "n38_0_01_stage0_preflight.json"

CORPUS_FILES: list[tuple[str, Path]] = [
    ("simple_zh_texts", ROOT / "data" / "simple_zh" / "simple_zh_texts.jsonl"),
    ("shared_core", ROOT / "data" / "simple_zh" / "shared_core.jsonl"),
    ("dialogue_extended_clean", ROOT / "data" / "simple_zh" / "dialogue_extended_clean.jsonl"),
    ("p3b_dialogue_subset", ROOT / "data" / "p3b_dialogue_subset.jsonl"),
    ("p3b_dialogue_fresh", ROOT / "data" / "p3b_dialogue_fresh.jsonl"),
    (
        "ultradata_sft",
        ROOT / "data" / "ultradata" / "derived" / "ultradata_sft_nothink_simplezh.jsonl",
    ),
    (
        "ultradata_holdout",
        ROOT / "data" / "ultradata" / "derived" / "ultradata_hetero_holdout_simplezh.jsonl",
    ),
]

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>"]


def _extract_text(obj: dict[str, Any]) -> str:
    if "text" in obj and isinstance(obj["text"], str):
        return obj["text"]
    if "messages" in obj and isinstance(obj["messages"], list):
        parts = []
        for m in obj["messages"]:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(f"<|{m.get('role', 'user')}|>{m['content']}")
        return "\n".join(parts)
    if "conversations" in obj and isinstance(obj["conversations"], list):
        parts = []
        for m in obj["conversations"]:
            if isinstance(m, dict) and isinstance(m.get("value"), str):
                parts.append(m["value"])
        return "\n".join(parts)
    return json.dumps(obj, ensure_ascii=False)


def _bucket_for_text(text: str) -> str:
    h = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if h < 980:
        return "train"
    if h < 990:
        return "val"
    return "sealed_eval"


def audit_corpora(max_docs_per_file: int) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    seen_hashes: set[str] = set()
    stats: list[dict[str, Any]] = []
    train_sample: list[str] = []
    val_sample: list[str] = []
    train_bytes = 0
    val_bytes = 0

    for name, path in CORPUS_FILES:
        if not path.exists():
            stats.append({"name": name, "exists": False})
            continue
        file_bytes = path.stat().st_size
        n_docs = 0
        n_unique = 0
        sample_bytes = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if n_docs >= max_docs_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = _extract_text(obj).strip()
                if len(txt) < 8:
                    continue
                n_docs += 1
                sample_bytes += len(line.encode("utf-8"))
                dh = hashlib.sha256(txt.encode("utf-8")).hexdigest()
                if dh not in seen_hashes:
                    seen_hashes.add(dh)
                    n_unique += 1
                    b = _bucket_for_text(txt)
                    if b == "train" and train_bytes < 25_000_000:
                        train_sample.append(txt)
                        train_bytes += len(txt.encode("utf-8"))
                    elif b == "val" and val_bytes < 4_000_000:
                        val_sample.append(txt)
                        val_bytes += len(txt.encode("utf-8"))
        unique_ratio = round(n_unique / max(1, n_docs), 4)
        est_total_unique_mb = round((file_bytes / 1048576.0) * unique_ratio, 1)
        stats.append(
            {
                "name": name,
                "exists": True,
                "size_mb": round(file_bytes / 1048576.0, 1),
                "sampled_docs": n_docs,
                "sampled_unique_docs": n_unique,
                "unique_ratio": unique_ratio,
                "est_unique_mb": est_total_unique_mb,
            }
        )
    return stats, train_sample, val_sample


def train_and_verify_tokenizer(
    train_texts: list[str], val_texts: list[str], out_path: Path
) -> dict[str, Any]:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(train_texts, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))

    # Verify compression on val_texts and 0% OOV on rare CJK
    probe_texts = val_texts[:200] if val_texts else train_texts[:200]
    total_chars = sum(len(t) for t in probe_texts)
    total_bytes = sum(len(t.encode("utf-8")) for t in probe_texts)
    total_toks = sum(len(tok.encode(t).ids) for t in probe_texts)

    oov_probes = [
        "饕餮餮餮，魑魅魍魉；请把青瓷茶盏放到红木方桌上。",
        "龑龘靐齉，生僻字回环往复测试：甲乙丙丁戊己庚辛。",
    ]
    roundtrip_ok = all(tok.decode(tok.encode(p).ids) == p for p in oov_probes)

    # Materialize val_tokens.bin
    val_ids: list[int] = []
    eos_id = tok.token_to_id("<|eos|>")
    for t in val_texts:
        val_ids.extend(tok.encode(t).ids)
        val_ids.append(eos_id)
        if len(val_ids) >= 1_500_000:
            break
    arr = np.array(val_ids, dtype=np.uint16)
    val_bin = out_path.parent / "val_tokens.bin"
    arr.tofile(val_bin)

    tok_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": tok_sha,
        "val_chars_per_token": round(total_chars / max(1, total_toks), 4),
        "val_bytes_per_token": round(total_bytes / max(1, total_toks), 4),
        "oov_roundtrip_exact": roundtrip_ok,
        "val_tokens_materialized": int(arr.size),
    }


def _bench_step(
    preset_name: str,
    *,
    threads: int,
    dtype_str: str,
    seq_len: int,
    batch_size: int,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    torch.set_num_threads(threads)
    cfg = PRESETS[preset_name]
    model = TaijiOpenChineseLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    x = torch.randint(1, cfg.vocab_size, (batch_size, seq_len))
    y = torch.randint(1, cfg.vocab_size, (batch_size, seq_len))
    use_bf16 = dtype_str == "bfloat16"

    for i in range(warmup + steps):
        if i == warmup:
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if use_bf16:
            with torch.amp.autocast("cpu", dtype=torch.bfloat16):
                _, loss, _ = model(x, labels=y)
        else:
            _, loss, _ = model(x, labels=y)
        assert loss is not None
        loss.backward()
        opt.step()

    elapsed = time.perf_counter() - t0
    toks_per_step = batch_size * seq_len
    tps = (steps * toks_per_step) / max(1e-6, elapsed)
    params = model.count_parameters()
    proj = {}
    for hrs in (12, 24, 48):
        tot_toks = tps * hrs * 3600.0
        proj[f"{hrs}h"] = {
            "total_tokens_m": round(tot_toks / 1e6, 1),
            "tokens_per_param": round(tot_toks / params, 2),
        }
    return {
        "preset": preset_name,
        "params": params,
        "threads": threads,
        "dtype": dtype_str,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "sec_per_step": round(elapsed / steps, 3),
        "tokens_per_sec": round(tps, 1),
        "projections": proj,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    max_docs = 2000 if args.smoke else 40000
    stats, train_sample, val_sample = audit_corpora(max_docs)
    tok_info = train_and_verify_tokenizer(train_sample, val_sample, DATA_OUT / "tokenizer_8k.json")

    # Sweep threads and dtype on mid_18m
    sweep: list[dict[str, Any]] = []
    thread_list = [12] if args.smoke else [8, 12, 16, 20]
    for dt in ("float32", "bfloat16"):
        for th in thread_list:
            try:
                res = _bench_step(
                    "mid_18m",
                    threads=th,
                    dtype_str=dt,
                    seq_len=256 if args.smoke else 512,
                    batch_size=2 if args.smoke else 4,
                    warmup=1 if args.smoke else 2,
                    steps=2 if args.smoke else 5,
                )
                sweep.append(res)
            except Exception as exc:
                sweep.append({"preset": "mid_18m", "threads": th, "dtype": dt, "error": str(exc)})

    valid_sweep = [s for s in sweep if "tokens_per_sec" in s]
    winner = max(valid_sweep, key=lambda s: float(s["tokens_per_sec"]))

    presets_bench: list[dict[str, Any]] = []
    for pname in ("small_8m", "mid_18m", "large_30m"):
        presets_bench.append(
            _bench_step(
                pname,
                threads=int(winner["threads"]),
                dtype_str=str(winner["dtype"]),
                seq_len=256 if args.smoke else 512,
                batch_size=2 if args.smoke else 4,
                warmup=1 if args.smoke else 2,
                steps=2 if args.smoke else 6,
            )
        )

    split_manifest = {
        "schema": "taiji-n38-split-manifest-v1",
        "split_rule": "sha256(text)[:8] % 1000 -> 0..979 train, 980..989 val, 990..999 sealed_eval",
        "tokenizer": tok_info,
        "corpora": stats,
    }
    (DATA_OUT / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    payload = {
        "schema": "taiji-n38-stage0-preflight-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "smoke": args.smoke,
        "corpora": stats,
        "tokenizer": tok_info,
        "winner_config": {"threads": winner["threads"], "dtype": winner["dtype"]},
        "sweep": sweep,
        "presets_benchmark": presets_bench,
        "discipline": {"check_reads": 0, "final_reads": 0, "confirm_reads": 0},
    }
    if not args.smoke:
        REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
        REPORT_OUT.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "smoke": args.smoke,
                "winner": payload["winner_config"],
                "presets": [
                    {
                        "preset": p["preset"],
                        "params": p["params"],
                        "tps": p["tokens_per_sec"],
                        "24h_m_toks": p["projections"]["24h"]["total_tokens_m"],
                        "24h_tok_per_param": p["projections"]["24h"]["tokens_per_param"],
                    }
                    for p in presets_bench
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
