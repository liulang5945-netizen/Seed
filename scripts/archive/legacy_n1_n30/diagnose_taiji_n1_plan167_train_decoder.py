"""Fixed train-only CC-2 readout attribution; no learning or dev access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

CPU_START = time.process_time()
WALL_START = time.monotonic()

import torch  # noqa: E402

from scripts.training import evaluate_taiji_r2_daily_dialogue_recovery_v2 as frozen  # noqa: E402
from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base  # noqa: E402
from taiji.language_alignment import LanguageAlignmentTrainer, content_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
READINESS = (
    ROOT
    / "reports/taiji_r2_daily_dialogue_recovery_v2_20261021_extension_v1_paired_eval_recovery_v2_formal_001_readiness.json"
)
OUTPUT = ROOT / "reports/taiji_n1_plan167_train_decoder_result_20261025.json"
PROGRESS = ROOT / "reports/taiji_n1_plan167_train_decoder_progress_20261025.json"
MODES = ("configured", "protected", "oracle_plan")
MAX_BYTES = 128


def guard() -> None:
    # Retain 30 seconds inside the hard package cap for state restoration/publication.
    if time.process_time() - CPU_START > 1770 or time.monotonic() - WALL_START > 2370:
        raise TimeoutError("N1-167 fixed diagnostic CPU/wall ceiling reached")


def select_train(corpus: Any) -> list[Any]:
    selected = []
    for episode in corpus.for_split("train"):
        if (
            re.search(r"[\u4e00-\u9fff]", episode.response)
            and 48 <= len(episode.target_bytes) <= 384
            and len(episode.prompt_bytes) <= 512
            and not any(byte < 32 and byte not in (9, 10, 13) for byte in episode.target_bytes)
        ):
            selected.append(episode)
        if len(selected) == 4:
            return selected
    raise ValueError("fewer than four train episodes meet the preregistered filter")


def restore(trainer: Any, snapshot: dict[str, Any]) -> None:
    trainer.model.restore(snapshot)
    trainer._configure_developmental_mode()
    trainer._configure_response_start_readout()
    trainer._configure_response_phase_readout()
    trainer._configure_response_plan_readout()


def prime(trainer: Any, episode: Any, mode: str) -> Any:
    trainer._prime(episode)
    if mode == "protected":
        return None
    config, model = trainer.config, trainer.model
    readout = None
    if config.response_phase_readout:
        model.begin_response_phase()
        readout = model.response_phase_readout
    if config.response_plan_readout:
        model.begin_response_plan()
        readout = model.response_plan_readout
        if mode == "oracle_plan":
            # Deliberately label-informed dynamic state, never a weight update.
            readout._plan_state = (
                trainer._response_plan_target(episode).detach().to(readout.device).clone()
            )
            state = model.snapshot()
            state.motor_probabilities = model.response_plan_probabilities().detach().clone()
            model._state = state
    elif mode == "oracle_plan":
        raise ValueError("oracle plan is unavailable without the configured plan readout")
    return readout


def probabilities(trainer: Any, mode: str, position: int) -> torch.Tensor:
    model, config = trainer.model, trainer.config
    if mode != "protected":
        if config.response_plan_readout:
            return model.response_plan_probabilities().detach().cpu()
        if config.response_start_readout and position == 0:
            return model.response_start_probabilities().detach().cpu()
        if config.response_phase_readout:
            return model.response_phase_probabilities().detach().cpu()
    return model._state.motor_probabilities.detach().cpu().clone()


def masked(prob: torch.Tensor, remaining: int, lead: int) -> torch.Tensor:
    allowed = LanguageAlignmentTrainer._utf8_allowed(remaining, lead)
    result = torch.zeros_like(prob)
    result[allowed] = prob[allowed]
    mass = float(result.sum())
    if mass <= 0 or not math.isfinite(mass):
        raise ValueError("non-finite or empty legal next-byte distribution")
    return result / mass


def advance(trainer: Any, symbol: int, mode: str, readout: Any) -> None:
    trainer._observe(symbol, learn=False, predictive_readout=readout)
    if (
        mode != "protected"
        and trainer.config.response_plan_readout
        and trainer.config.response_plan_variant == "factorized_v1"
    ):
        trainer.model.advance_response_plan_phase()


def summarize_tf(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = {}
    for lo, hi in ((0, 16), (16, 32), (32, 48), (48, 64), (64, 128)):
        part = rows[lo:hi]
        bins[f"{lo}-{hi - 1}"] = {
            "count": len(part),
            "top1_raw": sum(row["raw_top1"] == row["target"] for row in part) / max(1, len(part)),
        }
    return {
        "positions": len(rows),
        "top1_raw": sum(row["raw_top1"] == row["target"] for row in rows) / len(rows),
        "top1_masked": sum(row["masked_top1"] == row["target"] for row in rows) / len(rows),
        "nll_raw_bits": sum(row["nll_raw_bits"] for row in rows) / len(rows),
        "nll_masked_bits": sum(row["nll_masked_bits"] for row in rows) / len(rows),
        "first_error": next(
            (i for i, row in enumerate(rows) if row["raw_top1"] != row["target"]), None
        ),
        "phase_bins": bins,
        "rows": rows,
    }


def teacher_forced(trainer: Any, episode: Any, mode: str) -> tuple[dict[str, Any], torch.Tensor]:
    readout = prime(trainer, episode, mode)
    remaining = lead = 0
    rows, distributions = [], []
    for position, target in enumerate(episode.target_bytes[:MAX_BYTES]):
        guard()
        prob = probabilities(trainer, mode, position)
        legal = masked(prob, remaining, lead)
        if not bool(torch.isfinite(prob).all()):
            raise ValueError("non-finite raw probability")
        rows.append(
            {
                "target": target,
                "raw_top1": int(prob.argmax()),
                "masked_top1": int(legal.argmax()),
                "nll_raw_bits": -math.log2(max(float(prob[target]), 1e-30)),
                "nll_masked_bits": -math.log2(max(float(legal[target]), 1e-30)),
            }
        )
        distributions.append(prob)
        advance(trainer, target, mode, readout)
        remaining, lead = trainer._advance_utf8(remaining, lead, target)
    return summarize_tf(rows), torch.stack(distributions)


def rollout(trainer: Any, episode: Any, mode: str) -> bytes:
    readout = prime(trainer, episode, mode)
    remaining = lead = 0
    output = bytearray()
    for position in range(MAX_BYTES):
        guard()
        symbol = int(masked(probabilities(trainer, mode, position), remaining, lead).argmax())
        output.append(symbol)
        advance(trainer, symbol, mode, readout)
        remaining, lead = trainer._advance_utf8(remaining, lead, symbol)
        if output.endswith(b"<|end|>"):
            break
    raw = bytes(output)
    for cut in range(len(raw), max(0, len(raw) - 4), -1):
        try:
            raw[:cut].decode("utf-8")
            return raw[:cut]
        except UnicodeDecodeError:
            pass
    return raw


def prefix_match(raw: bytes, target: bytes) -> float:
    length = min(MAX_BYTES, len(target))
    hit = 0
    for left, right in zip(raw, target[:length], strict=False):
        if left != right:
            break
        hit += 1
    return hit / max(1, length)


def js_bits(left: torch.Tensor, right: torch.Tensor) -> float:
    p = left.double().clamp_min(1e-30)
    q = right.double().clamp_min(1e-30)
    p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
    midpoint = 0.5 * (p + q)
    return float(
        (
            0.5 * (p * (p / midpoint).log2()).sum(-1) + 0.5 * (q * (q / midpoint).log2()).sum(-1)
        ).mean()
    )


def run() -> dict[str, Any]:
    torch.set_num_threads(1)
    if OUTPUT.exists():
        raise ValueError("N1-167 output exists; do not overwrite prior diagnostic evidence")
    report: dict[str, Any] = {
        "schema": "taiji-n1-plan167-train-decoder-v1",
        "status": "running",
        "parameter_updates": 0,
        "optimizer_updates": 0,
        "dev_queries": 0,
        "can_promote": False,
        "records": [],
        "endpoint_summaries": {},
        "source_sha256": {},
    }
    sources = [
        Path(__file__),
        Path(base.__file__),
        Path(frozen.__file__),
        READINESS,
        ROOT / "taiji/language_alignment.py",
        ROOT / "taiji/model.py",
        ROOT / "taiji/organs.py",
        ROOT / "taiji/response_plan_target.py",
        ROOT / "plans/reference/PLAN-N1-167_train_decoder_attribution_20261025.md",
        ROOT / "plans/reference/SPEC-N1-167_train_decoder_attribution_20261025.md",
    ]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): frozen.sha256_file(path) for path in sources
    }
    try:
        readiness = frozen.read_json(READINESS)
        corpus = base._load_assistant_train_corpus()  # split-only scan; train payloads only.
        selected = select_train(corpus)
        report["selection"] = [
            {
                "episode_id": ep.episode_id,
                "prompt_sha256": hashlib.sha256(ep.prompt_bytes).hexdigest(),
                "target_sha256": hashlib.sha256(ep.target_bytes).hexdigest(),
                "prompt_bytes": len(ep.prompt_bytes),
                "target_bytes": len(ep.target_bytes),
                "train_response": ep.response,
            }
            for ep in selected
        ]
        report["corpus_digest"] = corpus.digest
        for endpoint in readiness["endpoints"]:
            guard()
            checkpoint = Path(endpoint["path"])
            before_sha = frozen.sha256_file(checkpoint)
            if before_sha != endpoint["sha256"]:
                raise ValueError("checkpoint identity drift before diagnostic")
            trainer, _ = frozen._load_endpoint_trainer(endpoint, corpus)
            snapshot = trainer.model.checkpoint()
            digest = content_digest(snapshot)
            endpoint_id = endpoint["endpoint_id"]
            report["endpoint_summaries"][endpoint_id] = {
                "checkpoint_sha256": before_sha,
                "config": {
                    name: getattr(trainer.config, name)
                    for name in (
                        "response_plan_readout",
                        "response_plan_variant",
                        "response_plan_target_geometry",
                        "response_phase_readout",
                        "response_start_readout",
                    )
                },
                "model_digest": digest,
            }
            for index, episode in enumerate(selected):
                restore(trainer, snapshot)
                training_metric = trainer._target_pass(episode, learn=False)
                restore(trainer, snapshot)
                actual_raw = trainer._constrained_generate(episode, max_generation_bytes=MAX_BYTES)
                native_distributions = None
                native_tf = None
                for mode in MODES:
                    row = {
                        "endpoint_id": endpoint_id,
                        "episode_id": episode.episode_id,
                        "mode": mode,
                        "label_informed": mode == "oracle_plan",
                        "training_path_readonly": training_metric,
                    }
                    if mode == "oracle_plan" and not trainer.config.response_plan_readout:
                        report["records"].append({**row, "status": "not_applicable"})
                        continue
                    restore(trainer, snapshot)
                    tf, distributions = teacher_forced(trainer, episode, mode)
                    restore(trainer, snapshot)
                    raw = rollout(trainer, episode, mode)
                    if mode == "configured":
                        if raw != actual_raw:
                            raise ValueError(
                                "instrumented configured rollout differs from the actual decoder"
                            )
                        native_distributions, native_tf = distributions, tf
                    text, utf8_valid, _, boundary, stop = trainer._generated_text(raw)
                    row.update(
                        status="complete",
                        tf=tf,
                        generated_text=text,
                        generated_bytes_hex=raw.hex(),
                        generated_bytes_length=len(raw),
                        utf8_valid=utf8_valid,
                        boundary=boundary,
                        stop_reason=stop,
                        target_prefix_match=prefix_match(raw, episode.target_bytes),
                    )
                    restore(trainer, snapshot)
                    if content_digest(trainer.model.checkpoint()) != digest:
                        raise ValueError("diagnostic did not restore the original model")
                    report["records"].append(row)
                other = selected[(index + 1) % len(selected)]
                swapped = replace(
                    episode,
                    system=other.system,
                    context=other.context,
                    history=other.history,
                    user_input=other.user_input,
                    unknown_policy=other.unknown_policy,
                )
                restore(trainer, snapshot)
                wrong_tf, wrong_distributions = teacher_forced(trainer, swapped, "configured")
                native_row = next(
                    row
                    for row in reversed(report["records"])
                    if row["endpoint_id"] == endpoint_id
                    and row["episode_id"] == episode.episode_id
                    and row["mode"] == "configured"
                )
                native_row["prompt_swap"] = {
                    "other_episode_id": other.episode_id,
                    "mean_js_bits": js_bits(native_distributions, wrong_distributions),
                    "top1_agreement": float(
                        (native_distributions.argmax(-1) == wrong_distributions.argmax(-1))
                        .float()
                        .mean()
                    ),
                    "nll_raw_bits_delta": wrong_tf["nll_raw_bits"] - native_tf["nll_raw_bits"],
                }
                restore(trainer, snapshot)
                temporary = PROGRESS.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(PROGRESS)
            if (
                content_digest(trainer.model.checkpoint()) != digest
                or frozen.sha256_file(checkpoint) != before_sha
            ):
                raise ValueError("endpoint changed during train-only diagnosis")
            report["endpoint_summaries"][endpoint_id]["read_only_verified"] = True
        if len(report["records"]) != 48:
            raise ValueError("fixed 48-cell diagnostic matrix is incomplete")
        report["status"] = "complete"
    except Exception as exc:
        report.update(
            status="partial" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    report.update(
        cpu_seconds=time.process_time() - CPU_START, wall_seconds=time.monotonic() - WALL_START
    )
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": len(result["records"]),
                "cpu_seconds": result["cpu_seconds"],
                "error": result.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
