"""Matched warm-start/fresh-initialization full-turn dialogue training arms."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_r2_daily_dialogue_full_turn_pretraining as base
from taiji import LanguageAlignmentTrainer, Taiji, TaijiConfig
from taiji.internalization import content_digest

PLAN = ROOT / "plans/reference/PLAN-R2-DAILY-DIALOGUE-WARMSTART-MATCHED_20261021.md"
PLAN_SHA256 = "1c3628f2fcf419a37215ae9a6b67d991e64bbb4c6e1f3f638e90ee30b39110cf"
CHECKPOINT_ROOT = ROOT / "data/checkpoints/r2_daily_dialogue_warmstart_matched_20261021"
REPORT_ROOT = ROOT / "reports"
REPORT_PREFIX = REPORT_ROOT / "taiji_r2_daily_dialogue_warmstart_matched_20261021"
FRESH_SEED = 20261023
CPU_LIMIT_PER_ARM = 120_000.0
MIN_FREE_MEMORY_GIB = 8.0
ARM_NAMES = ("warm_start", "fresh_same_capacity")


class Refuse(RuntimeError):
    """A matched dialogue-transfer precondition failed."""


def _sha256(path: Path) -> str:
    return base.sha256(path)


def _arm_paths(arm: str) -> dict[str, Path]:
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "checkpoint_dir": CHECKPOINT_ROOT / arm,
        "readiness": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_{arm}_readiness.json"),
        "attempt": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_{arm}_attempt.json"),
        "progress": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_{arm}_progress.jsonl"),
        "result": REPORT_PREFIX.with_name(f"{REPORT_PREFIX.name}_{arm}_result.json"),
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resource_snapshot() -> dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=1.0)
    available_gib = psutil.virtual_memory().available / 1024**3
    if available_gib < MIN_FREE_MEMORY_GIB:
        raise Refuse(f"available memory below {MIN_FREE_MEMORY_GIB} GiB: {available_gib:.2f}")
    if cpu_percent > 85.0:
        raise Refuse(f"system CPU exceeds parallel-start ceiling: {cpu_percent:.1f}%")
    return {
        "logical_cpu_count": os.cpu_count(),
        "system_cpu_percent_sample": cpu_percent,
        "available_memory_gib": round(available_gib, 3),
    }


def _source_paths() -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                *base._source_paths(),
                PLAN,
                ROOT / "tests/taiji_native/test_r2_daily_dialogue_warmstart_matched_contract.py",
                Path(__file__).resolve(),
            )
        )
    )


def _source_hashes() -> dict[str, str]:
    return {str(path.resolve()): _sha256(path) for path in _source_paths()}


def _validate_inputs() -> tuple[Any, Any, dict[str, Any], dict[str, Any], dict[str, str]]:
    if _sha256(PLAN) != PLAN_SHA256:
        raise Refuse("matched warm-start plan hash differs")
    pretrain_corpus, assistant_corpus, manifest = base._validate_sources()
    baseline = torch.load(base.BASELINE_CHECKPOINT, map_location="cpu", weights_only=True)
    baseline_digest = content_digest(
        {key: value for key, value in baseline.items() if key != "checkpoint_digest"}
    )
    if (
        baseline.get("checkpoint_digest") != baseline_digest
        or baseline.get("checkpoint_digest")
        != "1a77364a42633686dc9e1e9ee6a13a7e09b4f07770e3841d873bacd9eadaeaeb"
        or baseline.get("corpus_digest")
        != base._read_json(base.ASSISTANT_MANIFEST).get("corpus_digest")
        or base.sha256(base.BASELINE_CHECKPOINT) != base.BASELINE_CHECKPOINT_SHA256
    ):
        raise Refuse("warm-start parent checkpoint provenance differs")
    config = baseline.get("model", {}).get("config")
    if not isinstance(config, dict) or config.get("region_sizes") != [112, 80, 56]:
        raise Refuse("warm-start parent is not the frozen native 300k R2 architecture")
    hashes = _source_hashes()
    return pretrain_corpus, assistant_corpus, manifest, baseline, hashes


def _new_model(arm: str, baseline: dict[str, Any]) -> Taiji:
    if arm == "warm_start":
        model = Taiji.from_checkpoint(baseline["model"], device="cpu")
    elif arm == "fresh_same_capacity":
        config_payload = dict(baseline["model"]["config"])
        config_payload["seed"] = FRESH_SEED
        model_config = TaijiConfig.from_dict(config_payload)
        model = Taiji(model_config, device="cpu")
        model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
    else:
        raise ValueError(f"unknown arm: {arm}")
    if not model.response_plan_readout_enabled:
        raise Refuse("paired arm is missing the shared response-plan readout")
    return model


def _new_pretrainer(model: Taiji, corpus: Any) -> LanguageAlignmentTrainer:
    return LanguageAlignmentTrainer(
        model,
        corpus,
        config=base._pretraining_config(),
        code_revision=base._fresh_code_revision(),
    )


def _record(stream, payload: dict[str, Any]) -> None:
    base.write_jsonl(stream, payload)


def prepare(arm: str) -> dict[str, Any]:
    paths = _arm_paths(arm)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"matched arm is write-once and already exists: {arm}")
    resource = _resource_snapshot()
    pretrain_corpus, assistant_corpus, manifest, baseline, source_hashes = _validate_inputs()
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=False)
    progress = paths["progress"].open("x", encoding="utf-8", newline="\n")
    start_cpu = time.process_time()
    preflight_budget = base.CpuBudget(CPU_LIMIT_PER_ARM)
    attempt = {
        "schema": "taiji-r2-warmstart-matched-attempt-v1",
        "arm": arm,
        "status": "preflight_started",
        "plan_sha256": PLAN_SHA256,
        "source_hashes": source_hashes,
        "parameter_updates": 0,
        "generation_queries": 0,
        "can_promote": False,
    }
    _write_atomic(paths["attempt"], attempt)
    try:
        if shutil.disk_usage(CHECKPOINT_ROOT.parent).free < 512 * 1024**2:
            raise Refuse("less than 512 MiB remains for the matched arm")
        model = _new_model(arm, baseline)
        pretrainer = _new_pretrainer(model, pretrain_corpus)
        parameters = model.parameter_count()
        baseline_model = Taiji.from_checkpoint(baseline["model"], device="cpu")
        if parameters != baseline_model.parameter_count():
            raise Refuse("warm/fresh paired model parameter counts differ")
        zero_path = paths["checkpoint_dir"] / "full_turn_zero.pt"
        zero_payload = pretrainer.checkpoint()
        base._atomic_save(zero_payload, zero_path)
        zero_restore = base._independent_restore(zero_path, "pretraining", preflight_budget)
        if zero_restore.get("checkpoint_digest") != zero_payload.get("checkpoint_digest"):
            raise Refuse("full-turn zero checkpoint did not independently restore")
        probe = LanguageAlignmentTrainer.from_checkpoint(zero_payload, pretrain_corpus)
        probe_result = probe.train(epochs=1, max_episodes=1)
        first_bytes = len(pretrain_corpus.episodes[0].target_bytes)
        if (
            probe_result.get("status") != "completed"
            or probe.global_step != first_bytes
            or probe.episode_count != 1
            or pretrainer.checkpoint()["checkpoint_digest"] != zero_payload["checkpoint_digest"]
        ):
            raise Refuse("full-turn one-episode credit probe failed or mutated its parent")

        sft_model = Taiji.from_checkpoint(zero_payload["model"], device="cpu")
        sft_canary, encoder = base._assistant_sft_trainer(
            sft_model, assistant_corpus, base._fresh_code_revision()
        )
        sft_zero_path = paths["checkpoint_dir"] / "assistant_sft_canary_zero.pt"
        sft_zero_payload = sft_canary.checkpoint()
        base._atomic_save(sft_zero_payload, sft_zero_path)
        sft_restore = base._independent_restore(sft_zero_path, "assistant_sft", preflight_budget)
        if sft_restore.get("checkpoint_digest") != sft_zero_payload.get("checkpoint_digest"):
            raise Refuse("assistant calibration canary checkpoint did not independently restore")
        sft_probe = LanguageAlignmentTrainer.from_checkpoint(sft_zero_payload, assistant_corpus)
        sft_probe_result = sft_probe.train(epochs=1, max_episodes=1)
        if (
            sft_probe_result.get("status") != "completed"
            or sft_probe.global_step != len(assistant_corpus.episodes[0].target_bytes)
            or sft_canary.checkpoint()["checkpoint_digest"] != sft_zero_payload["checkpoint_digest"]
        ):
            raise Refuse("assistant calibration credit probe failed or mutated its parent")

        if _source_hashes() != source_hashes:
            raise Refuse("a frozen input changed during matched-arm preflight")
        readiness = {
            "schema": "taiji-r2-warmstart-matched-readiness-v1",
            "status": "ready",
            "arm": arm,
            "plan_sha256": PLAN_SHA256,
            "source_hashes": source_hashes,
            "resource_snapshot": resource,
            "pretraining_corpus_digest": pretrain_corpus.digest,
            "assistant_train_corpus_digest": assistant_corpus.digest,
            "pretraining_episodes": len(pretrain_corpus.episodes),
            "pretraining_families": len({item.family_id for item in pretrain_corpus.episodes}),
            "pretraining_target_bytes_per_epoch": sum(
                len(item.target_bytes) for item in pretrain_corpus.episodes
            ),
            "assistant_train_episodes": len(assistant_corpus.episodes),
            "model_parameters": parameters,
            "model_config": model.checkpoint()["config"],
            "warm_start_parent_checkpoint_sha256": base.BASELINE_CHECKPOINT_SHA256,
            "fresh_seed": FRESH_SEED if arm == "fresh_same_capacity" else None,
            "zero_checkpoint": {
                "path": str(zero_path.resolve()),
                "sha256": _sha256(zero_path),
                **zero_restore,
            },
            "assistant_canary_zero_checkpoint": {
                "path": str(sft_zero_path.resolve()),
                "sha256": _sha256(sft_zero_path),
                **sft_restore,
            },
            "credit_probes": {
                "pretraining_target_bytes": first_bytes,
                "assistant_target_bytes": len(assistant_corpus.episodes[0].target_bytes),
                "parent_checkpoints_unchanged": True,
            },
            "cpu_seconds_preflight": round(
                time.process_time() - start_cpu + preflight_budget.child_seconds, 6
            ),
            "cpu_limit_seconds": CPU_LIMIT_PER_ARM,
            "formal_updates": 0,
            "generation_queries": 0,
            "can_promote": False,
        }
        base.write_new_json(paths["readiness"], readiness)
        _record(
            progress,
            {
                "event": "checkpoint_and_credit_preflight_pass",
                "readiness_sha256": _sha256(paths["readiness"]),
            },
        )
        attempt.update(
            {
                "status": "ready",
                "readiness_sha256": _sha256(paths["readiness"]),
                "cpu_seconds_preflight": readiness["cpu_seconds_preflight"],
            }
        )
        _write_atomic(paths["attempt"], attempt)
        return readiness
    except BaseException as exc:
        attempt.update(
            {
                "status": "preflight_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "cpu_seconds": round(
                    time.process_time() - start_cpu + preflight_budget.child_seconds, 6
                ),
            }
        )
        _write_atomic(paths["attempt"], attempt)
        raise
    finally:
        progress.flush()
        progress.close()


def execute(arm: str) -> dict[str, Any]:
    paths = _arm_paths(arm)
    if not paths["attempt"].is_file() or not paths["readiness"].is_file():
        raise Refuse(f"{arm} has not passed checkpoint and credit preflight")
    attempt = base._read_json(paths["attempt"])
    readiness = base._read_json(paths["readiness"])
    if attempt.get("status") != "ready" or readiness.get("status") != "ready":
        raise Refuse(f"{arm} preflight is not ready")
    preflight_cpu = float(readiness.get("cpu_seconds_preflight", 0.0))
    budget = base.CpuBudget(max(1.0, CPU_LIMIT_PER_ARM - preflight_cpu))
    pretrain_corpus, assistant_corpus, _, baseline, source_hashes = _validate_inputs()
    if source_hashes != readiness.get("source_hashes"):
        raise Refuse("matched-arm inputs changed since checkpoint preflight")
    zero_path = Path(readiness["zero_checkpoint"]["path"])
    if _sha256(zero_path) != readiness["zero_checkpoint"]["sha256"]:
        raise Refuse("matched-arm zero checkpoint changed after preflight")
    pretrainer = LanguageAlignmentTrainer.from_checkpoint(
        torch.load(zero_path, map_location="cpu", weights_only=True), pretrain_corpus
    )
    if (
        pretrainer.checkpoint()["checkpoint_digest"]
        != readiness["zero_checkpoint"]["checkpoint_digest"]
    ):
        raise Refuse("matched-arm full-turn zero bundle does not restore exactly")
    if _resource_snapshot()["available_memory_gib"] < MIN_FREE_MEMORY_GIB:
        raise Refuse("insufficient memory to start matched-arm training")

    attempt.update(
        {
            "status": "training",
            "training_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    _write_atomic(paths["attempt"], attempt)
    progress = paths["progress"].open("a", encoding="utf-8", newline="\n")
    active_trainer: LanguageAlignmentTrainer | None = pretrainer
    active_stage = "full_turn_pretraining"
    pretrain_count = 0
    pretrain_updates = 0
    sft_count = 0
    sft_updates = 0
    try:

        def on_pretrain(record: dict[str, Any]) -> None:
            nonlocal pretrain_count, pretrain_updates
            pretrain_count += 1
            pretrain_updates += int(record["observations"])
            if pretrain_count % 128 == 0:
                _record(
                    progress,
                    {
                        "event": "full_turn_pretraining_progress",
                        "episode": pretrain_count,
                        "episodes_planned": base.PRETRAIN_EPISODE_COUNT * base.PRETRAIN_EPOCHS,
                        "target_byte_updates": pretrain_updates,
                        "cpu_seconds": round(preflight_cpu + budget.used(), 3),
                        "available_memory_gib": round(
                            psutil.virtual_memory().available / 1024**3, 3
                        ),
                    },
                )
                print(
                    f"r2-warmstart-{arm} episode={pretrain_count}/{base.PRETRAIN_EPISODE_COUNT * base.PRETRAIN_EPOCHS} updates={pretrain_updates} cpu={preflight_cpu + budget.used():.1f}",
                    flush=True,
                )
            if psutil.virtual_memory().available < 4 * 1024**3:
                raise Refuse("available memory fell below 4 GiB during matched training")
            budget.guard()

        pretrain_result = pretrainer.train(epochs=base.PRETRAIN_EPOCHS, progress=on_pretrain)
        if (
            pretrain_result.get("status") != "completed"
            or pretrain_count != base.PRETRAIN_EPISODE_COUNT * base.PRETRAIN_EPOCHS
            or pretrain_updates != base.PRETRAIN_TARGET_BYTES_PER_EPOCH * base.PRETRAIN_EPOCHS
            or pretrainer.global_step != pretrain_updates
        ):
            raise Refuse(
                "full-turn matched-arm training did not meet the fixed episode/update budget"
            )

        pretrain_final = paths["checkpoint_dir"] / "full_turn_final.pt"
        pretrain_payload = pretrainer.checkpoint()
        base._atomic_save(pretrain_payload, pretrain_final)
        pretrain_restore = base._independent_restore(pretrain_final, "pretraining", budget)
        if pretrain_restore.get("checkpoint_digest") != pretrain_payload.get("checkpoint_digest"):
            raise Refuse("full-turn matched-arm final checkpoint restore failed")
        _record(
            progress,
            {
                "event": "full_turn_final_restored",
                "checkpoint_sha256": _sha256(pretrain_final),
                "checkpoint_digest": pretrain_payload["checkpoint_digest"],
                "restore": pretrain_restore,
            },
        )

        sft_model = Taiji.from_checkpoint(pretrain_payload["model"], device="cpu")
        sft_trainer, encoder = base._assistant_sft_trainer(
            sft_model, assistant_corpus, base._fresh_code_revision()
        )
        active_trainer = sft_trainer
        active_stage = "assistant_sft"
        sft_zero = paths["checkpoint_dir"] / "assistant_sft_zero.pt"
        sft_zero_payload = sft_trainer.checkpoint()
        base._atomic_save(sft_zero_payload, sft_zero)
        sft_zero_restore = base._independent_restore(sft_zero, "assistant_sft", budget)
        if sft_zero_restore.get("checkpoint_digest") != sft_zero_payload.get("checkpoint_digest"):
            raise Refuse("assistant calibration zero checkpoint restore failed")
        sft_probe = LanguageAlignmentTrainer.from_checkpoint(sft_zero_payload, assistant_corpus)
        probe_result = sft_probe.train(epochs=1, max_episodes=1)
        if (
            probe_result.get("status") != "completed"
            or sft_probe.global_step != len(assistant_corpus.episodes[0].target_bytes)
            or sft_trainer.checkpoint()["checkpoint_digest"]
            != sft_zero_payload["checkpoint_digest"]
        ):
            raise Refuse("assistant calibration one-episode credit probe failed")

        def on_sft(record: dict[str, Any]) -> None:
            nonlocal sft_count, sft_updates
            sft_count += 1
            sft_updates += int(record["observations"])
            if sft_count % 64 == 0:
                _record(
                    progress,
                    {
                        "event": "assistant_sft_progress",
                        "episode": sft_count,
                        "episodes_planned": base.ASSISTANT_TRAIN_EPISODE_COUNT,
                        "target_byte_updates": sft_updates,
                        "cpu_seconds": round(preflight_cpu + budget.used(), 3),
                    },
                )
                print(
                    f"r2-warmstart-{arm} sft_episode={sft_count}/{base.ASSISTANT_TRAIN_EPISODE_COUNT} updates={sft_updates} cpu={preflight_cpu + budget.used():.1f}",
                    flush=True,
                )
            budget.guard()

        sft_result = sft_trainer.train(epochs=base.ASSISTANT_CALIBRATION_EPOCHS, progress=on_sft)
        if (
            sft_result.get("status") != "completed"
            or sft_count != base.ASSISTANT_TRAIN_EPISODE_COUNT
            or sft_updates != base.ASSISTANT_TRAIN_TARGET_BYTES
            or sft_trainer.global_step != sft_updates
        ):
            raise Refuse("assistant calibration did not meet the fixed episode/update budget")
        sft_final = paths["checkpoint_dir"] / "assistant_sft_final.pt"
        sft_final_payload = sft_trainer.checkpoint()
        base._atomic_save(sft_final_payload, sft_final)
        sft_final_restore = base._independent_restore(sft_final, "assistant_sft", budget)
        if sft_final_restore.get("checkpoint_digest") != sft_final_payload.get("checkpoint_digest"):
            raise Refuse("assistant calibration final checkpoint restore failed")

        if _source_hashes() != source_hashes:
            raise Refuse("a frozen input changed during matched-arm training")
        result = {
            "schema": "taiji-r2-warmstart-matched-result-v1",
            "status": "complete",
            "complete": True,
            "arm": arm,
            "plan_sha256": PLAN_SHA256,
            "source_hashes": source_hashes,
            "model_parameters": sft_trainer.model.parameter_count(),
            "pretraining": {
                "episodes": pretrain_count,
                "target_byte_updates": pretrain_updates,
                "global_step": pretrainer.global_step,
                "checkpoint": {
                    "path": str(pretrain_final.resolve()),
                    "sha256": _sha256(pretrain_final),
                    "restore": pretrain_restore,
                },
            },
            "assistant_calibration": {
                "episodes": sft_count,
                "target_byte_updates": sft_updates,
                "global_step": sft_trainer.global_step,
                "target_encoder_format": encoder.FORMAT,
                "checkpoint": {
                    "path": str(sft_final.resolve()),
                    "sha256": _sha256(sft_final),
                    "restore": sft_final_restore,
                },
            },
            "checkpoint_restores": {
                "full_turn_final": pretrain_restore,
                "assistant_zero": sft_zero_restore,
                "assistant_final": sft_final_restore,
            },
            "cpu_seconds_total": round(preflight_cpu + budget.used(), 6),
            "cpu_limit_seconds": CPU_LIMIT_PER_ARM,
            "generation_queries": 0,
            "dev_queries": 0,
            "reference_answer_scoring_queries": 0,
            "can_promote": False,
        }
        base.write_new_json(paths["result"], result)
        attempt.update(
            {
                "status": "complete",
                "result_sha256": _sha256(paths["result"]),
                "cpu_seconds_total": result["cpu_seconds_total"],
            }
        )
        _write_atomic(paths["attempt"], attempt)
        _record(
            progress,
            {
                "event": "arm_complete",
                "result_sha256": _sha256(paths["result"]),
                "cpu_seconds_total": result["cpu_seconds_total"],
            },
        )
        return result
    except BaseException as exc:
        partial = None
        if active_trainer is not None:
            partial_path = paths["checkpoint_dir"] / f"{active_stage}_partial.pt"
            try:
                if not partial_path.exists():
                    partial_payload = active_trainer.checkpoint()
                    base._atomic_save(partial_payload, partial_path)
                    partial = {
                        "path": str(partial_path.resolve()),
                        "sha256": _sha256(partial_path),
                        "checkpoint_digest": partial_payload.get("checkpoint_digest"),
                    }
            except Exception as save_exc:
                partial = {"save_error": f"{type(save_exc).__name__}: {save_exc}"}
        failure = {
            "schema": "taiji-r2-warmstart-matched-result-v1",
            "status": "partial" if pretrain_count or sft_count else "failed",
            "complete": False,
            "arm": arm,
            "plan_sha256": PLAN_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
            "pretraining_episodes": pretrain_count,
            "pretraining_updates": pretrain_updates,
            "assistant_episodes": sft_count,
            "assistant_updates": sft_updates,
            "partial_checkpoint": partial,
            "cpu_seconds_total": round(preflight_cpu + budget.used(), 6),
            "can_promote": False,
        }
        if not paths["result"].exists():
            base.write_new_json(paths["result"], failure)
        attempt.update(
            {
                "status": failure["status"],
                "error": failure["error"],
                "result_sha256": _sha256(paths["result"]),
            }
        )
        _write_atomic(paths["attempt"], attempt)
        raise
    finally:
        progress.flush()
        progress.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = prepare(args.arm) if args.prepare_only else execute(args.arm)
        print(
            json.dumps(
                {
                    "status": result.get("status", "ready"),
                    "arm": args.arm,
                    "model_parameters": result.get("model_parameters"),
                    "cpu_seconds": result.get(
                        "cpu_seconds_total", result.get("cpu_seconds_preflight")
                    ),
                    "readiness": result.get("readiness_sha256"),
                    "result": result.get("result_sha256"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            f"R2 matched warm-start {args.arm} stopped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
