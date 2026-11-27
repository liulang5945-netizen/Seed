"""Fixed eight-round isolated train fit, with fresh-process checkpoint verification."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

CPU_START = time.process_time()
WALL_START = time.monotonic()
CHILD_CPU = 0.0

import torch  # noqa: E402

from scripts.training import diagnose_taiji_n1_plan167_train_decoder as diagnostic  # noqa: E402
from scripts.training import summarize_taiji_n1_plan167_train_decoder as analysis  # noqa: E402
from taiji import LanguageAlignmentTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/taiji_n1_plan168_isolated_train_fit_20261025.json"
PROGRESS = REPORT.with_name(REPORT.stem + "_progress.json")
CHECKPOINTS = ROOT / "data/checkpoints/n1_plan168_train_fit_20261025"


def guard():
    if time.process_time() - CPU_START + CHILD_CPU > 1540 or time.monotonic() - WALL_START > 2140:
        raise TimeoutError(
            "N1-168 reached its limit with 60 seconds reserved for partial save/verify"
        )


def selected(corpus):
    previous = diagnostic.frozen.read_json(diagnostic.OUTPUT)
    by_id = {episode.episode_id: episode for episode in corpus.for_split("train")}
    result = [by_id[row["episode_id"]] for row in previous["selection"]]
    if len(result) != 4 or sum(len(ep.target_bytes) for ep in result) != 1165:
        raise ValueError("N1-168 fixed selection differs from N1-167")
    return result


def signature(trainer, episode):
    snapshot = trainer.model.checkpoint()
    diagnostic.prime(trainer, episode, "configured")
    prob = diagnostic.probabilities(trainer, "configured", 0).contiguous().numpy().tobytes()
    result = hashlib.sha256(prob).hexdigest()
    diagnostic.restore(trainer, snapshot)
    return result


def train_one(trainer, episode, epoch):
    if trainer.config.response_repeats != 1:
        raise ValueError("fixed learning protocol requires the existing response_repeats=1")
    metrics = trainer._target_pass(episode, learn=True)
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("training produced a non-finite metric")
    if int(metrics["observations"]) != len(episode.target_bytes):
        raise ValueError("training target byte accounting differs")
    trainer.global_step += int(metrics["observations"])
    trainer.episode_count += 1
    record = {
        "epoch": epoch,
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "target_bytes": len(episode.target_bytes),
        "prefix_bytes": len(episode.prompt_bytes),
        "observations": int(metrics["observations"]),
        "response_accuracy": metrics["accuracy"],
        "response_mean_surprise": metrics["mean_surprise"],
        "protocol": "N1-168",
    }
    trainer.history.append(record)
    return record


def process_cpu(process):
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    api.GetProcessTimes.restype = wintypes.BOOL
    creation, exited, kernel, user = [wintypes.FILETIME() for _ in range(4)]
    if not api.GetProcessTimes(
        int(process._handle),
        ctypes.byref(creation),
        ctypes.byref(exited),
        ctypes.byref(kernel),
        ctypes.byref(user),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        ((kernel.dwHighDateTime << 32) + kernel.dwLowDateTime)
        + ((user.dwHighDateTime << 32) + user.dwLowDateTime)
    ) / 10000000


def verify_saved(path, trainer, episode):
    global CHILD_CPU
    expected = trainer.checkpoint()["checkpoint_digest"]
    expected_signature = signature(trainer, episode)
    started = time.monotonic()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scripts.training.run_taiji_n1_plan168_isolated_train_fit",
            "--verify",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        stdout, stderr = process.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        CHILD_CPU += process_cpu(process)
        raise TimeoutError("fresh-process restore exceeded its fixed 90-second limit") from None
    charged = process_cpu(process)
    CHILD_CPU += charged
    if process.returncode != 0:
        raise ValueError(f"fresh-process restore failed: {stderr.strip()}")
    receipt = json.loads(stdout)
    if (
        receipt["checkpoint_digest"] != expected
        or receipt["first_distribution_sha256"] != expected_signature
    ):
        raise ValueError("checkpoint payload or next-step distribution changed after fresh restore")
    receipt.update(
        process_cpu_seconds_through_exit=charged,
        wall_seconds=time.monotonic() - started,
        path=str(path),
        sha256=diagnostic.frozen.sha256_file(path),
    )
    return receipt


def evaluate(trainer, episodes, epoch):
    snapshot = trainer.model.checkpoint()
    records = []
    for episode in episodes:
        guard()
        diagnostic.restore(trainer, snapshot)
        tf, _ = diagnostic.teacher_forced(trainer, episode, "configured")
        diagnostic.restore(trainer, snapshot)
        raw = trainer._constrained_generate(episode, max_generation_bytes=128)
        text, _, _, _, stop = trainer._generated_text(raw)
        hits, characters = analysis.cjk_accuracy(tf["rows"])
        records.append(
            {
                "epoch": epoch,
                "episode_id": episode.episode_id,
                "tf_top1": tf["top1_raw"],
                "tf_nll_bits": tf["nll_raw_bits"],
                "cjk_correct": hits,
                "cjk_characters": characters,
                "generated_text": text,
                "generated_bytes_hex": raw.hex(),
                "target_prefix_match": diagnostic.prefix_match(raw, episode.target_bytes),
                "stop_reason": stop,
            }
        )
    diagnostic.restore(trainer, snapshot)
    if diagnostic.content_digest(trainer.model.checkpoint()) != diagnostic.content_digest(snapshot):
        raise ValueError("milestone evaluation changed the trained state")
    return records


def parameter_tensors(value, path=""):
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(parameter_tensors(child, f"{path}/{key}"))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            result.update(parameter_tensors(child, f"{path}/{index}"))
    elif isinstance(value, torch.Tensor) and (
        path.endswith(("weight", "weights", "bias", "values", "plan_bridge"))
    ):
        result[path] = value.detach().cpu()
    return result


def parameter_changes(before, after):
    result = {}
    for group in ("fabric", "predictive_context", "predictive_readout", "response_plan_readout"):
        left, right = parameter_tensors(before.get(group)), parameter_tensors(after.get(group))
        if set(left) != set(right):
            raise ValueError("parameter ownership changed during a fixed-architecture fit")
        result[group] = {
            "tensors": len(left),
            "changed_tensors": sum(not torch.equal(left[key], right[key]) for key in left),
            "l2_change": sum(
                float((left[key].double() - right[key].double()).square().sum()) for key in left
            )
            ** 0.5,
        }
    return result


def run():
    torch.set_num_threads(1)
    if REPORT.exists() or CHECKPOINTS.exists():
        raise ValueError("N1-168 artifacts already exist; never overwrite a prior run")
    CHECKPOINTS.mkdir(parents=True)
    report = {
        "schema": "taiji-n1-plan168-isolated-fit-v1",
        "status": "running",
        "training_records": [],
        "milestones": [],
        "restores": {},
        "dev_queries": 0,
        "can_promote": False,
        "source_sha256": {},
    }
    for path in (
        Path(__file__),
        Path(diagnostic.__file__),
        diagnostic.OUTPUT,
        ROOT / "plans/reference/PLAN-N1-168_isolated_train_fit_20261025.md",
        ROOT / "plans/reference/SPEC-N1-168_isolated_train_fit_20261025.md",
    ):
        report["source_sha256"][str(path.relative_to(ROOT))] = diagnostic.frozen.sha256_file(path)
    trainer = None
    parent = None
    parent_sha = None
    try:
        corpus = diagnostic.base._load_assistant_train_corpus()
        episodes = selected(corpus)
        endpoint = next(
            row
            for row in diagnostic.frozen.read_json(diagnostic.READINESS)["endpoints"]
            if row["endpoint_id"] == "full_turn_1m"
        )
        parent = Path(endpoint["path"])
        parent_sha = diagnostic.frozen.sha256_file(parent)
        if parent_sha != "5654b9eac004e98525e55661e6606be8de1f40027bebfa547ae02ee378efa4ca":
            raise ValueError("N1-168 parent is not the preregistered full-turn checkpoint")
        trainer, _ = diagnostic.frozen._load_endpoint_trainer(endpoint, corpus)
        report["parent_code_revision"] = trainer.code_revision
        trainer.code_revision = diagnostic.base._fresh_code_revision()
        report["child_code_revision"] = trainer.code_revision
        before_model = trainer.model.checkpoint()
        initial_episode_count, initial_step = trainer.episode_count, trainer.global_step
        zero = CHECKPOINTS / "zero.pt"
        trainer.save(zero)
        report["restores"]["zero"] = verify_saved(zero, trainer, episodes[0])
        report["milestones"].extend(evaluate(trainer, episodes, 0))
        for epoch in range(1, 9):
            for episode in episodes:
                guard()
                report["training_records"].append(train_one(trainer, episode, epoch))
                PROGRESS.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            trainer._finish_developmental_epoch(epoch)
            if epoch in {1, 2, 4, 8}:
                report["milestones"].extend(evaluate(trainer, episodes, epoch))
        if (
            trainer.global_step - initial_step != 9320
            or trainer.episode_count - initial_episode_count != 32
        ):
            raise ValueError("fixed additional/cumulative training counts do not match")
        report.update(
            status="complete",
            additional_updates=9320,
            cumulative_updates=trainer.global_step,
            additional_episodes=32,
            cumulative_episodes=trainer.episode_count,
        )
        report["parameter_changes"] = parameter_changes(before_model, trainer.model.checkpoint())
    except Exception as exc:
        report.update(
            status="partial" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        if trainer is not None:
            final = CHECKPOINTS / ("final.pt" if report["status"] == "complete" else "partial.pt")
            trainer.save(final)
            try:
                report["restores"]["terminal"] = verify_saved(final, trainer, episodes[0])
            except Exception as exc:
                report.update(status="failed", restore_error=str(exc))
            report["actual_episode_count"] = trainer.episode_count
            report["actual_target_byte_updates"] = trainer.global_step
        if parent is not None:
            report["parent_sha256"] = parent_sha
            report["parent_unchanged"] = diagnostic.frozen.sha256_file(parent) == parent_sha
            if not report["parent_unchanged"]:
                report["status"] = "failed"
        report.update(
            parent_cpu_seconds=time.process_time() - CPU_START,
            child_cpu_seconds=CHILD_CPU,
            cpu_seconds=time.process_time() - CPU_START + CHILD_CPU,
            wall_seconds=time.monotonic() - WALL_START,
        )
        with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "additional_updates",
                    "actual_target_byte_updates",
                    "cpu_seconds",
                    "error",
                    "restore_error",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.run:
        return run()
    torch.set_num_threads(1)
    corpus = diagnostic.base._load_assistant_train_corpus()
    payload = torch.load(args.verify, map_location="cpu", weights_only=True)
    trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
    if trainer.checkpoint()["checkpoint_digest"] != payload["checkpoint_digest"]:
        raise ValueError("fresh-process full trainer digest differs")
    print(
        json.dumps(
            {
                "checkpoint_digest": payload["checkpoint_digest"],
                "first_distribution_sha256": signature(trainer, selected(corpus)[0]),
                "global_step": trainer.global_step,
                "episode_count": trainer.episode_count,
                "model_digest": diagnostic.content_digest(trainer.model.checkpoint()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
