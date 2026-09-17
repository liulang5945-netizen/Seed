"""H3.7B dev validation package: bounded two-arm repair-effectiveness check.

Per the frozen dev validation package (M5_R2_H3_7B_CONSISTENCY_REPAIR_20260917.md):
two epochs per arm, dev evaluation at zero-step and post-training from saved
checkpoint probes, final unread.  Engineering verification of the repair
chain's dev-side behaviour, not a capability or generalization claim.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

from taiji import (
    LanguageAlignmentConfig,
    LanguageAlignmentTrainer,
    LanguageEpisodeCorpus,
    Taiji,
    TaijiConfig,
    checkpoint_roundtrip_preflight,
    factorized_response_plan_preflight,
)
from taiji.response_plan_target import (
    ByteAlignedResponsePlanTargetEncoder,
    FactorizedResponsePlanTargetEncoder,
)

DATASET = Path("tests/fixtures/r2_h3_5a_response_plan_v3.jsonl")
DIGEST = "0bc5b5540d4b622f907942f42d7b809e825932e50c6e6c505a3cd091edabd691"
EPOCHS = 2
EXPECTED_EPISODES = 12 * EPOCHS
EXPECTED_UPDATES = 537 * EPOCHS
REPORT_FORMAT = "taiji-r2-h3-7b-dev-validation-v1"


def assert_finite(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("non-finite checkpoint tensor")
    elif isinstance(value, Mapping):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)


def dev_summary(evaluation: dict) -> dict:
    return {
        "teacher_forced_mean_surprise": round(float(evaluation["teacher_forced_mean_surprise"]), 6),
        "teacher_forced_accuracy": round(float(evaluation["teacher_forced_accuracy"]), 6),
        "exact_response_rate": round(float(evaluation["exact_response_rate"]), 6),
        "utf8_valid_rate": round(float(evaluation["utf8_valid_rate"]), 6),
        "response_boundary_rate": round(float(evaluation["response_boundary_rate"]), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("legacy_target", "byte_aligned"), required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    corpus = LanguageEpisodeCorpus.from_jsonl([DATASET])
    if corpus.digest != DIGEST:
        raise RuntimeError("frozen corpus digest mismatch")
    if args.verify:
        payload = torch.load(args.verify, map_location="cpu", weights_only=False)
        trainer = LanguageAlignmentTrainer.from_checkpoint(payload, corpus)
        if trainer.checkpoint()["checkpoint_digest"] != payload["checkpoint_digest"]:
            raise RuntimeError("fresh-process checkpoint restore mismatch")
        trainer.train(epochs=1, max_episodes=1)
        print(
            json.dumps(
                {
                    "restored": payload["checkpoint_digest"],
                    "continued": trainer.checkpoint()["checkpoint_digest"],
                }
            )
        )
        return 0

    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 512 * 1024**2:
        raise RuntimeError("insufficient checkpoint disk reserve")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    model = Taiji(
        TaijiConfig.capacity_profile(
            300_000,
            seed=20260917,
            additional_predictive_readouts=1,
            response_plan_width=48,
        )
    )
    model.enable_response_plan_readout(plan_width=48, variant="factorized_v1")
    aligned = args.arm == "byte_aligned"
    encoder_type = (
        ByteAlignedResponsePlanTargetEncoder if aligned else FactorizedResponsePlanTargetEncoder
    )
    trainer = LanguageAlignmentTrainer(
        model,
        corpus,
        code_revision=revision,
        config=LanguageAlignmentConfig(
            response_plan_readout=True,
            response_plan_width=48,
            response_plan_variant="factorized_v1",
            response_plan_target_geometry=(
                "h3_7b_byte_aligned_chunks" if aligned else "h3_7_factorized_response_chunks"
            ),
        ),
        response_plan_target_encoder=encoder_type.fit(model, corpus),
    )
    if model.parameter_count() > 300_000:
        raise RuntimeError("parameter budget exceeded")

    def fresh_check(path: Path) -> dict:
        output = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "scripts.training.validate_taiji_r2_h3_7b_dev",
                "--output",
                str(args.output),
                "--arm",
                args.arm,
                "--verify",
                str(path),
            ],
            text=True,
            timeout=180,
        )
        result = json.loads(output)
        probe = LanguageAlignmentTrainer.from_checkpoint(trainer.checkpoint(), corpus)
        probe.train(epochs=1, max_episodes=1)
        if result["continued"] != probe.checkpoint()["checkpoint_digest"]:
            raise RuntimeError("fresh-process continuation differs")
        return result

    report = {
        "format": REPORT_FORMAT,
        "version": 1,
        "arm": args.arm,
        "code_revision": revision,
        "corpus_digest": corpus.digest,
        "epochs": EPOCHS,
        "status": "preflight",
        "final_evaluated": False,
        "effective_parameters": model.parameter_count(),
    }
    report_path = args.output / "report.json"

    def persist() -> None:
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    persist()
    try:
        report["checkpoint_preflight"] = checkpoint_roundtrip_preflight(
            trainer, directory=args.output / "preflight"
        )
        report["factorized_preflight"] = factorized_response_plan_preflight(trainer)
        zero = trainer.save(args.output / "zero.pt")
        report["zero_fresh_process"] = fresh_check(zero)
        zero_probe = LanguageAlignmentTrainer.from_checkpoint(
            torch.load(zero, map_location="cpu", weights_only=False), corpus
        )
        report["dev_zero"] = dev_summary(zero_probe.evaluate("dev"))
        report["status"] = "training"
        persist()
        report["training"] = trainer.train(epochs=EPOCHS, max_episodes=12)
        if trainer.episode_count != EXPECTED_EPISODES or trainer.global_step != EXPECTED_UPDATES:
            raise RuntimeError("frozen update budget mismatch")
        assert_finite(trainer.checkpoint())
        child = trainer.save(args.output / "checkpoint.pt")
        report["child_fresh_process"] = fresh_check(child)
        child_probe = LanguageAlignmentTrainer.from_checkpoint(
            torch.load(child, map_location="cpu", weights_only=False), corpus
        )
        report["dev_trained"] = dev_summary(child_probe.evaluate("dev"))
        report["dev_transfer"] = {
            "surprise_delta": round(
                report["dev_trained"]["teacher_forced_mean_surprise"]
                - report["dev_zero"]["teacher_forced_mean_surprise"],
                6,
            ),
            "accuracy_delta": round(
                report["dev_trained"]["teacher_forced_accuracy"]
                - report["dev_zero"]["teacher_forced_accuracy"],
                6,
            ),
        }
        report["checkpoint"] = str(child.resolve())
        report["status"] = "completed_dev_evaluation"
        persist()
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "status": report["status"],
                    "dev_transfer": report["dev_transfer"],
                }
            )
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        persist()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
