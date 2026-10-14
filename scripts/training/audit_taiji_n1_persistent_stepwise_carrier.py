"""Independently audit the N1 persistent stepwise carrier checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji.model import Taiji  # noqa: E402


def _load_weights_only(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"weights-only checkpoint load failed: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint envelope is not a dictionary")
    return value


def _carrier_from_checkpoint(payload: dict[str, Any], location: str) -> Any:
    if location == "native":
        envelope = payload.get("taiji")
        if not isinstance(envelope, dict):
            raise RuntimeError("Seed checkpoint is missing its native Taiji envelope")
        kernel = envelope.get("kernel")
    else:
        kernel = payload.get("substrate")
    if not isinstance(kernel, dict):
        raise RuntimeError(f"Seed checkpoint is missing its {location} kernel")
    return kernel.get(Taiji.PERSISTENT_STEPWISE_CARRIER_KEY)


def _payload_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("carrier payload is missing its episode list")
    manifest = []
    for episode in episodes:
        vectors = episode.get("vectors")
        if not isinstance(vectors, torch.Tensor):
            raise RuntimeError("carrier payload contains a non-tensor episode")
        manifest.append(
            {
                "episode_id": str(episode["episode_id"]),
                "serial": int(episode["serial"]),
                "source": dict(episode["source"]),
                "steps": int(vectors.shape[0]),
                "vectors_sha256": hashlib.sha256(vectors.numpy().tobytes()).hexdigest(),
            }
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--expect-episodes", type=int, default=None)
    parser.add_argument("--expect-context-dim", type=int, default=None)
    parser.add_argument("--expect-legacy-absent", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if args.expect_legacy_absent and args.report:
        parser.error("--expect-legacy-absent cannot be combined with --report")
    if not args.expect_legacy_absent and (
        args.report is None
        or args.expect_episodes is None
        or args.expect_context_dim is None
    ):
        parser.error("open-carrier audit requires --report, --expect-episodes, and --expect-context-dim")

    saved = _load_weights_only(checkpoint_path)
    native_carrier = _carrier_from_checkpoint(saved, "native")
    compatibility_carrier = _carrier_from_checkpoint(saved, "compatibility")
    if args.expect_legacy_absent:
        if native_carrier is not None or compatibility_carrier is not None:
            raise RuntimeError("legacy checkpoint unexpectedly contains a carrier payload")
        del saved
        gc.collect()
        from api.seed_runtime import SeedRuntime

        runtime = SeedRuntime.load(checkpoint_path)
        substrate = runtime.model.substrate
        if substrate.persistent_stepwise_carrier:
            raise RuntimeError("legacy checkpoint restored recording enabled")
        if substrate._persistent_stepwise_carrier.episode_count != 0:
            raise RuntimeError("legacy checkpoint did not restore an empty carrier")
        print(json.dumps({"legacy_carrier_absent": True, "recording_enabled": False}))
        return 0

    if native_carrier is None or compatibility_carrier is None:
        raise RuntimeError("saved Seed checkpoint is missing one carrier payload")
    if _payload_manifest(native_carrier) != _payload_manifest(compatibility_carrier):
        raise RuntimeError("native and compatibility carrier payloads disagree")
    del native_carrier, compatibility_carrier, saved
    gc.collect()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    carrier_report = report.get("persistent_stepwise_carrier")
    if not isinstance(carrier_report, dict) or not carrier_report.get("enabled"):
        raise RuntimeError("probe report does not identify an enabled persistent carrier")
    if int(carrier_report.get("episodes", -1)) != args.expect_episodes:
        raise RuntimeError("probe report episode count does not match the expected count")
    if int(carrier_report.get("context_dim", -1)) != args.expect_context_dim:
        raise RuntimeError("probe report context dimension does not match the expected value")
    expected_manifest = carrier_report.get("episode_manifest")
    if not isinstance(expected_manifest, list) or len(expected_manifest) != args.expect_episodes:
        raise RuntimeError("probe report is missing its complete episode manifest")
    if carrier_report.get("model_parameter_digest_before") != carrier_report.get(
        "model_parameter_digest_after"
    ):
        raise RuntimeError("probe report shows model parameter changes during the carrier face")
    if report.get("instrument_guard", {}).get("persistent_carrier_weights_unchanged") is not True:
        raise RuntimeError("probe report does not attest unchanged model parameters")

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint_path)
    substrate = runtime.model.substrate
    carrier = substrate._persistent_stepwise_carrier
    if substrate.persistent_stepwise_carrier:
        raise RuntimeError("restored carrier must leave recording disabled")
    if carrier.context_dim != args.expect_context_dim:
        raise RuntimeError("restored carrier context dimension does not match")
    if carrier.episode_count != args.expect_episodes:
        raise RuntimeError("restored carrier episode count does not match")
    actual_manifest = carrier.episode_manifest()
    if actual_manifest != expected_manifest:
        raise RuntimeError("restored episode ids, source metadata, lengths, or vector contents differ")
    for expected in expected_manifest:
        episode_id = str(expected["episode_id"])
        vectors = substrate.read_persistent_stepwise_episode(episode_id)
        if vectors.dtype != torch.float32:
            raise RuntimeError(f"episode {episode_id} did not restore as float32")
        if tuple(vectors.shape) != (int(expected["steps"]), args.expect_context_dim):
            raise RuntimeError(f"episode {episode_id} restored with the wrong vector shape")
        if not torch.isfinite(vectors).all().item():
            raise RuntimeError(f"episode {episode_id} contains non-finite vectors")
    missing = substrate.read_persistent_stepwise_episode("missing-episode-must-not-fallback")
    if tuple(missing.shape) != (0, args.expect_context_dim):
        raise RuntimeError("missing episode lookup did not return an empty trace")

    first_nonempty = next(
        (substrate.read_persistent_stepwise_episode(item["episode_id"]) for item in expected_manifest
         if int(item["steps"]) > 0),
        None,
    )
    feedback = (
        torch.zeros(args.expect_context_dim, dtype=torch.float32)
        if first_nonempty is None
        else first_nonempty[0]
    )
    substrate.fabric.step(
        substrate.sensor.encode(0),
        substrate._state.regions,
        learn=False,
        episodic_feedback=feedback,
        restructure=False,
        adapt_homeostasis=False,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "episodes": carrier.episode_count,
                "steps": carrier.total_steps,
                "context_dim": carrier.context_dim,
                "raw_vector_bytes": carrier.total_steps * carrier.context_dim * 4,
                "recording_enabled_after_restore": substrate.persistent_stepwise_carrier,
                "missing_episode_returns_empty": tuple(missing.shape)
                == (0, args.expect_context_dim),
                "feedback_shape_accepted": True,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
