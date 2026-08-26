"""Profile native sparse Taiji neuron/runtime paths on CPU and CUDA.

This is a runtime baseline, not a training job.  It measures the existing
identity-preserving sparse region/network implementation, verifies that its
checkpoint payload survives a device round trip, and compares one identical
initial state across CPU/CUDA when CUDA is available.  The result is intended
to decide whether a fused or sparse kernel is justified by measured hotspots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taiji import AdaptiveNeuronNetwork, AdaptiveNeuronRegion  # noqa: E402

PROFILE_FORMAT = "taiji-native-runtime-profile-v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "taiji_native_runtime_profile_20260826.json"
DEFAULT_INPUT_DIM = 64
DEFAULT_UNIT_COUNT = 48
DEFAULT_FAN_IN = 12
DEFAULT_WARMUP_TICKS = 6
DEFAULT_TIMED_TICKS = 48
DEFAULT_PROFILE_TICKS = 8
OUTPUT_TOLERANCE = 1e-5


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _make_network() -> AdaptiveNeuronNetwork:
    regions = tuple(
        AdaptiveNeuronRegion(
            region_id=region_id,
            input_dim=DEFAULT_INPUT_DIM,
            unit_ids=tuple(f"{region_id}.u{index}" for index in range(DEFAULT_UNIT_COUNT)),
            fan_in=DEFAULT_FAN_IN,
            generator=_generator(71 + index),
        )
        for index, region_id in enumerate(("source", "relay", "target"))
    )
    network = AdaptiveNeuronNetwork(
        regions,
        execution_order=("source", "relay", "target"),
    )
    for index, (source_id, target_id) in enumerate((("source", "relay"), ("relay", "target"))):
        proposal = network.propose_connection_add(
            source_region_id=source_id,
            target_region_id=target_id,
            evidence_ids=(f"profile:route:{index}",),
            fan_in=DEFAULT_FAN_IN,
        )
        network.apply_topology_proposal(proposal, generator=_generator(101 + index))
    return network


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _payload_equal(item_left, item_right)
            for item_left, item_right in zip(left, right, strict=True)
        )
    return left == right


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    source = torch.linspace(
        -1.0,
        1.0,
        DEFAULT_INPUT_DIM,
        device=device,
        dtype=torch.float32,
    )
    return {
        "source": source,
        "relay": torch.zeros(DEFAULT_INPUT_DIM, device=device),
        "target": torch.zeros(DEFAULT_INPUT_DIM, device=device),
    }


def _region_runner(
    network: AdaptiveNeuronNetwork,
) -> tuple[Callable[[], torch.Tensor], torch.Tensor]:
    region = network.regions[0]
    input_activity = _inputs(region.device)["source"]

    def run() -> torch.Tensor:
        return region.step(input_activity)

    return run, input_activity


def _network_runner(
    network: AdaptiveNeuronNetwork,
) -> tuple[Callable[[], dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    external_inputs = _inputs(network.regions[0].device)
    connection_ids = network.connection_ids

    def run() -> dict[str, torch.Tensor]:
        return network.step(
            external_inputs,
            connection_ids=connection_ids,
            max_connections=len(connection_ids),
        )

    return run, external_inputs


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_seconds(
    runner: Callable[[], Any],
    *,
    ticks: int,
    device: torch.device,
) -> float:
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(int(ticks)):
        runner()
    _synchronize(device)
    return max(time.perf_counter() - start, 1e-12)


def _profile(
    runner: Callable[[], Any],
    *,
    device: torch.device,
    profile_ticks: int,
) -> list[dict[str, Any]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        for _ in range(int(profile_ticks)):
            runner()
    events = sorted(
        profile.key_averages(),
        key=lambda item: float(getattr(item, "self_cpu_time_total", 0.0)),
        reverse=True,
    )
    top_ops: list[dict[str, Any]] = []
    for event in events[:10]:
        top_ops.append(
            {
                "name": str(event.key),
                "calls": int(event.count),
                "self_cpu_time_us": round(float(event.self_cpu_time_total), 3),
                "self_cuda_time_us": round(
                    float(getattr(event, "self_device_time_total", 0.0)),
                    3,
                ),
            }
        )
    return top_ops


def _benchmark_device(
    network: AdaptiveNeuronNetwork,
    *,
    warmup_ticks: int,
    timed_ticks: int,
    profile_ticks: int,
) -> dict[str, Any]:
    device = network.regions[0].device
    region_runner, _ = _region_runner(network)
    network_runner, _ = _network_runner(network)
    with torch.inference_mode():
        for _ in range(int(warmup_ticks)):
            region_runner()
            network_runner()
        region_seconds = _elapsed_seconds(
            region_runner,
            ticks=timed_ticks,
            device=device,
        )
        network_seconds = _elapsed_seconds(
            network_runner,
            ticks=timed_ticks,
            device=device,
        )
        top_region_ops = _profile(
            region_runner,
            device=device,
            profile_ticks=profile_ticks,
        )
        top_network_ops = _profile(
            network_runner,
            device=device,
            profile_ticks=profile_ticks,
        )
    return {
        "device": str(device),
        "region_ticks": int(timed_ticks),
        "network_ticks": int(timed_ticks),
        "region_seconds": round(region_seconds, 6),
        "network_seconds": round(network_seconds, 6),
        "region_ticks_per_second": round(float(timed_ticks) / region_seconds, 3),
        "network_ticks_per_second": round(float(timed_ticks) / network_seconds, 3),
        "region_tick_ms": round(region_seconds * 1000.0 / float(timed_ticks), 4),
        "network_tick_ms": round(network_seconds * 1000.0 / float(timed_ticks), 4),
        "top_region_ops": top_region_ops,
        "top_network_ops": top_network_ops,
    }


def _max_output_error(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    errors = [
        (left[region_id].detach().cpu() - right[region_id].detach().cpu()).abs().max()
        for region_id in left
    ]
    return 0.0 if not errors else float(torch.stack(errors).max().item())


def _checkpoint_gate(
    payload: dict[str, Any],
    *,
    cuda_available: bool,
) -> dict[str, Any]:
    cpu_restored = AdaptiveNeuronNetwork.from_payload(
        payload,
        generator=_generator(999),
        device="cpu",
    )
    cpu_roundtrip = _payload_equal(payload, cpu_restored.to_payload())
    cpu_reference = AdaptiveNeuronNetwork.from_payload(
        payload,
        generator=_generator(1000),
        device="cpu",
    )
    cpu_runner, _ = _network_runner(cpu_reference)
    with torch.inference_mode():
        cpu_runner()
        continued_reference = cpu_runner()
    cpu_continuation = AdaptiveNeuronNetwork.from_payload(
        payload,
        generator=_generator(1001),
        device="cpu",
    )
    restored_runner, _ = _network_runner(cpu_continuation)
    with torch.inference_mode():
        restored_runner()
        restored_continuation = restored_runner()
    continuation_error = _max_output_error(continued_reference, restored_continuation)
    result: dict[str, Any] = {
        "cpu_roundtrip": cpu_roundtrip,
        "cpu_continuation_max_abs_error": continuation_error,
        "cpu_continuation": continuation_error <= OUTPUT_TOLERANCE,
        "cross_device_roundtrip": None,
        "cross_device_continuation_max_abs_error": None,
        "cross_device_continuation": None,
    }
    if cuda_available:
        cuda_network = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=_generator(1002),
            device="cuda",
        )
        cuda_payload = cuda_network.to_payload()
        cross_device_roundtrip = _payload_equal(payload, cuda_payload)
        cuda_runner, _ = _network_runner(cuda_network)
        cuda_reference = AdaptiveNeuronNetwork.from_payload(
            payload,
            generator=_generator(1003),
            device="cuda",
        )
        cuda_reference_runner, _ = _network_runner(cuda_reference)
        with torch.inference_mode():
            cpu_runner_for_device = AdaptiveNeuronNetwork.from_payload(
                payload,
                generator=_generator(1004),
                device="cpu",
            )
            cpu_device_runner, _ = _network_runner(cpu_runner_for_device)
            cpu_output = cpu_device_runner()
            cuda_output = cuda_runner()
            cuda_reference_runner()
            cuda_reference_next = cuda_reference_runner()
            cpu_reference_next = cpu_device_runner()
        output_error = _max_output_error(cpu_output, cuda_output)
        continuation_error = _max_output_error(cpu_reference_next, cuda_reference_next)
        result.update(
            {
                "cross_device_roundtrip": cross_device_roundtrip,
                "cross_device_output_max_abs_error": output_error,
                "cross_device_output": output_error <= OUTPUT_TOLERANCE,
                "cross_device_continuation_max_abs_error": continuation_error,
                "cross_device_continuation": continuation_error <= OUTPUT_TOLERANCE,
            }
        )
    return result


def evaluate(
    *,
    warmup_ticks: int = DEFAULT_WARMUP_TICKS,
    timed_ticks: int = DEFAULT_TIMED_TICKS,
    profile_ticks: int = DEFAULT_PROFILE_TICKS,
) -> dict[str, Any]:
    payload = _make_network().to_payload()
    cuda_available = bool(torch.cuda.is_available())
    cpu_network = AdaptiveNeuronNetwork.from_payload(
        payload,
        generator=_generator(2001),
        device="cpu",
    )
    cpu_result = _benchmark_device(
        cpu_network,
        warmup_ticks=warmup_ticks,
        timed_ticks=timed_ticks,
        profile_ticks=profile_ticks,
    )
    cuda_result: dict[str, Any] | None = None
    cuda_error: str | None = None
    if cuda_available:
        try:
            cuda_network = AdaptiveNeuronNetwork.from_payload(
                payload,
                generator=_generator(2002),
                device="cuda",
            )
            cuda_result = _benchmark_device(
                cuda_network,
                warmup_ticks=warmup_ticks,
                timed_ticks=timed_ticks,
                profile_ticks=profile_ticks,
            )
        except Exception as exc:  # pragma: no cover - hardware-specific failure
            cuda_error = f"{type(exc).__name__}: {exc}"
    checkpoint = _checkpoint_gate(payload, cuda_available=cuda_available and cuda_error is None)
    checks = {
        "cpu_profile": bool(
            cpu_result["region_ticks_per_second"] > 0
            and cpu_result["network_ticks_per_second"] > 0
            and cpu_result["top_region_ops"]
            and cpu_result["top_network_ops"]
        ),
        "cpu_checkpoint_roundtrip": bool(checkpoint["cpu_roundtrip"]),
        "cpu_checkpoint_continuation": bool(checkpoint["cpu_continuation"]),
        "cuda_status_explicit": not cuda_available or cuda_result is not None,
        "cuda_profile": not cuda_available
        or bool(
            cuda_result is not None
            and cuda_result["region_ticks_per_second"] > 0
            and cuda_result["network_ticks_per_second"] > 0
        ),
        "cross_device_checkpoint": not cuda_available or bool(checkpoint["cross_device_roundtrip"]),
        "cross_device_output": not cuda_available or bool(checkpoint.get("cross_device_output")),
        "cross_device_continuation": not cuda_available
        or bool(checkpoint.get("cross_device_continuation")),
    }
    return {
        "format": PROFILE_FORMAT,
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "cuda_device_name": (torch.cuda.get_device_name(0) if cuda_available else None),
        },
        "workload": {
            "input_dim": DEFAULT_INPUT_DIM,
            "unit_count_per_region": DEFAULT_UNIT_COUNT,
            "fan_in": DEFAULT_FAN_IN,
            "regions": ["source", "relay", "target"],
            "connections": ["connection:source->relay", "connection:relay->target"],
            "warmup_ticks": int(warmup_ticks),
            "timed_ticks": int(timed_ticks),
            "profile_ticks": int(profile_ticks),
        },
        "implementation": {
            "device_transfer_guard": True,
            "cached_norm_constants": True,
            "reused_network_scratch_vectors": True,
            "custom_cuda_kernel": False,
        },
        "cpu": cpu_result,
        "cuda": cuda_result,
        "cuda_error": cuda_error,
        "checkpoint": checkpoint,
        "gate": {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "output_tolerance": OUTPUT_TOLERANCE,
            "criterion": (
                "native sparse region and network runtime must expose CPU hotspots, "
                "save and restore exact topology/state, and preserve numerical outputs "
                "across CUDA when CUDA is available; unavailable CUDA is reported rather "
                "than being claimed"
            ),
            "boundary": (
                "This profile establishes a measured baseline only. It does not justify "
                "a fused or custom sparse kernel until the recorded hotspots and a CUDA "
                "capable host are reviewed."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS)
    parser.add_argument("--timed-ticks", type=int, default=DEFAULT_TIMED_TICKS)
    parser.add_argument("--profile-ticks", type=int, default=DEFAULT_PROFILE_TICKS)
    args = parser.parse_args()
    report = evaluate(
        warmup_ticks=args.warmup_ticks,
        timed_ticks=args.timed_ticks,
        profile_ticks=args.profile_ticks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
