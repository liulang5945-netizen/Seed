"""Profile the live paired graph and price real Adam on disjoint shadow ownership."""

# ruff: noqa: E402 -- retain startup costs
from __future__ import annotations

import cProfile
import hashlib
import json
import pstats
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.evidence_a_learner_22_5 import load_learner
from scripts.training.paired_episode_22_5 import episode_loss
from taiji.u2_history import atomic_load, detached_tree, value_digest


def shadow_price(model, optimizer):
    original = list(model.parameters())
    temporary = [torch.nn.Parameter(p.detach().clone()) for p in original]
    for source, target in zip(original, temporary, strict=True):
        target.grad = None if source.grad is None else source.grad.detach().clone()
    live_storage = {p.data_ptr() for p in original}
    assert all(p.data_ptr() not in live_storage for p in temporary)
    shadow = torch.optim.Adam(temporary, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    start = time.process_time()
    shadow.step()
    fresh_cost = time.process_time() - start
    shadow.load_state_dict(detached_tree(optimizer.state_dict()))
    live_moments = {
        value.data_ptr()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    }
    assert all(
        value.data_ptr() not in live_moments
        for state in shadow.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    for _ in range(2):
        shadow.step()
    samples = []
    for _ in range(8):
        start = time.process_time()
        for _ in range(4):
            shadow.step()
        samples.append((time.process_time() - start) / 4)
    return {
        "fresh_step_cpu": fresh_cost,
        "inherited_moment_step_cpu_samples": samples,
        "median_inherited_step_cpu": statistics.median(samples),
        "shadow_optimizer_steps": 35,
        "live_parameters_disjoint": True,
        "live_moments_disjoint": True,
    }


def profile():
    torch.set_num_threads(1)
    output = ROOT / "reports/taiji_u2_paired_compute_profile_20261113.json"
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_5_3_compute_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    design = json.loads(
        (ROOT / "plans/manifests/taiji_u2_p3_5_2_learning_contract_20261113.json").read_text(
            encoding="utf-8"
        )
    )
    data_manifest = json.loads((ROOT / design["data_manifest"]).read_text(encoding="utf-8"))
    data = atomic_load(data_manifest["data_path"])
    assert value_digest(data) == data_manifest["data_digest"]
    rows = {}
    for name, parent in design["parents"].items():
        assert (
            hashlib.sha256(Path(parent["checkpoint"]).read_bytes()).hexdigest() == parent["sha256"]
        )
        model, optimizer, _ = load_learner(parent["checkpoint"])
        before = value_digest(model.state_dict()), value_digest(optimizer.state_dict())
        samples = []
        for index in contract["profile_episode_indices"]:
            model.zero_grad(set_to_none=True)
            start = time.process_time()
            loss = episode_loss(model, data["train"], data["train"]["episodes"][index])
            loss.backward()
            samples.append(time.process_time() - start)
        counts: Counter[str] = Counter()
        handles = []

        def observer(module_name, counter=counts):
            def record(_module, _inputs, _output):
                counter[module_name] += 1

            return record

        for module_name, module in model.named_modules():
            if module_name in {
                "encoder.rgb",
                "convolutions.0",
                "evidence_reader.key",
                "evidence_reader.value",
                "joint_readout",
                "byte_head",
                "frontend.convolutions.0",
                "frontend.encoder.rgb",
            }:
                handles.append(module.register_forward_hook(observer(module_name)))
        profiler = cProfile.Profile()
        try:
            model.zero_grad(set_to_none=True)
            profiler.enable()
            episode_loss(model, data["train"], data["train"]["episodes"][0]).backward()
            profiler.disable()
        finally:
            for handle in handles:
                handle.remove()
        stats = cast(Any, pstats.Stats(profiler)).stats
        selected = sorted(
            (row[3], file, line, function, row[1], row[2])
            for (file, line, function), row in stats.items()
            if "u2_" in file or "paired_" in file
        )
        adam = shadow_price(model, optimizer)
        assert before == (value_digest(model.state_dict()), value_digest(optimizer.state_dict()))
        rows[name] = {
            "forward_backward_cpu_samples": samples,
            "median_forward_backward_cpu": statistics.median(samples),
            "module_calls_per_episode": dict(counts),
            "profile_cumulative_wall_top": [
                {
                    "function": function,
                    "file": file,
                    "line": line,
                    "calls": calls,
                    "cumulative_wall": cumulative,
                    "self_wall": own,
                }
                for cumulative, file, line, function, calls, own in selected[-15:][::-1]
            ],
            "adam": adam,
            "learner_parameters_and_optimizer_unchanged": True,
        }
    report = {
        "schema": "taiji-u2-paired-compute-profile-v1",
        "scope": "Live graph forward/backward; module/cProfile counters are intrusive and used for diagnosis. Real Adam steps act on disjoint shadow tensors only. No live learner update or task accuracy evaluation.",
        "profiles": rows,
        "optimizer_steps": sum(row["adam"]["shadow_optimizer_steps"] for row in rows.values()),
        "shadow_optimizer_steps": 70,
        "learner_optimizer_steps": 0,
        "check_reads": 0,
        "final_reads": 0,
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
        "source_sha256": hashlib.sha256(
            Path(__file__).read_text(encoding="utf-8-sig").encode()
        ).hexdigest(),
    }
    assert (
        report["cpu_seconds"] <= contract["cpu_cap"]
        and report["wall_seconds"] <= contract["wall_cap"]
    )
    write_report(output, report)
    print(
        json.dumps(
            {
                name: {
                    "module_calls": row["module_calls_per_episode"],
                    "forward_backward_cpu": row["median_forward_backward_cpu"],
                    "adam_cpu": row["adam"]["median_inherited_step_cpu"],
                }
                for name, row in rows.items()
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    profile()
