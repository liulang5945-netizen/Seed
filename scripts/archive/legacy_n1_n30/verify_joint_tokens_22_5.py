"""Local joint-token equations, mixed dependence and full directional derivative."""

# ruff: noqa: E402 -- include startup in the local verification cost
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_history import value_digest
from taiji.u2_joint_tokens import JointTokenFusion

CONTRACT = ROOT / "plans/manifests/taiji_u2_joint_tokens_local_contract_20261113.json"
REPORT = ROOT / "reports/taiji_u2_joint_tokens_local_20261113.json"


def verify():
    if REPORT.exists():
        raise FileExistsError(REPORT)
    torch.set_num_threads(1)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    for name, digest in contract["source_pins"].items():
        assert (
            hashlib.sha256((ROOT / name).read_text(encoding="utf-8-sig").encode()).hexdigest()
            == digest
        )
    reports = {}
    with StepGuard() as guard:
        for precision in ("float32", "float64"):
            dtype = getattr(torch, precision)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(contract["seed"])
                model = JointTokenFusion(8).to(dtype=dtype)
            generator = torch.Generator().manual_seed(37)
            inputs = [
                torch.randn(shape, generator=generator, dtype=torch.float64)
                .to(dtype)
                .requires_grad_()
                for shape in ((4, 8), (5, 8), (5, 8), (2, 3, 8))
            ]
            query, evidence, metadata, h = inputs
            valid = torch.tensor([True, False, True, True, True])
            signature = value_digest(model.state_dict())
            feature = model(query, evidence, metadata, valid, h)
            target = torch.arange(1, 17, dtype=dtype).reshape(2, 8) / 16
            loss = (feature * target).sum()
            named = list(model.named_parameters())
            gradients = torch.autograd.grad(loss, [*model.parameters(), *inputs])
            assert all(
                bool(torch.isfinite(value).all()) and float(value.norm()) > 0 for value in gradients
            )
            length = math.sqrt(
                sum(float(value.detach().double().square().sum()) for value in gradients)
            )
            directions = [value.detach() / length for value in gradients]
            step = contract["difference_step"][precision]
            values = []
            with torch.no_grad():
                for sign in (-1, 1):
                    shifted = copy.deepcopy(model)
                    for parameter, direction in zip(
                        shifted.parameters(), directions[: len(named)], strict=True
                    ):
                        parameter.add_(direction, alpha=sign * step)
                    moved = [
                        value.detach() + sign * step * direction
                        for value, direction in zip(inputs, directions[len(named) :], strict=True)
                    ]
                    result = shifted(moved[0], moved[1], moved[2], valid, moved[3])
                    values.append(float((result * target).sum()))
                finite_difference = (values[1] - values[0]) / (2 * step)
                tolerance = contract["derivative_atol"][precision] + contract[
                    "derivative_rtol"
                ] * abs(length)
                assert abs(finite_difference - length) < tolerance
                assert abs(finite_difference + length) > tolerance
                mixed = (
                    model(query, evidence, metadata, valid, h)
                    - model(-query, evidence, metadata, valid, h)
                    - model(query, -evidence, metadata, valid, h)
                    + model(-query, -evidence, metadata, valid, h)
                )
                assert float(mixed.norm()) > contract["minimum_mixed_norm"]
            assert value_digest(model.state_dict()) == signature
            assert float(feature.detach().square().mean(-1).max()) < 1
            reports[precision] = {
                "parameters": sum(value.numel() for value in model.parameters()),
                "parameter_tensors": len(named),
                "parameter_gradient_norms": {
                    name: float(value.norm())
                    for (name, _), value in zip(named, gradients[: len(named)], strict=True)
                },
                "input_gradient_norms": [float(value.norm()) for value in gradients[len(named) :]],
                "analytic_directional_derivative": length,
                "finite_directional_derivative": finite_difference,
                "sign_reversed_control_rejected": True,
                "mixed_norm": float(mixed.norm()),
                "maximum_output_rms_squared": float(feature.detach().square().mean(-1).max()),
                "parameter_digest": signature,
            }
    assert (
        time.process_time() < contract["cpu_cap"]
        and time.monotonic() - STARTED < contract["wall_cap"]
    )
    report = {
        "schema": "taiji-u2-joint-tokens-local-v1",
        "verdict": "passed_local_joint_token_operator",
        "precisions": reports,
        "optimizer_steps": guard.calls,
        "subject_integration_verified": False,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    write_report(REPORT, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": report["cpu_seconds"],
                "mixed_norms": {name: value["mixed_norm"] for name, value in reports.items()},
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    verify()
