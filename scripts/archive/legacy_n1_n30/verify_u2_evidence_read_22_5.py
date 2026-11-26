"""Local reader checks; this verifier does not admit a full subject candidate."""

# ruff: noqa: E402 -- include startup in the measured interval
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training.check_report_run_conditions import write_report
from scripts.training.verify_taiji_u2_metadata_groups_22_4 import StepGuard
from taiji.u2_evidence_read import QueryEvidenceReader
from taiji.u2_history import assert_values_equal, value_digest


def verify():
    report_path = ROOT / "reports/taiji_u2_evidence_reader_local_20261112.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    torch.set_num_threads(1)
    records = {}
    with StepGuard() as blocked:
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(20261112)
            reader = QueryEvidenceReader(8).to(dtype=dtype)
            request = torch.randn(8, dtype=dtype, requires_grad=True)
            h = torch.randn(2, 8, dtype=dtype)
            context = torch.randn(2, 8, dtype=dtype)
            encoded = torch.randn(7, 8, dtype=dtype, requires_grad=True)
            metadata = torch.randn(7, 8, dtype=dtype, requires_grad=True)
            valid = torch.tensor([True, True, True, True, True, False, False])
            parameters_before = value_digest(reader.state_dict())
            result, attention, available = reader(request, h, context, encoded, metadata, valid)
            assert available and result.shape == context.shape
            assert attention.shape == (2, 2, 2, 7)
            assert bool(torch.isfinite(result).all())
            assert bool((attention[..., ~valid] == 0).all())
            torch.testing.assert_close(attention.sum(-1), torch.ones_like(attention.sum(-1)))
            order = torch.randperm(7)
            permuted, weights, _ = reader(
                request, h, context, encoded[order], metadata[order], valid[order]
            )
            torch.testing.assert_close(result, permuted)
            torch.testing.assert_close(attention[..., order], weights)
            absent = encoded.detach().clone()
            absent[~valid] = float("nan")
            masked, _, _ = reader(request, h, context, absent, metadata, valid)
            torch.testing.assert_close(masked, result, atol=0, rtol=0)
            empty, empty_weights, available = reader(
                request, h, context, encoded, metadata, torch.zeros_like(valid)
            )
            assert not available and torch.equal(empty, context)
            assert bool((empty_weights == 0).all())
            empty, empty_weights, available = reader(
                request, h, context, encoded[:0], metadata[:0], valid[:0]
            )
            assert not available and torch.equal(empty, context) and empty_weights.shape[-1] == 0
            other_request = request.roll(1)
            other_tokens = encoded.roll(1, 0)
            q_only, _, _ = reader(other_request, h, context, encoded, metadata, valid)
            x_only, _, _ = reader(request, h, context, other_tokens, metadata, valid)
            both, _, _ = reader(other_request, h, context, other_tokens, metadata, valid)
            interaction = float((result - q_only - x_only + both).detach().abs().max())
            assert interaction > 1e-10
            # A scalar objective checks credit through every named matrix.
            (result.square().mean() + result.mean()).backward()
            norms = {
                name: float(p.grad.norm())
                for name, p in reader.named_parameters()
                if p.grad is not None
            }
            assert len(norms) == len(list(reader.parameters()))
            assert all(value > 0 and value < float("inf") for value in norms.values())
            assert (
                encoded.grad is not None and metadata.grad is not None and request.grad is not None
            )
            assert bool((encoded.grad[~valid] == 0).all())
            assert bool((metadata.grad[~valid] == 0).all())
            assert float(encoded.grad[valid].norm()) > 0
            assert float(metadata.grad[valid].norm()) > 0
            assert float(request.grad.norm()) > 0
            assert parameters_before == value_digest(reader.state_dict())
            restored = QueryEvidenceReader(8).to(dtype=dtype)
            restored.load_state_dict(reader.state_dict())
            actual = restored(request, h, context, encoded, metadata, valid)
            assert_values_equal((result, attention), actual[:2], "local_reader_restore")
            rejected = []
            for bad_valid in (valid.float(), valid[:-1]):
                try:
                    reader(request, h, context, encoded, metadata, bad_valid)
                except ValueError:
                    rejected.append(True)
            assert len(rejected) == 2
            try:
                reader(
                    request,
                    h,
                    context,
                    encoded.new_zeros(129, 8),
                    metadata.new_zeros(129, 8),
                    torch.ones(129, dtype=torch.bool),
                )
            except ValueError as error:
                assert str(error) == "evidence_reader_token_budget_stop"
            else:
                raise AssertionError("unbounded_evidence_was_accepted")
            records[str(dtype)] = {
                "verdict": "passed_local_reader_checks",
                "parameters": sum(p.numel() for p in reader.parameters()),
                "parameter_grad_norms": norms,
                "encoded_grad_norm": float(encoded.grad.norm()),
                "metadata_grad_norm": float(metadata.grad.norm()),
                "request_grad_norm": float(request.grad.norm()),
                "query_content_mixed_difference": interaction,
                "checks": [
                    "finite",
                    "normalized_masked_attention",
                    "token_permutation",
                    "invalid_nan_mask",
                    "all_missing",
                    "zero_tokens",
                    "query_content_interaction",
                    "all_parameter_credit",
                    "input_credit",
                    "invalid_token_zero_credit",
                    "parameters_unchanged",
                    "state_dict_restore",
                    "invalid_shape_rejected",
                    "token_budget_rejected",
                ],
            }
    report: dict[str, Any] = {
        "schema": "taiji-u2-evidence-reader-local-v1",
        "verdict": "passed_local_reader_only",
        "scope": "Standalone module; subject normal entry, provenance admission, canonical history rebuild and independent-process restoration remain pending. Random synthetic inputs; no task data or optimizer updates.",
        "precisions": records,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "taiji/u2_evidence_read.py", Path(__file__))
        },
        "optimizer_steps": blocked.calls,
        "task_data_reads": 0,
        "check_reads": 0,
        "final_reads": 0,
        "capability_qualifications": [],
        "cpu_seconds": time.process_time(),
        "wall_seconds": time.monotonic() - STARTED,
        "run_conditions": {"device": "cpu", "threads": 1, "torch": torch.__version__},
    }
    assert report["cpu_seconds"] <= 30 and report["wall_seconds"] <= 60
    write_report(report_path, report)
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "cpu_seconds": report["cpu_seconds"],
                "precisions": records,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    verify()
