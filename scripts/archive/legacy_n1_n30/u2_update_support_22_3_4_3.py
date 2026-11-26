"""Recipe-owned gradients and verified checkpoint I/O for U2 engineering updates."""

from __future__ import annotations

import hashlib
import json
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any

import torch

from scripts.training.u2_attempt_ledger_22_3_4_3 import LedgerError
from taiji.u2_history import assert_values_equal, atomic_load, value_digest
from taiji.u2_recovery import ReferenceExecutor, U2Runtime, _optimizer_layout, _restore_optimizer
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture


class ReadbackError(RuntimeError):
    exit_code = 4


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(runtime: U2Runtime) -> dict[str, Any]:
    return {
        "state_sha256": value_digest(runtime.state),
        "parameter_sha256": value_digest(runtime.model.state_dict()),
        "moment_sha256": value_digest(
            None if runtime.optimizer is None else runtime.optimizer.state_dict()
        ),
        "history_reference": runtime.history.reference(),
        "revision": runtime.model.parameter_revision,
    }


def distributions(runtime: U2Runtime) -> dict[str, Any]:
    with torch.no_grad():
        return {
            "next_output": runtime.output()[0].exp().tolist(),
            "policy": runtime.model.policy(runtime.state)["log_policy"].exp().tolist(),
        }


def fresh_load(path: Path) -> U2Runtime:
    envelope = atomic_load(path)
    executor = ReferenceExecutor.from_snapshot(envelope["executor_snapshot"])
    return CheckedRuntime.from_envelope(envelope, executor=executor)


def readback(path: Path, expected: dict[str, Any]) -> tuple[U2Runtime, dict[str, Any]]:
    started = time.process_time()
    try:
        restored = fresh_load(path)
        actual = fingerprint(restored)
        assert_values_equal(expected, actual, "checkpoint.readback")
    except Exception as error:
        raise ReadbackError(f"Checkpoint read-back failed: {path}: {error}") from error
    return restored, {
        "path": str(path),
        "sha256": file_sha(path),
        "bytes": path.stat().st_size,
        "readback_ok": True,
        "cpu_seconds": time.process_time() - started,
        "fingerprint": actual,
    }


class CheckedRuntime(U2Runtime):
    """Every save, including durable intents, immediately verifies a fresh load."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readbacks: list[dict[str, Any]] = []

    def save(self, path) -> Path:
        path = Path(path)
        if (
            path.name in {"parent.bin", "step1.bin", "step2-orig.bin", "step2-recovery.bin"}
            and path.exists()
        ):
            raise FileExistsError(path)
        expected = fingerprint(self)
        destination = super().save(path)
        if os.name == "nt":
            directory_sync = "unsupported"
        else:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory_sync = "done"
        _, proof = readback(destination, expected)
        proof["dir_sync"] = directory_sync
        self.readbacks.append(proof)
        return destination


def optimizer_for(model, manifest):
    options = dict(manifest["optimizer"])
    if options.pop("type") != "Adam":
        raise ValueError("Only the frozen Adam recipe is supported")
    options["betas"] = tuple(options["betas"])
    return torch.optim.Adam(model.parameters(), **options)


def recipe_records(fixture, group):
    return fixture["records"] if group == "F1" else (fixture["records"][0],)


def prepare_parent(group: str, manifest: dict[str, Any], directory: Path) -> CheckedRuntime:
    dtype = getattr(torch, manifest["groups"][group]["dtype"])
    model = U2Subject(U2Config.small()).to(dtype=dtype)
    runtime = CheckedRuntime(
        model, executor=ReferenceExecutor(), subject_id="engineering-parent-" + group
    )
    fixture = synthetic_fixture(model.c, dtype)
    runtime.set_goal(manifest["groups"][group]["goal"])
    runtime.set_records(recipe_records(fixture, group))
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    identity = runtime.issue(runtime.choose(mask), dt=manifest["recipe"]["dt"])
    runtime.execute_pending(checkpoint=directory / "_preparation-intent.bin")
    runtime.set_process_time(manifest["recipe"]["dt"])
    event = fixture["events"][0]
    runtime.receive(
        U2Packet(
            "engineering-parent-observation",
            "sensor",
            "world",
            0.0,
            manifest["recipe"]["dt"],
            manifest["recipe"]["dt"],
            event["observations"],
            event["coordinates"],
            event["outcome"],
            identity,
            feedback_record_ids=("red",),
        )
    )
    runtime.begin_new_window()
    runtime.optimizer = optimizer_for(model, manifest)
    return runtime


def gradients_for(runtime: U2Runtime, group: str, manifest: dict[str, Any]):
    model = runtime.model
    previous = fingerprint(runtime)
    config = manifest["groups"][group]
    fixture = synthetic_fixture(model.c, model.native_w.dtype)
    state = model.initial("engineering-shadow-" + group)
    state = model.set_goal(state, config["goal"])
    records = recipe_records(fixture, group)
    state = model.set_records(state, records)
    receipts = []
    for index, event in enumerate(fixture["events"]):
        mask = torch.zeros(9, dtype=torch.bool)
        mask[0] = True
        state, decision = model.choose_operation(state, mask)
        state, identity = model.issue(state, decision, dt=manifest["recipe"]["dt"])
        end = (index + 1) * manifest["recipe"]["dt"]
        state["process_time"] = end
        packet = U2Packet(
            f"shadow-{group}-{index}",
            "sensor",
            "world",
            index * manifest["recipe"]["dt"],
            end,
            end,
            event["observations"],
            event["coordinates"],
            event["outcome"],
            identity,
            feedback_record_ids=("red",),
        )
        state, result = model.receive(state, packet)
        receipts.append(result)
    pieces = {
        "predictive": -torch.stack(
            [receipt["predictive_logp"].mean() for receipt in receipts]
        ).mean(),
        "actor": torch.stack([receipt["actor"] for receipt in receipts]).mean(),
        "critic": torch.stack([receipt["critic"] for receipt in receipts]).mean(),
        "verbatim": model.native_w.new_zeros(()),
    }
    if group == "F2":
        target = config["target"].encode("utf-8")
        if len(records) != 1 or records[0].raw != target:
            raise ValueError("Verbatim request/target mismatch")
        utterance = model.begin_utterance(state)
        goal = model.goal(utterance["goal_raw"])
        terms = []
        for symbol in [*target, 256]:
            log_probability, _, output = model.output(
                utterance, goal, utterance["prediction_action"]
            )
            terms.append(-log_probability[symbol])
            utterance = model.accept_byte(utterance, symbol, output)
        pieces["verbatim"] = torch.stack(terms).mean()
    loss = sum(config["weights"][key] * value for key, value in pieces.items())
    named = dict(model.named_parameters())
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    values = dict(
        zip(
            named,
            (None if gradient is None else gradient.detach().clone() for gradient in gradients),
            strict=True,
        )
    )
    if any(
        value is not None and not bool(torch.isfinite(value).all()) for value in values.values()
    ):
        raise ValueError("Nonfinite engineering gradient")
    floor = manifest["activity_floors"][config["dtype"]]["gradient_norm"]
    for name in config["active_parameters"]:
        if values[name] is None or float(values[name].norm()) < floor:
            raise ValueError(f"Frozen active-gradient floor failed: {name}")
    assert_values_equal(previous, fingerprint(runtime), "shadow_training_did_not_change_parent")
    return values, {
        "total": float(loss.detach()),
        **{name: float(value.detach()) for name, value in pieces.items()},
    }


def fork_candidate(runtime, gradients, *, fault=False, manifest):
    values = runtime.optimizer.state_dict()
    bundle = runtime.model.export_bundle(runtime.state, values)
    candidate, _ = U2Subject.from_bundle(bundle)
    optimizer = _restore_optimizer(
        candidate, _optimizer_layout(runtime.model, runtime.optimizer), values
    )
    assert_values_equal(
        value_digest(runtime.model.state_dict()), value_digest(candidate.state_dict())
    )
    assert_values_equal(value_digest(values), value_digest(optimizer.state_dict()))
    if fault:
        with torch.no_grad():
            candidate.text_input.bias.add_(manifest["fault_candidate"]["text_input.bias_add"])
            candidate.critic.last.bias.add_(manifest["fault_candidate"]["critic.last.bias_add"])
    else:
        for name, parameter in candidate.named_parameters():
            parameter.grad = None if gradients[name] is None else gradients[name].clone()
        optimizer.step()
    candidate.mark_parameter_update()
    return candidate, optimizer


class StepCounter:
    def __init__(self, ledger, claim_root: Path | None = None):
        self.ledger = ledger
        self.claim_root = claim_root
        self.current = None
        self.called: set[str] = set()

    def wrapper(self, original):
        @wraps(original)
        def step(optimizer, *args, **kwargs):
            state = self.ledger.read()
            identity = self.current
            if (
                identity not in state["opened"]
                or identity in state["closed"]
                or identity in self.called
                or identity.startswith("fault-")
            ):
                raise LedgerError("Optimizer step has no unique durable open attempt")
            if identity in state["reconciled"]:
                raise LedgerError("Reconciled attempt cannot execute an optimizer step")
            if self.claim_root is not None:
                self.claim_root.mkdir(parents=True, exist_ok=True)
                with (self.claim_root / (identity + ".json")).open(
                    "x", encoding="utf-8", newline="\n"
                ) as stream:
                    stream.write(
                        json.dumps(
                            {
                                "attempt_id": identity,
                                "run_id": self.ledger.run["run_id"],
                                "open_seq": state["opened"][identity]["seq"],
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
            self.called.add(identity)
            return original(optimizer, *args, **kwargs)

        return step


def parameter_rows(parent, post, gradients):
    before = parent.model.state_dict()
    after = post.model.state_dict()
    return [
        {
            "parameter": name,
            "gradient_norm": None if gradient is None else float(gradient.norm()),
            "max_parameter_change": float((before[name] - after[name]).abs().max()),
        }
        for name, gradient in gradients.items()
    ]


def check_adam_equations(parent, post, gradients, manifest, group):
    options = manifest["optimizer"]
    beta1, beta2 = options["betas"]
    tolerance = manifest["tolerances"][manifest["groups"][group]["dtype"]]
    old = dict(parent.model.named_parameters())
    new = dict(post.model.named_parameters())
    results = []
    with torch.no_grad():
        for name, parameter in old.items():
            gradient = gradients[name]
            previous = parent.optimizer.state.get(parameter, {})
            actual = post.optimizer.state.get(new[name], {})
            if gradient is None:
                assert_values_equal(parameter, new[name], name + ".unused_parameter")
                assert_values_equal(previous, actual, name + ".unused_moment")
                results.append({"parameter": name, "status": "unused_by_loss"})
                continue
            mean = previous.get("exp_avg", torch.zeros_like(parameter)) * beta1 + gradient * (
                1 - beta1
            )
            variance = previous.get(
                "exp_avg_sq", torch.zeros_like(parameter)
            ) * beta2 + gradient.square() * (1 - beta2)
            step = float(previous.get("step", 0)) + 1
            expected = parameter - options["lr"] * (mean / (1 - beta1**step)) / (
                (variance / (1 - beta2**step)).sqrt() + options["eps"]
            )
            assert float(actual["step"]) == step, name
            assert torch.allclose(actual["exp_avg"], mean, **tolerance), name + ".mean"
            assert torch.allclose(actual["exp_avg_sq"], variance, **tolerance), name + ".variance"
            assert torch.allclose(new[name], expected, **tolerance), name + ".update"
            results.append(
                {
                    "parameter": name,
                    "status": "adam_equation_matched",
                    "max_abs_delta": float((new[name] - expected).abs().max()),
                }
            )
    return results
