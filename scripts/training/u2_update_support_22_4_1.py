"""Recipe-owned gradients and verified checkpoint I/O for the 22.4.1 u1 update."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from scripts.training.u2_update_support_22_3_4_3 import (
    CheckedRuntime,
    ReadbackError,
    file_sha,
    fingerprint,
)
from taiji.u2_history import assert_values_equal, atomic_load, atomic_save, value_digest
from taiji.u2_recovery import ReferenceExecutor
from taiji.u2_subject import U2Config, U2Packet, U2Subject, synthetic_fixture

DEFAULT_ARM = "u1-candidate-float32"
# 兼容旧调用点：u1 的臂名字面量逐字保留为默认值；任何一次实际执行都必须显式传 arm。
ARM = DEFAULT_ARM
ADAM_FLOAT32_TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}


def arm_for(run: dict[str, Any]) -> str:
    """臂名不再硬编码：一律从 run.json 的 update_id 读取（u1/u2 各自绑定自己的臂）。"""
    identity = run["update_id"]
    if not isinstance(identity, str) or not identity:
        raise ValueError("run.json update_id must be a non-empty string")
    return identity


def normalized_sha256(path: Path) -> str:
    """UTF8 without BOM, universal newline to LF (frozen hash semantics)."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8-sig").encode()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


class U1Runtime(CheckedRuntime):
    """Every canonical checkpoint is create-once and read back by a fresh load."""

    CREATE_ONCE = {"parent.bin", "post.bin"}

    def save(self, path) -> Path:
        path = Path(path)
        if path.name in self.CREATE_ONCE and path.exists():
            raise FileExistsError(path)
        return super().save(path)


def optimizer_for(model, manifest):
    options = dict(manifest["optimizer"])
    if options.pop("type") != "Adam":
        raise ValueError("Only the frozen Adam recipe is supported")
    options.pop("basis", None)
    options["betas"] = tuple(options["betas"])
    return torch.optim.Adam(model.parameters(), **options)


def parent_digest(dtype: torch.dtype) -> str:
    from taiji.u2_history import value_digest

    base = U2Subject(U2Config.small()).to(dtype=dtype)
    return value_digest(base.state_dict())


REPO_ROOT = Path(__file__).resolve().parents[2]


class BaselineRuntime:
    """22.4.4 冻结的 Transformer 基线，作为 u2 臂的执行图（不是 U2 元数据候选）。

    该图没有主体 state、没有历史对象，因此 checkpoint 的五个字段里
    `state_sha256` 记的是执行图身份摘要，`history_reference` 是稳定的图标签；
    参数与优化器动量仍按 u1 的同一口径做摘要与读回。
    """

    CREATE_ONCE = {"parent.bin", "post.bin"}
    SCHEMA = "taiji-u2-22.4.1-baseline-checkpoint-v1"

    def __init__(self, model, optimizer, graph: dict[str, Any], arm: str) -> None:
        self.model = model
        self.optimizer = optimizer
        self.graph = dict(graph)
        self.arm = arm
        self.revision = 0
        self.readbacks: list[dict[str, Any]] = []

    def named_parameters(self):
        return self.model.named_parameters()


def baseline_fingerprint(runtime: BaselineRuntime) -> dict[str, Any]:
    """与 u1 同名的五个读回字段；本图无主体 state，state 字段记执行图身份。"""
    graph = runtime.graph
    return {
        "state_sha256": canonical_sha256(
            {
                "schema": "taiji-u2-22.4.1-baseline-execution-graph-v1",
                "arm": runtime.arm,
                "architecture_id": graph["architecture_id"],
                "param_total": graph["param_total"],
                "param_tensors": graph["param_tensors"],
                "param_inventory_sha256": graph["param_inventory_sha256"],
                "identity_sha256": graph["identity_sha256"],
            }
        ),
        "parameter_sha256": value_digest(runtime.model.state_dict()),
        "moment_sha256": value_digest(
            None if runtime.optimizer is None else runtime.optimizer.state_dict()
        ),
        "history_reference": "baseline:" + graph["identity_sha256"][:16],
        "revision": runtime.revision,
    }


def readback_baseline(
    path: Path, expected: dict[str, Any], manifest: dict[str, Any], *, arm: str
) -> tuple[BaselineRuntime, dict[str, Any]]:
    """Fresh rebuild from the frozen identity, load state/moments, compare five fields."""
    started = time.process_time()
    try:
        payload = atomic_load(path)
        if payload.get("schema") != BaselineRuntime.SCHEMA:
            raise ValueError(f"unexpected baseline checkpoint schema: {payload.get('schema')}")
        if payload.get("arm") != arm:
            raise ValueError(f"checkpoint arm {payload.get('arm')} != {arm}")
        model, graph = baseline_graph(manifest, arm=arm)
        if payload.get("graph") != graph:
            raise ValueError("checkpoint graph differs from the frozen identity")
        model.load_state_dict(payload["model"])
        optimizer = optimizer_for(model, manifest)
        optimizer.load_state_dict(payload["optimizer"])
        restored = BaselineRuntime(model, optimizer, graph, arm)
        restored.revision = int(payload["revision"])
        actual = baseline_fingerprint(restored)
        assert_values_equal(expected, actual, "baseline_checkpoint.readback")
        if expected.get("revision") != actual["revision"]:
            raise ValueError(f"revision {actual['revision']} != {expected.get('revision')}")
    except Exception as error:  # noqa: BLE001 - read-back must fail closed
        raise ReadbackError(f"Baseline checkpoint read-back failed: {path}: {error}") from error
    return restored, {
        "path": str(path),
        "sha256": file_sha(path),
        "bytes": Path(path).stat().st_size,
        "readback_ok": True,
        "cpu_seconds": time.process_time() - started,
        "fingerprint": actual,
    }


def save_baseline(runtime: BaselineRuntime, path: Path, manifest: dict[str, Any]) -> Path:
    """Create-once write, then a fresh load must reproduce the five fields."""
    path = Path(path)
    if path.name in BaselineRuntime.CREATE_ONCE and path.exists():
        raise FileExistsError(path)
    expected = baseline_fingerprint(runtime)
    atomic_save(
        {
            "schema": BaselineRuntime.SCHEMA,
            "arm": runtime.arm,
            "graph": runtime.graph,
            "revision": runtime.revision,
            "model": runtime.model.state_dict(),
            "optimizer": runtime.optimizer.state_dict(),
        },
        path,
    )
    if os.name == "nt":
        directory_sync = "unsupported"
    else:
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_sync = "done"
    _, proof = readback_baseline(path, expected, manifest, arm=runtime.arm)
    proof["dir_sync"] = directory_sync
    runtime.readbacks.append(proof)
    return path


def baseline_graph(manifest: dict[str, Any], *, arm: str = DEFAULT_ARM) -> tuple[Any, dict]:
    """按身份清单重建 u2 的执行图，并核对参数量/张量数/清单摘要逐项一致。"""
    row = manifest["arms"][arm]
    reference = row.get("baseline_identity_ref")
    if not reference:
        raise ValueError(f"{arm} 没有 baseline_identity_ref")
    identity_path = REPO_ROOT / reference
    identity_sha = normalized_sha256(identity_path)
    if identity_sha != row.get("baseline_identity_sha256"):
        raise ValueError(f"{arm} 的身份清单哈希与臂行 pin 不一致")
    identity = json.loads(Path(identity_path).read_text(encoding="utf-8"))
    architecture = identity["architecture"]
    module_path = REPO_ROOT / architecture["module_path"]
    if normalized_sha256(module_path) != architecture["module_sha256"]:
        raise ValueError("身份清单冻结的基线模块源码已变动")
    from scripts.training.baseline_identity_22_4_4 import build_arm, param_counts

    model = build_arm(architecture["architecture_id"], architecture["free_quantity"]["value"])
    total, tensors = param_counts(model)
    inventory = canonical_sha256(
        [
            [name, list(parameter.shape), str(parameter.dtype).removeprefix("torch.")]
            for name, parameter in sorted(model.named_parameters())
        ]
    )
    if (total, tensors) != (identity["param_total"], identity["param_tensors"]):
        raise ValueError(f"重建图 {total}/{tensors} 与身份清单不一致")
    if inventory != identity["param_inventory_sha256"]:
        raise ValueError("重建图的参数清单摘要与身份清单不一致")
    return model, {
        "architecture_id": architecture["architecture_id"],
        "param_total": total,
        "param_tensors": tensors,
        "param_inventory_sha256": inventory,
        "identity_sha256": identity_sha,
        "identity_ref": str(reference),
    }


def prepare_parent(
    manifest: dict[str, Any], directory: Path, *, arm: str = DEFAULT_ARM
) -> U1Runtime | BaselineRuntime:
    recipe = manifest["recipe"]
    dtype = getattr(torch, manifest["arms"][arm]["dtype"])
    if arm != DEFAULT_ARM:
        model, graph = baseline_graph(manifest, arm=arm)
        if model.position.dtype != dtype:
            raise ValueError(f"{arm} 的执行图 dtype {model.position.dtype} 与臂行 {dtype} 不一致")
        return BaselineRuntime(model, optimizer_for(model, manifest), graph, arm)
    base = U2Subject(U2Config.small()).to(dtype=dtype)
    model = base.metadata_candidate()
    runtime = U1Runtime(model, executor=ReferenceExecutor(), subject_id="engineering-parent-" + arm)
    fixture = synthetic_fixture(model.c, dtype)
    runtime.set_goal(recipe["u1_shadow"]["goal"])
    runtime.set_records(fixture["records"])
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    identity = runtime.issue(runtime.choose(mask), dt=recipe["dt"])
    runtime.execute_pending(checkpoint=directory / "_preparation-intent.bin")
    runtime.set_process_time(recipe["dt"])
    event = fixture["events"][0]
    runtime.receive(
        U2Packet(
            "engineering-parent-observation",
            "sensor",
            "world",
            0.0,
            recipe["dt"],
            recipe["dt"],
            event["observations"],
            event["coordinates"],
            event["outcome"],
            identity,
            feedback_record_ids=tuple(recipe["feedback_records"]),
        )
    )
    runtime.begin_new_window()
    runtime.optimizer = optimizer_for(model, manifest)
    return runtime


def _tensor_digests(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(
                value.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest(),
        }
    raise TypeError(type(value))


def recipe_digest(manifest: dict[str, Any]) -> str:
    return canonical_sha256(manifest["recipe"])


def _record_fields(record) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "raw_hex": record.raw.hex(),
        "version": record.version,
        "source": record.source,
        "scope": record.scope,
        "observed_at": record.observed_at,
        "writable": record.writable,
    }


def _baseline_slice_descriptor(
    manifest: dict[str, Any], model, *, arm: str = DEFAULT_ARM
) -> dict[str, Any]:
    from scripts.training.baseline_identity_22_4_4 import fragment_inputs

    recipe = manifest["recipe"]
    shadow = recipe["u2_shadow"]
    config = U2Config.small()
    fixture = synthetic_fixture(config, model.position.dtype)
    observations, _ = fragment_inputs()
    return {
        "graph": {
            "architecture_id": model.architecture_id,
            "param_total": int(sum(parameter.numel() for parameter in model.parameters())),
            "param_tensors": len(list(model.named_parameters())),
        },
        "seed": config.seed,
        "dtype": manifest["arms"][arm]["dtype"],
        "dt": recipe["dt"],
        "goal": shadow["goal"],
        "records": [
            _record_fields(record)
            for record in fixture["records"]
            if record.record_id in set(recipe["feedback_records"])
        ],
        "target_hex": shadow["target"].encode("utf-8").hex(),
        "observations": {name: _tensor_digests(value) for name, value in observations.items()},
        "feedback_records": list(recipe["feedback_records"]),
    }


def slice_descriptor(
    manifest: dict[str, Any], runtime_model, *, arm: str = DEFAULT_ARM
) -> dict[str, Any]:
    if arm != DEFAULT_ARM:
        return _baseline_slice_descriptor(manifest, runtime_model, arm=arm)
    recipe = manifest["recipe"]
    fixture = synthetic_fixture(runtime_model.c, runtime_model.native_w.dtype)
    observable = fixture["events"][: recipe["observations_per_shadow_episode"]]
    return {
        "config": "U2Config.small",
        "seed": runtime_model.c.seed,
        "dtype": manifest["arms"][arm]["dtype"],
        "dt": recipe["dt"],
        "goal": recipe["u1_shadow"]["goal"],
        "records": [_record_fields(record) for record in fixture["records"]],
        "observations": [
            {
                "observations": {
                    name: _tensor_digests(value) for name, value in event["observations"].items()
                },
                "coordinates": {
                    name: _tensor_digests(value) for name, value in event["coordinates"].items()
                },
                "outcome": _tensor_digests(event["outcome"]),
            }
            for event in observable
        ],
        "feedback_records": list(recipe["feedback_records"]),
    }


def slice_digest(manifest: dict[str, Any], runtime_model, *, arm: str = DEFAULT_ARM) -> str:
    return canonical_sha256(slice_descriptor(manifest, runtime_model, arm=arm))


def active_parameters(model, manifest: dict[str, Any], *, arm: str = DEFAULT_ARM) -> list[str]:
    names = list(manifest["arms"][arm]["active_parameters"])
    group_units = getattr(getattr(model, "c", None), "group_units", None)
    if group_units is not None:
        for key in model.group_extensions:
            names.extend([f"group_extensions.{key}.incoming", f"group_extensions.{key}.outgoing"])
    return names


def _baseline_trunk(model, observations: dict, text_context: torch.Tensor) -> torch.Tensor:
    """与冻结 forward 前半段逐行相同，只是把 STEPS 从张量形状取出来。"""
    sequence = model.frontend.step_tokens(observations, text_context)
    hidden = model.record_in(sequence) + model.position[None]
    steps = hidden.shape[1]
    mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=hidden.device), diagonal=1)
    attended, _ = model.attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
    hidden = model.norm_attention(hidden + attended)
    return model.norm_feedforward(hidden + model.feedforward(hidden))


def baseline_shadow_terms(
    model, manifest: dict[str, Any], *, arm: str = DEFAULT_ARM
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """F2 口径：verbatim 项按基线的 byte_head 逐字教师强制，不再是零张量。

    目标来源：`recipe.u2_shadow.target`（已冻结的 train_label），不引入新反馈通道。
    actor/critic 在该图上没有对应物，按不存在处理为 0。
    """
    from scripts.training.baseline_identity_22_4_4 import fragment_inputs

    recipe = manifest["recipe"]
    shadow = recipe["u2_shadow"]
    target_bytes = shadow["target"].encode("utf-8")
    observations, _ = fragment_inputs()
    observations = {name: value.to(model.position.dtype) for name, value in observations.items()}
    text_context = model.frontend.text_units(shadow["goal"])
    hidden = _baseline_trunk(model, observations, text_context)
    last = hidden[:, -1]
    batch = last.shape[0]

    def byte_logits(previous_byte: int) -> torch.Tensor:
        previous = model.frontend.embedding(
            torch.full((batch,), previous_byte, dtype=torch.long, device=last.device)
        )
        return model.byte_head(torch.cat([last, previous], -1))

    def cross_entropy(logits: torch.Tensor, byte: int) -> torch.Tensor:
        return functional.cross_entropy(
            logits, torch.full((batch,), byte, dtype=torch.long, device=last.device)
        )

    predictive = cross_entropy(byte_logits(255), target_bytes[0])
    losses = []
    previous_byte = 255
    for byte in target_bytes:
        losses.append(cross_entropy(byte_logits(previous_byte), byte))
        previous_byte = byte
    verbatim = torch.stack(losses).mean()
    weights = manifest["arms"][arm]["weights"]
    pieces = {
        "predictive": predictive,
        "actor": last.new_zeros(()),
        "critic": last.new_zeros(()),
        "verbatim": verbatim,
    }
    return sum(weights[name] * value for name, value in pieces.items()), pieces


def baseline_run_conditions(model) -> dict[str, Any]:
    """与 u1 的 run_conditions 同键；本图没有 native_w，取第一个参数作 device/dtype 见证。"""
    parameter = next(iter(model.parameters()))
    return {
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "python": sys.version,
        "torch": torch.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def baseline_outputs(runtime: BaselineRuntime, manifest: dict[str, Any]) -> dict[str, Any]:
    """该图的可观测输出：下一字节分布（256 项）与结构空的 policy。"""
    from scripts.training.baseline_identity_22_4_4 import fragment_inputs

    model = runtime.model
    observations, _ = fragment_inputs()
    with torch.no_grad():
        observations = {
            name: value.to(model.position.dtype) for name, value in observations.items()
        }
        text_context = model.frontend.text_units(manifest["recipe"]["u2_shadow"]["goal"])
        hidden = _baseline_trunk(model, observations, text_context)
        last = hidden[:, -1]
        batch = last.shape[0]
        previous = model.frontend.embedding(
            torch.full((batch,), 255, dtype=torch.long, device=last.device)
        )
        logits = model.byte_head(torch.cat([last, previous], -1))
        return {
            "next_output": logits[0].softmax(-1).tolist(),
            "policy": [],
            "policy_note": "该执行图没有策略头；policy 是结构空，policy_l1 恒为 0。",
        }


def shadow_terms(model, manifest: dict[str, Any], *, arm: str = DEFAULT_ARM):
    """u1 是冻结的 shadow 片段；u2 走基线的 F2 口径。"""
    if arm != DEFAULT_ARM:
        return baseline_shadow_terms(model, manifest, arm=arm)
    recipe = manifest["recipe"]
    dt = recipe["dt"]
    fixture = synthetic_fixture(model.c, model.native_w.dtype)
    state = model.initial("engineering-shadow-" + arm)
    state = model.set_goal(state, recipe["u1_shadow"]["goal"])
    state = model.set_records(state, fixture["records"])
    receipts = []
    for index, event in enumerate(fixture["events"][: recipe["observations_per_shadow_episode"]]):
        mask = torch.zeros(9, dtype=torch.bool)
        mask[0] = True
        state, decision = model.choose_operation(state, mask)
        state, identity = model.issue(state, decision, dt=dt)
        end = (index + 1) * dt
        state["process_time"] = end
        state, result = model.receive(
            state,
            U2Packet(
                f"shadow-{arm}-{index}",
                "sensor",
                "world",
                index * dt,
                end,
                end,
                event["observations"],
                event["coordinates"],
                event["outcome"],
                identity,
                feedback_record_ids=tuple(recipe["feedback_records"]),
            ),
        )
        receipts.append(result)
    weights = manifest["arms"][arm]["weights"]
    pieces = {
        "predictive": -torch.stack(
            [receipt["predictive_logp"].mean() for receipt in receipts]
        ).mean(),
        "actor": torch.stack([receipt["actor"] for receipt in receipts]).mean(),
        "critic": torch.stack([receipt["critic"] for receipt in receipts]).mean(),
        "verbatim": model.native_w.new_zeros(()),
    }
    return sum(weights[name] * value for name, value in pieces.items()), pieces


def gradients_for(
    runtime: U1Runtime | BaselineRuntime,
    manifest: dict[str, Any],
    *,
    arm: str = DEFAULT_ARM,
    fingerprint_fn=fingerprint,
):
    """Loss and gradients for one arm. The fingerprint hook stays u1-identical by default."""
    model = runtime.model
    previous = fingerprint_fn(runtime)
    loss, pieces = shadow_terms(model, manifest, arm=arm)
    named = dict(model.named_parameters())
    measured = torch.autograd.grad(loss, list(named.values()), allow_unused=True)
    values = dict(
        zip(
            named,
            (None if gradient is None else gradient.detach().clone() for gradient in measured),
            strict=True,
        )
    )
    if any(
        value is not None and not bool(torch.isfinite(value).all()) for value in values.values()
    ):
        raise ValueError("Nonfinite engineering gradient")
    floor = manifest["activity_floors"][manifest["arms"][arm]["dtype"]]
    names = active_parameters(model, manifest, arm=arm)
    if any(
        values[name] is None or float(values[name].norm()) < floor["gradient_norm"]
        for name in names
    ):
        from scripts.training.u2_update_executor_22_4_1 import ActivityGateError

        rows = [
            {
                "parameter": name,
                "gradient_norm": None if values[name] is None else float(values[name].norm()),
                "change": 0.0,
                "change_floor": floor["max_parameter_change"],
                "margin": 0.0,
                "barely_live": True,
                "gradient_ok": values[name] is not None
                and float(values[name].norm()) >= floor["gradient_norm"],
                "change_ok": False,
                "note": "no optimizer step was taken; the gradient floor failed first",
            }
            for name in names
        ]
        error = ActivityGateError(
            "Frozen active-gradient floor failed: "
            + str([row["parameter"] for row in rows if not row["gradient_ok"]])
        )
        error.receipt = {
            "parameters": rows,
            "min_margin": 0.0,
            "barely_live_parameters": [row["parameter"] for row in rows],
            "floors": floor,
        }
        raise error
    assert_values_equal(previous, fingerprint_fn(runtime), "shadow_training_did_not_change_parent")
    return values, {
        "total": float(loss.detach()),
        **{name: float(value.detach()) for name, value in pieces.items()},
    }


def fork_candidate(runtime: U1Runtime, gradients, manifest: dict[str, Any]):
    from scripts.training.u2_update_support_22_3_4_3 import fork_candidate as fork

    return fork(runtime, gradients, manifest=manifest)


def check_adam_equations(parent: U1Runtime, post: U1Runtime, gradients, manifest) -> list:
    """Recompute Adam's update for every parameter; tolerance is the frozen 22.4.1 one."""
    options = manifest["optimizer"]
    beta1, beta2 = options["betas"]
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
            assert torch.allclose(actual["exp_avg"], mean, **ADAM_FLOAT32_TOLERANCE), name + ".mean"
            assert torch.allclose(actual["exp_avg_sq"], variance, **ADAM_FLOAT32_TOLERANCE), (
                name + ".variance"
            )
            assert torch.allclose(new[name], expected, **ADAM_FLOAT32_TOLERANCE), name + ".update"
            results.append(
                {
                    "parameter": name,
                    "status": "adam_equation_matched",
                    "max_abs_delta": float((new[name] - expected).abs().max()),
                }
            )
    return results


def parameter_rows(parent: U1Runtime, post: U1Runtime, gradients) -> list[dict[str, Any]]:
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


def activity_receipt(
    model: U2Subject,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    gradients,
    manifest: dict[str, Any],
    *,
    arm: str = DEFAULT_ARM,
) -> dict[str, Any]:
    floor = manifest["activity_floors"][manifest["arms"][arm]["dtype"]]
    rows = []
    for name in active_parameters(model, manifest, arm=arm):
        change = float((before_state[name] - after_state[name]).abs().max())
        margin = change / floor["max_parameter_change"]
        rows.append(
            {
                "parameter": name,
                "gradient_norm": float(gradients[name].norm()),
                "change": change,
                "change_floor": floor["max_parameter_change"],
                "margin": margin,
                "barely_live": margin < 10,
                "gradient_ok": float(gradients[name].norm()) >= floor["gradient_norm"],
                "change_ok": change >= floor["max_parameter_change"],
            }
        )
    failed = [row["parameter"] for row in rows if not (row["gradient_ok"] and row["change_ok"])]
    receipt = {
        "parameters": rows,
        "min_margin": min(row["margin"] for row in rows),
        "barely_live_parameters": [row["parameter"] for row in rows if row["barely_live"]],
        "floors": floor,
    }
    if failed:
        from scripts.training.u2_update_executor_22_4_1 import ActivityGateError

        error = ActivityGateError(f"Frozen activity floor failed: {failed}")
        error.receipt = receipt
        raise error
    return receipt


def output_displacement(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "next_output_l1": float(
            (torch.tensor(before["next_output"]) - torch.tensor(after["next_output"])).abs().sum()
        ),
        "policy_l1": float(
            (torch.tensor(before["policy"]) - torch.tensor(after["policy"])).abs().sum()
        ),
    }
