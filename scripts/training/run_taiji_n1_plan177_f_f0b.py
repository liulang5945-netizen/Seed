"""N177 F0b：完整候选（producer+V4 冻结，value encoder+byte bias 训练）的短成本/恢复标定。

12 run = 4 臂 × 3 seed，总包上限 2,000 process-CPU 秒 / 1,500 wall 秒；每 run 的额度由
"包上限减去实测 setup，再 12 等分"现算（实测优先，不抄 F0a 的组件地板价）。

关键口径：**每一步更新都要重新跑冻结链**（producer 解析 → V4 query-only next-byte logits
→ bias），而不是在预计算 logits 上只训一个头——后者会把 F1 的价格低估几个量级。

按 owner 2026-10-29 裁定：训练面收窄到 N177.6 producer 能解析出的 4 个实体
（钥匙/杯子/书/盒子，16 个家族），另半边在报告里明写成未覆盖。

每 run：零检查点存盘 → 新进程恢复并比对前向 → 1 步预热（不计）→ 计量窗口 →
终点耐久存盘 → 新进程恢复并比对前向/优化器/cursor。只读 train，不读 query/holdout/sealed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_COLORS,
    build_train_episodes,
    e1_families,
)
from taiji.joint_state_value_encoder import FeatureByteBiasV1, ValueByteEncoderV1  # noqa: E402

BASE_CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f0b_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_f0b_20261029.json"
CHECKPOINT_FORMAT = "taiji-n177-f-f0b-checkpoint-v1"
ARMS = ("main_credit_on", "main_credit_stop", "copy_only", "no_state")
COVERED_ENTITIES = ("钥匙", "杯子", "书", "盒子")
UNCOVERED_ENTITIES = ("球", "手机", "笔", "眼镜")
BATCH = 4
ENVELOPE_CPU = 2000.0
ENVELOPE_WALL = 1500.0
WARMUP_UPDATES = 1
MAX_UPDATES = 64
# 终点存盘 + 两次新进程重放不在更新循环里，必须从额度里预留，否则每 run 会超几秒
# （2026-10-29 首跑实测：12 run 各超 6-9 秒，总 wall 1,596 s 超出 1,500 包络 96 s）。
ENDPOINT_RESERVE_CPU = 8.0
ENDPOINT_RESERVE_WALL = 12.0
BYTE_VOCAB = 257


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_bytes(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode("utf-8")), dtype=torch.long)


def load_trainer() -> Any:
    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    cache = JointT1TraceCacheV1.from_payload(
        torch.load(TRACE_CACHE, map_location="cpu", weights_only=False)
    )
    payload = torch.load(BASE_CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    trainer.input_trace_cache = None
    return trainer


_EPISODES: dict[str, Any] | None = None


def episode_for(family_id: str) -> Any:
    global _EPISODES
    if _EPISODES is None:
        _EPISODES = {
            item.family_id: item
            for item in build_train_episodes()
            if item.episode_id.endswith(":0")
        }
    return _EPISODES[family_id]


def train_items() -> list[dict[str, Any]]:
    """16 个已覆盖家族的元数据；链路在每一步更新里现跑，不用预计算 logits。"""

    items: list[dict[str, Any]] = []
    for family in e1_families()["train"]:
        if family.entity not in COVERED_ENTITIES:
            continue
        items.append(
            {
                "family_id": family.family_id,
                "entity": family.entity,
                "value": family.value,
                "target_byte": int(list(family.value.encode("utf-8"))[0]),
                "value_bytes": _value_bytes(family.value),
            }
        )
    return items


def chain_logits(trainer: Any, item: dict[str, Any]) -> torch.Tensor:
    """一步冻结链：producer 解析事实/查询，V4 给出 query-only 下一字节 logits。"""

    episode = episode_for(item["family_id"])
    with torch.no_grad():
        adapter, joint = trainer._new_episode_state(f"f0b:{episode.episode_id}")
        logits: torch.Tensor | None = None
        for turn in episode.turns:
            trace, joint = trainer._append_turn(adapter, joint, turn)
            parse = trainer.producer.predict(joint, trace)
            if getattr(parse, "act", None) == "query" and logits is None:
                source = joint.input_bindings[-1].neural_source_id
                evidence = trainer.model._joint_evidence_cache(joint, selected_source_ids=(source,))
                base, _hidden = trainer.model.next_logits(joint.language_state, cache=evidence)
                logits = base.detach().clone()
        if logits is None:
            raise RuntimeError(f"episode {episode.episode_id} yielded no query logits")
    return logits


class _Arm:
    def __init__(self, *, arm: str, seed: int, feature_dim: int = 64, byte_dim: int = 32) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown F0b arm: {arm}")
        torch.manual_seed(seed)
        random.seed(seed)
        self.arm = arm
        self.seed = int(seed)
        self.feature_dim = feature_dim
        self.byte_dim = byte_dim
        self.encoder = ValueByteEncoderV1(byte_dim=byte_dim, hidden_dim=feature_dim)
        self.bias = FeatureByteBiasV1(feature_dim=feature_dim, byte_vocab=BYTE_VOCAB)
        self.cursor = 0
        trainable: list[torch.nn.Parameter] = list(self.bias.parameters())
        if arm in {"main_credit_on", "main_credit_stop"}:
            trainable += list(self.encoder.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.0)

    def feature(self, item: dict[str, Any]) -> torch.Tensor:
        if self.arm == "no_state":
            return torch.zeros(self.feature_dim)
        if self.arm == "copy_only":
            vector = torch.zeros(self.feature_dim)
            vector[E1_COLORS.index(item["value"])] = 1.0
            return vector
        value_bytes = item.get("value_bytes")
        if value_bytes is None:
            value_bytes = _value_bytes(item["value"])
        encoded = self.encoder(value_bytes)
        if self.arm == "main_credit_stop":
            return encoded.detach()
        return encoded

    def step(self, trainer: Any, batch: list[dict[str, Any]]) -> dict[str, float]:
        losses = []
        hits = 0
        for item in batch:
            base = chain_logits(trainer, item)
            logits = base + self.bias(self.feature(item))
            target = torch.tensor([item["target_byte"]], dtype=torch.long)
            losses.append(F.cross_entropy(logits.reshape(1, -1), target))
            with torch.no_grad():
                hits += int(logits.argmax()) == item["target_byte"]
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.cursor += 1
        return {"batch_loss": float(loss.detach()), "batch_hits": float(hits)}

    def parameters_digest(self) -> dict[str, str]:
        return {
            "encoder": content_digest(self.encoder.state_dict()),
            "bias": content_digest(self.bias.state_dict()),
        }

    def checkpoint(self) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "arm": self.arm,
            "seed": self.seed,
            "cursor": self.cursor,
            "feature_dim": self.feature_dim,
            "byte_dim": self.byte_dim,
            "byte_vocab": BYTE_VOCAB,
            "encoder_state": {
                key: value.clone() for key, value in self.encoder.state_dict().items()
            },
            "bias_state": {key: value.clone() for key, value in self.bias.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "torch_rng_state": torch.random.get_rng_state().clone(),
            "python_rng_state": random.getstate(),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> _Arm:
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("F0b checkpoint has an unexpected format")
        arm = cls(
            arm=payload["arm"],
            seed=int(payload["seed"]),
            feature_dim=int(payload["feature_dim"]),
            byte_dim=int(payload["byte_dim"]),
        )
        arm.encoder.load_state_dict(
            {key: value.clone() for key, value in payload["encoder_state"].items()}
        )
        arm.bias.load_state_dict(
            {key: value.clone() for key, value in payload["bias_state"].items()}
        )
        arm.optimizer.load_state_dict(payload["optimizer"])
        arm.cursor = int(payload["cursor"])
        torch.random.set_rng_state(payload["torch_rng_state"].clone())
        random.setstate(payload["python_rng_state"])
        return arm


def forward_digest(arm: _Arm, trainer: Any, item: dict[str, Any]) -> dict[str, Any]:
    base = item["base_logits"] if "base_logits" in item else chain_logits(trainer, item)
    with torch.no_grad():
        logits = base + arm.bias(arm.feature(item))
        values = [float(x) for x in logits.tolist()]
    return {
        "logits_digest": content_digest({"logits": values}),
        "argmax": int(logits.argmax()),
        "parameters": arm.parameters_digest(),
        "optimizer": content_digest(arm.optimizer.state_dict()),
        "cursor": arm.cursor,
    }


def child_replay(checkpoint_path: Path, probe_path: Path) -> dict[str, Any]:
    script = (
        "import json, sys, torch\n"
        "torch.set_num_threads(1)\n"
        "from scripts.training.run_taiji_n1_plan177_f_f0b import _Arm, forward_digest\n"
        "payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)\n"
        "item = torch.load(sys.argv[2], map_location='cpu', weights_only=False)\n"
        "arm = _Arm.from_checkpoint(payload)\n"
        "print(json.dumps(forward_digest(arm, None, item)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint_path), str(probe_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1"),
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh replay failed: {completed.stderr[-800:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_one(
    arm_name: str,
    seed: int,
    items: list[dict[str, Any]],
    trainer: Any,
    run_root: Path,
    cpu_allowance: float,
    wall_allowance: float,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    probe_item = {key: items[0][key] for key in ("family_id", "entity", "value", "target_byte")}
    probe_item["base_logits"] = chain_logits(trainer, items[0])
    probe_path = run_root / "probe_item.pt"
    torch.save(probe_item, probe_path)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    arm = _Arm(arm=arm_name, seed=seed)
    zero_path = run_root / "zero.pt"
    zero_expectation = forward_digest(arm, trainer, probe_item)
    torch.save(arm.checkpoint(), zero_path)
    zero_child = child_replay(zero_path, probe_path)
    zero_restore_ok = (
        zero_child["logits_digest"] == zero_expectation["logits_digest"]
        and zero_child["parameters"] == zero_expectation["parameters"]
    )

    warmup_cpu = time.process_time()
    warmup_wall = time.perf_counter()
    arm.step(trainer, [items[index % len(items)] for index in range(BATCH)])
    warmup = {
        "process_cpu_seconds": time.process_time() - warmup_cpu,
        "wall_seconds": time.perf_counter() - warmup_wall,
    }

    measured: list[dict[str, Any]] = []
    boundary_reason = "max_updates"
    while arm.cursor < WARMUP_UPDATES + MAX_UPDATES:
        if time.process_time() - cpu_start >= cpu_allowance - ENDPOINT_RESERVE_CPU:
            boundary_reason = "cpu_allowance"
            break
        if time.perf_counter() - wall_start >= wall_allowance - ENDPOINT_RESERVE_WALL:
            boundary_reason = "wall_allowance"
            break
        offset = BATCH * arm.cursor
        batch = [items[(offset + index) % len(items)] for index in range(BATCH)]
        step_cpu = time.process_time()
        step_wall = time.perf_counter()
        stats = arm.step(trainer, batch)
        measured.append(
            {
                "update": arm.cursor,
                "batch_loss": stats["batch_loss"],
                "batch_hits": stats["batch_hits"],
                "process_cpu_seconds": time.process_time() - step_cpu,
                "wall_seconds": time.perf_counter() - step_wall,
            }
        )

    endpoint_path = run_root / f"endpoint_update_{arm.cursor:04d}.pt"
    endpoint_expectation = forward_digest(arm, trainer, probe_item)
    torch.save(arm.checkpoint(), endpoint_path)
    endpoint_child = child_replay(endpoint_path, probe_path)
    endpoint_restore_ok = (
        endpoint_child["logits_digest"] == endpoint_expectation["logits_digest"]
        and endpoint_child["optimizer"] == endpoint_expectation["optimizer"]
        and endpoint_child["cursor"] == endpoint_expectation["cursor"]
    )

    measured_cpu = [row["process_cpu_seconds"] for row in measured]
    measured_wall = [row["wall_seconds"] for row in measured]
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    return {
        "arm": arm_name,
        "seed": seed,
        "run_root": _rel(run_root),
        "updates": arm.cursor,
        "measured_updates": len(measured),
        "boundary_reason": boundary_reason,
        "zero_checkpoint_sha256": _sha256_file(zero_path),
        "endpoint_checkpoint_sha256": _sha256_file(endpoint_path),
        "zero_restore_ok": zero_restore_ok,
        "endpoint_restore_ok": endpoint_restore_ok,
        "zero_forward": zero_expectation["logits_digest"],
        "endpoint_forward": endpoint_expectation["logits_digest"],
        "endpoint_parameters": endpoint_expectation["parameters"],
        "warmup": warmup,
        "measured": measured,
        "cost": {
            "process_cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "cpu_per_measured_update": (
                sum(measured_cpu) / len(measured_cpu) if measured_cpu else None
            ),
            "wall_per_measured_update": (
                sum(measured_wall) / len(measured_wall) if measured_wall else None
            ),
        },
        "within_allowance": cpu_seconds <= cpu_allowance and wall_seconds <= wall_allowance,
    }


def main(argv: list[str] | None = None) -> int:
    global MAX_UPDATES
    parser = argparse.ArgumentParser(description="N177 F0b 12-run 短成本/恢复标定")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    parser.add_argument(
        "--runs",
        type=int,
        default=12,
        help="只跑前 N 个 run（仅用于冒烟；正式件必须 12）",
    )
    args = parser.parse_args(argv)
    MAX_UPDATES = args.max_updates
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("F0b report is write-once")
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock = args.run_root / "run_root.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"F0b run root already locked: {lock}") from error
    os.write(handle, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())
    os.close(handle)

    overall_cpu_start = time.process_time()
    overall_wall_start = time.perf_counter()
    trainer = load_trainer()
    items = train_items()
    if len(items) != 16:
        raise RuntimeError(f"F0b expects 16 train items, got {len(items)}")
    setup_cpu = time.process_time() - overall_cpu_start
    setup_wall = time.perf_counter() - overall_wall_start
    cpu_allowance = (ENVELOPE_CPU - setup_cpu) / 12.0
    wall_allowance = (ENVELOPE_WALL - setup_wall) / 12.0

    runs = []
    planned = [(seed, arm) for seed in (0, 1, 2) for arm in ARMS][: args.runs]
    for seed, arm_name in planned:
        run_root = args.run_root / f"{arm_name}__seed_{seed}"
        runs.append(
            run_one(arm_name, seed, items, trainer, run_root, cpu_allowance, wall_allowance)
        )
    total_cpu = time.process_time() - overall_cpu_start
    total_wall = time.perf_counter() - overall_wall_start
    verdicts = {
        "twelve_runs": len(runs) == 12,
        "training_faces_are_sixteen_families": len(items) == 16,
        "every_zero_restore_ok": all(run["zero_restore_ok"] for run in runs),
        "every_endpoint_restore_ok": all(run["endpoint_restore_ok"] for run in runs),
        "every_run_within_allowance": all(run["within_allowance"] for run in runs),
        "every_run_did_a_measured_update": all(run["measured_updates"] > 0 for run in runs),
        "within_envelope": total_cpu <= ENVELOPE_CPU and total_wall <= ENVELOPE_WALL,
    }
    report = {
        "format": "taiji-n177-f-f0b-v1",
        "date": "2026-10-29",
        "scope": (
            "4 arms x 3 seeds; frozen producer/V4 re-run every update, trained value "
            "encoder + byte bias; train-only 16 families on the four entities the N177.6 "
            "producer parses; no query/holdout/sealed read"
        ),
        "entities": list(COVERED_ENTITIES),
        "uncovered_entities": list(UNCOVERED_ENTITIES),
        "train_items": [
            {"family_id": item["family_id"], "entity": item["entity"], "value": item["value"]}
            for item in items
        ],
        "batch": BATCH,
        "warmup_updates": WARMUP_UPDATES,
        "max_updates": MAX_UPDATES,
        "per_run_allowance": {
            "process_cpu_seconds": cpu_allowance,
            "wall_seconds": wall_allowance,
            "source": "envelope minus measured setup, split 12 ways",
        },
        "setup_cost": {"process_cpu_seconds": setup_cpu, "wall_seconds": setup_wall},
        "base_checkpoint": _rel(BASE_CHECKPOINT),
        "base_checkpoint_sha256": _sha256_file(BASE_CHECKPOINT),
        "trace_cache": _rel(TRACE_CACHE),
        "trace_cache_sha256": _sha256_file(TRACE_CACHE),
        "runs": runs,
        "verdicts": verdicts,
        "all_verdicts_ok": all(verdicts.values()),
        "cost": {"process_cpu_seconds": total_cpu, "wall_seconds": total_wall},
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    lock.unlink(missing_ok=True)
    for key, value in verdicts.items():
        print(f"{key}={value}")
    print(f"ALL_VERDICTS_OK={report['all_verdicts_ok']} runs={len(runs)}")
    print(f"cost={report['cost']}")
    return 0 if report["all_verdicts_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
