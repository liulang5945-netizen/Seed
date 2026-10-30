"""N178 正式阶段：parser 加压训练（3 seed）+ 在已登记的 face2 上只读评估解析覆盖。

训练：父端点 = N177.6 correct/seed0；每 seed 从该父重新起步，按"含旧值轮 / 未覆盖实体"
加权采样、只对 parser 损失反传、只更新 producer，1,280 步、批 2；端点存盘。
评估：父端点（baseline）与 3 个端点各自在 face2 的 64 条查询上跑一次，按模板类 × 实体组
报"能否抽出实体"。face2 在训练前已登记（sha256 固定），此处只读。

预算 15,000 process-CPU 秒；每 seed 上限 4,500 秒；触顶写 partial，不补跑。
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import (  # noqa: E402
    _make_turn,
    build_train_episodes,
    e1_families,
)
from taiji.joint_state_consumption_face2 import face2_contract, face2_items  # noqa: E402
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1  # noqa: E402
from taiji.joint_t1_trainer import JointT1TrainerV1  # noqa: E402

PARENT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
RUN_ROOT = ROOT / "data/checkpoints/n1_plan178_20261030"
REPORT = ROOT / "reports/taiji_n1_plan178_20261030.json"
FACE2 = ROOT / "reports/taiji_n178_face2_contract_20261030.json"
COVERED = ("钥匙", "杯子", "书", "盒子")
UPDATES = 1280
BATCH = 2
SEED_CPU_CAP = 4500.0
TOTAL_CPU_CAP = 15000.0


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _trainer() -> JointT1TrainerV1:
    cache = JointT1TraceCacheV1.from_payload(
        torch.load(TRACE_CACHE, map_location="cpu", weights_only=False)
    )
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    trainer.input_trace_cache = None
    return trainer


def _weights(episodes: list[Any], entity_by_family: dict[str, str]) -> torch.Tensor:
    weights = []
    for episode in episodes:
        text = "".join(
            bytes(turn.frame.payload).decode("utf-8", "replace") for turn in episode.turns
        )
        weight = 1.0
        if ("之前" in text) or ("以前" in text):
            weight += 2.0
        if entity_by_family[episode.family_id] not in COVERED:
            weight += 2.0
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.float64)
    return tensor / tensor.sum()


def _face2_rows(trainer: JointT1TrainerV1) -> list[dict[str, Any]]:
    family = e1_families()["train"][0]
    rows = []
    for item in face2_items():
        turn = _make_turn(
            family=family,
            episode_id=f"face2:{item.item_id}",
            variant=0,
            turn_index=0,
            text=item.text,
            answer="",
            act="query",
            spans=(),
        )
        adapter, joint = trainer._new_episode_state(f"face2:{item.item_id}")
        trace, joint = trainer._append_turn(adapter, joint, turn)
        parse = trainer.producer.predict(joint, trace)
        spans = {span.role: span.text for span in getattr(parse, "spans", ())}
        rows.append(
            {
                "item_id": item.item_id,
                "template_class": item.template_class,
                "entity": item.entity,
                "entity_covered": item.entity in COVERED,
                "parsed_entity": spans.get("entity") == item.entity,
                "predicted_act": getattr(parse, "act", None),
            }
        )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = collections.Counter()
    for row in rows:
        cells[
            (
                row["template_class"],
                "covered" if row["entity_covered"] else "uncovered",
                row["parsed_entity"],
            )
        ] += 1
    out: dict[str, Any] = {}
    for template in ("ordinary", "old"):
        for group in ("covered", "uncovered"):
            hits = cells[(template, group, True)]
            total = hits + cells[(template, group, False)]
            out[f"{template}_{group}"] = {"parsed": hits, "total": total}
    out["overall"] = {
        "parsed": sum(row["parsed_entity"] for row in rows),
        "total": len(rows),
    }
    return out


def _train_one(
    seed: int,
    episodes: list[Any],
    weights: torch.Tensor,
    updates: int,
    run_root_base: Path,
    seed_cpu_cap: float = SEED_CPU_CAP,
) -> dict[str, Any]:
    trainer = _trainer()
    # 种必须在**加载检查点之后**设：from_native_checkpoint 会恢复 torch_rng_state，
    # 先设就会被覆盖（2026-10-30 实测：三个 seed 端点逐字节相同）。
    torch.manual_seed(seed)
    run_root = run_root_base / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    cpu_start = time.process_time()
    steps = 0
    last_loss = None
    status = "completed_updates"
    for _ in range(updates):
        if time.process_time() - cpu_start >= seed_cpu_cap:
            status = "seed_cpu_cap"
            break
        picks = torch.multinomial(weights, BATCH, replacement=True).tolist()
        terms = []
        for index in picks:
            episode = episodes[index]
            adapter, joint = trainer._new_episode_state(episode.episode_id)
            turn_losses = []
            for turn in episode.turns:
                trace, joint = trainer._append_turn(adapter, joint, turn)
                semantic_loss, _ = trainer._semantic_loss(joint, trace, turn)
                turn_losses.append(semantic_loss)
            terms.append(torch.stack(turn_losses).mean())
        loss = torch.stack(terms).mean()
        trainer.producer_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.producer_optimizer.step()
        trainer.language_trainer.optimizer.zero_grad(set_to_none=True)
        steps += 1
        last_loss = float(loss.detach())
    endpoint = run_root / f"endpoint_update_{steps:04d}.pt"
    torch.save(trainer.checkpoint(), endpoint)
    return {
        "seed": seed,
        "updates": steps,
        "status": status,
        "endpoint": _rel(endpoint),
        "last_parser_loss": last_loss,
        "process_cpu_seconds": time.process_time() - cpu_start,
        "episodes_seen": steps * BATCH,
        "_trainer": trainer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N178 parser 加压正式阶段")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--seed-cpu-cap", type=float, default=SEED_CPU_CAP)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("N178 report is write-once")
    if not FACE2.is_file():
        raise FileNotFoundError("face2 must be registered before training")
    registered = json.loads(FACE2.read_text(encoding="utf-8"))
    if face2_contract()["face_sha256"] != registered["face_sha256"]:
        raise ValueError("face2 changed after registration")

    entity_by_family = {family.family_id: family.entity for family in e1_families()["train"]}
    episodes = list(build_train_episodes())
    weights = _weights(episodes, entity_by_family)

    baseline_rows = _face2_rows(_trainer())
    runs = []
    total_cpu = 0.0
    for seed in (int(item) for item in args.seeds.split(",") if item.strip()):
        if total_cpu >= TOTAL_CPU_CAP:
            break
        run = _train_one(seed, episodes, weights, args.updates, args.run_root, args.seed_cpu_cap)
        trainer = run.pop("_trainer")
        total_cpu += run["process_cpu_seconds"]
        run["face2"] = _summary(_face2_rows(trainer))
        runs.append(run)
        print(
            f"seed {seed}: updates={run['updates']} cpu={run['process_cpu_seconds']:.1f} "
            f"loss={run['last_parser_loss']} face2={run['face2']['overall']}",
            flush=True,
        )
    report = {
        "format": "taiji-n178-v1",
        "date": "2026-10-30",
        "scope": (
            "parser-only pressure on the existing train course; face2 (pre-registered, "
            "64 unseen query phrasings) read once after training; no V4 update"
        ),
        "parent": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "face2_sha256": registered["face_sha256"],
        "updates_per_seed": UPDATES,
        "batch": BATCH,
        "baseline_face2": _summary(baseline_rows),
        "runs": runs,
        "total_process_cpu_seconds": total_cpu,
        "total_cpu_cap": TOTAL_CPU_CAP,
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    print(
        json.dumps(
            {"baseline": report["baseline_face2"], "total_cpu": round(total_cpu, 1)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
