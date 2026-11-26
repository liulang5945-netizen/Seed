"""22.8 S2 (path A): same RouteB subject across 3 sessions x 32 joint-segment rounds (0 check).

Subject: fresh 30-unit RouteBGraph with five task heads (appearance 24-class, dynamics
3-class final bin, occlusion binary, correction 4-class subset, no-prompt 3-action
choice).  Each round trains the shared units plus the round's head on that type's pool;
sessions are separated by capsule save/load recovery; a label correction is applied at
the start of session 3 and its net gain/old retention measured.  Gates J1-J6 per the
amended pre-registration.  Zero check/final.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training.check_report_run_conditions import write_report  # noqa: E402
from scripts.training.curate_s2_22_6_pool_20261120 import (  # noqa: E402
    policy_utilities,
    tick_stream,
)
from scripts.training.first_a_support_22_5_v2 import generate_examples  # noqa: E402
from scripts.training.probe_s2_22_7_0_alpha_adapter_v3_20261120 import (  # noqa: E402
    adapter_v3_features,
)
from scripts.training.probe_s2_22_7_joint24_20261120 import joint_label  # noqa: E402
from scripts.training.probe_s2_action_consequence_20261120 import build_partition  # noqa: E402
from taiji.u2_multihead import MultiHeadGraph  # noqa: E402
from taiji.u2_neural_units import (  # noqa: E402
    UnitGraphSpec,
    apply_capsule,
    build_capsule,
    load_capsule,
    save_capsule,
)

OUT = ROOT / "reports/taiji_u2_22_8_s2_20261120.json"
CAP_CPU = 3850.0
SEEDS = (20261120, 20261121, 20261122)
SESSIONS = 3
ROUNDS = 32
STEPS_PER_ROUND = 8
BATCH = 32
LR = 3e-3
WARMUP = 50
OUTPUT = 24
POSITIONS = ("左边", "中间", "右边", "未知")
CHANNELS = ("rgb", "audio", "proprio", "question", "goal", "feedback")
ACTION_DIMS = (37, 38)
TYPES = (("appearance", 8), ("dynamics", 8), ("occlusion", 6), ("correction", 4), ("no_prompt", 6))
HEAD = {"appearance": 0, "dynamics": 1, "occlusion": 2, "correction": 3, "no_prompt": 4}


def lr_at(step: int, total: int) -> float:
    if step < WARMUP:
        return LR * float(step + 1) / float(WARMUP)
    span = max(1, total - WARMUP)
    return LR * 0.5 * (1.0 + math.cos(math.pi * float(step - WARMUP) / float(span)))


def channels(features: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: features[:, index * 8 : (index + 1) * 8] for index, name in enumerate(CHANNELS)}


def build_data(seed: int) -> dict[str, Any]:
    appearance_train = generate_examples("train", 256)
    appearance_fit = generate_examples("fit", 32)
    x_app = torch.stack([adapter_v3_features(item) for item in appearance_train]).float()
    y_app = torch.tensor([joint_label(item) for item in appearance_train])
    x_fit = torch.stack([adapter_v3_features(item) for item in appearance_fit]).float()
    y_fit = torch.tensor([joint_label(item) for item in appearance_fit])
    x_occ = x_app.clone()
    x_occ[:, 0:24] = 0.0
    y_occ = torch.ones(x_app.shape[0], dtype=torch.long)
    x_occ_fit = x_fit.clone()
    x_occ_fit[:, 0:24] = 0.0
    y_occ_fit = torch.ones(x_fit.shape[0], dtype=torch.long)
    action_rows = build_partition("train", 256, seed)
    x_dyn = torch.stack([row["features"] for row in action_rows]).float()
    y_dyn = torch.tensor([row["final_bin"] for row in action_rows])
    dyn_fit_rows = build_partition("fit", 64, seed + 1)
    x_dyn_fit = torch.stack([row["features"] for row in dyn_fit_rows]).float()
    y_dyn_fit = torch.tensor([row["final_bin"] for row in dyn_fit_rows])
    no_prompt = []
    for world in range(32):
        ticks = tick_stream(seed + world)
        present = list(ticks[0]["observed"])
        for name in present:
            truth = ticks[-1]["truth"][name]
            observed = [tick["observed"][name] for tick in ticks]
            utility = policy_utilities({"truth": truth, "observed": observed})
            context = torch.zeros(48)
            for index, other in enumerate(present[:3]):
                value = ticks[0]["observed"][other]
                context[index * 4 + POSITIONS.index(value)] = 1.0
            context[12 + present.index(name)] = 1.0
            actions = ("rest_then_answer", "observe_t0_answer", "observe_deadline_answer")
            values = [utility[action] for action in actions]
            no_prompt.append(
                {"features": context, "action": int(np.argmax(values)), "values": values}
            )
    conformity = torch.ones(x_app.shape[0], dtype=torch.long)
    return {
        "x_app": x_app,
        "y_app": y_app,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_occ": x_occ,
        "y_occ": y_occ,
        "x_occ_fit": x_occ_fit,
        "y_occ_fit": y_occ_fit,
        "x_dyn": x_dyn,
        "y_dyn": y_dyn,
        "x_dyn_fit": x_dyn_fit,
        "y_dyn_fit": y_dyn_fit,
        "no_prompt": no_prompt,
        "x_app_fit": x_fit,
        "y_app_fit": y_fit,
        "conformity": conformity,
    }


def build_model(seed: int) -> MultiHeadGraph:
    torch.manual_seed(seed)
    spec = UnitGraphSpec(
        groups=6, units_per_group=5, input_dim=8, hidden_dim=8, output_dim=OUTPUT, seed=seed
    )
    return MultiHeadGraph(spec, heads=5)


def head_logits(graph: MultiHeadGraph, features: torch.Tensor, head: int) -> torch.Tensor:
    gates = torch.ones(features.shape[0], graph.units.n_units)
    return graph(channels(features), gates, graph.initial_state(), head=head)["readout"]


def train_batch(
    graph: MultiHeadGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    head: int,
    steps: int,
    generator: torch.Generator,
    lr_scale: float = 1.0,
) -> None:
    optimizer = torch.optim.Adam(graph.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_scale * LR
        index = torch.randint(0, features.shape[0], (BATCH,), generator=generator)
        logits = head_logits(graph, features[index], head)
        k = int(labels.max()) + 1
        loss = loss_fn(logits[:, :k], labels[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        graph.units.project_parameters_()


def evaluate(graph: MultiHeadGraph, bundle: dict) -> dict[str, float]:
    with torch.no_grad():
        app = head_logits(graph, bundle["x_app_fit"], HEAD["appearance"])[:, :24]
        app_acc = float((app.argmax(-1) == bundle["y_app_fit"]).float().mean())
        dyn = head_logits(graph, bundle["x_dyn_fit"], HEAD["dynamics"])[:, :3]
        dyn_acc = float((dyn.argmax(-1) == bundle["y_dyn_fit"]).float().mean())
        masked = bundle["x_dyn_fit"].clone()
        masked[:, list(ACTION_DIMS)] = 0.0
        dyn_masked = head_logits(graph, masked, HEAD["dynamics"])[:, :3]
        dyn_masked_acc = float((dyn_masked.argmax(-1) == bundle["y_dyn_fit"]).float().mean())
        occ = head_logits(graph, bundle["x_occ_fit"], HEAD["occlusion"])[:, :2]
        occ_acc = float((occ.argmax(-1) == bundle["y_occ_fit"]).float().mean())
        np_features = torch.stack([item["features"] for item in bundle["no_prompt"]])
        np_logits = head_logits(graph, np_features, HEAD["no_prompt"])[:, :3]
        np_actions = np_logits.argmax(-1).tolist()
        np_utility = float(
            np.mean([bundle["no_prompt"][i]["values"][a] for i, a in enumerate(np_actions)])
        )
        fixed = [
            float(np.mean([item["values"][action] for item in bundle["no_prompt"]]))
            for action in range(3)
        ]
        idle = float(np.mean([action == 0 for action in np_actions]))
    return {
        "appearance": app_acc,
        "dynamics": dyn_acc,
        "dynamics_masked": dyn_masked_acc,
        "action_margin": dyn_acc - dyn_masked_acc,
        "occlusion": occ_acc,
        "no_prompt_utility": np_utility,
        "best_fixed_utility": max(fixed),
        "idle_rate": idle,
    }


def correction(generator: torch.Generator) -> dict[str, float]:
    return {}


def run_seed(seed: int) -> dict[str, Any]:
    bundle = build_data(seed)
    graph = build_model(seed)
    generator = torch.Generator().manual_seed(seed + 17)
    h_flip = HEAD["correction"]
    flipped = (bundle["y_app"] + 12) % 24
    rows = []
    recovery = []
    correction_metrics: dict[str, float] | None = None
    for session in range(1, SESSIONS + 1):
        before = evaluate(graph, bundle)
        if session == 3:
            corrected_mask = (bundle["y_app"] % 4) == 0
            pre_flip = head_logits(graph, bundle["x_app"][corrected_mask], h_flip)[:, :24]
            pre_acc = float((pre_flip.argmax(-1) == flipped[corrected_mask]).float().mean())
            train_batch(graph, bundle["x_app"], flipped, h_flip, 200, generator, lr_scale=0.5)
            post_flip = head_logits(graph, bundle["x_app"][corrected_mask], h_flip)[:, :24]
            post_acc = float((post_flip.argmax(-1) == flipped[corrected_mask]).float().mean())
            train_batch(
                graph, bundle["x_app"], bundle["y_app"], h_flip, 200, generator, lr_scale=1.0
            )
            retained = head_logits(graph, bundle["x_app"][~corrected_mask], h_flip)[:, :24]
            old_acc = float(
                (retained.argmax(-1) == bundle["y_app"][~corrected_mask]).float().mean()
            )
            correction_metrics = {
                "corrected_subset_gain": post_acc - pre_acc,
                "corrected_subset_after": post_acc,
                "old_subset_after": old_acc,
                "retention_damage": max(0.0, 1.0 - old_acc),
            }
        order = []
        for name, count in TYPES:
            order.extend([name] * count)
        shuffle = torch.randperm(ROUNDS, generator=generator).tolist()
        for position in shuffle:
            kind = order[position]
            if kind == "appearance":
                train_batch(
                    graph,
                    bundle["x_app"],
                    bundle["y_app"],
                    HEAD["appearance"],
                    STEPS_PER_ROUND,
                    generator,
                )
            elif kind == "dynamics":
                train_batch(
                    graph,
                    bundle["x_dyn"],
                    bundle["y_dyn"],
                    HEAD["dynamics"],
                    STEPS_PER_ROUND,
                    generator,
                )
            elif kind == "occlusion":
                train_batch(
                    graph,
                    bundle["x_occ"],
                    bundle["y_occ"],
                    HEAD["occlusion"],
                    STEPS_PER_ROUND,
                    generator,
                )
            elif kind == "correction":
                train_batch(
                    graph,
                    bundle["x_app"],
                    bundle["y_app"],
                    HEAD["correction"],
                    STEPS_PER_ROUND,
                    generator,
                )
            elif kind == "no_prompt":
                features = torch.stack([item["features"] for item in bundle["no_prompt"]])
                labels = torch.tensor([item["action"] for item in bundle["no_prompt"]])
                train_batch(graph, features, labels, HEAD["no_prompt"], STEPS_PER_ROUND, generator)
        after = evaluate(graph, bundle)
        capsule = build_capsule(graph)
        with tempfile.TemporaryDirectory(dir=ROOT / ".seed_test_tmp") as directory:
            path = save_capsule(Path(directory) / f"session{session}.json", capsule)
            loaded = load_capsule(path)
        fresh = build_model(seed)
        apply_capsule(fresh, loaded)
        digest_a = {key: value.sum().item() for key, value in graph.state_dict().items()}
        digest_b = {key: value.sum().item() for key, value in fresh.state_dict().items()}
        recovery.append(
            {
                "session": session,
                "max_abs_digest_diff": max(abs(digest_a[k] - digest_b[k]) for k in digest_a),
            }
        )
        rows.append({"session": session, "before": before, "after": after})
    final = evaluate(graph, bundle)
    return {
        "seed": seed,
        "sessions": rows,
        "final": final,
        "recovery": recovery,
        "correction": correction_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines(0)[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    started = time.process_time()
    torch.set_num_threads(1)
    (ROOT / ".seed_test_tmp").mkdir(exist_ok=True)
    rows = [run_seed(seed) for seed in SEEDS]
    finals = [row["final"] for row in rows]
    app = [item["appearance"] for item in finals]
    dyn = [item["dynamics"] for item in finals]
    margin = [item["action_margin"] for item in finals]
    recovery_ok = all(
        all(check["max_abs_digest_diff"] <= 1e-10 for check in row["recovery"]) for row in rows
    )
    no_prompt = [item["no_prompt_utility"] for item in finals]
    best_fixed = [item["best_fixed_utility"] for item in finals]
    idle = [item["idle_rate"] for item in finals]
    correction_rows = [row["correction"] for row in rows]
    correction_ok = all(
        item is not None
        and item["corrected_subset_gain"] >= 0.05
        and item["retention_damage"] <= 0.02
        for item in correction_rows
    )
    gates = {
        "j1_appearance": min(app) >= 0.60,
        "j2_actions": min(dyn) >= 0.45 and min(margin) >= 0.05,
        "j3_correction": correction_ok,
        "j4_recovery": recovery_ok,
        "j5_no_prompt": all(
            no_prompt[i] >= best_fixed[i] + 0.05 and idle[i] <= 0.30 for i in range(len(SEEDS))
        ),
        "j6_breakdown": all(
            key in finals[0] for key in ("appearance", "dynamics", "occlusion", "no_prompt_utility")
        ),
    }
    if all(gates.values()):
        verdict = "joint_life_segment_verified"
    elif any(gates.values()):
        verdict = "joint_life_segment_partial"
    else:
        verdict = "joint_life_segment_negative"
    payload = {
        "schema": "taiji-u2-22-8-s2-v1",
        "card": "22.8 S2 path-A joint life segment on a RouteB subject",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_conditions": {
            "threads": 1,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "protocol": {
            "seeds": list(SEEDS),
            "sessions": SESSIONS,
            "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "types": TYPES,
        },
        "rows": rows,
        "finals": finals,
        "correction": correction_rows,
        "gates": gates,
        "verdict": verdict,
        "discipline": {"check_reads": 0, "final_reads": 0},
        "cost": {"cpu": round(time.process_time() - started, 2)},
    }
    write_report(Path(args.out), payload)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "gates": gates,
                "finals": finals,
                "correction": correction_rows,
                "cpu": payload["cost"]["cpu"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if verdict == "joint_life_segment_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
