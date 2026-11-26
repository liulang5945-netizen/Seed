"""R4 preflight: real train episodes through the three objects, no optimizer step.

R4 execution is gated on the user's review of R2/R4.  This preflight does the
engineering checks that can be done without training:

* load only the train split and replay the 8-turn timeline per object;
* build teacher-forced losses for every scored turn with the correct store
  state at that turn (no future facts);
* run one forward/backward per object and check finite loss/gradients;
* zero-step bundle save/load in-process and in a fresh process.

It never calls ``optimizer.step()``, never evaluates query/holdout, and never
writes a trained checkpoint.  ``optimizer_updates=0`` is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from taiji.conditional_language_v3 import TaskCoreConfigV3  # noqa: E402
from taiji.corrected_old_baseline import (  # noqa: E402
    CorrectedOldBaselineV1,
    CorrectedOldBundleV1,
    DiscreteFactStoreV1,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.native_content_core import (  # noqa: E402
    NativeContentBundleV1,
    NativeContentCoreConfigV1,
    NativeContentCoreV1,
    NativeContentStoreV1,
)
from taiji.ordinary_baseline import OrdinaryBundleV1, OrdinaryEndToEndV1  # noqa: E402

TRAIN = ROOT / "plans/manifests/native_core_r1_train_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r4_preflight_20261030.json"
EPISODES = 4
OWNER, SCOPE = "train", "preflight"
END_MARKER = b"<|end|>"

TIME_FOR_CATEGORY = {
    "current_query": "current",
    "previous_query": "previous",
    "unknown_query": "unknown",
}
FACT_ROLES = {"fact_v0", "correction", "distractor"}


def _native_config() -> NativeContentCoreConfigV1:
    return NativeContentCoreConfigV1(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        content_width=32,
        seed=401,
        position_extension_seed=402,
        read_seed=403,
    )


def _baseline_config() -> TaskCoreConfigV3:
    return TaskCoreConfigV3(
        width=64,
        embedding_width=32,
        key_width=32,
        slots=8,
        chunk_bytes=8,
        max_observation_bytes=512,
        max_input_evidence_bytes=512,
        max_output_position=128,
        max_generation_bytes=64,
        seed=411,
        position_extension_seed=412,
    )


def _load_episodes(limit: int = EPISODES) -> list[dict[str, Any]]:
    payload = json.loads(TRAIN.read_text(encoding="utf-8"))
    if len(payload["episodes"]) < limit:
        raise ValueError("train split has fewer episodes than the preflight batch")
    return payload["episodes"][:limit]


def _session(episode: dict[str, Any]) -> str:
    return episode["episode_id"].replace(":", "_")


def _ingest_native(bundle: NativeContentBundleV1, episode: dict[str, Any], turn: dict[str, Any]):
    raw = turn["text"].encode("utf-8")
    bundle.ingest(
        raw_span=raw,
        source_input_id=f"{episode['episode_id']}:{turn['index']}",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        owner=OWNER,
        scope=SCOPE,
        session=_session(episode),
        turn_index=turn["index"],
        revision=0 if turn["role"] == "fact_v0" or turn["role"] == "distractor" else 1,
        valid_from=turn["index"],
        entity=turn["entity"],
        attribute=turn["attribute"],
        value=turn["value"],
        status="resolved",
    )


def _ingest_corrected(bundle: CorrectedOldBundleV1, episode: dict[str, Any], turn: dict[str, Any]):
    bundle.ingest(
        source_text=turn["text"],
        entity=turn["entity"],
        attribute=turn["attribute"],
        value=turn["value"],
        source_input_id=f"{episode['episode_id']}:{turn['index']}",
        owner=OWNER,
        scope=SCOPE,
        session=_session(episode),
        turn_index=turn["index"],
        revision=0 if turn["role"] == "fact_v0" or turn["role"] == "distractor" else 1,
        valid_from=turn["index"],
    )


def native_turn_logits(
    bundle: NativeContentBundleV1, episode: dict[str, Any], turn: dict[str, Any]
) -> torch.Tensor:
    assert turn["category"] is not None
    condition = bundle.condition(
        query=turn["text"].encode("utf-8"),
        time_intent=TIME_FOR_CATEGORY[turn["category"]],
        owner=OWNER,
        scope=SCOPE,
        session=_session(episode),
    )
    return bundle.model.teacher_forced_conditioned(
        condition,
        bytes(turn["target"]["target_bytes_utf8"]),
        scope_id=SCOPE,
        session_id=_session(episode),
    )


def corrected_turn_logits(
    bundle: CorrectedOldBundleV1, episode: dict[str, Any], turn: dict[str, Any]
) -> torch.Tensor:
    assert turn["category"] is not None
    logits, _ = bundle.score_target(
        query=turn["text"].encode("utf-8"),
        target=bytes(turn["target"]["target_bytes_utf8"]),
        time_intent=TIME_FOR_CATEGORY[turn["category"]],
        owner=OWNER,
        scope=SCOPE,
        session=_session(episode),
    )
    return logits


def ordinary_turn_logits(
    bundle: OrdinaryBundleV1, episode: dict[str, Any], turn: dict[str, Any]
) -> torch.Tensor:
    assert turn["category"] is not None
    del episode
    return bundle.score_target(
        query=turn["text"].encode("utf-8"),
        target=bytes(turn["target"]["target_bytes_utf8"]),
        time_intent=TIME_FOR_CATEGORY[turn["category"]],
    )


def build_losses() -> dict[str, dict[str, Any]]:
    episodes = _load_episodes()
    models = {
        "new_core_v1": NativeContentCoreV1(_native_config()),
        "corrected_old_v1": CorrectedOldBaselineV1(_baseline_config()),
        "ordinary_v1": OrdinaryEndToEndV1(_baseline_config()),
    }
    totals = {
        "new_core_v1": {"loss": torch.zeros(()), "bytes": 0, "turns": 0},
        "corrected_old_v1": {"loss": torch.zeros(()), "bytes": 0, "turns": 0},
        "ordinary_v1": {"loss": torch.zeros(()), "bytes": 0, "turns": 0},
    }
    for episode in episodes:
        native = NativeContentBundleV1(
            model=models["new_core_v1"], store=NativeContentStoreV1(()), cursor={}
        )
        corrected = CorrectedOldBundleV1(
            model=models["corrected_old_v1"], store=DiscreteFactStoreV1(()), cursor={}
        )
        ordinary = OrdinaryBundleV1(model=models["ordinary_v1"], history=(), cursor={})
        for turn in episode["turns"]:
            if turn["role"] in FACT_ROLES:
                _ingest_native(native, episode, turn)
                _ingest_corrected(corrected, episode, turn)
            if not turn["scored"]:
                ordinary.observe(turn["text"])
                continue
            target = torch.tensor(turn["target"]["target_bytes_utf8"], dtype=torch.long)
            for name, logits in (
                ("new_core_v1", native_turn_logits(native, episode, turn)),
                ("corrected_old_v1", corrected_turn_logits(corrected, episode, turn)),
                ("ordinary_v1", ordinary_turn_logits(ordinary, episode, turn)),
            ):
                totals[name]["loss"] = totals[name]["loss"] + F.cross_entropy(
                    logits, target, reduction="sum"
                )
                totals[name]["bytes"] += int(len(target))
                totals[name]["turns"] += 1
            ordinary.observe(turn["text"])
    return {
        "episodes": len(episodes),
        "models": models,
        "objects": {
            name: {
                "loss": item["loss"] / item["bytes"],
                "target_bytes": item["bytes"],
                "scored_turns": item["turns"],
            }
            for name, item in totals.items()
        },
    }


def _finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _restore_probe() -> dict[str, Any]:
    native = NativeContentBundleV1.create(_native_config())
    raw = "雨伞的颜色是红色。".encode()
    native.ingest(
        raw_span=raw,
        source_input_id="preflight:restore:0",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        owner="probe",
        scope="probe",
        session="probe",
        turn_index=0,
        revision=0,
        entity="雨伞",
        attribute="颜色",
        value="红色",
        status="resolved",
    )
    query = "雨伞的颜色是什么？".encode()
    expected = native.probe_digest(query=query)
    path = ROOT / ".seed_test_tmp" / "r4_preflight_native.pt"
    if path.exists():
        path.unlink()
    native.save(path)
    loaded = NativeContentBundleV1.load(path)
    in_process = loaded.probe_digest(query=query)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys; "
        "from taiji.native_content_core import NativeContentBundleV1; "
        "b=NativeContentBundleV1.load(sys.argv[1]); "
        "print(b.probe_digest(query=sys.argv[2].encode()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), query.decode("utf-8")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    fresh = result.stdout.strip()
    return {
        "in_process_equal": in_process == expected,
        "fresh_process_equal": fresh == expected,
        "expected_digest": expected,
        "in_process_digest": in_process,
        "fresh_process_digest": fresh,
    }


def _probe() -> dict[str, Any]:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()
    built = build_losses()
    report: dict[str, Any] = {
        "schema": "taiji-native-core-r4-preflight-v1",
        "episodes": built["episodes"],
        "train_only": True,
        "query_read": False,
        "holdout_read": False,
        "optimizer_updates": 0,
        "training_started": False,
        "sealed_text_read": 0,
        "can_promote": False,
        "objects": {},
    }
    for name, payload in built["objects"].items():
        model = built["models"][name]
        loss = payload["loss"]
        model.zero_grad(set_to_none=True)
        loss.backward()
        groups: dict[str, bool] = {}
        if hasattr(model, "credit_groups"):
            for group, names in model.credit_groups().items():
                groups[group] = any(
                    parameter.grad is not None
                    for parameter_name, parameter in model.named_parameters()
                    if parameter_name in names
                )
        finite = _finite_gradients(model)
        report["objects"][name] = {
            "loss_nats": float(loss.detach()),
            "finite_loss": bool(torch.isfinite(loss)),
            "finite_gradients": finite,
            "gradient_groups": groups,
            "target_bytes": payload["target_bytes"],
            "scored_turns": payload["scored_turns"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
        model.zero_grad(set_to_none=True)
    report["restore"] = _restore_probe()
    report["cpu_seconds"] = time.process_time() - started_cpu
    report["wall_seconds"] = time.perf_counter() - started_wall
    failures = [
        f"{name}: non-finite loss"
        for name, item in report["objects"].items()
        if not item["finite_loss"]
    ] + [
        f"{name}: non-finite gradients"
        for name, item in report["objects"].items()
        if not item["finite_gradients"]
    ]
    if not report["restore"]["in_process_equal"] or not report["restore"]["fresh_process_equal"]:
        failures.append("zero-step restore differs")
    report["status"] = "r4_preflight_passed" if not failures else "r4_preflight_failed"
    report["failures"] = failures
    report["source_sha256"] = {
        "scripts/training/preflight_native_content_core_r4.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
    }
    report["digest"] = content_digest(
        {
            "objects": {
                name: {key: value for key, value in item.items() if key != "gradient_groups"}
                for name, item in report["objects"].items()
            },
            "restore": report["restore"],
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.report.exists():
        raise SystemExit(f"report already exists (write-once): {args.report}")
    report = _probe()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps({"status": report["status"], "failures": report["failures"]}, ensure_ascii=False)
    )
    return 0 if report["status"] == "r4_preflight_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
