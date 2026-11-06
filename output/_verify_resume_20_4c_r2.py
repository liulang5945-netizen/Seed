"""Load-only checkpoint verification for the 20.4c r2 stop (no forward, no step)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(r"e:\Seed")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.rule_learning_20_1_net import (  # noqa: E402
    RuleLearningGraph,
    _group_of,
    copy_subject,
)
from taiji.selection_routing_checkpoint import load_routing_bundle  # noqa: E402

CKPT = ROOT / "output/taiji_package20_4c_r2_checkpoints/B_EH_20261105_075.pt"
RECORDED_SHA = "13e294f0e1df52be2c1ff2969e4bc66508bfd3256fa730700ea74a074202d101"
BUNDLE = ROOT / "output/taiji_package19_compare/28031-B_stop_lock-step300.pt"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(tensors: dict) -> str:
    rows = []
    for name, value in sorted(tensors.items()):
        rows.append(
            (
                name,
                str(value.dtype),
                tuple(value.shape),
                sha(value.detach().cpu().numpy().tobytes()),
            )
        )
    return sha(repr(rows).encode("utf-8"))


def state_dict_digest(state_dict: dict) -> str:
    rows = []
    for key, value in sorted(state_dict["state"].items()):
        for name, item in sorted(value.items()):
            if isinstance(item, torch.Tensor):
                rows.append(
                    (
                        key,
                        name,
                        str(item.dtype),
                        tuple(item.shape),
                        sha(item.detach().cpu().numpy().tobytes()),
                    )
                )
            else:
                rows.append(
                    (
                        key,
                        name,
                        float(item) if isinstance(item, (int, float)) else str(item),
                    )
                )
    groups = [
        {key: (list(value) if isinstance(value, list) else value) for key, value in group.items()}
        for group in state_dict["param_groups"]
    ]
    rows.append(("param_groups", json.dumps(groups, sort_keys=True)))
    return sha(repr(rows).encode("utf-8"))


def rng_digest(state: torch.Tensor) -> str:
    return sha(state.cpu().numpy().tobytes())


def main() -> None:
    file_sha = sha(CKPT.read_bytes())
    first = torch.load(CKPT, map_location="cpu", weights_only=False)
    second = torch.load(CKPT, map_location="cpu", weights_only=False)

    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    candidate = load_routing_bundle(BUNDLE)[0]
    gate = dict(bundle["growth_gates"][0]) if bundle.get("growth_gates") else None
    graph = RuleLearningGraph(
        copy_subject(candidate.subject),
        arm=first["arm"],
        growth_gates=gate,
        addressed_store_enabled=False,
    )
    load_report = graph.subject.load_state_dict(first["subject"], strict=True)
    for name, parameter in graph.subject.named_parameters():
        parameter.requires_grad_(_group_of(name) != "world")
    optimizer = torch.optim.Adam(
        [p for p in graph.subject.parameters() if p.requires_grad], lr=1e-3
    )
    optimizer.load_state_dict(second["optimizer"])
    torch.set_rng_state(second["rng"])

    weights_first = tensor_digest(dict(first["subject"]))
    weights_second = tensor_digest(dict(second["subject"]))
    weights_model = tensor_digest(dict(graph.subject.state_dict()))
    opt_first = state_dict_digest(first["optimizer"])
    opt_second = state_dict_digest(second["optimizer"])
    rng_first = rng_digest(first["rng"])
    rng_second = rng_digest(second["rng"])

    evidence = {
        "schema": "taiji-rule-learning-20-4c-r2-stop-evidence-v1",
        "evidence": "20_4c_r2_stop",
        "stop_reason": "user_reboot",
        "run": {"arm": first["arm"], "seed": int(first["seed"])},
        "global_step": int(first["updates"]),
        "cursor": int(first["cursor"]),
        "checkpoint": {
            "path": str(CKPT.relative_to(ROOT)),
            "sha256": file_sha,
            "recorded_sha256": RECORDED_SHA,
            "sha_matches_recording": file_sha == RECORDED_SHA,
        },
        "reload_only": {
            "no_forward": True,
            "no_step": True,
            "weights_digest": weights_first,
            "weights_identical_across_loads": weights_first == weights_second,
            "weights_identical_after_restore": weights_first == weights_model,
            "optimizer_digest": opt_first,
            "optimizer_identical_across_loads": opt_first == opt_second,
            "optimizer_state_entries": len(first["optimizer"]["state"]),
            "rng_digest": rng_first,
            "rng_identical_across_loads": rng_first == rng_second,
            "cursor_identical_across_loads": first["cursor"] == second["cursor"],
            "accum_identical_across_loads": first["accum"] == second["accum"],
            "state_dict_load": {
                "missing_keys": list(load_report.missing_keys),
                "unexpected_keys": list(load_report.unexpected_keys),
            },
        },
        "resume_verified": all(
            [
                file_sha == RECORDED_SHA,
                weights_first == weights_second == weights_model,
                opt_first == opt_second,
                rng_first == rng_second,
                first["cursor"] == second["cursor"],
                first["accum"] == second["accum"],
                not load_report.missing_keys,
                not load_report.unexpected_keys,
            ]
        ),
    }
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    (ROOT / "reports/taiji_package20_4c_r2_stop_20261106.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
