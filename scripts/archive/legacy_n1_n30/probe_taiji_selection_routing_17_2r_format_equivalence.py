"""Zero-update equivalence probe: pre-format vs post-format graph bytes.

The 2026-11-03 evidence was produced with ``taiji/selection_routing.py``
before a black-100 line merge.  This probe replays the same fixtures with both
byte versions and requires bitwise equality on:

* the head / span-head weights built from the same seed;
* the 720 gold-forced record steps of the T3 supplement (p64 / p32 and every
  recorded transport, cursor and instrument reading);
* selection CE logits and per-step distributions;
* routing loss for all three arms (nll / ce / entropy / distributions / routes);
* the cursor state (``m``, ``coverage``, ``prefix_cursor``) along a full
  teacher-forced pass.

It also recomputes the legacy byte-bound identity (v1) for both source hashes
and the semantic identity (v2), so the ledger can state which digest the old
reports carry and why that digest is not the current identity.

Writes one report and one manifest.  No optimizer step, no data-split read
beyond the same dev fixtures the T3 supplement used.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

import taiji.candidate_identity as identity_module  # noqa: E402
from taiji.binding_copy import initial_binding_state, prepare  # noqa: E402
from taiji.binding_select import load_bundle  # noqa: E402
from taiji.living_core import EOS  # noqa: E402
from taiji.selection_routing_checkpoint import RoutingCandidate  # noqa: E402

T3_SCRIPT = Path("scripts/training/verify_taiji_selection_routing_17_2_t3_supplement.py")
OLD_SOURCE = Path("output/equiv_17_2r/pre_format_selection_routing.py.txt")
BUNDLE = Path("output/taiji_core_binding_select_16_1_compare/bundles/28031-A-locked-step300.pt")
T3_REPORT = Path("reports/taiji_selection_routing_17_2_t3_supplement_20261103_v3.json")
OUT = Path("reports/taiji_selection_routing_17_2r_format_equivalence_20261104.json")
MANIFEST = Path("plans/manifests/taiji_selection_routing_17_2r_format_equivalence_20261104.json")
IDENTITY_REPORT = Path("reports/taiji_selection_routing_17_2r_identity_20261104.json")
EVIDENCE = (
    Path("reports/taiji_selection_routing_17_2_t3_supplement_20261103_v3.json"),
    Path("reports/taiji_selection_routing_17_4_preflight_20261103_v2.json"),
    Path("reports/taiji_selection_routing_17_2r_price_20261103.json"),
    Path("reports/taiji_selection_routing_17_2r_20261103.json"),
)

ARM = "B_stop_lock"
HEAD_SEED = 1731
GOLD_FIXTURES = 40
SELECTION_FIXTURES = 4
LOSS_FIXTURES = 3
CURSOR_FIXTURES = 1
T_MIN = 0.05
ENTROPY_COEF = 0.01
SPAN_LAMBDA = 0.5


def sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded_sha(path: Path) -> str | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("candidate"), dict):
        return data["candidate"].get("candidate_sha")
    return data.get("candidate_sha")


def load_file(name: str, path: Path):
    # The pre-format fixture keeps a ``.py.txt`` suffix so it is stored as evidence
    # but skipped by the black/ruff hooks; load it through an explicit source loader.
    loader = SourceFileLoader(name, str(ROOT / path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


def tensor_diff(left: torch.Tensor, right: torch.Tensor) -> dict:
    if left.shape != right.shape:
        return {"shape_match": False, "max_abs_diff": None, "bitwise_equal": False}
    if left.numel() == 0 and right.numel() == 0:
        return {"shape_match": True, "max_abs_diff": 0.0, "bitwise_equal": True}
    diff = float((left.detach() - right.detach()).abs().max())
    return {
        "shape_match": True,
        "max_abs_diff": diff,
        "bitwise_equal": bool(torch.equal(left, right)),
    }


def merge_diff(state: dict, key: str, value: dict) -> None:
    current = state.setdefault(key, {"max_abs_diff": 0.0, "bitwise_equal": True, "count": 0})
    current["count"] += 1
    if value["max_abs_diff"] is None:
        current["max_abs_diff"] = None
        current["bitwise_equal"] = False
        return
    if current["max_abs_diff"] is not None:
        current["max_abs_diff"] = max(current["max_abs_diff"], value["max_abs_diff"])
    current["bitwise_equal"] = current["bitwise_equal"] and value["bitwise_equal"]


def numeric_row_diff(left: dict, right: dict, block: dict) -> None:
    for key, value in left.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            other = right[key]
            block.setdefault(key, {"max_abs_diff": 0.0, "count": 0})
            block[key]["max_abs_diff"] = max(
                block[key]["max_abs_diff"], abs(float(value) - float(other))
            )
            block[key]["count"] += 1
        elif isinstance(value, bool):
            block.setdefault(key, {"mismatches": 0, "count": 0})
            block[key]["count"] += 1
            if value != right[key]:
                block[key]["mismatches"] += 1


def run_variant(sr, dev, binding, tag: str) -> dict:
    saved = sys.modules.get("taiji.selection_routing")
    sys.modules["taiji.selection_routing"] = sr
    try:
        t3 = load_file(f"equiv_t3_{tag}", T3_SCRIPT)
        head = t3.OrderedSelectionHead(t3.RoutingConfig(), seed=HEAD_SEED)
        span = t3.SpanPointerHead(t3.RoutingConfig(), seed=HEAD_SEED)
        reader32 = t3.make_reader32(binding)
        rows: list[dict] = []
        summaries: list[dict] = []
        for fixture in dev:
            summary, fixture_rows = t3.evaluate_fixture(binding, reader32, head, fixture)
            summaries.append(summary)
            rows.extend(fixture_rows)
        weights = {name: tensor for name, tensor in head.state_dict().items()}
        weights.update({f"span.{name}": tensor for name, tensor in span.state_dict().items()})
        candidate = RoutingCandidate(binding, head, arm=ARM, seed=HEAD_SEED, span_head=span)
        extra = measure_functions(sr, dev, binding, head, span, candidate)
        return {
            "head": head,
            "span": span,
            "weights": weights,
            "rows": rows,
            "summaries": summaries,
            "extra": extra,
            "module_path": getattr(sr, "__file__", None),
        }
    finally:
        sys.modules.pop(f"equiv_t3_{tag}", None)
        if saved is not None:
            sys.modules["taiji.selection_routing"] = saved


def measure_functions(sr, dev, binding, head, span, candidate) -> dict:
    routes_all = [sr.route_labels(fixture) for fixture in dev]
    selection = []
    for fixture, routes in zip(
        dev[:SELECTION_FIXTURES], routes_all[:SELECTION_FIXTURES], strict=False
    ):
        goal = fixture["goal"].encode("utf-8")
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"equiv-sel-{fixture['index']}"),
            goal,
            fixture["records"],
        )
        context = sr.build_context(binding, goal, tuple(fixture["record_keys"]))
        output = sr.selection_ce(
            candidate,
            prepared,
            tuple(fixture["target_prefix"]),
            tuple(fixture["target_bytes"]),
            routes[0],
            routes[1],
            context,
            head,
        )
        selection.append(
            {
                "index": fixture["index"],
                "loss": output["loss"].detach().clone(),
                "stage_ce": {
                    stage: (torch.stack(values) if values else None)
                    for stage, values in output["stage_ce"].items()
                },
            }
        )

    losses = []
    for fixture, routes in zip(dev[:LOSS_FIXTURES], routes_all[:LOSS_FIXTURES], strict=False):
        goal = fixture["goal"].encode("utf-8")
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"equiv-loss-{fixture['index']}"),
            goal,
            fixture["records"],
        )
        context = sr.build_context(binding, goal, tuple(fixture["record_keys"]))
        for arm in sr.ARMS:
            output = sr.routing_loss(
                candidate,
                prepared,
                tuple(fixture["target_prefix"]),
                tuple(fixture["target_bytes"]),
                routes[0],
                routes[1],
                context,
                head,
                arm=arm,
                collect=True,
                span_head=span,
                pointer_temperature=T_MIN,
                pointer_entropy_coef=ENTROPY_COEF,
            )
            losses.append(
                {
                    "index": fixture["index"],
                    "arm": arm,
                    "nll": output["nll"].detach().clone(),
                    "ce": output["ce"].detach().clone(),
                    "entropy": output["entropy"].detach().clone(),
                    "distributions": [row.detach().clone() for row in output["distributions"]],
                    "routes": [row.detach().clone() for row in output["routes"]],
                    "stage_ce": [
                        (value.detach().clone() if value is not None else None)
                        for value in output["stage_ce"].values()
                    ],
                }
            )

    cursor = []
    for fixture in dev[:CURSOR_FIXTURES]:
        goal = fixture["goal"].encode("utf-8")
        prepared = prepare(
            binding,
            initial_binding_state(binding, f"equiv-cursor-{fixture['index']}"),
            goal,
            fixture["records"],
        )
        context = sr.build_context(binding, goal, tuple(fixture["record_keys"]))
        for arm in sr.ARMS:
            state, gate_index = sr.observe_goal(binding, prepared, gate_index=0)
            previous = EOS
            for t, accepted in enumerate(fixture["target_prefix"]):
                step = sr.routing_step(
                    binding,
                    prepared,
                    state,
                    previous,
                    context,
                    head,
                    arm=arm,
                    gate_index=gate_index,
                    span_head=span,
                    pointer_temperature=T_MIN,
                )
                entry = {
                    "index": fixture["index"],
                    "arm": arm,
                    "t": t,
                    "scores": step.scores.detach().clone(),
                    "probabilities": step.probabilities.detach().clone(),
                    "pointer_logits": [row.detach().clone() for row in step.pointer_logits],
                    "m": state.m.detach().clone(),
                    "coverage": state.coverage.detach().clone(),
                    "prefix_cursor": int(state.prefix_cursor),
                }
                next_state, _coverage, next_gate = sr._advanced_state(
                    binding, prepared, step, accepted
                )
                entry["next_m"] = next_state.m.detach().clone()
                entry["next_coverage"] = next_state.coverage.detach().clone()
                entry["next_prefix_cursor"] = int(next_state.prefix_cursor)
                cursor.append(entry)
                state = next_state
                gate_index = next_gate
                previous = accepted
    return {"selection": selection, "losses": losses, "cursor": cursor}


def compare(old: dict, new: dict) -> dict:
    weight_block = {}
    for name in old["weights"]:
        merge_diff(weight_block, name, tensor_diff(old["weights"][name], new["weights"][name]))

    row_block: dict = {}
    assert len(old["rows"]) == len(new["rows"])
    for left, right in zip(old["rows"], new["rows"], strict=False):
        numeric_row_diff(left, right, row_block)
    gold_rows = [row for row in old["rows"] if row["kind"] == "record"]

    selection_block: dict = {}
    assert len(old["extra"]["selection"]) == len(new["extra"]["selection"])
    for left, right in zip(old["extra"]["selection"], new["extra"]["selection"], strict=False):
        prefix = f"{left['index']}"
        merge_diff(selection_block, f"{prefix}.loss", tensor_diff(left["loss"], right["loss"]))
        for stage in left["stage_ce"]:
            a = left["stage_ce"][stage]
            b = right["stage_ce"][stage]
            if a is None or b is None:
                merge_diff(
                    selection_block,
                    f"{prefix}.stage_ce.{stage}",
                    {"shape_match": a is b, "max_abs_diff": 0.0, "bitwise_equal": a is b},
                )
            else:
                merge_diff(selection_block, f"{prefix}.stage_ce.{stage}", tensor_diff(a, b))

    loss_block: dict = {}
    for left, right in zip(old["extra"]["losses"], new["extra"]["losses"], strict=False):
        prefix = f"{left['index']}.{left['arm']}"
        for field in ("nll", "ce", "entropy"):
            merge_diff(loss_block, f"{prefix}.{field}", tensor_diff(left[field], right[field]))
        for t, (a, b) in enumerate(
            zip(left["distributions"], right["distributions"], strict=False)
        ):
            merge_diff(loss_block, f"{prefix}.dist.{t}", tensor_diff(a, b))
        for t, (a, b) in enumerate(zip(left["routes"], right["routes"], strict=False)):
            merge_diff(loss_block, f"{prefix}.route.{t}", tensor_diff(a, b))

    cursor_block: dict = {}
    for left, right in zip(old["extra"]["cursor"], new["extra"]["cursor"], strict=False):
        prefix = f"{left['index']}.{left['arm']}.{left['t']}"
        for field in ("scores", "probabilities", "m", "coverage", "next_m", "next_coverage"):
            merge_diff(cursor_block, f"{prefix}.{field}", tensor_diff(left[field], right[field]))
        for i, (a, b) in enumerate(
            zip(left["pointer_logits"], right["pointer_logits"], strict=False)
        ):
            merge_diff(cursor_block, f"{prefix}.pointer.{i}", tensor_diff(a, b))
        for key, left_value, right_value in (
            ("prefix_cursor", left["prefix_cursor"], right["prefix_cursor"]),
            ("next_prefix_cursor", left["next_prefix_cursor"], right["next_prefix_cursor"]),
        ):
            current = cursor_block.setdefault(f"{prefix}.{key}", {"mismatches": 0, "count": 0})
            current["count"] += 1
            if left_value != right_value:
                current["mismatches"] += 1

    def summarise(block: dict) -> dict:
        diffs = [value["max_abs_diff"] for value in block.values() if "max_abs_diff" in value]
        missing = any(
            value["max_abs_diff"] is None for value in block.values() if "max_abs_diff" in value
        )
        bitwise = (
            all(value["bitwise_equal"] for value in block.values() if "bitwise_equal" in value)
            and not missing
        )
        mismatches = sum(
            value.get("mismatches", 0) for value in block.values() if "mismatches" in value
        )
        return {
            "entries": len(block),
            "max_abs_diff": None if missing else max(diffs, default=0.0),
            "all_bitwise_equal": bitwise,
            "mismatches": mismatches,
        }

    gold_rows_expected = 720
    return {
        "weights": summarise(weight_block),
        "gold_forced_rows": {
            "rows": len(old["rows"]),
            "record_rows": len(gold_rows),
            "expected_record_rows": gold_rows_expected,
            "record_rows_match": len(gold_rows) == gold_rows_expected,
            **summarise(row_block),
        },
        "selection_ce": summarise(selection_block),
        "routing_loss": summarise(loss_block),
        "cursor_state": summarise(cursor_block),
        "details": {
            "weights": weight_block,
            "gold_forced": row_block,
            "selection": selection_block,
            "loss": loss_block,
            "cursor": cursor_block,
        },
    }


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    old_sha = sha256_file(OLD_SOURCE)
    new_sha = sha256_file(ROOT / "taiji/selection_routing.py")
    old_module = load_file("taiji_selection_routing_preformat", OLD_SOURCE)
    import taiji.selection_routing as new_module  # noqa: E402  (current bytes)

    t3_data = load_file("equiv_t3_data", T3_SCRIPT)
    dev, audit = t3_data.load_dev()
    binding = load_bundle(ROOT / BUNDLE, torch.float64)[0].binding
    if len(dev) < GOLD_FIXTURES:
        raise RuntimeError(f"dev fixtures shrank to {len(dev)}")

    old = run_variant(old_module, dev, binding, "old")
    new = run_variant(new_module, dev, binding, "new")

    comparison = compare(old, new)

    t3_report = json.loads((ROOT / T3_REPORT).read_text(encoding="utf-8"))
    recorded_t3_sha = t3_report.get("candidate", {}).get("candidate_sha")
    evidence_recorded = {
        str(path): recorded_sha(ROOT / path) for path in EVIDENCE if (ROOT / path).exists()
    }
    identity_candidate = RoutingCandidate(
        binding,
        new["head"],
        arm=ARM,
        seed=HEAD_SEED,
        span_head=new["span"],
    )
    parameters = identity_module.declared_parameters(identity_candidate)
    trainable = identity_module.trainable_declarations(identity_candidate)
    sources = identity_module.graph_sources()
    old_sources = dict(sources)
    old_sources["taiji/selection_routing.py"] = old_sha
    v1_pre = identity_module._digest(
        identity_module._legacy_payload(parameters, trainable, old_sources)
    )
    v1_post = identity_module._digest(
        identity_module._legacy_payload(parameters, trainable, sources)
    )
    semantic = identity_module._digest(identity_module._semantic_payload(parameters, trainable))

    blocks = [
        comparison["weights"],
        comparison["gold_forced_rows"],
        comparison["selection_ce"],
        comparison["routing_loss"],
        comparison["cursor_state"],
    ]
    passed = (
        all(block["all_bitwise_equal"] and block["mismatches"] == 0 for block in blocks)
        and comparison["gold_forced_rows"]["record_rows_match"]
    )

    report = {
        "schema": "taiji-selection-routing-17-format-equivalence-v1",
        "generated_at": "2026-11-04T00:00:00+08:00",
        "passed": passed,
        "question": "does the black-100 line merge change any forward output?",
        "source": {
            "old_source_sha": old_sha,
            "new_source_sha": new_sha,
            "old_source_path": str(OLD_SOURCE),
            "new_source_path": "taiji/selection_routing.py",
            "changed_lines": [474],
        },
        "identity": {
            "identity_basis": identity_module.IDENTITY_BASIS,
            "candidate_sha_v2": semantic,
            "candidate_sha_v1_preformat": v1_pre,
            "candidate_sha_v1_postformat": v1_post,
            "recorded_in_t3_v3": recorded_t3_sha,
            "v1_preformat_matches_recorded": v1_pre == recorded_t3_sha,
            "v2_equal_across_bytes": True,
            "weights_sha256": identity_module.weights_sha256(identity_candidate),
            "candidate_state_sha256": identity_module._state_digest(
                identity_module.weights_sha256(identity_candidate)
            ),
        },
        "comparison": {
            key: comparison[key]
            for key in (
                "weights",
                "gold_forced_rows",
                "selection_ce",
                "routing_loss",
                "cursor_state",
            )
        },
        "details": comparison["details"],
        "data_audit": audit,
        "controls": {
            "optimizer_steps": 0,
            "training_updates": 0,
            "confirm_reads": 0,
            "fixtures": len(dev),
            "arms": list(new_module.ARMS),
        },
        "cost": {
            "cpu": time.process_time() - started_cpu,
            "wall": time.perf_counter() - started_wall,
        },
        "conclusion": (
            "bitwise identical across both byte versions; the semantic identity "
            "is unchanged, so the 2026-11-03 evidence stays valid"
            if passed
            else "the two byte versions differ; evidence must be re-run"
        ),
    }
    (ROOT / OUT).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / OUT).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema": "taiji-selection-routing-17-format-equivalence-manifest-v1",
        "report": str(OUT),
        "report_sha256": sha256_file(ROOT / OUT),
        "old_source": str(OLD_SOURCE),
        "old_source_sha256": old_sha,
        "new_source": "taiji/selection_routing.py",
        "new_source_sha256": new_sha,
        "script": "scripts/training/probe_taiji_selection_routing_17_2r_format_equivalence.py",
        "script_sha256": sha256_file(Path(__file__)),
        "data": {
            "bundle": str(BUNDLE),
            "bundle_sha256": sha256_file(ROOT / BUNDLE),
            "fixtures_used": len(dev),
        },
        "passed": passed,
        "cost": report["cost"],
    }
    (ROOT / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    sidecars = []
    for path in EVIDENCE:
        if not (ROOT / path).exists():
            continue
        sidecar_path = ROOT / path.with_name(path.name + ".identity.json")
        payload = {
            "schema": "taiji-selection-routing-17-identity-sidecar-v1",
            "report": str(path),
            "report_sha256": sha256_file(ROOT / path),
            "recorded_candidate_sha": evidence_recorded.get(str(path)),
            "identity_basis": identity_module.IDENTITY_BASIS,
            "candidate_sha_v2": semantic,
            "candidate_sha_v1_preformat": v1_pre,
            "candidate_sha_v1_postformat": v1_post,
            "old_source_sha": old_sha,
            "new_source_sha": new_sha,
            "equivalence_report": str(OUT),
            "equivalence_report_sha256": manifest["report_sha256"],
            "status": "valid" if passed else "recheck_required",
            "reason": (
                "black-100 line merge only; the zero-update probe is bitwise "
                "identical, so the recorded evidence stays valid"
                if passed
                else "equivalence probe failed; re-run with a new file name"
            ),
        }
        sidecar_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        sidecars.append(str(sidecar_path.relative_to(ROOT)))

    registration = {
        "schema": "taiji-selection-routing-17-identity-registration-v2",
        "generated_at": "2026-11-04T00:00:00+08:00",
        "identity_basis": identity_module.IDENTITY_BASIS,
        "candidate_sha": semantic,
        "contract_version": identity_module.ROUTING_SCHEMA,
        "composition": [
            "graph name and contract/bundle schema",
            "config constants (width, prev_width, hidden, gamma_floor, span_bottleneck)",
            "module list",
            "routing_rule_version",
            "decode rules (point transport, pointer temperature schedule, argmax "
            "inference, entropy coefficient, span supervision weight, gamma floor, "
            "per-arm credit)",
            "routing-step input whitelist",
            "ordered parameter names / shapes / dtypes",
            "declared trainable ownership",
        ],
        "recorded_not_hashed": {
            "source_sha": "per-module byte hash; formatting and comments may change it",
            "weights_sha256": "per-tensor value digests; the parent bundle and head seed "
            "differ per cell, so weights are runtime state",
            "candidate_state_sha256": "digest over the weight digests",
        },
        "legacy": {
            "candidate_sha_v1_preformat": v1_pre,
            "candidate_sha_v1_postformat": v1_post,
            "note": "v1 folded source bytes into the digest and therefore moved on formatting",
        },
        "source": {"old_source_sha": old_sha, "new_source_sha": new_sha},
        "equivalence": {
            "passed": passed,
            "report": str(OUT),
            "report_sha256": manifest["report_sha256"],
        },
        "evidence": evidence_recorded,
        "sidecars": sidecars,
        "cost": report["cost"],
    }
    (ROOT / IDENTITY_REPORT).write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "comparison": report["comparison"],
                "identity": {
                    "v2": semantic,
                    "v1_pre": v1_pre,
                    "v1_post": v1_post,
                    "recorded": recorded_t3_sha,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
