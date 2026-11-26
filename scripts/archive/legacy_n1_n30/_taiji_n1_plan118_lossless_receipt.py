"""PLAN84 lossless, train-only receipt validation before any confirmation loading."""

import math
from pathlib import Path

from scripts.training import run_taiji_n1_transition_factorial_paired as train

COUNTS = dict(losses=24576, measurements=24576, rows=4800)
RWL_GROUPS = ("reader", "writer", "local")
NORMALIZED_RWL_SCOPE = "shared_normalized_rwl_radius"
META = (
    "seed",
    "arm",
    "completed_updates",
    "milestones",
    "restore_checks",
    "signature_clone_updates",
    "final_steps",
    "new_session_count",
    "score_queries",
    "qa_sequences",
    "training_sufficient",
    "complete",
)


def checked_rows(record, views):
    seed, arm = record["seed"], record["arm"]
    specs = {v: {item["id"]: dict(item=item) for item in items} for v, items in views.items()}
    for view in ("original", "rename"):
        entries = [
            entry
            for item in views[view]
            if item["kind"] == "transition"
            for entry in train.calibration.engineering.factorial.variants(item)
        ]
        specs[f"factorial_{view}"] = {entry["case_id"]: entry for entry in entries}
    expected = {
        (step, view, key)
        for step in train.MILESTONES
        for view, items in specs.items()
        for key in items
    }
    actual = {}
    for row in record["rows"]:
        view = row["evaluation_set"]
        key = row.get("case_id") if view.startswith("factorial_") else row["id"]
        address = row["completed_updates"], view, key
        if address not in expected or address in actual or row["seed"] != seed or row["arm"] != arm:
            raise ValueError("exact five phases, six views, unique cases and model required")
        spec = specs[view][key]
        item = spec["item"]
        labels = [*item["answer"].encode(), 256]
        if (
            row["id"] != item["id"]
            or row["kind"] != item["kind"]
            or row["target_symbols"] != labels
        ):
            raise ValueError("known train target/identity differs")
        if view.startswith("factorial_") and any(
            row[k] != spec[k] for k in ("parent_id", "condition", "initial_direction", "turn")
        ):
            raise ValueError("factorial condition identity differs")
        predicted = row["teacher_symbols"]
        if len(predicted) != len(labels) or any(
            type(x) is not int or not 0 <= x <= 256 for x in predicted
        ):
            raise ValueError("full teacher symbols required")
        if (
            row["teacher_exact"] != (predicted == labels)
            or row["teacher_tokens_correct"]
            != sum(a == b for a, b in zip(predicted, labels, strict=True))
            or row["free_exact"]
            != (
                bytes.fromhex(row["raw_hex"]) == item["answer"].encode()
                and row["stop_reason"] == "boundary"
            )
            or not math.isfinite(row["teacher_ce"])
            or row["teacher_ce"] < 0
        ):
            raise ValueError("raw predictions and stored score differ")
        actual[address] = row
    if len(actual) != 4800 or set(actual) != expected:
        raise ValueError("all4800 exact rows required")
    for step in train.MILESTONES:
        selected = {
            view: [actual[step, view, key] for key in items] for view, items in specs.items()
        }
        for view in ("original", "rename"):
            for row in selected[f"factorial_{view}"]:
                if row["condition"] == "original" and any(
                    row[k] != actual[step, view, row["parent_id"]][k]
                    for k in train.calibration.engineering.factorial.BASE_FIELDS
                ):
                    raise ValueError("native factorial row must reproduce original score exactly")
        summary = train.runner.core.summarize(selected["original"])[str(seed)][arm]
        renamed = train.runner.core.summarize(selected["rename"])[str(seed)][arm]
        summary.update(
            pair_exact=sum(r["free_exact"] for r in selected["pair"]),
            rename_free_exact=renamed["free_exact"],
            rename_free_by_kind=renamed["free_by_kind"],
            rename_pair_exact=sum(r["free_exact"] for r in selected["rename_pair"]),
            factorial={
                v: train.calibration.quartet_summary(selected[f"factorial_{v}"])
                for v in ("original", "rename")
            },
            completed_updates=step,
        )
        if summary != record["milestones"][list(train.MILESTONES).index(step)]:
            raise ValueError("raw scores and complete milestone summary differ")


def checked_trajectory(record, items):
    groups = train.engineering.bounded.POLICY["groups"]
    for step, (loss, measurement) in enumerate(
        zip(record["losses"], record["measurements"], strict=True)
    ):
        item = items[((step % 6144) // 2) % 192]
        expected_qa = (
            4 if item["kind"] == "transition" else 2 if item["kind"] in ("lookup", "unknown") else 1
        )
        if (
            loss["qa_sequences"] != expected_qa
            or any(
                not math.isfinite(loss[k]) or loss[k] < 0 for k in ("total", "qa", "observation")
            )
            or not math.isclose(
                loss["total"], loss["qa"] + 0.25 * loss["observation"], rel_tol=1e-5, abs_tol=1e-6
            )
        ):
            raise ValueError("exact scheduled QA and finite unchanged loss weighting required")
        if (
            measurement["step"] != step
            or measurement["cycle"] != step // 6144
            or set(measurement["groups"]) != set(groups)
        ):
            raise ValueError("ordered fixed updates and all five groups required")
        validate_adam_proposal_measurement(measurement)


def validate_adam_proposal_measurement(measurement):
    groups = measurement["groups"]
    shared_flags = [groups[name].get("cap_scope") == NORMALIZED_RWL_SCOPE for name in RWL_GROUPS]
    shared_fields = (
        "group_normalized_proposal",
        "shared_normalized_proposal_l2",
        "shared_normalized_radius",
        "shared_normalized_scale",
        "shared_normalized_actual_l2",
    )
    has_shared_metadata = any(
        any(field in groups[name] for field in shared_fields) for name in RWL_GROUPS
    )
    shared = all(shared_flags)
    if has_shared_metadata != shared:
        raise ValueError("finite bounded actual Adam proposal evidence required")

    for name, group in groups.items():
        values = (
            "actual_l2",
            "cap",
            "gradient_l2",
            "parameter_l2",
            "proposed_l2",
            "scale",
        )
        if (
            any(not math.isfinite(group[key]) or group[key] < 0 for key in values)
            or group["cap"] <= 0
            or not 0 < group["scale"] <= 1
            or group["clipped"] != (group["scale"] < 1)
        ):
            raise ValueError("finite bounded actual Adam proposal evidence required")
        if (not shared or name not in RWL_GROUPS) and group["actual_l2"] > (
            group["cap"] * 1.0001 + 1e-6
        ):
            raise ValueError("finite bounded actual Adam proposal evidence required")

    if not shared:
        return

    if any(any(field not in groups[name] for field in shared_fields) for name in RWL_GROUPS):
        raise ValueError("finite bounded actual Adam proposal evidence required")
    normalized_proposals = [groups[name]["group_normalized_proposal"] for name in RWL_GROUPS]
    radius = groups[RWL_GROUPS[0]]["shared_normalized_radius"]
    if (
        any(not math.isfinite(value) or value < 0 for value in normalized_proposals)
        or not math.isfinite(radius)
        or not math.isclose(radius, math.sqrt(3.0), rel_tol=1e-7, abs_tol=1e-9)
    ):
        raise ValueError("finite bounded actual Adam proposal evidence required")
    proposal_norm = math.sqrt(sum(value**2 for value in normalized_proposals))
    expected_scale = min(1.0, radius / proposal_norm) if proposal_norm else 1.0
    expected_actual = proposal_norm * expected_scale
    if not math.isfinite(expected_scale) or not math.isclose(
        expected_scale,
        groups[RWL_GROUPS[0]]["shared_normalized_scale"],
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise ValueError("finite bounded actual Adam proposal evidence required")

    for name, normalized_proposal in zip(RWL_GROUPS, normalized_proposals, strict=True):
        group = groups[name]
        if (
            any(not math.isfinite(group[field]) for field in shared_fields[1:])
            or group["cap_scope"] != NORMALIZED_RWL_SCOPE
            or not math.isclose(
                group["proposed_l2"] / group["cap"], normalized_proposal, rel_tol=1e-5, abs_tol=1e-6
            )
            or not math.isclose(
                group["shared_normalized_proposal_l2"], proposal_norm, rel_tol=1e-5, abs_tol=1e-6
            )
            or not math.isclose(
                group["shared_normalized_radius"], radius, rel_tol=1e-7, abs_tol=1e-9
            )
            or not math.isclose(
                group["shared_normalized_scale"], expected_scale, rel_tol=1e-5, abs_tol=1e-6
            )
            or not math.isclose(
                group["shared_normalized_actual_l2"], expected_actual, rel_tol=1e-5, abs_tol=1e-6
            )
            or not math.isclose(group["scale"], expected_scale, rel_tol=1e-5, abs_tol=1e-6)
            or not math.isclose(
                group["actual_l2"],
                group["proposed_l2"] * expected_scale,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            or not math.isclose(
                group["actual_l2"] / group["cap"],
                normalized_proposal * expected_scale,
                rel_tol=1e-5,
                abs_tol=1e-6,
            )
            or group["actual_l2"] > group["cap"] * radius * 1.0001 + 1e-6
        ):
            raise ValueError("finite bounded actual Adam proposal evidence required")

    actual_radius = math.sqrt(
        sum((groups[name]["actual_l2"] / groups[name]["cap"]) ** 2 for name in RWL_GROUPS)
    )
    shared_actuals = [groups[name]["shared_normalized_actual_l2"] for name in RWL_GROUPS]
    if (
        not math.isfinite(actual_radius)
        or actual_radius > radius * 1.0001 + 1e-6
        or any(
            not math.isclose(value, expected_actual, rel_tol=1e-5, abs_tol=1e-6)
            for value in shared_actuals
        )
    ):
        raise ValueError("finite bounded actual Adam proposal evidence required")


def validate_model(model, record, *, views=None):
    seed, arm = model["seed"], model["arm"]
    identity = dict(seed=seed, arm=arm, protocol=train.PROTOCOL, phase="complete")
    train.shard.validate(record, identity, COUNTS)
    if any(model.get(k) != record.get(k) for k in META):
        raise ValueError("main receipt and full shard metadata differ")
    if (
        not record["complete"]
        or record["status"] != "complete"
        or record["completed_updates"] != 24576
        or record["new_session_count"] != 43008
        or record["score_queries"] != 4800
        or record["qa_sequences"] != 55296
        or sum(x["qa_sequences"] for x in record["losses"]) != 55296
        or record["signature_clone_updates"] != 8
        or [x["completed_updates"] for x in record["milestones"]] != list(train.MILESTONES)
    ):
        raise ValueError("full fixed quartet trajectory required")
    steps = record["final_steps"]
    if (
        len(steps) != 41
        or sum(n.startswith("writer.") for n in steps) != 4
        or any(
            v != (None if arm == "cut" and n.startswith("writer.") else 24576)
            for n, v in steps.items()
        )
    ):
        raise ValueError("all41 same-arm final Adam counts required")
    checks = record["restore_checks"]
    if (
        len(checks) != 2
        or {Path(c["path"]).name for c in checks}
        != {f"s{seed}_{arm}_zero.pt", f"s{seed}_{arm}_final.pt"}
        or any(not c["equal"] or c["signature_clone_updates"] != 2 for c in checks)
    ):
        raise ValueError("exact zero/final independent quartet restore pair required")
    sufficient = train.calibration.sufficient(record["milestones"][-1])
    if record["training_sufficient"] != sufficient:
        raise ValueError("stored gate differs from frozen six gates")
    if views is None:
        _, _, views = train.known_views()
    checked_rows(record, views)
    checked_trajectory(record, views["original"])
    return sufficient


def read_model(model):
    seal = model["shard"]
    identity = dict(seed=model["seed"], arm=model["arm"], protocol=train.PROTOCOL, phase="complete")
    if seal["counts"] != COUNTS:
        raise ValueError("exact lossless complete evidence counts required")
    return train.shard.read_checked(Path(seal["path"]), seal, identity=identity)


def validate(receipt, *, read_record=read_model, views=None):
    expected = {(s, a) for s in train.engineering.SEEDS for a in train.engineering.ARMS}
    models = receipt["models"]
    if (
        not receipt["complete"]
        or receipt["status"] != "complete"
        or receipt["protocol"] != train.PROTOCOL
        or receipt["code_identity"] != train.identity()
        or len(models) != 9
        or {(m["seed"], m["arm"]) for m in models} != expected
        or receipt["completed_updates"] != 221184
        or receipt["qa_sequences"] != 497664
        or receipt["new_session_count"] != 387072
        or receipt["score_queries"] != 43200
        or receipt["signature_clone_updates"] != 72
        or receipt["synthetic_qa_sequences"] != 288
        or not 0 <= receipt["cpu_seconds"] <= 14400
        or not all(
            receipt[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
        or any(receipt[k] != 0 for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
    ):
        raise ValueError("complete frozen84 nine-model quartet receipt required")
    if views is None:
        _, _, views = train.known_views()
    positions = []
    for model in models:
        sufficient = validate_model(model, read_record(model), views=views)
        if model["arm"] == "position":
            positions.append(sufficient)
    if len(positions) != 3 or not all(positions) or not receipt["training_sufficient"]:
        raise ValueError("all three position six gates required before confirmation")
    return dict(
        models=9,
        independent_restores=18,
        known_rows=43200,
        updates=221184,
        model_loads=0,
        confirmation_queries=0,
    )
