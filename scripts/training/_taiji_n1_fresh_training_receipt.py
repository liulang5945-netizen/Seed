"""Read-only complete PLAN77 receipt validation before any confirmation loading."""

from pathlib import Path

from scripts.training import run_taiji_n1_fresh_paired_training as train

COUNTS = dict(losses=24576, measurements=24576, rows=2880)


def validate_model(model, record):
    seed, arm = model["seed"], model["arm"]
    identity = dict(seed=seed, arm=arm, protocol=train.PROTOCOL, phase="complete")
    train.shard.validate(record, identity, COUNTS)
    for key in (
        "seed",
        "arm",
        "completed_updates",
        "milestones",
        "restore_checks",
        "signature_clone_updates",
        "final_steps",
        "new_session_count",
        "training_sufficient",
        "complete",
    ):
        if model.get(key) != record.get(key):
            raise ValueError("main receipt and full shard differ")
    if (
        not record["complete"]
        or record["status"] != "complete"
        or record["completed_updates"] != 24576
        or record["new_session_count"] != 24576
        or record["signature_clone_updates"] != 8
        or sum(x["qa_sequences"] for x in record["losses"]) != 36864
        or [x["completed_updates"] for x in record["milestones"]] != list(train.MILESTONES)
    ):
        raise ValueError("full fixed per-model trajectory required")
    steps = record["final_steps"]
    if len(steps) != 41 or sum(n.startswith("writer.") for n in steps) != 4:
        raise ValueError("all41 named final Adam states required")
    if any(
        value != (None if arm == "cut" and name.startswith("writer.") else 24576)
        for name, value in steps.items()
    ):
        raise ValueError("same-arm final Adam counts differ")
    checks = record["restore_checks"]
    if len(checks) != 2 or {Path(c["path"]).name for c in checks} != {
        f"s{seed}_{arm}_zero.pt",
        f"s{seed}_{arm}_final.pt",
    }:
        raise ValueError("exact zero/final independent restore pair required")
    if any(not c["equal"] or c["signature_clone_updates"] != 2 for c in checks):
        raise ValueError("independent restore equality required")
    sufficient = train.runner.sufficient(record["milestones"][-1])
    if record["training_sufficient"] != sufficient:
        raise ValueError("stored training gate differs from original four gates")
    return sufficient


def read_model(model):
    seal = model["shard"]
    identity = dict(seed=model["seed"], arm=model["arm"], protocol=train.PROTOCOL, phase="complete")
    if seal["counts"] != COUNTS:
        raise ValueError("exact complete evidence counts required")
    record = train.shard.read_checked(Path(seal["path"]), seal, identity=identity)
    validate_model(model, record)
    return record


def validate(receipt, *, read_record=read_model):
    expected = {(seed, arm) for seed in train.engineering.SEEDS for arm in train.engineering.ARMS}
    models = receipt["models"]
    if (
        not receipt["complete"]
        or receipt["status"] != "complete"
        or receipt["protocol"] != train.PROTOCOL
        or receipt["code_identity"] != train.identity()
        or len(models) != 9
        or {(m["seed"], m["arm"]) for m in models} != expected
        or receipt["completed_updates"] != 221184
        or receipt["signature_clone_updates"] != 72
        or receipt["synthetic_qa_sequences"] != 144
        or not 0 <= receipt["cpu_seconds"] <= 9000
        or not all(
            receipt[k] for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
        )
        or any(receipt[k] != 0 for k in ("dev_queries", "holdout_queries", "confirmation_queries"))
    ):
        raise ValueError("complete frozen77 nine-model receipt required")
    position_gates = []
    for model in models:
        record = read_record(model)
        sufficient = validate_model(model, record)
        if model["arm"] == "position":
            position_gates.append(sufficient)
    if len(position_gates) != 3 or not all(position_gates) or not receipt["training_sufficient"]:
        raise ValueError("all three position original gates required before confirmation")
    return dict(
        models=9,
        independent_restores=18,
        known_rows=25920,
        updates=221184,
        model_loads=0,
        confirmation_queries=0,
    )
