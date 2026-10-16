"""Synthetic counter/score layouts only; never read checkpoints or confirmation data."""

from copy import deepcopy

import pytest

from scripts.training import _taiji_n1_transition_factorial_training_receipt as receipt


def views():
    original = [
        dict(
            receipt.train.engineering.SYNTHETIC,
            id=f"synthetic-{i}",
            kind=receipt.train.paired.KINDS[i % 4],
        )
        for i in range(192)
    ]
    pairs = [
        dict(item, id="pair-" + item["id"], kind="lookup")
        for item in original
        if item["kind"] in ("lookup", "unknown")
    ]
    return dict(original=original, pair=pairs, rename=original, rename_pair=pairs)


def synthetic_model(seed=11, arm="position", correct=True):
    known = views()
    entries = {v: [dict(item=item) for item in items] for v, items in known.items()}
    for view in ("original", "rename"):
        entries[f"factorial_{view}"] = [
            entry
            for item in known[view]
            if item["kind"] == "transition"
            for entry in receipt.train.calibration.engineering.factorial.variants(item)
        ]
    rows = []
    for step in receipt.train.MILESTONES:
        for view, specs in entries.items():
            for spec in specs:
                item = spec["item"]
                labels = [*item["answer"].encode(), 256]
                raw = item["answer"].encode() if correct else b"wrong"
                row = dict(
                    id=item["id"],
                    kind=item["kind"],
                    seed=seed,
                    arm=arm,
                    completed_updates=step,
                    evaluation_set=view,
                    target_symbols=labels,
                    teacher_symbols=labels if correct else [0] * len(labels),
                    teacher_tokens_correct=len(labels) if correct else 0,
                    teacher_exact=correct,
                    teacher_ce=0.0 if correct else 1.0,
                    raw_hex=raw.hex(),
                    output_hex=raw.hex(),
                    stop_reason="boundary",
                    generated_steps=len(raw) + 1,
                    free_exact=correct,
                )
                row.update({k: v for k, v in spec.items() if k != "item"})
                rows.append(row)
    n = 48 if correct else 0
    summary = dict(
        items=192,
        teacher_exact=192 if correct else 0,
        free_exact=192 if correct else 0,
        teacher_token_accuracy=1.0 if correct else 0.0,
        mean_episode_ce=0.0 if correct else 1.0,
        free_by_kind={k: n for k in receipt.train.paired.KINDS},
        pair_exact=96 if correct else 0,
        rename_free_exact=192 if correct else 0,
        rename_free_by_kind={k: n for k in receipt.train.paired.KINDS},
        rename_pair_exact=96 if correct else 0,
        factorial={
            v: dict(
                items=192,
                parents=48,
                by_condition={
                    c: n for c in receipt.train.calibration.engineering.factorial.CONDITIONS
                },
                joint_four_exact=n,
            )
            for v in ("original", "rename")
        },
    )
    group = dict(
        actual_l2=0.0,
        cap=0.001,
        gradient_l2=0.0,
        parameter_l2=1.0,
        proposed_l2=0.0,
        scale=1.0,
        clipped=False,
    )
    losses = [
        dict(total=1.25, qa=1.0, observation=1.0, qa_sequences=x) for x in (2, 2, 1, 1, 4, 4, 2, 2)
    ] * 3072
    measurements = [
        dict(
            step=i,
            cycle=i // 6144,
            groups={g: group for g in receipt.train.engineering.bounded.POLICY["groups"]},
        )
        for i in range(24576)
    ]
    record = dict(
        seed=seed,
        arm=arm,
        protocol=receipt.train.PROTOCOL,
        phase="complete",
        complete=True,
        status="complete",
        completed_updates=24576,
        new_session_count=43008,
        score_queries=4800,
        qa_sequences=55296,
        signature_clone_updates=8,
        training_sufficient=correct,
        losses=losses,
        measurements=measurements,
        rows=rows,
        milestones=[dict(deepcopy(summary), completed_updates=s) for s in receipt.train.MILESTONES],
        final_steps={
            **{f"writer.{i}": None if arm == "cut" else 24576 for i in range(4)},
            **{f"other.{i}": 24576 for i in range(37)},
        },
        restore_checks=[
            dict(path=f"s{seed}_{arm}_{phase}.pt", equal=True, signature_clone_updates=2)
            for phase in ("zero", "final")
        ],
    )
    return {k: record[k] for k in receipt.META}, record, known


def test_full_model_layout_and_raw_milestones_match():
    model, record, known = synthetic_model(13, "cut")
    assert receipt.validate_model(model, record, views=known)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "condition",
        "native",
        "score",
        "summary",
        "qa",
        "gradient",
        "step",
        "history",
        "adam",
        "restore",
    ],
)
def test_changed_evidence_cannot_pass_receipt(change):
    model, record, known = synthetic_model(11, "cut")
    if change == "duplicate":
        record["rows"][-1] = record["rows"][-2]
    elif change == "condition":
        record["rows"][-1]["turn"] = "invalid"
    elif change == "native":
        row = next(
            r
            for r in record["rows"]
            if r["evaluation_set"] == "factorial_original" and r["condition"] == "original"
        )
        row["teacher_ce"] = 0.01
    elif change == "score":
        record["rows"][0]["free_exact"] = False
    elif change == "summary":
        record["milestones"][0]["free_exact"] -= 1
    elif change == "qa":
        record["losses"][0] = dict(record["losses"][0], total=99.0)
    elif change == "gradient":
        record["measurements"][0]["groups"] = deepcopy(record["measurements"][0]["groups"])
        record["measurements"][0]["groups"]["reader"]["actual_l2"] = 1.0
    elif change == "step":
        record["measurements"][0]["step"] = 1
    elif change == "history":
        record["new_session_count"] -= 1
    elif change == "adam":
        record["final_steps"]["writer.0"] = 24576
    else:
        record["restore_checks"][0]["equal"] = False
    with pytest.raises(ValueError):
        receipt.validate_model(model, record, views=known)


def test_batch_rejects_weak_position_but_preserves_weak_controls(monkeypatch):
    monkeypatch.setattr(receipt.train, "identity", lambda: dict(synthetic=True))
    monkeypatch.setattr(
        receipt.train.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("receipt must load no models"),
    )
    models = []
    records = {}
    for seed in receipt.train.engineering.SEEDS:
        for arm in receipt.train.engineering.ARMS:
            model, record, known = synthetic_model(seed, arm, correct=arm != "terminal")
            models.append(model)
            records[seed, arm] = record
    batch = dict(
        complete=True,
        status="complete",
        protocol=receipt.train.PROTOCOL,
        code_identity=receipt.train.identity(),
        models=models,
        completed_updates=221184,
        qa_sequences=497664,
        new_session_count=387072,
        score_queries=43200,
        signature_clone_updates=72,
        synthetic_qa_sequences=288,
        cpu_seconds=13000,
        sources_unchanged=True,
        code_unchanged=True,
        cpu_accounting_complete=True,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        training_sufficient=True,
    )

    def read(model):
        return records[model["seed"], model["arm"]]

    assert receipt.validate(batch, read_record=read, views=known)["known_rows"] == 43200
    model, record, _ = synthetic_model(12, "position", False)
    batch["models"][3] = model
    records[12, "position"] = record
    with pytest.raises(ValueError, match="six gates"):
        receipt.validate(batch, read_record=read, views=known)
    for key, value in (
        ("confirmation_queries", 1),
        ("cpu_seconds", 14401),
        ("new_session_count", 1),
    ):
        bad = dict(batch, **{key: value})
        with pytest.raises(ValueError, match="complete frozen84"):
            receipt.validate(bad, read_record=read, views=known)


def test_double_seal_checked_before_record_use(tmp_path):
    model, record, known = synthetic_model()
    path = tmp_path / "evidence.json.gz"
    identity = dict(seed=11, arm="position", protocol=receipt.train.PROTOCOL, phase="complete")
    seal = receipt.train.shard.write_new(path, record, identity, receipt.COUNTS)
    model["shard"] = seal
    assert receipt.read_model(model)["rows"] == record["rows"]
    bad = dict(model, shard=dict(seal, uncompressed_sha256="0" * 64))
    with pytest.raises(ValueError, match="uncompressed"):
        receipt.read_model(bad)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="compressed"):
        receipt.read_model(model)
