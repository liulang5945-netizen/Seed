"""PLAN66: reproduce one frozen trajectory while observing detached update statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import run_taiji_n1_relation_paired as runner  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

paired = runner.paired
SOURCE_SHA = "d982909f6722d37040fb21f7d869aed6817f3a42a3c4b08300b6746ef14e2922"
OFFLINE = ROOT / "reports/taiji_n1_relation_seed_drift_20261015.json"
OFFLINE_SHA = "6e36f68a9327140586f5a61d23be9f9fa502f9240316dd664e3c9158a7439ef8"
REPORT = ROOT / "reports/taiji_n1_relation_update_replay_20261015.json"
PROTOCOL = dict(
    schema="n1-relation-update-replay-v1",
    source_sha256=SOURCE_SHA,
    offline_sha256=OFFLINE_SHA,
    seed=13,
    arm="position",
    updates=6144,
    milestones=[0, 3072, 6144],
    qa_sequences=9216,
    cpu_limit_seconds=300,
    loss_absolute_tolerance=1e-7,
    loss_relative_tolerance=1e-6,
    teacher_free_groups=1728,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)


def identity():
    return dict(protocol=PROTOCOL, source_code=runner.identity(), runner=paired.sha(Path(__file__)))


def loss_matches(actual, expected):
    if actual["qa_sequences"] != expected["qa_sequences"]:
        return False
    for field in ("total", "qa", "observation"):
        a, b = actual[field], expected[field]
        difference = abs(a - b)
        if (
            not math.isfinite(a)
            or not math.isfinite(b)
            or difference > 1e-7
            or difference / max(abs(a), abs(b), 1e-12) > 1e-6
        ):
            return False
    return True


def instrumented_step(model, optimizer, item, session, second):
    """Use the frozen training function and wrap only its existing Adam step."""
    parameters = dict(model.named_parameters())
    measurements = []
    original_step = optimizer.step

    @torch.no_grad()
    def observed_step(*args, **kwargs):
        before = {name: p.detach().clone() for name, p in parameters.items()}
        groups = {
            name: dict(
                gradient_squared=0.0, gradient_max=0.0, parameter_squared=0.0, update_squared=0.0
            )
            for name in ("reader", "local", "context", "relation", "writer")
        }
        gradients = {}
        for name, parameter in parameters.items():
            gradient = parameter.grad
            norm = 0.0 if gradient is None else float(torch.linalg.vector_norm(gradient.detach()))
            maximum = 0.0 if gradient is None else float(gradient.detach().abs().max())
            gradients[name] = maximum
            labels = [runner.core.parameter_group(name)]
            if name.startswith("writer."):
                labels.append("writer")
            for label in labels:
                group = groups[label]
                group["gradient_squared"] += norm**2
                group["gradient_max"] = max(group["gradient_max"], maximum)
                group["parameter_squared"] += float(torch.linalg.vector_norm(before[name])) ** 2
        returned = original_step(*args, **kwargs)
        updates = {}
        for name, parameter in parameters.items():
            norm = float(torch.linalg.vector_norm(parameter.detach() - before[name]))
            updates[name] = norm
            labels = [runner.core.parameter_group(name)]
            if name.startswith("writer."):
                labels.append("writer")
            for label in labels:
                groups[label]["update_squared"] += norm**2
        measurements.append(
            dict(
                groups={
                    label: dict(
                        gradient_l2=math.sqrt(value["gradient_squared"]),
                        gradient_max=value["gradient_max"],
                        parameter_l2=math.sqrt(value["parameter_squared"]),
                        update_l2=math.sqrt(value["update_squared"]),
                        relative_update=math.sqrt(value["update_squared"])
                        / max(math.sqrt(value["parameter_squared"]), 1e-30),
                    )
                    for label, value in groups.items()
                },
                largest_gradient_parameter=max(gradients, key=gradients.get),
                largest_gradient_max=max(gradients.values()),
                largest_update_parameter=max(updates, key=updates.get),
                largest_update_l2=max(updates.values()),
            )
        )
        return returned

    optimizer.step = observed_step
    try:
        loss = runner.train_step(model, optimizer, item, session, "position", second)
    finally:
        optimizer.step = original_step
    if len(measurements) != 1:
        raise ValueError("exactly one unchanged Adam step must be observed")
    return loss, measurements[0]


def readiness():
    if paired.sha(runner.REPORT) != SOURCE_SHA or paired.sha(OFFLINE) != OFFLINE_SHA:
        raise ValueError("frozen63/65 reports required")
    record = json.loads(runner.REPORT.read_text(encoding="utf-8"))
    offline = json.loads(OFFLINE.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or record["code_identity"] != runner.identity()
        or record["confirmation_queries"] != 0
        or not offline["complete"]
        or offline["model_queries"] != 0
    ):
        raise ValueError("complete unchanged63/65 and unqueried v4 required")
    runner.readiness()
    guards = {**record["source_guards"], str(runner.REPORT): SOURCE_SHA, str(OFFLINE): OFFLINE_SHA}
    for check in record["restore_checks"]:
        if not check["equal"]:
            raise ValueError("all prior independent restores required")
        guards[check["path"]] = check["sha256"]
    if not all(paired.sha(Path(p)) == digest for p, digest in guards.items()):
        raise ValueError("original source/checkpoint bytes changed")
    selected = {
        phase: next(
            Path(c["path"])
            for c in record["restore_checks"]
            if Path(c["path"]).name == f"s13_position_{phase}.pt"
        )
        for phase in ("zero", "final")
    }
    return record, guards, selected


def audit():
    folder = Path(get_external_path("data/checkpoints")) / "n1_relation_update_replay_20261015"
    if REPORT.exists() or folder.exists():
        raise FileExistsError("exclusive exact replay; no restart or overwrite")
    budget = paired.CpuBudget(300)
    record, guards, paths = readiness()
    zero = torch.load(paths["zero"], map_location="cpu", weights_only=True)
    target = torch.load(paths["final"], map_location="cpu", weights_only=True)
    for phase, payload, count in (("zero", zero, 0), ("final", target, 6144)):
        if (
            payload["protocol"] != runner.PROTOCOL
            or payload["seed"] != 13
            or payload["arm"] != "position"
            or payload["completed_updates"] != count
        ):
            raise ValueError(f"wrong frozen {phase} source")
    train = runner.core.checked_train()
    curriculum = runner.core.checked_curriculum(train)
    probes = [
        runner.core.renaming.rename_item(e, p["names"])
        for e, p in zip(train, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=train,
        pair=runner.core.checked_pairs(train),
        rename=probes,
        rename_pair=runner.core.build_pairs(probes),
    )
    expected_losses = [r for r in record["losses"] if r["seed"] == 13 and r["arm"] == "position"]
    expected_rows = [r for r in record["rows"] if r["seed"] == 13 and r["arm"] == "position"]
    if len(expected_losses) != 6144 or len(expected_rows) != 1728:
        raise ValueError("complete source trajectory required")
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        source_checkpoints={k: str(v) for k, v in paths.items()},
        completed_updates=0,
        observations=[],
        losses=[],
        rows=[],
        restore_checks=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    model = optimizer = None
    try:
        model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(zero["candidate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(zero["candidate"]["optimizer"])
        if paired.fingerprint(model.bundle(optimizer)) != paired.fingerprint(zero["candidate"]):
            raise ValueError("zero full bundle differs after restore")
        saved = runner.save(model, optimizer, folder / "zero.pt", 13, "position", 0)
        result["restore_checks"].append(
            runner.independent(folder / "zero.pt", saved["expected_signature"], budget)
        )
        for step in range(6145):
            if step in (0, 3072, 6144):
                rows, _ = runner.evaluate(model, 13, "position", step, views, budget)
                if rows != [r for r in expected_rows if r["completed_updates"] == step]:
                    raise ValueError(f"milestone{step} failed exact row reproduction")
                result["rows"].extend(rows)
            if step == 6144:
                break
            budget.guard(reserve=10)
            item, second = runner.core.renaming.train_item(step, train, curriculum)
            loss, observed = instrumented_step(
                model, optimizer, item, f"relation63-train-13-position-{step}", second
            )
            result["completed_updates"] += 1
            result["losses"].append(loss)
            result["observations"].append(
                dict(
                    observed,
                    step=step,
                    id=item["id"],
                    kind=item["kind"],
                    view="renamed" if step % 2 else "original",
                )
            )
            if not loss_matches(loss, expected_losses[step]):
                raise ValueError(f"step{step} loss failed exact replay tolerance")
            if (step + 1) % 1024 == 0:
                print(
                    f"relation66 updates={step+1} qa={loss['qa']:.7f} cpu={budget.used():.2f}",
                    flush=True,
                )
        saved = runner.save(model, optimizer, folder / "final.pt", 13, "position", 6144)
        result["full_final_bundle_equal"] = paired.fingerprint(
            saved["candidate"]
        ) == paired.fingerprint(target["candidate"])
        result["final_signature_equal"] = (
            saved["expected_signature"] == target["expected_signature"]
        )
        if not result["full_final_bundle_equal"] or not result["final_signature_equal"]:
            raise ValueError("replay final full bundle/signature differs from63")
        result["restore_checks"].append(
            runner.independent(folder / "final.pt", saved["expected_signature"], budget)
        )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no restart")
    finally:
        if result["status"] != "complete" and model is not None and optimizer is not None:
            try:
                runner.save(
                    model,
                    optimizer,
                    folder / "partial.pt",
                    13,
                    "position",
                    result["completed_updates"],
                )
                result["partial_checkpoint"] = str(folder / "partial.pt")
            except Exception as error:
                result["partial_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            sources_unchanged=all(paired.sha(Path(p)) == digest for p, digest in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and result["completed_updates"] == 6144
            and len(result["observations"]) == 6144
            and len(result["rows"]) == 1728
            and sum(r["qa_sequences"] for r in result["losses"]) == 9216
            and len(result["restore_checks"]) == 2
            and result["cpu_seconds"] <= 300
            and all(
                result[k]
                for k in (
                    "sources_unchanged",
                    "code_unchanged",
                    "cpu_accounting_complete",
                    "full_final_bundle_equal",
                    "final_signature_equal",
                )
            )
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    torch.set_num_threads(1)
    if parser.parse_args().execute_approved:
        result = audit()
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("complete", "completed_updates", "cpu_seconds", "error")
                }
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
