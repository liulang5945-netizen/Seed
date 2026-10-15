"""PLAN45: second-hop address/value interventions; never deployed as inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import audit_taiji_n1_read_equivariance as prior  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402
from taiji.native_question_aligned_evidence_candidate import QuestionReadState  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeReadState  # noqa: E402

paired, source = prior.paired, prior.source
REPORT = ROOT / "reports/taiji_n1_cross_read_intervention_20261015.json"
CONDITIONS = ("native", "address", "values", "both")


def intervened_step(model, tape, session, state, previous, address=None, values=None):
    """Mirror native two hops; intervention affects the final read only."""
    if tape.session != session or (state.input_tape is not None and state.input_tape is not tape):
        raise ValueError("same-session tape required")
    conditioned, alignment = state.conditioned_tape, state.alignment
    if conditioned is None:
        conditioned, alignment = model.condition(tape, state.question_states)
    relative = state.relative
    if relative.tape is not None and relative.tape is not conditioned:
        raise ValueError("same conditioned tape required")
    _, hidden, _ = NativeEvidenceCandidate.step(
        model, conditioned, session, relative.hidden, previous
    )
    weights = torch.softmax(conditioned.keys @ model.query(hidden) / model.config.key**0.5, dim=0)
    if relative.address is not None:
        controls = model.location(hidden)
        location = model.transport(
            conditioned, relative.address, torch.softmax(controls[:3], dim=0)
        )
        gate = torch.sigmoid(controls[3])
        weights = (1 - gate) * weights + gate * location
    if address is not None:
        if (
            address.shape != weights.shape
            or not torch.isfinite(address).all()
            or (address < 0).any()
            or not torch.isclose(address.sum(), address.new_tensor(1.0))
        ):
            raise ValueError("replayed address must be a matching probability vector")
        weights = address
    table = conditioned.values if values is None else values
    if table.shape != conditioned.values.shape:
        raise ValueError("replayed value table must preserve position and feature dimensions")
    read = weights @ table
    hidden = model.decoder(torch.cat((model.symbol(previous), read)), hidden)
    logits = model.readout(torch.cat((hidden, read)))
    return (
        logits,
        QuestionReadState(
            RelativeReadState(hidden, weights, conditioned),
            state.question_states,
            tape,
            conditioned,
            alignment,
        ),
        weights,
    )


@torch.no_grad()
def measure(model, item, session, reference=None, condition="native", mirror_native=False):
    if item["split"] != "train":
        raise ValueError("train-derived inputs only")
    if session in model._observations or condition not in CONDITIONS:
        raise ValueError("unique session and frozen condition required")
    original = model.step
    addresses, tables = [], []

    def step(tape, scope, state, previous):
        index = len(addresses)
        if condition == "native" and not mirror_native:
            output = original(tape, scope, state, previous)
        else:
            if condition != "native" and (
                reference is None or tape.positions != reference["positions"]
            ):
                raise ValueError("position-matched learned native reference required")
            output = intervened_step(
                model,
                tape,
                scope,
                state,
                previous,
                reference["addresses"][index] if condition in ("address", "both") else None,
                reference["values"] if condition in ("values", "both") else None,
            )
        _, new_state, weights = output
        addresses.append(weights.detach().clone())
        if not tables:
            tables.append(new_state.conditioned_tape.values.detach().clone())
        return output

    try:
        paired.observe(model, paired.project(item), session)
        tape = model.evidence(session)
        model.step = step
        target = item["answer"].encode()
        labels = torch.tensor([*target, 256])
        logits, _ = model.teacher_logits(tape, session, item["question"].encode(), target)
        symbols = logits.argmax(-1)
        return dict(
            id=item["id"],
            teacher_symbols=symbols.tolist(),
            target_symbols=labels.tolist(),
            teacher_tokens_correct=int((symbols == labels).sum()),
            teacher_exact=bool(torch.equal(symbols, labels)),
            visible_correct=int((symbols[:-1] == labels[:-1]).sum()),
            visible_tokens=len(target),
            boundary_correct=bool(symbols[-1] == 256),
            teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels)),
            addresses=addresses,
            values=tables[0],
            positions=tape.positions,
            logits=logits,
        )
    finally:
        model.__dict__.pop("step", None)
        model._observations.pop(session, None)


def validate_keys(keys):
    expected = {
        (model, view, item["id"], condition)
        for model in ("41", "43")
        for view in ("original", "pair")
        for item in prior.checked_views()[view]
        for condition in ("reference", *CONDITIONS)
    }
    if len(keys) != 2880 or len(set(keys)) != 2880 or set(keys) != expected:
        raise ValueError("all 2880 unique teacher sequences required")


def summarize(rows):
    groups = {}
    for row in rows:
        for combination in ("all", row["free_combination"]):
            key = (row["model"], row["view"], row["kind"], row["condition"], combination)
            groups.setdefault(key, []).append(row)
    return [
        dict(
            model=k[0],
            view=k[1],
            kind=k[2],
            condition=k[3],
            combination=k[4],
            sequences=len(v),
            teacher_exact=sum(r["teacher_exact"] for r in v),
            visible_correct=sum(r["visible_correct"] for r in v),
            visible_tokens=sum(r["visible_tokens"] for r in v),
            boundary_correct=sum(r["boundary_correct"] for r in v),
            mean_ce=sum(r["teacher_ce"] for r in v) / len(v),
        )
        for k, v in groups.items()
    ]


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic; no automatic repeat")
    previous = json.loads(prior.REPORT.read_text(encoding="utf-8"))
    if not previous["complete"] or previous["code_identity"]["audit_sha256"] != paired.sha(
        Path(prior.__file__)
    ):
        raise ValueError("complete frozen 44 required")
    guards = {**previous["source_guards"], str(prior.REPORT): paired.sha(prior.REPORT)}
    if any(paired.sha(Path(p)) != h for p, h in guards.items()):
        raise ValueError("frozen sources changed")
    code = dict(
        source=source.code_identity(),
        prior=paired.sha(Path(prior.__file__)),
        audit=paired.sha(Path(__file__)),
    )
    result = dict(
        schema="n1-cross-read-v1",
        cpu_limit_seconds=60,
        teacher_queries=0,
        free_queries=0,
        parameter_updates=0,
        dev_queries=0,
        holdout_queries=0,
        rows=[],
        controls=[],
        model_guards=[],
        source_guards=guards,
        code_identity=code,
        complete=False,
        status="running",
        can_promote=False,
    )
    keys = []
    try:
        views = prior.checked_views()
        _, _, old_path = source.transfer_source()
        record = json.loads(source.REPORT.read_text(encoding="utf-8"))
        (new_entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
        controls = {(r["model"], r["view"], r["id"]): r for r in previous["controls"]}
        combinations = {
            (r["model"], r["view"], r["id"]): r["free_combination"] for r in previous["rows"]
        }
        for name, path in (("41", old_path), ("43", Path(new_entry["path"]))):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = source.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(payload["candidate"]["optimizer"])
            before = paired.fingerprint(model.bundle(optimizer))
            if not optimizer.state or before != paired.fingerprint(payload["candidate"]):
                raise ValueError("full checkpoint restore mismatch")
            for view, renamed_view in (("original", "rename"), ("pair", "rename_pair")):
                for item, renamed in zip(views[view], views[renamed_view], strict=True):
                    reference = None
                    for condition in ("reference", *CONDITIONS):
                        if time.process_time() >= 58:
                            raise paired.BudgetStop("58 second reserve")
                        measured = measure(
                            model,
                            item if condition == "reference" else renamed,
                            f"cross-read45-{name}-{view}-{item['id']}-{condition}",
                            reference,
                            "native" if condition == "reference" else condition,
                        )
                        result["teacher_queries"] += 1
                        keys.append((name, view, item["id"], condition))
                        if condition in ("reference", "native"):
                            old = controls[
                                name, view if condition == "reference" else renamed_view, item["id"]
                            ]
                            if (
                                any(
                                    old[k] != measured[k]
                                    for k in ("teacher_symbols", "teacher_tokens_correct")
                                )
                                or abs(old["teacher_ce"] - measured["teacher_ce"]) > 1e-6
                            ):
                                raise ValueError("native control mismatch")
                            result["controls"].append(
                                dict(
                                    model=name,
                                    view=view,
                                    id=item["id"],
                                    condition=condition,
                                    reproduced=True,
                                )
                            )
                        if condition == "reference":
                            reference = measured
                        result["rows"].append(
                            dict(
                                model=name,
                                view=view,
                                id=item["id"],
                                condition=condition,
                                kind=item.get("original_kind", item["kind"]),
                                free_combination=combinations[name, view, item["id"]],
                                **{
                                    k: v
                                    for k, v in measured.items()
                                    if k not in ("addresses", "values", "positions", "logits", "id")
                                },
                            )
                        )
            unchanged = before == paired.fingerprint(model.bundle(optimizer))
            result["model_guards"].append(dict(model=name, unchanged=unchanged))
            if not unchanged:
                raise ValueError("parameters/Adam/RNG/history changed")
        validate_keys(keys)
        result.update(complete=True, status="complete", summary=summarize(result["rows"]))
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = code == dict(
            source=source.code_identity(),
            prior=paired.sha(Path(prior.__file__)),
            audit=paired.sha(Path(__file__)),
        )
        result["cpu_seconds"] = time.process_time()
        result["complete"] = (
            result["complete"]
            and result["sources_unchanged"]
            and result["code_unchanged"]
            and result["cpu_seconds"] <= 60
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute_approved:
        output = audit()
        print(
            json.dumps(
                {k: output.get(k) for k in ("complete", "teacher_queries", "cpu_seconds", "error")}
            )
        )
        raise SystemExit(0 if output["complete"] else 1)
    print(json.dumps(dict(read_only=True, teacher_sequences=2880, cpu_limit_seconds=60)))
