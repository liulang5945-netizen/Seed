"""PLAN50: frozen local-binding ablation and offline alignment summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import audit_taiji_n1_read_equivariance as views_source  # noqa: E402
from scripts.training import run_taiji_n1_local_question as source  # noqa: E402
from taiji.native_question_aligned_evidence_candidate import QuestionReadState  # noqa: E402

paired = source.paired
REPORT = ROOT / "reports/taiji_n1_local_binding_ablation_20261015.json"
SOURCE_SHA = "7347b34b7a987fc1014aaac91e2168f9532e1dba4e607f5b50d6650f69b93af3"
CONDITIONS = ("native", "bypass")


@torch.no_grad()
def alignment_statistics(model, tape, state, alignment, salience, item):
    """Metadata enters only these post-forward measurements, never the model."""
    question = item["question"]
    entities = [n for n in item["entities"] if n in question]
    if len(entities) != 1:
        raise ValueError("one question entity required for offline measurement")
    name = entities[0]
    start = question.index(name)
    entity_mask = torch.zeros(len(question), dtype=torch.bool)
    entity_mask[start : start + len(name)] = True
    evidence_mask = torch.zeros(len(tape.positions), dtype=torch.bool)
    offset = 0
    for observation in tape.sources:
        if observation.content.startswith(name.encode() + b" "):
            evidence_mask[offset : offset + len(name)] = True
        offset += len(observation.content)
    features = torch.cat([model.local_features(s.content) for s in tape.sources])
    confidence = (model.local_features(state.question_bytes) @ features.T).max(-1).values
    present = bool(evidence_mask.any())
    return dict(
        question_entity_observed=present,
        entity_salience_mass=float(salience[entity_mask].sum()),
        entity_salience_uniform=float(entity_mask.float().mean()),
        entity_max_cosine=float(confidence[entity_mask].mean()),
        template_max_cosine=float(confidence[~entity_mask].mean()),
        same_entity_alignment_mass=(
            float(alignment[entity_mask][:, evidence_mask].sum(-1).mean()) if present else None
        ),
        same_entity_uniform=float(evidence_mask.float().mean()) if present else None,
    )


@torch.no_grad()
def measure(model, item, session, condition):
    if item["split"] != "train":
        raise ValueError("only train-derived inputs allowed")
    if condition not in CONDITIONS or session in model._observations:
        raise ValueError("unique session and frozen condition required")
    original_binding = model.bind_question
    statistics = {}

    def bind(tape, state):
        if condition == "bypass":
            return QuestionReadState(state.relative, state.question_states), None, None
        bound, alignment, salience = original_binding(tape, state)
        statistics.update(alignment_statistics(model, tape, state, alignment, salience, item))
        return bound, alignment, salience

    try:
        paired.observe(model, paired.project(item), session)
        model.bind_question = bind
        tape = model.evidence(session)
        target = item["answer"].encode()
        logits, addresses = model.teacher_logits(tape, session, item["question"].encode(), target)
        labels = torch.tensor([*target, 256])
        predicted = logits.argmax(-1)
        return dict(
            id=item["id"],
            teacher_symbols=predicted.tolist(),
            target_symbols=labels.tolist(),
            teacher_tokens_correct=int((predicted == labels).sum()),
            teacher_exact=bool(torch.equal(predicted, labels)),
            teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels)),
            visible_correct=int((predicted[:-1] == labels[:-1]).sum()),
            visible_tokens=len(target),
            boundary_correct=bool(predicted[-1] == 256),
            addresses=addresses,
            logits=logits,
            alignment=statistics or None,
        )
    finally:
        model.__dict__.pop("bind_question", None)
        model._observations.pop(session, None)


def validate_keys(keys):
    expected = {
        (view, item["id"], condition)
        for view, items in views_source.checked_views().items()
        for item in items
        for condition in CONDITIONS
    }
    if len(keys) != 1152 or len(set(keys)) != 1152 or set(keys) != expected:
        raise ValueError("all 1152 unique teacher sequences required")


def summarize(rows):
    result = []
    groups = {(r["view"], r["kind"], r["condition"]) for r in rows}
    for view, kind, condition in sorted(groups):
        selected = [
            r for r in rows if (r["view"], r["kind"], r["condition"]) == (view, kind, condition)
        ]
        result.append(
            dict(
                view=view,
                kind=kind,
                condition=condition,
                items=len(selected),
                teacher_exact=sum(r["teacher_exact"] for r in selected),
                visible_correct=sum(r["visible_correct"] for r in selected),
                visible_tokens=sum(r["visible_tokens"] for r in selected),
                mean_ce=sum(r["teacher_ce"] for r in selected) / len(selected),
                boundary_correct=sum(r["boundary_correct"] for r in selected),
            )
        )
    return result


def alignment_summaries(rows):
    result = []
    for view in ("original", "pair", "rename", "rename_pair"):
        for present in (False, True):
            selected = [
                r["alignment"]
                for r in rows
                if r["view"] == view
                and r["condition"] == "native"
                and r["alignment"]["question_entity_observed"] == present
            ]
            if not selected:
                continue
            result.append(
                dict(
                    view=view,
                    question_entity_observed=present,
                    items=len(selected),
                    **{
                        key: (
                            sum(s[key] for s in selected) / len(selected)
                            if selected[0][key] is not None
                            else None
                        )
                        for key in (
                            "entity_salience_mass",
                            "entity_salience_uniform",
                            "entity_max_cosine",
                            "template_max_cosine",
                            "same_entity_alignment_mass",
                            "same_entity_uniform",
                        )
                    },
                )
            )
    return result


def code_identity():
    return dict(
        source=source.code_identity(),
        audit=paired.sha(Path(__file__)),
        views=paired.sha(Path(views_source.__file__)),
    )


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic; no repeat")
    if paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen49 report digest mismatch")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete frozen49 required")
    (entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    path = Path(entry["path"])
    if paired.sha(path) != entry["sha256"]:
        raise ValueError("frozen49 final checkpoint changed")
    views = views_source.checked_views()
    guards = {
        str(p): paired.sha(p)
        for p in (
            source.REPORT,
            path,
            source.BASELINE,
            source.calibration.MANIFEST,
            source.PAIR_MANIFEST,
            source.CURRICULUM,
            ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json",
        )
    }
    result = dict(
        schema="n1-local-binding-ablation-v1",
        cpu_limit_seconds=60,
        teacher_queries=0,
        free_queries=0,
        parameter_updates=0,
        dev_queries=0,
        holdout_queries=0,
        rows=[],
        controls=[],
        source_guards=guards,
        code_identity=code_identity(),
        complete=False,
        status="running",
        can_promote=False,
    )
    keys = []
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
            raise ValueError("fixed49 final protocol required")
        model = source.LocalQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.load_state_dict(payload["candidate"]["optimizer"])
        before = paired.fingerprint(model.bundle(optimizer))
        if before != paired.fingerprint(payload["candidate"]) or len(optimizer.state) != len(
            list(model.parameters())
        ):
            raise ValueError("full trained checkpoint restore differs")
        controls = {
            (r["evaluation_set"], r["id"]): r
            for r in record["rows"]
            if r["completed_updates"] == 6144
        }
        for view, items in views.items():
            for item in items:
                native = None
                for condition in CONDITIONS:
                    if time.process_time() >= 58:
                        raise paired.BudgetStop("58 second diagnostic reserve")
                    row = measure(
                        model, item, f"local-binding50-{view}-{item['id']}-{condition}", condition
                    )
                    result["teacher_queries"] += 1
                    keys.append((view, item["id"], condition))
                    if condition == "native":
                        native = row
                        control = controls[view, item["id"]]
                        if (
                            any(
                                row[k] != control[k]
                                for k in ("teacher_symbols", "teacher_tokens_correct")
                            )
                            or abs(row["teacher_ce"] - control["teacher_ce"]) > 1e-6
                        ):
                            raise ValueError("native teacher control differs")
                        result["controls"].append(dict(view=view, id=item["id"], reproduced=True))
                    result["rows"].append(
                        dict(
                            view=view,
                            id=item["id"],
                            kind=item.get("original_kind", item["kind"]),
                            condition=condition,
                            native_free_exact=controls[view, item["id"]]["free_exact"],
                            **{
                                k: v
                                for k, v in row.items()
                                if k not in ("id", "addresses", "logits")
                            },
                            address_tv_to_native=(
                                (row["addresses"] - native["addresses"]).abs().sum(-1) / 2
                            ).tolist(),
                        )
                    )
        validate_keys(keys)
        result["model_unchanged"] = before == paired.fingerprint(model.bundle(optimizer))
        if not result["model_unchanged"]:
            raise ValueError("parameters/Adam/RNG/original sessions changed")
        result.update(
            complete=True,
            status="complete",
            summary=summarize(result["rows"]),
            alignment_summary=alignment_summaries(result["rows"]),
        )
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = result["code_identity"] == code_identity()
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
    print(json.dumps(dict(read_only=True, teacher_sequences=1152, cpu_limit_seconds=60)))
