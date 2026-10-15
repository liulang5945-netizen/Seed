"""Approved PLAN39: frozen train-only fusion ablation, no parameter changes."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_question_alignment as replacing  # noqa: E402
from scripts.training import run_taiji_n1_residual_question as residual  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeEvidenceTape  # noqa: E402

REPORT = ROOT / "reports/taiji_n1_question_fusion_ablation_20261015.json"
SOURCE_SHAS = {
    "replacing": "bc4db237e0823d283116c78d066295dafdcd28bf0f058805d986775c5095d19e",
    "residual": "dd6a5ff3ba64c591eb0a6d006188106e81a5316bac5adca0848549257f3c0a97",
}
CONDITIONS = ("native", "bypass")


def checked_item(item):
    if item["split"] != "train":
        raise ValueError("only train model queries allowed")
    return item


def control_equal(row, old):
    return (
        all(row[k] == old[k] for k in ("raw_hex", "stop_reason", "teacher_tokens_correct"))
        and abs(row["teacher_ce"] - old["teacher_ce"]) <= 1e-6
    )


def entity_mask(item):
    checked_item(item)
    question = item["question"].encode()
    mask = torch.zeros(len(question), dtype=torch.bool)
    for entity in item["entities"]:
        text = entity.encode()
        start = question.find(text)
        if start >= 0:
            mask[start : start + len(text)] = True
    if int(mask.sum()) != 10:
        raise ValueError("unique ten-byte question entity required for offline profile")
    return mask


def bypass(tape, alignment):
    return (
        RelativeEvidenceTape(
            tape.session,
            tape.sources,
            tape.positions,
            tape.states,
            tape.keys,
            tape.values,
            tape.shift_indices,
        ),
        alignment,
    )


def profile(item, tape, alignment, attentions):
    mask = entity_mask(item).to(alignment.device)
    mass = alignment.mean(0)
    target = item["answer"].encode()
    span = []
    for source in tape.sources:
        offset = source.content.find(target)
        if offset >= 0:
            span.extend(
                i
                for i, (name, index) in enumerate(tape.positions)
                if name == source.source and offset <= index < offset + len(target)
            )
    teacher = torch.stack(attentions[: len(target)])
    entity_mass = float(mass[mask].sum())
    return {
        "entity_mass": entity_mass,
        "entity_bytes": int(mask.sum()),
        "template_mass": float(mass[~mask].sum()),
        "template_bytes": int((~mask).sum()),
        "entity_density": float(mass[mask].mean()),
        "template_density": float(mass[~mask].mean()),
        "answer_span_mass": float(teacher[:, span].sum(-1).mean()) if span else None,
    }


def complete_rows(rows):
    required = {
        (name, condition, f"train-{index:03d}")
        for name in ("replacing", "residual")
        for condition in CONDITIONS
        for index in range(192)
    }
    keys = [(r["model"], r["condition"], r["id"]) for r in rows]
    return len(keys) == len(required) and set(keys) == required


def identity():
    return {
        "audit": paired.sha(Path(__file__)),
        "replacing": replacing.code_identity(),
        "residual": residual.code_identity(),
    }


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive fusion diagnostic exists")
    frozen_code = identity()
    train = residual.checked_train()
    result = dict(
        schema="n1-question-fusion-ablation-v1",
        rows=[],
        code_identity=frozen_code,
        source_sha256=SOURCE_SHAS,
        parameter_updates=0,
        dev_queries=0,
        holdout_queries=0,
        teacher_queries=0,
        free_queries=0,
        can_promote=False,
        cpu_limit_seconds=60,
        status="running",
    )
    guards, inputs = [], []
    try:
        for name, runner in (("replacing", replacing), ("residual", residual)):
            if paired.sha(runner.REPORT) != SOURCE_SHAS[name]:
                raise ValueError("frozen source report changed")
            baseline = json.loads(runner.REPORT.read_text(encoding="utf-8"))
            if (
                not baseline["complete"]
                or baseline["completed_updates"] != 9216
                or baseline["code_identity"] != runner.code_identity()
            ):
                raise ValueError("complete frozen source required")
            finals = [c for c in baseline["restore_checks"] if Path(c["path"]).name == "final.pt"]
            if len(finals) != 1:
                raise ValueError("unique final required")
            entry = finals[0]
            path = Path(entry["path"])
            if paired.sha(path) != entry["sha256"]:
                raise ValueError("frozen model changed")
            inputs.append((path, entry["sha256"]))
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload["protocol"] != runner.PROTOCOL or payload["completed_updates"] != 9216:
                raise ValueError("frozen checkpoint protocol mismatch")
            cls = (
                runner.QuestionAlignedEvidenceCandidate
                if name == "replacing"
                else runner.ResidualQuestionEvidenceCandidate
            )
            model = cls.restore_bundle(payload["candidate"])
            before = paired.fingerprint(model.bundle())
            native_condition, native_step = model.condition, model.step
            old_rows = {r["id"]: r for r in baseline["rows"] if r["completed_updates"] == 9216}
            for item in train:
                checked_item(item)
                for condition in CONDITIONS:
                    if time.process_time() >= 58:
                        raise paired.BudgetStop("fusion diagnostic reserve reached")
                    captured, attentions = {}, []

                    def conditioned(
                        self,
                        tape,
                        states,
                        mode=condition,
                        function=native_condition,
                        capture=captured,
                    ):
                        conditioned_tape, alignment = function(tape, states)
                        capture.update(tape=tape, alignment=alignment)
                        return (
                            bypass(tape, alignment)
                            if mode == "bypass"
                            else (conditioned_tape, alignment)
                        )

                    def stepped(self, *args, function=native_step, collected=attentions):
                        output = function(*args)
                        collected.append(output[2])
                        return output

                    model.condition = MethodType(conditioned, model)
                    model.step = MethodType(stepped, model)
                    session = f"fusion-{name}-{condition}-{item['id']}"
                    try:
                        row = runner.measure(model, item, "position", session)
                        result["teacher_queries"] += 1
                        result["free_queries"] += 1
                        if condition == "native" and not control_equal(row, old_rows[item["id"]]):
                            raise ValueError("native final train control mismatch")
                        row.update(
                            model=name,
                            condition=condition,
                            profile=profile(
                                item, captured["tape"], captured["alignment"], attentions
                            ),
                        )
                        result["rows"].append(row)
                    finally:
                        model._observations.pop(session, None)
            model.condition, model.step = native_condition, native_step
            guards.append(before == paired.fingerprint(model.bundle()))
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result.update(
        cpu_seconds=time.process_time(),
        models_unchanged=len(guards) == 2 and all(guards),
        source_files_unchanged=len(inputs) == 2
        and all(paired.sha(p) == s for p, s in inputs)
        and all(
            paired.sha(r.REPORT) == SOURCE_SHAS[n]
            for n, r in (("replacing", replacing), ("residual", residual))
        ),
        manifest_unchanged=paired.sha(residual.calibration.MANIFEST)
        == residual.calibration.MANIFEST_SHA,
        code_unchanged=identity() == frozen_code,
    )
    summary = {}
    for name in ("replacing", "residual"):
        native = {
            r["id"]: r for r in result["rows"] if r["model"] == name and r["condition"] == "native"
        }
        changed = {
            r["id"]: r for r in result["rows"] if r["model"] == name and r["condition"] == "bypass"
        }
        summary[name] = {
            c: {
                k: sum(
                    r["free_exact"]
                    for r in result["rows"]
                    if r["model"] == name and r["condition"] == c and r["kind"] == k
                )
                for k in paired.KINDS
            }
            for c in CONDITIONS
        }
        summary[name]["paired"] = {
            "native_only": sum(
                native[i]["free_exact"] and not changed[i]["free_exact"]
                for i in native.keys() & changed.keys()
            ),
            "bypass_only": sum(
                changed[i]["free_exact"] and not native[i]["free_exact"]
                for i in native.keys() & changed.keys()
            ),
        }
    result["summary"] = summary
    result["complete"] = (
        result["status"] == "complete"
        and complete_rows(result["rows"])
        and result["teacher_queries"] == result["free_queries"] == 768
        and result["cpu_seconds"] <= 60
        and all(
            result[k]
            for k in (
                "models_unchanged",
                "source_files_unchanged",
                "manifest_unchanged",
                "code_unchanged",
            )
        )
    )
    paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = audit()
    print(json.dumps({k: output[k] for k in ("complete", "cpu_seconds", "summary")}))
    raise SystemExit(0 if output["complete"] else 1)
