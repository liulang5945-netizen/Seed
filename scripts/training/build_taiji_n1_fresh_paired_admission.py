"""PLAN76: zero-inference fresh paired admission, new sealed face and lossless layout."""

from __future__ import annotations

import argparse
import gzip
import json
import random
import string
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import _taiji_n1_evidence_shard as shard  # noqa: E402
from scripts.training import build_taiji_n1_confirmation_v4 as face  # noqa: E402
from scripts.training import (
    run_taiji_n1_fresh_full_history_calibration as calibration,
)  # noqa: E402

paired = calibration.paired
KINDS = face.KINDS
DIRECTIONS = face.DIRECTIONS
MANIFEST = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
REPORT = ROOT / "reports/taiji_n1_fresh_paired_admission_20261016.json"
CALIBRATION_SHA = "61dced2fe042f6f0e320b7e354afff227c732f82ec4acd601b18d8417f848e5a"
FORMAL = dict(
    schema="n1-fresh-full-history-paired-training-v1",
    seeds=[11, 12, 13],
    arms=["position", "terminal", "cut"],
    initialization=calibration.engineering.INITIALIZATION,
    updates_per_model=24576,
    total_updates=221184,
    total_qa_sequences=331776,
    milestones=[0, 6144, 12288, 18432, 24576],
    known_score_rows=25920,
    independent_restores=18,
    signature_clone_updates=72,
    synthetic_qa_sequences=144,
    cpu_limit_seconds=9000,
    parameters=62056,
    update_policy=calibration.update.POLICY,
    cycle_length=6144,
    cycles=4,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)
PROTOCOL = dict(
    schema="n1-fresh-paired-admission-v1",
    cpu_limit_seconds=30,
    model_loads=0,
    model_queries=0,
    parameter_updates=0,
    confirmation_queries=0,
    face_seed=76,
)


def identity():
    return dict(
        protocol=PROTOCOL,
        formal=FORMAL,
        calibration=calibration.identity(),
        builder=paired.sha(Path(__file__)),
        shard=paired.sha(Path(shard.__file__)),
        prior_face=paired.sha(Path(face.__file__)),
    )


def exclusions():
    names, words = face.exclusions()
    calibration.runner.seal.check_all()
    prior = json.loads(face.MANIFEST.read_text(encoding="utf-8"))
    names.update(n for e in prior["episodes"] for n in e["entities"])
    words.update(
        text.removesuffix(".").split()[-1]
        for e in prior["episodes"]
        for text in e["observations"]
        if " uses " in text
    )
    return names, words


def build():
    names, words = exclusions()
    rng = random.Random(76)

    def novel(length, used):
        while True:
            value = "".join(rng.choices(string.ascii_lowercase, k=length))
            if value not in used:
                used.add(value)
                return value

    episodes = []
    for index in range(32):
        kind = KINDS[index % 4]
        name, other, third = (novel(10, names) for _ in range(3))
        first, second = novel(6, words), novel(6, words)
        if kind == "lookup":
            observations = [f"{name} uses {first}.", f"{other} uses {second}."]
            rng.shuffle(observations)
            question, answer = f"material of {name}?", first
        elif kind == "revision":
            observations = [f"{name} uses {first}.", f"{name} now uses {second}."]
            question, answer = f"material of {name}?", second
        elif kind == "transition":
            start, turn = rng.randrange(4), rng.choice((-1, 1))
            observations = [
                f"{name} faces {DIRECTIONS[start]}.",
                f"{name} turns {'right' if turn==1 else 'left'}.",
            ]
            question, answer = f"direction of {name}?", DIRECTIONS[(start + turn) % 4]
        else:
            observations = [f"{other} uses {first}.", f"{third} uses {second}."]
            question, answer = f"material of {name}?", "unknown"
        episodes.append(
            dict(
                id=f"v5-confirm-{index:03d}",
                split="holdout",
                kind=kind,
                entities=[name, other, third],
                observations=observations,
                question=question,
                answer=answer,
            )
        )
    return dict(
        schema="n1-fresh-confirmation-v5",
        seed=76,
        excluded_source_sha256=[face.sha(p) for p in (*face.SOURCES, face.MANIFEST)],
        episodes=episodes,
    )


def validate(manifest):
    if manifest != build():
        raise ValueError("exact fixed-seed76 fresh confirmation required")
    names, words = exclusions()
    episodes = manifest["episodes"]
    new_names = [n for e in episodes for n in e["entities"]]
    new_words = [
        text.removesuffix(".").split()[-1]
        for e in episodes
        for text in e["observations"]
        if " uses " in text
    ]
    counts = dict(Counter(e["kind"] for e in episodes))
    if (
        len(episodes) != 32
        or len(set(new_names)) != 96
        or set(new_names) & names
        or len(new_words) != 48
        or len(set(new_words)) != 48
        or set(new_words) & words
        or counts != {k: 8 for k in KINDS}
        or any(e["split"] != "holdout" or len(e["observations"]) != 2 for e in episodes)
    ):
        raise ValueError("disjoint complete32/four-class fresh confirmation required")
    return dict(items=32, by_kind=counts, entities=96, materials=48, model_queries=0)


def readiness():
    if paired.sha(calibration.REPORT) != CALIBRATION_SHA:
        raise ValueError("complete frozen75 calibration required")
    record = json.loads(calibration.REPORT.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or not record["training_sufficient"]
        or record["code_identity"] != calibration.identity()
        or record["completed_updates"] != 24576
        or len(record["restore_checks"]) != 2
        or not calibration.runner.sufficient(record["milestones"][-1])
    ):
        raise ValueError("unchanged75 fixed terminal and original four gates required")
    guards, _ = calibration.readiness()
    guards[str(calibration.REPORT)] = CALIBRATION_SHA
    for check in record["restore_checks"]:
        if not check["equal"] or paired.sha(Path(check["path"])) != check["sha256"]:
            raise ValueError("75 original full recovery bytes changed")
        guards[check["path"]] = check["sha256"]
    exclusions()
    for path in (*face.SOURCES, face.MANIFEST, face.REPORT):
        guards[str(path)] = paired.sha(path)
    return record, guards


def write_lf_new(path, record):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def execute():
    if REPORT.exists() or MANIFEST.exists():
        raise FileExistsError("one exclusive fresh face/admission; no regeneration")
    record, guards = readiness()
    raw = calibration.REPORT.read_bytes()
    started = time.process_time()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    if gzip.decompress(compressed) != raw:
        raise ValueError("full real evidence compression must round-trip exactly")
    compression_cpu = time.process_time() - started
    manifest = build()
    summary = validate(manifest)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        formal_protocol=FORMAL,
        source_guards=guards,
        face_summary=summary,
        model_loads=0,
        model_queries=0,
        parameter_updates=0,
        confirmation_queries=0,
        can_promote=False,
        forecast_cpu_seconds=9 * record["cpu_seconds"],
        evidence_compression=dict(
            source_bytes=len(raw),
            compressed_bytes=len(compressed),
            full_round_trip_equal=True,
            cpu_seconds=compression_cpu,
        ),
        evidence_layout=dict(
            schema=shard.SCHEMA,
            compression="gzip6-mtime0",
            shards=9,
            per_model_counts=dict(losses=24576, measurements=24576, rows=2880),
        ),
    )
    if time.process_time() >= 29:
        raise ValueError("admission CPU reserve exhausted before exclusive face write")
    write_lf_new(MANIFEST, manifest)
    result["manifest"] = dict(
        path=str(MANIFEST), sha256=paired.sha(MANIFEST), bytes=MANIFEST.stat().st_size
    )
    result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
    result["code_unchanged"] = identity() == result["code_identity"]
    result["cpu_seconds"] = time.process_time()
    result["complete"] = (
        result["sources_unchanged"] and result["code_unchanged"] and result["cpu_seconds"] <= 30
    )
    write_lf_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.execute_approved:
        result = execute()
        print(
            json.dumps(
                {
                    k: result.get(k)
                    for k in ("complete", "cpu_seconds", "manifest", "forecast_cpu_seconds")
                }
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL, formal=FORMAL)))
