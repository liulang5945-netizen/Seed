"""Representation probe: can the *existing* text representation separate
"the same content said differently" from "unrelated content"?

The address line is closed at the byte level (anchored iff the record occurs
verbatim).  It has no trainable freedom and does not prove semantic matching.
This round asks the next question with zero training and zero new parameters:

    take 20 goals (4 objects x 5 places).  Each goal is written four ways -- the
    entity descriptions (``describe_goal``) plus the three paraphrases
    (``needs`` / ``hope`` / ``state``).  That is 80 sentences, 4 per goal.
    A *same* pair is two wordings of one goal (20 x C(4,2) = 120).
    An *unrelated* pair is two sentences whose goals share neither object nor
    place: 120 unrelated goal pairs x 16 sentence pairs = 1,920.  The relation is
    enumerated completely instead of subsampling one sentence per goal, because
    a subsample would have to choose a template after seeing the fixture.  The
    imbalance (120 vs 1,920) is handled by the AUC and by per-distribution
    quantiles; no threshold is applied to the counts.  Cross-goal pairs that
    share exactly one entity are a *near* group (70 goal pairs, 1,120 sentence
    pairs): printed but never entering the distributions or the verdict.

For each representation we print **both distributions** -- same-pair distance
and unrelated-pair distance -- with min / p10 / p25 / median / p75 / p90 / max,
the Mann-Whitney AUC, the gap ``p10(unrelated) - p90(same)``, and three overlap
measures (unrelated at or below p90(same), same at or above p10(unrelated), and
the histogram overlap coefficient).  A high accuracy with heavy overlap is not
separation, so the overlap is reported next to the accuracy, not behind it.

Representations (all read-only, no fitting):

* R1 -- byte histogram ``content_feature`` (128 bins, L2-normalised; this is the
  existing content key, ``content_centered`` left at its default False).
* R2 -- mean of the project encoder's context vectors
  (``reference.encoder.context(raw).mean(0)``); the weights are the random
  initialisation at seed 20261031, i.e. an *untrained* encoder.
* R3 -- the frozen sentence-transformer, only as an upper bound.  It needs a
  local cache; if the cache is absent it is reported ``unavailable`` and the
  criteria are not touched.
* B  -- character-bigram Jaccard (lexical baseline, same protocol).

Frozen verdict (first hit wins, per representation):

* ``separates``    iff gap > 0 and AUC >= 0.95 and 1-NN accuracy >= 0.90;
* ``overlaps``     iff gap <= 0 or AUC < 0.90;
* ``inconclusive`` otherwise.

Overall: all of R1/R2/B ``overlaps`` -> ``representation_not_supported``;
R1 ``separates`` and R2 ``overlaps`` -> ``lexical_overlap_only``;
R2 ``separates`` -> ``representation_supported_without_training``;
anything else -> ``inconclusive``.  R3 never decides the verdict.

Discipline: 0 training / 0 optimizer steps / 0 Adam / 0 confirm / 0 new
parameters; ``process_guard`` first; single-instance lock; 60 s watchdog; CPU
cap 200 s; readings written then renamed; dual runner sha.  No
``python -m black``.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import math
import os
import sys
import threading
import time
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.goal_paraphrase import STYLES, paraphrase_goal  # noqa: E402
from taiji.living_core import LivingCoreConfig, LivingCoreReference  # noqa: E402
from taiji.native_goal_world_wide import OBJECTS, PLACES, describe_goal  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

MANIFEST = ROOT / "plans/manifests/taiji_representation_20261109.json"
REPORT = ROOT / "reports/taiji_representation_20261109.json"
PROGRESS = ROOT / "reports/taiji_representation_20261109_progress.jsonl"
LOCK = ROOT / "reports/taiji_representation_20261109.lock"
WATCHDOG_MARKER = ROOT / "reports/taiji_representation_20261109_watchdog.json"
CPU_ABORT = ROOT / "reports/taiji_representation_20261109_cpu_abort.json"

VARIANTS = ("describe",) + STYLES
AUC_SEPARATES = 0.95
AUC_OVERLAP = 0.90
NN_SEPARATES = 0.90
HIST_BINS = 15
CPU_CAP = 200.0
WATCHDOG_SECONDS = 60.0
LOCK_STALE_SECONDS = 600.0
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FIXTURE_FILES = (
    "taiji/native_goal_world_wide.py",
    "taiji/goal_paraphrase.py",
    "taiji/living_core.py",
)

RUNNER_SHA_START = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]

DEFINITION = {
    "fixture": (
        "OBJECTS x PLACES = 4 x 5 = 20 goals.  Each goal is rendered four ways: "
        "describe_goal (entity descriptions) and the three paraphrases needs / hope / state. "
        "80 sentences, 4 per goal, no sentence reused."
    ),
    "same_pair": "two distinct wordings of the same (object, place) goal: 20 x C(4,2) = 120.",
    "unrelated_pair": (
        "two sentences whose goals differ in object AND place: 120 unrelated goal pairs x 16 "
        "sentence pairs = 1,920.  Enumerated completely, not subsampled: picking one rendering per "
        "goal would be a template choice made after seeing the fixture.  A cross-goal pair that "
        "shares exactly one entity is a near pair (70 goal pairs, 1,120 sentence pairs); near pairs "
        "are printed and excluded from the distributions and the verdict."
    ),
    "imbalance": (
        "same 120 vs unrelated 1,920.  The AUC and the per-distribution quantiles are defined for "
        "unequal counts, and no verdict threshold is applied to the counts themselves."
    ),
    "distance": "1 - cosine for R1/R2/R3 (R1 and R3 are unit vectors; R2 is normalised here).",
    "R1": (
        "LivingCoreReference.content_feature: 128-bin byte histogram, L2-normalised, "
        "content_centered kept at its default False.  Order-free by construction."
    ),
    "R2": (
        "mean over positions of reference.encoder.context(raw), 128 dims.  Weights are the random "
        "initialisation of LivingCoreReference(seed=20261031); no gradient step is taken, so this "
        "is the untrained encoder, not a learned representation."
    ),
    "R3": (
        "instruments.document_embedding.DocumentEmbedder "
        "(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, mean pooling, L2-normalised). "
        "Upper bound only.  It is used iff the weights are already in the local HuggingFace cache; "
        "no download is attempted and the criteria do not depend on it."
    ),
    "B": "character-bigram Jaccard distance, 1 - |A & B| / |A | B|, same pair protocol.",
    "distribution": (
        "for each representation both distributions are reported in full: min, p10, p25, median, "
        "p75, p90, max, mean, over the frozen 120 same pairs and 120 unrelated pairs."
    ),
    "overlap": (
        "three readings per representation: unrelated pairs at or below p90(same); same pairs at "
        "or above p10(unrelated); histogram overlap coefficient sum_i min(s_i/n_s, u_i/n_u) over "
        "15 equal-width bins spanning the pooled range.  Overlap is reported next to accuracy "
        "because a high accuracy can come from a threshold that happens to sit in the middle."
    ),
    "classification": (
        "leave-one-out 1-NN over the 80 sentences: a sentence is correct iff its nearest other "
        "sentence belongs to the same goal.  Cosine distance, ties broken by smaller index."
    ),
    "gap": "p10(unrelated) - p90(same); positive means the central bands do not overlap.",
    "auc": "Mann-Whitney P(distance(same) < distance(unrelated)), ties counted as 0.5.",
    "verdict_order": (
        "R1, R2 and B decide the overall verdict; R3 is an upper bound and never decides it.  "
        "All three overlaps -> representation_not_supported; R1 separates and R2 overlaps -> "
        "lexical_overlap_only; R2 separates -> representation_supported_without_training; "
        "otherwise inconclusive."
    ),
}

CRITERIA = {
    "C1_freeze": (
        "the manifest is written before the first reading with the runner sha, the three fixture "
        "file hashes, the pair definitions and the verdict thresholds; the run refuses to start if "
        "a fixture file changed after the freeze."
    ),
    "C2_counts": (
        "80 sentences, 120 same pairs, 1,920 unrelated pairs, 1,120 near pairs; every pair is "
        "flagged same, unrelated or near before any distance is computed."
    ),
    "C3_distributions": (
        "for R1, R2, B and (when cached) R3 both distributions are printed with the seven "
        "quantiles, the AUC, the gap and the three overlap readings."
    ),
    "C4_classification": "leave-one-out 1-NN accuracy is reported on the same 80 sentences.",
    "C5_verdict": (
        "per representation separates iff gap > 0 and AUC >= 0.95 and 1-NN >= 0.90; overlaps iff "
        "gap <= 0 or AUC < 0.90; otherwise inconclusive.  The overall verdict follows the frozen "
        "order and is not revisited after the reading."
    ),
    "C6_no_fit": (
        "0 training, 0 optimizer steps, 0 Adam, 0 confirm, 0 new parameters; the only tensors "
        "allocated are the random-initialised reference and the frozen cached model."
    ),
}

TORCH_VERSION = torch.__version__


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def append_progress(entry: dict[str, Any]) -> None:
    LAST_READING[0] = time.time()
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def watchdog() -> None:
    def loop() -> None:
        while True:
            time.sleep(1.0)
            idle = time.time() - LAST_READING[0]
            if idle > WATCHDOG_SECONDS:
                write_json(
                    WATCHDOG_MARKER,
                    {"fired": True, "idle_seconds": round(idle, 3), "at": time.time()},
                )
                os._exit(70)
            if time.process_time() - START_CPU > CPU_CAP:
                write_json(
                    CPU_ABORT,
                    {"aborted": True, "cpu": round(time.process_time() - START_CPU, 3)},
                )
                os._exit(71)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


def release_lock() -> None:
    LOCK.unlink(missing_ok=True)


def acquire_lock() -> None:
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < LOCK_STALE_SECONDS:
            raise SystemExit(f"another run holds the lock ({age:.0f}s old): {LOCK}")
        LOCK.unlink()
    try:
        handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            raise SystemExit(f"lock race: {LOCK}") from exc
        raise
    os.write(handle, str(os.getpid()).encode("ascii"))
    os.close(handle)
    atexit.register(release_lock)


def fixture_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FIXTURE_FILES}


def build_fixture() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in OBJECTS:
        for place in PLACES:
            for variant in VARIANTS:
                if variant == "describe":
                    text = describe_goal(obj, place)
                else:
                    text = paraphrase_goal(obj, place, variant)
                rows.append(
                    {
                        "goal": f"{obj}|{place}",
                        "object": obj,
                        "place": place,
                        "variant": variant,
                        "text": text,
                    }
                )
    return rows


def build_pairs(rows: list[dict[str, Any]]) -> tuple[list, list, list]:
    same, unrelated, near = [], [], []
    for left, right in combinations(range(len(rows)), 2):
        a, b = rows[left], rows[right]
        if a["goal"] == b["goal"]:
            same.append((left, right))
        elif a["object"] != b["object"] and a["place"] != b["place"]:
            unrelated.append((left, right))
        else:
            near.append((left, right))
    return same, unrelated, near


def bigrams(text: str) -> set[str]:
    return {text[index : index + 2] for index in range(len(text) - 1)}


def bigram_distance(first: str, second: str) -> float:
    left, right = bigrams(first), bigrams(second)
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def quantile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": round(ordered[0], 6),
        "p10": round(quantile(ordered, 0.10), 6),
        "p25": round(quantile(ordered, 0.25), 6),
        "median": round(quantile(ordered, 0.50), 6),
        "p75": round(quantile(ordered, 0.75), 6),
        "p90": round(quantile(ordered, 0.90), 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def auc_lower(same: list[float], unrelated: list[float]) -> float:
    pooled = sorted([(value, 0) for value in same] + [(value, 1) for value in unrelated])
    ranks = [0.0] * len(pooled)
    index = 0
    while index < len(pooled):
        end = index
        while end + 1 < len(pooled) and pooled[end + 1][0] == pooled[index][0]:
            end += 1
        average = (index + 1 + end + 1) / 2.0
        for slot in range(index, end + 1):
            ranks[slot] = average
        index = end + 1
    unrelated_rank = sum(rank for rank, (_, label) in zip(ranks, pooled, strict=True) if label)
    n_same, n_unrelated = len(same), len(unrelated)
    return (unrelated_rank - n_unrelated * (n_unrelated + 1) / 2.0) / (n_same * n_unrelated)


def histogram_overlap(same: list[float], unrelated: list[float]) -> float:
    low = min(min(same), min(unrelated))
    high = max(max(same), max(unrelated))
    if high == low:
        return 1.0
    width = (high - low) / HIST_BINS
    left = [0] * HIST_BINS
    right = [0] * HIST_BINS
    for value in same:
        left[min(int((value - low) / width), HIST_BINS - 1)] += 1
    for value in unrelated:
        right[min(int((value - low) / width), HIST_BINS - 1)] += 1
    return sum(min(a / len(same), b / len(unrelated)) for a, b in zip(left, right, strict=True))


def distance_reading(same_values: list[float], unrelated_values: list[float]) -> dict[str, Any]:
    same_block = distribution(same_values)
    unrelated_block = distribution(unrelated_values)
    gap = unrelated_block["p10"] - same_block["p90"]
    below = sum(1 for value in unrelated_values if value <= same_block["p90"])
    above = sum(1 for value in same_values if value >= unrelated_block["p10"])
    return {
        "same": same_block,
        "unrelated": unrelated_block,
        "gap_p10_unrelated_minus_p90_same": round(gap, 6),
        "auc": round(auc_lower(same_values, unrelated_values), 6),
        "unrelated_at_or_below_p90_same": {
            "n": below,
            "of": len(unrelated_values),
            "fraction": round(below / len(unrelated_values), 6),
        },
        "same_at_or_above_p10_unrelated": {
            "n": above,
            "of": len(same_values),
            "fraction": round(above / len(same_values), 6),
        },
        "histogram_overlap": round(histogram_overlap(same_values, unrelated_values), 6),
        "histogram_bins": HIST_BINS,
    }


def nearest_neighbour(matrix: list[list[float]], goals: list[str]) -> dict[str, Any]:
    rows = []
    correct = 0
    for index, distances in enumerate(matrix):
        best, best_distance = None, None
        for other, distance in enumerate(distances):
            if other == index:
                continue
            if best_distance is None or distance < best_distance:
                best, best_distance = other, distance
        hit = goals[best] == goals[index]
        correct += int(hit)
        rows.append(
            {
                "index": index,
                "goal": goals[index],
                "nearest": best,
                "nearest_goal": goals[best],
                "distance": round(best_distance, 6),
                "correct": hit,
            }
        )
    return {
        "accuracy": round(correct / len(matrix), 6),
        "correct": correct,
        "of": len(matrix),
        "rows": rows,
    }


def cosine_matrix(vectors: torch.Tensor) -> list[list[float]]:
    unit = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-12)
    similarity = unit @ unit.T
    return (1.0 - similarity).tolist()


def classify(reading: dict[str, Any], accuracy: float) -> str:
    gap = reading["gap_p10_unrelated_minus_p90_same"]
    auc = reading["auc"]
    if gap > 0 and auc >= AUC_SEPARATES and accuracy >= NN_SEPARATES:
        return "separates"
    if gap <= 0 or auc < AUC_OVERLAP:
        return "overlaps"
    return "inconclusive"


def load_frozen_embedder() -> tuple[Any, dict[str, Any]]:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    cached = cache.is_dir() and any(
        MODEL_ID.split("/")[-1] in entry.name for entry in cache.iterdir()
    )
    if not cached:
        return None, {
            "status": "unavailable",
            "reason": "model_not_cached",
            "model_id": MODEL_ID,
            "cache_dir": str(cache),
            "note": "no download attempted; the criteria do not depend on this arm",
        }
    try:
        from instruments.document_embedding import DocumentEmbedder

        embedder = DocumentEmbedder(model_id=MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - an unavailable arm must not kill the run
        return None, {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "model_id": MODEL_ID,
        }
    return embedder, {"status": "available", "payload": embedder.to_payload()}


def representation(name: str, texts: list[str], reference: LivingCoreReference) -> torch.Tensor:
    if name == "R1":
        return torch.stack([reference.content_feature(text) for text in texts]).to(torch.float64)
    if name == "R2":
        with torch.no_grad():
            rows = [reference.encoder.context(text.encode("utf-8")).mean(0) for text in texts]
        return torch.stack(rows).to(torch.float64)
    raise ValueError(f"unknown representation {name!r}")


def phase_freeze() -> dict[str, Any]:
    rows = build_fixture()
    same, unrelated, near = build_pairs(rows)
    manifest = {
        "schema": "taiji-representation-v1",
        "date": "2026-11-09",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runner_sha256": RUNNER_SHA_START,
        "fixture_sha256": fixture_hashes(),
        "fixture_files": list(FIXTURE_FILES),
        "question": (
            "with 20 goals written four ways each, does an existing text representation put the "
            "120 same-content pairs closer than the 120 unrelated pairs -- and do the two distance "
            "distributions actually separate, or only look separated under a threshold?"
        ),
        "counts": {
            "goals": len(OBJECTS) * len(PLACES),
            "variants": list(VARIANTS),
            "sentences": len(rows),
            "same_pairs": len(same),
            "unrelated_pairs": len(unrelated),
            "near_pairs_excluded_from_verdict": len(near),
        },
        "thresholds": {
            "auc_separates": AUC_SEPARATES,
            "auc_overlap": AUC_OVERLAP,
            "nn_separates": NN_SEPARATES,
            "histogram_bins": HIST_BINS,
        },
        "representation_config": {
            "reference_seed": LivingCoreConfig().seed,
            "content_centered": LivingCoreConfig().content_centered,
            "sc_sfk_enabled": LivingCoreConfig().sc_sfk_enabled,
            "torch_version": TORCH_VERSION,
            "R3_policy": (
                "used iff the weights are already in the local HuggingFace cache; otherwise "
                "unavailable and excluded from the verdict"
            ),
        },
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "cost_estimate": {
            "estimate_cpu_seconds": 30.0,
            "note": (
                "80 sentences x 2 encoders (128 dims, ~20 bytes each) + 3,160 pair distances + "
                "4 x 80x80 matrices; the reference is constructed once.  Below the 200 CPU cap."
            ),
        },
    }
    if MANIFEST.exists():
        raise SystemExit(f"manifest already frozen: {MANIFEST}")
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(json.dumps(manifest["cost_estimate"], ensure_ascii=False))
    return manifest


def phase_run(manifest: dict[str, Any]) -> dict[str, Any]:
    if fixture_hashes() != manifest["fixture_sha256"]:
        raise SystemExit("a fixture file changed after the freeze")
    rows = build_fixture()
    same, unrelated, near = build_pairs(rows)
    expected = manifest["counts"]
    counts = {
        "sentences": len(rows),
        "same_pairs": len(same),
        "unrelated_pairs": len(unrelated),
        "near_pairs_excluded_from_verdict": len(near),
    }
    for key, value in counts.items():
        if value != expected[key]:
            raise SystemExit(f"{key}: {value} != frozen {expected[key]}")
    append_progress({"phase": "fixture_built", **counts})

    texts = [row["text"] for row in rows]
    goals = [row["goal"] for row in rows]
    reference = LivingCoreReference()
    reference.eval()

    readings: dict[str, Any] = {}
    for name in ("R1", "R2"):
        vectors = representation(name, texts, reference)
        matrix = cosine_matrix(vectors)
        same_values = [matrix[i][j] for i, j in same]
        unrelated_values = [matrix[i][j] for i, j in unrelated]
        reading = distance_reading(same_values, unrelated_values)
        reading["nearest_neighbour"] = nearest_neighbour(matrix, goals)
        reading["verdict"] = classify(reading, reading["nearest_neighbour"]["accuracy"])
        reading["pairs"] = {
            "same": [{"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in same],
            "unrelated": [
                {"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in unrelated
            ],
            "near_excluded": [
                {"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in near
            ],
        }
        readings[name] = reading
        append_progress(
            {
                "phase": "representation_done",
                "name": name,
                "verdict": reading["verdict"],
                "auc": reading["auc"],
            }
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "verdict": reading["verdict"],
                    "same_median": reading["same"]["median"],
                    "unrelated_median": reading["unrelated"]["median"],
                    "gap": reading["gap_p10_unrelated_minus_p90_same"],
                    "auc": reading["auc"],
                    "nn": reading["nearest_neighbour"]["accuracy"],
                },
                ensure_ascii=False,
            )
        )

    lexical_values = [bigram_distance(texts[i], texts[j]) for i, j in same]
    lexical_unrelated = [bigram_distance(texts[i], texts[j]) for i, j in unrelated]
    lexical = distance_reading(lexical_values, lexical_unrelated)
    lexical["nearest_neighbour"] = nearest_neighbour(
        [
            [0.0 if i == j else bigram_distance(texts[i], texts[j]) for j in range(len(texts))]
            for i in range(len(texts))
        ],
        goals,
    )
    lexical["verdict"] = classify(lexical, lexical["nearest_neighbour"]["accuracy"])
    readings["B"] = lexical
    append_progress({"phase": "representation_done", "name": "B", "verdict": lexical["verdict"]})
    print(
        json.dumps(
            {
                "name": "B",
                "verdict": lexical["verdict"],
                "same_median": lexical["same"]["median"],
                "unrelated_median": lexical["unrelated"]["median"],
                "gap": lexical["gap_p10_unrelated_minus_p90_same"],
                "auc": lexical["auc"],
                "nn": lexical["nearest_neighbour"]["accuracy"],
            },
            ensure_ascii=False,
        )
    )

    embedder, frozen_status = load_frozen_embedder()
    if embedder is not None:
        vectors = embedder.embed(texts).to(torch.float64)
        matrix = cosine_matrix(vectors)
        reading = distance_reading(
            [matrix[i][j] for i, j in same], [matrix[i][j] for i, j in unrelated]
        )
        reading["nearest_neighbour"] = nearest_neighbour(matrix, goals)
        reading["verdict"] = classify(reading, reading["nearest_neighbour"]["accuracy"])
        readings["R3"] = reading
    else:
        readings["R3"] = {"status": "unavailable", "detail": frozen_status}
    append_progress(
        {"phase": "representation_done", "name": "R3", "status": frozen_status["status"]}
    )

    decision_arms = {name: readings[name]["verdict"] for name in ("R1", "R2", "B")}
    if all(value == "overlaps" for value in decision_arms.values()):
        verdict = "representation_not_supported"
    elif decision_arms["R1"] == "separates" and decision_arms["R2"] == "overlaps":
        verdict = "lexical_overlap_only"
    elif decision_arms["R2"] == "separates":
        verdict = "representation_supported_without_training"
    else:
        verdict = "inconclusive"

    disclosures = [
        "R2 is the untrained encoder at seed 20261031: no gradient step, no fitted head. Its "
        "reading is a property of the initialisation plus the input bytes, not of learning.",
        "R1 is an order-free byte histogram. If it separates, the separation is lexical (the two "
        "wordings share the object and place names); that is reported as lexical_overlap_only, "
        "not as semantic matching.",
        "near pairs (cross-goal, one shared entity) are printed with their distances but never "
        "enter the distributions or the verdict.",
        "overlap is reported next to accuracy on purpose: a high 1-NN accuracy can come from a "
        "threshold that happens to sit between two heavily overlapping distributions, and this "
        "round must not read that as separation.",
        "the 20-goal fixture is built from entity descriptions and three sentence templates. It "
        "does not contain free prose; no claim about open-domain paraphrase is possible.",
        "R3 is an upper bound only; when the local cache is absent it is unavailable and no "
        "download is attempted.",
        "0 training, 0 optimizer steps, 0 Adam, 0 confirm, 0 new parameters.",
    ]
    report = {
        "schema": "taiji-representation-v1",
        "date": "2026-11-09",
        "verdict": verdict,
        "decision_arms": decision_arms,
        "runner_sha_start": RUNNER_SHA_START,
        "runner_sha_end": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "fixture_sha256": fixture_hashes(),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_frozen_at": manifest.get("frozen_at"),
        "counts": counts,
        "thresholds": manifest["thresholds"],
        "representation_config": manifest["representation_config"],
        "definition": DEFINITION,
        "criteria": CRITERIA,
        "readings": readings,
        "sentences": [
            {
                "index": index,
                "goal": row["goal"],
                "variant": row["variant"],
                "text": row["text"],
            }
            for index, row in enumerate(rows)
        ],
        "disclosures": disclosures,
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
        },
    }
    write_json(REPORT, report)
    append_progress({"phase": "done", "verdict": verdict})
    print(json.dumps({"verdict": verdict, "decision_arms": decision_arms}, ensure_ascii=False))
    return report


def main(argv: list[str]) -> int:
    assert_no_black_processes()
    if "--freeze" in argv:
        acquire_lock()
        watchdog()
        phase_freeze()
        return 0
    if not MANIFEST.exists():
        raise SystemExit("manifest missing: run --freeze first")
    acquire_lock()
    watchdog()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    phase_run(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
