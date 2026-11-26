"""R3 upper-bound arm for the representation probe.

The frozen round (``probe_taiji_representation_20261109.py``, runner sha
``b04ef3de…``) judged ``representation_not_supported`` on three arms: the byte
histogram R1, the untrained project encoder R2, and the character-bigram
baseline B.  Its R3 arm -- a frozen sentence-transformer -- was reported
``unavailable`` because the weights were not in the local cache and the frozen
policy forbade a download.

This script fills that single cell without touching the frozen artifact:

* it reuses the frozen fixture and pair set by importing the frozen runner as a
  module (no edit, no re-freeze, no change to the recorded runner sha);
* it downloads the preregistered model id on first use, then embeds the same 80
  sentences and reports the same distribution block;
* it writes a separate report, ``taiji_representation_20261109_r3.json``, and
  never modifies ``taiji_representation_20261109.json``.

Scope: R3 is an **upper bound only**.  It never enters the overall verdict and
cannot upgrade ``representation_not_supported``.  Its job is to answer one
question the frozen round could not: *is there any separation available on this
fixture at all, from a frozen general-purpose text encoder?*  Zero training,
zero optimizer steps, zero new parameters.

Discipline: ``process_guard`` first; single-instance lock; CPU cap 200 s; an
idle watchdog long enough for a model download; readings written then renamed;
0 training / 0 Adam / 0 confirm.  No ``python -m black``.
"""

from __future__ import annotations

import atexit
import errno
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.probe_taiji_representation_20261109 as frozen  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

REPORT = ROOT / "reports/taiji_representation_20261109_r3.json"
PROGRESS = ROOT / "reports/taiji_representation_20261109_r3_progress.jsonl"
LOCK = ROOT / "reports/taiji_representation_20261109_r3.lock"
FROZEN_REPORT = ROOT / "reports/taiji_representation_20261109.json"
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CPU_CAP = 200.0
IDLE_LIMIT = 900.0
LOCK_STALE_SECONDS = 600.0

START_CPU = time.process_time()
START_WALL = time.time()
LAST_READING = [time.time()]
RUNNER_SHA = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
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
            if idle > IDLE_LIMIT:
                os._exit(70)
            if time.process_time() - START_CPU > CPU_CAP:
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


def main() -> int:
    assert_no_black_processes()
    acquire_lock()
    watchdog()

    frozen_report = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    if frozen_report["verdict"] != "representation_not_supported":
        raise SystemExit(f"unexpected frozen verdict: {frozen_report['verdict']}")

    rows = frozen.build_fixture()
    same, unrelated, near = frozen.build_pairs(rows)
    texts = [row["text"] for row in rows]
    goals = [row["goal"] for row in rows]
    append_progress({"phase": "fixture_built", "sentences": len(texts), "same": len(same)})

    status: dict[str, Any] = {"status": "unavailable", "model_id": MODEL_ID}
    reading: dict[str, Any] | None = None
    try:
        from instruments.document_embedding import DocumentEmbedder

        append_progress({"phase": "loading_model", "model_id": MODEL_ID})
        embedder = DocumentEmbedder(model_id=MODEL_ID)
        status = {"status": "available", "payload": embedder.to_payload()}
        append_progress({"phase": "model_loaded", "revision": embedder.revision})
        vectors = embedder.embed(texts).to(torch.float64)
    except Exception as exc:  # noqa: BLE001 - an unavailable arm must not kill the run
        status = {
            "status": "unavailable",
            "model_id": MODEL_ID,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        vectors = None
    append_progress({"phase": "embedded", "status": status["status"]})

    if vectors is not None:
        matrix = frozen.cosine_matrix(vectors)
        reading = frozen.distance_reading(
            [matrix[i][j] for i, j in same], [matrix[i][j] for i, j in unrelated]
        )
        reading["nearest_neighbour"] = frozen.nearest_neighbour(matrix, goals)
        reading["verdict"] = frozen.classify(reading, reading["nearest_neighbour"]["accuracy"])
        reading["role"] = "upper_bound_only_not_in_verdict"
        reading["pairs"] = {
            "same": [{"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in same],
            "unrelated": [
                {"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in unrelated
            ],
            "near_excluded": [
                {"i": i, "j": j, "distance": round(matrix[i][j], 6)} for i, j in near
            ],
        }

    report = {
        "schema": "taiji-representation-r3-v1",
        "date": "2026-11-09",
        "arm": "R3",
        "role": "upper_bound_only_not_in_verdict",
        "frozen_report": str(FROZEN_REPORT.relative_to(ROOT)),
        "frozen_report_sha256": hashlib.sha256(FROZEN_REPORT.read_bytes()).hexdigest(),
        "frozen_verdict": frozen_report["verdict"],
        "frozen_runner_sha256": frozen_report["runner_sha_start"],
        "runner_sha256": RUNNER_SHA,
        "counts": {
            "sentences": len(texts),
            "same_pairs": len(same),
            "unrelated_pairs": len(unrelated),
            "near_pairs_excluded_from_verdict": len(near),
        },
        "model": status,
        "reading": reading,
        "disclosures": [
            "R3 is an upper bound only. It cannot upgrade representation_not_supported; the "
            "frozen decision arms R1/R2/B are unchanged.",
            "The frozen report and manifest are not modified; this is a separate artifact with "
            "its own runner sha.",
            "R3 is a frozen general-purpose encoder: no fine-tuning, no classifier, no new "
            "parameters. Its separation (or lack of it) speaks to whether the fixture admits a "
            "semantic bridge, not to what the project's own representation can learn.",
            "if the weights had to be downloaded, the model payload records the resolved revision "
            "and config digest so the reading is reproducible.",
        ],
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
            "idle_limit_seconds": IDLE_LIMIT,
        },
    }
    write_json(REPORT, report)
    append_progress({"phase": "done", "status": status["status"]})
    if reading is None:
        print(json.dumps({"R3": "unavailable", "detail": status}, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "R3": reading["verdict"],
                    "same_median": reading["same"]["median"],
                    "unrelated_median": reading["unrelated"]["median"],
                    "gap": reading["gap_p10_unrelated_minus_p90_same"],
                    "auc": reading["auc"],
                    "overlap": reading["histogram_overlap"],
                    "nn": reading["nearest_neighbour"]["accuracy"],
                    "cost": report["cost"],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
