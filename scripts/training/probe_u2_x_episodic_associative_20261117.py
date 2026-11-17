"""22.5.X zero-training feasibility: five localized memory gaps + unit-graph bridge.

Pre-registered in plans/reference/DESIGN-TAIJI_U2_EPISODIC_ASSOCIATIVE_20261117.md
section 9 BEFORE any reading. Face S = EpisodicMemoryStore (record level long-term
store); Face G = RouteBGraph over unit-input-dms-v1 (the current unit representation).

The generating chain and the binary diagnostic head are reused from the W probes
byte-for-byte; the k-class head is new and is anchored against the reused binary
head at k=2. Zero graph optimizer steps, zero weight writes, zero sealed reads:
the dataset is fully synthetic and no project data file is opened.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from scripts.training import probe_u2_w_dms_20261117 as wprobe
from scripts.training import run_u1_unit_read_20261116 as u1
from taiji import u2_unit_input_dms as dms
from taiji.contracts import EpisodicMemoryRecord
from taiji.episodic_memory import EpisodicMemoryStore
from taiji.u2_history import value_digest
from taiji.u2_neural_units import RouteBGraph, UnitGraphSpec

SMOKE_REPORT = ROOT / "reports/taiji_u2_x_feasibility_smoke_20261117.json"
PRICE_REPORT = ROOT / "reports/taiji_u2_x_feasibility_price_20261117.json"
RUN_REPORT = ROOT / "reports/taiji_u2_x_feasibility_20261117.json"
CARD = "22.5.X"
PREREG = "plans/reference/DESIGN-TAIJI_U2_EPISODIC_ASSOCIATIVE_20261117.md#9"

GRAPH_SEEDS = (2026111701, 2026111702, 2026111703)
DELAYS = (0, 1, 2, 4, 8)
STD_DELAY = dms.DELAY_STEPS
PLACES = ("kitchen", "hall", "garden", "cellar")
N_PLACE_EVENTS = 8
N_PLACE_QUERIES = 32
N_CONF_RECORDS = 128
N_SESSION_RECORDS = 8
N_TARGETS = 8
INTERFERENCE = (0, 8, 32, 64)
SMALL_CAPACITY = 8
LARGE_CAPACITY = 1024

TRIPLE_FIT_MIN = 0.75
POSITIVE_TRIPLE_MIN = 0.80
DERANGED_TRIPLE_MAX = 0.50
PLACE_NARROWING_EXACT = 1.0
PLACE_IN_CUE_TOP1_MIN = 0.80
PLACE_ATTRIBUTE_ONLY_TOP1_MAX = 0.50
PLACE_ARM_GAP_MIN = 0.30
PERMUTED_ORDERING_ABS_DIFF_MAX = 0.10
CROSS_SESSION_RETENTION_MIN = 0.80
GATE_ON_EXCLUSION_EXACT = 1.0

DESIGN_CAP_CPU = 120.0
DESIGN_CAP_WALL = 240.0
CAP_MARGIN = 2.0
RESERVE_CPU = 5.0
RESERVE_WALL = 10.0
PRICE_REPEATS = 5
STORE_SEED = 20261117
N_PERMUTATIONS = 20


def timing_stats(samples: list[tuple[float, float]]) -> dict[str, float]:
    cpu = sorted(sample[0] for sample in samples)
    wall = sorted(sample[1] for sample in samples)
    n = len(cpu)
    median = cpu[n // 2] if n % 2 else (cpu[n // 2 - 1] + cpu[n // 2]) / 2.0
    return {
        "repeats": n,
        "cpu_mean": round(sum(cpu) / n, 8),
        "cpu_median": round(median, 8),
        "cpu_max": round(cpu[-1], 8),
        "wall_mean": round(sum(wall) / n, 8),
        "wall_max": round(wall[-1], 8),
    }


def criteria() -> dict[str, Any]:
    return {
        "triple_fit_min": TRIPLE_FIT_MIN,
        "positive_triple_min": POSITIVE_TRIPLE_MIN,
        "deranged_triple_max": DERANGED_TRIPLE_MAX,
        "place_narrowing_exact": PLACE_NARROWING_EXACT,
        "place_in_cue_top1_min": PLACE_IN_CUE_TOP1_MIN,
        "place_attribute_only_top1_max": PLACE_ATTRIBUTE_ONLY_TOP1_MAX,
        "place_arm_gap_min": PLACE_ARM_GAP_MIN,
        "permuted_ordering_abs_diff_max": PERMUTED_ORDERING_ABS_DIFF_MAX,
        "cross_session_retention_min": CROSS_SESSION_RETENTION_MIN,
        "gate_on_exclusion_exact": GATE_ON_EXCLUSION_EXACT,
        "graph_seeds": list(GRAPH_SEEDS),
        "delays": list(DELAYS),
        "places": list(PLACES),
        "exits": [
            "budget_stop",
            "instrument_invalid",
            "task_not_learnable",
            "episodic_associative_feasible",
        ],
    }


def fit_multi(
    train_x: torch.Tensor,
    train_y: list[int],
    fit_x: torch.Tensor,
    fit_y: list[int],
    k: int,
) -> dict[str, Any]:
    """k-class diagnostic head. Same LBFGS settings as the reused binary head."""
    weights = torch.zeros(train_x.shape[1], k, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(train_y, dtype=torch.long)
    optimizer = torch.optim.LBFGS(
        [weights],
        max_iter=wprobe.LBFGS_ITER,
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(train_x @ weights, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        train_pred = (train_x @ weights).argmax(-1)
        fit_pred = (fit_x @ weights).argmax(-1)
    return {
        "k": k,
        "train_accuracy": float((train_pred == target).to(torch.float64).mean()),
        "fit_accuracy": float((fit_pred == torch.tensor(fit_y)).to(torch.float64).mean()),
        "fit_distinct": len(set(fit_pred.tolist())),
        "fit_pred": fit_pred.tolist(),
    }


def head_anchor(dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """At k=2 the new head must reproduce the reused binary head exactly."""
    graph = RouteBGraph(UnitGraphSpec(dtype="float64", seed=GRAPH_SEEDS[0]))
    train_x = torch.stack([wprobe.run_episode(graph, ep) for ep in dataset["train"]])
    fit_x = torch.stack([wprobe.run_episode(graph, ep) for ep in dataset["fit"]])
    train_y = [dms.label(ep) for ep in dataset["train"]]
    fit_y = [dms.label(ep) for ep in dataset["fit"]]
    binary = wprobe.fit_head(train_x, train_y, fit_x, fit_y)
    multi = fit_multi(train_x, train_y, fit_x, fit_y, 2)
    keys = ("train_accuracy", "fit_accuracy", "fit_distinct")
    return {
        "binary_head": {key: binary[key] for key in keys},
        "multi_head_k2": {key: multi[key] for key in keys},
        "exact": all(binary[key] == multi[key] for key in keys),
        "reused_helper": "scripts/training/probe_u2_w_dms_20261117.py:fit_head",
    }


def graph_features(
    graph: RouteBGraph, dataset: dict[str, list[dict[str, Any]]], delay: int
) -> tuple[torch.Tensor, torch.Tensor]:
    train_x = torch.stack([wprobe.run_episode(graph, ep, delay=delay) for ep in dataset["train"]])
    fit_x = torch.stack([wprobe.run_episode(graph, ep, delay=delay) for ep in dataset["fit"]])
    return train_x, fit_x


def oracle_features(episodes: list[dict[str, Any]], delay_index: int) -> torch.Tensor:
    rows = []
    for episode in episodes:
        vector = torch.zeros(4 + 8 + len(DELAYS), dtype=torch.float64)
        vector[episode["sample"]] = 1.0
        vector[4 + episode["position"]] = 1.0
        vector[12 + delay_index] = 1.0
        rows.append(vector)
    return torch.stack(rows)


def triple_of(
    what: dict[str, Any], where: dict[str, Any], when_pred: list[int], fit_y: dict[str, list[int]]
) -> float:
    hits = sum(
        int(w == y and p == q and d == r)
        for w, y, p, q, d, r in zip(
            what["fit_pred"],
            fit_y["what"],
            where["fit_pred"],
            fit_y["where"],
            when_pred,
            fit_y["when"],
            strict=True,
        )
    )
    return hits / len(fit_y["what"])


def bridge_face(graph: RouteBGraph, dataset: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    feats = {delay: graph_features(graph, dataset, delay) for delay in DELAYS}
    std_index = DELAYS.index(STD_DELAY)
    n_fit = len(dataset["fit"])
    fit_y = {
        "what": [ep["sample"] for ep in dataset["fit"]],
        "where": [ep["position"] for ep in dataset["fit"]],
        "when": [std_index] * n_fit,
    }
    train_y = {
        "what": [ep["sample"] for ep in dataset["train"]],
        "where": [ep["position"] for ep in dataset["train"]],
    }
    when_train_x = torch.cat([feats[delay][0] for delay in DELAYS])
    when_fit_x = torch.cat([feats[delay][1] for delay in DELAYS])
    when_train_y = [i for i in range(len(DELAYS)) for _ in dataset["train"]]
    when_fit_y = [i for i in range(len(DELAYS)) for _ in dataset["fit"]]

    what = fit_multi(feats[STD_DELAY][0], train_y["what"], feats[STD_DELAY][1], fit_y["what"], 4)
    where = fit_multi(feats[STD_DELAY][0], train_y["where"], feats[STD_DELAY][1], fit_y["where"], 8)
    when = fit_multi(when_train_x, when_train_y, when_fit_x, when_fit_y, len(DELAYS))
    when_std = when["fit_pred"][std_index * n_fit : (std_index + 1) * n_fit]
    measured = triple_of(what, where, when_std, fit_y)

    pos_train = torch.cat([oracle_features(dataset["train"], i) for i in range(len(DELAYS))])
    pos_fit = torch.cat([oracle_features(dataset["fit"], i) for i in range(len(DELAYS))])
    pos_what = fit_multi(
        oracle_features(dataset["train"], std_index),
        train_y["what"],
        oracle_features(dataset["fit"], std_index),
        fit_y["what"],
        4,
    )
    pos_where = fit_multi(
        oracle_features(dataset["train"], std_index),
        train_y["where"],
        oracle_features(dataset["fit"], std_index),
        fit_y["where"],
        8,
    )
    pos_when = fit_multi(pos_train, when_train_y, pos_fit, when_fit_y, len(DELAYS))
    pos_when_std = pos_when["fit_pred"][std_index * n_fit : (std_index + 1) * n_fit]
    positive = triple_of(pos_what, pos_where, pos_when_std, fit_y)

    der_what = fit_multi(
        feats[STD_DELAY][0],
        [(y + 1) % 4 for y in train_y["what"]],
        feats[STD_DELAY][1],
        fit_y["what"],
        4,
    )
    der_where = fit_multi(
        feats[STD_DELAY][0],
        [(y + 1) % 8 for y in train_y["where"]],
        feats[STD_DELAY][1],
        fit_y["where"],
        8,
    )
    der_when = fit_multi(
        when_train_x,
        [(y + 1) % len(DELAYS) for y in when_train_y],
        when_fit_x,
        when_fit_y,
        len(DELAYS),
    )
    der_when_std = der_when["fit_pred"][std_index * n_fit : (std_index + 1) * n_fit]
    deranged = triple_of(der_what, der_where, der_when_std, fit_y)

    return {
        "what": {key: what[key] for key in ("k", "train_accuracy", "fit_accuracy", "fit_distinct")},
        "where": {
            key: where[key] for key in ("k", "train_accuracy", "fit_accuracy", "fit_distinct")
        },
        "when": {key: when[key] for key in ("k", "train_accuracy", "fit_accuracy", "fit_distinct")},
        "triple_fit": measured,
        "positive_triple_fit": positive,
        "deranged_triple_fit": deranged,
        "derangement": "train labels shifted by +1 mod k, fit labels untouched",
        "std_delay": STD_DELAY,
        "when_fit_rows": len(when_fit_y),
        "collapsed_heads": [
            name
            for name, head in (("what", what), ("where", where), ("when", when))
            if head["fit_distinct"] < 2
        ],
        "collapse_note": (
            "a collapsed head on real features is a finding about the representation, "
            "not an instrument failure; the ruler's range is established by the oracle "
            "and deranged arms instead"
        ),
        "checks": {
            "positive_control_reaches_gate": positive >= POSITIVE_TRIPLE_MIN,
            "deranged_below_band": deranged <= DERANGED_TRIPLE_MAX,
            "ruler_has_dynamic_range": positive >= POSITIVE_TRIPLE_MIN
            and deranged <= DERANGED_TRIPLE_MAX,
        },
        "gates": {"triple_reaches_gate": measured >= TRIPLE_FIT_MIN},
    }


def _record(
    memory_id: str,
    episode_id: str,
    tick: int,
    cue: torch.Tensor,
    *,
    place: str | None = None,
    object_id: str | None = None,
    attribute: str | None = None,
) -> EpisodicMemoryRecord:
    return EpisodicMemoryRecord(
        memory_id=memory_id,
        episode_id=episode_id,
        tick=tick,
        cue=cue,
        provenance="experienced",
        object_ids=() if object_id is None else (object_id,),
        attribute_ids=((attribute if attribute is not None else f"place={place}"),),
    )


def _place_of(record: EpisodicMemoryRecord) -> str:
    for item in record.attribute_ids:
        if item.startswith("place="):
            return item.split("=", 1)[1]
    return ""


def _onehot(index: int, width: int) -> torch.Tensor:
    vector = torch.zeros(width, dtype=torch.float64)
    vector[index] = 1.0
    return vector


def place_face() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(STORE_SEED)
    arms: dict[str, Any] = {}
    narrowing: dict[str, bool] = {}
    for arm in ("place_in_cue", "place_attribute_only"):
        store = EpisodicMemoryStore(capacity=LARGE_CAPACITY, cue_dim=len(PLACES))
        bound: dict[str, set[str]] = {place: set() for place in PLACES}
        for index, place in enumerate(PLACES):
            for event in range(N_PLACE_EVENTS):
                memory_id = f"{arm}-{index}-{event}"
                if arm == "place_in_cue":
                    cue = _onehot(index, len(PLACES)) + 0.05 * torch.randn(
                        len(PLACES), generator=generator, dtype=torch.float64
                    )
                else:
                    cue = torch.full((len(PLACES),), 0.5, dtype=torch.float64)
                store.write(_record(memory_id, "sessionA", event, cue, place=place))
                bound[place].add(memory_id)
        correct = 0
        for query in range(N_PLACE_QUERIES):
            index = query % len(PLACES)
            cue = _onehot(index, len(PLACES)) + 0.05 * torch.randn(
                len(PLACES), generator=generator, dtype=torch.float64
            )
            hits = store.retrieve(cue, limit=1)
            correct += int(bool(hits) and _place_of(hits[0].record) == PLACES[index])
        arms[arm] = correct / N_PLACE_QUERIES
        for place in PLACES:
            hits = store.retrieve(
                _onehot(PLACES.index(place), len(PLACES)),
                limit=LARGE_CAPACITY,
                attribute=f"place={place}",
            )
            narrowing[f"{arm}:{place}"] = {hit.record.memory_id for hit in hits} == bound[place]
    gap = arms["place_in_cue"] - arms["place_attribute_only"]
    narrowing_rate = sum(narrowing.values()) / len(narrowing)
    return {
        "place_in_cue_top1": arms["place_in_cue"],
        "place_attribute_only_top1": arms["place_attribute_only"],
        "arm_gap": gap,
        "narrowing_rate": narrowing_rate,
        "narrowing_faces": len(narrowing),
        "queries_per_arm": N_PLACE_QUERIES,
        "checks": {
            "narrowing_exact": narrowing_rate == PLACE_NARROWING_EXACT,
            "arm_gap_wide_enough": gap >= PLACE_ARM_GAP_MIN,
            "ruler_has_dynamic_range": gap >= PLACE_ARM_GAP_MIN,
        },
        "gates": {
            "in_cue_reaches_gate": arms["place_in_cue"] >= PLACE_IN_CUE_TOP1_MIN,
            "attribute_only_below_band": arms["place_attribute_only"]
            <= PLACE_ATTRIBUTE_ONLY_TOP1_MAX,
        },
    }


def _error_rates(pairs: list[tuple[float, bool]]) -> tuple[float, float]:
    ordered = sorted(pairs, key=lambda item: item[0])
    half = len(ordered) // 2
    low = ordered[:half]
    high = ordered[half:]
    low_err = sum(int(not ok) for _, ok in low) / len(low)
    high_err = sum(int(not ok) for _, ok in high) / len(high)
    return high_err, low_err


def confidence_face() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(STORE_SEED + 1)
    store = EpisodicMemoryStore(capacity=LARGE_CAPACITY, cue_dim=len(PLACES))
    fields = {field.name for field in dataclasses.fields(EpisodicMemoryRecord)}
    for index in range(N_CONF_RECORDS):
        place_index = index % len(PLACES)
        scale = 0.05 + 1.50 * (index / N_CONF_RECORDS)
        cue = _onehot(place_index, len(PLACES)) + scale * torch.randn(
            len(PLACES), generator=generator, dtype=torch.float64
        )
        store.write(_record(f"conf-{index}", "sessionA", index, cue, place=PLACES[place_index]))
    pairs: list[tuple[float, bool]] = []
    alphas: list[float] = []
    for query in range(N_CONF_RECORDS):
        place_index = query % len(PLACES)
        other = (place_index + 1 + query // len(PLACES)) % len(PLACES)
        if other == place_index:
            other = (other + 1) % len(PLACES)
        alpha = 0.05 + 0.55 * (query / N_CONF_RECORDS)
        cue = (1.0 - alpha) * _onehot(place_index, len(PLACES)) + alpha * _onehot(
            other, len(PLACES)
        )
        cue = cue + 0.02 * torch.randn(len(PLACES), generator=generator, dtype=torch.float64)
        alphas.append(alpha)
        hits = store.retrieve(cue, limit=1)
        score = float(hits[0].score) if hits else 0.0
        ok = bool(hits) and _place_of(hits[0].record) == PLACES[place_index]
        pairs.append((score, ok))
    high_err, low_err = _error_rates(pairs)
    outcomes = [ok for _, ok in pairs]
    base_scores = [score for score, _ in pairs]
    permuted: list[float] = []
    for trial in range(N_PERMUTATIONS):
        scores = list(base_scores)
        random.Random(STORE_SEED + trial).shuffle(scores)
        perm_high, perm_low = _error_rates(list(zip(scores, outcomes, strict=True)))
        permuted.append(abs(perm_high - perm_low))
    ordering = high_err < low_err
    permuted_mean = sum(permuted) / len(permuted)
    error_count = sum(int(not ok) for ok in outcomes)
    oracle_high, oracle_low = _error_rates([(1.0 if ok else 0.0, ok) for ok in outcomes])
    oracle_ordering = oracle_high < oracle_low
    return {
        "record_has_confidence_field": "confidence" in fields,
        "record_fields": sorted(fields),
        "high_confidence_error_rate": high_err,
        "low_confidence_error_rate": low_err,
        "observed_abs_diff": abs(high_err - low_err),
        "error_count": error_count,
        "query_alpha_range": [min(alphas), max(alphas)],
        "permuted_abs_diff_mean": permuted_mean,
        "permuted_abs_diff_max": max(permuted),
        "permutations": N_PERMUTATIONS,
        "oracle_statistic_high_error_rate": oracle_high,
        "oracle_statistic_low_error_rate": oracle_low,
        "queries": len(pairs),
        "confidence_proxy": "retrieve cosine score (no per-record confidence exists)",
        "stimulus": (
            "competing partial cue (1-alpha)*onehot(place) + alpha*onehot(other), "
            "alpha 0.05-0.60; record noise scale 0.05-1.55"
        ),
        "checks": {
            "record_confidence_absent": "confidence" not in fields,
            "errors_exist": error_count > 0,
            "oracle_statistic_ordering_holds": oracle_ordering,
            "permuted_ordering_within_band": permuted_mean <= PERMUTED_ORDERING_ABS_DIFF_MAX,
            "ruler_has_dynamic_range": oracle_ordering
            and error_count > 0
            and permuted_mean <= PERMUTED_ORDERING_ABS_DIFF_MAX,
        },
        "gates": {"calibration_ordering_holds": ordering},
    }


def conflict_face() -> dict[str, Any]:
    cue = _onehot(0, len(PLACES))

    def build() -> EpisodicMemoryStore:
        store = EpisodicMemoryStore(capacity=LARGE_CAPACITY, cue_dim=len(PLACES))
        store.write(
            _record(
                "stale",
                "sessionA",
                0,
                cue,
                place=PLACES[0],
                object_id="obj0",
                attribute="color=red",
            )
        )
        store.write(
            _record(
                "fresh",
                "sessionA",
                1,
                cue,
                place=PLACES[0],
                object_id="obj0",
                attribute="color=blue",
            )
        )
        return store

    default = build()
    default_hits = default.retrieve(cue, limit=LARGE_CAPACITY)
    default_ids = {hit.record.memory_id for hit in default_hits}
    contradiction = default.count == 2 and {"stale", "fresh"} <= default_ids

    overwrite = build()
    overwrite.write(
        _record(
            "stale", "sessionA", 2, cue, place=PLACES[0], object_id="obj0", attribute="color=blue"
        )
    )
    overwrite_hits = overwrite.retrieve(cue, limit=LARGE_CAPACITY)
    overwrite_top = overwrite_hits[0].record.attribute_ids if overwrite_hits else ()
    overwrite_stale = any("color=red" in hit.record.attribute_ids for hit in overwrite_hits)

    expired = build()
    removed = expired.expire("stale")
    expired_hits = expired.retrieve(cue, limit=LARGE_CAPACITY)
    expired_top = expired_hits[0].record.attribute_ids if expired_hits else ()
    expired_stale = any("color=red" in hit.record.attribute_ids for hit in expired_hits)

    corrected_ok = (
        "color=blue" in overwrite_top
        and not overwrite_stale
        and "color=blue" in expired_top
        and not expired_stale
        and removed == 1
    )
    return {
        "default_reachable_ids": sorted(default_ids),
        "default_count": default.count,
        "contradiction_persists": contradiction,
        "conflict_marker_api_present": False,
        "overwrite_top_attribute": list(overwrite_top),
        "overwrite_stale_reachable": overwrite_stale,
        "expire_removed": removed,
        "expire_top_attribute": list(expired_top),
        "expire_stale_reachable": expired_stale,
        "expired_memory_ids": list(expired.expired_memory_ids),
        "checks": {
            "ruler_has_dynamic_range": contradiction and corrected_ok,
        },
        "gates": {
            "default_contradiction_persists": contradiction,
            "corrected_top1_is_current": corrected_ok,
            "stale_not_reachable_after_correction": not (overwrite_stale or expired_stale),
        },
    }


def forgetting_face() -> dict[str, Any]:
    cue = _onehot(0, len(PLACES))

    def retention(capacity: int, interference: int) -> tuple[float, int, EpisodicMemoryStore]:
        store = EpisodicMemoryStore(capacity=capacity, cue_dim=len(PLACES))
        targets = [f"target-{i}" for i in range(N_TARGETS)]
        for index, memory_id in enumerate(targets):
            store.write(_record(memory_id, "sessionA", index, cue, place=PLACES[0]))
        for index in range(interference):
            store.write(_record(f"inter-{index}", "sessionB", index, cue, place=PLACES[1]))
        present = {hit.record.memory_id for hit in store.retrieve(cue, limit=capacity)}
        kept = sum(int(memory_id in present) for memory_id in targets)
        return kept / N_TARGETS, store.evicted_count, store

    large = {n: retention(LARGE_CAPACITY, n) for n in INTERFERENCE}
    large_rates = [large[n][0] for n in INTERFERENCE]
    small_before = retention(SMALL_CAPACITY, 0)
    small_after = retention(SMALL_CAPACITY, N_TARGETS)
    flat = max(large_rates) == min(large_rates)
    evicted_exact = small_after[1] == N_TARGETS
    drops = small_after[0] < small_before[0]
    return {
        "large_capacity_retention": {str(n): large[n][0] for n in INTERFERENCE},
        "large_capacity_evicted": {str(n): large[n][1] for n in INTERFERENCE},
        "retention_flat": flat,
        "small_capacity": SMALL_CAPACITY,
        "small_before_retention": small_before[0],
        "small_after_retention": small_after[0],
        "small_after_evicted_count": small_after[1],
        "expected_evicted": N_TARGETS,
        "store_has_decay_attribute": hasattr(small_after[2], "decay"),
        "checks": {
            "evicted_count_exact": evicted_exact,
            "store_has_no_decay": not hasattr(small_after[2], "decay"),
            "ruler_has_dynamic_range": flat and drops,
        },
        "gates": {
            "retention_flat_without_decay": flat,
            "retention_drops_under_capacity": drops,
        },
    }


def session_face() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(STORE_SEED + 2)
    store = EpisodicMemoryStore(capacity=LARGE_CAPACITY, cue_dim=len(PLACES))
    cues: dict[str, torch.Tensor] = {}
    for session in ("sessionA", "sessionB"):
        for index in range(N_SESSION_RECORDS):
            place_index = index % len(PLACES)
            cue = _onehot(place_index, len(PLACES)) + 0.05 * torch.randn(
                len(PLACES), generator=generator, dtype=torch.float64
            )
            memory_id = f"{session}-{index}"
            store.write(_record(memory_id, session, index, cue, place=PLACES[place_index]))
            if session == "sessionA":
                cues[memory_id] = cue
    payload = store.checkpoint()
    restored = EpisodicMemoryStore.from_checkpoint(payload)
    roundtrip = value_digest({r.memory_id: r.to_payload() for r in store.records}) == value_digest(
        {r.memory_id: r.to_payload() for r in restored.records}
    )
    correct = 0
    for memory_id, cue in cues.items():
        hits = restored.retrieve(cue, limit=1, episode_id="sessionA")
        correct += int(bool(hits) and hits[0].record.memory_id == memory_id)
    retention_rate = correct / N_SESSION_RECORDS
    leak_hits = restored.retrieve(next(iter(cues.values())), limit=LARGE_CAPACITY)
    leakage = any(hit.record.episode_id == "sessionB" for hit in leak_hits)
    gated = restored.retrieve(
        next(iter(cues.values())), limit=LARGE_CAPACITY, episode_id="sessionA"
    )
    exclusion = sum(int(hit.record.episode_id == "sessionA") for hit in gated) / len(gated)
    return {
        "checkpoint_roundtrip_exact": roundtrip,
        "checkpoint_format": payload["format"],
        "cross_session_retention": retention_rate,
        "session_queries": N_SESSION_RECORDS,
        "gate_off_leakage": leakage,
        "gate_off_hit_count": len(leak_hits),
        "gate_on_exclusion_rate": exclusion,
        "gate_on_hit_count": len(gated),
        "checks": {
            "checkpoint_roundtrip_exact": roundtrip,
            "gate_on_exclusion_exact": exclusion == GATE_ON_EXCLUSION_EXACT,
            "ruler_has_dynamic_range": leakage and exclusion == GATE_ON_EXCLUSION_EXACT,
        },
        "gates": {
            "cross_session_retention_reaches_gate": retention_rate >= CROSS_SESSION_RETENTION_MIN,
            "gate_off_leaks": leakage,
        },
    }


STORE_FACES = {
    "S1_place": place_face,
    "S2_confidence": confidence_face,
    "S3_conflict": conflict_face,
    "S4_forgetting": forgetting_face,
    "S5_session": session_face,
}


def run_store_faces() -> dict[str, Any]:
    return {name: face() for name, face in STORE_FACES.items()}


def global_checks(
    dataset: dict[str, list[dict[str, Any]]], anchor: dict[str, Any]
) -> dict[str, Any]:
    total = sum(len(items) for items in dataset.values())
    oracle = sum(
        int(dms.label(item) == int(any(a == item["sample"] for a in item["test"])))
        for split in dataset
        for item in dataset[split]
    )
    store = EpisodicMemoryStore(capacity=4, cue_dim=len(PLACES))
    record = _record("roundtrip", "sessionA", 0, _onehot(0, len(PLACES)), place=PLACES[0])
    store.write(record)
    restored = EpisodicMemoryStore.from_checkpoint(store.checkpoint())
    return {
        "encoding_oracle": oracle == total,
        "encoding_oracle_hits": oracle,
        "encoding_oracle_total": total,
        "partition_disjoint": not (
            {(item["sample"], item["position"]) for item in dataset["train"]}
            & {(item["sample"], item["position"]) for item in dataset["fit"]}
        ),
        "multi_head_anchor_exact": anchor["exact"],
        "store_record_roundtrip": value_digest({r.memory_id: r.to_payload() for r in store.records})
        == value_digest({r.memory_id: r.to_payload() for r in restored.records}),
    }


def discipline() -> dict[str, Any]:
    return {
        "graph_optimizer_steps": 0,
        "weight_writes": 0,
        "head_optimizer": "LBFGS on diagnostic heads only, never on the graph",
        "check_reads": 0,
        "final_reads": 0,
        "confirm_reads": 0,
        "sealed_partitions_read": [],
        "project_data_files_opened": 0,
        "dataset_origin": "fully synthetic via probe_u2_w_dms_20261117.make_dataset",
    }


def classify(
    budget_stopped: bool, uncovered: list[str], seeds: list[dict[str, Any]], store: dict[str, Any]
) -> str:
    if budget_stopped:
        return "budget_stop"
    if uncovered:
        return "instrument_invalid"
    if any(row["bridge"]["triple_fit"] < TRIPLE_FIT_MIN for row in seeds):
        return "task_not_learnable"
    store_ranged = all(row["checks"]["ruler_has_dynamic_range"] for row in store.values())
    if len(seeds) == len(GRAPH_SEEDS) and store_ranged:
        return "episodic_associative_feasible"
    return "instrument_invalid"


def collect_uncovered(
    globals_: dict[str, Any], seeds: list[dict[str, Any]], store: dict[str, Any]
) -> list[str]:
    uncovered = [f"global:{key}" for key, value in globals_.items() if value is False]
    for row in seeds:
        uncovered.extend(
            f"seed{row['seed']}:{key}"
            for key, value in row["bridge"]["checks"].items()
            if not value
        )
        uncovered.extend(
            f"seed{row['seed']}:{key}" for key, value in row["checks"].items() if not value
        )
    for name, row in store.items():
        uncovered.extend(f"{name}:{key}" for key, value in row["checks"].items() if not value)
    return uncovered


def measure(seeds: tuple[int, ...], cap_cpu: float, cap_wall: float) -> dict[str, Any]:
    started_cpu, started_wall = time.process_time(), time.monotonic()
    dataset = wprobe.make_dataset()
    anchor = head_anchor(dataset)
    globals_ = global_checks(dataset, anchor)
    rows: list[dict[str, Any]] = []
    budget_stopped = False
    for seed in seeds:
        if (
            time.process_time() - started_cpu >= cap_cpu - RESERVE_CPU
            or time.monotonic() - started_wall >= cap_wall - RESERVE_WALL
        ):
            budget_stopped = True
            break
        graph = RouteBGraph(UnitGraphSpec(dtype="float64", seed=seed))
        before = value_digest({n: t.detach().clone() for n, t in graph.state_dict().items()})
        bridge = bridge_face(graph, dataset)
        after = value_digest({n: t.detach().clone() for n, t in graph.state_dict().items()})
        rows.append(
            {
                "seed": seed,
                "bridge": bridge,
                "checks": {"params_unchanged": before == after},
            }
        )
    store = run_store_faces()
    uncovered = collect_uncovered(globals_, rows, store)
    verdict = classify(budget_stopped, uncovered, rows, store)
    return {
        "dataset": dataset,
        "anchor": anchor,
        "globals": globals_,
        "seeds": rows,
        "store": store,
        "uncovered": uncovered,
        "budget_stopped": budget_stopped,
        "verdict": verdict,
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }


def base_payload(mode: str, measured: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "taiji-u2-x-feasibility-v1",
        "card": CARD,
        "phase": mode,
        "prereg": PREREG,
        "criteria": criteria(),
        "criteria_digest": value_digest(criteria()),
        "encoding_contract": dms.encoding_contract(),
        "data_digest": wprobe.dataset_digest(measured["dataset"]),
        "dataset_sizes": {split: len(items) for split, items in measured["dataset"].items()},
        "head_anchor": measured["anchor"],
        "global_checks": measured["globals"],
        "seeds": measured["seeds"],
        "store_faces": measured["store"],
        "uncovered": measured["uncovered"],
        "budget_stopped": measured["budget_stopped"],
        "verdict": measured["verdict"],
        "discipline": discipline(),
        "cost": measured["cost"],
        "scope_note": (
            "feasible means only: the current unit representation carries the event triple "
            "and the five store gaps are precisely localized (including absence by construction). "
            "It does not mean episodic/associative memory works, nor infinite context."
        ),
        "prior_readings_not_remeasured": [
            "reports/taiji_m5_episodic_field_20260821.json",
            "reports/taiji_p4_episodic_recall_baseline_20260825.json",
            "reports/taiji_g5_episodic_write_four_grid_v3_20260926.json",
            "reports/taiji_episodic_read_sharing_face_20261011.json",
        ],
    }


def price() -> int:
    torch.set_num_threads(1)
    started_cpu, started_wall = time.process_time(), time.monotonic()
    dataset = wprobe.make_dataset()
    graph = RouteBGraph(UnitGraphSpec(dtype="float64", seed=GRAPH_SEEDS[0]))
    feature_samples = []
    for _ in range(PRICE_REPEATS):
        cpu0, wall0 = time.process_time(), time.monotonic()
        graph_features(graph, dataset, STD_DELAY)
        feature_samples.append((time.process_time() - cpu0, time.monotonic() - wall0))
    train_x, fit_x = graph_features(graph, dataset, STD_DELAY)
    train_y = [ep["sample"] for ep in dataset["train"]]
    fit_y = [ep["sample"] for ep in dataset["fit"]]
    head_samples = []
    for _ in range(PRICE_REPEATS):
        cpu0, wall0 = time.process_time(), time.monotonic()
        fit_multi(train_x, train_y, fit_x, fit_y, 4)
        head_samples.append((time.process_time() - cpu0, time.monotonic() - wall0))
    cpu0, wall0 = time.process_time(), time.monotonic()
    run_store_faces()
    store_cpu, store_wall = time.process_time() - cpu0, time.monotonic() - wall0
    feature_stats = timing_stats(feature_samples)
    head_stats = timing_stats(head_samples)
    per_store_face = store_cpu / len(STORE_FACES)
    heads_per_seed = 9
    features_per_seed = len(DELAYS)

    def basis(feature_cpu: float, head_cpu: float) -> float:
        return (
            len(GRAPH_SEEDS) * (features_per_seed * feature_cpu + heads_per_seed * head_cpu)
            + len(STORE_FACES) * per_store_face
        )

    median_basis = basis(feature_stats["cpu_median"], head_stats["cpu_median"])
    mean_basis = basis(feature_stats["cpu_mean"], head_stats["cpu_mean"])
    max_basis = basis(feature_stats["cpu_max"], head_stats["cpu_max"])
    measured_cpu = mean_basis
    cap_cpu = min(measured_cpu * CAP_MARGIN, DESIGN_CAP_CPU)
    cap_wall = min(cap_cpu * 2.0, DESIGN_CAP_WALL)
    payload = {
        "schema": "taiji-u2-x-feasibility-price-v1",
        "card": CARD,
        "phase": "price",
        "prereg": PREREG,
        "criteria_digest": value_digest(criteria()),
        "timing": "batched: one features call covers 96 episodes; one head fit is 300 LBFGS iterations",
        "repeats": PRICE_REPEATS,
        "aggregation": "cap basis = mean over repeats; median and max disclosed alongside",
        "cap_basis_note": (
            "a single contention outlier can make the max basis non-predictive (the first "
            "pricing pass measured head cpu_max at 32x cpu_median), so the cap is set from the "
            "mean basis, which already carries the outlier's contribution; the live ratio is "
            "disclosed in head_outlier_ratio_max_over_median and the max basis is not adopted"
        ),
        "features_timing": feature_stats,
        "head_timing": head_stats,
        "head_outlier_ratio_max_over_median": round(
            head_stats["cpu_max"] / head_stats["cpu_median"], 4
        ),
        "median_basis_cpu": round(median_basis, 4),
        "mean_basis_cpu": round(mean_basis, 4),
        "max_basis_cpu": round(max_basis, 4),
        "per_features_call_cpu": feature_stats["cpu_mean"],
        "per_head_fit_cpu": head_stats["cpu_mean"],
        "per_store_face_cpu": round(per_store_face, 8),
        "store_faces_cpu": round(store_cpu, 4),
        "store_faces_wall": round(store_wall, 4),
        "features_per_seed": features_per_seed,
        "heads_per_seed": heads_per_seed,
        "seeds": len(GRAPH_SEEDS),
        "measured_cpu": round(measured_cpu, 4),
        "cap_margin": CAP_MARGIN,
        "cap_cpu": round(cap_cpu, 4),
        "cap_wall": round(cap_wall, 4),
        "design_cap_cpu": DESIGN_CAP_CPU,
        "design_cap_wall": DESIGN_CAP_WALL,
        "reserve_cpu": RESERVE_CPU,
        "reserve_wall": RESERVE_WALL,
        "low_tick_limit": (
            "single-call timing below the ~0.0156 s process_time resolution is a tick lottery, "
            "not a measurement; every timed unit here is a batch"
        ),
        "discipline": discipline(),
        "cost": {
            "cpu": round(time.process_time() - started_cpu, 4),
            "wall": round(time.monotonic() - started_wall, 4),
        },
    }
    u1.write_report(PRICE_REPORT, payload)
    print(
        f"PRICE measured_cpu={payload['measured_cpu']} cap_cpu={payload['cap_cpu']} "
        f"cap_wall={payload['cap_wall']} per_features={payload['per_features_call_cpu']} "
        f"per_head={payload['per_head_fit_cpu']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--price", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.price:
        return price()
    if args.smoke:
        measured = measure(GRAPH_SEEDS[:1], DESIGN_CAP_CPU, DESIGN_CAP_WALL)
        payload = base_payload("smoke", measured)
        payload["verdict"] = "smoke_passed" if not measured["uncovered"] else "smoke_failed"
        u1.write_report(SMOKE_REPORT, payload)
        print(
            f"SMOKE verdict={payload['verdict']} uncovered={len(measured['uncovered'])} "
            f"triple={[round(r['bridge']['triple_fit'], 4) for r in measured['seeds']]} "
            f"cpu={payload['cost']['cpu']}"
        )
        return 0 if payload["verdict"] == "smoke_passed" else 1
    if not PRICE_REPORT.is_file():
        raise SystemExit("price_report_missing: run --price before --run")
    plan = json.loads(PRICE_REPORT.read_text(encoding="utf-8"))
    measured = measure(GRAPH_SEEDS, float(plan["cap_cpu"]), float(plan["cap_wall"]))
    payload = base_payload("run", measured)
    payload["caps"] = {"cap_cpu": plan["cap_cpu"], "cap_wall": plan["cap_wall"]}
    payload["price_report"] = "reports/taiji_u2_x_feasibility_price_20261117.json"
    u1.write_report(RUN_REPORT, payload)
    print(
        f"X verdict={payload['verdict']} uncovered={len(measured['uncovered'])} "
        f"triple={[round(r['bridge']['triple_fit'], 4) for r in measured['seeds']]} "
        f"cpu={payload['cost']['cpu']} wall={payload['cost']['wall']}"
    )
    return 0 if payload["verdict"] == "episodic_associative_feasible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
