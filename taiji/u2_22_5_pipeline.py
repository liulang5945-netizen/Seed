"""Unified 22.5.0 -> 22.5.1 training, online-correction & verification pipeline.

Connects the 22.5.1 joint lineage (S0 architecture revision: ``d=32``,
``readout="pixel-salience-v2"``, ``record_metadata=True``,
``separate_requests=True``) to the correction-and-retention evidence via
``model.record_organ_candidate()`` with zero monkey-patching, zero external
heads, and bit-exact parent freezing.  The frozen 22.5.0 v1 A parent
(``readout="pixel-salience"``) keeps its own identity and is rebuilt/loaded
separately (S1 ``R-REBUILD-1``); these helpers never mix the two.

All evaluation helpers operate strictly on caller-supplied splits (e.g.
``train`` / ``fit``) and never read sealed ``check`` or ``final`` partitions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, cast

import torch

from .u2_events import observation_intent
from .u2_history import value_digest
from .u2_subject import U2Config, U2Packet, U2Record, U2Subject

LABELS = ("左边", "中间", "右边", "未知")
POSITION_LABELS = LABELS[:3]
SUFFIX = "现在在哪里？"
DEFAULT_TEMPLATES = (
    "通知：{entity}改为{label}。",
    "最新记录：{entity}现在在{label}。",
    "更正消息：现在{label}放着{entity}。",
)


@dataclass(frozen=True)
class ATrainConfig:
    seed: int = 20261110
    d: int = 32
    updates: int = 800
    batch_size: int = 8
    lr: float = 4e-3
    readout_lr_mult: float = 2.5
    min_lr: float = 4e-4
    warmup_steps: int = 60
    plateau_frac: float = 0.6
    first_byte_weight: float = 2.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class BOrganTrainConfig:
    seed: int = 20261110
    steps: int = 300
    batch_size: int = 4
    lr: float = 1.5e-2
    min_lr: float = 1e-3
    grad_clip: float = 1.0
    route_loss_weight: float = 1.0
    span_aux_weight: float = 0.5


def build_a_subject(
    seed: int = 20261110, d: int = 32, readout: str = "pixel-salience-v2"
) -> U2Subject:
    """Build the S0 A subject (default: v2 readout + separate requests)."""
    config = replace(U2Config.small(), seed=seed, readout=readout, d=d)
    return U2Subject(config).metadata_candidate().request_candidate()


def make_record(raw: bytes, record_id: str, version: int = 0, observed_at: float = 0.0) -> U2Record:
    return U2Record(
        record_id=record_id,
        raw=raw,
        version=version,
        source="sensor",
        scope="world",
        observed_at=observed_at,
        writable=True,
    )


def prepare_episode(
    model: U2Subject,
    public: dict[str, Any],
    identity: str,
    records: list[U2Record] | None = None,
) -> dict[str, Any]:
    """Assimilate a public scene, attach optional records, bind request, begin utterance."""
    state = model.set_goal(model.initial(identity), public["goal"])
    state = observation_intent(model, state, "recorded-observation", 0.05)
    packet = U2Packet(
        "recorded-public-frame",
        "sensor",
        "world",
        public["t_start"],
        public["t_end"],
        0.0,
        public["observations"],
        public["coordinates"],
        public["outcome"],
        "recorded-observation",
    )
    state, _ = model.assimilate(state, packet)
    if records:
        state = model.set_records(state, records)
    state = model.set_request(state, "current-request", public["request"])
    return cast(dict[str, Any], model.begin_utterance(state))


def byte_sequence_loss(
    model: U2Subject,
    public: dict[str, Any],
    identity: str,
    target: bytes,
    records: list[U2Record] | None = None,
    first_weight: float = 2.0,
) -> torch.Tensor:
    """Teacher-forced byte-level cross-entropy directly through ``U2Subject.output``."""
    state = prepare_episode(model, public, identity, records=records)
    goal = model.language_goal(state)
    action = state["prediction_action"]
    terms = []
    for idx, symbol in enumerate([*target, 256]):
        logp, _, details = model.output(state, goal, action)
        weight = first_weight if idx == 0 else 1.0
        terms.append(-logp[symbol] * weight)
        state = model.accept_byte(state, symbol, details)
    return torch.stack(terms).mean()


def decode_episode(
    model: U2Subject,
    public: dict[str, Any],
    identity: str,
    records: list[U2Record] | None = None,
    mask_first_eos: bool = True,
) -> tuple[bytes, bool]:
    """Greedy UTF-8-safe decoding through ``U2Subject.output``."""
    state = prepare_episode(model, public, identity, records=records)
    goal = model.language_goal(state)
    action = state["prediction_action"]
    data = bytearray()
    for _ in range(65):
        logp, _, details = model.output(state, goal, action)
        if mask_first_eos and not data:
            logp = logp.clone()
            logp[256] = -torch.inf
        symbol = int(logp.detach().argmax())
        state = model.accept_byte(state, symbol, details)
        if symbol == 256:
            return bytes(data), True
        data.append(symbol)
    return bytes(data), False


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    plateau_frac: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    plateau_end = max(warmup_steps, int(total_steps * plateau_frac))
    if step < plateau_end:
        return base_lr
    denom = max(total_steps - plateau_end, 1)
    progress = min(max((step - plateau_end) / float(denom), 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate_task_a(model: U2Subject, examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate Task A (record-free) accuracy and per-label breakdown."""
    correct = 0
    per_label: dict[str, list[int]] = {label: [0, 0] for label in LABELS}
    rows = []
    with torch.no_grad():
        for ex in examples:
            content, eos = decode_episode(model, ex["public"], ex["id"])
            ok = bool(eos and content == ex["target"])
            correct += int(ok)
            truth = ex["target"].decode("utf-8")
            if truth in per_label:
                per_label[truth][0] += int(ok)
                per_label[truth][1] += 1
            rows.append(
                {"id": ex["id"], "output": content.decode("utf-8", "replace"), "correct": ok}
            )
    return {
        "correct": correct,
        "denominator": len(examples),
        "accuracy": correct / max(len(examples), 1),
        "per_label": {k: f"{v[0]}/{v[1]}" for k, v in per_label.items()},
        "rows": rows,
    }


def train_task_a(
    train_examples: list[dict[str, Any]],
    cfg: ATrainConfig = ATrainConfig(),
    fit_examples: list[dict[str, Any]] | None = None,
    eval_every: int = 0,
) -> tuple[U2Subject, list[dict[str, Any]]]:
    """Train a 22.5.0 A-gate subject with separate readout/base LR & grad clipping."""
    torch.manual_seed(cfg.seed)
    model = build_a_subject(seed=cfg.seed, d=cfg.d)
    readout_params = [
        p for n, p in model.named_parameters() if n.startswith(("pixel_readout.", "delta_head."))
    ]
    base_params = [
        p
        for n, p in model.named_parameters()
        if not n.startswith(("pixel_readout.", "delta_head."))
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": base_params, "lr": cfg.lr, "weight_decay": 0.0},
            {"params": readout_params, "lr": cfg.lr * cfg.readout_lr_mult, "weight_decay": 1e-4},
        ],
        betas=(0.9, 0.999),
    )
    rng = random.Random(cfg.seed + 9)
    n_train = len(train_examples)
    order = list(range(n_train))
    rng.shuffle(order)
    cursor = 0
    history: list[dict[str, Any]] = []

    for step in range(cfg.updates):
        lr = cosine_warmup_lr(
            step, cfg.updates, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.plateau_frac
        )
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * cfg.readout_lr_mult
        batch = []
        for _ in range(cfg.batch_size):
            if cursor >= n_train:
                rng.shuffle(order)
                cursor = 0
            batch.append(train_examples[order[cursor]])
            cursor += 1
        optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [
                byte_sequence_loss(
                    model,
                    ex["public"],
                    ex["id"],
                    ex["target"],
                    first_weight=cfg.first_byte_weight,
                )
                for ex in batch
            ]
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(base_params, cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(readout_params, cfg.grad_clip)
        optimizer.step()

        if fit_examples and eval_every > 0 and (step + 1) % eval_every == 0:
            reading = evaluate_task_a(model, fit_examples)
            history.append(
                {
                    "step": step + 1,
                    "loss": round(float(loss.detach()), 4),
                    "lr": round(lr, 6),
                    "fit_accuracy": round(reading["accuracy"], 4),
                    "per_label": reading["per_label"],
                }
            )
    return model, history


def build_b_items(
    examples: list[dict[str, Any]], templates: tuple[str, ...] = DEFAULT_TEMPLATES
) -> list[dict[str, Any]]:
    """Construct Task B revised-fact items from Task A examples (all non-unknown labels)."""
    items = []
    for ex in examples:
        truth = ex["target"].decode("utf-8")
        if truth not in POSITION_LABELS:
            continue
        idx = POSITION_LABELS.index(truth)
        revised = POSITION_LABELS[(idx + 1) % 3]
        counter = POSITION_LABELS[(idx + 2) % 3]
        req_text = ex["public"]["request"].decode("utf-8")
        entity = req_text[: -len(SUFFIX)] if req_text.endswith(SUFFIX) else req_text
        records = [t.format(entity=entity, label=revised).encode("utf-8") for t in templates]
        counter_records = [
            t.format(entity=entity, label=counter).encode("utf-8") for t in templates
        ]
        items.append(
            {
                "example": ex,
                "entity": entity,
                "truth": truth,
                "revised": revised,
                "counter": counter,
                "target": revised.encode("utf-8"),
                "counter_target": counter.encode("utf-8"),
                "records": records,
                "counter_records": counter_records,
            }
        )
    return items


def train_record_organ_on_parent(
    parent: U2Subject,
    train_examples: list[dict[str, Any]],
    cfg: BOrganTrainConfig = BOrganTrainConfig(),
    templates: tuple[str, ...] = DEFAULT_TEMPLATES,
) -> tuple[U2Subject, dict[str, Any]]:
    """Fork ``parent`` via ``record_organ_candidate()`` and train ONLY ``record_organ``.

    Trains end-to-end through ``U2Subject.output()`` on a balanced mix of:
      1. single relevant correction records (Task B),
      2. single irrelevant distractor records (Task D backoff -> parent A answer),
      3. multi-record scenes (1 relevant + 1-2 distractors in arbitrary order),
      4. superseding records (older fact + newer correction for the same entity).

    All inherited parent parameters remain frozen and verified bit-identical.
    """
    model = parent.record_organ_candidate()
    parent_keys = {k for k in model.state_dict() if not k.startswith("record_organ.")}
    before_digest = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})

    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("record_organ."))

    train_items = build_b_items(train_examples, templates=templates)
    # Precompute frozen parent byte log-probabilities & UTF-8 masks under no_grad:
    # parent weights are frozen, so computing these once speeds up organ training
    # by ~50x while keeping the exact differentiable mix_log_probabilities loss.
    from .u2_operators import utf8_mask
    from .u2_record_organ import mix_log_probabilities

    parent_cache: dict[tuple[str, bytes], list[tuple[torch.Tensor, torch.Tensor]]] = {}
    with torch.no_grad():
        for item in train_items:
            ex = item["example"]
            for tgt in (ex["target"], item["target"]):
                key = (ex["id"], tgt)
                if key in parent_cache:
                    continue
                st = prepare_episode(parent, ex["public"], ex["id"], records=None)
                goal = parent.language_goal(st)
                action = st["prediction_action"]
                seq_rows = []
                for sym in [*tgt, 256]:
                    mask = utf8_mask(
                        st["utf8_remaining"],
                        st["utf8_lead"],
                        parent.c.max_output_bytes - st["data_bytes"],
                        device=parent.native_w.device,
                    )
                    logp, _, details = parent.output(st, goal, action)
                    seq_rows.append((logp.detach(), mask))
                    st = parent.accept_byte(st, sym, details)
                parent_cache[key] = seq_rows

    gate_scalars = [
        p
        for n, p in model.record_organ.named_parameters()
        if n in {"organ_mix", "recency_logit"} or n.startswith("route_head.")
    ]
    organ_other = [
        p
        for n, p in model.record_organ.named_parameters()
        if not (n in {"organ_mix", "recency_logit"} or n.startswith("route_head."))
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": organ_other, "lr": cfg.lr, "weight_decay": 1e-4},
            {"params": gate_scalars, "lr": cfg.lr * 3.0, "weight_decay": 0.0},
        ]
    )
    rng = random.Random(cfg.seed + 31)
    bce = torch.nn.functional.binary_cross_entropy

    for step in range(cfg.steps):
        lr = cosine_warmup_lr(step, cfg.steps, cfg.lr, cfg.min_lr, warmup_steps=20)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 3.0

        losses = []
        for _ in range(cfg.batch_size):
            item = rng.choice(train_items)
            other = rng.choice(train_items)
            while other["entity"] == item["entity"]:
                other = rng.choice(train_items)
            other2 = rng.choice(train_items)
            while other2["entity"] == item["entity"]:
                other2 = rng.choice(train_items)
            t_idx = rng.randrange(len(templates))
            mode = rng.randrange(6)
            ex = item["example"]

            def _distractor_rec(ent_item: dict[str, Any], idx: int, tag: str) -> Any:
                raw = (
                    templates[idx]
                    .format(entity=ent_item["entity"], label=ent_item["revised"])
                    .encode("utf-8")
                )
                return make_record(raw, tag, version=1)

            if mode == 0:
                # Single relevant correction record (Task B).
                recs = [make_record(item["records"][t_idx], f"b-{step}", version=1)]
                target = item["target"]
                should_route = 1.0
            elif mode in (1, 4):
                # Single irrelevant distractor record (Task D -> keep parent A truth).
                recs = [_distractor_rec(other, t_idx, f"d-{mode}-{step}")]
                target = ex["target"]
                should_route = 0.0
            elif mode == 2:
                # Multi-record: 1 relevant + 2 distractors in shuffled order.
                recs = [
                    _distractor_rec(other, rng.randrange(len(templates)), f"m-d1-{step}"),
                    make_record(item["records"][t_idx], f"m-rel-{step}", version=1),
                    _distractor_rec(other2, rng.randrange(len(templates)), f"m-d2-{step}"),
                ]
                rng.shuffle(recs)
                target = item["target"]
                should_route = 1.0
            elif mode == 3:
                # Superseding correction: old record (counter) followed by newer record (revised).
                old_rec = make_record(item["counter_records"][t_idx], f"sup-old-{step}", version=1)
                new_rec = make_record(
                    item["records"][(t_idx + 1) % len(templates)], f"sup-new-{step}", version=2
                )
                recs = [old_rec, new_rec]
                target = item["target"]
                should_route = 1.0
            else:
                # Multi-distractor negative: 2 irrelevant records (Task D -> keep parent A truth).
                recs = [
                    _distractor_rec(other, rng.randrange(len(templates)), f"md1-{step}"),
                    _distractor_rec(other2, rng.randrange(len(templates)), f"md2-{step}"),
                ]
                target = ex["target"]
                should_route = 0.0

            plan = model.record_organ.plan(ex["public"]["request"], recs)
            cached_rows = parent_cache[(ex["id"], target)]
            terms = []
            span_terms = []
            for offset, (sym, (parent_logp, mask)) in enumerate(
                zip([*target, 256], cached_rows, strict=True)
            ):
                route_prob, organ_probs = model.record_organ.emit(plan, offset, mask=mask)
                mixed_logp = mix_log_probabilities(parent_logp, route_prob, organ_probs, mask)
                weight = 2.0 if offset == 0 else 1.0
                terms.append(-mixed_logp[sym] * weight)
                if should_route > 0.5:
                    span_terms.append(-organ_probs[sym].clamp_min(1e-9).log() * weight)
            seq_loss = torch.stack(terms).mean()
            route_target = torch.tensor(should_route, dtype=plan["match_prob"].dtype)
            route_aux = bce(plan["match_prob"].clamp(1e-6, 1.0 - 1e-6), route_target)
            item_loss = seq_loss + cfg.route_loss_weight * route_aux
            if span_terms:
                item_loss = item_loss + cfg.span_aux_weight * torch.stack(span_terms).mean()
            losses.append(item_loss)

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.record_organ.parameters()), cfg.grad_clip)
        optimizer.step()

    after_digest = value_digest({k: v for k, v in model.state_dict().items() if k in parent_keys})
    if before_digest != after_digest:
        raise RuntimeError("parent_weights_modified_during_organ_training")
    return model, {
        "parent_bit_identical": before_digest == after_digest,
        "organ_mix": round(float(model.record_organ.organ_mix.detach()), 4),
    }


def evaluate_22_5_1_suite(
    model: U2Subject,
    parent: U2Subject,
    fit_examples: list[dict[str, Any]],
    templates: tuple[str, ...] = DEFAULT_TEMPLATES,
    with_task_a: bool = True,
    with_controls: bool = True,
    with_recovery: bool = True,
) -> dict[str, Any]:
    """Evaluate full 22.5.1 matrix: A retention, B correction, D distractor, multi-record,
    superseding, causal cut, feedback permutation, and bundle recovery.

    ``with_task_a`` / ``with_controls`` / ``with_recovery`` let a caller measure a
    second template face without re-decoding task A (template independent) or the
    causal controls (measured once on the primary face).
    """
    fit_items = build_b_items(fit_examples, templates=templates)
    n_b = len(fit_items)

    a_parent = evaluate_task_a(parent, fit_examples) if with_task_a else None
    a_child = evaluate_task_a(model, fit_examples) if with_task_a else None

    b_ok = d_ok = multi_ok = super_ok = cut_b_ok = perm_b_ok = frozen_b_ok = 0
    with torch.no_grad():
        for idx, item in enumerate(fit_items):
            ex = item["example"]
            k1 = 1
            while fit_items[(idx + k1) % n_b]["entity"] == item["entity"]:
                k1 += 1
            other = fit_items[(idx + k1) % n_b]
            k2 = k1 + 1
            while fit_items[(idx + k2) % n_b]["entity"] in (item["entity"], other["entity"]):
                k2 += 1
            other2 = fit_items[(idx + k2) % n_b]
            t_idx = idx % len(templates)

            rec_b = make_record(item["records"][t_idx], f"eval-b-{idx}", version=1)
            rec_d = make_record(
                templates[t_idx].format(entity=other["entity"], label=other["revised"]).encode(),
                f"eval-d-{idx}",
                version=1,
            )
            rec_d2 = make_record(
                templates[(t_idx + 1) % len(templates)]
                .format(entity=other2["entity"], label=other2["revised"])
                .encode(),
                f"eval-d2-{idx}",
                version=1,
            )
            rec_old = make_record(item["counter_records"][t_idx], f"eval-old-{idx}", version=1)
            rec_new = make_record(
                item["records"][(t_idx + 1) % len(templates)], f"eval-new-{idx}", version=2
            )

            out_b, eos_b = decode_episode(model, ex["public"], ex["id"], records=[rec_b])
            b_ok += int(eos_b and out_b == item["target"])

            # One distractor decode serves both counts: D-retention reads the
            # parent A truth, the permuted-feedback control reads the record's
            # correction from the same output.
            out_d, eos_d = decode_episode(model, ex["public"], ex["id"], records=[rec_d])
            d_ok += int(eos_d and out_d == ex["target"])
            perm_b_ok += int(eos_d and out_d == item["target"])

            # Relevant record placed in the middle of distractors (not index 0).
            out_m, eos_m = decode_episode(
                model, ex["public"], ex["id"], records=[rec_d, rec_b, rec_d2]
            )
            multi_ok += int(eos_m and out_m == item["target"])

            # Superseding correction (old version 1 vs new version 2).
            out_s, eos_s = decode_episode(model, ex["public"], ex["id"], records=[rec_old, rec_new])
            super_ok += int(eos_s and out_s == item["target"])

            if with_controls:
                # Frozen parent control (no organ).
                out_f, eos_f = decode_episode(parent, ex["public"], ex["id"], records=[rec_b])
                frozen_b_ok += int(eos_f and out_f == item["target"])

        if with_controls:
            # Causal knockout control: cut record_organ.
            model._verification_cuts = frozenset({"record_organ"})
            try:
                for idx, item in enumerate(fit_items):
                    ex = item["example"]
                    rec_b = make_record(
                        item["records"][idx % len(templates)], f"cut-{idx}", version=1
                    )
                    out_c, eos_c = decode_episode(model, ex["public"], ex["id"], records=[rec_b])
                    cut_b_ok += int(eos_c and out_c == item["target"])
            finally:
                model._verification_cuts = frozenset()

    # Cross-session bundle recovery check on a record-carrying state.
    recovery_exact: bool | None = None
    if with_recovery:
        probe_item = fit_items[0]
        probe_rec = make_record(probe_item["records"][0], "recovery-rec", version=1)
        probe_state = prepare_episode(
            model,
            probe_item["example"]["public"],
            probe_item["example"]["id"],
            records=[probe_rec],
        )
        bundle = model.export_bundle(probe_state)
        restored_model, restored_state = U2Subject.from_bundle(bundle)
        with torch.no_grad():
            orig_logp = model.output(
                probe_state, model.language_goal(probe_state), probe_state["prediction_action"]
            )[0]
            rest_logp = restored_model.output(
                restored_state,
                restored_model.language_goal(restored_state),
                restored_state["prediction_action"],
            )[0]
        valid_mask = torch.isfinite(orig_logp) & torch.isfinite(rest_logp)
        recovery_exact = bool(
            torch.equal(torch.isfinite(orig_logp), torch.isfinite(rest_logp))
            and float((orig_logp[valid_mask] - rest_logp[valid_mask]).abs().max()) == 0.0
        )

    return {
        "a_parent": f"{a_parent['correct']}/{a_parent['denominator']}" if a_parent else None,
        "a_child": f"{a_child['correct']}/{a_child['denominator']}" if a_child else None,
        "a_retention_drop": (
            round(a_parent["accuracy"] - a_child["accuracy"], 4)
            if a_parent is not None and a_child is not None
            else None
        ),
        "b_correction": f"{b_ok}/{n_b}",
        "b_accuracy": round(b_ok / max(n_b, 1), 4),
        "d_distractor_retention": f"{d_ok}/{n_b}",
        "d_accuracy": round(d_ok / max(n_b, 1), 4),
        "multi_record_b": f"{multi_ok}/{n_b}",
        "superseding_b": f"{super_ok}/{n_b}",
        "controls": (
            {
                "frozen_parent_b": f"{frozen_b_ok}/{n_b}",
                "organ_cut_b": f"{cut_b_ok}/{n_b}",
                "permuted_feedback_b": f"{perm_b_ok}/{n_b}",
            }
            if with_controls
            else None
        ),
        "recovery_exact": recovery_exact,
    }
