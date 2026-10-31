"""§10.3 bounded causal replay migration on the R021 reference graph.

The career journal is the single source of truth: ``run_career`` both produces the
original state and executes replays, so journal completeness is mechanically
checkable. Replay never updates parameters, claims rewards or executes external
actions; a budget refusal keeps the parent revision. No trainer and no task data.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from taiji.living_core import (
    EOS,
    LivingCoreConfig,
    LivingCoreReference,
    LivingState,
    RawSource,
)


@dataclass(frozen=True)
class CareerJournal:
    """Declared synthetic career: raw events plus the shared step context."""

    subject_id: str
    prompt: bytes
    target: bytes
    goal: bytes
    records: tuple[RawSource, ...]
    raw_nu: tuple[float, ...]
    known_nu: tuple[bool, ...]
    mode: int = 0
    now: int = 0
    allowed_write: bool = True

    def __post_init__(self) -> None:
        if not self.subject_id or not self.prompt or not self.target:
            raise ValueError("Career journal requires subject, prompt and target")
        if len(self.raw_nu) != 8 or len(self.known_nu) != 8:
            raise ValueError("Career journal requires eight nu readings")

    @property
    def event_count(self) -> int:
        # One microstep per prompt patch plus one prediction round per target byte
        # and the terminal EOS round; patches depend on the encoder patching rule.
        patches = (len(self.prompt) + 7) // 8
        return patches + len(self.target) + 1

    @property
    def journal_bytes(self) -> int:
        return (
            len(self.prompt)
            + len(self.target)
            + len(self.goal)
            + sum(len(record.raw) for record in self.records)
        )


@dataclass(frozen=True)
class MigrationBudget:
    max_events: int | None = None
    max_journal_bytes: int | None = None
    max_cpu_seconds: float | None = None
    max_wall_seconds: float | None = None


def _shared_kwargs(model: LivingCoreReference, journal: CareerJournal) -> dict:
    dtype = model.W.dtype
    return {
        "goal": journal.goal,
        "records": journal.records,
        "raw_nu": torch.tensor(journal.raw_nu, dtype=dtype),
        "known_nu": torch.tensor(journal.known_nu),
        "mode": journal.mode,
        "now": journal.now,
        "allowed": torch.full(
            (model.config.groups, model.config.fast_heads),
            journal.allowed_write,
            dtype=torch.bool,
        ),
    }


def career_step_specs(journal: CareerJournal) -> list[dict]:
    """Explicit step sequence: prompt patches, then one round per target byte + EOS."""
    specs: list[dict] = []
    for start in range(0, len(journal.prompt), 8):
        specs.append({"kind": "patch", "event_raw": journal.prompt[start : start + 8]})
    previous = EOS
    for byte in (*journal.target, EOS):
        specs.append(
            {
                "kind": "round",
                "event_raw": None if previous == EOS else bytes([previous]),
                "previous_byte": previous,
                "round_byte": byte,
            }
        )
        previous = byte
    return specs


def resume_career(
    model: LivingCoreReference,
    state: LivingState,
    journal: CareerJournal,
    start: int = 0,
    growth_gate_provider=None,
    limit: int | None = None,
):
    """Consume specs[start:stop] from the given state; returns (state, [(output, byte)])."""
    kwargs = _shared_kwargs(model, journal)
    specs = career_step_specs(journal)
    stop = len(specs) if limit is None else min(len(specs), start + limit)
    steps: list[tuple] = []
    for index in range(start, stop):
        spec = specs[index]
        step_kwargs = dict(kwargs)
        if growth_gate_provider is not None:
            step_kwargs["growth_gates"] = growth_gate_provider(index)
        if spec["kind"] == "patch":
            event = model.encoder.patches(spec["event_raw"])[0]
            output = model.step(state, event=event, **step_kwargs)
            steps.append((output, None))
            state = output.state
        else:
            if spec["event_raw"] is None:
                event = model.W.new_zeros(model.config.width)
            else:
                event = model.encoder.patches(spec["event_raw"])[0]
            output = model.step(
                state, event=event, previous_byte=spec["previous_byte"], **step_kwargs
            )
            steps.append((output, spec["round_byte"]))
            state = model.consume_output(output, spec["round_byte"])
    return state, steps


def run_career(model: LivingCoreReference, journal: CareerJournal, growth_gate_provider=None):
    """Consume the journal in visible order; returns (state, [(output, round_byte)]).

    ``growth_gate_provider`` optionally maps the career step index to the declared
    per-step branch gates consumed by gated newborn units (v1.2 candidate)."""
    state = model.initial_state(journal.subject_id)
    return resume_career(model, state, journal, 0, growth_gate_provider)


def journal_from_dict(data: dict) -> CareerJournal:
    return CareerJournal(
        subject_id=data["subject_id"],
        prompt=bytes(data["prompt"]),
        target=bytes(data["target"]),
        goal=bytes(data["goal"]),
        records=tuple(
            RawSource(
                source_id=record["source_id"],
                raw=bytes(record["raw"]),
                observed_at=record["observed_at"],
                version=record["version"],
                owner=record.get("owner", "owner"),
                scope=record.get("scope", "scope"),
                real=record.get("real", True),
            )
            for record in data["records"]
        ),
        raw_nu=tuple(data["raw_nu"]),
        known_nu=tuple(data["known_nu"]),
        mode=data.get("mode", 0),
        now=data.get("now", 0),
        allowed_write=data.get("allowed_write", True),
    )


def journal_to_dict(journal: CareerJournal) -> dict:
    return {
        "subject_id": journal.subject_id,
        "prompt": list(journal.prompt),
        "target": list(journal.target),
        "goal": list(journal.goal),
        "records": [
            {
                "source_id": record.source_id,
                "raw": list(record.raw),
                "observed_at": record.observed_at,
                "version": record.version,
                "owner": record.owner,
                "scope": record.scope,
                "real": record.real,
            }
            for record in journal.records
        ],
        "raw_nu": list(journal.raw_nu),
        "known_nu": list(journal.known_nu),
        "mode": journal.mode,
        "now": journal.now,
        "allowed_write": journal.allowed_write,
    }


BUNDLE_SCHEMA = "taiji-subject-bundle-r024-v1"


def save_subject(
    path,
    model: LivingCoreReference,
    state: LivingState,
    *,
    career: CareerJournal,
    journal_cursor: int = 0,
    optimizer=None,
    eligibility=None,
    journal_entries=None,
    pending=None,
    growth_gates=None,
    extra=None,
) -> str:
    """Write the declared subject bundle; returns the file sha256."""
    from taiji.growth_credit import GatedGrowthCore

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "subject_id": state.subject_id,
        "revision": state.revision,
        "step": state.step,
        "config": {
            "groups": model.config.groups,
            "width": model.config.width,
            "units": model.config.units,
            "adapter_rank": model.config.adapter_rank,
            "fast_heads": model.config.fast_heads,
            "key_width": model.config.key_width,
            "value_width": model.config.value_width,
            "fast_limit": model.config.fast_limit,
            "max_event_bytes": model.config.max_event_bytes,
            "max_source_bytes": model.config.max_source_bytes,
            "max_target_bytes": model.config.max_target_bytes,
            "seed": model.config.seed,
        },
        "growth_parent_units": (
            model.growth_parent_units if isinstance(model, GatedGrowthCore) else None
        ),
        "growth_unit_branches": (
            model.growth_unit_branches if isinstance(model, GatedGrowthCore) else None
        ),
        "growth_gates": growth_gates,
        "parameters": {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        },
        "h": state.h.detach().cpu().clone(),
        "fast": state.fast.detach().cpu().clone(),
        "nu_ema": state.nu_ema.detach().cpu().clone(),
        "coverage": state.coverage.detach().cpu().clone(),
        "prior": state.prior.detach().cpu().clone(),
        "keys": [[source, position] for source, position in state.keys],
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "optimizer_config": (
            None
            if optimizer is None
            else {
                "lr": optimizer.param_groups[0]["lr"],
                "betas": list(optimizer.param_groups[0]["betas"]),
                "eps": optimizer.param_groups[0]["eps"],
            }
        ),
        "eligibility": None if eligibility is None else eligibility.detach().cpu().clone(),
        "career": journal_to_dict(career),
        "journal_cursor": journal_cursor,
        "journal_entries": journal_entries or [],
        "pending": pending,
        "rng_state": torch.get_rng_state().clone(),
        "extra": extra or {},
    }
    torch.save(bundle, path)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_subject(path) -> dict:
    return torch.load(path, weights_only=True)


def resume_subject(bundle: dict):
    """Rebuild (model, state, optimizer) from a saved bundle in any process."""
    from taiji.growth_credit import GatedGrowthCore

    config = LivingCoreConfig(
        **{
            key: value
            for key, value in bundle["config"].items()
            if key in LivingCoreConfig.__dataclass_fields__
        }
    )
    grown = bundle["growth_parent_units"] is not None
    model = GatedGrowthCore(config) if grown else LivingCoreReference(config)
    model = model.to(torch.float64)
    parameters = bundle["parameters"]
    if grown:
        model.R = torch.nn.Parameter(parameters["R"].clone())
        model.P = torch.nn.Parameter(parameters["P"].clone())
        model.growth_parent_units = bundle["growth_parent_units"]
        model.growth_unit_branches = bundle["growth_unit_branches"]
        rest = {name: value for name, value in parameters.items() if name not in ("R", "P")}
        model.load_state_dict(rest, strict=False)
    else:
        model.load_state_dict(parameters)
    state = LivingState(
        subject_id=bundle["subject_id"],
        revision=bundle["revision"],
        step=bundle["step"],
        h=bundle["h"],
        fast=bundle["fast"],
        nu_ema=bundle["nu_ema"],
        keys=tuple((source, position) for source, position in bundle["keys"]),
        coverage=bundle["coverage"],
        prior=bundle["prior"],
    )
    optimizer = None
    if bundle["optimizer_state"] is not None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=bundle["optimizer_config"]["lr"],
            betas=tuple(bundle["optimizer_config"]["betas"]),
            eps=bundle["optimizer_config"]["eps"],
        )
        optimizer.load_state_dict(bundle["optimizer_state"])
    torch.set_rng_state(bundle["rng_state"])
    return model, state, optimizer


def eligibility_step(
    eligibility: torch.Tensor, xs: torch.Tensor, decay: float = 0.99
) -> torch.Tensor:
    """Declared eligibility stand-in: e <- decay*e + mean_i ||x_i||_2 / sqrt(p)."""
    norms = xs.square().sum(-1).sqrt().mean(0) / math.sqrt(xs.shape[-1])
    return decay * eligibility + norms


def append_journal(entries: list, kind: str, **fields) -> dict:
    entry = {"seq": max((entry["seq"] for entry in entries), default=-1) + 1, "kind": kind}
    entry.update(fields)
    entries.append(entry)
    return entry


def journal_has_resolution(entries: list, outcome_id: str) -> bool:
    return any(
        entry["kind"] == "resolve_outcome" and entry.get("outcome_id") == outcome_id
        for entry in entries
    )


WORKER_FEE_PER_STEP = 1e-3


def run_worker_job(spec: dict) -> dict:
    """R024 restore worker job; runs in a fresh interpreter process (spawn).

    Modes: microstep (consume the next journal spec), update (declared one-step
    Adam on the teacher-forced replay branch), resolve (idempotent pending-outcome
    resolution against the current journal). Every mode records an RNG probe drawn
    immediately after restore. Deterministic; no task data.
    """

    torch.set_num_threads(1)
    bundle = load_subject(Path(spec["bundle"]))
    model, state, optimizer = resume_subject(bundle)
    rng_probe = torch.rand(3)
    journal = journal_from_dict(bundle["career"])
    gate_table = bundle["growth_gates"]
    if gate_table is None:
        provider = None
    else:

        def provider(index: int) -> dict:
            item = gate_table[index % len(gate_table)]
            if isinstance(item, dict):
                return dict(item)
            return {"copy": item[0], "generate": item[1]}

    eligibility = bundle["eligibility"]
    extra = dict(bundle["extra"])
    entries = [dict(entry) for entry in bundle["journal_entries"]]
    pending = bundle["pending"]
    result = {
        "mode": spec["mode"],
        "subject_id": bundle["subject_id"],
        "revision": bundle["revision"],
        "cursor_in": bundle["journal_cursor"],
        "rng_probe": rng_probe,
        "journal_entries_out": entries,
        "fee_total": extra.get("fee_total", 0.0),
    }
    fee_per_step = float(spec.get("fee_per_step", WORKER_FEE_PER_STEP))
    if spec["mode"] == "microstep":
        cursor = bundle["journal_cursor"]
        specs = career_step_specs(journal)
        if cursor < len(specs):
            new_state, steps = resume_career(model, state, journal, cursor, provider, limit=1)
            output, _ = steps[0]
            eligibility = eligibility_step(eligibility, output.trace["x"].detach())
            extra["fee_total"] = extra.get("fee_total", 0.0) + fee_per_step
            state = new_state
            result.update(
                {
                    "consumed": 1,
                    "cursor_out": cursor + 1,
                    "h": new_state.h.detach().clone(),
                    "probabilities": output.probabilities.detach().clone(),
                    "eligibility": eligibility,
                    "fee_total": extra["fee_total"],
                }
            )
        else:
            result.update({"consumed": 0, "cursor_out": cursor})
    elif spec["mode"] == "update":
        _, steps = resume_career(model, state, journal, 0, provider)
        terms = [output.probabilities[byte] for output, byte in steps if byte is not None]
        loss = -torch.stack(terms).log().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        parameters_after = {
            name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
        }
        cursor = bundle["journal_cursor"]
        specs_total = len(career_step_specs(journal))
        post_update = {}
        if cursor < specs_total:
            post_state, post_steps = resume_career(model, state, journal, cursor, provider, limit=1)
            post_output = post_steps[0][0]
            post_update = {
                "post_update_probabilities": post_output.probabilities.detach().clone(),
                "post_update_h": post_state.h.detach().clone(),
            }
        result.update(
            {
                "replay_loss": float(loss.detach()),
                "parameters_after": parameters_after,
                "post_update_consumed": bool(post_update),
                **post_update,
                "moment_max": max(
                    (
                        float(moment["exp_avg"].abs().max())
                        for moment in optimizer.state.values()
                        if isinstance(moment, dict) and "exp_avg" in moment
                    ),
                    default=0.0,
                ),
            }
        )
    elif spec["mode"] == "resolve":
        current = spec.get("current_entries", [])
        outcome_id = pending["outcome_id"]
        if pending["resolved"] or journal_has_resolution(current, outcome_id):
            result.update({"resolved": False, "reward_delta": 0.0, "guard": "already_resolved"})
        else:
            append_journal(entries, "resolve_outcome", outcome_id=outcome_id)
            result.update({"resolved": True, "reward_delta": pending["reward"], "guard": "applied"})
            pending = {**pending, "resolved": True}
    elif spec["mode"] == "activity_round":
        from taiji.native_activity import ActivityPolicy, env_from_extra, env_to_extra, run_round

        extra = dict(bundle["extra"])
        policy = ActivityPolicy(bundle["config"]["width"]).double()
        policy.load_state_dict(extra["policy_state"])
        env = env_from_extra(extra["activity_env"])
        state, env, entry, last_event_out, error_out = run_round(
            model,
            state,
            env,
            policy,
            entries,
            last_event=torch.tensor(extra["activity_env"]["last_event"], dtype=torch.float64),
            prediction_error_value=extra["activity_env"]["prediction_error"],
            gate_provider=provider,
            gate_index=bundle["journal_cursor"],
        )
        extra["activity_env"] = {
            **env_to_extra(env),
            "last_event": list(last_event_out.detach()),
            "prediction_error": error_out,
        }
        result.update(
            {
                "activity": entry["activity"],
                "decision_entry": entry,
                "h": state.h.detach().clone(),
                "fee_total": extra.get("fee_total", 0.0),
            }
        )
    else:
        raise ValueError(f"Unknown worker mode {spec['mode']}")
    if spec.get("out_bundle") is not None:
        save_subject(
            Path(spec["out_bundle"]),
            model,
            state,
            career=journal,
            journal_cursor=result.get("cursor_out", bundle["journal_cursor"]),
            optimizer=optimizer,
            eligibility=eligibility,
            journal_entries=entries,
            pending=pending,
            growth_gates=gate_table,
            extra=extra,
        )
    if spec.get("out") is not None:
        torch.save(result, spec["out"])
    return result


def parameter_digest(model: LivingCoreReference) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def states_bitwise_equal(left: LivingState, right: LivingState) -> bool:
    return (
        torch.equal(left.h, right.h)
        and torch.equal(left.fast, right.fast)
        and torch.equal(left.nu_ema, right.nu_ema)
        and torch.equal(left.coverage, right.coverage)
        and torch.equal(left.prior, right.prior)
        and left.keys == right.keys
        and left.step == right.step
    )


def replay_transaction(
    parent_model: LivingCoreReference,
    parent_state: LivingState,
    journal: CareerJournal,
    coordinate_change=None,
    budget: MigrationBudget | None = None,
    growth_gate_provider=None,
) -> dict:
    """Bounded causal replay: shadow rebuild, optional declared coordinate change.

    ``coordinate_change`` is a callable applied to the shadow model only. Replay
    performs no parameter updates, reward claims or external actions. A budget
    refusal keeps the parent revision and records the cost of the attempt.
    """
    budget = budget or MigrationBudget()
    events, bytes_total = journal.event_count, journal.journal_bytes
    if budget.max_events is not None and events > budget.max_events:
        return {"outcome": "refused_budget", "reason": "event_cap", "events": events}
    if budget.max_journal_bytes is not None and bytes_total > budget.max_journal_bytes:
        return {"outcome": "refused_budget", "reason": "byte_cap", "bytes": bytes_total}
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    parent_digest = parameter_digest(parent_model)
    shadow = copy.deepcopy(parent_model)
    encoder_changed = False
    if coordinate_change is not None:
        before = parameter_digest(shadow)
        coordinate_change(shadow)
        encoder_changed = parameter_digest(shadow) != before
    after_change = parameter_digest(shadow)
    with torch.no_grad():
        rebuilt_state, _ = run_career(shadow, journal, growth_gate_provider=growth_gate_provider)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    no_replay_mutation = parameter_digest(shadow) == after_change
    result = {
        "events": events,
        "journal_bytes": bytes_total,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "watermark": {"events_consumed": events, "journal_steps": journal.event_count},
        "no_replay_parameter_mutation": no_replay_mutation,
        "coordinate_change_applied": encoder_changed,
        "reward_ledger_untouched": True,
        "external_actions_executed": 0,
    }
    if budget.max_cpu_seconds is not None and cpu_seconds > budget.max_cpu_seconds:
        result.update({"outcome": "refused_budget", "reason": "cpu_cap"})
        return result
    if budget.max_wall_seconds is not None and wall_seconds > budget.max_wall_seconds:
        result.update({"outcome": "refused_budget", "reason": "wall_cap"})
        return result
    if coordinate_change is None:
        identical = states_bitwise_equal(rebuilt_state, parent_state)
        result.update(
            {
                "outcome": "identity_checked",
                "state_bitwise_equal": identical,
                "committed_state": rebuilt_state if identical else None,
            }
        )
        return result
    committed = LivingState(
        subject_id=parent_state.subject_id,
        revision=parent_state.revision + 1,
        step=rebuilt_state.step,
        h=rebuilt_state.h,
        fast=rebuilt_state.fast,
        nu_ema=rebuilt_state.nu_ema,
        keys=rebuilt_state.keys,
        coverage=rebuilt_state.coverage,
        prior=rebuilt_state.prior,
    )
    result.update(
        {
            "outcome": "committed",
            "committed_state": committed,
            "committed_model": shadow,
            "parent_parameter_digest": parent_digest,
        }
    )
    return result
