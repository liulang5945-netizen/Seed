"""20.1 formal graphs: CF1 (feedback binding fast memory) and B_EH (no F).

Both arms are the same module class around the frozen R021 subject.  CF1 may
write the fast memory (``LivingState.fast``) on released feedback batches only;
B_EH never writes it.  Nothing else differs, so with F closed the two graphs
must produce byte-identical logits from the same parent weights.

The module never imports the fixture/solver module: the runner passes plain
dicts whose keys are the response fields.  No rule, signature or truth table can
reach this file.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from taiji.living_core import EOS, LivingState, RawSource

ARMS = ("CF1", "B_EH")
FAST_ARM = "CF1"
ACTIONS = ("应送左箱。", "应送右箱。")
ANSWER_BYTES = len(ACTIONS[0].encode("utf-8"))
ANSWER_PREFIX = "应送"
ANSWER_SUFFIX = "箱。"
DECISION_POSITIONS = tuple(
    position
    for position in range(ANSWER_BYTES)
    if ACTIONS[0].encode("utf-8")[position] != ACTIONS[1].encode("utf-8")[position]
)
TEMPLATE_BYTES = {
    position: ACTIONS[0].encode("utf-8")[position]
    for position in range(ANSWER_BYTES)
    if position not in DECISION_POSITIONS
}
TEMPLATE_ALPHABET = frozenset(TEMPLATE_BYTES.values())

GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "encoder": ("encoder.",),
    "nu": ("nu_projection.",),
    "message": ("message_q.", "message_k.", "message_v.", "message_goal."),
    "fast_read": ("fast_queries.", "fast_outputs."),
    "episodic": ("episodic.",),
    "W": ("W",),
    "U": ("U",),
    "V": ("V",),
    "P": ("P",),
    "R": ("R",),
    "tau": ("tau_logit",),
    "xi": ("xi.",),
    "readout": ("readout.",),
    "expression": ("expression.",),
    "write_heads": ("write_heads.",),
    "world": ("world.",),
}
FROZEN_UNUSED_GROUPS = ("world",)


def _group_of(name: str) -> str:
    for group, prefixes in GROUP_PREFIXES.items():
        if name.startswith(prefixes) or name in prefixes:
            return group
    raise KeyError(f"parameter {name!r} has no declared group")


@dataclass(frozen=True)
class RuleState:
    living: LivingState
    journal: tuple[dict[str, Any], ...]
    revision: int


def state_digest(state: RuleState) -> str:
    living = state.living
    rows: list[tuple[Any, ...]] = [
        ("subject_id", living.subject_id),
        ("revision", living.revision),
        ("step", living.step),
        ("h", living.h.detach().cpu().numpy().tobytes()),
        ("fast", living.fast.detach().cpu().numpy().tobytes()),
        ("nu_ema", living.nu_ema.detach().cpu().numpy().tobytes()),
        ("keys", tuple(living.keys)),
        ("coverage", living.coverage.detach().cpu().numpy().tobytes()),
        ("prior", living.prior.detach().cpu().numpy().tobytes()),
    ]
    for entry in state.journal:
        rows.append(("journal", tuple(sorted(entry.items()))))
    payload = repr(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RuleLearningGraph(nn.Module):
    """CF1 / B_EH: one subject, one answer path, F gated by the arm."""

    def __init__(
        self,
        subject: nn.Module,
        *,
        arm: str,
        growth_gates: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.subject = subject
        self.arm = arm
        self.fast_enabled = arm == FAST_ARM
        grown = getattr(subject, "growth_parent_units", None) is not None
        self.step_kwargs: dict[str, Any] = (
            {"growth_gates": dict(growth_gates)} if grown and growth_gates else {}
        )

    # ---- freeze declarations -------------------------------------------------
    def parameter_table(self) -> list[dict[str, Any]]:
        rows = []
        for name, parameter in self.subject.named_parameters():
            group = _group_of(name)
            rows.append(
                {
                    "group": group,
                    "name": f"subject.{name}",
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype).replace("torch.", ""),
                    "elements": int(parameter.numel()),
                    "requires_grad": bool(parameter.requires_grad),
                    "outer_trainable": group not in FROZEN_UNUSED_GROUPS,
                    "session_frozen": True,
                    "inference_plastic": False,
                }
            )
        return rows

    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.parameter_table():
            counts[row["group"]] = counts.get(row["group"], 0) + row["elements"]
        return counts

    def state_table(self) -> list[dict[str, Any]]:
        subject = self.subject
        config = subject.config
        return [
            {
                "name": "h",
                "kind": "persistent",
                "shape": [config.groups, config.width],
                "dtype": "float64",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "fast",
                "kind": "persistent",
                "shape": [config.groups, config.fast_heads, config.key_width, config.value_width],
                "dtype": "float64",
                "restore": "value",
                "plastic": self.fast_enabled,
            },
            {
                "name": "nu_ema",
                "kind": "persistent",
                "shape": [8],
                "dtype": "float64",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "keys",
                "kind": "persistent",
                "shape": ["record_count"],
                "dtype": "tuple",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "coverage",
                "kind": "persistent",
                "shape": ["record_bytes"],
                "dtype": "float64",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "prior",
                "kind": "persistent",
                "shape": ["record_bytes"],
                "dtype": "float64",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "journal",
                "kind": "persistent",
                "shape": ["receipts"],
                "dtype": "json",
                "restore": "value",
                "plastic": False,
            },
            {
                "name": "previous_byte",
                "kind": "session",
                "shape": [1],
                "dtype": "int",
                "restore": "reset to EOS",
                "plastic": False,
            },
            {
                "name": "turn_cursor",
                "kind": "session",
                "shape": [1],
                "dtype": "int",
                "restore": "reset to 0",
                "plastic": False,
            },
        ]

    # ---- state ---------------------------------------------------------------
    def initial_state(self, world_id: str) -> RuleState:
        living = self.subject.initial_state(f"20.1-{world_id}")
        return RuleState(living, (), 0)

    def save_state(self, state: RuleState) -> dict[str, Any]:
        living = state.living
        return {
            "subject_id": living.subject_id,
            "revision": living.revision,
            "step": living.step,
            "h": living.h.detach().cpu().clone(),
            "fast": living.fast.detach().cpu().clone(),
            "nu_ema": living.nu_ema.detach().cpu().clone(),
            "keys": [[source, int(offset)] for source, offset in living.keys],
            "coverage": living.coverage.detach().cpu().clone(),
            "prior": living.prior.detach().cpu().clone(),
            "journal": copy.deepcopy(list(state.journal)),
            "graph_revision": state.revision,
        }

    def restore_state(self, payload: dict[str, Any]) -> RuleState:
        reference = self.subject.W
        living = LivingState(
            subject_id=payload["subject_id"],
            revision=int(payload["revision"]),
            step=int(payload["step"]),
            h=payload["h"].to(reference.dtype),
            fast=payload["fast"].to(reference.dtype),
            nu_ema=payload["nu_ema"].to(reference.dtype),
            keys=tuple((source, int(offset)) for source, offset in payload["keys"]),
            coverage=payload["coverage"].to(reference.dtype),
            prior=payload["prior"].to(reference.dtype),
        )
        return RuleState(
            living, tuple(copy.deepcopy(payload["journal"])), int(payload["graph_revision"])
        )

    def _records(self, released: list[dict[str, Any]], now: int) -> tuple[RawSource, ...]:
        return tuple(
            RawSource(
                source_id=item["source_id"],
                raw=item["feedback"].encode("utf-8"),
                observed_at=int(item.get("after_turn", 0)),
                version=int(item["version"]),
                owner="owner",
                scope="scope",
                real=True,
            )
            for item in released
        )

    def _allowed(self, enabled: bool) -> torch.Tensor:
        c = self.subject.config
        return torch.full((c.groups, c.fast_heads), bool(enabled), dtype=torch.bool)

    @staticmethod
    def _patch(subject: nn.Module, byte: int) -> torch.Tensor:
        if byte == EOS:
            return subject.W.new_zeros(subject.config.width)
        return subject.encoder.patches(bytes([byte]))[0]

    def _provenance(self, output, byte: int, position: int) -> dict[str, Any]:
        weights = output.copy_weights
        ids = output.sources.byte_ids
        keys = output.sources.keys
        refs = []
        for index in range(len(keys)):
            if int(ids[index]) == byte and float(weights[index]) > 0:
                source_id, offset = keys[index]
                refs.append((str(source_id), int(offset), float(weights[index])))
        if refs:
            return {"source": "feedback", "refs": refs[:4]}
        if byte in TEMPLATE_ALPHABET:
            return {"source": "template", "refs": [], "position": position}
        return {"source": "generate", "refs": []}

    @torch.no_grad()
    def run_turn(
        self,
        state: RuleState,
        turn: dict[str, Any],
        released: list[dict[str, Any]],
        *,
        write_enabled: bool | None = None,
        vocab: tuple[int, ...] | None = None,
        now: int = 0,
    ) -> dict[str, Any]:
        query = turn["query"].encode("utf-8")
        records = self._records(released, now)
        living = self.subject.observe(
            state.living,
            query,
            goal=query,
            records=records,
            allowed=self._allowed(False),
            now=now,
            **self.step_kwargs,
        )
        logits: list[torch.Tensor] = []
        emitted: list[int] = []
        constrained: list[int] = []
        provenance: list[dict[str, Any]] = []
        constrained_provenance: list[dict[str, Any]] = []
        previous = EOS
        allowed_ids = None if vocab is None else torch.tensor(vocab, dtype=torch.long)
        for position in range(ANSWER_BYTES):
            event = self._patch(self.subject, previous)
            output = self.subject.step(
                living,
                event=event,
                goal=query,
                records=records,
                previous_byte=previous,
                allowed=self._allowed(False),
                now=now,
                **self.step_kwargs,
            )
            probabilities = output.probabilities
            logits.append(probabilities)
            choice = int(torch.argmax(probabilities[:256]))
            emitted.append(choice)
            provenance.append(self._provenance(output, choice, position))
            if allowed_ids is not None:
                masked = torch.full_like(probabilities, float("-inf"))
                masked[allowed_ids] = probabilities[allowed_ids]
                picked = int(torch.argmax(masked[:256]))
                constrained.append(picked)
                constrained_provenance.append(self._provenance(output, picked, position))
            living = self.subject.consume_output(output, choice)
            previous = choice
        prediction_id = f"{turn['turn']}-{state.revision}"
        entry = {
            "prediction_id": prediction_id,
            "turn": int(turn["turn"]),
            "session": int(turn["session"]),
            "version": int(turn["version"]),
            "query_sha256": hashlib.sha256(query).hexdigest(),
            "model_version": self.arm,
            "output": bytes(emitted).hex(),
            "released": False,
            "applied": False,
        }
        child = RuleState(living, state.journal + (entry,), state.revision)
        return {
            "turn": int(turn["turn"]),
            "bytes": tuple(emitted),
            "constrained_bytes": tuple(constrained),
            "logits": torch.stack(logits),
            "provenance": provenance,
            "constrained_provenance": constrained_provenance,
            "state": child,
            "prediction_id": prediction_id,
        }

    @torch.no_grad()
    def release(
        self,
        state: RuleState,
        release: dict[str, Any],
        released: list[dict[str, Any]],
        *,
        write_enabled: bool | None = None,
        now: int = 0,
    ) -> tuple[RuleState, dict[str, Any]]:
        """Atomic batch release: every line is observed before any is readable."""
        write = self.fast_enabled if write_enabled is None else bool(write_enabled)
        records = self._records(released, now)
        living = state.living
        written = 0
        for item in release["items"]:
            feedback = item["feedback"].encode("utf-8")
            query = item["query"].encode("utf-8")
            before = living.fast.clone()
            living = self.subject.observe(
                living,
                feedback,
                goal=query,
                records=records,
                allowed=self._allowed(write),
                now=now,
                **self.step_kwargs,
            )
            if not torch.equal(before, living.fast):
                written += 1
        journal = []
        applied = 0
        for entry in state.journal:
            if (
                entry["turn"] in {item["turn"] for item in release["items"]}
                and not entry["applied"]
            ):
                entry = dict(
                    entry, released=True, applied=True, release_version=int(release["version"])
                )
                applied += 1
            journal.append(entry)
        child = RuleState(living, tuple(journal), state.revision + 1)
        return child, {
            "written": written,
            "applied_receipts": applied,
            "release_version": int(release["version"]),
            "after_turn": int(release["after_turn"]),
        }

    def candidate_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "fast_enabled": self.fast_enabled,
            "parameter_table": self.parameter_table(),
            "group_counts": self.group_counts(),
            "state_table": self.state_table(),
        }


def copy_subject(subject: nn.Module) -> nn.Module:
    clone = copy.deepcopy(subject)
    clone.eval()
    return clone
