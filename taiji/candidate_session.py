"""One task entry for one candidate: observe / answer / clarify / reset, raw first.

The rule this module enforces is the 11.2 card's: *store the raw model output before
any decoding*, and never let the display layer fill in content the model did not
produce. ``answer`` therefore returns ``raw`` and ``decoded`` separately, and ``decoded``
may only echo the raw top byte.

The loader is manifest-bound (11.1), the state comes from ``resume_subject``, and in
``frozen_eval`` mode any mutating call is refused.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import torch

from taiji.candidate_manifest import load_candidate
from taiji.living_core import EOS, LivingState
from taiji.state_migration import resume_subject

MUTATING = ("learn", "update", "fit")


def state_digest(state) -> str:
    digest = hashlib.sha256()
    for name in ("h", "fast", "nu_ema", "coverage", "prior"):
        tensor = getattr(state, name)
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CandidateSession:
    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        bundle_path: Path,
        mode: str = "frozen_eval",
    ) -> None:
        self.manifest, payload = load_candidate(root, manifest_path, bundle_path, mode)
        self._payload = copy.deepcopy(payload)
        self.mode = mode
        self.model, self.state, _optimizer = resume_subject(payload)
        self.gates = payload["growth_gates"][0]
        self.initial_digest = state_digest(self.state)
        self.goal = b""
        self.records: tuple[Any, ...] = ()
        self.transcript: list[dict] = []

    def _record(self, entry: dict) -> dict:
        self.transcript.append(entry)
        return entry

    def observe(self, text: str, records=()) -> dict:
        encoded = text.encode()
        self.goal = encoded
        self.records = tuple(records)
        with torch.no_grad():
            for patch in self.model.encoder.patches(encoded):
                output = self.model.step(
                    self.state,
                    event=patch,
                    goal=encoded,
                    records=self.records,
                    growth_gates=self.gates,
                )
                self.state = output.state
        return self._record(
            {"call": "observe", "bytes": len(encoded), "state": state_digest(self.state)}
        )

    def answer(self) -> dict:
        with torch.no_grad():
            output = self.model.step(
                self.state,
                event=self.model.W.new_zeros(self.model.config.width),
                goal=self.goal or b"x",
                records=self.records,
                previous_byte=EOS,
                growth_gates=self.gates,
            )
            probabilities = output.probabilities
            top = int(torch.argmax(probabilities))
            top_probability = float(probabilities[top])
        raw = {
            "top_byte": top,
            "top_probability": top_probability,
            "entropy": float(
                -sum(float(p) * math.log(float(p)) for p in probabilities if float(p) > 0.0)
            ),
        }
        decoded = None
        if top != EOS:
            try:
                decoded = bytes([top]).decode("utf-8")
            except UnicodeDecodeError:
                decoded = None
        return self._record({"call": "answer", "raw": raw, "decoded": decoded})

    def respond(self, max_bytes: int = 16) -> dict:
        produced: list[int] = []
        raw_steps: list[dict] = []
        previous = EOS
        with torch.no_grad():
            for _ in range(max_bytes):
                event = (
                    self.model.W.new_zeros(self.model.config.width)
                    if previous == EOS
                    else self.model.encoder.patches(bytes([previous]))[0]
                )
                output = self.model.step(
                    self.state,
                    event=event,
                    goal=self.goal or b"x",
                    records=self.records,
                    previous_byte=previous,
                    growth_gates=self.gates,
                )
                probabilities = output.probabilities
                top = int(torch.argmax(probabilities))
                raw_steps.append(
                    {
                        "top_byte": top,
                        "top_probability": float(probabilities[top]),
                    }
                )
                if top == EOS:
                    break
                produced.append(top)
                self.state = self.model.consume_output(output, top)
                previous = top
        raw = {"steps": raw_steps}
        text = bytes(produced).decode("utf-8", errors="replace")
        return self._record({"call": "respond", "raw": raw, "text": text})

    def answer_or_abstain(self, confidence: float = 0.5, entropy: float = 3.0) -> dict:
        """Abstain from the model's own distribution, never from a text template."""
        with torch.no_grad():
            output = self.model.step(
                self.state,
                event=self.model.W.new_zeros(self.model.config.width),
                goal=self.goal or b"x",
                records=self.records,
                previous_byte=EOS,
                growth_gates=self.gates,
            )
            probabilities = output.probabilities
            top = int(torch.argmax(probabilities))
            top_probability = float(probabilities[top])
            observed_entropy = float(
                -sum(float(p) * math.log(float(p)) for p in probabilities if float(p) > 0.0)
            )
        raw = {
            "top_byte": top,
            "top_probability": top_probability,
            "entropy": observed_entropy,
            "confidence_gate": confidence,
            "entropy_gate": entropy,
        }
        abstain = top_probability < confidence or observed_entropy > entropy
        if abstain:
            return self._record(
                {
                    "call": "answer_or_abstain",
                    "raw": raw,
                    "kind": "abstain",
                    "content": None,
                }
            )
        return self._record(
            {
                "call": "answer_or_abstain",
                "raw": raw,
                "kind": "answer",
                "content": top,
            }
        )

    def clarify(self, reason: str) -> dict:
        return self._record(
            {"call": "clarify", "reason": reason, "content": None, "kind": "clarify"}
        )

    def reset(self) -> dict:
        self.model, self.state, _optimizer = resume_subject(copy.deepcopy(self._payload))
        self.goal = b""
        self.records = ()
        return self._record({"call": "reset", "state": state_digest(self.state)})

    def save(self, path: Path) -> dict:
        snapshot = {
            "schema": "taiji-session-snapshot-v1",
            "candidate_id": self.manifest["candidate_id"],
            "state": {
                "subject_id": self.state.subject_id,
                "revision": self.state.revision,
                "step": self.state.step,
                "h": self.state.h,
                "fast": self.state.fast,
                "nu_ema": self.state.nu_ema,
                "keys": [list(pair) for pair in self.state.keys],
                "coverage": self.state.coverage,
                "prior": self.state.prior,
            },
            "goal": self.goal,
            "records": [list(record) for record in self.records],
        }
        torch.save(snapshot, path)
        return self._record({"call": "save", "state": state_digest(self.state)})

    def load(self, path: Path) -> dict:
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if snapshot.get("candidate_id") != self.manifest["candidate_id"]:
            raise ValueError("snapshot belongs to another candidate")
        state = snapshot["state"]
        self.state = LivingState(
            subject_id=state["subject_id"],
            revision=state["revision"],
            step=state["step"],
            h=state["h"],
            fast=state["fast"],
            nu_ema=state["nu_ema"],
            keys=tuple((source, position) for source, position in state["keys"]),
            coverage=state["coverage"],
            prior=state["prior"],
        )
        self.goal = snapshot["goal"]
        self.records = tuple(snapshot.get("records") or ())
        return self._record({"call": "load", "state": state_digest(self.state)})

    def outcome(self, intent: str, result: str, source: str) -> dict:
        before = state_digest(self.state)
        entry = self._record(
            {
                "call": "outcome",
                "intent": intent,
                "result": result,
                "source": source,
                "state_unchanged": state_digest(self.state) == before,
            }
        )
        return entry

    def act(self) -> dict:
        if self.mode == "frozen_eval":
            return self._record(
                {
                    "call": "act",
                    "refused": "no_action_channel",
                    "reason": "this candidate exposes no action path",
                }
            )
        return self._record({"call": "act", "refused": None})

    def refuse_mutation(self, name: str) -> dict:
        if self.mode == "frozen_eval" and name in MUTATING:
            return self._record({"call": name, "refused": "frozen_eval"})
        return self._record({"call": name, "refused": None})
