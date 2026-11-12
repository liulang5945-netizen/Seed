"""Opt-in U2 subject with versioned, public-packet evidence reading."""

from __future__ import annotations

import copy
import math

import torch

from .u2_evidence_read import QueryEvidenceReader
from .u2_history import value_digest
from .u2_subject import U2Subject

SCHEMA = "taiji-u2-query-evidence-subject-v1"


def evidence_digest(row):
    # The encoder revision changes during replay; the original packet remains factual.
    return value_digest({key: value for key, value in row.items() if key != "subject_version"})


class EvidenceSubject(U2Subject):
    """An explicit candidate profile; the base U2 subject keeps its original graph."""

    def __init__(self, config):
        if not config.separate_requests:
            raise ValueError("evidence_reader_requires_request_profile")
        super().__init__(config)
        self.evidence_reader = QueryEvidenceReader(config.d)

    @classmethod
    def from_parent(cls, parent):
        if isinstance(parent, cls):
            raise ValueError("evidence_candidate_already_forked")
        candidate = cls(parent.c).to(dtype=parent.native_w.dtype, device=parent.native_w.device)
        missing, unexpected = candidate.load_state_dict(parent.state_dict(), strict=False)
        if unexpected or set(missing) != {
            f"evidence_reader.{name}" for name in candidate.evidence_reader.state_dict()
        }:
            raise ValueError("evidence_parent_parameter_mismatch")
        candidate.parameter_revision = parent.parameter_revision + 1
        candidate.generation = parent.generation
        candidate.unit_ids = copy.deepcopy(parent.unit_ids)
        candidate.unit_branches = copy.deepcopy(parent.unit_branches)
        return candidate

    def initial(self, subject_id="u2-reference"):
        return {**super().initial(subject_id), "evidence_directory": {}, "revoked_evidence": []}

    def _register(self, state):
        directory = dict(state["evidence_directory"])
        for row in state["evidence"]:
            identity = row["packet_id"]
            if identity in state["seen_packets"]:
                digest = evidence_digest(row)
                if identity in directory and directory[identity] != digest:
                    raise ValueError("evidence_packet_identity_changed")
                directory[identity] = digest
        child = {**state, "evidence_directory": directory}
        self.evidence_tokens(child)
        return child

    def assimilate(self, state, packet):
        child, result = super().assimilate(state, packet)
        return self._register(child), result

    def receive(self, state, packet):
        child, result = super().receive(state, packet)
        return self._register(child), result

    def revoke_evidence(self, state, packet_id):
        if state["pending"] is not None or state["utterance_active"]:
            raise ValueError("evidence_revocation_requires_committed_boundary")
        if packet_id not in state["evidence_directory"]:
            raise ValueError("unknown_evidence_packet")
        return {**state, "revoked_evidence": sorted(set(state["revoked_evidence"]) | {packet_id})}

    def evidence_tokens(self, state):
        directory = state["evidence_directory"]
        revoked = state["revoked_evidence"]
        if (
            not isinstance(directory, dict)
            or revoked != sorted(set(revoked))
            or not set(revoked) <= set(directory)
        ):
            raise ValueError("invalid_evidence_directory")
        rows = {row["packet_id"]: row for row in state["evidence"]}
        if not set(directory) <= set(rows):
            raise ValueError("missing_canonical_evidence")
        active, encoded, coordinates = [], [], []
        for identity, digest in directory.items():
            row = rows[identity]
            if evidence_digest(row) != digest:
                raise ValueError("canonical_evidence_digest_mismatch")
            if (
                row["source"] != "sensor"
                or row["scope"] != "world"
                or identity not in state["seen_packets"]
                or row["receipt_id"] not in state["settled"]
                or not all(math.isfinite(row[k]) for k in ("t_start", "t_end", "t_arrival"))
                or not 0 <= row["t_start"] <= row["t_end"] <= state["world_time"] + 1e-10
                or row["t_arrival"] > state["process_time"]
            ):
                raise ValueError("evidence_source_not_admitted")
            if identity not in revoked and row["observations"]:
                active.append(row)
        if len(active) > 4:
            raise ValueError("evidence_event_budget_stop")
        for row in sorted(active, key=lambda r: (r["t_end"], r["packet_id"])):
            obs = {key: value.to(self.native_w) for key, value in row["observations"].items()}
            coords = {
                key: value.to(self.native_w).clone() for key, value in row["coordinates"].items()
            }
            if set(obs) != set(coords) or not set(obs) <= {"rgb", "audio", "proprio"}:
                raise ValueError("evidence_modality_not_admitted")
            tokens = self.encode(obs, coords)
            for modality in sorted(tokens):
                if coords[modality].shape != (tokens[modality].shape[0], 8):
                    raise ValueError("evidence_coordinate_shape")
                # Sensor metadata flags describe coordinate types, not token validity.
                # Absence is represented by the admitted modality set and proprio mask.
                metadata = coords[modality].clone()
                metadata[:, 3:5] = (
                    metadata[:, 3:5].sinh() + row["t_end"] - state["world_time"]
                ).asinh()
                encoded.append(tokens[modality])
                coordinates.append(metadata)
        if not encoded:
            return self.native_w.new_zeros(0, self.c.d), self.native_w.new_zeros(0, 8)
        result = torch.cat(encoded), torch.cat(coordinates)
        if result[0].shape[0] > 128:
            raise ValueError("evidence_reader_token_budget_stop")
        return result

    def branch_context(self, state, goal, action):
        original = super().branch_context(state, goal, action)
        if "query_evidence" in getattr(self, "_verification_cuts", ()):
            return original
        tokens, coordinates = self.evidence_tokens(state)
        context, _, _ = self.evidence_reader(
            goal,
            self.read_h(state, goal),
            original,
            tokens,
            coordinates,
            torch.ones(tokens.shape[0], dtype=torch.bool, device=tokens.device),
        )
        return context

    def export_bundle(self, state, optimizer_state=None):
        expected = {
            row["packet_id"]
            for row in state["evidence"]
            if row["packet_id"] in state["seen_packets"]
        }
        if set(state["evidence_directory"]) != expected:
            raise ValueError("incomplete_evidence_directory")
        self.evidence_tokens(state)
        bundle = super().export_bundle(state, optimizer_state)
        return {**bundle, "schema": SCHEMA}

    @classmethod
    def from_bundle(cls, bundle):
        if bundle["schema"] != SCHEMA:
            raise ValueError("evidence_subject_schema_mismatch")
        model, state = super().from_bundle({**bundle, "schema": "taiji-u2-subject-v4"})
        model.evidence_tokens(state)
        return model, state
