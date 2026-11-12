"""Full public-stream GRU and one-layer causal Transformer answer baselines."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from .u2_history import assert_values_equal, atomic_load, atomic_save, detached_tree
from .u2_operators import utf8_mask
from .u2_subject import U2Config, U2Subject
from .utf8_state import advance_utf8


class PublicFrontend(nn.Module):
    """Reuse the native trainable encoders, retaining all units and metadata."""

    encoder: nn.ModuleDict
    metadata: nn.Linear
    modality: nn.Parameter
    embedding: nn.Embedding
    text_input: nn.Linear
    convolutions: nn.ModuleList
    goal_score: nn.Linear
    goal_value: nn.Linear
    outcome: nn.Linear
    action: nn.Linear
    record_metadata: nn.Linear
    record_provenance: nn.Linear

    def __init__(self):
        super().__init__()
        source = U2Subject(U2Config.small()).metadata_candidate()
        self.c = source.c
        for name in (
            "encoder",
            "metadata",
            "modality",
            "embedding",
            "text_input",
            "convolutions",
            "goal_score",
            "goal_value",
            "outcome",
            "action",
            "record_metadata",
            "record_provenance",
        ):
            setattr(self, name, getattr(source, name))

    @property
    def native_w(self):
        return self.embedding.weight

    def text_units(self, raw):
        return U2Subject.text_units(cast(Any, self), raw)

    def goal(self, raw):
        return U2Subject.goal(cast(Any, self), raw)

    def encode(self, observations, coordinates, outcome):
        return U2Subject.encode(cast(Any, self), observations, coordinates, outcome)

    def record_units(self, records, world_time):
        return U2Subject.record_units(
            cast(Any, self), {"records": tuple(records), "world_time": world_time}
        )


class TaskSequenceBaseline(nn.Module):
    def __init__(self, architecture="transformer", *, seed=20261110, max_tokens=4096):
        super().__init__()
        if (
            architecture not in {"transformer", "gru"}
            or type(max_tokens) is not int
            or max_tokens < 1
        ):
            raise ValueError("unsupported_public_stream_baseline")
        self.architecture, self.seed, self.max_tokens = architecture, seed, max_tokens
        self.width = 28 if architecture == "transformer" else 40
        self.parameter_revision = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.frontend = PublicFrontend()
            if architecture == "gru":
                self.gru = nn.GRU(33, self.width, batch_first=True)
            else:
                self.record_in = nn.Linear(33, self.width)
                self.position = nn.Parameter(torch.randn(8, self.width) * 0.05)
                self.attention = nn.MultiheadAttention(self.width, 4, batch_first=True, dropout=0)
                self.norm_attention = nn.LayerNorm(self.width)
                self.norm_feedforward = nn.LayerNorm(self.width)
                self.feedforward = nn.Sequential(
                    nn.Linear(self.width, 4 * self.width),
                    nn.GELU(),
                    nn.Linear(4 * self.width, self.width),
                )
            self.byte_head = nn.Linear(self.width + 4, 257)

    def initial(self):
        weight = self.frontend.embedding.weight
        state = {
            "revision": self.parameter_revision,
            "goal_raw": b"",
            "request_raw": b"",
            "active": False,
            "prefix": b"",
            "previous": 256,
            "remaining": 0,
            "lead": 0,
            "data_bytes": 0,
            "count": 0,
            "last": weight.new_zeros(self.width),
            "ops": (),
        }
        if self.architecture == "gru":
            state["hidden"] = weight.new_zeros(1, 1, self.width)
        else:
            state["keys"] = weight.new_zeros(4, 0, 7)
            state["values"] = weight.new_zeros(4, 0, 7)
        return state

    def _append(self, state, content, metadata, kind, *, query_context=False):
        if state["revision"] != self.parameter_revision:
            raise ValueError("baseline_state_requires_reencoding")
        if state["count"] + len(content) > self.max_tokens:
            raise ValueError("baseline_sequence_budget_stop")
        context = self.frontend.goal(state["request_raw"] if query_context else state["goal_raw"])
        tag = content.new_zeros(9)
        tag[kind] = 1
        result = dict(state)
        for unit, coordinate in zip(content, metadata, strict=True):
            token = torch.cat([unit, context, coordinate, tag])
            if self.architecture == "gru":
                output, hidden = self.gru(token[None, None], result["hidden"])
                result["hidden"], result["last"] = hidden, output[0, -1]
            else:
                index = result["count"]
                frequency = torch.exp(
                    torch.arange(0, self.width, 2, device=token.device, dtype=token.dtype)
                    * (-math.log(10000) / self.width)
                )
                sinusoid = token.new_zeros(self.width)
                sinusoid[0::2], sinusoid[1::2] = (index * frequency).sin(), (
                    index * frequency
                ).cos()
                hidden = self.record_in(token) + self.position[index % 8] + sinusoid
                query, key, value = F.linear(
                    hidden, self.attention.in_proj_weight, self.attention.in_proj_bias
                ).chunk(3)
                query = query.reshape(4, 1, 7)
                keys = torch.cat([result["keys"], key.reshape(4, 1, 7)], 1)
                values = torch.cat([result["values"], value.reshape(4, 1, 7)], 1)
                attended = F.scaled_dot_product_attention(
                    query[None], keys[None], values[None], dropout_p=0
                )[0, :, 0].reshape(self.width)
                hidden = self.norm_attention(hidden + self.attention.out_proj(attended))
                result["last"] = self.norm_feedforward(hidden + self.feedforward(hidden))
                result["keys"], result["values"] = keys, values
            result["count"] += 1
        return result

    def _apply_event(self, state, kind, args, *, record=True):
        if state["revision"] != self.parameter_revision:
            raise ValueError("baseline_state_requires_reencoding")
        result = dict(state)
        if kind in {"goal", "request"}:
            raw = args["raw"].encode() if isinstance(args["raw"], str) else bytes(args["raw"])
            if not raw or len(raw) > 512:
                raise ValueError("baseline_text_outside_profile")
            if kind == "request" and state["active"]:
                raise ValueError("baseline_request_requires_answer_boundary")
            result["goal_raw" if kind == "goal" else "request_raw"] = raw
            if kind == "request":
                result.update(
                    active=True, prefix=b"", previous=256, remaining=0, lead=0, data_bytes=0
                )
            units = self.frontend.text_units(raw)
            result = self._append(
                result,
                units,
                units.new_zeros(len(units), 8),
                3 if kind == "goal" else 4,
                query_context=kind == "request",
            )
        elif kind == "observation":
            encoded = self.frontend.encode(
                args["observations"], args["coordinates"], args.get("outcome")
            )
            for name, units in encoded.items():
                coordinates = args["coordinates"].get(name, units.new_zeros(len(units), 8))
                result = self._append(
                    result,
                    units,
                    coordinates,
                    {"rgb": 0, "audio": 1, "proprio": 2, "outcome": 6}[name],
                )
        elif kind == "action":
            units = self.frontend.action(args["action"])[None]
            result = self._append(result, units, units.new_zeros(1, 8), 7)
        elif kind == "records":
            units, valid = self.frontend.record_units(args["records"], args["world_time"])
            result = self._append(result, units[valid], units.new_zeros(int(valid.sum()), 8), 5)
        elif kind == "byte":
            symbol = args["symbol"]
            if not result["active"]:
                raise ValueError("baseline_output_requires_request")
            mask = utf8_mask(
                result["remaining"],
                result["lead"],
                64 - result["data_bytes"],
                device=result["last"].device,
            )
            if type(symbol) is not int or not 0 <= symbol <= 256 or not bool(mask[symbol]):
                raise ValueError("invalid_baseline_byte")
            if symbol == 256:
                result["active"] = False
            else:
                result["prefix"] += bytes([symbol])
                result["data_bytes"] += 1
                result["remaining"], result["lead"] = advance_utf8(
                    result["remaining"], result["lead"], symbol
                )
                result["previous"] = symbol
                unit = self.frontend.text_units(result["prefix"])[-1:]
                result = self._append(result, unit, unit.new_zeros(1, 8), 8, query_context=True)
        else:
            raise ValueError("unsupported_baseline_public_operation")
        if record:
            result["ops"] = (*state["ops"], detached_tree({"kind": kind, "args": args}))
        return result

    def set_goal(self, state, raw):
        return self._apply_event(state, "goal", {"raw": raw})

    def set_request(self, state, raw):
        return self._apply_event(state, "request", {"raw": raw})

    def observe(self, state, observations, coordinates, outcome=None):
        return self._apply_event(
            state,
            "observation",
            {"observations": observations, "coordinates": coordinates, "outcome": outcome},
        )

    def action(self, state, action):
        return self._apply_event(state, "action", {"action": action})

    def records(self, state, records, world_time):
        return self._apply_event(
            state, "records", {"records": tuple(records), "world_time": world_time}
        )

    def output(self, state):
        if state["revision"] != self.parameter_revision or not state["active"]:
            raise ValueError("baseline_output_outside_active_parameter_window")
        previous = min(state["previous"], 255)
        embedding = self.frontend.embedding(torch.tensor(previous, device=state["last"].device))
        logits = self.byte_head(torch.cat([state["last"], embedding]))
        mask = utf8_mask(
            state["remaining"], state["lead"], 64 - state["data_bytes"], device=logits.device
        )
        return logits.masked_fill(~mask, -torch.inf).log_softmax(-1)

    def accept_byte(self, state, symbol):
        return self._apply_event(state, "byte", {"symbol": symbol})

    def reencode(self, ops):
        state = self.initial()
        for operation in ops:
            state = self._apply_event(state, operation["kind"], operation["args"])
        return state

    def save(self, path, state):
        source = hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
        return atomic_save(
            {
                "schema": "taiji-public-baseline-v1",
                "source": source,
                "architecture": self.architecture,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
                "parameters": detached_tree(self.state_dict()),
                "revision": self.parameter_revision,
                "state": detached_tree(state),
            },
            path,
        )

    @classmethod
    def load(cls, path):
        payload = atomic_load(path)
        if (
            payload["schema"] != "taiji-public-baseline-v1"
            or payload["source"]
            != hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
        ):
            raise ValueError("baseline_source_identity_mismatch")
        model = cls(payload["architecture"], seed=payload["seed"], max_tokens=payload["max_tokens"])
        tensor = next(iter(payload["parameters"].values()))
        model = model.to(dtype=tensor.dtype, device=tensor.device)
        model.load_state_dict(payload["parameters"])
        model.parameter_revision = payload["revision"]
        state = model.reencode(payload["state"]["ops"])
        assert_values_equal(detached_tree(state), payload["state"])
        return model, state
