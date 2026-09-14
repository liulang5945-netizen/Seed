"""Native raw-byte sensory and motor organs for Taiji."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from .config import TaijiConfig
from .developmental_synapse import DevelopmentalSynapseBank
from .sparse import SparseSynapses, bound_norm


class ByteSensor:
    """Map each raw byte and the boundary marker to one receptor population."""

    def __init__(self, config: TaijiConfig, *, device: torch.device | str = "cpu"):
        if config.alphabet_size < 257:
            raise ValueError("ByteSensor requires 256 bytes plus one boundary symbol")
        self.config = config
        self.device = torch.device(device)

    def encode(self, symbol: int) -> torch.Tensor:
        if not 0 <= int(symbol) < self.config.alphabet_size:
            raise ValueError(f"symbol {symbol} is outside the sensor alphabet")
        value = torch.zeros(self.config.alphabet_size, device=self.device)
        value[int(symbol)] = 1.0
        return value

    def symbols(
        self,
        data: bytes,
        *,
        include_boundary: bool = True,
        include_start_boundary: bool | None = None,
        include_end_boundary: bool | None = None,
    ) -> tuple[int, ...]:
        """Return a byte stream with independently controlled edge markers.

        ``include_boundary`` remains the compatibility switch for callers
        that want both markers.  The split controls let a resumable learner
        feed adjacent chunks without inserting a synthetic boundary between
        them: only the first chunk receives the opening marker and only the
        last chunk receives the closing marker.
        """

        if not isinstance(include_boundary, bool):
            raise TypeError("include_boundary must be a bool")
        if include_start_boundary is None:
            include_start_boundary = include_boundary
        if include_end_boundary is None:
            include_end_boundary = include_boundary
        if not isinstance(include_start_boundary, bool):
            raise TypeError("include_start_boundary must be a bool or None")
        if not isinstance(include_end_boundary, bool):
            raise TypeError("include_end_boundary must be a bool or None")
        body = tuple(int(value) for value in data)
        boundary = self.config.boundary_symbol
        prefix = (boundary,) if include_start_boundary else ()
        suffix = (boundary,) if include_end_boundary else ()
        return (*prefix, *body, *suffix)


class SparseReceptorBank:
    """Fold every cortical signal into a shared, bounded motor evidence space.

    Each input coordinate has exactly one fixed excitatory or inhibitory edge.
    Inputs are assigned evenly across receptor channels, so sparse compression
    never makes a cortical coordinate invisible to the action population.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        generator: torch.Generator,
        context_norm: float,
        device: torch.device | str = "cpu",
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("receptor dimensions must be positive")
        if out_features > in_features:
            raise ValueError("receptor count cannot exceed cortical inputs")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.context_norm = float(context_norm)
        self.device = torch.device(device)

        order = torch.randperm(self.in_features, generator=generator)
        channel = torch.empty(self.in_features, dtype=torch.long)
        channel[order] = torch.arange(self.in_features) % self.out_features
        polarity = (torch.randint(0, 2, (self.in_features,), generator=generator) * 2 - 1).to(
            torch.float32
        )
        counts = torch.bincount(channel, minlength=self.out_features).to(torch.float32)

        self.channel = channel.to(self.device)
        self.polarity = polarity.to(self.device)
        self.channel_scale = counts.rsqrt().to(self.device)

    def forward(self, cortical_state: torch.Tensor) -> torch.Tensor:
        if cortical_state.shape != (self.in_features,):
            raise ValueError(
                f"cortical state shape must be ({self.in_features},), "
                f"got {tuple(cortical_state.shape)}"
            )
        context = torch.zeros(self.out_features, device=self.device)
        context.scatter_add_(
            0,
            self.channel,
            cortical_state.to(self.device) * self.polarity,
        )
        context.mul_(self.channel_scale)
        norm = context.norm()
        if float(norm.item()) < 1e-8:
            return context
        scaled: torch.Tensor = context * (self.context_norm / norm)
        return scaled

    def to_payload(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "context_norm": self.context_norm,
            "channel": self.channel.detach().cpu().clone(),
            "polarity": self.polarity.detach().cpu().clone(),
        }

    def _payload_tensors(
        self, payload: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.in_features, self.out_features, self.context_norm)
        actual = (
            int(payload["in_features"]),
            int(payload["out_features"]),
            float(payload["context_norm"]),
        )
        if actual != expected:
            raise ValueError("receptor payload does not match architecture")
        channel = payload["channel"].detach().to(self.device, dtype=torch.long)
        polarity = payload["polarity"].detach().to(self.device, dtype=torch.float32)
        if channel.shape != (self.in_features,) or polarity.shape != (self.in_features,):
            raise ValueError("receptor payload shape does not match architecture")
        if bool((channel < 0).any()) or bool((channel >= self.out_features).any()):
            raise ValueError("receptor payload channel is outside the context space")
        if not bool(((polarity == -1.0) | (polarity == 1.0)).all()):
            raise ValueError("receptor payload polarity must be binary")
        counts = torch.bincount(channel, minlength=self.out_features).to(torch.float32)
        if bool((counts == 0).any()):
            raise ValueError("receptor payload leaves a context channel unconnected")
        return channel, polarity, counts.rsqrt().to(self.device)

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        channel, polarity, _channel_scale = self._payload_tensors(payload)
        if not torch.equal(channel, self.channel):
            raise ValueError("receptor channel map does not match architecture")
        if not torch.equal(polarity, self.polarity):
            raise ValueError("receptor polarity does not match architecture")

    def transplant_payload(self, payload: Mapping[str, Any]) -> None:
        """Adopt a validated map during an explicit architecture migration."""

        channel, polarity, channel_scale = self._payload_tensors(payload)
        self.channel = channel.detach().clone()
        self.polarity = polarity.detach().clone()
        self.channel_scale = channel_scale.detach().clone()


class ByteMotor:
    """A single action organ trained by local outcome error."""

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptors = SparseReceptorBank(
            config.cortical_context_dim,
            config.motor_context_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_context_dim,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)
        self.reward_baseline = 0.0
        self.reward_updates = 0

    def encode_context(self, cortical_state: torch.Tensor) -> torch.Tensor:
        return self.receptors.forward(cortical_state)

    def probabilities(
        self,
        context: torch.Tensor,
        *,
        episodic_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        evidence = self.synapses.forward(context) + self.bias
        if episodic_evidence is not None:
            if episodic_evidence.shape != (self.config.alphabet_size,):
                raise ValueError("episodic evidence dimension mismatch")
            evidence = evidence + episodic_evidence.to(self.device)
        evidence = evidence / float(self.config.motor_temperature)
        return torch.softmax(evidence, dim=0)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
    ) -> torch.Tensor:
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(observed_symbol)] = 1.0
        error = target - predicted.to(self.device)
        self._apply_error(context, error)
        return error

    @torch.no_grad()
    def learn_reward(
        self,
        context: torch.Tensor,
        policy_probabilities: torch.Tensor,
        action_symbol: int,
        reward: float,
    ) -> tuple[torch.Tensor, float]:
        """Apply a local three-factor action × eligibility × reward update."""

        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if policy_probabilities.shape != (self.config.alphabet_size,):
            raise ValueError("policy probability dimension mismatch")
        if not 0 <= int(action_symbol) < self.config.alphabet_size:
            raise ValueError("action is outside the motor alphabet")
        modulation = reward - self.reward_baseline
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(action_symbol)] = 1.0
        error = modulation * (target - policy_probabilities.to(self.device))
        self._apply_error(context, error)
        self.reward_baseline += self.config.reward_baseline_rate * modulation
        self.reward_updates += 1
        return error, modulation

    @torch.no_grad()
    def _apply_error(
        self,
        context: torch.Tensor,
        error: torch.Tensor,
    ) -> None:
        self.synapses.local_update(
            error,
            context,
            learning_rate=self.config.motor_learning_rate,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)

    def to_payload(self) -> dict[str, Any]:
        return {
            "receptors": self.receptors.to_payload(),
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
            "reward_baseline": self.reward_baseline,
            "reward_updates": self.reward_updates,
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        self.receptors.load_payload(payload["receptors"])
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("motor bias shape does not match architecture")
        self.bias = bias
        self.reward_baseline = float(payload["reward_baseline"])
        self.reward_updates = int(payload["reward_updates"])


class BytePredictiveContext:
    """Private F1 context with a sparse, locally plastic temporal residual.

    The organ receives only a read-only cortical state from the shared fabric.
    F1 decoder error can update the residual's existing temporal contacts, but
    no forward or learning path reaches shared fabric, F4 motor, episodic
    memory or identity state.  The residual starts at zero so an explicit
    v9→v10 migration preserves behaviour at its boundary when it adopts the
    former motor receptor map.
    """

    PAYLOAD_FORMAT = "taiji-byte-predictive-context-v1"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.receptors = SparseReceptorBank(
            config.cortical_context_dim,
            config.motor_context_dim,
            generator=generator,
            context_norm=config.motor_context_norm,
            device=self.device,
        )
        # A one-unit context cannot exclude its only input.  Normal profiles
        # forbid self contacts so the residual carries preceding context rather
        # than an instantaneous self gain.
        allow_self = config.motor_context_dim <= 1
        self.recurrent = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )

    def encode(
        self,
        cortical_state: torch.Tensor,
        *,
        prior_context: torch.Tensor | None,
        recurrent_override: DevelopmentalSynapseBank | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current context and exact prior trace used to form it."""

        base = self.receptors.forward(cortical_state)
        if prior_context is None:
            trace = torch.zeros(self.config.motor_context_dim, device=self.device)
        else:
            if prior_context.shape != (self.config.motor_context_dim,):
                raise ValueError("predictive context trace dimension mismatch")
            trace = prior_context.detach().to(self.device).clone()
        recurrent = (
            self.recurrent.forward(trace)
            if recurrent_override is None
            else recurrent_override.forward(trace)
        )
        residual = float(self.config.predictive_context_recurrent_gain) * recurrent
        context = bound_norm(base + residual, self.config.motor_context_norm)
        return context, trace

    @torch.no_grad()
    def learn(
        self,
        trace: torch.Tensor,
        feedback: torch.Tensor,
        *,
        learning_rate_scale: float = 1.0,
    ) -> None:
        """Apply F1 decoder feedback to private temporal contacts only."""

        learning_rate_scale = float(learning_rate_scale)
        if not math.isfinite(learning_rate_scale) or learning_rate_scale < 0.0:
            raise ValueError("learning_rate_scale must be finite and non-negative")

        if trace.shape != (self.config.motor_context_dim,):
            raise ValueError("predictive context trace dimension mismatch")
        if feedback.shape != (self.config.motor_context_dim,):
            raise ValueError("predictive context feedback dimension mismatch")
        # The first F1 tick has no predecessor.  Skipping it keeps the causal
        # eligibility trace explicit instead of relying on a zero update.
        if learning_rate_scale == 0.0 or not bool(trace.detach().abs().any()):
            return
        self.recurrent.local_update(
            feedback,
            trace,
            learning_rate=(self.config.predictive_context_learning_rate * learning_rate_scale),
            weight_decay=self.config.synapse_decay,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.PAYLOAD_FORMAT,
            "receptors": self.receptors.to_payload(),
            "recurrent": self.recurrent.to_payload(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported predictive context payload")
        # A v9→v10 migration may have transplanted the old motor map.  That
        # valid private topology is now persistent F1 state, so a later v10
        # restore must load it from its own content-addressed payload rather
        # than regenerate the new-organ seed topology.
        self.receptors.transplant_payload(payload["receptors"])
        self.recurrent.load_payload(payload["recurrent"])

    @torch.no_grad()
    def load_legacy_motor_payload(self, payload: Mapping[str, Any]) -> None:
        """Migrate v8/v9's fixed motor basis into the private F1 owner."""

        receptors = payload.get("receptors")
        if not isinstance(receptors, Mapping):
            raise ValueError("legacy motor payload is missing receptors")
        self.receptors.transplant_payload(receptors)
        self.recurrent.edge_weight.zero_()


class GatedMultiTimescaleTemporalResidual:
    """Optional F1 candidate with fast and slow causal eligibility traces.

    The candidate is deliberately separate from ``BytePredictiveContext`` so
    an old v10 checkpoint can remain byte-for-byte compatible when the
    candidate is not enabled.  Both residual banks start at zero: attaching
    the candidate therefore does not change the current prediction until it
    has learned.  The gate is activity-dependent, while the decay constants
    are explicit candidate metadata rather than hidden architecture values.
    """

    PAYLOAD_FORMAT = "taiji-gated-multiscale-temporal-residual-v1"
    PAYLOAD_VERSION = 1

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        fast_decay: float,
        slow_decay: float,
        gate_temperature: float,
        residual_gain: float,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.fast_decay = float(fast_decay)
        self.slow_decay = float(slow_decay)
        self.gate_temperature = float(gate_temperature)
        self.residual_gain = float(residual_gain)
        self._validate_hyperparameters()
        allow_self = config.motor_context_dim <= 1
        self.fast = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )
        self.slow = SparseSynapses(
            config.motor_context_dim,
            config.motor_context_dim,
            config.predictive_context_fan_in,
            generator=generator,
            init_scale=0.0,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
            allow_self=allow_self,
        )

    def _validate_hyperparameters(self) -> None:
        if not 0.0 <= self.fast_decay < 1.0:
            raise ValueError("fast_decay must be in [0, 1)")
        if not 0.0 < self.slow_decay < 1.0:
            raise ValueError("slow_decay must be in (0, 1)")
        if not self.gate_temperature > 0.0:
            raise ValueError("gate_temperature must be positive")
        if not self.residual_gain > 0.0:
            raise ValueError("residual_gain must be positive")

    def _slow_trace(
        self,
        prior_context: torch.Tensor,
        prior_slow_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if prior_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate prior context dimension mismatch")
        if prior_slow_context is None:
            prior_slow_context = torch.zeros_like(prior_context)
        elif prior_slow_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate slow context dimension mismatch")
        return self.slow_decay * prior_slow_context + (1.0 - self.slow_decay) * prior_context

    def encode(
        self,
        base_context: torch.Tensor,
        *,
        prior_context: torch.Tensor | None,
        prior_slow_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add a gated fast/slow residual and return the next slow trace."""

        if base_context.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate base context dimension mismatch")
        if prior_context is None:
            prior_context = torch.zeros_like(base_context)
        else:
            prior_context = prior_context.detach().to(self.device).clone()
        slow_trace = self._slow_trace(prior_context, prior_slow_context)
        fast_norm = prior_context.norm()
        slow_norm = slow_trace.norm()
        slow_gate = torch.sigmoid((slow_norm - fast_norm) / float(self.gate_temperature))
        fast_gate = 1.0 - slow_gate
        residual = float(self.residual_gain) * (
            fast_gate * self.fast.forward(prior_context) + slow_gate * self.slow.forward(slow_trace)
        )
        return (
            bound_norm(
                base_context + residual,
                self.config.motor_context_norm,
            ),
            slow_trace,
        )

    @torch.no_grad()
    def learn(
        self,
        prior_context: torch.Tensor,
        prior_slow_context: torch.Tensor | None,
        feedback: torch.Tensor,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        """Write both eligibility scales from the same causal F1 feedback."""

        if feedback.shape != (self.config.motor_context_dim,):
            raise ValueError("candidate feedback dimension mismatch")
        prior_context = prior_context.detach().to(self.device).clone()
        slow_trace = self._slow_trace(prior_context, prior_slow_context)
        self.fast.local_update(
            feedback,
            prior_context,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.slow.local_update(
            feedback,
            slow_trace,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    def parameter_tensors(self) -> tuple[torch.Tensor, ...]:
        return self.fast.edge_weight, self.slow.edge_weight

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.PAYLOAD_FORMAT,
            "version": self.PAYLOAD_VERSION,
            "fast_decay": self.fast_decay,
            "slow_decay": self.slow_decay,
            "gate_temperature": self.gate_temperature,
            "residual_gain": self.residual_gain,
            "fast": self.fast.to_payload(),
            "slow": self.slow.to_payload(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported gated temporal residual payload")
        if int(payload.get("version", -1)) != self.PAYLOAD_VERSION:
            raise ValueError("unsupported gated temporal residual version")
        values = {
            "fast_decay": float(payload["fast_decay"]),
            "slow_decay": float(payload["slow_decay"]),
            "gate_temperature": float(payload["gate_temperature"]),
            "residual_gain": float(payload["residual_gain"]),
        }
        self.fast_decay = values["fast_decay"]
        self.slow_decay = values["slow_decay"]
        self.gate_temperature = values["gate_temperature"]
        self.residual_gain = values["residual_gain"]
        self._validate_hyperparameters()
        self.fast.load_payload(payload["fast"])
        self.slow.load_payload(payload["slow"])


class BytePredictiveReadout:
    """Dedicated F1 next-byte decoder over its private predictive context.

    ``ByteMotor`` is an action policy: its reward update owns F4 decisions and
    must not be repurposed as a language loss.  The predictive readout owns
    separate physical synapses and a bias over the F1 private context, so a
    byte error never writes action policy or episodic value evidence.

    The legacy checkpoint migration intentionally seeds this decoder from the
    former shared motor decoder.  It preserves an already learned F1 surface
    at the migration boundary; all subsequent byte updates are isolated here.
    """

    PAYLOAD_FORMAT = "taiji-byte-predictive-readout-v1"

    def __init__(
        self,
        config: TaijiConfig,
        *,
        generator: torch.Generator,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.synapses = SparseSynapses(
            config.alphabet_size,
            config.motor_context_dim,
            config.motor_context_dim,
            generator=generator,
            init_scale=config.weight_init_scale,
            max_weight_norm=config.max_weight_norm,
            device=self.device,
        )
        self.bias = torch.zeros(config.alphabet_size, device=self.device)

    def probabilities(
        self,
        context: torch.Tensor,
        *,
        episodic_evidence: torch.Tensor | None = None,
        synapses_override: DevelopmentalSynapseBank | None = None,
    ) -> torch.Tensor:
        synapses = (
            self.synapses.forward(context)
            if synapses_override is None
            else synapses_override.forward(context)
        )
        evidence = synapses + self.bias
        if episodic_evidence is not None:
            if episodic_evidence.shape != (self.config.alphabet_size,):
                raise ValueError("episodic evidence dimension mismatch")
            evidence = evidence + episodic_evidence.to(self.device)
        evidence = evidence / float(self.config.motor_temperature)
        return torch.softmax(evidence, dim=0)

    def prediction_error(
        self,
        predicted: torch.Tensor,
        observed_symbol: int,
    ) -> torch.Tensor:
        if predicted.shape != (self.config.alphabet_size,):
            raise ValueError("predictive probability dimension mismatch")
        if not 0 <= int(observed_symbol) < self.config.alphabet_size:
            raise ValueError("observed symbol is outside the predictive alphabet")
        target = torch.zeros(self.config.alphabet_size, device=self.device)
        target[int(observed_symbol)] = 1.0
        return target - predicted.to(self.device)

    def context_feedback(
        self,
        error: torch.Tensor,
        *,
        synapses_override: DevelopmentalSynapseBank | None = None,
    ) -> torch.Tensor:
        """Project causal F1 error into the private context space."""

        if error.shape != (self.config.alphabet_size,):
            raise ValueError("predictive error dimension mismatch")
        # Callers take this before ``learn`` mutates decoder contacts, so the
        # temporal residual learns from the surface that made the prediction.
        synapses = self.synapses if synapses_override is None else synapses_override
        return synapses.backproject(error) / float(self.config.motor_temperature)

    @torch.no_grad()
    def learn(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        observed_symbol: int,
        *,
        preservation_probabilities: torch.Tensor | None = None,
        preservation_strength: float = 0.0,
        learning_rate_scale: float = 1.0,
    ) -> torch.Tensor:
        learning_rate_scale = float(learning_rate_scale)
        if not math.isfinite(learning_rate_scale) or learning_rate_scale < 0.0:
            raise ValueError("learning_rate_scale must be finite and non-negative")
        strength = float(preservation_strength)
        if not math.isfinite(strength) or strength < 0.0:
            raise ValueError("preservation_strength must be finite and non-negative")
        if strength > 0.0 and preservation_probabilities is None:
            raise ValueError("positive preservation_strength requires reference probabilities")
        error = self.prediction_error(predicted, observed_symbol)
        if preservation_probabilities is not None:
            if preservation_probabilities.shape != (self.config.alphabet_size,):
                raise ValueError("preservation probability dimension mismatch")
            reference = preservation_probabilities.to(self.device)
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("preservation probabilities must be finite")
            error = error + strength * (reference - predicted.to(self.device))
        if learning_rate_scale == 0.0:
            return error
        self.synapses.local_update(
            error,
            context,
            # M2-2f preserves the old F1 local-update scale while moving its
            # destination.  A future evidence-backed schedule may add a
            # dedicated rate, but the migration itself must not silently tune
            # the training semantics it is measuring.
            learning_rate=self.config.motor_learning_rate * learning_rate_scale,
            weight_decay=self.config.synapse_decay,
        )
        self.bias.add_(self.config.bias_learning_rate * learning_rate_scale * error)
        self.bias.sub_(self.bias.mean())
        self.bias.clamp_(-self.config.max_weight_norm, self.config.max_weight_norm)
        return error

    def to_payload(self) -> dict[str, Any]:
        return {
            "format": self.PAYLOAD_FORMAT,
            "synapses": self.synapses.to_payload(),
            "bias": self.bias.detach().cpu().clone(),
        }

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != self.PAYLOAD_FORMAT:
            raise ValueError("unsupported predictive readout payload")
        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("predictive readout bias shape does not match architecture")
        self.bias = bias

    def load_legacy_motor_payload(self, payload: Mapping[str, Any]) -> None:
        """Copy the pre-M2-2f shared decoder into this isolated F1 owner."""

        self.synapses.load_payload(payload["synapses"])
        bias = payload["bias"].detach().to(self.device).clone()
        if bias.shape != (self.config.alphabet_size,):
            raise ValueError("legacy motor bias shape does not match architecture")
        self.bias = bias
